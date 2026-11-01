"""Phase migration, symmetry, reset and contact-duration regression checks."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rsl_rl"))
sys.path.insert(0, str(ROOT / "deploy_mujoco"))
from deploy_mujoco.config_loader import load_config
from go2_hindstand_lab.phase import phase_observation
from go2_hindstand_lab.checkpoint import migrate_phase_checkpoint, expand_input
from go2_hindstand_lab.gait_diagnostics import GaitDiagnostics
from rsl_rl.modules import ActorCritic


class PhaseTests(unittest.TestCase):
    def test_evaluation_excludes_clearance_after_loss_of_upright(self):
        import numpy as np
        from go2_hindstand_lab.evaluation import summarize_trace
        trace = {"heights": np.full((200,1,2), .02), "forces": np.full((200,1,2), 75.),
                 "phase": np.full((200,1), .25), "position": np.zeros((200,1,3)),
                 "upright": np.ones((200,1))}
        trace["position"][...,2] = .52
        trace["position"][150:,:,2] = .2
        trace["heights"][150:] = .15
        report = summarize_trace(trace, .02, .2, trace["position"][0])
        self.assertFalse(report["upright_throughout"])
        self.assertAlmostEqual(report["first_upright_failure_s"], 3.02)
        self.assertEqual(report["steady_sole_peak_m"], [.02,.02])
        self.assertEqual(report["gait"]["mid_swing_above_4cm_fraction"], 0.)

    def test_clock_and_mirror(self):
        phase = torch.tensor([0., .25, .5, .75])
        clock = phase_observation(phase, torch.ones(4, dtype=torch.bool))
        torch.testing.assert_close(clock, torch.tensor([[0., 1.], [1., 0.], [0., -1.], [-1., 0.]]), atol=1e-6, rtol=0)
        torch.testing.assert_close(phase_observation((phase+.5)%1, torch.ones(4, dtype=torch.bool)), -clock, atol=1e-6, rtol=0)
        torch.testing.assert_close(phase_observation(phase, torch.zeros(4, dtype=torch.bool)), torch.zeros(4, 2))
        cfg = load_config("Go2_Handstand_PPO_Yu")
        matrix = torch.zeros(47, 47)
        for i, p in enumerate(cfg.algorithm.obs_permutation):
            matrix[int(abs(p)), i] = -1 if p < 0 else 1
        torch.testing.assert_close(matrix @ matrix, torch.eye(47))
        torch.testing.assert_close(matrix[-2:, -2:], -torch.eye(2))

    def test_migration_preserves_output_adam_and_next_update(self):
        old = ActorCritic(45, 86, 12, actor_hidden_dims=[16, 8], critic_hidden_dims=[16, 8])
        new = ActorCritic(47, 88, 12, actor_hidden_dims=[16, 8], critic_hidden_dims=[16, 8])
        opt = torch.optim.Adam(old.parameters(), lr=.0001)
        loss = old.actor(torch.randn(4,45)).square().mean() + old.critic(torch.randn(4,86)).square().mean() + old.std.square().mean()
        loss.backward(); opt.step()
        checkpoint = {"model_state_dict": old.state_dict(), "optimizer_state_dict": opt.state_dict(), "iter": 3300, "infos": None}
        original = copy.deepcopy(checkpoint)
        migrated = migrate_phase_checkpoint(checkpoint, new)
        self.assertEqual(migrated["iter"], 3300)
        for k, v in original["model_state_dict"].items():
            torch.testing.assert_close(checkpoint["model_state_dict"][k], v)
        new_opt = torch.optim.Adam(new.parameters())
        new_opt.load_state_dict(migrated["optimizer_state_dict"])
        for (name, a), (_, b) in zip(old.named_parameters(), new.named_parameters()):
            insertion = {"actor.0.weight":45, "critic.0.weight":48}.get(name)
            for field, value in opt.state[a].items():
                expected = expand_input(value, insertion) if insertion is not None and value.ndim == 2 else value
                torch.testing.assert_close(new_opt.state[b][field], expected)
        new_opt.zero_grad()
        (new.actor(torch.randn(4,47)).square().mean() + new.critic(torch.randn(4,88)).square().mean()).backward()
        new_opt.step()
        self.assertTrue(all(torch.isfinite(p).all() for p in new.parameters()))

    def test_air_timing_phase_and_censoring(self):
        d = GaitDiagnostics(1, "cpu", .02)
        def step(forces, heights=(0.,0.), phase=.25, moving=True):
            d.update(torch.tensor([heights]), torch.tensor([forces]), torch.tensor([phase]), torch.tensor([moving]))
        step((75.,75.))
        step((150.,0.), (0.,.02))
        step((150.,0.), (0.,.05))
        step((75.,75.))
        r = d.summary()
        self.assertEqual(float(r["right_completed_air_count"]), 1)
        self.assertAlmostEqual(float(r["right_completed_air_mean_s"]), .04, places=6)
        self.assertAlmostEqual(float(r["right_completed_air_peak_mean_m"]), .05, places=6)
        self.assertEqual(float(r["phase_match_given_single"]), 1)
        step((0.,150.), (.03,0.))
        d.reset([0])  # No spurious completed bout across reset.
        step((75.,75.))
        self.assertEqual(float(d.summary()["left_completed_air_count"]), 0)
        step((150.,0.), (0.,.03))
        step((150.,0.), (0.,.03), moving=False)
        step((75.,75.))
        self.assertEqual(float(d.summary()["right_completed_air_count"]), 1)
        with torch.inference_mode():
            step((150.,0.), (0.,.03))
        d.reset([0])  # Evaluation resets outside inference mode between command cases.

    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo runs locally")
    def test_mujoco_clock_reset_and_zero_command(self):
        from sim2sim_go2_handstand import Runner
        r = Runner()
        r.command[0] = .2
        for steps in (0, 1, 20, 40, 60, 80, 1001):
            r.control_steps = steps
            phase = (torch.tensor([steps]) * .02) % 1.6 / 1.6
            torch.testing.assert_close(torch.from_numpy(r.observation()[-2:]), phase_observation(phase, torch.tensor([True]))[0], atol=1e-5, rtol=0)
        r.command[:] = 0
        self.assertEqual(r.observation()[-2:].tolist(), [0., 0.])
        r.reset()
        self.assertEqual(r.control_steps, 0)


if __name__ == "__main__":
    unittest.main()
