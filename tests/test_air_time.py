"""Contact-driven gait state and completed-update checkpoint regressions."""
from pathlib import Path
import importlib.util
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import torch

from deploy_mujoco.config_loader import load_config
from go2_hindstand_lab.gait_events import GaitEvents, gait_enabled


class AirTimeTests(unittest.TestCase):
    def make(self, count=1):
        return GaitEvents(count, "cpu", .02, load_config().rewards)

    def step(self, state, forces=(75.,75.), heights=(0.,0.), enabled=True):
        state.update(torch.tensor([forces]).float().expand_as(state.air),
                     torch.tensor([heights]).float().expand_as(state.air),
                     torch.full((len(state.air),), enabled), torch.full((len(state.air),), 150.))
        return state.air_score.clone()

    def swing(self, state, seconds, leg=1, height=.06):
        forces, heights = ([150.,0.], [0.,height]) if leg == 1 else ([0.,150.], [height,0.])
        self.step(state)
        self.step(state, forces, heights)  # First missing contact is filtered.
        for _ in range(round(seconds/.02)):
            torch.testing.assert_close(self.step(state, forces, heights), torch.zeros_like(state.air_score))

    def test_signed_landing_only_and_normalization(self):
        for seconds, expected in ((.04, -2/3), (.12, 0.), (.18, .5), (.24, 1.)):
            state = self.make()
            self.swing(state, seconds)
            self.assertAlmostEqual(float(self.step(state)[0]), expected, places=5)
            for _ in range(5):
                self.assertEqual(float(self.step(state)[0]), 0.)

    def test_continuity_alternation_and_microsteps_do_not_advance_history(self):
        state = self.make()
        self.swing(state, .04)
        self.assertLess(float(state.contact_score[0]), .5)
        self.step(state)
        self.assertFalse(state.last_valid.any())
        self.swing(state, .18, height=.03)
        self.assertEqual(float(self.step(state)[0]), 0.)
        self.assertFalse(state.last_valid.any())
        self.swing(state, .18)
        self.assertAlmostEqual(float(state.contact_score[0]), 1., places=5)
        self.assertAlmostEqual(float(state.clearance_score[0]), 1., places=5)
        self.assertGreater(float(self.step(state)[0]), 0.)
        self.assertEqual(state.last_valid.tolist(), [[False,True]])
        self.swing(state, .18, leg=1)
        self.assertEqual(float(state.contact_score[0]), 0.)
        self.assertEqual(float(state.clearance_score[0]), 0.)
        self.assertEqual(float(self.step(state)[0]), 0.)
        self.assertEqual(state.last_valid.tolist(), [[False,True]])
        self.swing(state, .18, leg=0)
        self.assertGreater(float(self.step(state)[0]), 0.)
        self.assertEqual(state.last_valid.tolist(), [[True,False]])

    def test_overlong_hover_and_unsupported_flight_cannot_farm_positive_rewards(self):
        state = self.make()
        self.swing(state, .5)
        self.assertEqual(float(state.contact_score[0]), 0.)
        self.assertEqual(float(state.clearance_score[0]), 0.)
        self.assertEqual(float(self.step(state)[0]), 0.)
        self.assertFalse(state.last_valid.any())
        self.swing(state, .18)
        self.step(state, (0.,0.), (.06,.06))
        self.assertEqual(float(self.step(state)[0]), 0.)
        self.assertFalse(state.last_valid.any())
        self.step(state, (150.,0.), (0.,.06))
        self.assertEqual(float(self.step(state)[0]), 0.)  # One-frame contact dropout.
        for _ in range(3):
            self.step(state, (0.,0.), (.02,.02))
        self.assertEqual(float(self.step(state)[0]), -1.)  # Two short landings share a bounded penalty.

    def test_zero_reset_and_partial_reset_censor_bouts(self):
        state = self.make(2)
        self.swing(state, .18)
        state.reset([0])
        torch.testing.assert_close(self.step(state), torch.tensor([0.,.5]), atol=1e-6, rtol=0)
        self.swing(state, .18, leg=0)
        self.step(state, (0.,150.), (.06,0.), enabled=False)
        torch.testing.assert_close(self.step(state), torch.zeros(2))
        self.assertFalse(state.last_valid.any())
        torch.testing.assert_close(state.observation(torch.zeros(2,dtype=torch.bool)), torch.zeros(2,14))
        with torch.inference_mode():
            self.swing(state, .18)
        state.reset([0,1])
        self.assertTrue(torch.all(state.air == 0))

    def test_left_right_mirror_and_observation_contains_all_reward_memory(self):
        left, right = self.make(), self.make()
        self.swing(left, .18, leg=0)
        self.swing(right, .18, leg=1)
        torch.testing.assert_close(left.observation(torch.tensor([True])).reshape(1,7,2).flip(-1),
                                   right.observation(torch.tensor([True])).reshape(1,7,2))
        torch.testing.assert_close(self.step(left), self.step(right))
        self.assertEqual(left.last_valid.tolist(), [[True,False]])
        before = left.observation(torch.tensor([True])).clone()
        torch.testing.assert_close(left.observation(torch.tensor([True])), before)
        self.assertEqual(before.shape, (1,14))
        self.assertTrue(torch.all((before >= 0) & (before <= 1)))
        gate = gait_enabled(torch.tensor([True,True,True,False]), torch.tensor([1.,.6,1.,1.]),
                            torch.tensor([.52,.52,.3,.52]), .52)
        self.assertEqual(gate.tolist(), [True,False,False,False])

    @unittest.skipUnless(importlib.util.find_spec("tensorboard"), "Runner training dependencies run on server")
    def test_runner_checkpoints_count_completed_updates(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rsl_rl"))
        from rsl_rl.runners import OnPolicyRunner
        model = torch.nn.Linear(1, 1)
        runner = OnPolicyRunner.__new__(OnPolicyRunner)
        runner.alg = SimpleNamespace(actor_critic=model, optimizer=torch.optim.Adam(model.parameters()),
            act=Mock(return_value=torch.zeros(1,1)), process_env_step=Mock(), compute_returns=Mock(),
            update=Mock(return_value=(0.,0.,0.,0.)))
        runner.env = SimpleNamespace(num_envs=1, get_observations=lambda: torch.zeros(1,1),
            get_privileged_observations=lambda: torch.zeros(1,1),
            step=lambda a: (a,a,torch.zeros(1),torch.zeros(1,dtype=torch.bool),{}))
        runner.device, runner.writer, runner.log = "cpu", object(), Mock()
        runner.num_steps_per_env, runner.save_interval = 1, 2
        runner.current_learning_iteration = 4101
        with tempfile.TemporaryDirectory() as directory:
            runner.log_dir = directory
            runner.learn(3)
            self.assertEqual(runner.current_learning_iteration, 4104)
            self.assertEqual(runner.alg.update.call_count, 3)
            self.assertEqual(sorted(p.name for p in Path(directory).glob('*.pt')), ['model_4102.pt','model_4104.pt'])
            for count in (4102,4104):
                self.assertEqual(torch.load(Path(directory)/f'model_{count}.pt')['iter'], count)


if __name__ == '__main__':
    unittest.main()
