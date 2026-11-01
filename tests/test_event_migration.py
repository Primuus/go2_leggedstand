"""Input/Adam migration and deployment use the same observable event state."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'rsl_rl'))
sys.path.insert(0, str(ROOT / 'deploy_mujoco'))
from rsl_rl.modules import ActorCritic
from go2_hindstand_lab.checkpoint import migrate_event_checkpoint, replace_clock_inputs
from go2_hindstand_lab.gait_events import GaitEvents, gait_enabled


class EventMigrationTests(unittest.TestCase):
    def test_weights_and_adam_remapped_without_mutating_original(self):
        old = ActorCritic(47,88,12,actor_hidden_dims=[16,8],critic_hidden_dims=[16,8])
        new = ActorCritic(59,100,12,actor_hidden_dims=[16,8],critic_hidden_dims=[16,8])
        opt = torch.optim.Adam(old.parameters(), lr=.0001)
        (old.actor(torch.randn(4,47)).square().mean()+old.critic(torch.randn(4,88)).square().mean()+old.std.square().mean()).backward()
        opt.step()
        original = {'model_state_dict':old.state_dict(),'optimizer_state_dict':opt.state_dict(),'iter':4401}
        before = copy.deepcopy(original)
        migrated = migrate_event_checkpoint(original,new)
        self.assertEqual(migrated['iter'],4401)
        for k,v in before['model_state_dict'].items():
            torch.testing.assert_close(original['model_state_dict'][k],v)
        new_opt = torch.optim.Adam(new.parameters())
        new_opt.load_state_dict(migrated['optimizer_state_dict'])
        for (name,a),(_,b) in zip(old.named_parameters(),new.named_parameters()):
            insertion = {'actor.0.weight':45,'critic.0.weight':48}.get(name)
            expected = replace_clock_inputs(a,insertion) if insertion is not None else a
            torch.testing.assert_close(b,expected)
            for field,value in opt.state[a].items():
                expected = replace_clock_inputs(value,insertion) if insertion is not None and value.ndim==2 else value
                torch.testing.assert_close(new_opt.state[b][field],expected)
        new_opt.zero_grad()
        (new.actor(torch.ones(4,59)).square().mean()+new.critic(torch.ones(4,100)).square().mean()).backward()
        new_opt.step()
        self.assertTrue(torch.isfinite(new.actor[0].weight).all())
        self.assertTrue(torch.any(new.actor[0].weight[:,45:] != 0))

    @unittest.skipUnless(importlib.util.find_spec('mujoco'), 'MuJoCo runs locally')
    def test_mujoco_event_observation_matches_shared_tracker_once_per_step(self):
        from sim2sim_go2_handstand import Runner
        r = Runner()
        reference = GaitEvents(1,'cpu',.02,r.cfg.rewards)
        r.command[0] = .2
        for _ in range(20):
            r.step()
            heights,forces = r.feet_state()
            enabled = gait_enabled(torch.tensor([True]),torch.tensor([-r.observation()[3]]),
                                   torch.tensor([r.data.qpos[2]]),r.cfg.rewards.base_height_target)
            reference.update(torch.from_numpy(forces[None]),torch.from_numpy(heights[None]),enabled,r.body_weight)
            expected = reference.observation(torch.tensor([True]))[0]
            for _ in range(2):  # Re-reading the observation must not advance timers.
                torch.testing.assert_close(torch.from_numpy(r.observation()[45:]),expected)
            self.assertEqual(r.observation().shape,(59,))
        r.command[:] = 0
        self.assertEqual(r.observation()[45:].tolist(),[0.]*14)
        r.reset()
        self.assertFalse(r.gait.last_valid.any())
        self.assertFalse(r.gait.air.any())


if __name__ == '__main__':
    unittest.main()
