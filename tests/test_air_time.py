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

    def step(self, state, forces=(75.,75.), heights=(0.,0.), enabled=True, forward=(0.,0.), speed=.2):
        state.update(torch.tensor([forces]).float().expand_as(state.air),
                     torch.tensor([heights]).float().expand_as(state.air),
                     torch.full((len(state.air),), enabled), torch.full((len(state.air),), 150.),
                     torch.tensor([forward]).float().expand_as(state.air), torch.full((len(state.air),), speed))
        return state.air_score.clone()

    def swing(self, state, seconds, leg=1, height=.06):
        forces, heights = ([150.,0.], [0.,height]) if leg == 1 else ([0.,150.], [height,0.])
        self.step(state)
        self.step(state, forces, heights)  # First missing contact is filtered.
        for _ in range(round(seconds/.02)):
            torch.testing.assert_close(self.step(state, forces, heights), torch.zeros_like(state.air_score))

    def land(self, state, **kwargs):
        self.assertFalse(self.step(state, **kwargs).any())
        return self.step(state, **kwargs)

    def test_signed_landing_only_and_normalization(self):
        for seconds, expected in ((.04, -2/3), (.12, 0.), (.18, .5), (.24, 1.)):
            state = self.make()
            self.swing(state, seconds)
            self.assertAlmostEqual(float(self.land(state)[0]), expected, places=5)
            for _ in range(5):
                self.assertEqual(float(self.step(state)[0]), 0.)

    def test_history_advances_on_reliable_steps_independently_of_quality(self):
        state = self.make()
        self.swing(state, .04, height=.015)
        self.land(state)
        self.assertEqual(state.last_touch.tolist(), [[False,True]])
        self.assertFalse(state.completed.any())
        self.swing(state, .18, leg=0, height=.03)
        self.assertGreater(float(self.land(state)[0]), 0.)  # <4 cm still receives duration credit.
        self.assertEqual(state.last_touch.tolist(), [[True,False]])
        self.assertFalse(state.completed.any())
        self.swing(state, .18, leg=0)
        self.assertEqual(float(state.contact_score[0]), 0.)
        self.assertEqual(float(state.clearance_score[0]), 0.)
        self.assertEqual(float(self.land(state)[0]), 0.)
        self.swing(state, .18, leg=1)
        self.assertGreater(float(self.land(state, forward=(0.,.06))[0]), 0.)
        self.assertTrue(state.completed[0,1])

    def test_contact_flicker_requires_confirmation_and_freezes_landing_position(self):
        state = self.make()
        self.swing(state, .18)
        self.step(state, forward=(0.,.06))
        self.assertFalse(state.settled.any())
        self.step(state, (150.,0.), (0.,.06))  # One-frame contact spike is not landing.
        self.assertFalse(state.last_touch.any())
        self.step(state, forward=(0.,.04))
        self.step(state, forward=(0.,.15))  # Sliding during confirmation must not inflate stride.
        self.assertTrue(state.reliable[0,1])
        self.assertAlmostEqual(float(state.touchdown_span[0,1]), .04, places=6)
        state.reset([0])
        self.swing(state, .18, height=.005)
        self.land(state)
        self.assertFalse(state.last_touch.any())

    def test_force_dip_is_fractional_but_hover_and_flight_do_not_pay(self):
        state = self.make()
        self.swing(state, .16)
        self.step(state, (0.,0.), (0.,.06))
        self.assertEqual(float(state.clearance_score[0]), 0.)
        self.assertGreater(float(state.support_ratio[0,1]), .8)
        self.assertGreater(float(self.land(state)[0]), 0.)
        state.reset([0])
        self.swing(state, .5)
        self.assertEqual(float(state.contact_score[0]), 0.)
        self.assertEqual(float(state.clearance_score[0]), 0.)
        self.assertEqual(float(self.land(state)[0]), 0.)
        state.reset([0])
        for _ in range(12):
            self.step(state, (0.,0.), (.06,.06))
            self.assertEqual(float(state.clearance_score[0]), 0.)
        self.assertEqual(float(self.land(state)[0]), 0.)
        self.assertFalse(state.last_touch.any())

    def test_forward_placement_is_bounded_and_backward_or_overshoot_do_not_pay(self):
        scores=[]
        for span in (-.06, 0., .06, .12, .3):
            state=self.make();self.swing(state,.18)
            self.land(state,forward=(0.,span))
            scores.append(float(state.contact_score[0]))
        torch.testing.assert_close(torch.tensor(scores), torch.tensor([0.,0.,.5,0.,0.]), atol=1e-6, rtol=0)
        a,b=self.make(),self.make()
        self.swing(a,.18);self.swing(b,.18)
        self.land(a,forward=(0.,.06),speed=.2)
        self.land(b,forward=(0.,-.06),speed=-.2)
        torch.testing.assert_close(a.contact_score,b.contact_score)
        for speed,target in ((.2,.06),(.3,.09),(.5,.15)):
            state=self.make();self.swing(state,.18)
            self.land(state,forward=(0.,target),speed=speed)
            self.assertAlmostEqual(float(state.contact_score[0]),.5,places=6)

    def test_horizontal_forward_projection_ignores_lateral_splay_and_is_yaw_translation_invariant(self):
        from go2_hindstand_lab.gait_events import forward_foot_positions
        root=torch.tensor([[3.,5.,.52]])
        centers=torch.tensor([[[3.03,5.14,.02],[2.97,4.86,.02]]])
        forward=torch.tensor([[1.,0.,0.]])
        expected=torch.tensor([[.03,-.03]])
        torch.testing.assert_close(forward_foot_positions(centers,root,forward),expected)
        centers[:,:,1]+=torch.tensor([.2,-.2])
        torch.testing.assert_close(forward_foot_positions(centers,root,forward),expected)
        rot=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        shift=torch.tensor([10.,-7.,0.])
        torch.testing.assert_close(forward_foot_positions(centers@rot.T+shift,root@rot.T+shift,forward@rot.T),expected)

    def test_zero_reset_and_partial_reset_censor_bouts(self):
        state = self.make(2)
        self.swing(state, .18)
        state.reset([0])
        torch.testing.assert_close(self.land(state), torch.tensor([0.,.5]), atol=1e-6, rtol=0)
        self.swing(state, .18, leg=0)
        self.step(state, (0.,150.), (.06,0.), enabled=False)
        torch.testing.assert_close(self.land(state), torch.zeros(2))
        self.assertFalse(state.last_touch.any())
        torch.testing.assert_close(state.observation(torch.zeros(2,dtype=torch.bool)), torch.zeros(2,22))
        with torch.inference_mode():
            self.swing(state, .18)
        state.reset([0,1])
        self.assertTrue(torch.all(state.air == 0))

    def test_left_right_mirror_and_state_read_does_not_advance_timers(self):
        left, right = self.make(), self.make()
        self.swing(left, .18, leg=0)
        self.swing(right, .18, leg=1)
        self.step(left, (0.,150.), (.06,0.), forward=(.03,-.03))
        self.step(right, (150.,0.), (0.,.06), forward=(-.03,.03))
        torch.testing.assert_close(left.observation(torch.tensor([True])).reshape(1,11,2).flip(-1),
                                   right.observation(torch.tensor([True])).reshape(1,11,2))
        before=left.observation(torch.tensor([True])).clone()
        torch.testing.assert_close(left.observation(torch.tensor([True])), before)
        self.assertEqual(before.shape, (1,22))
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
