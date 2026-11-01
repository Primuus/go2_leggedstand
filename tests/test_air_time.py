"""Touchdown credit, censoring and completed-update checkpoint regressions."""
from pathlib import Path
import importlib.util
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import torch

from go2_hindstand_lab.air_time import FeetAirTime


class AirTimeTests(unittest.TestCase):
    def make(self, count=1):
        return FeetAirTime(count, "cpu", .02, .12, .12, .5)

    def step(self, timer, forces, enabled=True):
        timer.update(torch.tensor([forces], dtype=torch.float).expand_as(timer.time),
                     torch.full((len(timer.time),), enabled), torch.full((len(timer.time),), 150.))
        return timer.score.clone()

    def swing(self, timer, seconds, forces=(150., 0.)):
        self.step(timer, (75., 75.))
        self.step(timer, forces)  # One-frame contact filter at takeoff.
        for _ in range(round(seconds / timer.dt)):
            torch.testing.assert_close(self.step(timer, forces), torch.zeros_like(timer.score))

    def test_signed_landing_only_cap_and_support_foot(self):
        for seconds, expected in ((.04, -.08), (.12, 0.), (.20, .08), (.60, .12)):
            timer = self.make()
            self.swing(timer, seconds)
            self.assertAlmostEqual(float(self.step(timer, (75.,75.))[0]), expected, places=6)
            for _ in range(5):
                self.assertEqual(float(self.step(timer, (75.,75.))[0]), 0.)

    def test_zero_command_reset_and_partial_reset_censor_bouts(self):
        timer = self.make(2)
        self.swing(timer, .20)
        timer.reset([0])
        torch.testing.assert_close(self.step(timer, (75.,75.)), torch.tensor([0., .08]))
        self.swing(timer, .20)
        self.step(timer, (150.,0.), enabled=False)
        torch.testing.assert_close(self.step(timer, (75.,75.)), torch.zeros(2))
        self.assertTrue(torch.all(timer.time == 0))

    def test_unsupported_air_has_no_positive_credit_and_contact_glitch_is_ignored(self):
        timer = self.make()
        self.swing(timer, .20, forces=(0.,0.))
        self.assertEqual(float(self.step(timer, (75.,75.))[0]), 0.)
        self.step(timer, (150.,0.))
        self.assertEqual(float(self.step(timer, (75.,75.))[0]), 0.)
        self.swing(timer, .20)
        self.step(timer, (0.,0.))  # Opposite support lost during the swing.
        self.assertEqual(float(self.step(timer, (75.,75.))[0]), 0.)

    def test_left_right_symmetry_and_inference_mode_reset(self):
        left, right = self.make(), self.make()
        with torch.inference_mode():
            self.swing(left, .2, (0.,150.))
            self.swing(right, .2, (150.,0.))
            torch.testing.assert_close(self.step(left, (75.,75.)), self.step(right, (75.,75.)))
        left.reset([0])
        self.assertEqual(float(left.score[0]), 0.)

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
