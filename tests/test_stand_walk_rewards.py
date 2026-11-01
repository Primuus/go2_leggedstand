"""CPU regression checks for the authorized existing-reward refinements."""
import ast
import math
from pathlib import Path
import unittest

import torch

from deploy_mujoco.config_loader import load_config
from go2_hindstand_lab.reference_rewards import ReferenceRewards
from go2_hindstand_lab.gait_events import GaitEvents


def make_state(count=2):
    s = ReferenceRewards()
    s.cfg = load_config()
    s.num_envs, s.num_dof, s.device, s.dt = count, 12, "cpu", .02
    s.commands = torch.zeros(count, 4)
    s.base_lin_vel = torch.zeros(count, 3)
    s.base_ang_vel = torch.zeros(count, 3)
    s.rew_hanstand = torch.tensor(1.)
    s.dof_pos = torch.zeros(count, 12)
    s.descire_joint_pos = torch.zeros(12)
    s.rigid_state = torch.zeros(count, 4, 13)
    s.rigid_state[:, :, 2] = .02
    s.feet_indices = [0, 1, 2, 3]
    s.contact_foot_indices = [2, 3]
    s.contact_forces = torch.zeros(count, 4, 3)
    s.body_weight = torch.full((count,), 15.019 * 9.81)
    s.hind_sole_heights = torch.zeros(count, 2)
    s.feet_pos = torch.zeros(count, 2, 3)
    s.feet_pos[:, :, 2] = .67
    s.episode_length_buf = torch.full((count,), 20)  # phase=.25, right swing peak
    s.gait = GaitEvents(count, "cpu", .02, s.cfg.rewards)
    return s


class StandWalkRewards(unittest.TestCase):
    def test_standing_contact_and_event_rewards_do_not_depend_on_clock(self):
        s = make_state()
        s.commands[1,0] = .2
        s.contact_forces[:,:,2] = 75.
        s.gait.contact_score[1] = .75
        s.gait.clearance_score[1] = .5
        s.gait.air_score[1] = -.5
        for episode_step in (0, 1, 20, 40, 80, 1001):
            s.episode_length_buf[:] = episode_step
            torch.testing.assert_close(s._reward_contact(), torch.tensor([1.,.75]))
            torch.testing.assert_close(s._reward_feet_clearance(), torch.tensor([0.,.5]))
            torch.testing.assert_close(s._reward_feet_air_time(), torch.tensor([0.,-.5]))

    def test_front_height_is_relaxed_only_while_moving(self):
        s = make_state()
        s.commands[1, 0] = .2
        torch.testing.assert_close(s._reward_handstand_feet_height_exp() * 5., torch.tensor([5., 2.]))
        s.feet_pos[:, :, 2] += .02
        torch.testing.assert_close(s._reward_handstand_feet_height_exp() * 5., torch.tensor([5., 2.]) * math.exp(-.4))

    def test_linear_and_angular_tracking_denominators(self):
        s = make_state()
        s.commands[1, 0] = .2
        s.base_lin_vel[:, 2] = -.1
        torch.testing.assert_close(s._reward_tracking_lin_vel(), torch.tensor([math.exp(-4), math.exp(-.25)]))
        s.base_ang_vel[:, 0] = .1
        torch.testing.assert_close(s._reward_tracking_ang_vel(), torch.full((2,), math.exp(-.25)))
        s.base_lin_vel[0, 2] = 0
        s.base_lin_vel[1, 2] = -.2
        torch.testing.assert_close(s._reward_tracking_lin_vel(), torch.ones(2))

    def test_only_moving_rear_posture_is_relaxed(self):
        s = make_state()
        s.commands[1, 0] = .2
        s.dof_pos[:, 7] = 1
        torch.testing.assert_close(s._reward_default_pos(), torch.tensor([1., .2]))
        torch.testing.assert_close(s._reward_symmetric_joints(), torch.tensor([1., 0.]))
        s.dof_pos.zero_()
        s.dof_pos[:, 1] = 1
        for reward in (s._reward_default_pos, s._reward_symmetric_joints):
            torch.testing.assert_close(reward(), torch.ones(2))

    def test_reward_set_and_phase_are_preserved(self):
        s = make_state()
        active = [n for n in dir(s.cfg.rewards.scales) if not n.startswith('_') and getattr(s.cfg.rewards.scales, n) != 0]
        self.assertEqual(len(active), 24)
        self.assertEqual(s.cfg.rewards.scales.feet_air_time, 1.)
        self.assertEqual(s.cfg.env.num_observations, 67)
        self.assertEqual(s.cfg.env.num_privileged_obs, 108)
        self.assertEqual(s.cfg.rewards.target_foot_height, .06)
        self.assertEqual(s.cfg.rewards.scales.feet_clearance, 2.)
        self.assertFalse(s.cfg.commands.heading_command)
        root = Path(__file__).resolve().parents[1]
        paths = [root/'legged_gym/envs/Go2_Stand/Go2_Handstand/Go2_Handstand.py',
                 root/'go2_hindstand_lab/reference_rewards.py']
        methods = [{node.name: ast.dump(node) for node in ast.walk(ast.parse(p.read_text()))
                    if isinstance(node, ast.FunctionDef)} for p in paths]
        names = {n for n in methods[0] if n.startswith('_reward_') or n in ('_get_phase', '_get_gait_phase')}
        changed = {n for n in names if methods[0][n] != methods[1][n]}
        self.assertEqual(changed, {'_reward_tracking_lin_vel', '_reward_default_pos',
                                 '_reward_feet_clearance', '_reward_contact', '_reward_symmetric_joints',
                                 '_reward_handstand_feet_height_exp', '_reward_feet_air_time'})


if __name__ == '__main__':
    unittest.main()
