"""CPU regression checks for the authorized existing-reward refinements."""
import ast
import math
from pathlib import Path
import unittest

import torch

from deploy_mujoco.config_loader import load_config
from go2_hindstand_lab.reference_rewards import ReferenceRewards


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
    s.episode_length_buf = torch.full((count,), 20)  # phase=.25, right swing peak
    return s


class StandWalkRewards(unittest.TestCase):
    def test_contact_target_depends_on_command(self):
        s = make_state(4)
        s.commands[2:, 0] = .2
        s.contact_forces[:, 2, 2] = 50
        s.contact_forces[[0, 3], 3, 2] = 50
        self.assertEqual(s._reward_contact().tolist(), [True, False, True, False])

    def test_zero_command_has_no_clearance_incentive(self):
        s = make_state()
        s.commands[1, 0] = .2
        s.rigid_state[:, 3, 2] = .08  # 6 cm under the retained reference height convention
        torch.testing.assert_close(s._reward_feet_clearance(), torch.tensor([0., 1.]))
        s.rigid_state[:, 3, 2] = .02
        torch.testing.assert_close(s._reward_feet_clearance(), torch.tensor([0., math.exp(-1.8)]))

    def test_linear_tracking_is_tighter_without_changing_angular_tracking(self):
        s = make_state()
        s.commands[1, 0] = .2
        s.base_lin_vel[:, 2] = -.1
        torch.testing.assert_close(s._reward_tracking_lin_vel(), torch.tensor([math.exp(-4), math.exp(-.25)]))
        s.base_ang_vel[:, 0] = .1
        torch.testing.assert_close(s._reward_tracking_ang_vel(), torch.full((2,), math.exp(-.04)))
        s.base_lin_vel[0, 2] = 0
        s.base_lin_vel[1, 2] = -.2
        torch.testing.assert_close(s._reward_tracking_lin_vel(), torch.ones(2))

    def test_only_moving_rear_posture_is_relaxed(self):
        s = make_state()
        s.commands[1, 0] = .2
        s.dof_pos[:, 7] = 1
        for reward in (s._reward_default_pos, s._reward_symmetric_joints):
            torch.testing.assert_close(reward(), torch.tensor([1., .2]))
        s.dof_pos.zero_()
        s.dof_pos[:, 1] = 1
        for reward in (s._reward_default_pos, s._reward_symmetric_joints):
            torch.testing.assert_close(reward(), torch.ones(2))

    def test_reward_set_and_phase_are_preserved(self):
        s = make_state()
        active = [n for n in dir(s.cfg.rewards.scales) if not n.startswith('_') and getattr(s.cfg.rewards.scales, n) != 0]
        self.assertEqual(len(active), 23)
        self.assertEqual(s.cfg.env.num_observations, 45)
        self.assertEqual(s.cfg.env.num_privileged_obs, 86)
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
                                 '_reward_feet_clearance', '_reward_contact', '_reward_symmetric_joints'})


if __name__ == '__main__':
    unittest.main()
