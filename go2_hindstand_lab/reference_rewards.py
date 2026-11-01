# Copyright (c) 2026 Yu Songmin
# SPDX-License-Identifier: MIT
# Extracted reference methods with authorized stand/walk refinements; see doc/upstream_manifest.json.
import torch


class ReferenceRewards:
    def _moving_command(self):
        return ((torch.linalg.vector_norm(self.commands[:, :2], dim=1) > self.cfg.commands.standing_lin_threshold)
                | (self.commands[:, 2].abs() > self.cfg.commands.standing_ang_threshold))

    def _get_phase(self):
        cycle_time = self.cfg.rewards.cycle_time
        phase = (self.episode_length_buf * self.dt)%cycle_time/cycle_time
        return phase

    def _get_gait_phase(self):
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        stance_mask[:, 0] = phase<0.5
        stance_mask[:, 1] = phase>0.5
        return stance_mask

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.exp(-torch.abs(self.base_lin_vel[:, 0])*10)

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.exp(-torch.norm(torch.abs(self.base_ang_vel[:, 1:3]), dim=1))

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        # print(self.root_states[0,2])
        self.rew_hanstand=torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target)*5)
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target)*5)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.abs(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return  torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.soft_dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.soft_dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        x_error = torch.square(self.commands[:, 0] + self.base_lin_vel[:, 2])#站立起来的话，本体的x轴对应世界系的z，本体z轴对应世界系的-x
        y_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        # print(torch.mean(self.rew_hanstand),self.rew_hanstand.shape)
        sigma = torch.where(self._moving_command(), self.cfg.rewards.tracking_lin_sigma,
                            self.cfg.rewards.tracking_stand_sigma)
        return torch.exp(-(y_error+x_error)/sigma)*(self.rew_hanstand>0.70)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 0])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)*(self.rew_hanstand>0.70)

    def _reward_default_pos(self):
        # Keep the front-leg reference; relax the rear legs only during walking.
        error = torch.abs(self.dof_pos - self.descire_joint_pos)
        rear_factor = torch.where(self._moving_command(), self.cfg.rewards.moving_rear_posture_factor, 1.)
        return error[:, :6].sum(dim=1) + rear_factor * error[:, 6:].sum(dim=1)

    def _reward_default_pos_reward(self):
        # Penalize motion at zero commands
        return torch.exp(-torch.sum(torch.abs(self.dof_pos - self.descire_joint_pos)[:,:6], dim=1))*(self.rew_hanstand>0.70)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_handstand_feet_on_air(self):
        """
        脚部在空奖励：
        1. 使用 self.contact_forces 判断足部是否接触地面（通过预先设置的阈值）。
        2. 如果所有足部都没有接触地面，则奖励1，否则奖励为0（或取平均）。
        """
        # contact_forces: shape = (num_envs, num_bodies, 3)
        contact = torch.norm(self.contact_forces[:, self.feet_name_reward_indices , :], dim=-1) > 1.0
        # 如果所有足部均未接触地面，reward = 1；也可以使用 mean 得到部分奖励
        reward = (~contact).float().prod(dim=1)
        return reward

    def _reward_handstand_orientation(self):
        """
        姿态奖励：
        1. 使用 self.projected_gravity（机器人基座坐标系下的重力投影）来评估姿态。
        2. 目标重力方向通过配置传入（例如 [1, 0, 0] 表示目标为竖直向上）。
        3. 对比当前和目标重力方向的 L2 距离，偏差越大惩罚越大。
        """
        return torch.square((self.projected_gravity - self.target_gravity)).sum(dim=1)

    def _reward_default_hip_pos(self):
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        """
        joint_diff = torch.abs(self.dof_pos[:,0])+torch.abs(self.dof_pos[:,3])+torch.abs(self.dof_pos[:,6])+torch.abs(self.dof_pos[:,9])
        return joint_diff

    def _reward_ang_xz(self):
        return torch.abs(self.base_euler_xyz[:,0])*(self.rew_hanstand>0.70)#*(self.root_states[:,2]>0.45)

    def _reward_feet_clearance(self):
        return self.gait.clearance_score

    def _reward_contact(self):
        contact = self.contact_forces[:, self.contact_foot_indices, 2] > 1.
        standing = contact.all(dim=1).float() * (self.rew_hanstand > .70)
        return torch.where(self._moving_command(), self.gait.contact_score, standing)

    def _reward_feet_air_time(self):
        # RewardManager integrates rates with dt. This term is a one-step event.
        score = self.gait.air_score
        return score / self.dt if self.cfg.rewards.air_time_event_integral else score

    def _reward_symmetric_joints(self):
        # 左右关节对称（参考 go2_trot 的对称设置）：前后两对腿都参与
        # hip 左右正方向定义相反，比较前取反；thigh/calf 直接比较
        # dof 顺序 FL,FR,RL,RR：前对 FL-FR(0,1)，后对 RL-RR(2,3)
        dof = self.dof_pos.clone().view(self.num_envs, 4, int(self.num_dof/4))
        # Multiply the right side hips by -1 to match the sign of the left side:
        dof[:,1,0] *= -1   # FR hip
        dof[:,3,0] *= -1   # RR hip
        rear_factor = torch.where(self._moving_command(), 0., 1.)
        err = torch.sum(torch.abs(dof[:,0,:] - dof[:,1,:]),axis=1) + rear_factor * torch.sum(torch.abs(dof[:,2,:] - dof[:,3,:]),axis=1)
        return err*(self.rew_hanstand>0.70)

    def _reward_orientation_symmetry(self):
        # 左右姿态对称（重力投影版）：gy = -sin(roll)cos(pitch)
        # 倒立平衡时 roll≈0 ⇔ gy≈0；与 obs 中的 projected_gravity 同一来源，
        # 训练信号与策略观测严格一致（镜像下 gy→-gy，平方后不变，天然左右对称）
        return torch.square(self.projected_gravity[:, 1])

    def _reward_feet_height_symmetry(self):
        # 左右脚（倒立时的"手"）高度对称（trot feet_clearance 的成对思想）
        # 本任务前腿支撑：feet_name_reward=['FL_foot','FR_foot']，feet_pos 顺序 [FL, FR]
        left = self.feet_pos[:,0,2]
        right = self.feet_pos[:,1,2]
        return torch.abs(left - right)

    def _reward_handstand_feet_height_exp(self):
        # feet_indices_tensor = torch.tensor(feet_indices, dtype=torch.long, device=self.rigid_body_pos.device)
        feet_height_error = torch.abs(self.feet_pos[:,:,2]- 0.67).sum(dim=1)
        # print(torch.exp(-feet_height_error / (std**2)))
        # print(feet_height_error.shape)
        factor = torch.where(self._moving_command(), self.cfg.rewards.moving_front_height_factor, 1.)
        return torch.exp(-feet_height_error *10) * factor

    def _reward_alive(self):
        return 1.0
