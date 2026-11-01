"""Manager-based Isaac Lab task; PPO-facing order stays FL, FR, RL, RR."""
import torch
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, EventTermCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply

from deploy_mujoco.config_loader import load_config
from .asset import JOINT_NAMES, FOOT_NAMES, ROOT, prepare_urdf, asset_metadata
from .reference_rewards import ReferenceRewards
from .gait_events import GaitEvents, EVENT_OBS_SIZE, gait_enabled, forward_foot_positions
from .gait_diagnostics import GaitDiagnostics
from .terrain_lab import GroundRuntime
from .terrain_profile import speed_limit, FINISH_X, SIDE_LIMIT

REF = load_config()
SCALES = {name: getattr(REF.rewards.scales, name) for name in dir(REF.rewards.scales)
          if not name.startswith("_") and getattr(REF.rewards.scales, name) != 0}


class State(ReferenceRewards):
    def __init__(self, env):
        self.env, self.cfg = env, REF
        self.robot, self.sensor = env.scene["robot"], env.scene["contacts"]
        self.num_envs, self.device, self.num_dof, self.dt = env.num_envs, env.device, 12, env.step_dt
        self.joint_ids, names = self.robot.find_joints(JOINT_NAMES, preserve_order=True)
        assert names == JOINT_NAMES, names
        self.feet_indices, names = self.robot.find_bodies(FOOT_NAMES, preserve_order=True)
        assert names == FOOT_NAMES, names
        self.feet_name_reward_indices, self.contact_foot_indices = self.feet_indices[:2], self.feet_indices[2:]
        self.sensor_ids, names = self.sensor.find_bodies(self.robot.body_names, preserve_order=True)
        assert names == self.robot.body_names, names
        self.penalised_contact_indices = [i for i, n in enumerate(self.robot.body_names) if n not in FOOT_NAMES]
        self.base_sensor_ids, _ = self.sensor.find_bodies("base")
        self.default = torch.tensor([REF.init_state.default_joint_angles[n] for n in JOINT_NAMES], device=self.device)
        self.descire_joint_pos = torch.tensor([REF.init_state.descire_joint_angles[n] for n in JOINT_NAMES], device=self.device)
        self.target_gravity = torch.tensor(REF.asset.target_gravity, device=self.device)
        self.body_weight = self.robot.root_physx_view.get_masses().sum(dim=1).to(self.device) * abs(REF.sim.gravity[2])
        self.soft_dof_pos_limits = self.robot.data.soft_joint_pos_limits[0, self.joint_ids]
        self.previous_velocity = torch.zeros(env.num_envs, 12, device=env.device)
        self.commands = torch.zeros(env.num_envs, 4, device=env.device)
        self.ground = GroundRuntime(env)
        self.measured_heights = torch.zeros(env.num_envs, 1, device=env.device)
        self.tick = -1
        self.refresh_tick = -1
        self.actor_cache = None
        self.diagnostics = GaitDiagnostics(env.num_envs, env.device, env.step_dt)
        self.diagnostic_tick = -1
        self.gait = GaitEvents(env.num_envs, env.device, env.step_dt, REF.rewards)
        self.gait_tick = env.common_step_counter
        self.command_override = None
        self.sample(torch.arange(env.num_envs, device=env.device))

    def sample(self, ids):
        if self.command_override is not None:
            self.commands[ids] = self.command_override
            return
        for i, name in enumerate(("lin_vel_x", "lin_vel_y", "ang_vel_yaw", "heading")):
            lo, hi = getattr(REF.commands.ranges, name)
            self.commands[ids, i] = torch.empty(len(ids), device=self.device).uniform_(lo, hi)
        if self.ground.settings.mode == "rolling":
            limit = speed_limit(self.ground.levels[ids], self.ground.columns[ids])
            lower = REF.commands.ranges.lin_vel_x[0]
            self.commands[ids, 0] = lower + torch.rand(len(ids), device=self.device)*(limit-lower)
        self.commands[ids[torch.rand(len(ids), device=self.device) < REF.commands.standing_probability]] = 0

    def refresh(self):
        env, data = self.env, self.robot.data
        if self.refresh_tick == env.common_step_counter:
            return
        self.root_states = data.root_state_w
        self.base_lin_vel, self.base_ang_vel = data.root_lin_vel_b, data.root_ang_vel_b
        self.projected_gravity = data.projected_gravity_b
        self.base_euler_xyz = torch.stack(euler_xyz_from_quat(data.root_quat_w), -1)
        self.base_euler_xyz = torch.where(self.base_euler_xyz > torch.pi, self.base_euler_xyz - 2*torch.pi, self.base_euler_xyz)
        self.dof_pos, self.dof_vel = data.joint_pos[:, self.joint_ids], data.joint_vel[:, self.joint_ids]
        self.last_dof_vel = self.previous_velocity
        self.torques = data.applied_torque[:, self.joint_ids]
        self.actions, self.last_actions = env.action_manager.action, env.action_manager.prev_action
        self.rigid_state = data.body_link_state_w
        # Rear-foot collision spheres: local center (-.002, 0, 0), radius .022 m.
        rear_quat = data.body_link_quat_w[:, self.contact_foot_indices]
        offset = self.rigid_state.new_tensor([-.002, 0., 0.]).expand(env.num_envs, 2, 3)
        centers = self.rigid_state[:, self.contact_foot_indices, :3] + quat_apply(
            rear_quat.reshape(-1, 4), offset.reshape(-1, 3)).reshape(env.num_envs, 2, 3)
        forward = centers.new_tensor([0., 0., -1.]).expand(env.num_envs, 3)
        self.hind_forward = forward_foot_positions(centers, data.root_pos_w, quat_apply(data.root_quat_w, forward))
        ground_heights = self.ground.heights(centers)
        support = self.ground.update_support(ground_heights, env.episode_length_buf == 0)
        self.measured_heights = support[:, None]
        self.base_height = self.root_states[:, 2] - support
        self.hind_sole_heights = centers[..., 2] - .022 - ground_heights
        self.feet_pos = self.rigid_state[:, self.feet_name_reward_indices, :3].clone()
        self.feet_pos[:, :, 2] -= support[:, None]
        self.contact_forces = self.sensor.data.net_forces_w[:, self.sensor_ids]
        self.episode_length_buf = env.episode_length_buf
        if self.tick != env.common_step_counter:
            self.tick = env.common_step_counter
            if self.ground.settings.mode == "rolling":
                self.ground.curriculum.advance(data.root_pos_w-env.scene.env_origins,
                    self.commands[:, 0] * (env.episode_length_buf > 0), env.step_dt)
            ids = (env.episode_length_buf % int(REF.commands.resampling_time / env.step_dt) == 0).nonzero().flatten()
            self.sample(ids)
        if REF.commands.heading_command:
            # Upright heading: local -Z is horizontal; local +X points upward.
            forward = torch.zeros(env.num_envs, 3, device=env.device)
            forward[:, 2] = -1
            forward = quat_apply(data.root_quat_w, forward)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            error = torch.remainder(self.commands[:, 3] - heading + torch.pi, 2*torch.pi) - torch.pi
            self.commands[:, 2] = (.5 * error).clamp(-1, 1)
        # Initialize each environment's relative-height gate before reward terms.
        self._reward_base_height()
        if self.gait_tick != env.common_step_counter:
            enabled = gait_enabled(self._moving_command(), -self.projected_gravity[:, 0],
                                   self.base_height,
                                   REF.rewards.base_height_target) & (env.episode_length_buf > 0)
            self.gait.update(self.contact_forces[:, self.contact_foot_indices, 2], self.hind_sole_heights, enabled, self.body_weight, self.hind_forward, self.commands[:, 0])
            self.gait_tick = env.common_step_counter
        if self.diagnostic_tick != env.common_step_counter and torch.any(env.episode_length_buf > 0):
            self.diagnostics.update(self.hind_sole_heights, self.contact_forces[:, self.contact_foot_indices, 2],
                                    self._get_phase(), self._moving_command())
            self.diagnostic_tick = env.common_step_counter
        self.refresh_tick = env.common_step_counter


def state(env):
    if not hasattr(env, "reference_state"):
        env.reference_state = State(env)
    env.reference_state.refresh()
    return env.reference_state


def actor_observation(env):
    s = state(env)
    if s.actor_cache is None:
        scale = REF.normalization.obs_scales
        obs = torch.cat((s.base_ang_vel * scale.ang_vel, s.projected_gravity,
                         s.commands[:, :3] * s.commands.new_tensor([scale.lin_vel, scale.lin_vel, scale.ang_vel]),
                         (s.dof_pos - s.default) * scale.dof_pos, s.dof_vel * scale.dof_vel, s.actions,
                         s.gait.observation(s._moving_command())), -1)
        if REF.noise.add_noise:
            n = REF.noise.noise_scales
            noise = obs.new_tensor([n.ang_vel*scale.ang_vel]*3 + [n.gravity]*3 + [0.]*3 +
                                   [n.dof_pos*scale.dof_pos]*12 + [n.dof_vel*scale.dof_vel]*12 + [0.]*(12 + EVENT_OBS_SIZE))
            noise *= REF.noise.noise_level
            obs = obs + (2*torch.rand_like(obs)-1)*noise
        s.actor_cache = obs.clamp(-REF.normalization.clip_observations, REF.normalization.clip_observations)
    return s.actor_cache


def critic_observation(env):
    s = state(env)
    actor = actor_observation(env)
    fixed = actor.new_tensor([REF.domain_rand.friction, 0, 0, 0, 0] + [1]*24 +
                             [REF.domain_rand.joint_armature, REF.domain_rand.joint_friction,
                              REF.domain_rand.joint_damping, 0, 0]).expand(env.num_envs, -1)
    contacts = (s.contact_forces[:, s.feet_indices, 2] > 1.).float()
    return torch.cat((s.base_lin_vel * REF.normalization.obs_scales.lin_vel, actor, fixed, contacts), -1).clamp(
        -REF.normalization.clip_observations, REF.normalization.clip_observations)


def reward(env, name):
    s = state(env)
    return torch.as_tensor(getattr(s, "_reward_"+name)(), device=env.device).expand(env.num_envs)


def base_contact(env):
    s = state(env)
    return (s.sensor.data.net_forces_w[:, s.base_sensor_ids].norm(dim=-1) > 1.).any(-1)


def timeout(env):
    return env.episode_length_buf > env.max_episode_length


def terrain_success(env):
    s = state(env)
    if s.ground.settings.mode == "flat":
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    local = s.robot.data.root_pos_w-env.scene.env_origins
    return (local[:, 0] >= FINISH_X) & (local[:, 1].abs() <= .5) & (-s.projected_gravity[:, 0] >= .7) & (s.base_height >= .3)


def terrain_failure(env):
    s = state(env)
    if s.ground.settings.mode == "flat":
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    local = s.robot.data.root_pos_w-env.scene.env_origins
    out = (local[:, 0] <= -2.) | (local[:, 0] >= FINISH_X+.2) | (local[:, 1].abs() >= SIDE_LIMIT)
    return out | (s.base_height < .3) | (-s.projected_gravity[:, 0] < .7)


def fixed_physics(env, env_ids):
    robot = env.scene["robot"]
    materials = robot.root_physx_view.get_material_properties()
    materials[..., :2] = REF.domain_rand.friction
    materials[..., 2] = REF.domain_rand.restitution
    robot.root_physx_view.set_material_properties(materials, torch.arange(env.num_envs, dtype=torch.int32))
    robot.write_joint_damping_to_sim(REF.domain_rand.joint_damping)


def reset_pose(env, env_ids):
    robot = env.scene["robot"]
    root = robot.data.default_root_state[env_ids].clone()
    root[:, :3] += env.scene.env_origins[env_ids]
    robot.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root[:, 7:], env_ids=env_ids)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos[env_ids],
                                  torch.zeros_like(robot.data.default_joint_vel[env_ids]), env_ids=env_ids)
    if hasattr(env, "reference_state"):
        s = env.reference_state
        s.previous_velocity[env_ids] = 0
        s.diagnostics.reset(env_ids)
        s.gait.reset(env_ids)
        s.ground.curriculum.reset_episode(env_ids)
        s.ground.support_height[env_ids] = env.scene.env_origins[env_ids, 2]
        s.actor_cache = None
        s.refresh_tick = -1
        s.sample(env_ids)


@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="generator", collision_group=-1,
        terrain_generator=terrain_gen.TerrainGeneratorCfg(size=(24., 12.), border_width=2., num_rows=1, num_cols=1,
            curriculum=False, use_cache=False, sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.)}),
        max_init_terrain_level=0,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1., dynamic_friction=1., restitution=0.,
                                                       friction_combine_mode="multiply", restitution_combine_mode="multiply"))
    robot: ArticulationCfg = None
    contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=1, update_period=0.)


@configclass
class ObservationsCfg:
    @configclass
    class Policy(ObservationGroupCfg):
        obs = ObservationTermCfg(func=actor_observation)
        enable_corruption = False
        concatenate_terms = True
    @configclass
    class Critic(ObservationGroupCfg):
        obs = ObservationTermCfg(func=critic_observation)
        enable_corruption = False
        concatenate_terms = True
    policy: Policy = Policy()
    critic: Critic = Critic()


@configclass
class ActionsCfg:
    joints = mdp.JointPositionActionCfg(asset_name="robot", joint_names=JOINT_NAMES, preserve_order=True,
        scale=REF.control.action_scale, use_default_offset=False, offset=REF.init_state.default_joint_angles)


@configclass
class EventsCfg:
    physics = EventTermCfg(func=fixed_physics, mode="startup")
    reset = EventTermCfg(func=reset_pose, mode="reset")


@configclass
class DoneCfg:
    time_out = TerminationTermCfg(func=timeout, time_out=True)
    body_contact = TerminationTermCfg(func=base_contact)
    terrain_success = TerminationTermCfg(func=terrain_success, time_out=True)
    terrain_failure = TerminationTermCfg(func=terrain_failure)


RewardsCfg = configclass(type("RewardsCfg", (), {n: RewardTermCfg(func=reward, weight=w, params={"name": n}) for n, w in SCALES.items()}))


@configclass
class EnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=3.)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: DoneCfg = DoneCfg()
    commands = None
    curriculum = None
    terrain_settings: dict = {"mode": "flat", "seed": 1, "max_level": 4, "initial_level": 0}
    def __post_init__(self):
        self.decimation, self.sim.dt = REF.control.decimation, REF.sim.dt
        self.episode_length_s = REF.env.episode_length_s
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        info = asset_metadata()
        quat = REF.init_state.rot
        self.scene.robot = ArticulationCfg(prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UrdfFileCfg(asset_path=str(prepare_urdf()), usd_dir=str(ROOT / ".cache/isaaclab/usd"),
                fix_base=False, merge_fixed_joints=False, activate_contact_sensors=True,
                replace_cylinders_with_capsules=REF.asset.replace_cylinder_with_capsule,
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0., damping=0.)),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=True,
                    solver_position_iteration_count=4, solver_velocity_iteration_count=0),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, linear_damping=0., angular_damping=0.,
                    max_linear_velocity=1000., max_angular_velocity=1000., max_depenetration_velocity=1.)),
            init_state=ArticulationCfg.InitialStateCfg(pos=tuple(REF.init_state.pos), rot=(quat[3], *quat[:3]),
                joint_pos=REF.init_state.default_joint_angles, joint_vel={".*": 0.}),
            soft_joint_pos_limit_factor=.9,
            actuators={"legs": IdealPDActuatorCfg(joint_names_expr=JOINT_NAMES,
                stiffness=REF.control.stiffness["joint"], damping=REF.control.damping["joint"],
                friction=REF.domain_rand.joint_friction, armature=REF.domain_rand.joint_armature,
                effort_limit={n:float(v["effort"]) for n,v in info["limits"].items()},
                effort_limit_sim={n:float(v["effort"]) for n,v in info["limits"].items()},
                velocity_limit_sim={n:float(v["velocity"]) for n,v in info["limits"].items()})})


class Environment(ManagerBasedRLEnv):
    def step(self, actions):
        s = state(self)
        s.previous_velocity.copy_(s.robot.data.joint_vel[:, s.joint_ids])
        s.actor_cache = None
        return super().step(actions.clamp(-REF.normalization.clip_actions, REF.normalization.clip_actions))
