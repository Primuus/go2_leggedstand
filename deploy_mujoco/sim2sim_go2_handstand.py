"""Contact-driven handstand policy on MuJoCo; original 45/47/59 policies remain evaluable."""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import time
import sys

import mujoco
import numpy as np
import torch

from config_loader import ROOT, load_config
sys.path.insert(0, str(ROOT))
from go2_hindstand_lab.phase import phase_observation
from go2_hindstand_lab.gait_events import GaitEvents, ACTOR_OBS_SIZE, gait_enabled, forward_foot_positions
from go2_hindstand_lab.legacy_events import LegacyEventObservation
from go2_hindstand_lab.terrain_profile import GroundProfiles, FINISH_X, SIDE_LIMIT, SLOPE_DEGREES, geometry_metadata
from terrain import build_model


class Runner:
    def __init__(self, policy_path=None, terrain="flat", terrain_level=0, terrain_column=5, terrain_seed=1):
        self.cfg = load_config()
        self.terrain_mode, self.terrain_level, self.terrain_column = terrain, terrain_level, terrain_column
        self.terrain_seed = terrain_seed
        self.ground_profiles = GroundProfiles(terrain_seed)
        self.model = build_model(ROOT / "resources/robots/go2/mujoco/scene_flat.xml",
                                 terrain, terrain_level, terrain_column, terrain_seed)
        self.terrain_geom_ids = {self.model.geom("floor").id}
        self.model.opt.timestep = self.cfg.sim.dt
        self.data = mujoco.MjData(self.model)
        # Policy order is FL, FR, RL, RR; the reused XML actuator order differs.
        self.names = [f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR")
                      for joint in ("hip", "thigh", "calf")]
        joint_ids = [self.model.joint(name).id for name in self.names]
        self.qadr = self.model.jnt_qposadr[joint_ids]
        self.vadr = self.model.jnt_dofadr[joint_ids]
        self.aids = np.array([self.model.actuator(name[:-6]).id
                              for name in self.names])
        if not np.array_equal(self.model.actuator_trnid[self.aids, 0], joint_ids):
            raise ValueError("Joint/actuator mapping mismatch")
        self.default = np.array([self.cfg.init_state.default_joint_angles[n] for n in self.names])
        self.kp = self.cfg.control.stiffness["joint"]
        self.kd = self.cfg.control.damping["joint"]
        self.action = np.zeros(12, dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)
        self.policy = None
        self.num_obs = ACTOR_OBS_SIZE
        if policy_path is not None:
            self.policy = torch.jit.load(str(policy_path), map_location="cpu").eval()
            self.num_obs = dict(self.policy.named_parameters())["0.weight"].shape[1]
            if self.num_obs not in (45, 47, 59, ACTOR_OBS_SIZE):
                raise ValueError(f"Unsupported policy input width: {self.num_obs}")
            with torch.inference_mode():
                self.validate_action(self.policy(torch.zeros(1, self.num_obs)))
        self.foot_geoms = [self.model.geom(n + "_foot_collision_0").id for n in ("RL", "RR")]
        self.body_weight = torch.tensor([self.model.body_mass.sum() * abs(self.cfg.sim.gravity[2])], dtype=torch.float32)
        tracker = LegacyEventObservation if self.num_obs == 59 else GaitEvents
        self.gait = tracker(1, "cpu", self.cfg.sim.dt * self.cfg.control.decimation, self.cfg.rewards)
        self.reset()

    @staticmethod
    def validate_action(output):
        if output.shape != (1, 12) or not torch.isfinite(output).all():
            raise ValueError("Expected a finite [1, 12] action")

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = self.cfg.init_state.pos
        # Training quaternion is xyzw; MuJoCo free-joint quaternion is wxyz.
        q = self.cfg.init_state.rot
        self.data.qpos[3:7] = [q[3], q[0], q[1], q[2]]
        self.data.qpos[self.qadr] = self.default
        self.action.fill(0)
        self.control_steps = 0
        self.gait.reset([0])
        self.support_height = 0.
        mujoco.mj_forward(self.model, self.data)

    def observation(self):
        scale = self.cfg.normalization.obs_scales
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, self.data.qpos[3:7])
        gravity = rotation.reshape(3, 3).T @ np.array([0., 0., -1.])
        # Angular free-joint qvel is already in body coordinates.
        obs = np.concatenate((
            self.data.qvel[3:6] * scale.ang_vel, gravity,
            self.command * [scale.lin_vel, scale.lin_vel, scale.ang_vel],
            (self.data.qpos[self.qadr] - self.default) * scale.dof_pos,
            self.data.qvel[self.vadr] * scale.dof_vel, self.action,
        )).astype(np.float32)
        if self.num_obs == 47:
            moving = (np.linalg.norm(self.command[:2]) > self.cfg.commands.standing_lin_threshold
                      or abs(self.command[2]) > self.cfg.commands.standing_ang_threshold)
            clock = phase_observation(torch.tensor(self.phase(), dtype=torch.float32), torch.tensor(moving))
            obs = np.concatenate((obs, clock.numpy()))
        elif self.num_obs in (59, ACTOR_OBS_SIZE):
            obs = np.concatenate((obs, self.gait.observation(torch.tensor([self.moving_command()]))[0].numpy()))
        return np.clip(obs, -self.cfg.normalization.clip_observations,
                       self.cfg.normalization.clip_observations)

    def moving_command(self):
        return (np.linalg.norm(self.command[:2]) > self.cfg.commands.standing_lin_threshold
                or abs(self.command[2]) > self.cfg.commands.standing_ang_threshold)

    def ground_heights(self):
        if self.terrain_mode == "flat":
            return np.zeros(2)
        x = torch.tensor(self.data.geom_xpos[self.foot_geoms, 0][None], dtype=torch.float32)
        return self.ground_profiles.sole_ground(x, torch.tensor([self.terrain_level]),
                                                torch.tensor([self.terrain_column]))[0].numpy()

    @property
    def base_height(self):
        return self.data.qpos[2] - self.support_height

    def feet_state(self):
        heights = self.data.geom_xpos[self.foot_geoms, 2] - self.model.geom_size[self.foot_geoms, 0] - self.ground_heights()
        forces = np.zeros(2, dtype=np.float32)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            for leg, geom in enumerate(self.foot_geoms):
                if geom in (contact.geom1, contact.geom2):
                    other = contact.geom2 if contact.geom1 == geom else contact.geom1
                    if other in self.terrain_geom_ids:
                        force = np.zeros(6)
                        mujoco.mj_contactForce(self.model, self.data, i, force)
                        forces[leg] += (contact.frame.reshape(3, 3).T @ force[:3])[2] * (1 if contact.geom2 == geom else -1)
        return heights.astype(np.float32), forces

    def feet_forward(self):
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, self.data.qpos[3:7])
        return forward_foot_positions(torch.tensor(self.data.geom_xpos[self.foot_geoms][None], dtype=torch.float32),
            torch.tensor(self.data.qpos[None, :3], dtype=torch.float32),
            torch.tensor(-rotation.reshape(3, 3)[None, :, 2], dtype=torch.float32))

    def phase(self):
        return (self.control_steps * self.cfg.sim.dt * self.cfg.control.decimation) % self.cfg.rewards.cycle_time / self.cfg.rewards.cycle_time

    def step(self):
        if self.policy is not None:
            with torch.inference_mode():
                output = self.policy(torch.from_numpy(self.observation()).unsqueeze(0))
            self.validate_action(output)
            self.action[:] = np.clip(output[0].numpy(),
                                    -self.cfg.normalization.clip_actions,
                                    self.cfg.normalization.clip_actions)
        target = self.default + self.action * self.cfg.control.action_scale
        for _ in range(self.cfg.control.decimation):
            torque = self.kp * (target - self.data.qpos[self.qadr]) - self.kd * self.data.qvel[self.vadr]
            limits = self.model.actuator_ctrlrange[self.aids]
            self.data.ctrl[self.aids] = np.clip(torque, limits[:, 0], limits[:, 1])
            mujoco.mj_step(self.model, self.data)
        self.control_steps += 1
        mujoco.mj_forward(self.model, self.data)
        self.support_height += .2 * (float(self.ground_heights().mean()) - self.support_height)
        if self.num_obs in (59, ACTOR_OBS_SIZE):
            heights, forces = self.feet_state()
            enabled = gait_enabled(torch.tensor([self.moving_command()]),
                torch.tensor([-self.observation()[3]]), torch.tensor([self.base_height]),
                self.cfg.rewards.base_height_target)
            args = (torch.from_numpy(forces[None]), torch.from_numpy(heights[None]), enabled, self.body_weight)
            if self.num_obs == 59:
                self.gait.update(*args)
            else:
                self.gait.update(*args, self.feet_forward(), torch.tensor(self.command[:1]))
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("Non-finite simulation state")

    def key(self, code):
        # Viewer-focused controls avoid evdev permissions and extra input packages.
        if code == ord("W"):
            self.command[0] += 0.05
        elif code == ord("S"):
            self.command[0] -= 0.05
        elif code == ord("Q"):
            self.command[2] += 0.1
        elif code == ord("E"):
            self.command[2] -= 0.1
        elif code == 32:
            self.command.fill(0)
        elif code == ord("R"):
            self.reset_requested = True
        self.command[0] = np.clip(self.command[0], -0.2, self.cfg.commands.ranges.lin_vel_x[1])
        self.command[2] = np.clip(self.command[2], -1., 1.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--policy", type=Path, help="Exported 67-input stride policy (legacy 45/47/59-input accepted for comparison)")
    mode.add_argument("--check", action="store_true", help="Check model/PD only; does not test a learned policy")
    mode.add_argument("--terrain-preview", action="store_true", help="View rolling terrain and reference pose without stepping physics")
    mode.add_argument("--pose", action="store_true", help="View the initial reference pose without advancing physics")
    parser.add_argument("--terrain", choices=("flat", "rolling"), default="flat")
    parser.add_argument("--terrain-level", type=int, choices=range(5), default=0)
    parser.add_argument("--terrain-column", type=int, choices=range(10), default=5)
    parser.add_argument("--terrain-seed", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=60.)
    parser.add_argument("--speed", type=float, default=0.)
    parser.add_argument("--yaw-rate", type=float, default=0.)
    parser.add_argument("--report", type=Path, help="Write final diagnostic JSON")
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be finite and positive")
    if not math.isfinite(args.speed) or not math.isfinite(args.yaw_rate):
        parser.error("Commands must be finite")
    if args.speed > load_config().commands.ranges.lin_vel_x[1]:
        parser.error("--speed exceeds the configured forward command limit")
    if args.terrain_preview and args.terrain != "rolling":
        parser.error("--terrain-preview requires --terrain rolling")
    static = args.pose or args.terrain_preview
    torch.set_num_threads(1)
    runner = Runner(args.policy, args.terrain, args.terrain_level, args.terrain_column, args.terrain_seed)
    runner.command[:] = [args.speed, 0., args.yaw_rate]
    runner.reset_requested = False
    print(f"MuJoCo {mujoco.__version__}: {runner.num_obs} obs, 12 actions, "
          f"dt={runner.cfg.sim.dt}, decimation={runner.cfg.control.decimation}")
    print("Static initial pose (physics paused)" if static else
          "PD/model check only" if args.check else f"Policy: {args.policy}")
    print("W/S: forward command; Q/E: yaw; Space: zero command; R: reset")
    context = nullcontext(None)
    if not args.headless:
        from mujoco import viewer as mj_viewer
        context = mj_viewer.launch_passive(runner.model, runner.data, key_callback=runner.key)
    samples = []
    start_pos = runner.data.qpos[:3].copy()
    with context as viewer:
        if viewer is not None:
            viewer.cam.lookat[:] = runner.data.qpos[:3]
            viewer.cam.distance = 1.6
            viewer.cam.azimuth = 135.
            viewer.cam.elevation = -15.
            if args.terrain_preview:
                viewer.cam.lookat[:] = [2.3, 0., 0.]
                viewer.cam.distance = 3.8
                viewer.cam.azimuth = 90.
                viewer.cam.elevation = -15.
                # Mark the centreline at true scale, to make the low smooth relief visible.
                from go2_hindstand_lab.terrain_profile import profile
                x, z = profile(args.terrain_level, args.terrain_column, args.terrain_seed)
                selected = np.flatnonzero((x >= -0.5) & (x <= 6.))[::4]
                for a, b in zip(selected[:-1], selected[1:]):
                    index = viewer.user_scn.ngeom
                    if index >= viewer.user_scn.maxgeom:
                        break
                    geom = viewer.user_scn.geoms[index]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.eye(3).ravel(), np.array([1., .65, .1, 1.]))
                    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .003,
                        np.array([x[a], 0., z[a]+.004]), np.array([x[b], 0., z[b]+.004]))
                    viewer.user_scn.ngeom += 1
                print(f"Terrain preview: L{args.terrain_level}, max slope {SLOPE_DEGREES[args.terrain_level]} degrees; orange line follows the actual surface (no vertical exaggeration)", flush=True)
        dt = runner.cfg.sim.dt * runner.cfg.control.decimation
        for _ in range(math.ceil(args.duration / dt)):
            begin = time.monotonic()
            if viewer is not None and not viewer.is_running():
                break
            if runner.reset_requested:
                runner.reset()
                runner.reset_requested = False
                start_pos = runner.data.qpos[:3].copy()
                samples.clear()
            if not static:
                runner.step()
            samples.append((runner.base_height, -runner.observation()[3]))
            if args.terrain == "rolling" and (runner.data.qpos[0] >= FINISH_X or runner.data.qpos[0] <= -2. or abs(runner.data.qpos[1]) >= SIDE_LIMIT):
                print("Terrain lane boundary reached; stopping replay")
                break
            if viewer is not None:
                if not static:
                    viewer.cam.lookat[:] = runner.data.qpos[:3]
                viewer.sync()
                time.sleep(max(0., dt - (time.monotonic() - begin)))
    report = {
        "terrain": {"mode": args.terrain, "level": args.terrain_level, "column": args.terrain_column, "seed": args.terrain_seed, "geometry": geometry_metadata()},
        "mode": "initial_pose" if static else "model_pd_check" if args.check else "policy",
        "simulated_seconds_since_reset": float(runner.data.time),
        "command": runner.command.tolist(),
        "displacement_world_m": (runner.data.qpos[:3] - start_pos).tolist(),
        "mean_base_height_m": float(np.mean([s[0] for s in samples])) if samples else None,
        "mean_upright_cosine": float(np.mean([s[1] for s in samples])) if samples else None,
        "finite_state": bool(np.isfinite(runner.data.qpos).all()),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
