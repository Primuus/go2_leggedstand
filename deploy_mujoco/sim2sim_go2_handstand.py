"""Reference 45D handstand policy on the locally reused Go2 MuJoCo model."""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np
import torch

from config_loader import ROOT, load_config


class Runner:
    def __init__(self, policy_path=None):
        self.cfg = load_config()
        self.model = mujoco.MjModel.from_xml_path(str(
            ROOT / "resources/robots/go2/mujoco/scene_flat.xml"
        ))
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
        if policy_path is not None:
            self.policy = torch.jit.load(str(policy_path), map_location="cpu").eval()
            with torch.inference_mode():
                self.validate_action(self.policy(torch.zeros(1, 45)))
        self.reset()

    @staticmethod
    def validate_action(output):
        if output.shape != (1, 12) or not torch.isfinite(output).all():
            raise ValueError("Expected a finite [1, 12] action from a [1, 45] observation")

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = self.cfg.init_state.pos
        # Training quaternion is xyzw; MuJoCo free-joint quaternion is wxyz.
        q = self.cfg.init_state.rot
        self.data.qpos[3:7] = [q[3], q[0], q[1], q[2]]
        self.data.qpos[self.qadr] = self.default
        self.action.fill(0)
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
        return np.clip(obs, -self.cfg.normalization.clip_observations,
                       self.cfg.normalization.clip_observations)

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
        self.command[0] = np.clip(self.command[0], -0.2, 0.6)
        self.command[2] = np.clip(self.command[2], -1., 1.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, help="Exported 45-input TorchScript policy")
    parser.add_argument("--check", action="store_true", help="Check model/PD only; does not test a learned policy")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=60.)
    parser.add_argument("--speed", type=float, default=0.)
    parser.add_argument("--yaw-rate", type=float, default=0.)
    parser.add_argument("--report", type=Path, help="Write final diagnostic JSON")
    args = parser.parse_args()
    if (args.policy is None) == (not args.check):
        parser.error("Select exactly one of --policy or --check")
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be finite and positive")
    if not math.isfinite(args.speed) or not math.isfinite(args.yaw_rate):
        parser.error("Commands must be finite")
    torch.set_num_threads(1)
    runner = Runner(args.policy)
    runner.command[:] = [args.speed, 0., args.yaw_rate]
    runner.reset_requested = False
    print(f"MuJoCo {mujoco.__version__}: 45 obs, 12 actions, "
          f"dt={runner.cfg.sim.dt}, decimation={runner.cfg.control.decimation}")
    print("PD/model check only" if args.check else f"Policy: {args.policy}")
    print("W/S: forward command; Q/E: yaw; Space: zero command; R: reset")
    context = nullcontext(None)
    if not args.headless:
        from mujoco import viewer as mj_viewer
        context = mj_viewer.launch_passive(runner.model, runner.data, key_callback=runner.key)
    samples = []
    start_pos = runner.data.qpos[:3].copy()
    with context as viewer:
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
            runner.step()
            samples.append((runner.data.qpos[2], -runner.observation()[3]))
            if viewer is not None:
                viewer.cam.lookat[:] = runner.data.qpos[:3]
                viewer.cam.distance = 2.5
                viewer.sync()
                time.sleep(max(0., dt - (time.monotonic() - begin)))
    report = {
        "mode": "model_pd_check" if args.check else "policy",
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
