"""Compare exported policies using the same gait metrics as Isaac Lab evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deploy_mujoco"))
import mujoco
import numpy as np
import torch
from sim2sim_go2_handstand import Runner
from go2_hindstand_lab.evaluation import summarize_trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", type=Path, nargs="+", required=True)
    parser.add_argument("--duration", type=float, default=20.)
    parser.add_argument("--speeds", type=float, nargs="+", default=[0., .2])
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = {"backend": "MuJoCo", "version": mujoco.__version__, "cases": []}
    for path in args.policies:
        runner = Runner(path)
        # Foot geometry belongs to the calf because the XML merges fixed joints.
        foot_geoms = [runner.model.geom(n + "_foot_collision_0").id for n in ("RL", "RR")]
        dt = runner.cfg.sim.dt * runner.cfg.control.decimation
        for speed in args.speeds:
            runner.reset()
            runner.command[:] = [speed, 0., 0.]
            start = runner.data.qpos[None, :3].copy()
            trace = {k: [] for k in ("heights", "forces", "phase", "position", "upright")}
            for _ in range(round(args.duration / dt)):
                runner.step()
                mujoco.mj_forward(runner.model, runner.data)
                heights = runner.data.geom_xpos[foot_geoms, 2] - runner.model.geom_size[foot_geoms, 0]
                forces = np.zeros(2)
                for i in range(runner.data.ncon):
                    contact = runner.data.contact[i]
                    for leg, geom in enumerate(foot_geoms):
                        if geom in (contact.geom1, contact.geom2):
                            other = contact.geom2 if contact.geom1 == geom else contact.geom1
                            if runner.model.geom_type[other] == mujoco.mjtGeom.mjGEOM_PLANE:
                                force = np.zeros(6)
                                mujoco.mj_contactForce(runner.model, runner.data, i, force)
                                forces[leg] += force[0] * abs(contact.frame[2])
                values = {"heights": heights[None], "forces": forces[None],
                          "phase": np.array([runner.phase()]), "position": runner.data.qpos[None, :3].copy(),
                          "upright": np.array([-runner.observation()[3]])}
                for key, value in values.items():
                    trace[key].append(value.copy())
            result = summarize_trace(trace, dt, speed, start)
            result.update(policy=str(path), policy_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), input_width=runner.num_obs)
            report["cases"].append(result)
            print(json.dumps(result), flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
