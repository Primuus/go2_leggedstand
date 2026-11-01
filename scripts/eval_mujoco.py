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
from go2_hindstand_lab.terrain_profile import FINISH_X, SIDE_LIMIT, geometry_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", type=Path, nargs="+", required=True)
    parser.add_argument("--duration", type=float, default=20.)
    parser.add_argument("--speeds", type=float, nargs="+", default=[0., .2])
    parser.add_argument("--terrain", choices=("flat", "rolling"), default="flat")
    parser.add_argument("--terrain-level", type=int, choices=range(5), default=0)
    parser.add_argument("--terrain-column", type=int, choices=range(10), default=5)
    parser.add_argument("--terrain-seed", type=int, default=1)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = {"backend": "MuJoCo", "version": mujoco.__version__, "cases": []}
    for path in args.policies:
        runner = Runner(path, args.terrain, args.terrain_level, args.terrain_column, args.terrain_seed)
        # Foot geometry belongs to the calf because the XML merges fixed joints.
        foot_geoms = [runner.model.geom(n + "_foot_collision_0").id for n in ("RL", "RR")]
        dt = runner.cfg.sim.dt * runner.cfg.control.decimation
        for speed in args.speeds:
            runner.reset()
            runner.command[:] = [speed, 0., 0.]
            start = runner.data.qpos[None, :3].copy()
            trace = {k: [] for k in ("heights", "forces", "phase", "position", "upright", "foot_forward", "base_height")}
            for _ in range(round(args.duration / dt)):
                runner.step()
                mujoco.mj_forward(runner.model, runner.data)
                heights, forces = runner.feet_state()
                values = {"foot_forward": runner.feet_forward().numpy(), "heights": heights[None], "forces": forces[None],
                          "base_height": np.array([runner.base_height]),
                          "phase": np.array([runner.phase()]), "position": runner.data.qpos[None, :3].copy(),
                          "upright": np.array([-runner.observation()[3]])}
                for key, value in values.items():
                    trace[key].append(value.copy())
                if args.terrain == "rolling" and (runner.data.qpos[0] >= FINISH_X or runner.data.qpos[0] <= -2. or abs(runner.data.qpos[1]) >= SIDE_LIMIT):
                    break
            result = summarize_trace(trace, dt, speed, start, body_weight=runner.body_weight.numpy())
            result.update(policy=str(path), policy_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), input_width=runner.num_obs, terrain=dict(mode=args.terrain, level=args.terrain_level, column=args.terrain_column, seed=args.terrain_seed, geometry=geometry_metadata()))
            report["cases"].append(result)
            print(json.dumps(result), flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
