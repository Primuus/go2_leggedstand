"""Deterministic fixed-command evaluation and common Isaac Lab/MuJoCo metrics."""
import hashlib
from pathlib import Path
import numpy as np
import torch

from .gait_diagnostics import GaitDiagnostics
from .gait_events import ACTOR_OBS_SIZE
from .phase import phase_observation


def summarize_trace(trace, dt, speed, start, burn_in_s=2.):
    arrays = {k: np.asarray(v) for k, v in trace.items()}
    count, num_envs = arrays["heights"].shape[:2]
    first = min(round(burn_in_s / dt), count)
    failed = (arrays["position"][..., 2] < .3) | (arrays["upright"] < .7)
    failures = np.flatnonzero(failed.any(axis=1))
    last = int(failures[0]) if failures.size else count
    heights = arrays["heights"][first:last]
    diagnostics = GaitDiagnostics(num_envs, "cpu", dt)
    for i in range(first, last):
        diagnostics.update(torch.tensor(arrays["heights"][i]), torch.tensor(arrays["forces"][i]),
                           torch.tensor(arrays["phase"][i]), torch.full((num_envs,), speed != 0.))
    result = {"duration_s": count * dt, "command_vx": speed, "num_envs": num_envs,
              "burn_in_s": burn_in_s,
              "upright_throughout": not bool(failures.size),
              "first_upright_failure_s": (last+1)*dt if failures.size else None,
              "gait_window_end_s": last*dt,
              "displacement_world_mean_m": (arrays["position"][-1] - start).mean(0).tolist(),
              "forward_speed_mean_m_s": float((arrays["position"][-1, :, 0] - start[:, 0]).mean() / (count*dt)),
              "base_height_mean_m": float(arrays["position"][..., 2].mean()),
              "base_height_min_m": float(arrays["position"][..., 2].min()),
              "upright_cosine_min": float(arrays["upright"].min()),
              "finite": all(np.isfinite(a).all() for a in arrays.values()),
              "steady_sole_peak_m": heights.max(axis=(0, 1)).tolist() if heights.size else None,
              "steady_sole_p95_m": np.percentile(heights, 95, axis=(0, 1)).tolist() if heights.size else None,
              "gait": {k: float(v) for k, v in diagnostics.summary().items()}}
    # Completed, meaningful lift segments, matching the previous 3300 report criteria.
    bouts = [[], []]
    contact = arrays["forces"] > 1.
    for env_id in range(num_envs):
        for leg in range(2):
            begin = None
            for i in range(first+1, last):
                if contact[i-1, env_id, leg] and not contact[i, env_id, leg]:
                    begin = i
                if contact[i, env_id, leg] and begin is not None:
                    duration = (i-begin)*dt
                    peak = float(arrays["heights"][begin:i, env_id, leg].max())
                    if duration >= .04 - 1e-8 and peak > .003:
                        bouts[leg].append((duration, peak))
                    begin = None
    result["valid_air_bouts"] = [{"count": len(b),
        "duration_median_s": float(np.median([v[0] for v in b])) if b else None,
        "peak_median_m": float(np.median([v[1] for v in b])) if b else None} for b in bouts]
    return result


@torch.inference_mode()
def evaluate_policies(env, paths, seconds, speeds=(0., .2)):
    from .task import REF, state
    REF.noise.add_noise = False
    report = {"backend": "Isaac Lab", "observation_noise": False, "cases": []}
    s = state(env)
    for path in paths:
        policy = torch.jit.load(str(path), map_location=env.device).eval()
        width = dict(policy.named_parameters())["0.weight"].shape[1]
        assert width in (45, 47, ACTOR_OBS_SIZE)
        for speed in speeds:
            s.command_override = torch.tensor([speed, 0., 0., 0.], device=env.device)
            obs, _ = env.reset()
            start = (s.robot.data.root_pos_w-env.scene.env_origins).cpu().numpy().copy()
            trace = {k: [] for k in ("heights", "forces", "phase", "position", "upright")}
            resets = 0
            with torch.inference_mode():
                for _ in range(round(seconds/env.step_dt)):
                    inputs = obs["policy"][:, :width]
                    if width == 47:
                        inputs = torch.cat((obs["policy"][:, :45], phase_observation(s._get_phase(), s._moving_command())), -1)
                    action = policy(inputs)
                    obs, _, terminated, truncated, _ = env.step(action)
                    if (terminated | truncated).any():
                        resets += int((terminated | truncated).sum())
                        break  # Do not mix reset poses into a continuous walking trace.
                    s = state(env)
                    values = {"heights": s.hind_sole_heights,
                              "forces": s.contact_forces[:, s.contact_foot_indices, 2],
                              "phase": s._get_phase(), "position": s.robot.data.root_pos_w-env.scene.env_origins,
                              "upright": -s.projected_gravity[:, 0]}
                    for key, value in values.items():
                        trace[key].append(value.cpu().numpy().copy())
            result = summarize_trace(trace, env.step_dt, speed, start) if trace["heights"] else {"duration_s": 0.}
            result.update(policy=str(path), policy_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                          input_width=width, early_resets=resets)
            report["cases"].append(result)
            print("EVAL_CASE", str(path), speed, result, flush=True)
    return report
