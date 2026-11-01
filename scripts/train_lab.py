"""Train the reference 45D task on Isaac Lab using the bundled reference PPO."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rsl_rl"))

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=1500)
parser.add_argument("--save_interval", type=int, default=100)
parser.add_argument("--run_name", default="reference")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--validate_only", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Same startup ordering as the working ry_unitree_rl_lab entry point.
import torch
import torch._inductor.utils
import torchvision
import typing_extensions
torch.set_num_threads(8)
if args.device.startswith("cuda"):
    torch.cuda.set_device(torch.device(args.device))
    torch.zeros(1, device=args.device)
    torch.cuda.synchronize(args.device)
launcher = AppLauncher(dict(vars(args), multi_gpu=False))
app = launcher.app

from go2_hindstand_lab.task import Environment, EnvCfg
from go2_hindstand_lab.adapter import ReferenceVecEnv
from go2_hindstand_lab.validation import validate
from deploy_mujoco.config_loader import load_config
from rsl_rl.runners import OnPolicyRunner
import rsl_rl


def to_dict(obj):
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {k: to_dict(getattr(obj, k)) for k in dir(obj) if not k.startswith("_") and not callable(getattr(obj,k))}
    return obj


def main():
    assert Path(rsl_rl.__file__).resolve().is_relative_to(ROOT / "rsl_rl"), rsl_rl.__file__
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = ROOT / "logs/go2_handstand" / f"{stamp}_{args.run_name}"
    log_dir.mkdir(parents=True, exist_ok=False)
    print("TRAIN_LOG_DIR", log_dir, flush=True)
    cfg = EnvCfg()
    cfg.scene.num_envs, cfg.sim.device, cfg.seed = args.num_envs, args.device, args.seed
    env = Environment(cfg=cfg)
    report = validate(env)
    (log_dir / "validation.json").write_text(json.dumps(report, indent=2))
    print("PREFLIGHT_PASSED", json.dumps(report), flush=True)
    train = to_dict(load_config("Go2_Handstand_PPO_Yu"))
    train["runner"]["max_iterations"] = args.max_iterations
    train["runner"]["save_interval"] = args.save_interval
    (log_dir / "training_config.json").write_text(json.dumps(train, indent=2))
    sources = list((ROOT / "go2_hindstand_lab").glob("*.py")) + [Path(__file__), ROOT / "legged_gym/envs/Go2_Stand/Go2_Handstand/Go2_Handstand_Config.py"]
    (log_dir / "provenance.json").write_text(json.dumps({"source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "backend":"Isaac Lab", "rsl_rl_path":rsl_rl.__file__, "num_envs":args.num_envs,
        "max_iterations":args.max_iterations, "seed":args.seed, "initialization":"scratch"}, indent=2))
    if not args.validate_only:
        runner = OnPolicyRunner(ReferenceVecEnv(env), train, log_dir=str(log_dir), device=args.device)
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
        for param in runner.alg.actor_critic.parameters():
            assert torch.isfinite(param).all(), "Non-finite trained network"
        export = log_dir / "exported"
        export.mkdir()
        torch.jit.script(runner.alg.actor_critic.actor.cpu()).save(str(export / "policy.pt"))
        print("TRAINING_COMPLETE", str(log_dir), flush=True)
    env.close()


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
else:
    app.close()
