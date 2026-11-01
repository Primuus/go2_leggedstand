"""Train the contact-driven task on Isaac Lab using the bundled reference PPO."""
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
parser.add_argument("--max_iterations", type=int, default=1500, help="Number of updates to run, additional to a resumed checkpoint")
parser.add_argument("--resume", type=Path, help="Resume model and optimizer from a training checkpoint")
parser.add_argument("--resume_iteration", type=int, help="Explicit completed-update count for a legacy checkpoint with stale iter metadata")
parser.add_argument("--migrate_events", action="store_true", help="Replace 47/88 clock inputs with 67/108 contact/stride inputs and migrate Adam")
parser.add_argument("--migrate_stride", action="store_true", help="Migrate 59/100 event checkpoint to 67/108 stride state; retain 12 compatible state columns; reinitialize changed leg history and new stride state")
parser.add_argument("--prepare_only", action="store_true", help="Validate, restore/migrate and export a checkpoint without PPO updates")
parser.add_argument("--eval_policies", type=Path, nargs="+", help="Evaluate exported policies deterministically; no training")
parser.add_argument("--evaluation_seconds", type=float, default=20.)
parser.add_argument("--eval_speeds", type=float, nargs="+", default=[0., .2], help="Fixed forward commands for evaluation")
parser.add_argument("--save_interval", type=int, default=100)
parser.add_argument("--run_name", default="reference")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--validate_only", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.resume is not None and not args.resume.is_file():
    parser.error(f"Checkpoint does not exist: {args.resume}")
if (args.migrate_events or args.migrate_stride) and args.resume is None:
    parser.error("Migration requires --resume")
if args.migrate_events and args.migrate_stride:
    parser.error("Choose only one migration source")
if args.prepare_only and (args.resume is None or args.eval_policies or args.validate_only):
    parser.error("--prepare_only requires --resume and cannot combine with evaluation/validate_only")
if args.resume_iteration is not None and (args.resume is None or args.resume_iteration < 0):
    parser.error("--resume_iteration requires --resume and a nonnegative count")

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
    updates = 0 if args.prepare_only or args.validate_only or args.eval_policies else args.max_iterations
    train["runner"]["max_iterations"] = updates
    train["runner"]["save_interval"] = args.save_interval
    (log_dir / "training_config.json").write_text(json.dumps(train, indent=2))
    (log_dir / "env_config.json").write_text(json.dumps(to_dict(load_config()), indent=2))
    sources = list((ROOT / "go2_hindstand_lab").glob("*.py")) + [Path(__file__), ROOT / "legged_gym/envs/Go2_Stand/Go2_Handstand/Go2_Handstand_Config.py",
        ROOT / "rsl_rl/rsl_rl/runners/on_policy_runner.py"]
    provenance = {"source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "backend":"Isaac Lab", "rsl_rl_path":rsl_rl.__file__, "num_envs":args.num_envs,
        "max_iterations":updates, "seed":args.seed, "initialization":"scratch"}
    runner = None
    if args.eval_policies:
        from go2_hindstand_lab.evaluation import evaluate_policies
        evaluation = evaluate_policies(env, args.eval_policies, args.evaluation_seconds, args.eval_speeds)
        (log_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
        print("EVALUATION_COMPLETE", str(log_dir), flush=True)
    if not args.validate_only and not args.eval_policies:
        runner = OnPolicyRunner(ReferenceVecEnv(env), train, log_dir=str(log_dir), device=args.device)
        if args.resume is not None:
            if args.migrate_events or args.migrate_stride:
                from go2_hindstand_lab.checkpoint import migrate_event_checkpoint
                migrated = migrate_event_checkpoint(torch.load(args.resume, map_location="cpu"), runner.alg.actor_critic, old_count=14 if args.migrate_stride else 2)
                runner.alg.optimizer.load_state_dict(migrated["optimizer_state_dict"])
                runner.current_learning_iteration = migrated["iter"]
                provenance["event_migration"] = migrated["event_migration"]
                print("EVENT_MIGRATION_PASSED", migrated["event_migration"], flush=True)
            else:
                runner.load(str(args.resume), load_optimizer=True)
            if args.resume_iteration is not None:
                provenance["checkpoint_stored_iteration"] = runner.current_learning_iteration
                runner.current_learning_iteration = args.resume_iteration
                provenance["resume_iteration_override"] = args.resume_iteration
            # The reference loader restores Adam but not PPO's adaptive LR field.
            runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]["lr"]
            provenance.update(initialization="resume", resume_checkpoint=str(args.resume.resolve()),
                resume_checkpoint_sha256=hashlib.sha256(args.resume.read_bytes()).hexdigest(),
                optimizer_restored=True, restored_learning_rate=runner.alg.learning_rate)
            print("RESUMED", str(args.resume), "iteration", runner.current_learning_iteration,
                  "learning_rate", runner.alg.learning_rate,
                  "optimizer_state_entries", len(runner.alg.optimizer.state), flush=True)
        provenance.update(start_iteration=runner.current_learning_iteration,
                          final_iteration=runner.current_learning_iteration + updates)
    (log_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    if runner is not None:
        env.reference_state.diagnostics.clear_statistics()
        if args.prepare_only:
            runner.save(str(log_dir / f"prepared_model_{runner.current_learning_iteration}.pt"))
        else:
            runner.learn(num_learning_iterations=updates, init_at_random_ep_len=True)
        for param in runner.alg.actor_critic.parameters():
            assert torch.isfinite(param).all(), "Non-finite trained network"
        export = log_dir / "exported"
        export.mkdir()
        torch.jit.script(runner.alg.actor_critic.actor.cpu()).save(str(export / "policy.pt"))
        print("PREPARATION_COMPLETE" if args.prepare_only else "TRAINING_COMPLETE", str(log_dir), flush=True)
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
