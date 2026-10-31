"""Load the reference config without importing Isaac Gym's task registry."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_config():
    # These config-only files have exactly two inheritance imports between them.
    # Resolve them in one namespace, without executing envs/__init__.py.
    namespace = {}
    paths = [
        "legged_gym/envs/base/base_config.py",
        "legged_gym/envs/base/legged_robot_config.py",
        "legged_gym/envs/Go2_Stand/Go2_Handstand/Go2_Handstand_Config.py",
    ]
    for relative in paths:
        path = ROOT / relative
        tree = ast.parse(path.read_text(), filename=str(path))
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.ImportFrom) and node.module in (
                "base_config", "legged_gym.envs.base.legged_robot_config"
            )
        )]
        exec(compile(tree, str(path), "exec"), namespace)
    return namespace["Go2_Handstand_Cfg_Yu"]()
