#!/usr/bin/env bash
set -e
GO2_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$GO2_PROJECT_ROOT/rsl_rl:$GO2_PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$GO2_PROJECT_ROOT"
exec /workspace/isaaclab/_isaac_sim/python.sh "$@"
