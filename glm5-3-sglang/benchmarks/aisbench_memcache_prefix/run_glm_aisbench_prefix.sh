#!/usr/bin/env bash
# GLM-5.3 Ascend4P2D: independent-prefix warmup -> realMemCache proof -> AISBench.
# Usage: bash run_glm_aisbench_prefix.sh plan|check|run [Python runner arguments]
set -euo pipefail
KIT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
exec "$PYTHON_BIN" "$KIT_DIR/run_prefix_matrix.py" "${@:-plan}"
