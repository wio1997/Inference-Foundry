#!/usr/bin/env bash
# Install scripts/audit_prefix_storage.py once inside the P audit container.
# It needs the SAME SGLang/MemCache SDK/environment and the existing audited data.
set -euo pipefail
: "${AUDIT_HOST:?set SSH alias for the P host}"
: "${AUDIT_CONTAINER:?set owned P service container}"
: "${AUDIT_ENV_SCRIPT:?set SDK environment launcher inside container}"
: "${AUDIT_PYTHON_SCRIPT:?set installed audit_prefix_storage.py path}"
: "${AUDIT_PROFILE_PATH:?set profile JSON path inside audit container}"
# ssh sends one shell command; quote every argument for that remote shell.
REMOTE_COMMAND=$(python3 - "$AUDIT_CONTAINER" "$AUDIT_PROFILE_PATH" "$AUDIT_ENV_SCRIPT" "$AUDIT_PYTHON_SCRIPT" "$@" <<'PY'
import shlex, sys
container, profile, env_script, script, *args = sys.argv[1:]
print(shlex.join(['docker', 'exec', '-i', '-e', 'GLM_BENCH_PROFILE=' + profile,
                  container, 'bash', env_script, 'python3', script, *args]))
PY
)
exec ssh -T "$AUDIT_HOST" "$REMOTE_COMMAND"
