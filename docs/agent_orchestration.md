# Agent orchestration

The primary agent is GPT-6 Sol. It owns the DeepSeek Extreme product line,
performance attribution, experiment variables, Runtime and cross-module work,
architecture choices, achievable-bound judgement and KEEP/REJECT/PIVOT. All
other model outputs are inputs to Sol review, not project decisions.

## Review models

Use GPT-6 Astra Medium for independent checks on important conclusions,
complex profiling attribution, experiment design and competing root causes.
Use GPT-6 Astra High for key architecture questions, long-running failure to
converge, and independent review of the achievable bound or proximity to the
performance limit. No invocation quota applies; select a task with a clear
question, evidence set and acceptance criteria. Record the actual model used.

## Zcode + DeepSeek execution

Prefer Zcode for bounded, low-risk work with an objective completion check:
environment and service setup, start/stop, benchmark and repeat runs under
already-frozen parameters, existing scripts, profile/log/trace capture,
process/NPU checks, data extraction, result organization, narrow source
location and simple validation. Give it the exact input, command or allowed
operations, output path and acceptance check. Do not default to Zcode for
cross-module edits, performance root-cause judgement, variable selection,
KEEP/REJECT/PIVOT, architecture or limit decisions.

For a read-only investigation, create a task file and call:

    python3 scripts/delegate_zcode.py --task-file TASK.txt --record-dir RECORD_DIR --mode plan

For explicitly scoped shell execution, use --mode build and state allowed
commands and paths in the task file. Never use Zcode's yolo mode through the
wrapper. The wrapper records configured and observed model identities
separately, exit code, timeout, task digest, stdout and stderr. Check
model_verified from actual output before attributing a result to DeepSeek;
a configured model alone is insufficient. A timeout or successful process
exit alone does not establish task correctness.

Sol reviews produced files, logs, exact parameters and invariants, then
records valid/invalid Runs in TaskCtl and commits the recovery package and
necessary evidence to main. Keep credentials out of task files and committed
outputs. Agent execution logs are provenance; they enter product evidence only
after Sol validates the result.

## Current headless execution limitation (Run110, 2026-09-25)

The local Zcode CLI accepted --mode build and completed a bounded check, but
its headless permission layer denied docker exec, npu-smi and local HTTP with
"No permission client configured for Bash". Its JSON output had no provider
model identifier; a self-reported model in the answer is not independent
execution evidence. The run took188.845s and27 provider requests for a
simple check. Treat this route as unavailable for service/NPU operations until
permissions and actual-model reporting are verified. Sol uses direct tools
for time-sensitive mechanical work when this fallback is necessary, while
preserving TaskCtl and Git evidence. Run110 is INVALID, not an endorsement
of those delegated results.
