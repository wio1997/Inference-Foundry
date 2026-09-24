# Agent orchestration

The primary agent is GPT-6 Sol. It owns the DeepSeek Extreme product line,
performance attribution, experiment variables, Runtime and cross-module work,
architecture choices, achievable-bound judgement and KEEP/REJECT/PIVOT. All
other model outputs are inputs to Sol review, not project decisions.

## Checkpoint continuity

Evidence capture, TaskCtl updates, recovery-pack compaction, Git commit/push,
interim conclusions, and service cleanup are checkpoints within the active
turn. After each checkpoint, Sol rechecks the largest removable gap, evidence
and next_action. If the next step is clear, executable, high value and
unblocked, continue in the same turn. Sol resolves competing candidates such
as GMM compute versus exposed communication by obtaining discriminating
evidence and making the product decision. End the turn only for a genuine
external permission/resource need, indispensable user information, a major
architecture fork not decidable from available evidence, material execution
risk, or a forced runtime boundary. This rule changes stopping behavior only;
the frozen product objective, model roles, correctness gate and formal E2E
standard remain in force.

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

## Headless permission diagnosis (Runs110-114, 2026-09-25)

Run110 remains INVALID for its delegated task: in headless `--mode build`,
26 Bash calls and one WebFetch call requiring permission were denied with
`No permission client configured`. A retrospective audit of Zcode's local
runtime log independently confirms all 27 model streams used DeepSeek
`deepseek-flash`; the earlier claim that its actual model was unknown is
superseded by `evidence/20260925_loop039_delegate/run110/model_permission_audit.json`.

Run112 proves Bash itself works in build mode for a simple `pwd`. Runs113
and 114 compare the same read-only Docker status query: direct headless
`--mode yolo` executed it successfully (`running`), whereas `--mode
build` stopped it before execution with the permission-client error. Zcode
CLI help says headless `--prompt` defaults to yolo; the delegation wrapper
sets build explicitly. This mode change explains the observed regression.
The earlier exact invocation history is not fully established, so avoid
claiming every prior success used yolo.

Do not treat a Zcode process exit code of zero as proof a tool ran: inspect
the model I/O tool result and command exit status. The yolo trial was an
isolated, read-only diagnostic. Keep scoped production delegation on build
unless an interactive approval client is available or the execution policy
is deliberately changed and reviewed. Sol executes blocked commands
directly meanwhile, with TaskCtl and Git evidence.
