# Agent orchestration

The active Codex session is the GPT-6 Sol primary agent. It chooses and reviews
every task. The repository TaskCtl records Task/Loop/Run evidence; it is not a
model router and its schema remains unchanged.

For a bounded, independently verifiable task with a concrete acceptance
check, create a task text file and run:

    python3 scripts/delegate_zcode.py --task-file TASK.txt --record-dir RECORD_DIR

The wrapper verifies the configured Zcode main model is a DeepSeek model, then
spawns one Zcode CLI process in plan mode. It stores its exit status, elapsed
time, selected model name, task digest, stdout, and stderr. Keep task text
specific and review the output against the requested check. A successful
process exit alone does not establish correctness. If the task needs an edit,
the primary agent applies and verifies the change.

Use GPT-6 Sol as the primary agent for investigation, Runtime changes,
integration and final decisions. Astra Medium can supply an independent review
of important evidence or experiment design; Astra High is reserved for costly
architectural choices or unresolved, high-uncertainty failures. DeepSeek/Zcode
can handle clearly scoped verifiable work, including a complex investigation
when its boundary and acceptance check are explicit. No model has a required
invocation quota. The primary agent reviews every delegated result and may
execute the task itself when that is clearer or faster.

Do not put credentials in task files or committed output. Keep agent evidence
separate from Runtime experimental results; add a TaskCtl artifact reference
only when an agent result materially contributes to a Run.
