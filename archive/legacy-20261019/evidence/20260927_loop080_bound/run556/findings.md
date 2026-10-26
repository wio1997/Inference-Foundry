# Run556 — preflight rejected residual container workers

Status: **INVALID before source installation, service or NPU execution.** The controller returned exit1 because its stopped-container preflight found 91 non-zombie processes against a maximum of 10. Inspection showed many old Python `multiprocessing.spawn`/`forkserver` processes; no VLLM worker or NPU process was running. Cleanup verified the service stopped and eight NPUs idle. `restore_exit=not_needed` and `frontier_restore_exit=not_needed`; no source-before snapshot or performance sample was acquired.

The dedicated container was restarted. Afterwards `ps` showed only PID1 `sleep infinity` plus the inspection command, the installed ModelRunner, fixed serving and Extreme runtime matched pristine SHA `004dbd0d`, `137f1cc6`, `eb4b5142`, and eight NPUs had no running processes. The same reviewed protocol was retried under fresh Run557. No Bound endpoint or Current formal TPS changed.
