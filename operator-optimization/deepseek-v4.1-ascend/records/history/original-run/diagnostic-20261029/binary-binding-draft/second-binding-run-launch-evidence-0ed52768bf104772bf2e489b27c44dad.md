# Second selective MC2 binding diagnostic — launch evidence

- Timestamp (UTC): 2026-10-29, see STATE.json `second_binding_launch_20261029.timestamp_utc`
- Authorization: explicit user authorization for one isolated diagnostic on their own 169
- Launch command executed exactly once:
  `python3 diagnostic-20261029/binary-binding-draft/second-binding-run/launch_second_background.py --execute`
- Working directory: `runs/169-vllm-v41`
- Launch exit code: 0

## Preflight (before launch)

- `controller-reservation.json`: absent (checked by `ls` and `find diagnostic-20261029 -maxdepth 3`)
- Active second-run process: none (`ps` grep for `second-binding-run`, `orchestrate_binding`, `launch_second`, `start_binding`, `supervise_binding` returned no rows)

## Exact launcher output

```json
{
  "handle": "0ed52768bf104772bf2e489b27c44dad",
  "created_unix": 1793249898.46964,
  "state": "CONTROLLER_STARTED_NOT_DIAGNOSTIC_SUCCESS",
  "stdout": "/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binary-binding-draft/second-binding-run/controller-0ed52768bf104772bf2e489b27c44dad.stdout",
  "stderr": "/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/binary-binding-draft/second-binding-run/controller-0ed52768bf104772bf2e489b27c44dad.stderr",
  "pid": 57144
}
```

## Reservation

- Path: `diagnostic-20261029/binary-binding-draft/second-binding-run/controller-reservation.json`
- Created, 669 bytes; handle/pid match the launcher output.

## Controller process

- PID: 57144
- Command: `Python .../second-binding-run/orchestrate_binding.py --execute`
- PPID 1, `Ss` (detached session as intended)
- Alive at check: yes (`ps -p 57144 -o pid,ppid,etime,stat,command`)
- stdout first line: `Recovery watchdog ready; running owned-container preflight.`
- stderr: empty at check

## Scope of this record

Launch/controller-started only. Not evidence of capture success, profile completeness, or baseline restoration. Per instruction, no waiting for model loading or workload; no retry; no code or container/NPU operation from this side.
