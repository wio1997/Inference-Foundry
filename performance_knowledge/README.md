# Performance Knowledge

This is the decision layer between raw Runs and new performance hypotheses. The historical source is the private [`dsv4f-w4a8-ascend-logs`](https://github.com/wio1997/dsv4f-w4a8-ascend-logs) repository, pinned in `sources.json`. Its Round verdicts describe their **own** workload and code. They are priors and counterexamples, never an Extreme KEEP/REJECT or a throughput bound.

## Candidate preflight

1. State the proposed mechanism and current shape, dtype, TP/DP, Runtime owner, Graph mode and expected critical-path edge.
2. Search the pinned Round summaries with precise terms, for example:

   ```bash
   python3 scripts/performance_knowledge.py sync
   python3 scripts/performance_knowledge.py search 'compressor wkv overlap' --limit 5
   python3 scripts/performance_knowledge.py search 'dspark draft graph' --limit 5
   ```

   The checkout stays outside Foundry at `/data/wio/performance_history/dsv4f-w4a8-ascend-logs` (override with `PERFORMANCE_HISTORY_REPO`). Search reads only `TASK.md`, `REPORT.md` and `RESULT.md`. Open the relevant Round's `evidence/RUN.md`, analysis and raw material only when a claim needs verification. `sync` uses the pinned commit; changing the pin is an explicit catalog update.
3. Record in the candidate design: query and source revision; old test conditions; mechanism; measured result; demonstrated failure/root cause versus inference; resource contention or why local gain did not reach E2E; changed conditions that warrant a new test. If no relevant Round exists, record that briefly.
4. Use the current Dual-Bound DAG to check whether the candidate changes necessary work, a legal schedule, useful tokens/cycle or serving wall. A historical verdict does not replace current same-state correctness and repeated frozen formal E2E.
5. After a new Run, add or update one concise `entries.jsonl` record with source Run/evidence and the observed applicability limit. Validate with `python3 scripts/performance_knowledge.py validate`; link it from TaskCtl and the current Performance Map/Bound when it changes a decision.

The first five entries cover the currently active DSA overlap and owner-compact questions. They were selected **on demand**; this catalog is not a summary of every historical Round. `entries.jsonl` records factual observations and test conditions. A `status` of `prior` is historical only; `current_diagnostic` does not assert Product gain.

## Record fields

`id`, `topic`, `mechanism`, `source` (repository, immutable commit or Run ref, relative path), `environment`, `observed`, `failure_or_limit`, `revalidate_when`, `extreme_relation`, `status`. Prefer the smallest evidence set that supports the claim. Write “unknown” when root cause or exposed E2E effect was not measured. Do not turn profiler overlap rectangles, kernel microbench gains, or historical KEEP/REVERT into a current Product bound.
