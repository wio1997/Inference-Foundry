# GLM-RUN-0242 — one bounded H1 diagnostic

Type DIAGNOSTIC, not formal performance. Reset/review: GLM-OPT-0002/research/engine_commit_20261006/RESET.md and review.md. Historical grammar/cap/cadence directions parked. Parent remote b23845e4.

Hypothesis: under current cap2/resources3, a FIFO result published before an extra schedule is delayed by enqueue priority and delays actual user output or eligible Decode.
Distinguishing evidence: same-host monotonic core schedule/execute/sample receive/FIFO commit, all16 nativeworker copy/reply and PP accepted+draft publication/consumer/metadata events; optional native torchNPU profiler on one 4x64 diagnostic window; native streaming IDs/usage/DONE. No new device synchronization. Actual output-copy and MQ publish timestamps are upper bounds; future.done is never read by hook. Startup/restore and profiling overhead are not steady state E2E measures.

Decision table: complete FIFO replies published before extra schedule, positive commit→effective output delay -> conditional single native control candidate; evidence proving replies unready/commit not on needed Decode/output chain -> close H1; lacking timing/rank/device/output coverage -> INCONCLUSIVE and no large E2E. Do not infer negative readiness from a late synchronize return.

Scope: unique existing controller only; trace-only fresh D0 epoch temporarily, fixed native math/resources/PP/K2/cap2/issue8192/threshold1024/cadence1; D1 retained. Mark oldD0 epoch retired without migrating STORE. Finally restore original D0 code and V14 public with freshly compiled native domains, preserve D1 STORE (two completed native fixtures compared before/after) and all old raw. No original plugin/installed source edits. No drain-first/performance patch/cap scan. Correctness CPU native method/Future FIFO, empty/deferred grammar/exception semantics passed before server work; real request correctness audited in this Run.

Budget one diagnostic, max4 requests64outputs each, controller timeout5400s including two model startups and guaranteed best-effort restoration. Not evidence for stable capacity, full product gain or Current. Unique ids/files, no replay.
