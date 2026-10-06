# Run256 — completed bounded MC2 capability-cache comparison

The frozen 21-request supplement completed with controller phase exit0 at 2026-10-06T19:57:30.765341Z. This is one independent A1/B1/A2/B2 bracket; Run255's incomplete bracket remains separate. No new NPU profile, scan, kernel arithmetic, routing, collective or parameter change.

Both modes passed short8 and naturalEOS23 golden correctness before timing. Every complete request returned the same 23 IDs/content, prompt58, stop and58 external KV hits. The four timed complete requests all recorded11 drafts/11 draft tokens/11 accepted tokens/position0=11/invalid0. Cumulative work matches; exact acceptance ordering and target-only/kernel counts are not established. SSE groups also match here, but are not a correctness invariant.

| Preassigned pair | D TPOT ms/token | D wall seconds | Complete PD wall seconds |
|---|---:|---:|---:|
| A1→B1 |225.711765→194.973060 (13.62% lower)|5.379359→4.607018 (14.36% lower)|5.752673→5.090636 (11.51% lower)|
| A2→B2 |223.946609→192.152319 (14.20% lower)|5.314716→4.565718 (14.09% lower)|5.787416→4.953456 (14.41% lower)|

D savings0.772341/0.748999s both exceed frozen D drift0.064642s. P contributes−0.110304/+0.084961s respectively to total PD savings and is accounted separately. TPOT is independently recomputed from raw first/last token arrivals; D wall includes stream completion after the last token. Two timed short8 observations per phase are all retained in [independent reduction](comparison_reduced.json), with four drafts/four accepts; warm and correctness requests are excluded from those timings.

Each phase's native witness proves the same16 workers/start epochs and the same rebuilt native SHA0dc998… with the expected mode. This common binary isolates repeated predicate versus fixed-lifetime cached predicate; it does not measure a fresh stock-binary versus production-binary pair. The production change is [two local static bools](../../research/decode_path_20261006/mc2_capability.patch). The diagnostic binary adds only the observed mode selector around that mechanism. CPU import/ABI/schema, concurrency and negative capability paths are preserved in the referenced build/import evidence.

Final recovery completed in a separate stock-native D epoch321785/start306162978; all16 mappings use the original library whose SHA is83fb9a0e…. P249/root1916718/start303049910 remains unchanged. Protected sources, health200 and idle checks passed. Candidate was never installed into the original Python/native environment. [Restoration](restoration.json), [native mappings](stock_recovery_native_witness.json), [guards](guards_after.json), [phase](mc2compare.phase.json) and [original artifact identities](artifact_identity.json) are direct evidence. The original PLANNED manifest and handoff-running result remain historical inputs; actual completion is recorded separately.

Frozen decision: **scoped POSITIVE repeated complete PD mechanism evidence**. H6's bounded research question is closed; formal promotion is **INCONCLUSIVE/PARKED**, Current=None, active stack empty. Full dynamic API/standard workload/SLA capacity is unaccepted, and~192–195ms/token is far above the existing18/40ms SLA targets. Do not add H4/H5/H6 percentages or promote their untested stack. 本阶段没有新增代码级性能 KEEP。
