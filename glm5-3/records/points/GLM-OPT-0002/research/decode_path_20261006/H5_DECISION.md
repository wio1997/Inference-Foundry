# H5 single-stream MoE events: measured result and promotion boundary

Sol reviewed the five-file patch, producer/consumer sources, original Run249 queue/CANN joins, native rank records, Run253 raw SSE/arrivals and model witnesses. This is a new decision; original planned manifests and running Job handoffs remain unchanged.

The fixed eager W8A8 configuration disables shared-expert stream overlap. Four explicit records and three shared-consumer waits per MoE submit dependencies between operations already ordered on the same current stream. The helper returns None in that configuration; true overlap retains the original event operations and checks its required input dependency. Necessary HCCL/DCP/MC2 internal joins, output-copy synchronization and arithmetic remain intact. No LoRA or general live config-refresh correctness is claimed.

CPU five-file AST normalization and false/true/None/missing-input checks passed. Native actual imports and 16×11 cases passed: independent byte values, stock/candidate equality, input preservation, retained output lifetime and a real auxiliary-stream true-overlap dependency. All64 resident worker mode witnesses match their owned namespace PIDs/config and event return policy. Complete and short PD outputs retain exact token IDs/chunks and full external KV hits.

| Same-resident pair | Complete generation TPOT | Reduction | D request wall | D reduction | Total PD wall |
|---|---:|---:|---:|---:|---:|
| A1 → B1 |229.346 →218.544ms/token|4.71%|5.451206 →5.187993s|4.83%|5.922158 →5.587866s|
| A2 → B2 |226.595 →219.210ms/token|3.26%|5.382251 →5.198689s|3.41%|5.771890 →5.586090s|

All four complete requests return the same sentence,23 token IDs, chunks1 followed by eleven2-token chunks, natural EOS stop. Same messages/max96/thinking0/temperature0; unique cache salt per request prevents earlier samples satisfying the new prompt locally. All58 prompt tokens hit external KV. P249 unchanged; D253 loaded once; A/B/A/B in the same16 workers; H4 gather patch absent; profiler OFF throughout. Warm samples are excluded.

D generation first-last span shrinks237.642/162.476ms. Baseline A1/A2 span drift60.525ms (TPOT2.751ms), patched B1/B2 drift14.641ms (TPOT0.665ms). D-wall deltas263.214/183.561ms both exceed observed baseline D drift68.956ms. These are repeated positive code evidence for the declared bounded fixture. This small observed drift is not a statistical confidence interval. Short8 medians250.891→243.907 and264.363→244.450ms/token show2.78/7.53% reduction, but the first short pair is within wider baseline phase drift and is not the promotion basis.

Total PD shrinks334.292/185.801ms. P alone contributes71.079/2.239ms. Do not credit total5.64/3.22% PD reduction wholly to D or infer a cumulative gain with H4. D generation is measured between arrivals on the same client166 clock, excluding P; no P/D profiler timestamp subtraction is used.

**Decision: retain the patch and positive scoped code evidence; INCONCLUSIVE/PARKED for formal PERF_KEEP and product promotion.** The full standard dynamic workload/SLO contract has not been accepted, and this fixture's mean TPOT remains about219ms against the product18/40ms percentile goals. A valid full sentence and two comparable matched pairs do establish the bounded mechanism; they do not establish SLA capacity, API completeness or product Current. This decision is not caused by unrelated API backlog, an invented three-pair requirement or absence of a new heavy profile. No further identical repeats, large workload or scan are queued merely to accumulate samples. Current=None, active PERF_KEEP stack empty. 本阶段没有新增代码级性能 KEEP。

The actual baseline is a diagnostic shim: both modes use the new None-safe consumers; stock modes record original events, patched modes call the pure candidate helper. Helper AST and other four files match pure candidate artifacts. Selector overhead is shared, and pure-patch deployment without instrumentation is not measured. Finally selector0 and all five original disk sources were restored. Fresh read-only verification at15:48:54Z found P249/D253 healthy/idle,32 owned NPU workers,67 original relevant source hashes, and exited controller/phase. Resident D keeps the selector in memory with stock event behavior. Patch is not active in the product.

Evidence: [patch](moe_event.patch), [identity](moe_event_patch_identity.json), [native and original timing reducer](reduce_event_compare.py), [Run253 summary](../../runs/GLM-RUN-0253/execution_summary.json), [fresh site](final_site.json), [independent source review](ASTRA_MOE_EVENT_REVIEW.md). Original raw is indexed in evidence_index.json; full logs/source artifacts remain at their server paths.
