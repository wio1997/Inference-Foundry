# Run600 independent Astra review

Verdict: **SCOPED PASS for exact-artifact Host interval arithmetic.** No service/NPU work was performed. This is not a device-ready DAG, idle-time estimate, removable-time estimate or Product Bound.

## Independent identity and arithmetic checks

Reviewed reducer SHA256:a5099417f0e97453f7152b4185b96a5c3e9f28c4fe5f0181fa9f23ac1ea81c46.
Reviewed host_envelope.json SHA256:25e5ca85bfb9b291335f60e47ce5fde95f0defa5466a0c697f5ed3769ae6c7d5.

Verified all42 input hashes (8 original mark files,32 original call files,1 existing scheduler admission and1 existing phase admission). Independently regrouped raw marks and recomputed all32 rank/cohort envelopes, disjoint call unions, outside-call remainders and final-execute-to-built intervals. All agree. Additionally verified raw Host timestamps are ordered, calls are ordered/nonoverlapping, and every call's execute_mark_count places it between the corresponding execute entry and the next entry. Those stronger checks pass on the current data.

| Cohort | allrank Host envelope ms | rank-local outside-forward ms range | final execute->built ms range | cached count sum |
|---|---:|---:|---:|---:|
|5|2879.432|593.776–862.447|19.575–118.332|181|
|6|3513.892|948.502–1200.686|20.053–111.006|175|
|7|3506.801|952.941–1246.293|19.809–118.286|183|
|8|3213.699|977.849–1253.201|19.797–121.471|162|

The per-rank disjoint _model_forward sum is legitimately equal to its interval union for these files. 'outside_forward_host_envelope' is the complement inside that rank's first execute->runtime_built interval, including Host work and asynchronous queue/dependency effects. It is not evidence of device inactivity, off-CPU sleep or avoidability.

The final execute path triggers handoff before ordinary _model_forward. Its entry->runtime_built interval includes earlier execute processing, validation/handoff and construction, and may wait for earlier work. It is NOT isolated seed time, isolated build time or CPU-only service. Rank-duration spread is not HCCL wait attribution.

The allrank min-start/max-built envelope assumes one shared same-host monotonic clock domain, consistent with the capture setup; this reducer does not certify historical time namespace/offset identity. It is a Host envelope rather than an all8 resource critical path. Never combine its value with a sum of per-call maximum-rank durations.

## Exact meaning of cached.num_output_tokens

Installed vLLM v1/core/sched/scheduler.py constructs CachedRequestData.num_output_tokens from:

    req.num_output_tokens + req.num_output_placeholders

Request.num_output_tokens is len(_output_token_ids); async placeholders represent pending scheduled output slots. Therefore the captured181/175/183/162 totals are **scheduler output-position accounting, potentially including pending placeholders**. They are not necessarily actual generated-token counts, and are not worker-published, SSE-emitted or client-received token counts. They cannot supply handoff p_i, remaining=1024-count, or an external/Runtime prefix ownership join. The source of these fields is stronger than merely calling them 'scheduler state', and should be explicit in findings/model columns. This source interpretation does not establish the actual historical placeholder split, which was not captured.

The request ID/order gate added in the reviewed version correctly prevents count-vector comparisons across permuted slots. The old40-file version lacks this additional admission join and should remain superseded.

## Final gate revision

Reverified all42 current input hashes and all32 rank-cohort ordinal/shape/interval conditions. The revised reducer now requires strictly increasing execute_mark_count, each call inside its corresponding execute-entry interval, and the existing phase admission's per-cohort call count and ordered padded shapes. These close the earlier call reorder/drop/shape weaknesses for this admitted call inventory; actual interval values are unchanged. Scope now explicitly states async placeholders and excludes certified generated outputs.

No numeric error was found. Remaining scope limits are not new blockers for this exact artifact: the script still relies on the prior scheduler/phase admissions, rather than independently reconstructing their full original identities; its input hashing is provenance, not an immutable external trust anchor. Raw mark-kind/monotonic validation could be stronger for arbitrary future inputs, while independent inspection confirmed all current mark timestamps are ordered.

The earlier40/41-input artifacts are superseded by this42-input artifact. Findings numerical table, final-execute interpretation, placeholder warning and non-Bound statements are accurate. The findings first paragraph was also corrected to42 inputs including phase admission and the new ordinal/shape gates; this final wording was rechecked.

## Bound scope

The scope string correctly says scheduler counts are not certified published outputs, outside-forward remainder is unclassified, and not_numeric_bound=true. The final revision explicitly includes the placeholder qualification. Run600 reuses only Run341's own W0 and does not numerically combine Run239 or Run597 timing. It identifies where the next identity/completion capture should discriminate preparation work from ready/wait/control intervals. It establishes no lower bound, no removable gap and no formal performance change.
