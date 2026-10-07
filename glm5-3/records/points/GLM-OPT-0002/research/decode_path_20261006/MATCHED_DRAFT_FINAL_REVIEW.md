# Conditional matched-draft final independent review

2026-10-07. Offline only. Verified `pd_compare.py` SHA256 `bd0a11f423efe427a704f9cb8b9ac776981e347e8136c6f2ba45c24af33ae3c8`; read current design, script, workflow/recovery AST test, decision test results, actual cache collector, and inherited request timing/RPC consumers. No import/execution of operational scripts or service requests.

## Verdict

The four prior findings are addressed. **No new concrete measurement/control-flow blocker found in the inspected draft.** This remains conditional: Run273's independently reduced raw correctness PASS and candidate residency are mandatory; Run272 FAILED is not a substitute. The future functional_plan/pinned collector and diagnostic paths are not yet present in this draft directory, so final plan identity is a freeze-time check, not verified here.

1. Comparison evidence and effective retention are now separate. Initial decision is PENDING/H11=false. Final publication follows terminal warm, witness and both guards. Terminal failure/recovery and comparison errors invalidate H11; recovery failure also writes INVALIDATED before propagating. A measurement-positive evidence file is intentionally retained but no longer masquerades as the final decision.
2. Actual strict collector rejects absent/ambiguous/incomplete, nonfinite/noninteger/negative counters, changed engines and invalid decreasing deltas. Complete requests require P external0, D external queries=hits=prompt and both local hits0. Four complete endpoint query/hit signatures must match before POSITIVE. This establishes the declared fresh-salt/local0 scope, not formal93%.
3. Actual167 recovery files are checked read-only; normal10 and recovery-inclusive11 requests are explicit. There is one recover call site and one recovery warm, with candidate epoch required. Recovery is not retried after failure.
4. Fixed graph priming produces a new transition before A1. Each warm requires exactly last+1 across16 ranks; each complete requires the same transition as its warm. This prevents treating the diagnostic's old final-mode witness as new warm evidence and ensures the first two heavy observer rows are consumed outside measured complete calls.

## Routing and timing

The inherited d.rpc uses diagnostic script __file__/ROOT and its original operational epoch. Only x.request/m.write roots are moved to the new comparison output root, with a new plan run_id salt prefix. It invokes specific guard/switch/witness/transfer/recovery actions, not the old controller/workflow. The design explicitly allows operational selector/epoch/live evidence to advance while preserving historical diagnostic results. Pin Run273 as diagnostic_root and ensure its unused-recovery check is the one used.

Actual request timing starts just before P HTTP and ends after D stream completion; cache/spec-counter snapshots and transfer checks are outside that interval. P wall ends at P response, so D wall includes the small existing client metadata/HTTP preparation interval as well as D service/output. This is consistent across A/B but should not be called pure device decode time. TPOT derives from token-bearing SSE arrival span divided by output token count minus one, not individual engine-step durations. Exact content/IDs/EOS and aggregate MTP counters remain required; aggregate signatures are not an ordered acceptance trace.

Both repeated D and complete-PD savings must exceed their respective A drift, and TPOT must improve in both pairs. NaN/nonpositive timing inputs are rejected. No favorable-pair retry, no stock-stack percentage addition, no inference that a difference in P latency is MTP code gain. This is a narrow two-pair research decision with limited noise characterization, accurately labeled.

## CPU evidence limits / freeze boundary

Reviewed six workflow scenarios and decision counterexamples. They execute real workflow/recover AST but replace lifecycle/request operations with doubles; they prove the revised terminal decision logic and bounds, not remote ownership, timing, counter publication latency or runtime graph execution. The tests' root-routing assertions cover reassigned client globals; actual inherited x.request source supplies the concrete write-path evidence.

Before any execution, freeze the supplied script hash, strict collector hash/path, Run273 diagnostic bundle/source identities, new unique run_id/output directory and controller ownership. The script is intentionally generic and relies on that frozen plan; design text alone cannot force diagnostic_root to273. If Run273 remains failed/incomplete or has consumed recovery, the prerequisites must reject without measurement. This review does not assert those prerequisites have passed.

H6/H5 remain retained regardless of outcome. POSITIVE_LIMITED_RESEARCH is conditional active-research evidence only; Current/formal_product_KEEP remain unset/false. No performance Run is started or authorized by this review.
