# Independent conditional matched-draft review

2026-10-07. Offline source review only. Inspected `mtp_graph_matched_draft/pd_compare.py`, actual Run272 diagnostic request/RPC/witness/transfer/recovery consumers, and inherited Run251/253 request implementation. No script import/execution or service operation. This draft lacks its future functional_plan.json locally; plan pins/cache collector implementation cannot yet be fully checked.

## Verdict

The conditional same-worker A1/B1/A2/B2 design is reasonable **after** actual correctness passes and candidate residency is proven. It is not ready to call final until the terminal-verdict ambiguity and cache-condition checks below are resolved. No current Graph correctness or gain is inferred while Run272 runs.

## Concrete findings

1. **Positive result can outlive failed terminal health.** `decision.json` is written with POSITIVE_LIMITED_RESEARCH/H11_research_stack=true before terminal guard/switch/warm. If terminal health fails and recover succeeds, retained_stack says H11=false but embeds the old positive decision; the script then raises. A consumer reading decision.json alone can wrongly promote H11. Preserve measurement evidence, but publish a separate final/effective verdict only after terminal witnesses/guards, and explicitly invalidate retention eligibility on recovery or finalization failure. Comparison completed is not the same as safe retained stack. Existing Current=None/formal_product_KEEP=false is appropriately scoped but does not eliminate this inconsistency.

2. **Declared matched cache workload is not fully asserted here.** Complete requests assert both local hits zero and D external queries=hits=prompt_tokens. They do not assert matching/nonnegative local query counts, P external conditions, or monotonic/raw counter-reset validity. The cache collector may supply some checks, but its implementation/path is not supplied in this draft directory, so this remains unknown. Freeze and verify actual per-endpoint delta validity and the intended query equality rather than treating hits=0 alone as equivalent cache work. Do not demand an invented 93% condition: this comparison is explicitly fresh-salt/local0.

3. **Failure bounds need their own request allowance.** Normal path makes nine requests (four warm+four complete+terminal warm). A failed terminal warm can be followed by one recovery warm, so the recovery-inclusive cap is not nine. Freeze separate normal and failure maxima; do not silently add another recovery if the recovery warm fails. Current code makes only one recover() call, and epoch guard prevents reusing an already consumed baseline_recovery epoch.

## Root/RPC and recovery interfaces

- Imported diagnostic keeps `d.ROOT`, `d.PLAN` and rpc's `__file__` at Run272. Remote RPC subprocess therefore uses the original resident epoch/selector/source identities, intentionally.
- `d.x.ROOT=m.ROOT=ROOT` moves request raw/result writes to the new comparison root. Actual inherited x.request uses x globals; its cache_salt is `PLAN['run_id']+'-'+label`, and changing x.PLAN run_id makes this comparison's unique labels distinct from Run272. d.request retains original golden/content checks and MTP counter parser, which is appropriate when fixtures are unchanged. `oldclient.ROOT` is not reassigned, but this request chain uses x.request, not oldclient's workflow; no demonstrated old-root request write was found.
- RPC transfer_gate reads the original active epoch native log and checks exactly16 success ranks for remote_request_id plus absence of transfer failures. It does not merely trust HTTP output.
- Recovery invokes specific cleanup/restore/recoverymode/launch/poll actions, not the parent workflow/controller. Actual restore covers proposer and runner files; retained H6/H5 stay in the inherited native launch environment. Recovery uses the unused Run272 baseline_recovery epoch and requires candidate epoch first. New controller ownership is checked before entering workflow, but delegated parent actions themselves retain their original plan; freeze that ownership reuse explicitly.
- d.ready writes readiness through reassigned m.ROOT into the comparison root while its RPC polls the original lifecycle root. This separation is intentional and should have a CPU root-routing regression test to protect future refactors.

## Measurement and promotion boundaries

Counter snapshots and transfergate run outside x.request's P/D timing window. Warm request plus witness should consume the two heavy observer rows before complete measurement when each mode transition actually occurs; require that transition witness, not just an old mode witness. Actual witness selects the latest transition but does not itself prove it belongs to this warm; record/check transition monotonicity or mode/warm linkage across A1/B1/A2/B2. No current violation is asserted.

Inherited complete request checks exact IDs/content/stop/prompt size. Workload signatures compare scheduler aggregate draft/accepted counters, not ordered acceptance trajectories or SSE chunks. Wording must retain that limitation. Require both paired D and PD savings above their own A drift and positive TPOT differences as written; do not cherry-pick a pair or rerun for favorable signatures. Finite valid timings and counter/reset validity should be explicit in the frozen reducer.

POSITIVE_LIMITED_RESEARCH would support this narrow retained research stack only after successful terminal guards. It is not formal API/SLA/80K/600/93% proof, not Current promotion, and not cumulative stock-versus-H6/H5/H11 gain. No performance Run is initiated or authorized by this review.
