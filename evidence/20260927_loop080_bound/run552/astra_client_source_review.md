# Run552 — independent Run551 client and source-map review

2026-09-27. **PASS for the client chronology component and scoped source map; not a live readiness PASS.** Final Run551 source-preflight SHA256 **63a301870d36ad4a4c9ab7ecb27721f9adc894f882ccf33d653ec60f1a57bba3**. Client SHA256 **12fb38f53d5927f3731e6584d9a1ef0c5715bc343972021a15c14938d4d57872**; validator SHA256 **def4614d4d385c3c8ba62f622cfc64f515abdcf416daa3f8ace0c10d44918677**. No service, HTTP network, NPU execution, production-source mutation or performance claim.

## Independent checks

`audit.py` creates a CPU-only audit wrapper around the real asyncio.Semaphore implementation and fake SSE transport. For48 requests it observes the actual acquire-return and release-call intervals and verifies they lie within the client's timestamp brackets. The wrapper calls the original semaphore implementation; it is used only in the test, not proposed for production. All48 acquire/release bracket checks pass. Five extra marker mutations reject: acquisition after POST start, DONE after end, release upper bound before end, invented unique parent and noninteger clock. The existing validator's1 positive/7 negative cases pass. An explicitly wide release bracket that overlaps later acquisitions remains valid without inventing a parent.

Independently reran the17 Run528 fake-transport cases against the new client: expected exits, exact payload conservation and terminal-output handling all pass. The request-body function AST equals the prior validated client's AST; the transport review also compares the body with frozen `bench.py`. Changes are timestamp reads and fields around the existing semaphore/SSE path. They add no new await, request option, serving body field, semaphore implementation or in-wall file write. They can still perturb event-loop timing, so output remains instrumented diagnostic evidence.

Artifacts: `independent_checks.json`, `actual_semaphore_cpu.json`, `transport_cpu.json` and test logs. No completed NPU experiment was repeated.

## Correct interpretation of the markers and validator

For a completed successful request, the source places acquire-attempt before entering the async context, acquired after its acquire returns, stream-DONE at receipt of the fragment completing DONE, end before leaving the semaphore context, and release-completed-by after its `__aexit__` returns. Thus the actual semaphore release is bracketed by `[end, release_completed_by]`; end is a lower bound, not the release time. The acquired mark is after actual acquisition. The DONE mark is fragment receipt, not server transmission time. The independent test confirms these placements on the exercised ordinary path; cancellation/aborted phases do not become valid samples.

The validator counts intervals `[acquired,end]` that are certainly occupied. Its `guaranteed_active_peak` is the peak of those intervals, **not the exact peak of all possible permit-holding intervals**. A peak above12 disproves the record; a peak≤12 alone does not reconstruct exact semaphore state or prove a uniquely feasible transfer. Source ownership supplies the use of a real Semaphore(12). Do not relabel this output as exact release occupancy.

For each acquisition beyond the first12, the returned set contains earlier requests whose end lower bound is no later than acquisition. This is a **conservative candidate superset**, not a set of individually proven transferred permits or a simultaneous assignment of permit edges. Release may occur later within its bracket; some candidates may be removed by a tighter global feasibility analysis. The validator correctly reports zero certified unique-parent edges and does not choose a predecessor from timestamp rank matching. A later Scheduling solver must use the brackets/candidate relation plus collective permit conservation, or retain the uncertainty; it cannot pick every favorable edge independently.

This validator is a chronology component, not a substitute for the existing frozen-client admission. It reads summary rows, does not independently replay raw SSE/body/dataset/phase identities, and does not join server requests or clock domains. It must run alongside the existing full client/server/phase gates. Those gates establish the exact48 dataset rows and body, terminal usage/DONE payloads, response identities, phase barrier and cross-process Host clock identity before its candidate sets enter a product ledger. Neither script promotes instrumented durations to Current, device-ready time, strict Resource work or a feasible Scheduling bound.

## Source-map and completion-contract review

The inspected installed Runner and Runtime hashes match the pins. The map correctly separates original `_model_forward`, ordinary sample/proposer, CPU draft-ID delivery, handoff construction, first Target input preparation, acceptance/state/proposer and eventual cohort drain. The target layout is **12 slots×8 positions=96 Target rows**; read the document's “12×96 handoff” wording in that sense, not as1152 rows.

`propose_draft_token_ids` stores the draft tensor and invokes `_copy_draft_token_ids_to_cpu`; that invocation by itself is not a verified CPU visibility or device/KV completion certificate. The exact configured branch, producer stream, copy/event generation, Host consumption wait and next handoff use still require mapping. A caller-stream event after `_model_forward` similarly says nothing about an unjoined private KV/cache/HCCL producer. The source map correctly leaves these open.

The prior-slot history threshold is not output readiness. Histories are copied to CPU only after cohort completion in the existing path. For an earlier per-slot certificate, map the actual selected acceptance/history-copy generation, preserve the retained prefix and its source ordering, and bind a completion marker or honest interval to those producers. The chosen terminal cycle must be derived from the new run's q/G history; a fixed historic cycle number is only a sampling hint. An event after a history copy does not on its own make KV/state safe to recycle or establish an API publication path. The current one-cycle-lagged Host progress mirror cannot replace the device completion witness.

## Next smallest implementation gate

The client component is ready for composition into the eventual guarded diagnostic. It does not justify starting that diagnostic by itself. Next implement the minimal selected Runtime-cohort5→6 (measured1→2) lineage hooks and reducer with two explicit pieces:

1. One selected prior-slot retained-prefix generation and immutable-storage/completion interval, leaving early publication and safe slot/KV reuse unknown unless separately proved.
2. The chosen successor's actual cache/residual admission and ordinary sample→draft-ID producer/copy→existing consumer-wait→handoff→first Target input-generation chain, with per-rank owner and stream/event identities.

First preview exact source anchors and hashes; test missing/wrong generation, rank, FULL string, event scope and false readiness promotion on CPU. Do not add waits to make an unobserved edge appear complete. If a side-stream endpoint cannot be certified, retain the valid Host/lineage fields and keep that timing edge null. Before live use, independently check the actual-container imports, reversible install/restore, controller phase/cleanup guards and no-mark controls required for any timing transfer. The missing selected completion contracts and their executable collector/reducer are the smallest remaining implementation blocker; a new formal performance sweep or broader kernel scan is not required to resolve it.

All strict Bound endpoints remain unchanged. The work can improve release/dependency evidence; it does not certify fresh F, hardware C-plus/B, an alternative schedule or E2E benefit.
