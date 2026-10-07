# Run281 draft driver review

Offline read-only 2026-10-07. Read complete driver, CPU workflow result20, v2 shim/contracts, actual pad-geometry CSV rows, and inherited Run251/253 request implementation. No edits to driver, plan, service or frozen artifacts.

**No new device-path blocker found.** Initialization exception goes directly to owned cleanup/restore/recovery; no request is sent against a partial bank. The only candidate requests are A short, B short, B natural complete, retained A short. If any fails, recovery does not re-enter the candidate and uses at most one extra request. First error propagates even after successful recovery. Source installation occurs with no device owners; allowlists include exact before/candidate/original identities so a partial multi-file install can be restored. P249 is only guarded/requested, never stopped or patched. Runtime switch checks owned D idle and H11mode0 before writing mode/observe together.

Shape/dtype checks are supported by the supplied actual CSV: hidden pad BF16 [2,6144]→[16,6144], router FLOAT [2,256]→[16,256]. v2 exact-production AST comparison proves dispatched candidate functions are7bb35e. Native format2 equality/new_zeros still requires device validation; a mismatch must reject, not relax. Private target output is a diagnostic condition, not a proven universal output allocation property. Expected partial-bank initialization failure is recoverable by the declared single epoch.

## Concrete reporting issue

The finally block unconditionally writes retained_stack with H6=True/H5=True/H13=False/target_FULL=True even when recovery itself fails or the initial before-guard fails (terminal_verified=False). It correctly marks diagnostic failed, but these stack flags then describe intent rather than observed residency. Before freezing, make unverified retained state explicit (null/unknown or an intended_stack field) whenever terminal_ok is false. Do not let independent reduction interpret those booleans as recovery success. This does not require another model request.

## Required frozen plan/artifacts

* H13_sources must contain exactly the intended prepare/runner replacements **and rotary cleanup entry**: rotary candidate==original b6 baseline, before==resident e289 observer. Pin each artifact hash and target path. Recovery runner original must be current4c helper, not an earlier generic baseline; prepare original54e; preserve H6/H5/proposer6a and current H9off wrapper. Avoid contradictory legacy runner/target fields: guard overwrites legacy runner mapping with H13_sources, so explicitly verify final composed mapping in both epochs.
* before_source_hashes, protected_source_hashes, resident roles/start ticks/host namespaces, library/backend/schema identity, H5 helper hash and exact existing roots/modes. Freeze candidate/recovery native args: targetFULL bucket2, compileNONE, TP/DCP16/SPoff, current async/Noop, H11mode0 and H12physical baseline. No P-side source/config mutation.
* h13_mode.bin initially [0,1]; H9/H11 modes zero; MC2 one; new empty candidate/recovery witness namespaces and epoch directories. Immutable witness/source identities must be owned by this run. Driver witness checks all16 rank/pid sets, pool/output identity and geometry. A stronger rank→pid binding requires a trusted rank mapping if available; current sets prove membership, not independently permuted pairing.
* Exact short8 and complete23 prompt/token/content/stop fixtures, P/D routes, unique run-id+label salts, accepted helper/PD client source hashes. Pin the imported Run253→251 client because its exact short-token assertion and external-hit check are not duplicated in this driver. Complete workload counters are evidence, not an exact acceptance-signature correctness requirement.
* Four normal/five failure-inclusive request limit, one recovery epoch, fixed timeouts, cleanup scope, terminal decision and no-performance claim. Freeze driver/shim/CPU/production-equivalence hashes together, not just production7bb.

## Evidence limits / smaller checks

Recovery warm invokes the accepted exact short/KV-hit client but does not call this driver's all16 transfer_gate; candidate's four requests do. If frozen recovery contract explicitly promises all16 transfer evidence, add the existing-log read for that recovery request without another generation. Otherwise label recovery's evidence accurately as exact output + existing aggregate native KV evidence + all16 stack witnesses.

Before the first request, ready() relies on engine initialization success; bank capture assertions and immutable capture witness are generated in that initialization. A read-only all16 capture-witness/source check before the first request would make admission explicit but should not demand runtime/H11 witnesses that only exist after a request. Do not use witness() as that pre-request gate unchanged.

H12 observer cleanup is common baseline preparation, not H13 gain. Four correctness requests do not establish performance. Run277 stays FAILED, Run279 correctness and Run280 INCONCLUSIVE remain distinct historical outcomes. No additional profile or parameter scan is justified by this review.
