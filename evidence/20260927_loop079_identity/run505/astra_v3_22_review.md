# Run505 — independent V3.22 Bound-ledger review

2026-09-27. **PASS / ACCEPT the final pinned V3.22 as a conditional evidence-ledger update.** No new finite Resource, Hardware, Scheduling or Product endpoint, proven output writer, private-stream completion, or numeric Current-to-Bound gap is admitted. Formal Current remains **571.681 tok/s**. This verdict applies to the exact final hashes below, after the two corrections described here.

- Generator `scripts/extreme_bound_calibration_v3_22.py`: `864fa4890bb6c7e8ba6ca2d3cd2c849e0cd1350137c86e59252635105a9d463f`.
- Run504 `bound_calibration_v3_22.json`: `994a4aa69be470e4ad3124e7a0ae08e272fa979a88fb4509753c7c345208ed3d`.
- Reviewed baseline Run489 V3.21: `269bf2182ef61bce200470b671a50e4124689156a1350c384ecf1f0be0612bf8`.

Scope: read-only source/evidence inspection and CPU Python checks using temporary input copies. This review wrote only Run505 evidence and temporary test files; it did not start/query a service, execute NPU work, install patches, modify shared scripts/model/TaskCtl, or perform a new acquisition.

## Independent checks

`cpu_review_results.json` contains **41 passing assertions and one explicit guard-coverage characterization**. `astra_audit.py` makes the checks reproducible; `reviewed_generator.py` preserves the exact audited generator.

- Regeneration is byte-identical to Run504. The standalone CLI also reproduces the bytes from a minimal temporary tree with three Python modules and eight pinned direct inputs. Its output is confined to the temporary tree.
- All eight direct evidence hashes match; independently corrupting each input is rejected, including rank0 graph metadata. All **145** current input files in the frozen Run503 manifest independently match their recorded byte sizes and hashes, without a fallback file substitution.
- All **19** proof-DAG nodes independently recompute to the stored all-false state. The new work/traffic/capacity, necessary-floor, removable-time, actual-update-iteration, external-event-generation, typed-writer and terminal-to-caller fields remain null or explicitly uncertified as appropriate.
- Semantic negative fixtures deliberately update only their temporary expected hash so they reach the content gates. Changed Current, a true cached proof node, failed first-position source result, 127 rather than 128 patterns, failed validation/final admission, 39 slices, seven ranks, absent meta join, wrong task/stream counts, DSA rather than MLA backend, before branch, wrong configured stream and missing successful selected return all reject.
- Each of the six recognized numeric endpoint keys rejects a finite value. Shared generator/model bytes were checked again at audit end and remained unchanged.
- The complete V3.21→V3.22 delta is the three new bound-ladder records, one scheduling certificate record, input paths, revision and next-measurement text. Existing proof nodes, resource assumptions, endpoint values and scheduling caveats are preserved.

This audits a fixed, hash-pinned ledger transformation. The generator checks the manifest's hash rather than recomputing all its 145 entries; the independent check above supplies that verification for this review. It is not a general replacement for Run502 acquisition validation or for future evidence admission.

## Corrections incorporated before PASS

1. The initially inspected generator read rank0 graph metadata without including it in `HASHES`. The final version pins it to `a0083baa46b93affe7d7886ca9a4edb35e952281b19c97e5c0113aff9c7c0d98`; its corruption negative now rejects.
2. The initial Run502 node copied **45.986019 ms** from Run487's separate Run484 acquisition as `conditional_replay_envelope_ms`, without an explicit local provenance qualifier. The final version removes that numeric copy and explicitly states that the Run487 envelope is **not transferred to Run502**. The original Run487 observation and its restrictions remain in their existing standalone record.

No remaining mandatory correction is identified for this frozen ledger update.

## Resource and Hardware boundaries

Run497 proves the frozen CPU greedy-path first-position identity over 128 seven-draft equality patterns. It does not bind an actual request/generation through Runtime clipping, Scheduler pre-append admission, output publication and client receipt inside the formal window. V3.22 correctly keeps external retention/freshness uncertified and compulsory work null.

The **8,388,608** operation count is a conditional ordinary dense 4096×1024 projection calculation, not a positive unavoidable W-minus certificate. Loaded branch/kernel identity, fresh execution, allowed reuse/recomputation, legal implementation class and matching formal output-window scope remain obligations. The inherited strict DAG retains the positive-work, true-capacity and units/scope joins. The CPU lemma cannot close them.

The **5,412** tasks, stream census and **43** MEMCPY_ASYNC tasks per rank describe the current exporter graph. The dump lacks copy byte/source/destination information, so traffic stays null. Counts and attained service cannot establish an exact-board aggregate maximum capacity C-plus. No Hardware or Resource gate was dropped.

## Scheduling boundaries

The new node accurately summarizes Run503: valid same-acquisition selected graph/output ownership and actual `AscendMLAImpl.update_graph_params` binding, with four output **candidates**, unresolved actual update iteration counts, event/handle generations and terminal-to-caller continuation. It does not substitute configured stream102 or a successful outer return for effective work/completion. An after-submission update may release an in-flight replay wait; next-round-only preparation is not proved.

The candidate DAG retains its Task Id order/event-suffix assumptions. Neither all tasks reaching stream1 terminal notify nor a maximal address occurrence proves a typed last writer. Main RmsNorm remains ABI-supported; aux ReduceMean MIX/placeholder positions remain unresolved. Run503's additional boundary still applies: main/notify have no path back to already-ended internal graph stream0; equal stream numbers do not identify caller R1. The null terminal-to-caller field correctly prevents that promotion.

Run502 is a **post-drain graph-state** acquisition. It does not certify cycle64 dynamic argument values, a measured duration, unmarked Current exposure or a common all-rank clock. The final ledger's explicit non-transfer of Run487 time prevents combining different acquisitions into a measured producer envelope. Source statement ordering cannot be added into latency.

Necessary-path lower bounds still require unavoidable typed dependencies, relevant cross-rank/storage edges, justified node-duration lower bounds and formal arrival/handoff/publication-window ownership. Attainable schedules additionally require feasible storage/resource use, mixed compute/HBM/HCCL contention, concurrent attainable service, changed useful-token/cycle behavior and repeated correct formal E2E results. These obligations remain in the inherited DAG and scheduling missing lists. Marker overhead and matched trajectory controls remain unresolved; no scheduling gate was removed by the shorter new-node summary.

## Guard limitation and next step

The imported `require_null_endpoints` recognizes six older key names. It does **not** reject an arbitrary tree containing finite `necessary_path_floor_ms`, `removable_product_ms`, `compulsory_work_subset_operations` or `exact_board_aggregate_capacity_upper`. The frozen generator explicitly constructs these new fields as null, and its prior/input bytes are pinned, so this limitation does not invalidate the generated artifact. Before reusing the guard for arbitrary future model admission or numeric promotion, add a schema covering every new proof/endpoint field, validate types/units/scope, recompute the DAG and require the relevant actual certificate. The printed `finite_endpoints: 0` is not by itself an independent validator.

The recorded next step is appropriate: first source-only exact ReduceMean argument ABI and model-terminal semantics; then, only for unresolved joins, a separately preflighted minimal same-generation update/event/handle and typed-output-producer acquisition. Preserve zero-key/zero-zip distinctions, reset/replay generations, loaded-native correlation and overlapping writes. No extra waits or raw stream getters. This review accepts the evidence ledger; it does not authorize or preflight such an experiment. Resource W-minus and authoritative C-plus remain independent open tracks.
