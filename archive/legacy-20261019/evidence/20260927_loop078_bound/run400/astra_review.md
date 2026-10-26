# Astra independent review — Loop078 Run400

Verdict: ACCEPT as the revised source-pinned partial dependency inventory. The parking condition and buffer-reuse proof obligation are now explicitly recorded; neither resolves the missing timing or typed-edge semantics. It is not yet a complete executable DAG or a numerical Scheduling Bound. All duration floors and finite Product ceilings must remain null.

## Evidence checked

All five revised ledger source SHA256 values match current files, including fixed_decode.py (7d74f4...a6bafa). The pinned framework acl_graph.py also matches the file installed at /vllm-workspace/vllm-ascend/vllm_ascend/compilation/acl_graph.py inside the service container (6396ca...db3f6). Read the Runtime step, proposer count-copy/commit/parking methods, serving loop, graph replay condition, plus target_adapter.py and fixed_decode.py. fixed_decode.py is now pinned. target_adapter.py remains a relevant inspected dependency outside the five-source ledger and should be pinned before claiming source-complete Target data edges. No service/NPU execution or source mutation was performed.

## Correct structure and the two recorded constraints

The source supports Target→acceptance→state_advance→Draft preparation/model→draft_commit, current-stream serving-history staging before next-cycle buffer reuse, and a separate count-copy stream waiting on the producer current stream. Normal-path consumption of cycle c's host-count copy occurs in cycle c+1's proposer. That makes overlap with intervening Draft and Target work legally possible; it does not measure actual overlap or prove the copy is fully hidden.

1. **Conditional same-cycle parking join.** fixed_serving._park_completed calls proposer.park_completed_slots, whose first action is _commit_host_mirrors. The revised conditional_edges now correctly records count_copy_complete(c) → serving_progress_park(c) when parking occurs. This drains the current pending copy before sequence-mirror rewrites; next cycle's commit may consequently be a no-op. The original normal-path c→c+1 edge still appears in edges; interpret it as the pending-copy branch, since the new conditional parking edge can consume the event earlier. Branch-resolved endpoint semantics remain necessary. For Run394's sampled cycles64/65, zero initial outputs and at most8 tokens per cycle place every request below1024, so parking is absent there; the branch still matters for the complete workload.

2. **Unproved device-source reuse constraint.** The D2H copy reads state.num_sampled(c); fixed_decode.advance_state(c+1) writes the same buffer. The reviewed code orders producer c before copy c, and host-copy-buffer consumption before its next copy, but shows no explicit reverse stream wait that proves copy c has stopped reading before num_sampled is overwritten in c+1. The next host _commit_host_mirrors is called after c+1 state_advance has been submitted. The long intervening Target interval and passing mirror checks support observed behavior, not a universal happens-before proof. The revised unproven_hazards correctly records this as an unresolved read-before-overwrite requirement, **not** an observed race or an already enforced edge. Backend synchronization semantics or measured completion plus storage lifetime analysis must resolve it before a schedule transformation is declared legal.

## Edge semantics need separate domains

A host source-order edge does not normally mean the source device operation finishes before the destination host call begins. In particular, state_advance→host_commit_prev is host submission order; the CPU may reach that wait while current-stream device work is pending. Likewise count_copy_launch→Draft means copy submission precedes Draft submission, not that D2H service completes before Draft starts.

Use separate enqueue and completion endpoints, or an explicit typed-edge interpretation that prevents a longest-path solver from treating every listed edge as finish-to-start. The host's wait-call entry precedes or coincides with copy completion in some executions; its return and mirror update depend on completion. A single host_commit node conceals this distinction.

Useful explicit data edges currently implied only transitively by the fixed call order include Target hidden/aux output→Draft pack_hidden, acceptance sample/count tensors→Draft prepare, state last_sampled_tokens→Draft, draft_commit(c)→Target input IDs(c+1), and current acceptance/count buffer→serving history copy→next overwrite. Preserve these if evaluating reordered schedules, where original source-order edges may be removed.

## FULL Graph synchronization

The exact installed replay-branch condition is:

`runtime_mode == FULL and not (_EXTRA_CTX.is_draft_model and use_eagle) and not enable_enpu`

Only then does current_stream.synchronize precede aclgraph.replay. Capture/cache-miss and non-replay paths must not inherit that node automatically. Record runtime branch values, especially enable_enpu and schedule mode; a default constructor value is not proof of the instantiated value. This synchronizes the selected current stream, not every independent side stream. Graph/HCCL completion joins and rank arrivals remain unrepresented. If enable_enpu is true, the replacement ModelRunner ordering must be traced even when this wrapper skips the wait.

The source also has off/serial/overlap next-Target metadata schedules. Overlap mode adds a private stream, producer wait, completion event, next-cycle wait_event/commit, and parking invalidation. The current skeleton is conditional on the applicable mode, not a universal description of all three.

## Still missing for a bound

- Per-rank collective producer/arrival/completion and all8 dependencies; internal Graph branches and stream joins.
- Storage hazards for reused buffers and KV/state, host CPU work versus enqueue latency, Draft feedback subgraph and its underlying source pin. Naming “seven Markov steps” is not a source-expanded feedback DAG.
- Termination branch: no next cycle, pending-state drain, output/count D2H, host clipping, result assembly and publication; request arrivals, prefix/prefill, seed/handoff and legal refill.
- Actual concurrent resource occupancy, compulsory work/traffic, node lower bounds, instrumentation control and clock calibration.

Run395 medians are observed envelopes, not disjoint node costs. They must not be assigned wholesale to expanded nodes, summed, or used as minimum durations.

## What improves and what to do next

Scheduling uncertainty genuinely shrinks structurally: the inventory identifies the one-cycle delayed count consumer, host barriers, side-stream fork, serving staging, and conditional paths that a proposed schedule must respect. It also exposes a previously unstated buffer-reuse obligation. No numerical Scheduling floor, Algorithm compulsory-work floor, Hardware capacity bound or Product ceiling improves yet.

Highest-value next step is to settle the num_sampled read-before-overwrite contract and timestamp the existing copy completion and true next consumer on identified streams, without inserting a new hot-path synchronization that changes the schedule. Extend the same evidence to Graph/HCCL joins and rank arrival; include actual branch flags and an uninstrumented control. Then construct a resource-constrained DAG with justified costs.

Confidence: high for inspected source order, conditional parking and replay predicate; medium for overlap opportunity under the normal branch; insufficient for actual hidden overlap, full all8 critical path, or any finite ceiling.

Recheck covers the revised artifact: five pinned sources, 19 nodes, 22 main edges, one conditional parking edge and one unproven buffer-reuse hazard. Ledger SHA256: 7440f8e1b7f229d3416f56f4dd0dad4224075d92ffa04eb1d0966860da95bcba. The corrections are accepted as inventory updates; the host-enqueue/device-completion typed-edge limitation remains and prohibits automatic longest-path evaluation. Only astra_review.md was written.
