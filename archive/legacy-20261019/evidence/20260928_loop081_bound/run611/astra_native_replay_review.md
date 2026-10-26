# Run611 native Graph replay scoped Astra High addendum

**SCOPED PASS for the three exported Model45 event occurrences per rank and their instrumented timeline envelopes.** This is not a strict completion-time, Scheduling Bound or Product saving certificate. No service or device experiment was run.

Reviewed reducer SHA256: cfab978265fde98f84bad037ed3f482f72f666f4f2ddf29ad88c866216cadc3b.
Reviewed native_graph_replay.json SHA256: **9518ee739ece657c8cff467ff476d92e7cc939048945abccc1e9f7e5a009336d**. The task message's abbreviated/mistyped output SHA is superseded by this direct file digest.

## Independent partition challenge

I used a different partition method: group all Model45 native X records by physical-stream/task/batch/subtask **plus event name and task type**, sort each static key's three timestamps, then collect occurrence0/1/2 across keys. All eight ranks independently have5,412 keys, exactly three occurrences/key, and each resulting occurrence agrees with the reducer's event envelope to less than0.5us (absolute epoch timestamps converted to float lose sub-microsecond precision). This supports the temporal partition without using connection_id or the reducer's bisect algorithm.

Each exported first record is the stream1/task0 start anchor; each group's first timestamp lies inside the corresponding ordinal Target CPU scope. The reducer additionally checks temporal separation and complete equality of the static key sets. Together these strongly identify three internally consistent repeated static Graph event inventories in this profile.

The anchor-inside-scope relation is still temporal association. It does not independently certify a particular logical cycle, live tensor generation, or launch-to-device causality. Preserve nominal_cycle_if_window_exact and the existing carry-in/stop-fence caveats. A direct launch/flow join remains the next stronger ownership check.

“Completeness” here means each observed static Model45 key occurs once per partition. This does not prove that the exporter emitted every required native task, that this key set exactly equals the current captured Graph generation, or that all eager work is assigned. Do not use that word without this scope.

## HCCL and envelope checks

All24 occurrences independently contain exactly **260 hcom-named native events:130 allGather,87 reduceScatter,43 alltoall**. These are name-classified exported events, not proved necessary collectives, byte volumes or measured physical-link traffic. The older265-call chain cannot be equated with this Graph census; the five-count difference is not evidence of missing records without matching scope/ownership.

The reported **45.102–72.209ms** spans are reproduced as earliest exported Model45 start to latest exported Model45 timestamp-plus-duration. The ending event in the checked rank0 first occurrence is a zero-duration **NOTIFY_RECORD**, not a final arithmetic kernel with a measured completion duration. Prefer “exported native Graph event envelope/end marker” to “certified physical completion.”

The all24 exported native end markers lie after their associated Target CPU scope end by **42.9605–70.071ms** in the parser's common exported timeline. This statement is supported **as an exported-timeline observation**. It supports the existing conclusion that a Target Host scope return cannot be treated as the replay-completion boundary in this acquisition.

The earlier RAW-versus-MONOTONIC correction neither invalidates nor independently certifies the profiler's Host/device timeline calibration. This reducer has not established a bounded historical cross-domain mapping error, launch causality or absence of unexported work. Therefore do not describe these deltas as strict physical waiting durations or claim sub-microsecond accuracy from epoch-float subtraction.

## Bound implications and next step

The envelope contains recorded work, waits, rank skew, submission effects and profiler perturbation. It is neither necessary Target execution time nor removable Host/device slack. It cannot be subtracted from Product wall, multiplied by all cycles/layers, used as an HCCL transit time, or inserted as a strict node-duration floor.

This result does reduce Scheduling **ownership/partition uncertainty**: the evidence now supports three consistent Model45 exported event sequences rather than an unpartitioned Graph bag. Resource/Hardware, Scheduling/Execution and Product strict endpoints, plus numerical Current-to-limit gap, remain null; formal Current stays571.681tok/s.

Next correlate these occurrence anchors with actual Graph launch/flow and the surrounding eager acceptance/state/Draft ownership; cross-check task_time CSV and resolve rank7's eager-flow discrepancy from the previous review. Then identify the missing typed KV writer-generation→first-query and next-Target dependencies. Conditional Engineering service intervals require observer transfer and same-state control; no new live acquisition is justified merely to reproduce this already available inventory.
