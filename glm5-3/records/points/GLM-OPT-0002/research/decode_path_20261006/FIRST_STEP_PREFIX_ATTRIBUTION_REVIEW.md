# Independent first-step prefix attribution review

2026-10-07. Offline only. Read `jobs/GLM53-CURRENT-FULL-PROFILE-OFFLINE-20261007/first_step_prefix/{index.json,rank0.json.gz,rank13.json.gz,rank15.json.gz}` and Run276 profile-on complete events/result. Used exact decimal CSV timestamps for interval unions; no parser rerun, device request, or frozen-artifact modification. Existing Run284 Goal Review remains applicable.

## Stronger temporal grouping

All three exports contain 4,538 selected CSV rows and the same operator sequence. Two distinct embedding-to-ArgMax groups occur before the first host graph execute:

- CSV line120 embedding → line4217 ArgMax: **75 hidden Pad tasks**, each input1×6144/output16×6144; matching router pads occur in this group.
- CSV line4322 embedding → line4402 ArgMax: **one hidden Pad**, likewise one-token input, plus its router pad.

This independently strengthens the 75-target-MoE + one-MTP-MoE interpretation: it is not just a total-count coincidence. Names and timing identify two model/sampling phases; exact Python layer identity still needs the host correlation Root is reading. These boundaries do not include all target metadata preparation, logits work after the start of ArgMax, or output publication.

| Rank | First embedding → first ArgMax, ms | Second embedding → second ArgMax, ms |
|---|---:|---:|
| 0 | 326.682353 | 7.017820 |
| 13 | 326.853109 | 8.402272 |
| 15 | 326.536555 | 6.789104 |

All recorded rows have eager Model ID4294967295; the padding stream is47. Together with the actual scheduler/dispatcher source in the preceding review, this supports a substantial **one-token target execution before steady FULL replay**. It does not prove a graph implementation can remove the whole ~327ms envelope or that it reproduces Run284's first-token difference.

## Union audit: gaps are not yet host starvation

Window is from first selected device row to last selected row, not request start/end. CSV task coverage, including COMMUNICATION rows:

| Rank | Window extent, ms | CSV interval union, ms | Uncovered by CSV, ms | COMMUNICATION union, ms |
|---|---:|---:|---:|---:|
| 0 | 356.495037 | 331.359487 | 25.135550 | 225.349039 |
| 13 | 355.072665 | 332.950082 | 22.122583 | 235.290787 |
| 15 | 356.415639 | 193.303365 | 163.112274 | 128.773971 |

Union categories overlap and must not be summed. There are **no Event/Wait rows in this selected kernel CSV**. Therefore uncovered intervals mean only “not covered by these exported kernel/communication rows”: they may contain event waits, missing copy tasks, dependency waits, host submission delay or work outside this export. Conversely, a long COMMUNICATION interval can include peer rendezvous; it is not proof of active wire transfer or useful computation throughout.

The rank asymmetry is important: ranks0/13 can remain inside communication while rank15 is not covered by a CSV task. This is compatible with late supply by rank15, but does not establish it. Do not report rank15's163.112ms as removable frontend time. Match the same collectives' starts/ends and CANN enqueue/dequeue plus EVENT_WAIT hardware rows before assigning the source of delay. Likewise, the near-full CSV coverage on ranks0/13 does not refute host supply starvation elsewhere.

## First output versus first replay

The first ArgMax is temporally before the first graph call in all three local device/host timelines, followed by a second embedding and ArgMax. Thus a target sampled-token computation exists before replay. The first public token is a separate claim:

- Run276 SSE first token is154842, with `arrived_ns=3113496040053423`; events use client monotonic nanoseconds.
- Device/graph timestamps use the profiler clock near1791367888… microseconds. The SSE `created` field is request-level whole seconds, not a token-send timestamp.
- No clock bridge or same-host output-publication marker is supplied by this CSV package. Therefore **public first-token arrival before first replay is unknown**, even though first target sampling precedes it. Async execution can submit a following step before the client observes the previous output.

Do not align these clocks by matching end points or convert `D_first_s` into a device timestamp. Read an existing same-host output-copy/readiness/publication marker if available; otherwise retain the limitation.

## Decision

Continue the single offline first-step question. The new grouping makes it materially stronger than another unchanged H13 toggle: target one-token execution dominates the pre-replay model envelope, whereas the observed first draft phase is much smaller. Next decisive evidence is the **existing target-prefix CANN/queue/collective join**, especially rank15 versus0/13, and the actual initial-state SFA branch. This distinguishes eager producer delay from necessary device/peer work before any first-step graph patch is proposed. No new Run is justified merely by the CSV gap calculation.

Keep H6/H5 and current FULL; H13 remains off/INCONCLUSIVE. Do not subtract steady qlen2 duration from first qlen1 duration as a savings estimate: initial-state attention, metadata and profiler behavior differ. Global maximum removable time remains unknown.
