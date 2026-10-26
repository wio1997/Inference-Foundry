# Run412 — Astra independent completed B review

## Verdict and scope

**ACCEPT the scoped Run410 observations: generation64 D2H read completes before generation65 overwrite in all40 sampled rank/cohort records, followed by the recorded original-sync and first numeric Host count/sequence-mirror consumption.**

**INCONCLUSIVE for the later Draft/DSA consumer chain and the original combined gate.** first_clone and downstream are absent in40/40 records. No universal happens-before edge, original-schedule safety proof, cross-rank critical path, removable duration or finite Bound is established.

Read-only review of all40 completed schema2 captures,40 Runtime reports, both clients, server log, preflight, install/restore manifests and cleanup evidence. Only this requested review file was written. A1/service/NPU were not touched.

## Independent recomputation

Recomputed from raw JSON without relying on gate.json or analyzer summary as the source of truth.

- Exact coverage: five cohorts x ranks0..7; one B run ID; phases warmup for1..4 and diagnostic for5.
- Every record: schema2; schedule_mode=off; eager Draft; unparked normal_proposer; pending generation64; no generation/storage error;12 slots; cycles64/65 accepted counts in1..8.
- Four distinct timing events, distinct from the production event; R markers on the same recorded copy stream; W markers on a different common writer stream; all comparisons stay on the same device. Source identity is the same48-byte int32 tensor at the paired read/write sites.
- Per-cohort full accepted-count histories match across all8 ranks. Capture cycles join the corresponding Runtime reports:302,284,314,318,291.
- The Host sequence is ordered in every record: commit -> original synchronize -> destination count add -> first sequence-mirror add -> mirrors_done -> launch65. Count-add pointer matches the pinned CPU copy destination; sequence-add pointer belongs to the recorded mirrors. Entry/generation checks pass.

| Quantity | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Direct R_DONE64 -> W_PRE65 margin, ms |53.333221|53.746910|54.525200|
| Direct read envelope, us |2.140|2.420|3.420|
| Direct write envelope, us |4.920|5.360|5.980|
| Host synchronize bracket, us |11.361|14.731|22.991|
| Host count-add bracket, us |44.812|57.488|87.155|
| Host seq-add bracket, us |18.331|24.666|35.112|

All40 direct/anchor margin differences are <=0.000835419 ms. Per-cohort all8 margin ranges, ms:1=53.717522..53.779282;2=54.378658..54.525200;3=53.682941..53.752338;4=54.161362..54.251560;5=53.333221..53.433182.

The terminal anchor loses precision for microsecond envelopes: anchor-derived read intervals are1.953125..2.9296875 us and write intervals4.8828125..5.859375 us. Use direct nearby event intervals for the table. These are observed event brackets including instrumentation/queue effects, not isolated copy-kernel or indispensable-work costs.

## Preflight interpretation

Independently checked producer-script SHA, all8 devices,256 successful records, Host-visible copied data and nonnegative known-order direct intervals. Recomputed direct uncertainty from twice the largest observed adjacent interval/discrepancy:

- direct_uncertainty_ms=0.20068000257015228;
- anchor_uncertainty_ms=0.40136000514030457;
- reported local_marker_allowance_ms=0.6003240025701523;
- direct combined threshold=0.8010040051403046 ms.

All40 margins exceed both the direct allowance and combined threshold. The marker allowance includes an enqueue component whose individual samples are not retained in the report; its formula is source-reviewed, but that component cannot be independently reconstructed from the stored rows. Neither allowance is a hard error/overhead bound. The event wrapper, installed-source semantics and preflight support a strong empirical local observation, not a universal hardware timing theorem.

R_DONE is recorded after the original production completion event and W_PRE before the complete next overwrite expression. A positive margin is conservative for B. Added event initialization/Python hooks may alter slack; a large observed B gap is not permission to insert/remove dependencies in the original program.

## Frozen B and cleanup gates

Independently verified:

- 60 server POST log entries, matching server_post_count.txt.
- 48+12 client requests all successful with1024 output tokens. Combined timestamp sweep peak concurrency12.
- 40 Runtime reports, eight ranks in each cohort,60 unique Runtime request IDs, all8 per-cohort request order/cycle equality. FULL Target requested/active; zero post-handoff oracle and ModelRunner calls; every host_mirror_exact/pass true.
- 11 server Running/Waiting samples, maximum sum12. This supports clean concurrency alongside the process/request ledger; sampling alone does not prove an unsampled maximum.
- Client ledger SHA matches gate.json. Sum of per-cohort latest-rank wall durations independently recomputes85.7463466594927 s; this is an aggregate local-duration statistic, not a synchronized all8 makespan or formal Current.
- All seven cleanup status fields are0. Stop process probe is[], independent device probe has eight rows3435..3447 MB and no VLLM process rows.
- Install and restore manifests agree. before/after source SHA files match. Every current borrowed file also matches its expected original SHA at review time.

Thus B's external frozen-workload and restoration evidence passes this review. Client artifacts do not retain SSE request IDs; the join to Runtime is via clean lifecycle/counts/cohort trajectories, not a direct client-index-to-request-ID record. Do not claim stronger client attribution.

## Remaining consumer gap and forbidden extrapolations

The downstream absence is explained by source coverage: fixed handoff uses a non-None num_draft_tokens_cpu list and a runner shim with num_rejected_tokens_event=None, bypassing the padded+async utils clone/add_ path. DSpark metadata uses build_for_drafting, while the hooks target ordinary DSA builders. The drafting-specific decode Host max/item is uninstrumented; drafting prefill uses device sequence lengths. No-consumer-observed is not no-consumer-exists.

The recorded count add is a real first numeric read of the copied counts, and the recorded seq add is a real numeric mirror update. This closes those specific sites without proving the mirror's later use by Draft/DSA. Keep the original combined analyzer INCONCLUSIVE and publish the local subclaims explicitly rather than weakening its acceptance predicate retrospectively.

A0/A1 do not expose full per-cycle acceptance trajectories; their aggregate cycle/window statistics cannot establish trajectory equality. Per-record original_schedule_slack_supported is merely a threshold result. Global original_schedule_extrapolation_supported must remain false. No inference that54 ms is removable Host/copy overhead, that all ranks share one clock, or that the full state.num_sampled hazard is universally impossible is valid.

Algorithm/Resource floors, Hardware floors, executable Scheduling Bound and Product ceiling remain null. Confidence is high in the checked raw-data arithmetic and B-local ordering/Host sites, limited for extrapolation beyond those40 selected observations.
