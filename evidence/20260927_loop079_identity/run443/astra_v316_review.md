# Run443 — independent V3.16 Bound review

2026-09-27. Reviewed original Run442 and its generator against Run435/436, Run437/440, Run438/439 and Run441. Stdlib-only negative controls ran in temporary copies of both generators and their six inputs. No service, NPU workload, runtime source or existing evidence was modified. Reviewed versions are identified below.

## Verdict

**CHANGES REQUIRED before accepting original Run442.** Its Bound disposition is conservative: all 19 proof nodes resolve false, Current remains Run99 median 571.681 tok/s, and saved numeric endpoints are null. Run437 remains a sampled binding observation. However, Run441's claimed exact terminal payload is wrong by a factor of eight, and the advertised universal no-numeric-output guard misses an existing capacity endpoint.

## Run441 payload and timing

Installed CANN 9.1 source inspected through SSH 61.241.77.34-60008, active test container vllm-ascend26-dsv4f-w4a8 (hostname S900K3-49), gives:

- opbase_test/hccl_allgather_rootinfo_test.cc:131–134 divides data->count by rank_size, with ceiling division.
- Lines 136–139 assign input bytes=malloc_kSize and output bytes=malloc_kSize*rank_size.
- Lines 121–127 calculate algorithm bandwidth using aggregate output bytes divided by reported time.

Thus CLI data_size=3,102,720 at TP8/BF16 means **387,840 input bytes per rank**, and 3,102,720 output bytes per rank. Terminal [96,16160] BF16 requires 3,102,720 input bytes and 24,821,760 output bytes per rank: correct CLI minbytes/maxbytes are **24,821,760**. The existing scripts/loop074_run339_analyze.py already records this convention explicitly in CASES and its limits.

Keep 221.77/179.45/205.78 microseconds only as smaller-payload isolated observations. Do not scale their times to estimate the intended payload. Correct Run441's title/findings and V3.16's isolated_terminal_allgather name, payload and scope; a separately identified corrected run may supply the intended isolated sample.

Lines 165–178 record ACL events around the measured loop, synchronize the stream, then read elapsed time; lines 121–127 divide by 30. common/src/hccl_opbase_rootinfo_base.cc:163–207 prints root_rank's local interval without rank timing reduction. Prefer **reported root-rank stream-event mean per iteration, host overhead not excluded**. onlydevicetime=0 permits host submission/loading gaps in this event interval; it is not host-wall duration or an all-rank mean/makespan. Each invocation is an eight-rank MPI job.

Result checking runs an additional collective after the timed loop, then checks the final receive buffer (lines 180–186). Logs report success with no other-rank failure, supporting the tool's final-buffer check, not separate validation of all 30 timed iterations. BF16 uses the tool's half-width check path; do not imply exhaustive BF16 patterns. Algorithm bandwidth uses aggregate receive bytes, not physical per-link traffic.

Executable identity was independently verified in the parent's exact test container vllm-ascend26-dsv4f-w4a8: /usr/local/Ascend/cann-9.1.0/tools/hccl_test/bin/all_gather_test has SHA256 b276a969e778ab2d627643225fc9a29b45878b016999079de9ae705a73997b9d, matching Run441. Its installed source hashes match the reviewed source below. An initial check used legacy container dsv4ab from stale PROJECT_STATE; that distinct container's absent installed binary and different /tmp build do not challenge Run441 provenance. Preserve the exact container, command, rank/dtype and exit status in the corrected manifest.

A corrected matching-payload test still supplies no compulsory collective, true C_plus, unavoidable service minimum, torch backend completion contract, mixed contention, exposed Current cost or removable Product interval. Run439's native-completion prerequisite and original-path controls remain open.

## Run437 and DAG scope

Run437 correctly narrows uncertainty to selected explicit FULL graph input/storage binding and CP updater value prefixes. Actual layer-to-CP consumers, per-layer branch/native row composition, GMM/scale association, graph all-slot semantics and loaded binary identity remain open. Run440's common all-rank router-row permutation control still matters: valid histograms and endpoint metadata do not prove retained router-row identity. No full semantic oracle or external-token freshness certificate follows.

The saved 19-node DAG is structurally sound. New observation nodes cannot certify all43, compulsory work or capacity. The strict endpoint requires W-minus AND C-plus AND scope join; an independent projection proof need not require all43. A selected necessary-path relaxation remains independent of a full feasible overlap schedule. The revised missing text fixes the prior over-demand for all ranks/edges.

Validation checks topology, literal contract equality and Boolean statuses, not evidence truth, units, numeric sufficiency or attestation authority. Product contract strings on sampled observation nodes name the intended target context, not a witnessed formal 48-request timed window. The engineering endpoint is explicitly predictive-model-plus-formal-feasibility; it should not become a prerequisite for reporting a directly witnessed formal point.

## Temporary-copy negative controls

Baseline reproduction is byte-identical to original Run442.

| Mutation | Result |
|---|---|
| Add numeric_tps, numeric_s, latency_floor_s or finite_tps_upper_bound under nested certificate_graph list | Each rejected with endpoint path |
| Certify inherited sampled observation | Rejected |
| Dangling dependency, dependency cycle or changed node contract | Each rejected |
| Set Run437 all43 certificate true or CP certificate false | Each rejected |
| Set existing bound_ladder.hardware_resource.run431_installed_platform.maximum_capacity_bound=123 | **Accepted: uncovered existing endpoint** |
| Add proof_dag.numeric_tps=123 | **Accepted: unchecked subtree** |
| Replace five cohorts with five copies of first cohort | Accepted: length alone does not validate distinct cohorts |
| Change HCCL 221.77/205.78 while preserving anchor179.45 | Accepted: output still hardcodes old values |

The last two demonstrate a trusted-summary boundary, not a contradiction of independently reviewed raw Run437 evidence. Numeric escapes contradict the docstring's unconditional promise that this generator cannot emit finite endpoints from changed inherited input.

## Exact fixes

1. Correct Run441 payload/time semantics; store CLI output bytes, API input bytes, rank count and timing scope separately. Read timing from validated structured results/manifests with pinned logs, rather than a number substring in prose. Preserve the mistaken-payload run historically.
2. Apply endpoint checks to the whole model and include existing maximum_capacity_bound and derived unresolved_gap_tps. Prefer a schema or explicit endpoint-path registry for future fields. Keep all endpoints null and retain the no-certified-node guard. Add the capacity escape as a regression control.
3. Add direct PRIOR Run435 to input_paths: the current list contains Run434 but omits the actual loaded Run435 JSON. Pin direct input and generator hashes; snippet checks only establish phrase presence.
4. Replace stale next_measurement.specific_gate, which still prescribes broad Run437 capture, with the selected Run439 native-completion prerequisite and bounded slice gate. Preserve targeted retained-route work as the conditional alternative.
5. If the Run437 gate is intended as validation, assert expected distinct cohorts/ranks/cycles and bind the exact reviewed validator output hash. Otherwise label it a shallow scope-consistency check, not model correctness.

Historical arithmetic, storage sets, conditional cuts and attained samples retain qualifying scopes. No saved inherited field presently promotes a finite Resource/Scheduling/Product endpoint. Current and null endpoints should remain unchanged after correction.

## Reviewed SHA256

- V3.16 generator: 6d2ca6aec7a962c91552916838be19db5a22cc7b3c7b0a58e033b37731c71841
- Original Run442 JSON: f5db475f3c4261c8dc289ba28c06dd1f005936f9bc247905d008ee8de67bd5c6
- Original Run441 findings: ea863286796bbff555b7dc9ad27b781d5e92e7d435bd65e91990d04e9d743616
- Run437 validation: fcd5acd45b423bd3a92c2440f3d838ed0a7efac8d92ab9cd58e73c1867c3f292
- Run440 review: 55a0b9ea2b3cbdf08471275f4039a53dcc198a7edcba99e5a9ac4bd63003eb24
- Installed and /tmp AllGather source: 8a5339c7310160c8028dc4d481b9a44a9326979124c8ccf7b5b31732b9015d15
- Installed HCCL Test base source: 239c9b1d06937bf253ed82daba2762fd60e8cc75e5ac4e09f22e95cb814d5ff8
