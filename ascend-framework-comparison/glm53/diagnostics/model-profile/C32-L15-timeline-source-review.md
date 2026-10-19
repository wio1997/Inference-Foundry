# L15-D existing native timeline/source review

**Status:** `CLOSED_ACCEPTED` for bounded source/timeline evidence only. This note does not create a performance result, an optimization acceptance, an SLA pass, a causal attribution or a critical-path claim.

## Accepted evidence

- Master decision: `results/C32-L15-timeline-source-decision-v1.json` SHA `8136fdf8171aa9170f3ca7462fb64b907f2b49823ebb43b41d738ee71091b2e8` (`ACCEPTED`).
- Collection receipt: `runs/20261019T-L15-sglang-existing-D-timeline-source-v1/receipt.json` SHA `ea0e76dc6b3b778349cde0dded5944c7b811c722ab85ca2d2182202704fa7baa` (`pass=true`).
- Actual review: `results/C32-L15-D-timeline-source-actual-review-v1.json` SHA `70b0de421f85080079ab3548157681ad55d7e99b111571629c1ce19e219dbc4c` (`APPROVED_READ_ONLY`).
- Frozen collector: `scripts/glm53_C32_L15_D_timeline_source_v1.py` SHA `bdc451ea0927c63cd47d76bd7ec9ab4a46270221db952d48a18e6509a236e3a7`; wrapper SHA `8cdcbe25d8cd8e1b7dc9d80c743ea4bab07ea103ad366369b24adc88d6504408`.
- Local source copies: `runs/20261019T-L15-sglang-existing-D-timeline-source-v1/source/`; seven modules per node, 336,189 bytes per node, identical SHA sets across nodes. EagerRunner SHA `a04d7344319eccbeacefad3f60e559c716f412e680ed0f712645ef149238a6b5`; AscendBackend SHA `5d0722244c5c75109821c93db37b27f809f44fc79f5b76a8e351977cb0f1b997`; `eagle_info` SHA `0f8dabd7d0b0d7e7f179a1dd87f957b0172fbbae6436c6ddaef989b881b09590`.
- Original trace facts remain remote: node 36 rank 0 `21,607,164` bytes SHA `90f81d5a5674bbf69c69cfaadbd4c8b62f5d3b3ae8633d95b2ca36b7bbff779e`, mtime_ns `1792351435040000000`; node 41 rank 16 `21,673,099` bytes SHA `13fcaf77fa3c8e44c95638bac25d6104ce2b9bfc9ab1041a71264409532c9905`, mtime_ns `1792350998680000000`.
- Local supplement: `results/C32-L15-timeline-Decimal-enclosure-supplement-v1.json` SHA `60aaf5b76c10be7167d0db9b54eff08da2ce3973fcd95d6dcaf9018e9fd2c29b`; traceable local script `scripts/supplement_C32_L15_timeline_decimal_enclosure_v1.py` SHA `fd0cdb360dc286e810f4b00e53a512baa70f969f2d7fea302454f6696912b573`.

## What is known

- The original collector's `enclosing_step_spans=null` values were caused by string-valued `ts`; the original nulls and `missing_numeric_ts_or_dur` reasons are preserved unchanged.
- The supplement parses `Decimal(raw ts)` and `Decimal(str(dur))` with finite/non-negative checks, in raw units only, and tests geometric interval containment within the same context file, pid and tid. It does not convert to seconds, guess a clock base, join files/ranks or add overlaps.
- All 24 selected copy rows per node were re-checked:
  - node 36: 13 enclosed, 11 with no enclosing step span; node 41: 14 enclosed, 10 with no enclosing step span.
  - The two longest `acl_memcpy_host_to_device` rows on each rank are inside an outer `draft_extend` span and end before the next inner `step[DRAFT bs=1]` span.
  - Other long contained copies lie inside `draft` and/or an inner `step[DRAFT bs=1]` span. The unenclosed rows remain unenclosed and are not guessed into a phase.
- The installed `EagerRunner.load_batch` source contains the native branch
  `if envs.SGLANG_EAGER_INPUT_NO_COPY.get(): return replace(forward_batch)`;
  otherwise it uses `registry.fill_from(...)` and `registry.extract_buffer(...)`. This is a source observation, not a measurement of the operative environment value or of a runtime function return.
- The actual native API names carry an `AscendCL@` prefix, for example `AscendCL@aclrtMemcpy` (60 on node 36, 61 on node 41), `AscendCL@aclrtMemcpyAsync` (89 each), `AscendCL@aclmdlRIExecuteAsync` (2 each) and `AscendCL@aclmdlRICaptureGetInfo` (555 each). The L15 exact unprefixed selection therefore selected no `api:aclrtMemcpy`, `api:aclrtMemcpyAsync` or `graph_api` rows. The index names/counts are recorded as a selection limitation; no event interval or context is invented for them.
- `cat` is missing on 67,491 node-36 events and 67,676 node-41 events. Missing categories are preserved as a limitation, not treated as a reason to resample.
- No source errors, symlink candidates, relative imports or count overflows were observed in the accepted L15 source resolution. The selected events are a bounded sample, not full coverage.
- Import following remained bounded: 94 imports per node were not followed (45 prefix mismatch and 49 alias-token mismatch), and 3 of at most 4 extra modules were used. This is not proof that no other source path matters.

## What is not known

- The actual initial or live `SGLANG_EAGER_INPUT_NO_COPY` value on the 32 D model workers; no worker environment was measured by L15.
- Whether any selected host-copy event comes from the EagerRunner input registry rather than another copy path.
- Whether a long H2D host wait is avoidable, or how much of it is earlier asynchronous Graph work draining into a later blocking call. Host waits are not device time and must not be added to device intervals.
- Tensor ownership/lifetime safety if the no-copy live-tensor branch is actually used across eager forward, MTP/draft paths and asynchronous graph work.
- Whether the previously suspected pre-pad and post-pad attention-metadata initialization duplicate work, and whether the pre-pad call has required side effects. Source may contain pre/post-pad metadata duplication; side-effect and MTP-lifetime analysis remains pending.
- Any critical-path, overlap, bandwidth, utilization or optimization-benefit conclusion. The accepted measurement remains the historical L12D C32 result: 389.8740711045039 TPS with `aggregate_SLA_pass=false`.

## Guardrails

- `SGLANG_EAGER_INPUT_NO_COPY` is a **mechanism pending qualification**, not an optimization claim. No gain is attributed to it.
- Maximum H2D host wait may include Graph drain; it is not added to device time and is not called an avoidable copy.
- The pre/post-pad metadata duplication hypothesis is not proven; no source removal or initialization change is authorized.
- No cross-file, cross-pid/tid or cross-rank timestamp alignment; no overlap sum; no guessed clock unit/base.
- The existing C32 profile/measurement evidence remains E3. The L15 supplement and source read add bounded evidence, not a new performance result.

## Next prerequisite

`diagnostics/model-profile/loops/L16-D-native-eager-live-inputs-prerequisite-v1.json` is `PREPARING` with `decision=null`. Its local draft is `runs/20261019T-L15-accepted-L16-eager-input-prereq-local-draft-v1/l16-draft/` (`DRAFT_PENDING_PRIMARY_REVIEW`, not executed). The draft module SHA is `20120b22f227fc49c767bf6d5fe77912154c03b60ae02039496669d650bfd6c1`; wrapper SHA `d58a3791558c0af8221c69c2c1a9e54259a44d67a4fc8e80fa902bbc26e1c65b`; rendered host codes `36=2bb55d3cefa4b3af85a0ca531c6be5c7037d6676fab846227109fda031e98dfe`, `41=5016e2cd54ab9e58d9abc171a406405d63e7790fd423016171bff9238290cff1`. No actual review was created, nothing was dispatched or invoked, and no flag/service/model/Graph/draft/MTP/task-queue/MemCache/PD change is authorized.
