# Run399 — source-reviewed per-token route diagnostic (not installed)

Scope: CPU/source-only preparation after Run397. No service mutation, accelerator
compute, patch install, or measurement was performed. Sol must review before a
stopped-service install. This is a Bound numerator diagnostic, not a performance
candidate or evidence of compulsory HBM traffic.

## Files and invocation

- `scripts/loop078_route_capture.py`: inert helper; activates only when
  `EXTREME_BOUND_TOKEN_CAPTURE_DIR` is set before framework bootstrap.
- `scripts/loop078_route_capture_patch.py`: `check`, `install`, `restore`.
  `check --record <json>` reads six borrowed sources, generates patched text and
  compiles it in memory. Install requires a new `--state-dir`; restore verifies
  all installed/backup hashes before replacing any source. Keep backups.
- `scripts/loop078_route_capture_validate.py <capture-dir> --output <json>`:
  all-eight-rank offline validator, no torch dependency, refuses partial cohorts.
- `scripts/loop078_route_capture_selftest.py`: synthetic CPU positive and
  deliberate-corruption tests, including mocked hooks and two graph entries.

## Source-grounded capture

W4A8 IDs are produced in `quantization/methods/w4a8.py:539` and consumed by
`ops/fused_moe/moe_comm_method.py` dispatch. The hash C++ binding allocates int32
`[rows, 6]`. Target IDs are expected `[96, 6]`, 43 layers. Actual layer identity is
`RoutedExperts.layer_name`: Target `...layers.0–42.mlp.experts`, Draft
`mtp.0–2.mlp.experts` (normalized ordinal 43–45, preserving original name).
The Draft ModuleDict keys 43–45 are not its actual routed-layer prefixes.

The sixth borrowed source is `compilation/acl_graph.py`: its capture call wraps
`self.runnable` with an explicit ACLGraph-entry identity; replay records the
actual entry. Route/group references key on `(entry identity, batch descriptor,
layer name)`. At cohort-relative cycles 64/65, only the entries actually replayed
by Target are selected. Multiple captures with identical 96-row shape are safe
if they have distinct entries; registration order never selects an entry.
Same-entry pointer replacement, repeated selected replay, or anything other
than exactly 43 selected layers fails closed. Strong references intentionally
extend graph intermediate lifetimes; this needs a real startup smoke.

Draft is eager, as forced by `spec_decode/dspark_proposer.py`. Three MoE layers
run once, then seven Markov steps. Selected-cycle Draft IDs/group counts are
cloned immediately at dispatch, not identified by reusable eager addresses.
Draft input metadata is captured from its actual `set_inputs_first_pass`
override after `copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid`.
Q is 7 with sample_from_anchor, otherwise 8; expected full rows are 84 or 96.
Actual Q/shape/positions/sample offsets are validated, not inferred from Target.

Target IDs, group counts, map, input IDs, positions and active mask are device
snapshots after replay; raw acceptance counts are cloned before state advance.
The normal cohort count history is masked to zero for inactive slots, whereas
DSpark receives raw acceptance counts: both are kept and checked separately.
Draft64 output is checked against Target65 input candidate tokens for active
slots. All device copies use the producer/current stream. New D2H is confined
to the existing cohort-end drain/export location. No added hot-path sync.

## Validator and semantic boundary

The frozen AllGather dispatcher must have 256 experts and 32 local experts,
count-mode group_list, no log2phy, no EPLB and no forced balance. Per-layer global
ID histograms must exactly reproduce every rank's contiguous EP histogram and
expert_map. All eight replicated ID matrices and acceptance/position metadata
must agree. Missing ranks/layers, wrong graph identity, invalid top-6 IDs,
incorrect query offsets/positions or Draft64→Target65 handoff fail closed.

Target row `8*slot+j` maps to immutable cohort slot and captured absolute input
position. Retained row counts clip masked acceptance by remaining output quota.
The validator reports exact current and retained Target expert-union cardinality
per layer; Draft reports actual rows and current unions. Draft64 accepted token
attribution uses cycle65 count minus recovery/bonus. It is not a removable-Draft-
row claim: attention dependencies (including noncausal attention) remain.

Limitations and gates before interpreting real output:

- No NPU capture has been tested. Python graph-capture hook reachability, actual
  layer prefixes, all-gather unpadding, and route agreement are source-backed
  assumptions deliberately guarded by failures. A failure is not a result.
- External API request IDs are unavailable in FixedCohortServing. The recorded
  request identity is `(cohort, immutable slot)`, with initial positions/output
  counts. Join an external request manifest separately if API IDs are needed.
- Graph keys/pointers are process-local; validator does not compare their numeric
  values across ranks. It checks each rank's layer belongs to its own replay set.
- Global helper state assumes the frozen sequential fixed-cohort process; no
  concurrent runtime/thread use. Target self-replay audit is explicitly rejected.
- Selected snapshots add allocations and copy launches; graph reference retention
  changes pool lifetimes. Treat latency/TPS from this diagnostic as contaminated.
- Graph capture must happen with the environment enabled. Installing after graph
  creation cannot populate references. Restore only with service stopped.
- JSON integer dtype is not self-describing; int32 IDs are enforced in the live
  hook. Packed expert unions are storage sets, not compulsory physical HBM reads.

## Validation

`check` compiled all six generated borrowed sources, exit 0. Four new Python
scripts compiled, exit 0. Synthetic validator/mock-hook checks passed, exit 0,
including incorrect histogram, missing rank/layer, wrong graph, bad handoff,
wrong position and masked/raw acceptance mismatch rejection. Container-context
`from scripts import loop078_route_capture` printed `helper_import_ok`, exit 0;
it imports only standard-library modules and did not initialize torch/NPU.
See `source_check.json` and `validation.json`. Borrowed source hashes were
compared to the check record's originals after checks; no patch was installed.
