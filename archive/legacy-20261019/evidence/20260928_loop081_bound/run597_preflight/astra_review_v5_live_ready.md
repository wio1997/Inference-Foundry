# Run597 final independent Astra preflight — v5

## Verdict: LIVE-READY for one guarded diagnostic acquisition

The final reviewed collector/patch/wrapper are ready for a single warm48 + measured48, c12, 1024-output **diagnostic** acquisition, provided the wrapper runtime prestart/stop/restore gates pass. This is static/CPU readiness, not a guarantee that the live run will pass or remain unperturbed. No service, NPU workload, device query, production install or restore was performed by this review.

The initial NOT YET LIVE-READY review is preserved as `astra_review_initial_not_ready.md`; its concrete blockers have been closed. Final checks are in `astra_v5_checks.json`, `astra_selftest_v5.json` and the earlier independent one-shot/negative-site checks.

## Closed gates and evidence

- `prepared()` independently verifies all **five original source hashes**, unique insertion anchors, in-memory compilation and exact generated patched hashes against `patch_check_v5.json`. Helper SHA agrees with the selftest collector SHA.
- Independently reran the final CPU selftest into a separate artifact: valid full A0 fixture passes, **12 negatives** reject, and output exactly matches the parent selftest. The additional terminal-park positive passes before wrong-anchor/early-event negatives.
- Synthetic Target/DSpark/context lists are now independent. Separate negative-site checks confirm wrong DSpark token and context position fail their intended DSpark/context assertions, rather than a shared Target-list mutation. Duplicate/missing checkpoint negatives independently reject as well.
- Previous four-cohort A0 reconstruction remains 24,368 / 23,824 / 26,064 / 24,240 rows = **98,496**. Synthetic DSpark metadata validates checker formulas only; it is explicitly not captured Draft evidence.
- One-shot tests verify bind clears stale proposer state, the Draft input hook consumes its marker before capture (including failure), and an unarmed call does nothing. This prevents a final selected cycle from arming later ordinary prefill/seed calls.
- The revised validator joins 96 unique client records through response-ID prefix to rank0 Runtime cohort/slot, checks warmup/measured membership and dataset indices 0..47 per phase, then enforces all8 Runtime/basis identity and semantic hashes. The join logic independently succeeds on the admitted old Run566 96-request records. Client admission hash is retained.
- `bash -n` passes. Wrapper compares selftest helper SHA with the freshly generated patch manifest before install; mere existence of an old selftest is no longer sufficient.

## Installed-source hook timing

The new hook is at the unique return of `DSparkProposer.set_inputs_first_pass`, after every per-group copy/expand kernel and all `cad` query/sequence metadata updates. It is called from the Python `_propose` orchestration path before the actual model/context execution and outside the compiled decoder/Target graph. It observes real query IDs/positions, physical context positions and group slot buffers after their producer has been queued on the same execution stream. Clones preserve them before the next cycle overwrites reusable buffers; no explicit new synchronization or model invocation is added.

The source equations match the checker:

- Padded prepare keeps Target context 96 rows, with rejected padding retained; rejection is 8 minus RAW accepted prefix length, not masked count.
- Query start position is Target position + raw count; query q uses that position + q.
- Query0 token is the advanced anchor for active slots and frozen anchor for parked slots; later query tokens use the configured parallel token.
- Query sequence length is original Target sequence length minus rejection plus Q.
- sample_from_anchor gives Q=7 with samples 0..6; the alternate Q=8 source branch samples 1..7. The captured live configuration must be reviewed against the frozen model before extending its work formula.

Initial last/draft clones precede first Runtime work. The full next-draft history copy occurs after step has committed next_draft and before the next writer. Host park events record already available Host scalars; final-cycle park records are now checked against completion/final position and per-slot uniqueness. Raw acceptance and masked staged counts remain distinct.

## Safety and recovery

The wrapper holds the existing acquisition lock, requires a fresh output directory, checks service/container/NPU idleness and Host capacity before mutation, and snapshots source/script identities. Install backs up all five files before replacement. Cleanup stops only tagged owned service/client/helper processes, verifies stop before restore, and checks all original source hashes and unchanged helper/scripts. Unknown installed-source drift fails closed. `RUN568` marker strings retained in the wrapper are paired producer/consumer sentinels, not evidence of a different run; run identity is the explicit Run597 tag/output paths.

Admission requires the entire run plus cleanup to succeed. Retain run/stop/verify/restore/hash-comparison exit codes; a successful basis_admission file cannot override a nonzero cleanup result. The main reviewer has not exercised process stop/restore live during this read-only preflight.

## Remaining scope gates — do not promote beyond them

1. This is **new W0**, not any of Run99’s historical trajectories. Freeze algorithm/configuration; recording acceptance/output history does not authorize an algorithm change.
2. This captures complete cycle basis plus sparse Target, DSpark prepare and actual Draft input witnesses. It does not capture every layer/context projection, Markov operation, prefill/seed, initial KV generation or complete cache state. Captured slots establish observations and shapes; physical slot values are not yet independently verified against generation-aware block tables. Per-rank physical mappings correctly remain outside cross-rank semantic equality.
3. Do not dedup semantic states or classify all active/context rows as necessary fresh work based only on token/position equality. Keep entry-cache, context dependency and terminal-issued work uncertainty explicit. Group slots, causal/Q fields and schedule branch records require post-acquisition review before using a broader formula.
4. Current selected-cycle coverage covers fixed checkpoints including adjacent 0/1 and the collector emits a first-postpark witness. The checker enforces fixed selected checkpoints, but does not independently require that dynamically selected postpark cycle; verify it in final admission review. No claim of every-cycle physical metadata or complete KV correctness follows.
5. Additional copies, clones, Python calls and serialization perturb execution. JSON/fsync/D2H happen after ordinary cohort drain but **before serving return/client completion**, so their cost lies inside Product wall. First-run TPS is diagnostic. Any later timing transfer needs the previously specified perturbation and correctness gates; no zero-overhead claim is available.
6. Formal Current remains **571.681 tok/s** and strict Resource/Scheduling/Product endpoints remain null. After capture, use its evidence to refine conditional Runtime workload/dependency terms while carrying the missing Product work explicitly.

These remaining scope restrictions do not block acquiring the diagnostic evidence. Reject or narrow a live result for missing files/cycles/phase joins, failed formula/all8 checks, unsupported model/Graph branch, source drift, or any stop/restore/exit failure.

## Final reviewed identities

- `scripts/loop081_basis_capture_run597.py`: `4825cb32b65a941b5d081b40b76c9a810b154dcafe771b35c9f6991be715c02a`.
- `scripts/loop081_basis_patch_run597.py`: `bfd58d4fbab2dc175845bc7c68195d5045d5ed8ad6d78a167f01b7ad5bc15ae0`.
- `scripts/loop081_basis_selftest_run597.py`: `f5b1e25bc8fd107d527a8103ec46594f5efa81f6b30bbeb98b8af4f5771562ef`.
- `scripts/loop081_basis_validate_run597.py`: `d6d3d7eec66d66300e81776492ff34edf5931e03063fe286c39b7a519013e884`.
- `scripts/run_loop081_basis_run597.sh`: `cd0d5bf59eaef183d396038de234d1c26d61a63b88f5ec7cc16efc0791523b72`.
- `evidence/20260928_loop081_bound/run597_preflight/patch_check_v5.json`: `2543e39fe695ffbf9d61986bc650065b1a04eddcfb54cb0b87a3a6f6f22c9522`.
- `evidence/20260928_loop081_bound/run597_preflight/selftest.json`: `74ccfe2b8304b82f8ef8f1d6c04d3f8788642e775be100d47a296b0bd74ae8f5`.
