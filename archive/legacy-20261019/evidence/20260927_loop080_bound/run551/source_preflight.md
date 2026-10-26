# Run551 — source preflight for Run549 Acquisition A

Status: **source map and partial client-only implementation**. The Run549/550 device acquisition is not yet a ready-to-run patch. No service, borrowed-source edit, NPU action or timed claim was made. The following source bytes were checked on the stopped server; line numbers below refer only to these pins and must be rechecked before device hook implementation.

| Source | SHA256 |
|---|---|
| Installed `model_runner_v1.py` | `004dbd0d5b1a5c3f9533fa1d12327bd0ae421fe24bb2674544b2c4a25d74aaba` |
| `runtime/extreme_decode.py` | `eb4b5142cc475a51e06c6eec8238e9cfeb918b568e8a5f27b0a4bc7d26963499` |
| `runtime/fixed_serving.py` | `137f1cc6a460700e6f26e5837aa062022eb621198e33099b85988f8b7ef6743a` |
| `scripts/loop079_formal_ledger_client.py` | `6e0d19ae332249b32ba742179666c2c2ee9bce13df257368171d16e754f22a37` |
| `bootstrap/vllm_extreme_handoff.py` | `f644bd14ac1cb9c8365ba2464abbff989d7f2c4c2d58279a8e18a5bce918716f` |

## Exact current ownership sequence

The frozen diagnostic client `scripts/loop079_formal_ledger_client.py` uses `async with asyncio.Semaphore(12)` around each POST and SSE stream. It records start inside and end before exiting that block. Its timestamps can establish a release set and lower/upper chronology; they do **not** identify which predecessor's permit an individual successor acquired. The Run542 client/server ledger already verifies96 exact bodies/outputs and all8 Host request identity. A new client record should add pre-acquire, acquired, stream-DONE and post-context-release timestamps without changing semaphore implementation, plus verify the resulting c12 maximum and phase barrier. Unique parent remains unknown unless actual permit lineage is recorded and shown schedule-preserving.

At installed `model_runner_v1.py:1794`, `execute_model` receives `SchedulerOutput`, updates persistent batch state, prepares input/attention metadata and later calls original `_model_forward` (`:3660+`) for prefill/ordinary requests. `_model_forward` (`:4544+`) runs the model, possibly updates FULL Graph parameters, then performs FlashComm1 hidden/aux gather. Return is a Host dispatch boundary; it does not by itself prove all KV/cache/side-stream/HCCL work complete. `sample_tokens` (`:4026+`) samples and calls `propose_draft_token_ids` (`:4092+`), which stores `self._draft_token_ids` and copies them to CPU for the next scheduler wave. A new measurement must bind the chosen successor request through these actual ordinary stages; a `_model_forward` event alone does not prove seed readiness.

At the chosen 12×96 handoff, the ModelRunner constructs `TargetHandoffInputs`/`ExtremeHandoffInputs` from the actual `input_ids[:96]`, positions, attention/KV trees and DSpark bindings (`:2171–2342`), then calls `build_extreme_runtime`. With `EXTREME_RUNTIME_SERVE=1`, `FixedCohortServing(...initial_output_counts=[0]*12).run()` owns all further cycles (`:3380+`). In `runtime/extreme_decode.py:372+`, first `step()` prepares Target inputs, updates derived metadata, calls `target.execute`, acceptance, state advance and proposer, then commits Draft. The first Target consumer is therefore after a specific generation of `last_sampled_tokens`, `draft_tokens`, positions, slot mappings and metadata; record those storage/generation keys and the existing producer-to-consumer wait, rather than assuming a Python call boundary is device-ready.

`FixedCohortServing.run()` stages `acceptance.sampled_token_ids` and `state.num_sampled` into device histories each cycle, then queries the proposer Host progress mirror and parks completed slots. Only after the **whole cohort** ends does it copy the histories to CPU and construct the full `ModelRunnerOutput`. The Host progress mirror trails by one cycle; per-slot count-prefix positions reconstructed after drain do not timestamp device completion. A possible early-output counterfactual needs a selected slot's immutable history prefix, an event after its final producer and copy, a safe publication buffer, and KV/state/slot ownership that remains valid while the old cohort continues. Existing code has no such publication path. A first-slot threshold cycle is only a count-domain opportunity until those certificates and the client release→arrival chain exist.

## Cut-by-cut completion certificate to resolve before launch

| Cut | Owner and generation | Completion evidence that can enter Bound model | Current gap |
|---|---|---|---|
| Predecessor selected slot output prefix | `FixedCohortServing.run`, `token_history/count_history`, slot/cohort/cycle | Existing producer→device-history copy order, selected end event after copy, immutable range and accepted raw-token reconstruction; copy/event timed within one device | no per-slot end event or early publication; Host mirror one-step lag; parked KV/slot lifetime |
| Actual release and arrival | client semaphore/SSE DONE, POST; server add/Scheduler | measured client eligible release set, post-acquire/POST and exact server request ID join, common Host clock scope | no unique permit parent; no earlier-than-current publication |
| Residual prefill | Scheduler `scheduled_new_reqs`; ModelRunner `_model_forward`; OutputProcessor `PrefillStats` | actual83–85 residual request/shape, source-backed caller/side-stream/HCCL producer-to-next-consumer join, bounded event domain | previous Run341 caller-stream span excludes unjoined side effects; `PrefillStats` is not device-ready |
| Seed and KV/state | `sample_tokens`, `propose_draft_token_ids`, `_draft_token_ids`; KV cache | token/position/KV storage generation, final producer and existing wait before handoff/first Target | producer paths and stream joins not yet fully mapped |
| First required Target | handoff binds actual96 tensors; first `ExtremeDecodeRuntime.step` | event/input-generation before `target.execute` and joined consumer completion all8; tie selected retained raw output to first Target argmax | fresh required semantic F and Product publication still unproved |

## Required and optional admission

**Acquisition integrity required:** frozen96 POST and exact1024 usage/SSE, phase barrier, request/slot/rank join, correct FULL Graph branch, complete original Host ledger, source restoration, controller exit/stop/all8 idle. If any fail, mark run invalid and retain raw for fault analysis.

**Selected lineage required for a readiness claim:** selected predecessor output-prefix generation, the real eligible release set and successor request identity, actual residual shape and first Target input storage generation. Missing one invalidates that specific claimed frontier, but does not erase independently admitted Host/output observations.

**Per-edge optional timing:** a device event or native join can certify only its typed owner and clock domain. Unknown prefill side stream, KV/seed producer, rank join or publication time remains `null`; no fabricated zero, no cross-rank device-clock subtraction. Such missing edges prevent a finite legal concurrency-window or attained Scheduling interval, while other matched observations remain usable. The separate profiler/synchronized diagnostic is an intervention and cannot be numerically substituted into unmarked Current.

**A0/A1 controls mandatory for cost transfer:** run instrumentation-off matched controls and instrumented B with exact contract, cache/residual shape, request ownership, acceptance count and release differences. A lineage-only B can be admitted under perturbation; timing transfer to Current requires compatible controls. Different full token strings are not by themselves a failed match because replay/numerical variability was observed in Run542.

## Next implementation gate

The client eligible-release record is now staged as `scripts/loop080_frontier_client.py`: it copies the validated client, adds monotonic attempt/acquired/DONE/end/after-context marks and labels unique parent uncertified. It does not change the request body or semaphore. The separate validator admits48 c12 Host chronology and eligible release sets only. CPU fake transport passed48 requests/36 successors/peak12 and the prior17 transport cases;7 synthetic negatives reject wrong order, missing DONE, false parent, wrong output, duplicate index and wrong c12 scope. These are source-only checks, not live readiness or formal E2E.

Next implement selected measured cohort1→2 device/Host hooks. First do source-hash/anchor preview and CPU synthetic validator negatives for missing phase/rank/request/footer, wrong FULL serialization and unjoined event promotion. Review exact installation/restore and cleanup fail-closed behavior before live execution. Keep the prior-slot output-completion path explicitly `unknown` if its current source cannot certify safe output ownership; do not make an early-publication speed claim from count-history alone. A subsequent conditional all8 mixed-service B should be designed from A's actual shape/window and remain separate from the strict C⁺/B capacity proof.
