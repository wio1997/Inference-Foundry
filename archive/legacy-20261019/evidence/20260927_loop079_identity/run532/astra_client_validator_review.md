# Run532 — client validator review: final scoped PASS

2026-09-27. Source/CPU review only; no service/NPU. Final inspected validator `scripts/loop079_formal_ledger_client_validate.py` SHA256 **a2eed0d0c0c842c3d952ba8dd81d75774b5dd97c0906094f77892a7f5705c999**, archived as `validator_final.py`. **PASS for admission of this client's two frozen48-request artifact sets and its separate warmup-only gate.** It does not admit server lineage, native work, device readiness, fresh F, numerical equivalence or formal performance.

## Findings and corrections

The first displayed source only checked payload hashes/sidecar IDs and parsed the final usage. It could admit malformed earlier payloads, embedded IDs inconsistent with sidecar IDs, explicit errors, missing finish and duplicate usage; it also lacked actual concurrency, clock and phase-order checks. Sol corrected these before the first snapshot was copied. Therefore `validator_initial.py` is actually the intermediate **043ac0f99aaa7edea3e805785852565590599524df284b87e8a7daddf982652a** revision, not a preserved copy of the initially requested918cac version.

The intermediate version independently admitted three incorrect synthetic ledgers: no text delta while claiming success, backward event timestamps, and empty clock origins (`revision_cpu_review.json`). Final source fixes those as well as prompt-token consistency, timing presence and warmup-only uniqueness. Original failed checks are retained without promoting them.

An initial synthetic run used a toy dataset and was rejected at the newly added frozen-SHA gate for every case (`initial_cpu_review.json`). That run is **not evidence** of inner semantic validation. The final suite uses the actual frozen dataset bytes and permits the positive fixture before claiming negative-control success.

## Verified final gates

- Exact48 named request files and48 summary rows for each phase, identical per-request summary/file records, fixed phase/scope/n/success/fail/c12/max1024, frozen complete dataset SHA `4d9885088dac6b681c52c5ce12d39d192477ed95282f58c828da17e47fb2797d`, per-row dataset digest and body hash computed through the unchanged client body constructor.
- HTTP200/error-free/exact1024, valid prompt counts matching input_tokens, positive finite TTFT/TPOT and positive text-event count; requests stay inside phase wall intervals; measured interval follows warmup; actual interval-sweep peak is12, not merely a summary claim.
- Every event has sequence/hash/base64/response identity and a nondecreasing integer receipt time inside the request. Every non-DONE payload is JSON-parsed, embedded ID joined, errors rejected, single-choice/index0 checked and actual text-delta flag/count reconciled. Exactly one terminal usage equals row usage; finish must exist; DONE must be last and have exact bytes.
- All96 response IDs are distinct in two-phase mode; both clocks have nonempty matching boot/time-namespace/offset identity. Warmup-only mode admits exactly48 unique IDs and rejects a simultaneous measured-dir argument; missing measured-dir fails in ordinary mode. A warmup-only result has a distinct status and cannot be treated as two-phase completion.

## Independent CPU evidence

`negative_controls.py` constructs temporary artifact trees using actual frozen dataset bytes. It does not call a client, HTTP server or NPU. `final_cpu_review.json`: **20/20 expected outcomes**, two positives and18 negatives:

| Positive controls | Negative controls |
|---|---|
| Complete two-phase48+48; warmup-only48 | malformed earlier JSON; wrong embedded ID; explicit SSE error; no finish; duplicate usage; row/final usage mismatch; event outside request; clock mismatch; peak48; reversed phases; no-text success; backwards event time; empty clock origins; duplicate ID across phases; duplicate warmup ID; wrong body hash; wrong event byte hash; changed frozen dataset |

The revised no-finish control preserves its text-delta metadata, so it now fails on missing finish itself. The intermediate control was confounded by a missing sidecar delta flag; final control removes that ambiguity. The suite records exception details and requires the positive fixtures to pass. Source review confirms branch behavior in addition to those selected cases; this is not exhaustive schema fuzzing.

## Controller requirements and limits

No further validator change is required before building the guarded controller under this scope. Pin validator **and imported client source** together; the body check is only as stable as the imported function. Use actual container/module invocation, require nonzero validation to stop phase progression, keep fresh output directories, and reject stale result files from earlier failed invocations. The validator does not prove benchmark/controller exit codes or successful source restoration; those remain independent guards.

The artifact validator establishes internal consistency, not authenticity of arbitrary forged records. Hashes need the run manifest and actual acquisition provenance. Clock identity comparison does not calibrate Host/device clocks. TTFT/TPOT are checked for usable values, not independently recomputed from low-level network arrival. Fragment timestamps describe client receipt callbacks. Payload-preserving server/client joins and all8 Runtime correctness remain subsequent mandatory gates. Frozen input hash certifies dataset bytes, not inferred32768 tokenization; actual tokenized prompt/usage must be reconciled in the server ledger.

Current formal571.681 tok/s and every Bound endpoint remain unchanged. `review.sha256` pins this report, reviewed snapshots and CPU artifacts.
