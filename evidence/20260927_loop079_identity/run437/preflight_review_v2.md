# Run437 preflight review v2 — Astra

2026-09-27. Current four-script/source_check_v2 review only, with in-memory patch compile/hash checks and a fully mocked shell failure test. No service, NPU, docker command, framework import, source install or runtime modification executed in this review.

## Verdict: HOLD for one remaining cleanup P1

The previous validator promotion failures are substantially corrected. The acquisition is now explicitly conditional and does not claim a complete all43 row certificate. One stop-verification return-code bug still permits restoration after a failed process probe; fix before install/start.

## P1 — verify_stopped loses an intermediate failure under cleanup's set +e

cleanup disables errexit (`set +e`) and calls verify_stopped. Inside verify_stopped, the docker /proc probe and npu-smi command have no explicit error propagation. Bash returns the final Python HBM parser's exit status. Therefore a docker probe exit1 (live worker or inability to inspect container) followed by an apparently idle HBM parser exit0 makes verify_stopped return0. cleanup then restores sources despite unproved process stop.

This was reproduced without any real external command: extracted the current verify_stopped function, replaced curl/docker/npu-smi/python3 with shell functions (curl1, docker1, npu-smi0, final parser0), and observed `verify_rc=0`. Evidence: `cleanup_v2_mock.json`. This is a service-safety failure, not merely a reporting issue.

Minimum fix: explicit failure returns for each independent probe, e.g. docker ... redirection `|| return $?`, npu-smi ... `|| return $?`, and final parser status returned. Alternatively aggregate named status codes and return nonzero if any failed. Do not rely on restoring `set -e`: functions invoked in conditionals have subtler errexit behavior. Add the mocked docker-fails/HBM-passes case and npu-smi-fails case to CPU selftests. Existing cleanup stop_rc/restore_rc/SHA reporting can then enforce the intended stop-before-restore gate.

## Prior P1 fixes verified

- Runner now refuses nonempty acquisition directories, checks process/zombie states before install, records NPU/Host preflight, traps INT/TERM, reports cleanup codes and compares before/after SHA. Stop command failure skips restoration; restore failures propagate to final status. The remaining issue is the nested verifier failure noted above.
- Benchmark gate checks60 POSTs,48+12 error-free1024-token requests,40 capture and40 runtime files, then invokes offline validation.
- Capture records entry generation, actual graph args/kwargs descriptors, runtime input/position descriptors, and FULL/state pointer match flags. Validator requires exactly one selected FULL binding and descriptor equality. This now connects the snapshot source to captured/replayed input storage for the supported entry.
- Target logits indices now actually index the route matrix used for the conditional retained union. Target query prefixes are exact12×8, and logits/query/sequence/slot tensors are compared across ranks. The prior accepted-but-ignored permutation is fixed for conditional route attribution.
- CP snapshots now include seq_lens,input_positions,start_pos,local_query_start_loc,local_seq_lens. Validator checks their values against source formulas. I compared them to FixedTargetMetadataUpdater.update: start_pos=seq_lens-8; local sequence length is clipped by the rank's query-block offset, or zero for inactive local request segments. Prefix geometry is consistent with local12-row chunks.
- Ordered EP ranks and AllGather dispatcher are now checked along with maps/count parity. Output is `diagnostic_valid`, `all43_row_identity_certificate=False`; all43 row composition remains CONDITIONAL_NATIVE and external retention OPEN. Missing per-layer branch/layer→CP/native composition is explicitly acknowledged rather than promoted.

## Remaining limits / nonblocking hardening

1. This is an all43 route/metadata diagnostic, not the complete Run424 row certificate. Full logits→hidden row mapping does not prove all earlier layer rows stayed in that order. Current explicit conditional labels are appropriate.
2. CP descriptor validation checks shape capacity and values but not all dtype/stride/device expectations or exact binding ratio/layer membership. The current cp_value_map_valid should mean the five captured prefixes match formulas, not that every native attention binding is complete or correctly selected. Retain the named layer→CP/native limitation.
3. Entry generation is incremented at capture, but replay lookup is keyed by id(entry) without retaining the actual entry object. Frozen startup's persistent entry dictionary makes reuse unlikely; production-proof-grade capture should bind wrapper/capture lifetime explicitly. This is not a demonstrated current acquisition failure.
4. Helpers compare metadata rather than non-tensor values (tensor_tree records only non-tensor type). Treat the claim as input_ids/positions storage binding, not a proof that every runtime argument is unchanged.
5. source_check_v2 covers six borrowed patches; helper/validator scripts are compiled but not content-pinned in that manifest. Record reviewed helper/validator/runner hashes with acquisition evidence. Fail closed if edited after review.
6. Full source path joins and producing-stream completion remain conditioned on the known FULL graph path; sampled native tests do not prove arbitrary other graph branches. Validate acquisition output before any downstream numerical use.

## Checks

Six original/patched SHA pairs matched source_check_v2 and all generated files compiled in memory (exit0). The shell mock deliberately exposed verify_rc0 after a process-probe failure; no live external commands were called. No diagnostic capture files were consumed because acquisition has not started.

## Review-time hashes

- `loop079_row_capture.py`: `632d87dfbb1b84109eb0740c5bdca8446fa41ec075b2c55453faec5ad12de377`
- `loop079_row_capture_patch.py`: `6edb6247c1de807a4ea229ac17fa68b99e0ce4bf2beec5e26fa357028c47e103`
- `loop079_row_capture_validate.py`: `0d6f67f1ed879cbbf6419774402069a2f43893a702f84fe366fd24dac7181381`
- `run_loop079_row_capture_run437.sh`: `ab845a613ce44f61c165d2a6fd5b060885101e83723cb843478bf81b38f24589`
