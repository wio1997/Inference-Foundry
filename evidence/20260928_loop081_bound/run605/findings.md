# Run605 guarded dispatch diagnostic — invalid full packet

Run605 kept the frozen DSpark7 product behavior and ran warm48→measured48, 32K→1024 c12 on all8 ranks. Basis and Product admission passed: 48 measured clients, 32 worker files, 64 Runtime files and 281 Product raw files. The diagnostic client reported 581.947 tok/s over 84.461 s; event/metadata/JSON observers perturb this run, so it is not formal performance evidence.

The new dispatch admission failed exactly at `Draft input ordinal list`: all ordinary DSpark input/context hooks emitted zero rows. The ordinary proposal still executed. Source and Astra post-review identify the observer error: warmup handoff replaces `drafter.runner` with a FixedDP1 shim, so Run605's `_p602_marks` owner guard silently skipped the measured ordinary calls. Do not infer zero Draft work or zero context KV writes.

Run605 retains a narrower observed Target/Graph census: 44 ordinary Target/proposal pairs per rank, 30 NONE→FULL-wrapper fallthrough and 14 FULL replay across four cohorts; counts agree on all8. These fields do not form the complete producer→consumer DAG. No Scheduling/Execution, Resource/Hardware or Product bound endpoint or numerical Current→Bound gap changes.

`cleanup_status.txt` records `run_exit=1`, `final_exit=1`; stop, stopped/idle verification, restoration, source SHA/compare and script SHA/compare all returned 0. Health was down and all8 NPU process tables idle after cleanup. Original Run605 raw/admission failure is retained. Run606 introduces a canonical ContextVar ordinary-owner scope around the real proposal call and must pass a new all8 full admission. Its added ordinary context events mean Run605 and Run606 timings are not an unperturbed A/B comparison.
