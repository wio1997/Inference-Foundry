# Astra Challenger Review Package

1. Goal/contract: RESET.md及glm5-3/MISSION.md；完整功能、现有两机和正确算子，禁止改kernel/math、减少有效计数或放宽SLO。
2. Current: None / active PERF_KEEP为空。现役public241 V14、D0 nativeV2 TP8PP2DCP8 K2 cap2 resources3、D1 V1 TP4PP4DCP4 K1。最新远端b23845e4；性能baseline unknown。
3. Critical path: public/token prep→one replica native scheduler→execute RPC→PP target+MTP sample+postprocess+draft→async output copy-ready→MQ reply→FIFO scheduler.update_from_output→core output queue→SSE。read native PP draft broadcast implementation before assuming legality.
4. Main H1: ready oldest result might wait while new batch gets scheduled. Not established. Keep one hypothesis. Alternative batch grouping/metadata H2 is not activated.
5. Necessary source: sibling snapshots v1_engine_core.py, batch_queue_control.py, native_multiproc_executor.py, native_async_scheduler.py, native_scheduler.py, native_model_runner.py, native_async_utils.py, native_pp_utils.py, ascend_model_runner.py, ascend_worker.py, plugin_issue_budget_scheduler_v5.py/v3.py. source_identity.json pins actual container installation. Do not treat runtime/ versions as active.
6. Patch: none. Proposed conditional candidate only after evidence; reject FutureWrapper.done based early commit because it is lazy. Parent is writing observational trace, not activating performance patch.
7. Profile/trace: absent ready/commit/device correlated trace. Raw230 terminal_queue_native.stdout on166; RESET gives SHA and decisive serial106/107 line locators. CPU call wait is not device bubble.
8. Matched Run: none for code patch. Git manifest/summary230/231/234/235;230 serial106 cap3N4 wall14.595s versus107 cap2N4 8.826s, all token-vector hashes differ.231/234/235 16/14208 and all3SLOFAIL; salt/order/MTP confounded, cannot claim isolated product gain.
9. Negative evidence: native FutureWrapper.done is not readiness; CPU step counts are not physical device steps; direction cap2→3→2 remains confounded. Other causal falsifications unknown.
10. Options: one bounded diagnosisH1; STOP H1 if no ready critical delay and then choose measured gap. No architecture fork selected; no drain-first or cap/budget/cadence sweep.

Read necessary sources personally. Answer maxGap ranking (unknown allowed), route KEEP/STOP/PIVOT, top3 code questions (not active queue), explicit stop list, smallest distinguishing experiment, most likely evidence misread. Write review.md here. Do not operate services/devices or modify anything else. Missing raw access remains unknown; remote read-only SSH permitted if useful. Report fact/inference/unknown separately.
