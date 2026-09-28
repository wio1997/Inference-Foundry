# Run654 — idle all8 light Event capacity qualification

The proposed full-cohort outer-stage recorder needs at most `5×1025+1 = 5126` timing Events per rank per cohort. With service stopped and all eight 910B3 idle, eight independent container processes each allocated 5126 `torch.npu.Event(enable_timing=True)`, recorded/warmed them on its own current stream, drained, re-recorded and read first-relative elapsed times. All eight exited 0, had monotonic Event times, and released NPU processes.

| Idle one-rank action, all8 concurrent | Range across ranks |
|---|---:|
| Python Event construction | 12.722–15.936 ms |
| First record enqueue | 29.216–36.216 ms |
| First drain | 12.346–36.552 ms |
| Warm-query plus re-record enqueue | 16.340–21.434 ms |
| Second drain | 32.754–53.195 ms |
| Serial first-relative elapsed extraction | 8.841–10.864 ms |

Independent Astra High review gives a scoped pass and notes that the probe did not save all individual elapsed values or prove process overlap from the files alone. The script checked monotonicity but did not explicitly reject NaN/Inf; a corrected observer qualification must add finite/nonnegative guards. The wall timings labeled `rerecord_enqueue_ms` and `query_and_elapsed_ms` in raw JSON include warm-query work and exclude second-round query, respectively; the table names above describe their actual scopes.

This proves only idle-device Event capacity and baseline observer mechanics under installed `torch_npu 2.10.0.post4`; the probe has no model, HCCL, Graph, serving or competing work. First allocation/record/drain must occur before the measured Runtime timer; serial elapsed extraction after an existing device drain may still delay Product publication by roughly its own Host work. These numbers are **not** primitive costs, a Runtime saving or an observer correction to formal E2E. The full48 ON/OFF control, per-cohort identity, and actual stage-cost class transfer remain open. Formal Current 571.681 tok/s; Framework-only numerical bound remains unadmitted.
