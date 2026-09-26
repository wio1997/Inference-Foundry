# Run300 invalid diagnostic attempt

The first cross-layer census attempt exited 1. During the first request, the diagnostic constructor treated an empty `model.layers.0.self_attn.attn.kv_cache[0]` placeholder as a one-byte view and raised `typed view outside storage` on all eight workers. Eight `rank*.jsonl` files are empty; no registry, interval or liveness evidence was produced. The client stopped at 11/12 and is not a correctness or performance measurement. The service cleanup completed, active marker was removed and all eight NPUs returned to idle.

The view-envelope helper now treats `numel()==0` as a zero-byte interval and only checks occupied views against storage length. A container CPU unit check returned `[0,0]` for an empty BF16 tensor and `[0,24]` for a 2×3 FP32 tensor. Run301 repeats the same read-only diagnostic with this bug fix; no inference-path candidate changed.
