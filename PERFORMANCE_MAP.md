# Performance Map V0

Updated: 2026-09-20 UTC. Evidence maturity: E0 until the current DP1/TP8 baseline completes.

| Stage / mechanism | Current measurement | Evidence | Gap or unknown |
|---|---:|---|---|
| End-to-end 48×32K→1024, c12 | pending | `scripts/bench.py`, frozen dataset hash in `evidence/20260920_baseline/freeze.txt` | TTFT, TPOT, output TPS, run variance |
| Prefill device | pending | service and future trace | per chunk kernel/HCCL split, critical path |
| Decode target/draft | pending | service and future trace | target versus speculative draft, graph coverage, token acceptance |
| W4A8 weight traffic | ~21.1 GB active model weights per selected-expert path (model-wide approximation) | `evidence/20260920_baseline/weight_inventory.json` | true rank sharding, cache reuse, batch expert overlap, draft pass count |
| TP8 communication | pending | future HCCL trace | bytes, latency, overlap |
| Host / metadata / D2H | pending | future timestamps and source | exposed host overhead, synchronization migration |
| HBM bandwidth reference | 1.3 TB/s observed by older RMS kernel, not current workload | `/data/wio/vllm_ascend_26/results/r16_prefill_device_forward_candidate/evidence/RUN.md:81` | repeat on this source and relevant access pattern |

Historical constraints: DP2/TP4 R39 retained R17+R20 and rejected R29 for c12 throughput; R23 target decode Super Kernel regressed TPOT. These are workload/topology dependent and do not settle DP1/TP8.

Priority after E2 baseline: diagnose the largest exposed term rather than patching a presumed kernel. Do not assign a Compute or Framework label before matching device and host evidence.
