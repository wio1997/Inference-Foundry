# Results

## 2026-09-20 — Environment freeze and baseline setup

Status: in progress; no performance KEEP/REJECT yet.

- 8×910B3 health OK, no running NPU processes before startup; idle HBM ~3.4 GB/card.
- Container `dsv4ab`, image `quay.io/ascend/vllm-ascend:v0.26.0rc1`, image ID and privileged mode captured in `evidence/20260920_baseline/freeze.txt`.
- vLLM, vLLM-Ascend, torch-npu versions in `evidence/20260920_baseline/python_versions.txt`; source commits and dataset/model config hashes in `freeze.txt`.
- Model index hash and safetensors weight inventory in `evidence/20260920_baseline/`.
- Current DP1/TP8 service launched. Its result is pending; older DP1/TP8 aisbench numbers are not adopted as current baseline.

Benchmark command after ready:

```bash
docker exec dsv4ab python3 /data/wio/Inference_Foundry/scripts/bench.py \
  --dataset /data/wio/vllm_ascend_26/datasets/GSM8K-in32768-num48-DeepSeek-V4-Flash-0731-w4a8-repeatRate0.9.jsonl \
  --out /data/wio/Inference_Foundry/evidence/20260920_baseline/bench48.json \
  --limit 48 --concurrency 12 --max-tokens 1024
```

Measurement caveat: the custom streaming client timestamps HTTP first token and divides the remaining elapsed time by completion tokens minus one. Output TPS is total completed output tokens divided by the measured wall span. Compare only with the same client and workload unless cross-calibrated.
