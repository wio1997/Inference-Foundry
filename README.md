# GLM-5.3-Flash w8a8 on Ascend 910C — experimental source

This orphan-branch source snapshot derives from [DeepInfer](https://github.com/deepseek-ai/DeepInfer), base commit `553741557a57df44a46fb3ab72b6aafeaf945337`, plus the validated adaptation. The upstream Apache-2.0 license is retained in `DeepInfer/LICENSE`. Six image assets and vendor submodules are omitted. The source files are frozen from the tested snapshot; `portable/` is a later, unvalidated portability edit.

## Tested environment and scope

The source was tested on aarch64 Ascend 910_9382, driver 25.5.0, image `quay.io/ascend/vllm-ascend:v0.30.0rc2-a3` (image digest `sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b`), torch 2.10.0+cpu, torch-npu 2.10.0.post7 and flash-linear-attention-npu-a3 26.9.1+deva4a7958. CANN and custom OPPs came from that image; the precise CANN version is not yet recorded. These pins describe the tested image, not a validated image rebuild recipe.

The checkpoint was read only; weights are not included. The quantized weight index, config and template SHA-256 values were `dfaf0918f4c088c98b011def38475d9b1cb395f35ed5e279abc0c57610e23be4`, `33e63ec7fe607658be712bd6dd3c16c6549960d8e7f0483d34b939881b55f943`, and `34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891`. Model source manifest SHA-256: `7b6f73ccab6987daec9d29fdc46804a7a67a25c684222d5310914fb36c8b1857`.

Four cooperating ranks ran the full 45-layer model with a bounded 128-token context. One arithmetic reference case matched token IDs; a multi-turn recall case produced final answer 42 but reference reasoning diverged. Archive-v2 mid-generation restore reproduced 44 generated IDs on four ranks, both without cache and with a 128 MiB cache. Cache hit counts were zero; no reuse speedup is established. Generation stop 154827 and tokenizer EOS 154820 are distinct. These are scoped experimental results, not independent whole-model quality, performance or production acceptance. TP/EP production behavior, batching, service lifecycle and long-context quality remain unverified.

## Portable draft

`portable/worker.py` retains the restore and bounded generation experiment but parameterizes checkpoint, source and custom-library paths. It is a draft: the original local controller and host-specific launcher are not published, and this edited worker has not been run. A separate operator-owned launcher must implement resource locking, verify physical die-to-process mapping, prepare the request JSON and per-rank `go` barrier files, and launch four HCCL ranks. Do not treat the sample messages as a complete request JSON. Set `GLM_MODEL_DIR`, `GLM_SOURCE_DIR`, `GLM_ASCEND_LIBRARY`, `GLM_EXPERIMENT_RUN_DIR`, and `ASCEND_RT_VISIBLE_DEVICES` for each process. `compare_reference.py` accepts two compatible four-rank result JSON files; its changed rank mapping is unvalidated. No container deployment command is claimed reproducible yet.

Submodule commit pins are listed in `SUBMODULES.md`; fetch explicitly from their upstream URLs in `DeepInfer/.gitmodules`. Build and dependency compatibility outside the tested image are unverified.
