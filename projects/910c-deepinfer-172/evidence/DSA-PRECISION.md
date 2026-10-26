# 910C DSA precision evidence

This does not establish whole-model accuracy, a serving implementation, or
performance parity with the frozen vLLM baseline. The short-context engine's
512-token guard remains in place until real paged prefill integration passes.

## Independent evidence

The image's compiled Ascend extension loads directly through importlib; no
vLLM Python modules are imported. dsa-schema.output.txt records the actual
four operator schemas and confirms that schema registration did not initialize
the NPU. Custom OPP registration must occur **before** torch.npu.set_device:
v1 failed tiling lookup/used legacy system kernels; v2 corrected that order.
Both runs and their exact source payloads are retained under evidence/.

Eight CPU batching/causality/quantization contracts passed in both runs. v2
validated exact top512 sets for unfiltered C1/C2 including seq_len32769 and
2048 unique source candidate blocks, with no future keys. Both compressed MLA
cases passed the existing pointwise atol.0005+rtol.008 and relativeL2<=.006.
Candidate-consumer QLI had one differing selected key; SWA17 and SWA32769
failed the original FP32 pointwise reference. Those failures remain failures
under the original reference, even though their relative L2 was about .0021.

## Why a separate precision oracle is necessary

The actual installed source, verified by SHA256 on172 (dsa-source-audit.output.txt),
defines intermediate rounding absent from the ideal FP32 reference:

* QLI FixpSToL1 enables ReLU and DEQF16 with scale0x3a800000 (1/1024),
  converting the INT32 dot product to FP16. ProcessVec0 multiplies weights
  and query scales in FP16. ComputeWs reduces these FP16 operands in FP32;
  ProcessVec1 then multiplies the result by the FP16-stored key scale in FP32.
  The positive common 1/1024 factor does not affect ordering.
* SWA SoftmaxFlashV2Compute emits unnormalized exponentials; ProcessVec1SingleBuf
  rounds them to BF16 before cube PV. ProcessVec2 divides the FP32 numerator
  by the FP32 softmax sum, including the sink. Multiplying a fully FP32
  probability tensor by V describes a different arithmetic boundary.

An independent CPU replay of actual v3 input/output captures reproduced
**all four** source-precision top512 sets exactly, including the original
one-key candidate mismatch. SWA CPU replay passed the **unchanged** numeric
tolerances: relativeL2 .0000107632 for17 tokens and .0004071094 for32769.
The original FP32 comparisons are still reported and still fail. This is a
separate source-defined operator contract, not a relaxed tolerance. The replay
reports npu_initialized=false and does not modify production inputs/outputs.

Source hashes (under the image's custom_transformer ascendc/ directory):

| Source | SHA256 |
| --- | --- |
| quant_lightning_indexer_v2/arch22/quant_lightning_indexer_v2_service_cube_arch22.h | 2b586bb8c894fbb0ea6887c7d0bd69332f2ab54570294ee6f407ecf346296352 |
| quant_lightning_indexer_v2/arch22/quant_lightning_indexer_v2_service_vector_arch22.h | fcddf4e682e5c742f1190dbe673da909bcdf745dbd6dc1cb66f700fa9b41b8a8 |
| sparse_flash_mla/arch22/sparse_flash_mla_swa_block_vector.h | 72710e7c348fd6eed515355c62bf849dbb69361cb59d6e6a47cb30c93859e02a |

## Next integration gate

Run the source-defined oracle suite again on fresh NPU execution, keeping the
original FP32 diagnostic suite and evidence intact. Include actual checkpoint
index projections at layers2/20/24 (C2 source, candidate source, candidate
consumer), real INT8 query weights, source-owned keys and RoPE positions at32K.
Then validate multiple requests, vectorized prefill, source-sharing topology,
cache chunk boundaries and full model generation before changing the engine
context guard or attempting the fair C8/P90 benchmark.
