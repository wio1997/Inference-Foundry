# Active deployment target: Ascend W8A8 / INT8

The active target is DeepInfer loading and generating text with the existing
DeepSeek-V4.1-Flash-1017-w8a8 checkpoint on Ascend 910C. Port or integrate the
installed vLLM-Ascend implementation's quantization metadata, scales/offsets,
weight layouts, sharding and actual INT8 operators. BF16 mixed attention/norm
and CPU-offloaded Engram are allowed where required by that working scheme.

The native ModelSlim W8A8 entry now loads the complete40-layer text backbone
with resident INT8 experts and generates short text on16 logical910C devices.
Real-checkpoint INT8 primitive quantization/compute boundaries pass independent
CPU validation. The original ideal-CPU full-expert and full-layer pointwise
comparisons retain failures; short generation is not whole-model numerical
acceptance. Full BF16 expansion of another FP8/FP4 checkpoint is not this route.

## Native W8A8 status on172

`deepinfer.engine.ascend_w8a8` uses its own loader, model and HCCL expert
dispatch. In the recorded diagnostic, all40 layers loaded in32.965s and the
prompt `请计算1+1，只输出答案。` generated `答案：2`. The eager engine remains
single-sequence, EP16 with replicated dense modules, and max_tokens<=512.
Generation took29.401s including11 input and4 output tokens; this diagnostic
must not be compared with a serving throughput baseline.

`deepinfer.npu_ops.dsa910c` separately registers and invokes the pinned image's
CANN extension directly; it imports no vLLM Python engine, forward context,
model runner or scheduler. Call `load_ops()` **before the first NPU device
initialization**, otherwise CANN caches the wrong custom OPP search path.
The native indexer loads signed INT8 ModelSlim query weights, FLOAT index
key/weight projections, source-owned INT8 keys with FP16 scales and32K RoPE.
The native page cache supports source sharing and physical SWA page rings.

Actual NPU operator validation covers C1/C2 causal top512,32769-token C2 tails,
2048 candidate blocks, source/consumer indexers with real layer2/20/24 weights,
and BF16 paged SWA plus compressed MLA. An additional eight-request test has
20 packed queries, lengths1024..32769, reordered request slots and partial C2
top512 padding. All selected sets match the independent source-precision CPU
oracle; MLA passes pointwise atol.0005+rtol.008 and relativeL2<=.006. Thirteen
CPU contracts cover host coordinates, ring wrap, chunk boundaries and RoPE.

The ideal FP32 diagnostic failures are preserved. A separate CPU oracle models
the actual installed source's FP16 QLI score/weight-scale boundaries and BF16
unnormalized softmax exponentials before PV, without relaxing tolerances or
altering production values. Operator source SHA256s and raw failures/successes
are recorded in the parent control project's DSA-PRECISION.md and Task Runs.

```bash
PYTHONPATH="$PWD" python -m pytest --import-mode=importlib -q \
  tests/test_ascend910c_dsa.py tests/test_ascend910c_paged_cache.py
PYTHONPATH="$PWD" python examples/ascend910c_dsa_semantics.py \
  --model-dir /path/to/modelslim-w8a8 --output /project/runs/dsa-semantics.json
PYTHONPATH="$PWD" python examples/ascend910c_dsa_batch_probe.py \
  --config tests/data/ascend_w8a8_config.json --output /project/runs/dsa-batch.json
```

The separate `AscendW8A8PackedEngine` now wires DSA/index projections and page
caches into all40 layers, with vectorized C1/C2 compression and chunked Engram
CPU history/selected rows. Its actual16-rank TP8/EP16 short-generation run
produced the same IDs `[7624,768,20,1]` (`答案：2`) as the scalar entry;
load39.448s, generation15.177s including11 input tokens. The exact-select-all
short prefix uses the existing native eager attention; indexed long context
uses CANN. This is functional evidence, not a throughput comparison.

The scalar engine's512 guard remains. The packed engine completed a real
32768-token frozen-baseline prompt and generated8 tokens matching the saved
vLLM prediction prefix (` Bows are $5 each, a`). This run invoked2782 native
CANN attention calls and peaked at41,926,611,968 allocated bytes. Its195.466s
includes full cold prefill plus8 output tokens and is not a serving baseline.

Public CANN documentation and this image's custom ABI differ: the image
explicitly rejects `cmp_ratio=1` with `has_cmp_kv=False` and requires0. Match
operator calls to the pinned image and its actual source.

### Native DP2/TP8/EP16 service

`examples/ascend910c_w8a8_worker.py` runs16 resident native workers. Attention
is TP8; distinct requests split across2 DP groups and experts use global EP16.
Each TP rank owns a token slice for routed/shared FFN, followed by TP gather.
The24 resident experts per rank use two grouped INT8 matmuls per FFN rather
than a Python loop over individual experts. Shifted mHC and routing RMSNorm
use the pinned image's compiled operators.

`examples/ascend910c_w8a8_api.py` is a separate CPU HTTP process exposing
`/v1/completions`, `/health`, `/v1/models`, and native `deepinfer_*` metrics.
Only the deployed tokenizer is reused from vLLM. Requests and token events
travel through the project control directory. Prompts use the same raw
completion tokenizer semantics as the frozen benchmark; text-to-chat formatting
belongs to the caller. The present sampler supports greedy temperature0.

The fixed serving profile has4 requests per DP replica, context34816 and
chunk512. Cache snapshots include SWA physical rings, compressed KV, INT8 index
keys, FP16 index scales, C2 pending values and CPU Engram history. Snapshot
admission preserves cold seed anchors and does not create request-specific
children after restoring a prefix. Boundaries29440/29568 give actual hits
235904/262144 (89.990234%) for the frozen eight-request workload.

Native MC2 uses a dedicated HCCL communicator configured with
`ProcessGroupHCCL.Options().hccl_config={'hccl_buffer_size':2560}`. The A3 kernel
explicitly required2,145,386,496 bytes at the configured prefill capacity;
the ordinary200MiB group failed its window-size assertion. TP/control groups
retain the default size. The configuration interface is documented in
[the TorchNPU HCCL parameter guide](https://www.hiascend.com/document/detail/en/Pytorch/2600/ptmoddevg/Frameworkfeatures/docs/en/framework_feature_guide_pytorch/setting_HCCL_communicator_parameter.md).

Actual C8 required prewarm completed32768 input/32 output tokens per request:
all8 outputs matched the saved vLLM1024-token prediction prefixes exactly and
actual restored hits summed to235904. A subsequent formal attempt was marked
INVALID when cache admission evicted seed anchors; its partial timing is
excluded. The admission policy was corrected. Earlier startup failures and
raw evidence remain in the parent control project.

`w8a8_graph910c.NativeDecodeGraph` captures all40 target layers for four
requests per DP replica. CPU history/Engram selected-row staging and sampling
remain outside; CANN metadata/indexer/attention, cache writes and MC2 execute
inside the graph. Pinned host inputs and persistent device buffers retain their
addresses. Cold capture always replays before returning output.

The final profile uses standard input-only prompt lookup: n-gram orders8/4
propose up to63 tokens, and a64-query full-target graph verifies them. Each
emitted greedy token is checked by the complete40-layer target. A rejected
proposal restores logical lengths, CPU history and C2 pending state before a
native repair forward; speculative future cache rows remain invisible. The
reserved fifth slot is scratch storage for vectorized C2 writes. No saved
reference predictions enter the engine. DSpark/MTP is not implemented here.

Final native-v8 AISBench acceptance on172 (DP2/TP8/EP16,16 logical devices):

| Formal round | Output tokens/s | TTFT ms | TPOT ms | Success / full1024 exact |
|---|---:|---:|---:|---:|
| measure1 |716.497574|5502.0|4.6|8/8|
| measure2 |626.259413|5167.2|6.6|8/8|

Both rounds use32768 input/1024 output tokens, concurrency8 and actual prefix
hits235904/262144 (89.990234375%). Separate seed and C8 warmup are excluded.
Aggregate throughput is16384/(11.4333953+13.0808413)=668.3463273745184 tokens/s,
28.92% above the frozen vLLM-Ascend518.4171948453181 reference. Both individual
rounds exceed that target. All16 complete output texts match saved vLLM outputs
byte-for-byte. Peak allocated NPU bytes per rank were44,081,432,576.

Earlier native graph and CPU-staging baselines remain59.400212 and138.833430
aggregate tokens/s. Final throughput benefits from repetition within the fixed
prompt corpus and is not a general-input performance claim. Native TTFT is
higher than the tuned vLLM reference (885.7/776.8ms), despite higher aggregate
output throughput. SSE emits verified blocks; its token arrival pattern differs
from per-token decoding, so TPOT alone should not be read as uniform cadence.
Correctness acceptance is for this fixed greedy profile, not global model
accuracy or arbitrary sampling.

Raw AISBench data, request/event/output traces, source hashes, actual controllers
and container manifests are frozen under
`/data/tiankuan/wio/910c-deepinfer-172/final/native-v8`. Reproduction and original
vLLM restoration are delivered in the independent
`projects/910c-deepinfer-172` record folder on the user's Inference-Foundry branch.
The original vLLM evidence and container are retained.

The historical correctness reference below remains auxiliary evidence only.

# Ascend 910C correctness reference (experimental)

The opt-in `DEEPINFER_ASCEND_910C_REFERENCE=1` profile provides eager BF16
correctness paths on Ascend910_93xx. This historical BF16 reference does not establish full DeepSeek-V4.1
generation or performance; the native W8A8 acceptance above is a separate path. Default NVIDIA and 950 dispatch remains selected when the
profile is disabled.

Implemented and measured on Ascend910_9382:

- Explicit E4M3FN and packed E2M1 byte decode, unpacked UE8M0 or FP32 scales.
  Native FP8 tensors are avoided; incompatible packed runtime scales are rejected.
- ReplicatedLinear with decoded BF16 weights, FP32 accumulation, BF16/FP32 output,
  bias, output accumulation and padded strides; unquantized RMSNorm with residual.
- Public sparse MLA prefill/decode, including sink, natural-log LSE, lengths,
  duplicate indices, empty/all-masked rows, BF16/528-byte FP8/288-byte FP4 cache
  reads, and optional extra cache. Fused quantized output and batch-invariant
  guarantees are explicitly unsupported. Eager MLA uses its own schedule state.
- Public eager mHC expand, prenorm, Sinkhorn, pre/post/head mixing interfaces.
- Routed MoE primitives with real HCCL variable all-to-all, contiguous expert
  ownership and an explicit BF16 expert callback. This is not a DeepEP buffer
  replacement: EPLB, overlap and graph capture remain unsupported.
- Bounded checkpoint tensor/row reads, explicit Engram FP8 row decode, integer
  n-gram hashes including signed overflow, and BF16 Engram gate arithmetic.
  Engram helpers are not yet connected to the complete engine's fetch runtime.

Validation includes 104 regressions (14 upstream Sequence plus 90 reference
cases), two real FP8/FP4 matrices, actual two-rank routed experts, and complete
layer-zero BF16 arithmetic on two ranks. The layer probe covers embedding,
shifted mHC, attention, grouped output projection, shared and routed experts,
then compares seven stages with an independent CPU golden. Its output-trunk
relative L2 errors were 0.000539 and 0.001037. An Engram probe reads 24 real rows
(6336 bytes), executes the full real wkv projection and gate, and compares with
unchanged TileKernels CPU arithmetic (relative L2 4.21e-7). These are module
results, not equivalence to the model's native quantized activation/cache path.

## Run in an isolated environment

Configure visible devices before importing DeepInfer. The two-rank examples
require two available devices and use actual HCCL as part of the model module;
they do not run a separate communication smoke test. Do not share devices with
other workloads.

```bash
source scripts/ascend910c_reference_env.sh
export PYTHONPATH="$PWD"
export TILE_KERNELS_REFERENCE_SOURCE=/path/to/pinned/tile_kernels-2.1.0
python -m pytest --noconftest -q tests/engine/test_sequence.py tests/test_ascend910c_*.py
python examples/ascend910c_weight_smoke.py \
  --model-dir /path/to/fp8-fp4-checkpoint --output /path/to/weight-result.json
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  examples/ascend910c_layer_integration.py \
  --model-dir /path/to/fp8-fp4-checkpoint --output-dir /path/to/layer-results
python examples/ascend910c_engram_integration.py \
  --model-dir /path/to/fp8-fp4-checkpoint --output /path/to/engram-result.json
```

The golden source must be the pinned official TileKernels 2.1.0 sdist. Tests read
its unchanged pure PyTorch reference functions without importing its device
kernels. FlashMLA CPU golden functions are read from the pinned vendored source.
Examples never download weights. The layer probe reads one token per rank at
position zero, with no compressed KV, and is deliberately restricted to that
configuration. Its checkpoint read bound is 64 MiB per tensor; the Engram probe
explicitly raises this to 192 MiB for its 157 MiB projection matrix. Engram row
allocation remains bounded and does not materialize the huge embedding table.

## Checkpoint metadata and residency

If complete shards are present but an index omits tensors, build a separate view:

```bash
python scripts/prepare_checkpoint_view.py \
  --source /path/to/original-checkpoint --destination /path/to/new-view
python scripts/ascend910c_checkpoint_plan.py \
  --model-dir /path/to/new-view --output /path/to/residency-plan.json
```

The view validates shard headers, byte ranges, duplicates and configured Engram
coverage, then creates symlinks and a new index. It leaves the original index
and shards untouched. Create and consume it in the same filesystem namespace;
container-only absolute symlink targets will not resolve on the host.
Header validation does not establish full payload checksums or model accuracy.
ModelSlim and compressed-tensors W8A8 need separate loaders and are rejected.

For the measured 48-shard checkpoint, the index contained 96073 tensors and
omitted 12 Engram tensors already present in two shards. The corrected view
contains 96085 tensors and 510286023000 payload bytes. Fully materializing
BF16 weights while retaining scale tensors has a lower-bound estimate of
1550058261336 bytes, excluding cache, padding, activations and workspace. This
exceeds sixteen 64-GiB devices; streaming/offload or another supported compute
format is required. The layer/Engram probes demonstrate bounded reads, not an
accepted full-model residency policy.

## Toolchain and remaining engine work

The environment script selects CANN's bundled GCC 14.3 and matching libstdc++.
It does not install system packages. Native DeepInfer, DeepEP and DeepGEMM host
extensions were built with CANN 9.2.0, torch 2.10 and torch-npu 2.10; host import
does not prove that their 950 device kernels work on 910C. Upstream requests
newer torch versions, so current evidence is limited to the tested profile.
TileLang 0.1.15 requires a compatible TVM-FFI (0.1.12 was needed in this image).
With project-local dependencies and eager MLA metadata wiring, the official LLM
entry imported successfully and the same environment passed all 104 regressions.
Import success does not establish model construction, weight load or generation.

Remaining work includes full model weight/runtime wiring, bounded expert/table
residency, native activation and KV encode/store paths, compressed KV/indexer,
RoPE/fused transforms, sampling, 910C replacements for DeepEP UBMEM/URMA buffer
interfaces, full tokenizer/Engram integration, and end-to-end generation with
numerical regression. Same-chip IPC and simple BF16 graph proofs do not establish
cross-chip IPC, model graph capture or performance.
