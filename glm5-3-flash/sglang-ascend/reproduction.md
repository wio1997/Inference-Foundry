# GLM-5.3-Flash Ascend: source and AOT reproduction scope

This document describes the minimal contribution sources; it is not a new Ascend test or a claim that this contribution was benchmarked. The private-lab measured engine was 49002ca8aa4c64276905a4d7973a4213f54d6902. Apply the four ordered runtime patches to public SGLang ce0a8fea5f4245839d174b2b8c55692be1810a7f. The resulting new checkout has its own Git identity: compare the selected file SHA256 values with manifests/reproduction-sources.json rather than claiming its HEAD equals the old lab SHA.

## What is included

The runtime patches contain the paired Flash backend/recurrent/model/quantization integration plus R3 ordered HC4 normalization, R4 forward-local KPool CPU snapshots and R5 selected Sparse NoPE KV-length device reuse. Keep all matching companion imports. Do not overlay individual files on another installed SGLang version. The probe/build source selection is explicit and small; no API2 observer/control platform, weights, raw process environments, prebuilt libraries or large trace is required in this contribution.

The mHC boundary is **only** the FP32 normalization after the existing Torch softmax+epsilon: one column update followed by nineteen row/column pairs, HC=4, iterations=20, epsilon=1e-6. Projection, sigmoid, softmax, RMS, mixing and output dtypes stay in the original caller. The ordered template preserves the measured 910C Torch reduction association. The generated AscendC wrapper is recoverable from a committed 7111-byte text snapshot, SHA256 c6b161af7acf52e25019a730d201811179ac73cf58101850f6c61a3a51949331; carry its MIT notice and pinned upstream attribution.

## Isolation and build sources

Use the existing target container/toolchain, a dedicated candidate checkout, and separate build/cache/output directories. Set process-scoped PYTHONPATH to the candidate's python directory; do not perform a global editable installation, overwrite site-packages, replace torch/torch_npu/Triton/CANN, or modify weights. Do not add the isolated TileLang/TVM compiler paths to the model process.

Public compiler source is tile-ai/tilelang-ascend@2f22c6ac17a4235aa3de5ccf25729c3553b46e02. Pin TVM c2921fdaf795b1103d21abc962e83a209c7258d7 and the exact companion headers in manifests/reproduction-sources.json. Apply only the two pinned patches from that upstream checkout to its TVM tree after forward apply checks: tvm_backtrace_symbol_name_length.patch and tvm_slice_step_fix.patch. Preserve LF source bytes. Neither newer branch HEAD nor a wheel is an equivalent source identity.

The committed build_native.py configures CMake/Ninja with Ascend ON; CUDA/ROCm/LLVM/GTest/libbacktrace/ccache OFF, then builds tvm, tvm_runtime, tilelang and tilelang_module with parallel=4, without install. Its CLI is:

```bash
python3 -B "$CANDIDATE/lab/glm53_flash_ascend/repro/tilelang_reuse_r3/build_native.py" --source "$TILELANG_SOURCE" --build "$ISOLATED_BUILD" --output "$NEW_BUILD_EVIDENCE"
```

This is a prospective invocation of an existing tool, not an invocation executed for this contribution. Native compiler availability/ABI must be checked in the selected environment. The original public wheel failed GLIBC_2.38 vs target GLIBC_2.35; a source build worked in isolation, but importing the private compiler frontend in the CANN model process still failed. Do not reintroduce that route as the normal service entry.

The ordered mathematical source is mhc_sinkhorn_tilelang_npu.py. Its _sinkhorn_hc4_fp32 factory targets ascendc/platform A3. Compiler execution belongs in a separate process with only the pinned compiler paths and a dedicated cache. Existing export_public_tail_aot.py does **not** compile that factory or arbitrary generated CU: it validates and copies a pre-existing fixed LibraryGenerator cache, license and ELF; it never unpickles params.pkl or performs installation. A fresh compile-to-package driver and its reproducible compiler command are **not established by this contribution**. The generated CU also needs the fixed TileLang tl_templates/ascend headers and the compatible CANN toolchain; it is not a standalone C++ source independent of those headers.

## Exact binary gap: do not claim one-command reproduction of measured B49

The measured SO is 548872 bytes, SHA256 8b4ab5f9c4eb37f64b4d9746def519a2247a959119db6e19caae0297af988945; it is not in Git. Both the exporter and runtime deliberately accept only that selected hash. Source/template/CU availability does not prove byte-identical output from a fresh compiler invocation. Therefore an unchanged measured-B49 process needs the controlled original artifact. If a rebuild produces different bytes, do not bypass its hash guard or relabel it as the old test: bind the new binary/dependencies/source, adapt the explicit artifact identity in a new source commit, and run the fixed numerical/CANN/stream gates followed by full-model validation. Default-off Torch operation can be used without loading any AOT SO; it is a fallback, not the measured fused-B49 result.

## Runtime ABI and flags

The selected pure runtime is mhc_sinkhorn_tilelang_aot_npu.py. It verifies the library before ctypes.CDLL(RTLD_LOCAL), then calls void call(void* input, void* output, int64_t n, aclrtStream). It supplies torch.npu.current_stream().npu_stream and records input/output tensors on that stream; no manual init or host workspace is used. Input/output are distinct contiguous NPU FP32 [T,4,4] tensors. Empty T returns without launching. The current device must match the tensor. Do not preload the NPU library before a worker fork.

Measured optimized-path configuration, separate from shell connection configuration:

```bash
export GLM53_MHC_SINKHORN_FUSED=1
export GLM53_MHC_SINKHORN_IMPLEMENTATION=tilelang-aot
export GLM53_MHC_AOT_LIBRARY="$CONTROLLED_AOT_LIBRARY"
export GLM53_MHC_AOT_SHA256=8b4ab5f9c4eb37f64b4d9746def519a2247a959119db6e19caae0297af988945
export GLM53_KPOOL_FORWARD_METADATA_CACHE=1
export GLM53_SPARSE_KV_LENGTH_DEVICE_CACHE=1
export SGLANG_OPT_USE_TILELANG_MHC_PRE=0
export SGLANG_OPT_USE_TILELANG_MHC_POST=0
export SGLANG_OPT_FUSE_MHC_POST_PRE=0
export FLASH_PD_PERF_MODE=0
export FLASH_MHC_FIXED_ROWS=0
export FLASH_B103_PERSISTENT_OWNED_STORAGE=0
```

All three new performance flags default to 0. For staged rollback, R5 flag0 returns to the C17 call; R4 flag0 returns to ED216 metadata behavior; R3 fused0 returns to the Torch normalization. Disabling R3 alone while retaining R4/R5 is a different combination and must not be called the original A. Retain all state/layout/native-read-only contracts and bypasses; no Graph/spec/TBO/preplanned/CP guarantee is added.

Keep the tested single-node baseline budget2048: --device npu --dtype bfloat16 --quantization modelslim --model-impl sglang --tp-size 16 --dp-size 1 --ep-size 1 --attention-backend ascend --mamba-ssm-dtype float32 --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled --context-length 18432 --chunked-prefill-size 2048 --max-prefill-tokens 2048 --prefill-max-requests 4 --max-running-requests 4 --max-total-tokens 81920 --max-mamba-cache-size 64 --page-size 128 --mem-fraction-static 0.75 --disable-radix-cache. This is a parameter record, not a complete safe launch script or a new deployment authorization. Bind model/service paths and free resources separately. Do not enable MTP/PD/MemCache or change the mixed/dynamic/CP settings. Verify actual aggregate packed-extend budgets stay <=4096; a per-request limit alone does not prove that condition. Context capacity must include actual template input plus output. No automatic truncation is permitted. R6 did not show a uniform win for budget4096, so it is not the contribution default.

## Minimal correctness and source proof

Only use one confirmed-free logical device for a targeted probe; probe_cann_aot.py and test_boundary.py deliberately expect logical npu:0, which must be mapped by the authorized operator, not assumed to be physically free. They check exact candidate Git bytes and write fresh evidence outside the source tree. probe_cann_aot.py requires --candidate-tree, --expected-commit, --component-output-dir and --output-dir; the expected commit is the actual new candidate commit, not the old lab B49. Runtime loading additionally needs the explicit bound library variables above. The boundary gate is fixed at comb atol2e-7/rtol2e-6/max-relative2e-5 and BF16 downstream <=1ULP; preserve its empty/tail/odd-shape, invalid-input, batch independence and full synthetic caller checks. CANN format_cast before/after and a nondefault stream are also exercised. These are component tests, not accuracy or performance acceptance.

On every real service rank verify that GLM5Next loads the Flash backend and ModelSlim implementation from the candidate tree, and that the selected mHC implementation is tilelang-public-ordered-tail-aot-ctypes with the expected AOT SHA. The identity directory/candidate commit variables are process-scoped evidence roles; use a fresh directory each attempt, never a copied marker as execution proof. Keep the first-call synchronization in warmup before formal timing. Verify GC, diagnostics, observers and cache policy are equal across variants. Do not import the compiler frontend into the service merely to record its version.

Then proceed directly to bounded full-model checks: short smoke; natural-stop16K head/middle/tail, repeat/different/concurrent requests and persistent-state behavior; fixed formal actual16384 input/1024 output with ignore_eos=true, seed20261022, temperature0, top_p1, checkpoint-default thinking, prefix cache disabled, C1/C4. Exclude compilation/warmup and profiler from timing; retain all repeats/failures, TTFT/TPOT/E2E/tokens/successes and sampled per-rank memory. The old finite R3/R4/R5/R6 evidence is not a new precision certification or a benchmark of a rebuilt contribution.

## Relative source layout and license

The source bundle may sit below this contribution, but probes run against the actual SGLang candidate checkout. Preserve `python/sglang/...` for runtime imports. Preserve `lab/glm53_flash_ascend/repro/tilelang_reuse_r3/probe_cann_aot.py` and its hardcoded sibling subtree `lab/glm53_flash_ascend/repro/mhc_sinkhorn_r3/test_boundary.py` **inside that same candidate Git tree**; the probe computes each source path relative to the candidate and verifies its committed blob. Copy and commit the selected helper subtree before testing and use the resulting new candidate SHA. Moving only the top-level script into a different Foundry directory while pointing it at an external checkout is not supported by the original code. Other explicit helper/caller paths and required cache filenames are listed per file in manifests/reproduction-sources.json. Do not copy the observation/control platform to satisfy these small dependencies.

The generated CU includes `tl_templates/ascend/common.h` from the fixed compiler `src` include root and requires its pinned transitive AscendC/CATLASS/CANN headers. The original exporter requires existing cache files named `kernel.cu`, `wrapped_kernel.cu`, `kernel_lib.so`, `auto_gm_idx.pkl`; a single committed `wrapped_kernel.cu` snapshot is provided for inspection and avoids duplicated source in the contribution, but does not by itself form that binary cache.

Pinned primary-source license verification: tile-ai/tilelang-ascend@2f22c6ac17a4235aa3de5ccf25729c3553b46e02 `LICENSE` is MIT, 1127 bytes, SHA256 `bfd7d57808112905dce724b04e7b3bebb823d3e74737d5d83e9194464015017b`, Git blob `5104c90367b39d1372f0eb17efdf1be794b96ff1`. The exact full notice is included as `native/r3/LICENSE-MIT.txt`; keep that text beside the selected template/generated source and retain the original copyright/SPDX notice in the template. This was verified by a read-only GitHub contents API call at the complete commit, not inferred from the project name. It is an attribution check, not a complete audit of every SGLang/third-party dependency license.

Optional R4 CPU helper probes must receive the canonical helper through `--helper`; their historical sibling default is not a packaged runtime module. The optional R5 exact-AST caller probe additionally requires `--baseline-caller`: use C17 ascend_backend.py reconstructed in a separate checkout after patch03, while `--caller` points to final B49. Do not pass the final file as both inputs. Its exact C17 reference is listed separately and must never overwrite the B49 runtime file.

## Offline source reconstruction and candidate preparation

The Node verifier only uses a temporary Git index/object directory and the existing source object store. It checks all four cumulative runtime checkpoints (48/52/53/54 files) without editing the supplied source checkout, importing Python or loading NPU/native libraries:

```bash
node "$SUBMISSION/tools/verify_submission.cjs" --source-repo "$SGLANG_OBJECT_SOURCE"
```

For a future separately authorized candidate, create a new Git worktree at the exact public base, preserving that history; apply patches in order with `git apply --check` before each application. Do not initialize a new repository after copying a tree. Copy this submission's `lab/glm53_flash_ascend/repro/` and selected workload builders into the same candidate at their original relative paths, and commit all candidate changes before any probe or model execution. Bind that new actual HEAD and the actual native artifacts. A new Foundry publication commit is not one of the original tested engine commits.

`build_r3_workload.py` reconstructs synthetic prompts using the selected model's actual tokenizer/chat template. The original prompt hashes are in `manifests/workload-reference.json`; this publication does not copy raw prompts or regenerate them. Verify exact rendered input length and actual server token counts; never truncate. Keep both the service random seed (`--random-seed 20261022`) and each wire request seed fixed. The old ready-time GC freeze, diagnostics and 15-second observer were controlled by the original measurement implementation; the listed launch parameters alone do not recreate those controls. This export does not carry that complete controller. A future benchmark must declare and equalize those settings and bind its own evaluation source.

The published compact evidence supports code review and inspection of original per-service measurements. Full SSE responses, complete raw traces, native binaries and the complete original control/evaluation archive remain under the original access controls. Consequently this public export does not enable independent replay of every original raw measurement or the complete original ABI/code-generation closure. No new service, benchmark or probe was executed for this export.
