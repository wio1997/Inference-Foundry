# Run430 — native row-order contract review

Date: 2026-09-27. Scope: read-only current source/package and official source review; one pure-Python label oracle. No service operation, framework import, NPU operation, patch or borrowed-source edit.

## Verdict

Run424 can replace several unnamed assumptions with explicit source contracts. It **cannot yet issue an unconditional all-43 Target row certificate**. The smallest valuable next native experiment is a single-device **MoE V3 EP-range index → abs → masked unpermute** semantic test, not another end-to-end generation. It needs a separately authorized idle-device window; none was run here. CPU work can close the DSA permutation algebra, but cannot validate loaded HCCL or compiled kernels.

First unresolved native mapping: for the actual W4A8 `quant_mode=1`, partial `active_expert_range`, `row_idx_type=0`, prove each local expert route's `expanded_row_idx` addresses the corresponding packed row, and each excluded route is suppressed after `abs` and zero probability. Full-range permutation tests alone omit this case. Do not treat a negative index as a universal signed token identity.

## Version and history scope

Installed package metadata: CANN OPP `Version=9.1.0`, build timestamp `20260730_231653901`; torch_npu `2.10.0.post4`, `git_version=5dd8ef3f9b375b5ae4a83538d5785754148c3302`. Reading files does not establish loaded-process library identity.

The official Ascend/pytorch commit records op-plugin submodule `8b9c8534fa41eff367a41c155843daa530ab3a08` ([submodule metadata](https://api.github.com/repos/Ascend/pytorch/contents/third_party/op-plugin?ref=5dd8ef3f9b375b5ae4a83538d5785754148c3302)). This is materially stronger than assuming today's master matches the wheel; wheel build/local modifications still need provenance.

Targeted performance-knowledge searches (index commit `db3beef223e0b5acd81ccccb444600c1b22aac8a`): R11 draft MoE decomposition, R16 prefill device-forward candidate, R13 Target metadata. These establish prior call/communication context, not a current native permutation certificate; R13 uses an older DP2TP4/EP8 topology. No whole-history inference used. Run424 and Run420 remain the graph/request and external-output prerequisites.

## Directly supported mappings

### 1. MoE routing and unpermute

Current `vllm_ascend/device/device_op.py:116` wrapper invokes Python `npu_moe_init_routing_v2`. The version-matched [op-plugin adapter](https://github.com/Ascend/op-plugin/blob/8b9c8534fa41eff367a41c155843daa530ab3a08/op_plugin/ops/opapi/MoeInitRoutingV2KernelNpuOpApi.cpp), lines228 and322, selects **aclnnMoeInitRoutingV3** on A2 with CANN9.1; the V2 special branch is for older CANN or310P. Therefore CANN `aclnnMoeInitRouting`/V2 docs are not the relevant ABI just because names resemble the Python API.

Current dispatcher `ops/fused_moe/token_dispatcher.py:399–439` masks probabilities by `expert_map[topk_ids] != -1`, uses EP rank times local expert count for the half-open active range, requests count mode1, and carries returned indices into combine. `device_op.py:969` applies `torch.abs` before native unpermute. Need the actual frozen expert map and ordered EP ranks, not merely EP size.

Installed CANN source root is `/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/` (inside container). In `moe_init_routing_v3/moe_v3_row_idx_gather.h:185`, `CopyOut` writes packed ordinal at the original flattened route slot, establishing the **gather** direction for this template. `moe_v3_full_load_base.h:CopyOutDefaultGatherIdx` initializes indices to -1; this alone does not prove every selected template's excluded-slot behavior.

Installed `moe_token_unpermute/moe_token_unpermute.h:231–273` reads successive top-k indices and probabilities, gathers each selected packed token and accumulates into the corresponding output token. A zero probability avoids `CopyTokenIn` in the inspected template. Thus the supported intended map is `out[t] = sum_k p[t,k] * packed[abs(g[t*K+k])]`, with excluded zero-weight slots skipped. Need evidence for the actually selected tiling/kernel and all valid local slots; no assumption that `abs(-1)=1` itself is a valid route.

Current official [unpermute contract](https://github.com/Ascend/op-plugin/blob/1fd5cb5621bfff086357c7818a726ae2a888fedb/docs/zh/custom_APIs/torch_npu/torch_npu-npu_moe_token_unpermute.md) describes gather indices and token-major weighted sum, but its public index constraints exclude negative/repeated indices. This document was absent at the wheel's submodule path; it supports intended API semantics, not historical ABI identity. Old MindSpeed scatter/index_copy examples and WithEP docs are different APIs and cannot replace this proof.

### 2. Actual TND sparse attention query rows

This runtime calls `_C_ascend.npu_sparse_attn_sharedkv`, not generic `torch_npu.npu_sparse_flash_attention`. `device_op.py:693` selects it; `dsa_cp.py:1578–1642` supplies TND/PA_ND and local query prefix metadata. C++ binding `csrc/torch_binding.cpp:1096–1131` calls `aclnnSparseAttnSharedkv`.

Current custom `csrc/attention/sparse_attn_sharedkv/README.md` specifies attention output from Q and cumulative TND query prefixes. More directly, both `op_kernel/arch32/sparse_attn_sharedkv_{swa,scfa}_kernel.h` assign `info.attenOutOffset = tensorACoreOffset` (SWA594, SCFA575). The offset uses query prefix and local query/head index. Corresponding block-vector code writes output to that offset (SWA607, SCFA842). This supports output row t corresponding to input query row t for inspected templates, even when metadata partitions work across cores.

Missing: runtime custom-library/object SHA and selected tiling/template binding; validity of captured `cu_seqlens_q`/CP binding and metadata; all selected branches. Source-side correct output offsets do not show the metadata was generated for the current query batch. A generic same-shaped output check does not close this.

### 3. DSA A2A

`dsa_cp.py:1645–1675` transforms local `[L,P*H,D]` into send `[P*L,H,D]` by view/permute/contiguous, then equal-split `all_to_all_single`. Under group-rank-ordered send/receive semantics, destination group rank d receives `[source rank r, local row u, head d*H+h, feature]`; output row `r*L+u` keeps source label `(r,u)`. This is the documented collective model ([PyTorch2.10](https://docs.pytorch.org/docs/2.10/distributed.html#torch.distributed.all_to_all_single)), not an assertion about the backend binary.

Pure-Python oracle `cpu_mapping_check.json` passed 4608 labeled scalars for P8/L12/H2/D3. This closes only composition algebra. At96 rows, rank1/local0 is global12, hence request slot1/offset4 for eight verification rows per request; rank boundaries do not generally equal request boundaries.

Missing: frozen `skip_all_to_all`, ordered actual `device_group`, equal local L, native HCCL implementation and graph binding. If skip is true, this mapping does not apply. Rotary modifies features and needs its own row-preserving native contract.

### 4. Fused mm-reduce-scatter

Current `ops/linear_op.py:358–412` pads conditionally, obtains TP HCCL communicator, and invokes `DeviceOperator.npu_mm_reduce_scatter_base`; base wrapper forwards `reduce_op=sum`, scales/output dtype and comm mode. DSA o_proj/wo_b uses a separate padding predicate. Applicability must be recorded per actual layer branch.

Official [Ascend operator test](https://github.com/Ascend/op-plugin/blob/1fd5cb5621bfff086357c7818a726ae2a888fedb/test/test_custom_ops/test_mm_reduce_scatter_base.py) independently constructs rank-local matmuls, sums them, then returns consecutive dimension0 blocks indexed by group rank; its quantized reference preserves the same slicing after scale application. This directly supports intended map `out_r[u] = sum_s matmul_s[r*(M/P)+u]`, with no row permutation inside the block. It is a current upstream test, **not a passing test of this installation**. The [26.1 API](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu-npu_mm_reduce_scatter_base.md) is supplementary; it is not exact CANN9.1 build provenance.

Missing: version-matched fused kernel/backend identity, group rank map, quantized scale semantics and selected padding/unpadding chain. A standalone unquantized CPU matmul is insufficient to certify W8A8 fused execution.

## Minimum falsifiable next experiment (design only)

1. Before NPU: freeze wheel/CANN/custom-library checksums, actual route attributes and active range from the intended branch. Use the exact installed Python defaults; record schema without guessing gather/scatter mode. Stop if op-plugin provenance cannot be reconciled or topology differs.
2. One isolated device, two operator calls per case: init-routing → capture integer index/count/scale outputs; construct deterministic packed floating payloads from independently labeled `(token,k,expert)` routes; feed unpermute with the current wrapper's `abs` and masked probabilities. No model weights or service. Use actual-supported H/dtype and quant_mode1 (production H if minimum legal H is unknown), four tokens/top-k2 first, then actual top-k6 geometry. Include routes inside/below/above the EP interval, a token with zero local routes, ties in expert IDs across distinct tokens, and first packed row0. Nonuniform exact-representable probabilities distinguish k-order errors.
3. Fail closed unless every valid `g[t*K+k]` points to a packed row whose returned quantized data/scale reconstructs that token, local histogram equals counts exactly, no valid route aliases a different token, and output agrees with independent weighted labeled reference. Do not compare uninitialized packed tail bytes. Excluded slots must not contribute even if `abs` aliases a valid row. Full-range and partial-range controls must both pass.
4. This closes the first native EP-index pairing at tested geometry/tiling only. Repeat at exact96×top-k6 with actual expert count/range if the small case changes template. For graph certificate, follow with capture/replay of the same pair under pointer reuse and two distinct sentinel generations; a passing eager result alone is insufficient.
5. Next uncertainty after that is loaded custom-attention query-order/metadata binding: use exact supported TND/PA_ND shape, unequal per-request prefixes and known KV reference, compare each query row against independent CPU attention. Then a tiny distributed rank/row sentinel case can verify HCCL A2A and fusedRS group ordering. CPU emulation alone cannot close either backend contract.

## Gate and stop point

Proceed with the isolated MoE semantic experiment only if it serves the remaining identity question and resources are explicitly scheduled. Do not start another 43-layer frozen generation merely to rediscover this ABI gap. Until native checks plus Run424 frozen graph/branch/request metadata and Run420 external retention ledger are joined, use `ROW_IDENTITY_CONDITIONAL_NATIVE`. No finite Bound numerator or retained-row claim is promoted by this review.

Source inventory: `source_sha256.json` binds the seven local wrapper/custom-kernel files inspected here. It does not certify loaded shared objects. All review and CPU-oracle commands completed with exit0; one exploratory historical upstream path returned404, explicitly treated as unavailable above.

Correction after Run432 fixture inspection: actual router shape is96×6; eight denotes verification rows per request, not routed expert top-k.
