# Run628 Astra High negative formatted-slot audit

**Verdict: source-conditional no destination-row write is supported for [-1,31]; actual eager loaded-path certification remains UNKNOWN. Keep Run627's real-packet sentinel UNKNOWN.** Read-only source/artifact audit, no service/NPU experiment or candidate implementation.

## Source path and conditional derivation

1. Non-A5 `device_op.py::format_dsa_slot_mapping`844–847 stacks flat slot//block_size and flat slot%block_size. For block32, flat-1 becomes **[-1,31]**, not[-1,-1]. Non-A5 scatter710–712 invokes `_C_ascend.npu_scatter_nd_update_sk`.
2. `torch_binding.cpp:1950–1959` passes cache, actual indices, updates and actual cache strides to ACLNN. The installed ACLNN source checks types, makes indices/updates contiguous, sets var storage shape to view shape and launches the in-place AiCore op. No negative-slot special case occurs at this wrapper.
3. Tiling `ParseScatterShapes`463ff takes indexDim from the final indices dimension; `Tiling4LinearIndex`80ff derives coefficients from the first indexDim cache dimensions. Conditional on actual cache shape[34090,32,1,512] and actual indices[N,2], coefficients are **[32,1]**, capacity34090*32=1,090,880 indexed rows and scatterLength512 elements. Thus (-1)*32+31=-1, with no integer overflow for this sentinel.
4. Tiling chooses sorted when total indexed rows <=2^24-1. This geometry satisfies that test. Expected key is **11 for int32 indices,21 for int64 indices cast to int32**. These are conditional source predictions, not observed Run621 eager tiling keys. The actual invocation shape/dtype still needs its own binding; Run621 captured pre-refresh descriptors.
5. Sorted kernel source forms a descending index sequence. `scatter_nd_update_linear_index.h::ComputeForSort` explicitly casts int indices to float and back; -1 is exactly representable (not a bit reinterpretation NaN). `scatter_nd_update.h::UpdateSearchParam` restricts work to [start_,end_), via first value<end and last value>=start. Tiling/Core distribution gives nonnegative start, so -1 is excluded before CopyUpdateIn/CopyOut. View-stride destination address calculation only occurs for admitted indices.
6. Also inspected alternative supported paths: keys10/20 use an explicit signed linearIndex>=start && linearIndex<end guard in no_sort.h around101; key30 computes signed int64 linear index and applies the analogous guard in large_index.h around99. Under well-formed nonnegative range tiling and the same [32,1] coefficients, this sentinel is excluded there too. This does not certify arbitrary shapes/int64 overflow ranges or invalid tilings.
7. Kernel entry undefines HIGH_PERFORMANCE and dispatches the five keys to these paths. Packaged config says implMode=high_performance; that label must not be confused with the disabled source HP shortcut or used as loaded-path proof.

**Exact safe statement:** if the actual invocation uses this pinned source-equivalent implementation, well-formed actual[N,2] indices, this block32 geometry and the described tiling/range semantics, [-1,31] contributes no write to a destination cache row. The indices may still be loaded, linearized, sorted and synchronized; no zero-traffic, zero-work or removable-duration claim follows.

Do not generalize this into “any negative pair is ignored.” The implementation linearizes components before range checks; other component combinations and overflow may behave differently. Preserve raw pairs and normalize only under an explicit admitted adapter.

## Packaged candidates versus actual loaded evidence

Installed ascend910b config contains BF16/int32 candidate `ScatterNdUpdateSk_3b70bd1dfff14ea76aa6465d526da019` and BF16/int64 candidate `ScatterNdUpdateSk_ea313760ba67acd5000faf33911395ef`. Their JSON kernelList includes keys11/10/21/20/30. Independently hashed object bytes match their JSON sha256 values:

- int32 object: `ce291acee1f71b7d74c7b01781663cbba82dc2ec96fca5af613ecba240db6702`; JSON SHA `088dcd1ed3a3632bfac683388e4e0f327f5c85c1de14b194220c1a6e68533c43`.
- int64 object: `8a5d15f5ee2379f2761c7195f8f60b090b2956121039be6fabc4c8854bc95c8e`; JSON SHA `81020299bb94b426d734977b399350c4d8c0e0e68ae49f25a72032ecd986ec92`.

This proves packaged candidate identities, not source→compiled object equivalence or actual process/device loading. Run613 exact flow records identify a generic native `aclnnScatterNdUpdateSk_ScatterNdUpdateSkAiCore_ScatterNdUpdateSk`, stream47/task identity, not object hash or tiling parameters. Run621 exposes owner-level cache geometry, not the actual eager scatter ABI or loaded image. Run575 concerns a separate Target sparse-attention Graph association and itself does not certify loaded bytes; it cannot authenticate this scatter path.

**Smallest missing join:** in the next already-justified invocation packet, bind actual formatted indices/cache shape/dtype/stride and native flow to the loaded scatter implementation/object and tiling key/data (indexDim2, coefficients32/1, nonnegative destination ranges), with an applicable source/build or direct implementation-semantic certificate. Do not run another service just to rediscover packaged files. Without this join, the CPU lineage reducer must not silently drop [-1,31] from a real acquisition.

## Reproducible source hashes

Paths relative to `/data/wio/vllm_ascend_26/framework/vllm-ascend`; `S` below abbreviates `csrc/moe/scatter_nd_update_sk`.

| File | SHA256 |
|---|---|
| vllm_ascend/device/device_op.py | 67dda24f28de847f9aee87db9714e5d2293ebb46309ad13152538cf63ca5f1b4 |
| csrc/torch_binding.cpp | 20e1532b72600794ee3b5a39dbc8d6bd2eb89193e961664f5f47282a3028f2ce |
| S/op_host/op_api/aclnn_scatter_nd_update_sk.cpp | 0b8829cf8f3001c77f8e9fb372a442b1de9a1795b4d9a077beec4f1dca9d0cb7 |
| S/op_host/scatter_nd_update_sk_tiling.cpp | e0bf550fee77f1fe0ca268db10feaef174ce419edd5b48746bd2d73b3506eca5 |
| S/op_kernel/scatter_nd_update_sk.cpp | 3dcd3a4bb6fe2f39f43a381c939159fcc8c9e8f52408592d80470773b7cf9cfb |
| S/op_kernel/arch22/scatter_nd_update_common.h | b33f796fdd033582dee545d58d4c8f2f6e010a58c043083db15a575596e27b01 |
| S/op_kernel/arch22/scatter_nd_update_linear_index.h | 4a921d48312aad0b7425d0147e767a34071226e2f67cd8a343d62ed01b445274 |
| S/op_kernel/arch22/scatter_nd_update.h | 69ad8af7fed7eb7d981ac35a13f96f8a86fd0fafe2113b3d3dc2c7b525c46f18 |
| S/op_kernel/arch22/scatter_nd_update_no_sort.h | 6814692d2dc752ba87122b4de0e41b2ae4b806f0a0f54a668de4cc6afbdebad1 |
| S/op_kernel/arch22/scatter_nd_update_large_index.h | 7ab65d2ef9f2dfd797dafa030482d8f5fa5dd62136500cd8a41bb3a53a87874b |

Installed config path `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/config/ascend910b/scatter_nd_update_sk.json`, SHA `7471bd3433825945bc9487b7137999f04a88a1c2d11be026b4d8f85c42920f47`. Candidate objects/JSONs reside in the sibling kernel/ascend910b/scatter_nd_update_sk directory.

No real sentinel packet is admitted here. Physical-first-read, fresh Resource work, compulsory HBM and strict Resource/Scheduling/Product endpoints remain null. Formal Current571.681 tok/s is unchanged.
