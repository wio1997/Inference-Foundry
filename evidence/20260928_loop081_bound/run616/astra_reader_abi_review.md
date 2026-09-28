# Run616 — DSpark SWA eligible-reader source contract

**SCOPED PASS for a conditional source-derived logical reader contract; actual Run610/new-packet readers remain unproved.** Read-only review of current installed C++ sources and prior Run575 findings. No compilation, service, device experiment or algorithm change.

The source is sufficient to write a strict CPU oracle for the supported SWA/TND/PA_ND branch given exact live arguments, metadata and matched implementation. It is insufficient by itself to certify that these were the deployed bytes/tiling or that an eligible row physically reached HBM. Keep these three claims separate.

## Binding and branch conditions

torch_binding.cpp:1096–1128 forwards tensor arguments and windows to aclnnSparseAttnSharedkv and explicitly passes ori_kv.stride(0) as ori_kv_stride. Thus the observer needs the actual view data pointer/storage offset AND block stride; shape-derived contiguous stride is insufficient.

The SWA template is selected by ori_kv present and cmp_kv/cmp_sparse_indices absent (tiling.cpp:298–313), not by cmp_ratio alone. For the current DSpark compress_ratio<=1 caller, verify these actual optional arguments rather than inferring SWA from the Python compression field. Admission requires q TND, kv PA_ND, ori_mask_mode4, matching supported BF16/FP16 q/kv/output, one KV head, valid heads/dimensions, nonnegative windows and actual block geometry. This review's simple oracle additionally restricts each selected request to K>=Q>0, valid monotone query prefixes and in-bounds tables/slots. Other cases need their own source-exact handling, not clamping guesses.

The local README is stale: it says ori_sparse_indices is ineffective and cmp_ratio only4/128, while this source explicitly implements sparse original-KV lookup and SWA without cmp_kv. Do not use those README restrictions as the oracle.

## Logical row formulas

Let b be request, P=cu_seqlens_q, Q=P[b+1]-P[b], j in [0,Q), K=seqused_kv[b], L/R=ori_win_left/right. q global token index is t=P[b]+j. The TND/PA_ND init path uses cu_seqlens_q and seqused_kv (swa_kernel.h:324–366,440–452). The kernel sets mBaseSize=gSize (around229), so the s1StartIdx/s1EndIdx used by the window are this individual query j, not a multi-query approximation.

**No ori_sparse_indices:** source lines730–735 derive the inclusive interval
lo=max(0,K-Q+j-L), hi=min(K-1,K-Q+j+R).
Eligible logical KV positions are lo..hi if hi>=lo. For block size B, position k maps via physical block p=ori_block_table[b,k//B], row offset u=k%B. With the admitted single KV head and head dimension D, the element offset from the actual ori_kv tensor pointer is p*ori_kv_stride+u*D+d for d in [0,D). Convert to storage-relative bytes using the tensor view offset and element size. Do not assume physical block p belongs only to this request; shared/aliased blocks are possible and must enter the writer map.

**ori_sparse_indices present:** this is a distinct branch. Tiling checks require PA_ND/TND, int32 indices [T,Nkv,width], correct head count and positive128-aligned bounded width (tiling.cpp:1107–1143). GetOriSparseActualSeqLen (swa_kernel.h:373–391) scans the selected query/head row up to min(width,K), stopping at the FIRST negative value. Eligible entries are exactly that prefix. Preserve order and duplicate multiplicity for semantic checking; a set may be used only for storage-hazard membership. Later nonnegative entries after a negative sentinel are not eligible.

Each entry is an already-physical slot s. DataCopyPABySlots (common.h:174–198) maps p=s//B,u=s%B directly to the cache using the same block-stride formula. **Do not apply ori_block_table again and do not intersect this branch with the continuous L/R window.** The process branch replaces oriMaskLeft/right with 0..sparseLen-1 (swa_kernel.h:724–728). Windows remain relevant to producer/metadata coverage, but not as a second filter on these supplied slot IDs. Reject out-of-cache positive indices; the source prefix scan does not itself certify their bounds.

## Metadata and tiling coverage remain required

The above is the desired/source logical eligible domain. The actual kernel first reads metadata FA_CORE_ENABLE and core bN2/M/S2 start/end partitions (swa_kernel.h:530–540,664–669), which can omit or split work. The preallocated metadata tensor is1024 int32 values (4KiB; binding:1159 onward). Snapshot its content epoch as well as the arguments, not only its pointer. Verify enabled-core partitions cover each admitted query and needed S2 chunks exactly as intended; stale or incompatible metadata prevents promotion to an executed-reader witness.

The AICPU metadata generator is a separate implementation and can prioritize seqused_q when supplied, whereas the TND attention kernel derives Q from cu_seqlens_q. Record generator argument lineage and check they describe the same Q/K, not merely equal tensor shapes. ASCEND910 metadata uses mBaseSize=groupSize and S2 base512 (metadata_aicpu.cpp:281–287). Bind selected SoC/template/tiling fields and metadata generation; this source fact does not prove a captured invocation used that branch.

## Physical reads and binary identity

Cube paths invoke DataCopyPA/DataCopyPABySlots for both score and value matmuls (swa_block_cube.h:375–418,708–750). Source therefore describes planned GM-copy addresses under the selected task. It does not reveal actual first-load timing, cache hits, HBM transactions, burst/alignment overfetch, repeated traffic or inter-kernel visibility. DataCopyPABySlots even defensively substitutes slot0 for a negative passed to it; do not interpret that fallback as logical inclusion of invalid rows. An ABI-eligible row also need not have nonzero numerical influence after finite-precision softmax.

To bind this oracle to a live packet, preserve: Python caller and compiled torch extension identity; resolved aclnn op-api and tiling library loaded paths/build IDs/SHA; operator registration and source-to-object build provenance including flags/toolchain/target; selected AICore object SHA, function/template key and runtime kernel load/generation; matching AICPU metadata object and invocation arguments; actual dynamic tensors/format/strides/metadata. The permitted serving process's loaded-image evidence must be recorded in that acquisition, not inferred from files currently present on disk.

Run575 only joins an older FULL96 TARGET Graph task name and key to packaged objects, with FD0/TND/PA_ND SWA/CFA/SCFA. It explicitly does not prove device-loaded bytes or dynamic arguments and cannot be transferred to Run613's eager DSpark stream47 calls. Their SparseAttnSharedkv name and exact flow establish ownership, not the selected binary or tiling.

## Next source-only/preflight action

Implement the two formulas above as separate CPU oracle branches with exact integer offsets. Test nontrivial block permutation, non-contiguous block stride, tensor view offset, Q/K asymmetry, window edges, sparse sentinel in the middle, duplicate physical slots, out-of-bounds slot, branch optional-argument mismatch, metadata coverage gap and stale generator Q/K. These are logical fixtures; no device experiment is needed to implement/review them. Attach resulting conditional eligibility to typed writer-generation evidence. Keep executed-reader ownership UNKNOWN until metadata and loaded-implementation joins close. No strict Bound endpoint, compulsory byte count or Current-to-limit gap changes; formal Current571.681 tok/s.

## Source identities

torch_binding.cpp
SHA256 20e1532b72600794ee3b5a39dbc8d6bd2eb89193e961664f5f47282a3028f2ce

attention/sparse_attn_sharedkv/README.md
SHA256 42a1927c69e67ac8e02883b0a592ec432bd32302d18cbb71c164ac23a46b2623

attention/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv_tiling.cpp
SHA256 98fb6d201fd7657fe44c4ebe9a2445af377cb4b5db47861f0b52db2a8b8dacf1

attention/sparse_attn_sharedkv/op_kernel/sparse_attn_sharedkv.cpp
SHA256 846a1b620d3ea49990624556cc0259f85970fd37eafb7dcc0812be8f1c90e48b

attention/sparse_attn_sharedkv/op_kernel/sparse_attn_sharedkv_template_tiling_key.h
SHA256 c23764162a3bfb1cbd9ce9ccd6c3dd52719c514e2728f33aec2f39e5ed47a5a8

attention/sparse_attn_sharedkv/op_kernel/sparse_attn_sharedkv_common.h
SHA256 044b59865853d8c5a69083e81a39939be527486d016fa57dcf4b3e47c828c8b4

attention/sparse_attn_sharedkv/op_kernel/sparse_attn_sharedkv_metadata.h
SHA256 c134bc897936c0f57f0221b927cecde4f3b15e379f5c1d4c0760f11de85f0da7

attention/sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_swa_kernel.h
SHA256 fc6cff0a2e18ba4c124aeac7e2e9ab2746cc0a5954848e7e556778a5e898c7a0

attention/sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_swa_block_cube.h
SHA256 40c541ed870552bac0fdea0578da975ba07560f4ece44596c767192ce5eced48

attention/sparse_attn_sharedkv/op_kernel/arch32/sparse_attn_sharedkv_swa_block_vector.h
SHA256 50d9292af44269875221b530f689a39ebc5d3b863cfc76afc8ed2e5719b93a01

attention/sparse_attn_sharedkv_metadata/op_kernel_aicpu/sparse_attn_sharedkv_metadata_aicpu.cpp
SHA256 ba801a3c9904993e01efda8fc48117bfca6c1906b571ff0811238e4f88687b4f

