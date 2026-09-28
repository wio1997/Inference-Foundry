# Run617 independent CPU oracle review

**Formula/fixture SCOPED PASS; revise the input-contract wording and sparse-width gate before promoting this as an ABI admission oracle. NOT LIVE-READY.**

Reviewed script SHA25689d02235cbb7177319ccf742a017eecc81497d6e99238fc4b1a77323e1aad205 and output SHA2564a89f21c379d484d5c22bf203fe078b4112950d7d39394634e91543faf10ba65. I independently executed the script's CPU logic after AST-removing only final filesystem-output/print calls; no device/service operation and no artifact overwrite. The reconstructed result equals the saved JSON exactly:15 fixtures,7 passing arithmetic cases and8 expected rejections. I also checked the formulas against the pinned C++ branches reviewed in Run616.

Dense mapping and sparse prefix/sentinel behavior are correct under their declared single-query K>=Q>0 source branch. Sparse duplicate multiplicity is preserved, and sparse slots bypass block-table remapping/window filtering. The noncontiguous block-stride plus view-offset byte fixture correctly gives[620,624). The actual-W0, physical-HBM and both strict floor fields remain null. No execution-domain, generation, loaded-binary or performance claim is proved.

## Concrete corrections

1. Installed tiling.h defines SPARSE_LIMIT=2048; tiling.cpp requires width<=SPARSE_LIMIT. The helper checks only width>=128 and128 alignment. I supplied width2176 with first slot9 and observed[9] rather than rejection. Add the maximum gate and a negative fixture, and pin tiling.h (SHA below). This is a missing supported-ABI constraint, not an error in the prefix arithmetic.

2. The sparse path does not use block-table VALUES for addressing, but the installed C++ PA_ND CheckParaExistence still requires ori_block_table to be PRESENT (tiling.cpp:1381 onward; also earlier shape initialization). Therefore the block_table=None fixture is valid only as a pure mapping-unit demonstration. Rename/annotate it accordingly; never use it as evidence that a real missing-table invocation passes ABI admission. A future invocation validator must check presence even for sparse indices.

3. These fixtures are small mathematical examples, not hardware-supported shape tests: block_size4 and the address fixture's head_dim2 are deliberately below the native shape contract. Clearly separate this generic row/address helper from a complete live ABI/schema validator. The latter must validate real block/head geometry, dtype/int32 payloads, shape/strides/view bounds/NPU format, optional-argument-selected SWA mode, mask/layout, actual slot capacity, and integer range/overflow constraints before invoking the mapper. As a concrete type challenge, sparse_indices=[9.5,...] currently returns[9.5]; this must be rejected by the invocation schema, or by the helper if it is intended to perform that role.

The function docstring/module intro says “set” while returning an ordered list with duplicates; use “ordered eligible slot list” to avoid losing multiplicity in future consumers. Optional cache_slot_capacity is acceptable for symbolic arithmetic; actual storage admission must supply and validate capacity plus the byte bounds. Current conservative block-stride restriction is an admitted subset, not proof that every possible source view is supported.

## Scope that must stay conditional

The helper lacks actual query-prefix/request slicing, optional cmp_kv/cmp_sparse_indices, metadata coverage/generation, and loaded implementation arguments; it therefore cannot validate an entire invocation by itself. It cannot certify that eligible rows were actually scheduled: snapshot the1024-int32 metadata, bind generator Q/K semantics to consumer values, and prove enabled-core/M/S2 coverage. Preserve source-to-extension/op-api/tiling/AICPU/device-object identities and actual selection; Run575's older Target Graph linkage is not a substitute for current eager Draft loading.

A source eligibility result neither measures physical first read/HBM traffic nor proves numerical influence/mandatory work. Writer-generation and freshness admission remain separate. These findings do not invalidate the15 arithmetic fixtures; they prevent their reuse as an overly broad admission gate. Formal Current571.681tok/s and all strict Bound endpoints remain unchanged.

Additional source identity:
/data/wio/vllm_ascend_26/framework/vllm-ascend/csrc/attention/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv_tiling.h
SHA256 0a89fb2ab87434a914f1e0286f455ae0f18555f2848101a253f487fae0c774c1
