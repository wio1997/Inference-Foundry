# Loop036 metadata scalar source audit

Frozen path: `runtime/target_metadata.py::FixedTargetMetadataUpdater.update`
passes `seqused_kv=self.local_seq_lens` to SparseAttnSharedkvMetadata
with `layout_kv="PA_ND"`, and `actual_seq_lengths_key=self.local_seq_lens.clone()`
to VllmQuantLightningIndexerMetadata with `layout_key="PA_BSND"`.

Installed vllm-ascend 0.26 AICPU source:
- `csrc/attention/sparse_attn_sharedkv_metadata/op_kernel_aicpu/sparse_attn_sharedkv_metadata_aicpu.cpp`: `kvSeqSize_` is read from `max_seqlen_kv` at line 61. Its only other use is `GetS2SeqSize` at line 327, reached only when `seqUsedKv_` is absent and layout is not TND. This runtime passes the length tensor, so `GetS2SeqSize` reads its element per request.
- `csrc/attention/vllm_quant_lightning_indexer_metadata/op_kernel_aicpu/vllm_quant_lightning_indexer_metadata_aicpu.cpp`: `maxSeqlenK_` is read at line 55, checked nonnegative at 95, and used as a fallback in `GetS2SeqSize` at line 289 only when `actSeqLenKey_` is absent. This runtime passes the length tensor.
- Both op API wrappers forward these scalar attributes unchanged to AICPU. Recursive search under both op directories found no other consumers.

Candidate: set both scalar attributes to 1 in this fixed path, while retaining actual per-request length tensors. The value 1 is valid and its fallback branch is unreachable under the frozen binding. The opt-in shadow checks the first 1024 metadata elements for each distinct SAS ratio and QLI at every continuous update on all eight ranks against the existing dynamic `max().item()` path. Profiling uses the static path without shadow.
