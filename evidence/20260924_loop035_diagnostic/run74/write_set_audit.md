# Run74 source-backed cache write-set audit

Date: 2026-09-24 UTC. Scope: fixed c12, TP8, one 96-token target cycle plus DSpark proposer. No new NPU experiment was run.

Run73 identified all 67 target physical cache tensors, grouped 16/8/8/8/17/10 for groups 0..5, with no unmapped tensor. The eight rank manifests agree structurally. This audit follows the active DeepSeek V4 DSA-CP source and Ascend C implementation.

| Cache group | Active source | Physical write address source | Snapshot requirement |
| --- | --- | --- | --- |
| 0 | layer 2 compressed attention and indexer cache, ratio 4 | compressor_metadata output compress_slot_mapping and indexer_slot_mapping; generated from request start_pos, query_start_loc, and compressed block table | Capture every valid generated compressed slot for every group-0 cache tensor; inspect both indexer and attention streams |
| 1 | layer 3 compressed attention, ratio 128 | compressor_metadata output compress_slot_mapping | Capture every valid generated compressed slot for every group-1 cache tensor |
| 2, 3 | SWA context/target cache and DSpark draft cache | req_metadata.slot_mapping for target SWA; draft context/query slot mapping for proposal | Capture target and future proposal slots, including the native DSpark group-2 refresh |
| 4 | layer 2 attention and indexer compressor state, block size 2 | Ascend C WriteToCacheState uses state_block_table[batch, uncompressed_position / block_size], start_pos, and valid query rows | Capture every state page intersecting each request interval [start_pos, start_pos + query_len); include both state-cache aliases |
| 5 | layer 3 compressor state, block size 8 | Same WriteToCacheState rule | Capture every state page intersecting each request interval |

Source anchors:
- vllm_ascend/attention/context_parallel/dsa_cp.py: SWA scatter ~1522, compressed attention scatter ~1574, indexer scatter ~1727; compressor passes state block table and start position at ~1553 and ~1699.
- vllm_ascend/attention/dsa_v1.py:69-114: cached torch.ops._C_ascend.compressor_metadata call.
- .../ascendc/compressor_metadata/compressor_metadata.h:230-370: compressed position selects block-table page and output slot.
- .../ascendc/compressor/arch32/compressor_block_vec_perf.h:830-880: WriteToCacheState computes uncompressed page address; SaveState provides start and end positions.

The common group slot mappings used by RuntimeAssets.snapshot_slots do not certify these writes for groups 0, 1, 4, or 5. strict=True currently proves only that the supplied candidate indices are in bounds and nonempty. A completed Stock A / Product B / Stock C control must construct per-cache candidates from the actual active metadata and physical group source, count all cache tensors, reject invalid generated slots only when proven padding, and restore all mutated metadata aliases. Four large metadata tensors omitted by Run66 must be identified and either copied or shown immutable across all three legs. The A/C replay noise remains the comparison floor.

No semantic equivalence, acceptance recovery, or E2E performance conclusion follows from this source audit.

## Superseded by Run76

The source-level write-address families remain valid, but the Run73 per-cache
group labels came from a positional zip between two differently ordered lists.
Run76 proves 31 of 67 labels exclude the actual alias groups, and 42 cache
views alias layers in multiple groups. Do not apply the table above to
individual physical cache tensors using the Run73 group field. Build each
physical page candidate from the full layer-alias set and its group block table.
