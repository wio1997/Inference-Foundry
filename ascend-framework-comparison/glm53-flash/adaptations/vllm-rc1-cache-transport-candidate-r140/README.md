# Flash native cache/transport candidate after R140

This is source only. R127 bulk FP32/IEEE KDA math passes its fixed component gates; the native recurrent/decode/MTP rollback path is unchanged. The per-head ReLU correction retains R080 math provenance without CPU score traces.

R140 actually restored602368B from native Memcache and transferred the same three BF16/FP32 logical ranges with exactly one strict SDMA read. All P-before/D-after/P-after byte hashes match. Native workers and MetaService nevertheless abort during finalization after documented close0/destroy returned. WHOLE COMPONENT FAILS; no fullmodel/optimizedDP/Graph/MTP/cache80K600/performance qualification or accepted baseline. R141 only diagnoses that exit boundary.

The batch-registration adapter delegates all original logical vectors to the existing MF1.2 batch API once, letting the SDK coalesce shared2MiB pages. Payload vectors remain unchanged and no retry/RDMA fallback exists. R140 demonstrates the typed-data path only; the11-overlay bundle has not been loaded as a whole model.

The native PoolWorker candidate uses the already-supported Memcache metadata client API only when backendmemcache/kv_consumer/loadfalse/putfalse/whole-segment. It skips only that consumer’s MMC payload-buffer registration. Producer, enabled cache payload directions, other backends, native cache-key/block/address/state math and strict SDMA remain intact. Fullmodel gate pending.

The graph observer writes scalar metadata only after the actual native replay branch, enabled solely with FLASH_NATIVE_GRAPH_OBSERVER1 for qualification. Correlate selectedDP/same-node actualHTTP requestinterval/is_draftfalse; startupreplay is insufficient. Formal observer0 and transportbyteaudit0, after an actual audit0 functional gate. Never time CPU tensor hashes/traces as performance.

R139 actual Flash tokenizer154856 (model paddedembedding154880) generated and independently verified452 exact80000-ID sequences:74400prefix+four first-token-divergent5600suffixes atC1/16/32/64. This is file qualification only: no actual cachewarmup/modelrequests/600output or formal requests. Raw ID files stay remote/private and are excluded here. Use flash_warm_prefix_dataset_v2.py; v1’s154880 tokenizer-length assumption is historical and rejected.

Fixed image090eeee6…, MF/MMC1.2, CANN9.1.0/driver25.5.2; source-only official P DP2TP8 and D DP16TP1 GraphD5/nativehybridalign/whole-segment profile in config/vllm-rc1/native-memcache-performance-candidate. ProducerBM16worldsize16/56GiB each requested, consumersmetadata-only0BM. Actual capacity/C64 unknown. No SDK/image/driver upgrade or native math gate relaxation.
