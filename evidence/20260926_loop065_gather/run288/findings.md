# Run288 invalid before H003

The fixed one-layer async-immediate service loaded the 8-rank model but failed during vLLM KV block-table initialization, before Extreme handoff, Graph capture, target-layer branch, client requests or H003 measurement. Ranks4/6 reported torch.OutOfMemoryError in torch_npu CachingHostAllocator aclrtMallocHostWithCfg (error 207001), called from CpuGpuBuffer pinned host allocation. Launcher never reached health. This does not test the patch or scheduling hypothesis.

The orphaned launcher was terminated, the existing stop script released all8 NPUs to baseline ~3.4GB/card, and SHA-guarded borrowed dsa_cp.py restored to 27cbbf92a02f16301aad2a35091e258b8459dc77744bd8294f2f11a4c126004e. No runtime reports or benchmark JSON exist. After cleanup Host MemAvailable was ~190Gi and Mlocked ~26Mi; cause of allocator failure remains unknown. Retry as Run289 under same frozen product config, with no performance comparison if startup fails again.
