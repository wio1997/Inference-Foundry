# Run339 — CANN 9.1.0 HCCL Test, TP8 actual residual-prefill payloads

## Environment and methodology

Actual server: eight Ascend 910B3, driver `26.0.rc1`, CANN Toolkit `9.1.0` (`/usr/local/Ascend/cann-9.1.0`), Open MPI 4.1.2 installed in the dedicated test container for this run. Build uses the installed CANN HCCL Test source unchanged; its Makefile needed an explicit `-lmpi_cxx` link for Ubuntu Open MPI. Preserve build failure and corrected build logs plus binary SHA. Use the service's HCCL settings: `HCCL_BUFFSIZE=1024`, `TASK_QUEUE_ENABLE=1`, `HCCL_OP_EXPANSION_MODE=AIV`, `HCCL_INTRA_ROCE_ENABLE=1`, `HCCL_RDMA_CONNECT_TIMEOUT=17`. Official CANN 9.1.0 HCCL Test supports bfp16 on the A2 family and `-t 1` device-only timing when AIV mode and buffer >100 MB. Run 5 warmups and 30 measured iterations, result check enabled, all 8 ranks. Do not run beside serving.

The product API input-byte census is from Run337's 221 wrapped calls/rank, with missing43 DSA AllGather calls identified separately from source/Run333. HCCL Test `data_size` is **not always API input bytes**: installed source `hccl_allgather_rootinfo_test.cc::init_malloc_Ksize_by_data()` divides count by rank_size, so its `-b` denotes complete output bytes; ReduceScatter divides count by rank_size but uses `data_size` as total input; AllToAll uses `data_size` as input/output. Check output-reported size after alignment.

| Case | Product API input bytes/rank | HCCL Test `-b = -e` | dtype | Test family |
|---|---:|---:|---|---|
| hidden/DSA AllGather | 90,112 | 720,896 | bfp16 | all_gather_test |
| router-logits AllGather | 11,264 | 90,112 | fp32 | all_gather_test |
| extra BF16 AllGather | 360,448 | 2,883,584 | bfp16 | all_gather_test |
| MoE/DSA ReduceScatter | 720,896 | 720,896 | bfp16 | reduce_scatter_test |
| AllToAll | 720,896 | 720,896 | bfp16 | alltoall_test |

Do **not** pass `-i 0`: the installed source loops at fixed size indefinitely. With `-b = -e` and no `-i`, it chooses step 1 and executes one finite size. The first pilot used `-i 0` and was terminated after producing valid samples; it also used `-b 90112 bfp16` for AllGather, which corresponds to only 11,264 input bytes/rank and is not the product BF16 hidden payload. The pilot is a tool-behavior check, excluded from resource numbers.

## Interpretation

This estimates isolated no-arrival-wait HCCL Test service for each API family under the same dtype, 8-rank topology and HCCL environment. It does not preserve producer/consumer dependencies, model compute contention, graph scheduling, or the exact 264-op chain. HCCL Test `alg_bandwidth` is its own convention; retain measured microseconds and byte mapping rather than treating it as physical link bytes. Do not sum isolated medians into a Product critical path or call them a strict hardware lower bound. Use as attainable service calibration in a resource-constrained DAG only after confirming algorithm/runtime path comparability and measuring a product-like chain.
