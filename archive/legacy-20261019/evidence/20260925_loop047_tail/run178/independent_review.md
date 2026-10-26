# Run178 independent active-slot compaction review

Configured reviewer: GPT-6 Astra High; actual serving model ID is not independently exposed by the collaboration execution record. This is a read-only architecture review, not an executed runtime experiment.

Evidence reviewed: Run175 eight-rank active-slot census; Run176 c12 ABI source audit; Run177 same-service Stock target-size event slope; Run107 target compute/communication trace; Run150 GMM weight traffic.

- Run155 measured tail has 122 partially active cycles. More than 1 s/cohort requires >8.20 ms average target reduction over those cycles before any packing, metadata, graph selection, KV remapping, or synchronization cost. Run175's 2.553 s ideal linear exposure is not an achievable bound.
- Run177 c12-to-c1 event endpoints differ by 14.041 ms, so the endpoint alone does not falsify a >1 s gain. Bucket-weighting the observed active-count histogram gives only 0.570 s/cohort under zero switching/packing cost. The sampled scope is Stock `_model_forward`, not the exact Extreme target stage, and the sequential order lacked a final c12 drift control.
- Run107 profiler target compute interval union was 39.932 ms and exposed communication union 11.547 ms with 1.292 ms overlap. These figures cannot be added. Run150 representative GMM2 read 66 MB for 61 MB packed weights at approximately 995 GB/s; shared weight traffic could limit batch-size benefit.
- Smaller target graphs are architecturally plausible with 8 tokens/request and TP8. Implementation crosses graph and metadata banks, stable per-size addresses, logical-to-physical KV mapping, and live-row gather. Hidden, aux, and logits must scatter to canonical c12 slots. Preserve HCCL collective order and graph update-stream/event ordering. Parked rows require a valid dummy policy because DSpark consumes unmasked `acceptance.num_sampled`. The Host mirror commits pending async count copy before parking, so compaction can start only on subsequent cycles.
- Recommended gate: same-service explicit-sync c12 baseline repeated after smaller sizes, with 16-32 steady calls/rank, actual tokens and computed KV positions captured. Then Sol should assess whether the remaining practical headroom justifies the broad c12 ABI change. Do not claim formal E2E speedup from this diagnostic.

Sol accepts the drift-control measurement as the next bounded step. Correctness, formal E2E, and frozen-product contract remain required for any later KEEP verdict.
