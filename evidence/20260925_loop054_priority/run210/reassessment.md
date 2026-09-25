# Run210: current maximum-gap reassessment

Offline reconciliation, no service, NPU experiment or borrowed source edit. Sol reviewed Run116, Run145-152, Run195 and Run208-209, computed a Run107 kernel-family census, and used a configured Astra High read-only independent review as advisory input. Actual reviewer backend model ID is not independently verifiable.

Run116 showed about9.966ms profiled GMM sum and10.248ms communication time not overlapped by compute in the median target window. Run145/192 show the large first reduce-scatter span is mostly early-rank arrival wait with aligned ends. Run194 shows that same profiled proposer Host scope was8.62x a separate low-overhead run. Run195 has steady median8-rank target-duration spread0.090ms, though it does not establish absolute start-time alignment. The remaining collectives total about5.2ms in profile and Run152 adjacent FP32 gather fragments total only0.257ms/window.

Run148/150 one-card product-shape GMM counters show reads close to packed active weights, about1.055x W1 and1.075x W2 at roughly1118/995GB/s effective in-kernel bandwidth. This is not a hardware peak proof, but there is no identified legal GMM replacement or obvious repeated-weight traffic. Therefore neither Run116 gross sum is a credible independently removable10ms product Gap.

Run107 top-family median kernel sums include grouped matmul Swiglu6.375ms, grouped matmul3.582ms, QMM4.822ms, Compressor3.365ms, Scatter2.333ms, Indexer1.746ms and sparse attention1.619ms. These are profiled exposure only. Run151/197 mapped required compressor/indexer products and small scatter gross return; a new target implementation needs a specific safe mechanism rather than another census.

Astra High advisory identifies a narrower, untested Host mechanism: one real-weight prefill MoE custom-op subgraph. It avoids DSA/KV capture ownership but still contains dynamic routing, collective order, multistream events, context/layer-index and possibly mutable counters. Sol accepts this as the next bounded feasibility discriminator, not a throughput claim. First do source-closure and a single-layer/single-shape same-state capture test on all8 ranks; stop if the boundary cannot be bounded. Measure actual Host/device/latest-rank savings including refresh overhead before multiplying across43 layers or shapes. No formal E2E until correctness and stage effect.

Independent review: evidence/20260925_loop054_priority/independent_arch_review.md. Family census: family_census.json. Next: pivot Loop054 and open Loop055 for prefill MoE-only replay feasibility.
