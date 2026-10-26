# Astra High independent Run646 verdict

KEEP as conditional Host-operation-start timing evidence; not an available saving or Framework/Product Bound. Independent full16-window raw SHA, unique torch_to_npu flow/CPU-op exact start, duration partition and within-stream nonoverlap all pass.

Rank7 stream47 adjacent-task gaps16.353/14.012ms; next CPU op not begun at prior native end10.047/8.323ms. This is stronger than rank-local task-free coverage but depends on Host/native clock mapping and flow semantics. A sensitivity test that deducts a hypothetical20µs per positive fragment still leaves7.740/6.176ms; this20µs is not a certified error bound.

Next CPU-op labels do not establish cause or reveal primitive cost. `aclnnInplaceCopy` labels3.389/3.318ms, wait_event1.426/1.390ms, record_event0.782/0.433ms, aclnnMatmul0.666/0.692ms. These may be preceded by necessary CPU work, other stream issue, dependency/queue waiting or profiler overhead. Rank4 zero late-start time does not prove that timing is legal for rank7. Next offline join predecessor CPU operation, parent scope and enqueue/dequeue; only then select a bounded measurement of missing producer-ready/actual enqueue. Reviewer performed no NPU run or source edit.
