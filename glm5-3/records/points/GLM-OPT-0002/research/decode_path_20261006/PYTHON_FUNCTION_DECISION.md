# Run257 source context and remaining attribution

Execution/golden/stock ownership passed; the frozen source-stack diagnostic is **INCONCLUSIVE**. No repeat model request is permitted by its frozen budget, and none is queued. [Reducer and original profiles](../../runs/GLM-RUN-0257/source_functions_reduced.json) preserve the distinction.

Only one2334-prompt/8-token request was issued on P249/D256 stock. D13/D15 readers used py-spy0.4.2, nonblocking100Hz, no native/locals/subprocess collection. MainThread139/137 samples include132/122 ModelRunner ancestry; readers report46/37 errors and consumed2.18/2.23CPU seconds during approximately2.3s. Counts above100 do not override reader errors or prove atomic frames. Profile weights1.39/1.37s are not the actual request window. No quantitative source-function CPU/wall partition follows.

Corroborated locations narrow source inspection: W8A8DynamicFusedMoEMethod.apply→MoECommMethod.fused_experts→TokenDispatcher.token_dispatch234 / token_combine328→torch._ops.__call__1209. These are native MC2 calls inside the Python opaque handler, not time removable by deleting the outer MoERunner custom op. Prepare286/302/303, list all_gather, SFA KV/cache and DCP sparse/remap locations also appear. WorkerAsyncOutputCopy.event.synchronize has a real output-copy dependency; its samples do not establish a redundant barrier.

Original P/D roots,32workers/start ticks, health200/idle and allD16 original-native mappings are unchanged before/after. Native digest83fb9a0e… is global library identity, not a fabricated per-row digest. Reader setup59.999026ms occurred after P export validation and before D; the reported P wrapper wall includes this setup. TPOT257.810678ms is an observation under two observers, not a performance comparison.

Accept source locations only after matching actual source/consumer behavior. Reject sampled weight as CPU time, nonblocking/truncated frames as exact stacks, PythonKernelHolder ancestry as wrapper cost, and repeat sampling merely to improve coverage. Four control-layer exact entry/exit times remain unknown; existing device boundary gaps do not supply them. Continue saved raw/source attribution.
