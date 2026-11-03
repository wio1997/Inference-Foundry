# F08：已选 KV 写入与跨流区间

## 已有采样

复用 EP8 第 14 层 scope `737-20` 的同次 profiled 任务提取，见 `f08_ep8_pipeline_overlap.py` 和 `f08-ep8-pipeline-overlap.json`。输入 `diagnostic-20261029/target-layer-all-kernels-16rank.jsonl` SHA-256 为 `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`；只读该聚合和冻结源码，不读取请求、权重或张量元素，不运行新 NPU。提取之前已由同线程 layer scope、唯一 flow 和设备 task/CSV 对齐，方法及不完整性见 `F08-EP8-TENSOR-PROVENANCE.md`。scope 是主机提交归属，不是独立请求证明。

| 已选设备任务 | stream / task | 相对 stream 47 task 44167 的记录区间交集 |
| --- | --- | ---: |
| RmsNorm | 35 / 13274 | 33.8995 µs |
| InplacePartialRotaryMul | 35 / 13275 | 36.839 µs |
| ScatterNdUpdateSk | 35 / 13276 | 239.415 µs |

对照的 stream 47 / 44167 是形状为 `[1023,1280] × [1024,80,16,32]` 的 QuantBatchMatmul，记录区间 310.234 µs。它的形状与顺序和冻结源码 Q_b 的预期**相容**，但该设备任务未直接绑定具体 Python 模块、张量或事件；不能单靠这个确认它确为 Q_b。交集使用同 trace 设备事件的十进制原始时间和持续时间计算，是记录区间相交，并不证明相交区间内两个核均在运算、消耗不同资源、没有争用或其中任何一个位于服务关键路径。已选 scatter 的记录区间为 258.095 µs，其中大部分与此 Matmul 相交，所以不能把其 258.095 µs 直接相加到端到端延迟，或当作融合收益。

对选定 norm、RoPE、scatter 和 Matmul 的四个导出区间，用该 JSON 的十进制起止时间做并集：各自时长的算术和为 **641.0875 µs**，区间并集为 **330.934 µs**；差出的 310.1535 µs 是重复计入的记录交叠。scatter 与 Matmul 相交 **239.415 µs**，为该 scatter 记录时长的约 **92.76%**。这里只是这四条记录在同一 trace 坐标中的几何关系，既不是设备有效工作时间，也不是删除或融合后的加速上限；融合可能改变依赖、资源争用与其他任务进度，线上结果不能由此反事实推断。

同一 scope 已选 scatter 输入 cache `[17314,128,512]` BF16；后续 SparseFlashMla 47 / 44196 的第二个输入 `[17314,128,1,512]` BF16 与将 scatter 的 cache 增添单例维度**形状相容**。其开始晚于 scatter 记录结束 2803.744 µs；但形状、时间和层 scope 不证明相同存储、访问相同槽位或存在直接事件边。期间有其他已记录任务，也可能有未记录任务；时间差不等于等待或可被隐藏的时间。Indexer 47 / 44194 的第二输入 `[17314,64,1,128]` INT8 与上述 SWA cache 形状/dtype 不同，不能凭同层顺序说这个 scatter 写的就是该 indexer 的 key 缓存。indexer 开始晚于 scatter 记录结束 1775.944 µs，这同样仅是时间关系。

## 源码条件下融合的流水风险

冻结 `dsa_v41.py` SHA-256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`。在 `multistream_dsv4_dsa_overlap` 为真且 `cache_hidden` 非空时，`multistream_preprocess:440–495` 设计了主流 Q_b matmul 与辅助流 KV norm、RoPE 和条件 SWA scatter 的交叠：主流等待 `kv_matmul_done` 后提交 Q_b，辅助流等待 `part3_start` 后提交 norm/RoPE，符合条件时写 cache；主流在继续之前 `wait_stream(aux_stream)`，若 `write_cache_on_main` 为真则 join 后才由主流写。上述记录区间与此设计相容，但**不证明这次运行确实走多流分支或事件配对**。

如果候选融合把原先辅助流可执行的 norm/RoPE/scatter 整体移到主流、并排在 Q_b 前或后，可能串行化源码中原来可以重叠的部分；如果融合仍放在辅助流，也要保留输入就绪、主流 join 后可见、SWA cache 持久性与可能的资源争用。只是保证最后 attention 结果数学上相同不足以证明性能或事件安全；也不能因此断言融合一定变慢，具体影响取决于合法调度、算子资源及实际依赖。逐 tile 提前放行还需证明对应槽已就绪、cache 冲突和生命周期可见性，以及实际接口支持细粒度同步。

源码标准分支 `_write_swa_cache:297–308` 与 `_forward_attention:613–683` 引用该层 `swa_cache_layer.kv_cache[0]`，因此在这些源码分支和相同 runtime 对象等条件成立时，存在**逻辑 cache 状态的写→读**；indexer 则通过 `_write_compressed_source` 更新 key 和 `_select_sparse_indices` 选用其来源 cache，不能直接把两个 cache 当成同一个。以上源码条件与已选采样是否一一绑定仍未知。需要同次记录中的 runtime 分支和 cache storage 身份、view offset/stride、slot/block table 与当前 query 使用的槽集合，以及跨流 event record/wait 配对，才能确认所选 scatter 到 attention 的直接数据/同步关系；还需独立的无 profiler 对照与有效资源计数才能量化性能影响。当前不发起新采集。

复算 `python3 agents/fusion/f08_ep8_pipeline_overlap.py`；脚本 SHA-256 `6ce6247ffff89f806ad66c7179d095fa02983f5424ad34f9a47a6c2ca5cd7e8d`，聚合 JSON SHA-256 `510c88ca2632a214f224cb1d0cf9a344a6d0cf31dfa5fcdb59d9bea2029f2920`。没有更改目标实现。
