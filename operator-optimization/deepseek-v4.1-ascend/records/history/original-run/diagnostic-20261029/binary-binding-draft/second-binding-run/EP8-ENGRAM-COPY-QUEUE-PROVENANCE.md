# EP8 队列中的长 InplaceCopy：已有证据追溯

对两次第 14 层目标 enqueue→dequeue 等待区间，前一项 `inspect_queue_waits.json` 各列出 6 个超过 20 ms 的 `Dequeue@aclnnInplaceCopy`，共 12 条。基于同一次 EP8 离线 profiler DB 的 `CONNECTION_IDS` 反向关联，这 12 条在本次**已导出的数据库范围内**各匹配一个同名 enqueue 和一个同名 dequeue，没有观察到同 correlation 的其他队列名称。enqueue 先于 dequeue。ID 在未导出的记录或其他会话中是否复用不能由此证明。

12 条中每条都包含同一 worker 线程的一个 `aclnnInplaceCopy`，其内各包含一个 `Slice_Tiling`。这些是嵌套区间，不应把三者耗时相加；所见 dequeue 为 20.847–22.306 ms，内部 `Slice_Tiling` 为 20.756–21.785 ms。队列中出现这些先前提交的操作支持继续检查队列服务，但不能仅凭相邻或时间包含，断言其中任意一条对目标层延迟的因果贡献、可节省时间或可隐藏时间。

enqueue 同线程落在下列 profiler PyTorch 算子和记录调用栈内；表中形状/dtype 是导出 `aten::copy_` 的输入记录，并非 tensor 地址或完整生命周期：

| 来源与栈顶 | 层调用 746-20，6 条 | 层调用 746-65，6 条 |
| --- | --- | --- |
| `engram/common.py:65` `engram_gate`，`key.float()`；栈内 `aten::to → aten::_to_copy → aten::copy_` | 2 条；`[1023,4,5120]`，float / BF16 | 2 条；`[611,4,5120]`，float / BF16 |
| `engram/common.py:72` `engram_gate`；同上转换调用链 | 2 条；`[1023,5120]`，float / BF16 | 2 条；`[611,5120]`，float / BF16 |
| `engram/embedding.py:342` `embed_gathered`，`gathered.contiguous()`；`aten::contiguous → aten::clone → aten::copy_` | 2 条；`[8184,24]`，int / int | 2 条；`[4888,24]`，int / int |

栈上 `copy_` 的 dtype 顺序是 float 目的、`c10::BFloat16` 源；导出中 int 仅显示为 `int`，此处不猜具体位宽。common.py 第 72 行包含多个转换，形状与 `value.float()` 相容，但没有 tensor 身份，不能仅据此断言具体变量。12 个 enqueue 的已记录外层 marker 分别是 `NPUModelRunner.execute_model:746-5` 或 `:746-50`；没有同线程包含它们的第 14 层 marker，故不能把它们直接称作该层内部操作。调用栈明确提供 Engram 来源，这比单纯时间接近强，但仍依赖 profiler ID 与记录完整性。

## 源码与既有候选

只读检查显示，本次已停止的诊断容器与 stage 使用同一镜像 `sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b`，两者 rootfs 均只读，源码目录 `/vllm-workspace/vllm-ascend` 无覆盖挂载。stage 对应源码 `engram/common.py` SHA-256 `4797b8962264d137cdb142a9988825288283092fb797ddbd334c6ab2223eb58d`，`engram/embedding.py` 为 `a7741ed16522be4fe607e062e1c988c4e2435af7d258fe7db99f672d4508c4c9`；行号和容器镜像/挂载事实保存于 `engram-source-readonly-evidence.json`。这建立了源码一致性的强条件证据，不等同张量数据身份。

普通 Engram gate 的转换与组合边界已在 `agents/fusion/ENGRAM-LOCAL-RECOMBINATION.md` 和 `COMPRESSOR-ENGRAM-REVIEW.md` 中作为条件候选分析；已存在的 fused gate 受 rotation、硬件 profile 等约束，不能断言 A3 此次已命中，或新融合必然合法。原 F24 聚焦 hash、lookup/gather/dequant，与这里 gate 转换及 `gathered.contiguous()` 不能直接画等号。`embedding.py:340–342` 说明 contiguous 是为了满足查表内层步长，消除或重排必须先证明输入 stride、kernel 接口、跨卡及数值语义；本次尚无这样的证明，也无新增融合的实测结果。

## 覆盖、扰动与复算

本采样启用 profiler 的 stack/shape 等重型记录；此前导出报告 `SQLite Error: disk I/O error`，即便数据库 integrity_check 通过也不能证明完整。21 ms 是该受扰动 trace 的 host API 区间，不能直接当生产线上该操作固有耗时或可获得收益；也不代表 NPU kernel 执行 21 ms。目标记录仅涉及 EP8 两个区间，不能推广到其他 rank、并发或请求。

运行 `python3 run_trace_copy_provenance.py` 对 task-owned 离线 DB 做只读、低 CPU 优先级分析，结果 `copy-provenance-audit.json` 逐条列出相关 ID、输入 shape、调用栈和嵌套 API；输入 `inspect_queue_waits.json` SHA-256 `9d2eabbe60179181df3719427e4c5b948c4f5cd938af9492baa0f5350f3f4b84`，DB 已保存 SHA-256 `95b58de883ee272b7161bce7b213d15673c1925fd165bafbf4deacf7a6b74b23`，输出 SHA-256 `642b7582fa1b55d7075393cf1049e1bbddb56179b57cd656f930c4fd4b38310e`。本轮无 NPU 采集、无目标代码修改。
