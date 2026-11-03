# F08：当前 stage 容器已安装 ScatterNdUpdateSk 的索引与搬运语义

## 证据范围

只读取 169 任务的 stage 容器中 `/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/` 内文件，以及已冻结的 V4.1 Python 源与既有 profile 聚合。**当前容器文件尚未与历史两个设备任务实际加载的二进制、tiling 参数绑定**，因此以下源码语义仅用于候选假说与正确性约束，不能断定历史 L9/L30 运行了该源代码的特定分支。本次无新 NPU 负载、无模型实现修改。

容器中实际可读的自定义算子入口 `op_impl/ai_core/tbe/custom_transformer_impl/ascendc/scatter_nd_update_sk/scatter_nd_update_sk.cpp` SHA-256 `7beb62e25c2c5f5e5059054da4f937747e925cf1dec46f460fd7ac39b791f57d`；`arch22/scatter_nd_update_hp.h` SHA-256 `5741acc73f5ee48c1cfcde63cf1ebb7ddf2a748173c32ee45367a2a4198871bb`；`arch22/scatter_nd_update_common.h` SHA-256 `b33f796fdd033582dee545d58d4c8f2f6e010a58c043083db15a575596e27b01`；入口 API 头 `op_api/include/aclnnop/aclnn_scatter_nd_update_sk.h` SHA-256 `f5b0d646e33b251697dc97df6f18123923ef36ccdbd4a4884006393e5e7eeede`。当前源码目录与容器内安装的 kernel `.o` 同时存在，不代表历史进程当时使用的是这些二进制。源码入口另有 `scatter_nd_update_sk_apt.cpp`（SHA-256 `ebcdc3897945a7777a8f47e3655e98866464529e0f5ab51d6bb14d73d295375f`），不应将两入口的分支语义混同。

## 可确认的源码行为与约束

1. 当前 arch22 入口 `scatter_nd_update_sk.cpp` 第 58–70 行注释并实现：tiling key 10/11 调 `RunHp<..., int>`，20/21 调 `RunHp<..., int64_t>`，30 调 `RunLargeIndex`。入口注释称 HP 按索引分核，在每核融合线性索引及 scatter，不做 `SyncAll`；重复索引跨核写入不确定。此为该代码注释及控制流，不是历史任务选中 key 或其二进制的证明；也不是说其他 key 有同样的冲突语义。
2. 在 HP 的 `Process` 与 `ComputeLinearIndexFromIndices` 中，各 core 按 tiling 划分自己的索引行、分 tile 搬索引并计算线性索引；`ScatterTile` 据 `scatterTileNum` 和行对齐条件选择分支。`DoBatch` 先将一整批 updates 搬入，再逐行检查线性索引非负决定是否搬出；`ScatterTileSlice` 则先检查线性索引，负值直接跳过该行后续分片搬运。**所以即使后续确认存在 padding，也不能笼统说 padding 一定不读 updates：是否会读取取决于具体分支及硬件行为。** 这也不能反推出某条短任务更快的原因。
3. `ResolveOutOffset` 按 `isViewStride0` 选择连续偏移或首轴 stride 的视图偏移，API 首段显式接受 `strides`。`scatter_cache_sk` 的 Python 注释声称缓存平面可能是更大分配的视图，因此算子需要保留页 stride；代码将 cache `squeeze(-2)`，切片 indices，并将 updates 转换为 cache dtype 后 `.contiguous()`。这些是**源码分支可能行为**；既有导出 `NCL/ND/ND` 与 shape 不能唯一恢复首轴 stride、storage offset、tiling 分支或缓存命中。
4. 冻结的 `dsa_v41.py` SHA-256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`。对于普通构建路径，源码把负 `active_slots` 映射成 `[-1,-1]`；非负值以 `storage_block_size` 的整除与余数生成两维坐标。该 helper 的文档要求 `[-1,-1]` 跳过；HP 源码检查的是**线性化结果负值**，并非通用地逐维检查每个越界索引。不能据此声称任意非法坐标都被安全跳过，也不能推断历史 8180 行中 padding 数量或索引互异性。压缩路径还有 ratio、有效位置等附加条件，不能用 SWA 的分支替代它。

## 和历史短长样本的关系

已有同 EP3、同摘要、同 stream 的 L9/L30 对照，已导出 shape、dtype、格式与 block 一致，时长分别 16.680/265.060 微秒；详细选择和相关性见 `F08-SCATTER-VARIATION-AUDIT.md`、`F08-SCATTER-GAP-STRATIFICATION.md`。当前源码明确存在视图首轴 stride、索引分核与分 tile、批量/分片搬运和重复坐标等影响语义或工作量的条件；**已导出相同形状无法排除这些条件不同**。现有采集没有确认两个样本的有效索引数量、索引内容及重复、cache 地址/真实 stride、tiling key/具体参数、运行二进制或 task 内并行硬件争用，因此无法从新源码发现中排除“不同有效工作量”“不同布局或分支”“资源争用/时序口径”等备选解释，更不能宣布其中任一为根因。

后续允许证据时，应优先将每个 task 的运行时 op/二进制 hash 与 key、tiling 参数、cache strides、索引总数/负值数/重复数的**脱敏聚合**和捕获范围绑定；核实性能采集 clock/PMU 语义，再评估相关性。当前禁止新增 NPU 负载；不需要读取 token、prompt、权重或输出值。

## 只读复核入口

在 stage 容器内使用 `sha256sum` 和 `sed -n` 读取上述具体文件；本机 `python3 agents/fusion/f08_scatter_variation.py` 可复算既有已聚合样本。该容器没有 `rg` 时使用 `grep`。本次检查没有声称穷尽其他路径或历史安装状态。

ZCode 已运行 `python3 agents/fusion/snapshot_scatter_installed_sources.py`，只从任务容器复制上述四份当前源码并校验哈希到 `agents/fusion/frozen-source/cann-scatter-sk/`；主控独立校验四份副本的 SHA-256 均与上文一致。此快照便于复核源码，也不能证明历史进程加载了同一版本或由其构建的二进制。
