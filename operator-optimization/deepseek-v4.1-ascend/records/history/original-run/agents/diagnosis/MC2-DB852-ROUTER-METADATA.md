# db852bff 第 15、36 层 EP0/EP3：同次路由元数据清点

沿用 [已绑定的四条采样](MC2-DB852-L15-L36.md)，仅查各自原始 profiler SQLite 的 `COMPUTE_TASK_INFO`，以已唯一匹配的 `globalTaskId` 查询，每条恰好一行，并解析对应 `STRING_IDS`。L15/L36 在 EP0 的 ID 为 2574/3392，EP3 为 2380/3198；归属依赖前述 TASK 匹配和导出器语义，不能仅因 ID 一致跨库关联。

四条记录导出的元数据完全相同：`blockNum=24`、`mixBlockNum=48`，输入 shape 为 `1023,5120;24,5120,4608;24,2304,5120;1023,6;110592;122880;1023,6;1023`，输出 shape 为 `1023,5120;24`；`attrInfo` 为空字符串，`gridDim` 和 `blockDim` 为 `N/A`。这些是导出字段而非 tiling 值或硬件实际执行工作量；INT32[24] 输出形状与保存参考源码中的专家计数契约相容，但不能以形状证明调用实际使用同一二进制、某一槽位的数值或有效专家分布。

对这两个 rank 的 profile 目录，仅按文件名限定词 `tensor|tiling|dump|snapshot|graph` 清点：每个 rank 各有 5 片 `aging.additional.tensor_info`、1 片 `unaging.additional.tensor_info` 和相应 `.done` 文件；未发现名称命中其余限定词的文件。这个结果只覆盖该目录和限定词，不检查其他目录，也不能排除不同命名方式或 SQLite BLOB。切片只统计名称和大小，**未解码内容**；目前没有证明它们的版本、结构、与四次调用的对应关系，或是否含 tensor 元素。禁止据此断言有、或绝无逐专家 count。profiler 主库 schema 包含 shape、类型、attrs、TASK、PMU 等字段，本次没有逐表内容搜索，也没有对其他两个分析数据库重复 schema 清点。

因此历史四次调用中每专家 count、有效 mask、`maxOutputSize`、tiling 值、每专家有效 M 及同次 graph/replay 身份仍未知；不能根据 `[1023,6]` 推断专家分布或判定负载倾斜导致时长差异。静态计数输出与图/流生命周期约束见 [已有专家计数分析](RECAPTURE-MC2-EXPERT-COUNT-20261029.md)。后续如能找到**同版本官方解析器和字段契约**，再针对四条任务做隐私受限的定向解码；否则需未来受授权、绑定具体调用的采样，但当前任务禁新 NPU 负载，不能把新运行数值回填给历史样本。

复现：在任务根运行 `python3 agents/diagnosis/audit_mc2_db852_router_metadata.py`。脚本在自有 stage 容器打开 SQLite 只读、仅获取所列元数据和候选文件名称/大小，不导出张量、模型权重或请求文本，不触碰 NPU。脚本 SHA-256 `101ece9bca36dce01c8727a6b81c65b403c8e6b5e35d3eba62023b46fcb7c5c5`；输出 [mc2-db852-router-metadata.json](mc2-db852-router-metadata.json) SHA-256 `44a14dcd38b3e236c056ad5e39f11e97a179e0d03bda274a41cff330e071ebed`；绑定来源 `mc2-db852-extremes-pmu.json` SHA-256 `aa0f79ded9488c3658684f39958bd52aea2e9444f32ab3e5c94eabdc9214492b`。
