# GLM 32K P 侧 rank0：两个独立局部时间轴

**状态：两份历史 profile 的真实设备事件局部样本，不是同一请求、同一步骤、整模型时间线或性能排名。** 每框架各取一个 P rank0 稠密窗口，各自归零；两次采集属于不同服务 epoch，不能对齐横轴或把事件累计时长当墙钟。事件跨 stream 可重叠，SG 366 条与 vLLM 48 条仅是本次选窗和可识别事件的结果，不能据此判断哪个框架更慢。

## 1. 源数据与选窗

两份轨迹对应 32K 诊断、客户端配置 C16，不计作正式压测。SGLang 取 G240 冻结采集导出的 P rank0 trace；vLLM 取 G138 冻结原始采集经 G148 后处理的 P rank0 **派生 trace**，不是原始二进制。两份源端轨迹的哈希与归档 rank0 收据一致；vLLM 此前仅另核对 `torch.op_range` 和 `torch.op_mark` 两个 frozen raw 组件，不表示重跑了 SDK 转换或核对所有原始组件。

| 样本 | 源轨迹 SHA-256 | 源轨迹字节 | 自本 capture 首个白名单设备事件起的窗口偏移 | 250 ms 窗口可见事件 / anchor |
|---|---|---:|---:|---:|
| SGLang | `92b0b3ec44036c268c23c4eb68844313e77406a2701b7ce629ae85f349f09922` | 286874061 | 1934.111 ms | 366 / 276 |
| vLLM | `3b3dd739f60a1259511aeece38ac05a2c5cd6e54c00115611954e964fbe2f142` | 201336673 | 57.810 ms | 48 / 48 |

源端算法在各自 capture 的白名单设备事件中，以 anchor 事件开始数选取 250 ms 内最稠密窗口，同数取最早；如可见事件超过 1000，才缩短窗口。anchor 排除 wait 及部分 AICPU 类别，但输出保留与窗口相交的白名单事件，包括起点在窗外而被裁切者。两份记录最终均为 250 ms；窗口 offset 是各自局部相对偏移，不是跨框架时间差。仅凭已发布样本无法重新证明其为原 trace 全局最稠密窗口。

来源：[SGLang 相对样本](../evidence/p_timeline_036/sglang_rank0.json)、[vLLM 相对样本](../evidence/p_timeline_036/vllm_rank0.json)、[SGLang rank0 收据](../evidence/matched_SG_G181/profile_G240/CPU_p/review_g252/rank_0.json)、[vLLM rank0 收据](../evidence/G4_profile_G138/CPU_p/review/rank_0.json)。两份原始局部样本 SHA 分别为 `5c57c4352641087d7fc73b62eeb06698a4bfeebcb393dd055bfd87789cd6163c`、`8c09453d225eb7e567ac765f827998c783a5816aae4a0e22567ce57fd2821e4e`；先前静态可视化文件 SHA 为 `3a869d83d636b676ea4b6ab511960702814926ef27f468b11516d90fef65b9fb`，本仓不复制该文件或提取脚本。

## 2. 可见事件：只是选中窗口

| 类别 | SGLang 窗口 | vLLM 窗口 | 观察限制 |
|---|---:|---:|---|
| 普通 GroupedMatmul | 18 | 15 | 未匹配 shape、有效行、专家分布或 target/draft。 |
| fused GroupedMatmul+SwiGLU+quant | 本窗口未识别 | 15 | 未识别不代表整个采集或运行中不存在；不能相减得收益。 |
| KDA | 7 | 12 | 事件名和实现不同，未按逻辑 step 或 shape 配对。 |
| indexer / SparseFlashAttention | 各 2 | 各 3 | 同名不等于同输入或同工作量。 |
| DeepEP normal dispatch / combine | 10 / 9 | 本窗口未识别 | 加上各自 AICPU/notify 类别仍可能重叠，非通信墙钟。 |
| SDMA / device wait / notify wait | 209 / 36 / 25 | 本窗口未识别 | 事件层级、导出和命名均可能不同，不能视为跨框架等价项。 |

上述“未识别”仅针对白名单和该局部窗口，不等于全量 trace 中事件次数为零。每个 stream 编号只在自己的 capture 内有效；设备/通信/等待等事件可并发，不能简单相加成模型运行时间。相关的不同 shape 与 source 层级分析见 [P 侧逐事件证据](032-p-prefill-event-analysis.md)。

**精度警告：**源端提取器先筛选真实 `duration > 0`，输出时将起点和时长分别四舍五入到 **0.001 ms**。SGLang 的 **35 个事件在归档数据中显示 `duration_ms=0`**，这是舍入结果，不是源事件实际零时长或不存在；vLLM 该窗口没有这样的事件。SGLang 还有 2 个左裁切、1 个右裁切事件。缩放图中零显示时长仅用标记表示，不赋予非零宽度，也不能从中恢复真实时长。

## 3. 离线验证与查看

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/timeline_036.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/test_timeline_036.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_archive.py
# 可选：输出自包含静态双面板 HTML 到调用方选择的位置
PYTHONDONTWRITEBYTECODE=1 python3 scripts/timeline_036.py --html > timeline_036.html
```

[独立校验/渲染脚本](../scripts/timeline_036.py)只读仓内白名单摘要和 rank 收据：校验 trace SHA、事件名称/类别、rank0、250 ms、1000 条上限、相对坐标裁切、anchor/类别计数及取整边界；`--html` 只向 stdout 输出两个各自归零的 SVG 面板。事件表采用固定列、受限类别和数字 stream，不包含设备 PID、绝对路径、IP、请求文本或原始大 trace。[合成及归档测试](../scripts/test_timeline_036.py)覆盖越界、非有限值、非法类别/stream、rank、元数据及静态双面板脱敏。

源端提取既有轨迹和哈希核验是历史记录；本次归档在本机将两个局部 JSON 与发布版逐字段核对，没有重新访问设备或重跑原 trace。仅凭发布样本不能重建原 trace、完整工作单位、逐请求时间线或性能收益。后续需要相同请求/forward、shape、通信依赖与时钟口径证据，才可以做可证伪的机制定位与端到端验证。
