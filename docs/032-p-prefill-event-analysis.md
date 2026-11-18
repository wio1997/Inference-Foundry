# GLM-5.3-Flash：P 侧诊断轨迹逐事件分析

## 范围与结论

本稿仅消费已完成的离线摘要，范围是昇腾 910B1 上 GLM-5.3-Flash W8A8 的 **P 侧 32K 诊断窗口**：SGLang G181/G240 与 vLLM G4/G138 各选 rank 0、1，诊断输入为 32768，输出为 1024，客户端配置并发 16。诊断采集不计入正式性能成绩；不覆盖 D 侧或其他输入长度。相同输入数据集不证明两个窗口的逻辑 forward、有效行、padding 和 shape 一致。

**可确认的现象**：两框架两个 rank 均有 AllGather 事件；SGLang 还出现 DeepEP normal dispatch/combine。设备 CSV 中两框架部分 indexer 和 GMM shape 不同，SGLang 的 KDA 设备记录缺少 shape。按事件类型和标识符可以形成局部关联，但 CPU 到设备的完整因果链尚未贯通。

**不能据此得出的结论**：AllGather 是 MoE 专属、某项累计耗时就是可消除的墙钟损失、两框架算子总和之差解释吞吐差异，或某个候选已经带来端到端收益。事件处于多流和不同层级，可能重叠；窗口、工作量和跨主机时钟未对齐。

## 来源与完整性边界

[脱敏白名单摘要](../evidence/p-prefill-whitelist.json)包含来源对象 SHA、局部计数、事件分位数和筛选过的形状示例。以下 SHA 是来源记录里的原对象指纹，**不表示本次离线制稿重新读取或重算了远端文件**。

|框架 / rank|轨迹 SHA-256|设备 kernel CSV SHA-256|验证范围|
|---|---|---|---|
|SGLang 0|`92b0b3ec44036c268c23c4eb68844313e77406a2701b7ce629ae85f349f09922`|`760ec51155e49d2adadd17e20a4beaaf25887e1fec83b1ac461f278274672b79`|G240 冻结索引的 SHA、大小、mtime 均匹配|
|SGLang 1|`5cc44d4363565033f46e8ecbcce4d61c7b9a1c1cf2d71fcedf90d39f25d070ad`|`bd0e5b8eb7ac600fc13d7f7375360b8322c44651afe5bfae58b103e3ee434e78`|同上|
|vLLM 0|`3b3dd739f60a1259511aeece38ac05a2c5cd6e54c00115611954e964fbe2f142`|`9355cef2dbf5839023aea1dda5b0721f2ae5874e3087e1f454d84a7a842b6128`|轨迹为派生文件且与对应 rank 分析报告的 SHA 匹配；kernel CSV 两次读取一致|
|vLLM 1|`12b467eae38c4fa17d459827e4f991445fc87b37812b61b7a1d2d22a23adb52c`|`b5687cd1268035d2023d2bb55ee3de26c9531771e291d5832814ab6a3f6240b5`|同上|

SGLang 的 `operator_details.csv` 和 `op_statistic.csv` 也逐项匹配 G240 冻结索引的 SHA、大小和 mtime。vLLM 的 `trace_view.json` 和设备 CSV 来自 G138 冻结原始记录的 G148 派生分析副本，**不是原始二进制**。两个 rank 的 vLLM `kernel_details.csv` 两次读取 SHA 一致，但没有历史冻结索引的预期 SHA，不能称为历史索引认证；`operator_details.csv` 仅有单次稳定读取证据。原始 `torch.op_range` 与 `torch.op_mark` 在两个 rank 分别抽检匹配冻结索引；未重算全部原始二进制。

来源记录所列 P 侧冻结索引 SHA：SGLang `4e47296340f2f39e542c35fad47f9c80e42b16a69755c7830920aa854e7f443d`；vLLM `1a8abb67cf084e352f15223d934b79700e722d9155ddb5423d6e9c4cd1a2322a`。本次对离线摘要重新计算 SHA：来源说明 `83864f2f61ec805746ba0c477c32b2d5b1f87831a058034d1a00928b99527ba9`、逐 kernel 摘要 `0603dd338e51499697e52c8b8f3dfc862af86503689e5b296e584b654121ce03`；其余来源摘要和脚本的完整 SHA 均列于白名单 JSON。

## 单事件与形状观察

下表为各框架单个采集窗口中设备 CSV 的事件；`0/1` 对应 rank 0/1。p50 单位为微秒，采用排序后 `round((N-1)×0.5)` 的最近索引；这些不是单请求耗时，也不能跨框架相减。

|事件类别|SGLang：次数、p50|vLLM：次数、p50|边界|
|---|---|---|---|
|KDA 主核|272/272；4001/4013|272/272；3287/3281|SG shape 缺失；vLLM 还存在 transpose 核；不能认定同工作量|
|lightning indexer score|96/96；13889/13873|96/96；16241/16269|SG 有 query `8192,128`，vLLM 还含 `7680,128` 及不同 KV/批次字段|
|GMM|688/688；476/492|360/360；349/379，另有 360/360 融合 GMM+SwiGLU+quant|事件种类、调用数与 shape 不同，不能仅以次数或累计时长比较|
|AllGather 运行事件（不同事件层）|`HcclAllGather` 360/360|`allgatherAivKernel` 1867/1864|事件名称和层级不同；不能将全部事件认定为 MoE dispatch，不能将计数相加或直接比较|

例如，SGLang GMM 的 CSV 首维随事件变化；vLLM 的融合 GMM 记录有 `65536,4096` 和 `61440,4096`。这些是设备 CSV 原始形状字段，**不是已确认的有效 token 数**。`record_shapes=false` 说明 CPU profiler 未记录形状，不等于所有设备 CSV shape 均缺失。

此前按宽正则分组的摘要允许同一融合事件进入多个组，不能将组间时间求和；本稿设备事件按 CSV 完整名称分类。逐事件分位数、更多示例及对应 SHA 见白名单 JSON。

## 局部事件关联

- **enqueue/dequeue**：各采集窗口内，按 `correlation_id` 分组。例如 SGLang 两个 rank 的 dispatch 各有 344 个 enqueue、344 个 dequeue，唯一 ID 交集均为 344；GMM 各为 688，KDA 各为 272，indexer 各为 96。vLLM 对应 GMM 各为 720、KDA 各为 272、indexer 各为 96。ID 相同本身不是完整因果边。
- **间隔而非空闲时间**：同一 trace 中唯一配对的 `dequeue` 开始减去 `enqueue` 结束，SGLang dispatch rank 0/1 的 p50 分别为 4.62/3.19 微秒，p90 为 7.09/4.31 微秒；rank 1 有一个负间隔。vLLM rank 0 GMM p50 为 2.75 微秒、p90 为 42934.31 微秒。这只是标记之间的间隔，不证明设备空闲、可消除墙钟或跨框架差距。
- **launch/device**：抽样 flow 里可见 `enqueue_to_dequeue` 的两端；另有 `Node@launch` 与设备事件各自共享 `connection_id` 的样本。SGLang 样本包含 KDA、indexer、GMM 和 DeepEP AI_CPU kernel；vLLM 部分 connection_id 对应多个事件，不能强行一对一。已核对的 CPU `cpu_op` 样本本身没有 `correlation_id`。两类标识符各自提供局部证据，但**未证明 CPU-op → enqueue/dequeue → launch → device 全链映射**。

这些观察不证明多框架对应步骤属于同一请求，不跨主机直接相减时间戳，也不把通信层、设备核和 CPU 标记视为互斥事件。

## 后续验证边界

1. 先在原始轨迹中扩展抽样，厘清 `connection_id` 的边语义，以及它与 CPU、enqueue/dequeue 的连接；对齐相同逻辑 forward、shape、有效行和角色，不能只按事件名匹配。
2. 在已授权资源可用、不会干扰在飞任务时，针对缺失的形状或链路补采最小必要窗口；将真实任务、不同流重叠及跨主机时间口径明确留证。
3. 对通信、融合或 KDA 候选先验证语义与机制是否发生，再做受控同机端到端测试；未完成前不把局部事件、热点或预计收益写成根因或性能改善。
