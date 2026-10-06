# GLM Extreme — MISSION

checkpoint158之后的研究优先级以[AGENTS](AGENTS.md)为准：代码优化优先，Sol high承担性能架构师职责，旧HANDOFF/next_action仅作证据；不重写历史事实与裁决。

目标：为GLM-5.3 W8A8定制现场可用的极致优化推理框架，在标准P/D分离、既有SLA和完整功能下逼近当前两台服务器可达到的稳定有效服务容量。现有正确算子下，优化P Prefill→KV Transfer→D Decode→Output的真实完整请求关键路径、动态请求与资源调度。用户2026-10-06明确：DP1/TP16、DP2/TP8等是P池、D池各自内部的并行布局，不能将目标替换为完整请求副本。一次超过Stock是中间结果，最终报告当前最佳、剩余空间、适用条件和无法解决的限制。

## 专用化范围与现场动态能力

用户2026-10-06再次澄清：允许为GLM-5.3拥有完整请求/步骤控制权，成套重写scheduler、runner控制、状态与buffer组织、Graph派发、通信及PD编排；可删除或替换无实际GLM消费者的多模型/backend分派、通用对象/元数据转换、重复序列化及兼容抽象。vLLM/Ascend是可复用资源与源码起点，其既有架构不是产品约束。只有会增加实际关键路径/资源成本的开销才形成性能假说；不能仅凭层数多就断言有收益，也不要求先重写整仓库。

专用化固定模型及已核验的算子/状态语义，现场请求保持动态：内容、输入/输出长度、上下文增长、到达时机、并发、批次成员、采样、EOS/stop、取消与超时均不锁死到某次benchmark。动态合批/补位、背压与准入、资源回收及完整已有外部功能须成立；容量限制须真实、明确，不能靠截短输出、排除合法请求或冻结cohort换收益。稳定物理buffer、Graph分型和预分配可以优化执行，但不能成为固定request IDs、固定长度或固定到达序列的限制。

## 初始部署背景与当前目标

- 用户确认两台同规格、卡配置相同的Ascend910C服务器；结合前文按每台8卡16芯理解，实际rank/实例映射待核验。
- 初始背景为vLLM-Ascend0.27、PD已运行、参考DP1/TP16/EP16、GLM采用MTP；这不是当前现场事实。checkpoint159现役是无KV connector的完整请求副本，见[产品地图](CURRENT_PRODUCT_MAP.md)，与用户目标不符。Run101有真实PD功能证据，但没有完整PD产品验收或性能KEEP。
- 用户2026-10-06切换目标到GLM-5.3 W8A8，两机实际上传目录为`/data/tiankuan/wio/GLM-5.3-w8a8`，与旧权重独立；新服务模型名`glm-53`。新对话已获授权继续实现/验证/实验，早前上传期间仅资料约束不限制独立源码与CPU研究；上传未终态前仍不加载、不启动模型或设备测试。[checkpoint160](records/points/GLM-OPT-0002/research/pd_critical_path_20261006/CHECKPOINT160.md)核验全部索引引用文件的header结构、tokenizer和实际安装消费者；上传工具终态、payload校验和、fit/MTP/功能及真实完整PD性能仍待证明，不将旧GLM-5.2结果改标为5.3。
- 本阶段当前两台资源内收敛；未来1P3D等只是配置例子，不等于固定目标或物理机数。额外资源未具备的方向给当前结论。

## 性能与功能合同

完整接口与行为从实际框架恢复并持续补充；清单只用于覆盖，不限制能力。真实负载下比较有效吞吐、TTFT/TPOT/尾延迟、失败/拒绝、队列稳定性和资源成本。用户既有SLA的延迟阈值已记录在[验收实现](runtime/aisbench_slo.py)，本任务沿用以下严格小于门槛，不再将SLA写成未给定：

| 指标 | P50 | P75 | P90 | P99 |
| --- | --- | --- | --- | --- |
| TTFT | <4秒 | <8秒 | <12秒 | <30秒 |
| TPOT | <18毫秒 | — | <40毫秒 | — |

以上为分位数，不以平均值替代；未规定的分位门槛不自行补值。现有分析器按成功请求统计延迟，因此必须同时审计全量请求完成/失败/拒绝和实际有效输出，不能筛选快请求或少生成来达标；有限样本尤其小样本P99不能证明现场稳定尾延迟。固定请求数、长度、并发或salt只属于某个可复现工作包，不是产品永久限制。目标是在该SLA和完整合同下提高稳定有效容量，覆盖动态到达和混合请求行为。

正式比较固定相关条件，代码收益必须matched A/B；配置、部署和代码收益分开记账，没有真实完整E2E不得提升Current。只有代码级KEEP算主要性能进展。冷启动、实验迭代成本与稳态服务性能分开。

授权改调度、执行组织、状态/buffer、Graph、通信发起/等待、部署和Runtime边界，允许成套重构。当前不开发新计算kernel/fusion，不自动切换算子实现或减少模型必要工作。

## 后续交付与代码研究

已有现场/Current以真实Git、HANDOFF和Run恢复，不从初始发布状态重做。恢复后检查Goal Review信号，Sol直接研究代码与决定性证据，选择最大可信可消除Gap；不预定V2、grammar/compatibility或配置扫描路线。允许停止高投入但无Product Gain方向，删除不必要通用Runtime层并重构调度/状态/执行顺序。

标准PD为`Request → P Prefill → KV Transfer → D Decode → Output`；各Replica完整执行Prefill+Decode称Full Replica / Complete-request Placement，当前用户已排除其作为最终产品。历史部署与研究结果保留，只在匹配的问题范围内作证据。完整功能的合法生成请求及后续轮次须证明PD路径；不以short/tools/background/geometry不覆盖时D本地完整执行替代PD验收。取消、拒绝、完成及KV回收同属合同；内部helper输出只计成本，不计有效输出。具体目标纠正与源码/raw见[PD_SCOPE_CORRECTION](records/points/GLM-OPT-0002/research/engine_commit_20261006/PD_SCOPE_CORRECTION.md)。后续交付matched A/B、correctness、可重复完整E2E Gain和[代码性能账本](CODE_PERFORMANCE_LEDGER.md)，无新增KEEP明写“本阶段没有新增代码级性能 KEEP。”不重跑足够历史证据；必要连接入口未知则写缺项，不猜旧脚本/PID为现役。
