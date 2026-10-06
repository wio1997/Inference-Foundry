# GLM Extreme — MISSION

checkpoint158之后的研究优先级以[AGENTS](AGENTS.md)为准：代码优化优先，Sol high承担性能架构师职责，旧HANDOFF/next_action仅作证据；不重写历史事实与裁决。

目标：功能完整的GLM-5.3 W8A8标准P/D分离框架；现有正确算子下，优化P Prefill→KV Transfer→D Decode→Output的真实完整请求关键路径、动态请求与资源调度。用户2026-10-06明确：DP1/TP16、DP2/TP8等是P池、D池各自内部的并行布局，不能将目标替换为完整请求副本。一次超过Stock是中间结果，最终报告当前最佳、剩余空间、适用条件和无法解决的限制。

## 初始部署背景与当前目标

- 用户确认两台同规格、卡配置相同的Ascend910C服务器；结合前文按每台8卡16芯理解，实际rank/实例映射待核验。
- 初始背景为vLLM-Ascend0.27、PD已运行、参考DP1/TP16/EP16、GLM采用MTP；这不是当前现场事实。checkpoint159现役是无KV connector的完整请求副本，见[产品地图](CURRENT_PRODUCT_MAP.md)，与用户目标不符。Run101有真实PD功能证据，但没有完整PD产品验收或性能KEEP。
- 用户2026-10-06切换目标到GLM-5.3 W8A8，两机实际上传目录为`/data/tiankuan/wio/GLM-5.3-w8a8`，与旧权重独立；新服务模型名`glm-53`。上传完成、config/index/tokenizer与分片完整性、实际artifact/cache dtype、MTP及安装版本支持均待核验。上传期间只改配置与资料，不加载、不启动服务或测试；不将旧GLM-5.2结构、功能及性能结果改标为5.3证据。
- 本阶段当前两台资源内收敛；未来1P3D等只是配置例子，不等于固定目标或物理机数。额外资源未具备的方向给当前结论。

## 性能与功能合同

完整接口与行为从实际框架恢复并持续补充；清单只用于覆盖，不限制能力。真实负载下比较有效吞吐、TTFT/TPOT/尾延迟、失败/拒绝、队列稳定性和资源成本。沿用既有SLO；无SLO时报告折中曲线。正式比较固定相关条件，代码收益必须matched A/B；配置、部署和代码收益分开记账，没有真实完整E2E不得提升Current。只有代码级KEEP算主要性能进展。冷启动、实验迭代成本与稳态服务性能分开。

授权改调度、执行组织、状态/buffer、Graph、通信发起/等待、部署和Runtime边界，允许成套重构。当前不开发新计算kernel/fusion，不自动切换算子实现或减少模型必要工作。

## 后续交付与代码研究

已有现场/Current以真实Git、HANDOFF和Run恢复，不从初始发布状态重做。恢复后检查Goal Review信号，Sol直接研究代码与决定性证据，选择最大可信可消除Gap；不预定V2、grammar/compatibility或配置扫描路线。允许停止高投入但无Product Gain方向，删除不必要通用Runtime层并重构调度/状态/执行顺序。

标准PD为`Request → P Prefill → KV Transfer → D Decode → Output`；各Replica完整执行Prefill+Decode称Full Replica / Complete-request Placement，当前用户已排除其作为最终产品。历史部署与研究结果保留，只在匹配的问题范围内作证据。完整功能的合法生成请求及后续轮次须证明PD路径；不以short/tools/background/geometry不覆盖时D本地完整执行替代PD验收。取消、拒绝、完成及KV回收同属合同；内部helper输出只计成本，不计有效输出。具体目标纠正与源码/raw见[PD_SCOPE_CORRECTION](records/points/GLM-OPT-0002/research/engine_commit_20261006/PD_SCOPE_CORRECTION.md)。后续交付matched A/B、correctness、可重复完整E2E Gain和[代码性能账本](CODE_PERFORMANCE_LEDGER.md)，无新增KEEP明写“本阶段没有新增代码级性能 KEEP。”不重跑足够历史证据；必要连接入口未知则写缺项，不猜旧脚本/PID为现役。
