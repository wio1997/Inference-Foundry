# Independent Astra Challenger — checkpoint160

Reviewer：`gpt-6-astra`，独立subagent `pd_challenger`；只读，无服务器/文件/服务操作。输入包包括最终PD产品/SLA、CurrentNone/stack空、H1/H2已有反例、无性能patch/无matched5.3E2E、实际源码与Run101 raw。另提供实际Ascend V1 runner、executor及完整聚合类片段；没有让Reviewer只复述HANDOFF。

结论：**PIVOT，从预设completion-delay优化转为一个限域completion/admission诊断；保持严格PD，暂不写独立通知通道。最大可消除性能Gap unknown。**

源码事实：idle-D有0-token no-forward poll；有负载时completion capture在target-forward context退出，publication则跟随后续ModelRunnerOutput/MTP；executor收全部response队列，aggregator跨调用等expected workers；scheduler除remote completion外仍受token budget、运行slot、KV及已commit批次约束。transfer/reformat的CPU marker包含失败清理，reformat中NPU操作的返回不能独自证明设备可见。Run101只有wrapper/helper时间，没有ready/admission/device因果链。

排名是下一研究价值，不是已测耗时：①全rank成功/device-safe KV后错过合法D机会；②P/D重复且串行的原生准备（同时有合同动机）；③真实transfer/reformat同步与rank tail。最大已证明产品缺陷另外是optional PD fallback。仅①成为条件问题，②③不成为并行主线。

最小区分实验：权重ready前准备opt-in代码及离线夹具；ready后单个短baseline包含idle-D与decode-overlap receive。关联request/rank/step、成功device-ready、capture/publication、count→complete、scheduler receipt/promotion/admission、有效D execute/commit/公开输出，同时记录已commit batch、budget与slot；同机monotonic时钟，跨机误差单列。仅在晚观测确实错过合法机会时写候选。ready→admission envelope不是预测收益。

Stop List：完整副本/旧H1/PP phase作为PD优化证据；扩whitelist/fallback冒充完整PD；未经消费者证明删除rank barrier、错误处理、bootstrap或同步；先重写IPC/scheduler再找因果；用参数扫描/长E2E/旧权重代替5.3证据；把Review/diagnostic/功能配置KEEP算PERF_KEEP。

最易误读：把第一个worker的CPU“done”到D admission全算框架浪费；或把7.49/7.52/60.03s helper wall算可删除控制开销。Sol接受全部事实、反例与诊断优先级；rank0/CPU marker假ready、未证直接通知重写均否定。CurrentNone、active PERF_KEEP空、完整PD E2E Gain unknown。
