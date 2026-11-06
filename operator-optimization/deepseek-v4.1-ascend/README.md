# DeepSeek V4.1 · Ascend 算子优化

本分支保存算子优化分析、当前优先级和可复用证据。历史v180归档包括26个功能模块、360条语义行、35个模型候选族和7个框架边界候选族；该历史阶段不代表后续状态。**当前固定负载首选F31内CP＋Slot组合，已有约10.1%—11.2%的限定整批耗时降低；正常入口尚未实际运行验收。**F12有完整原生局部净值，HcPre行范数实现回放失败，两者生产收益UNKNOWN。

固定案例：DeepSeek-V4.1-Flash-1017-w8a8，Ascend910C/A3，16逻辑设备/8物理卡，CANN9.2.0，TP8/DP2/EP16，DSpark5，32K输入/1K输出、C8、P90文本负载。它与本仓库DeepSeek Extreme P0的W4A8/910B3/DSpark7案例独立，不继承后者的Current、KEEP、PID或next_action。

## 日常只读这些入口

首先看 **[优先级与详细方案](PRIORITY-PLAN.md)**：明确点名保留11项/9族，当前3方向/4改动，另有3项前端后备和4项长行/容量后备。行动顺序为CP＋Slot组合、F12真实传播、HcPre行平方和有效实现；后两项不是已证明的生产收益强排。**[进展看板](progress/README.md)**保存42条当前行动、各变体状态和追加纠正；F16保留长行条件，旧v168的1.20%撤回生产定价。登记、理解确认和引入已有实验均不等于本次新增实施。

1. [PRIORITY-PLAN.md](PRIORITY-PLAN.md)：封闭11项清单、42族取舍、工作量、已有结果、下一步及停止条件。
2. [HANDOFF.md](HANDOFF.md)：当前状态与恢复边界。
3. [REPORT.md](REPORT.md)：结果、覆盖、最低活动模块、理论对照和未知。
4. [CANDIDATES.md](CANDIDATES.md)：42个候选的具体组合思路及条件。
5. [ANALYSIS_WORKFLOW.md](ANALYSIS_WORKFLOW.md)：换一份profile后怎样按同一逻辑分析。
6. [REUSE_AND_DEDUP.md](REUSE_AND_DEDUP.md)：哪些结论直接复用，怎样避免重复无效工作。

不要把历史目录全部塞进上下文。先查事实/问题：

```sh
python tools/lookup.py F21
python tools/lookup.py DSpark
python tools/lookup.py alias
```

## 归档内容

- [records/latest/](records/latest/)：合同、当前图、42候选图、26模块汇总、84算子聚合、94条P5对照与状态。
- [records/history/](records/history/)：源分析过程、历轮报告/候选/覆盖/条件/问题/失败与修正记录、离线分析脚本和小型证据。
- [method/](method/)：融合扫描、性能建模、profiling分析和诊断的原方法合同。
- [knowledge/](knowledge/)：可复用事实及未决问题的适用条件、失效条件、证据。
- [progress/](progress/README.md)：每条候选的逻辑、状态、证据、下一步与追加纠正；[CHANGELOG](progress/CHANGELOG.md)和Git提交支持回溯。
- [inventory/summary.json](inventory/summary.json)：收录/排除/压缩/脱敏统计；[manifest.jsonl](inventory/manifest.jsonl)保存原始与发布SHA。

**不提交原始profile、trace、数据库、逐事件大导出、模型/容器、连接凭据或执行日志。**分析需要的聚合结果与结论保留；后续可以重新profile后复跑逻辑。这次归档没有采集新profile或操作服务器。

大于5MiB的历史文本按确定性gzip保存，避免把大JSON作为普通Git文件。原始分析文件中的绝对路径与原SHA保留；公开副本中若有凭据脱敏，manifest明确记录，不把脱敏后的副本声称为原始字节。

## 查历史引用与校验

```sh
python tools/resolve_reference.py '/Users/wio/work/166-vllm-analysis-continuation-20261030/shared-profile-analysis-v179/semantic-contract-sufficiency-review.json'
python tools/verify_archive.py             # 默认只校验当前入口，避免每轮重扫历史
python tools/verify_archive.py --full      # 下载/归档验收时才全量校验
python tools/render_progress.py --check   # 只核对当前进展页是否与登记一致
```

历史文档中的COMPLETE/UNKNOWN/失败/next_action均属于其原cohort与当时结论。历史归档入口是v180，当前行动以PRIORITY-PLAN及progress追加证据为准；旧脚本里硬编码的目录、taskId、指针、rank/group、时钟和运行flag不应直接套给新profile。历史控制脚本不自动执行。
