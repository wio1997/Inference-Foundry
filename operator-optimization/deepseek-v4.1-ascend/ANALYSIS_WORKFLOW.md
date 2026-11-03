# 分析流程：新profile怎样接入

本流程复用分析逻辑；历史任务ID、张量地址和计时数值不复用为新观测。此文不启动采集。

## 1. 固定新案例

保存模型/权重与源码版本（含dirty改动）、CANN/torch_npu、设备/核心/并行组、输入输出长度/并发/前缀复用、量化、graph/eager、采样参数和数值域。无profiler参考与诊断窗口分开。主指标、单位、分母、窗口和rank先定义。

用Git/source hash和配置差异判断失效范围：只变shape则重算工作量/代价；只换profile则重绑运行映射和统计；改量化/缓存/后端则使对应数值/状态合同失效，不把所有模块全部重读。

## 2. 只解析一次，先聚合再看局部

先读实际schema/表头，使用匹配安装版本的离线解析。全量统计operator次数、shape/dtype、累计kernel时间、适用区间并集、各engine正分母活动字段。保留missing/ambiguous，不填0。输出小摘要和完整索引，再按问题取局部事件；不把GB trace交给模型通读。

历史脚本入口可在下列目录查找；先用resolve_reference定位，复制到新run后只改输入路径/schema适配和配置，保留原版本：

- `shared-profile-analysis-v165/rank_profiles.py`：全部profile/operator聚合与连接基数检查。
- `shared-profile-analysis-v172/`：功能模块归属、reportedcycles加权活动率、双窗口/歧义和竞争原因。
- `shared-profile-analysis-v174/build_canonical_comparisons.py`：统一模块/候选/匹配子集P5记录。
- `shared-profile-analysis-v177/collect_operator_guard_census.py`：只对profile/加载类选中的函数建立精确条件引用；引用数不等于语义已证明。
- `shared-profile-analysis-v179/`：消费者/alias/state实质核验、DSpark/candidatebank/batch顺序修正。

不能把新CSV字段直接套旧列号，不能跨cohort把receiverwork除以另一轮时间。调用栈/连接ID用于归属；函数嵌套或时间相邻不等于数据依赖。

## 3. 建图并筛机会

维护生产者、所有消费者、输入/输出dtype、数值边界、别名/原地更新、持久状态、通信归属、value/request/parameter epoch。已有算子内融合先登记为基线事实。

按七类规则检查：生产消费两两、三节点链/分叉、继续扩展、拆分重组、共享输入、跨模块/层/卡/迭代、多消费者。依据数据/共享输入/资源/状态关系生成候选；不枚举所有候选笛卡尔积，也不因图上缺边就认定独立。变化后只复扫受影响邻域。

每个候选分别回答：能省什么逻辑工作/物化/搬运；语义条件是什么；还有谁读取中间结果；新活跃数据/重算/同步/资源竞争是什么。当前API不支持记UNSUPPORTED/CONDITIONAL，不等于理论机会不存在；反例否定的是具体改写，不是所有可能融合。

## 4. 只为一个可裁决问题取证

先查knowledge和原候选记录。写出：具体claim、已有答案、缺失字段、哪两种观测会改变结论、最小证据来源、停止条件。

优先新profile的局部关联。只有profile无法回答且会改变结论，才查对应符号/少量代码。例：三次gather与两次写入虽已在profile中出现，acceptedELSE的读取顺序仍需精确函数体；这次只补看26行，而不是重读框架。

保留原公开算子接口时，不要求先搞清全部私有参数。只有要直接删除/改写该私有helper时，才需要相应destination/extent/alias/retirement证据。一次探针只能支持它实际观测到的分支。

## 5. 重计数并比较

必要工作下界/能力上界匹配域后，取计算、各存储层/链路和必要依赖下界的最大值。融合子图整体重计数，不相加独立理想收益；当前workspace/allocatedtile不是必要流量/MAC行数。

T、Q、L必须同对象/同case/同计时边界才计算Q/T、L/T、T/L。额定clock/实测速率不当物理capacityupper。缺参数保留符号或INCOMPLETE。未实现候选的T为空。NPU活动率、MFU与端到端暴露分别记录。

## 6. 输出一个决策增量并停止无效循环

每轮只保存：新输入指纹；复用了哪些结论；新增/推翻的结论；候选状态变化；仍会影响下一决策的具体问题。没有新证据的已知UNKNOWN不生成另一份相同报告。

实质进展是新增可裁决证据、推翻假设、必要界被有依据收紧，或实现得到正确性/同条件性能验证。清点同一批文件、重建相同图、重复更名/配对、原样重写报告不算技术进展。

整个声明范围未分析完时保留缺口。完成条件是已声明范围/Case/规则一轮覆盖无新非等价候选且每项有分析答案；它不证明所有算法穷尽，也不证明候选已经更快。
