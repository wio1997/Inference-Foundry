# 同一窗口的详细性能账单

2026-11-10。用户要求将设备空档、具体算子和耗时完整细分。

- 介绍：统一复算idle-scopes-profile-20261110T045717Z的16卡，每卡11个闭合循环；仅处理已有trace和CSV，本轮新profile/模型请求/NPU测试/服务改动0。
- 到了哪里：设备79类计算/辅助操作与5类通信全部排行；保留每卡调用数、时长并集、单次分位数、完整shape/dtype/format签名、目标图内时间和PMU。输入准备按最深已记录CPU事件分配，消除嵌套重复；未覆盖区间保留。
- 结果：每卡平均839.32ms，算子任务454.87ms、空档322.15ms、未重叠通信61.76ms。空档内prepare input309.77ms；其中约155.35ms尚未绑定具体Python函数，明确列为待定位。目标DFC144.54ms/17.22%，量化矩阵乘43.57ms/5.19%，MatMulV2 34.37ms/4.10%，HcPre33.50ms/3.99%，QLI27.28ms/3.25%。
- 交付：REPORT.md、findings.json、rank-data.json，以及operator-ranking.csv/host-breakdown.csv/host-operations.csv。全局HTML默认页改为详细时间账单，旧采样对照保留并标明。
- 下一步：沿已有metadata-host / BatchFrame先价共同构造和发布；仅对155ms等会影响具体设计的函数归属补证。通用矩阵乘先按shape和实际消费者绑定，再定开发任务。已有耗时不等于可省收益，候选优化保持待确认。
