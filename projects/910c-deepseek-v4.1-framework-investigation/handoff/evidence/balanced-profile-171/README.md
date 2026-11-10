# 171 框架到算子性能数据包

## 当前入口：空档覆盖与融合地图

全局HTML默认展示实测Host/设备重叠、数据与CPU/缓冲约束，以及可调的条件覆盖预算。主台账72条历史记录+P01流水研究项、45条执行重审记录均映射到12模块；记录数不等于独立算子数。旧函数账单、算子、16卡时间线保留。

[条件模型与分析](../overlap-timeline-171/REPORT.md) · [全部候选位置](../overlap-timeline-171/candidate-coverage.csv)。本轮只复用旧trace，无新NPU测试或性能实现；默认尚未确认可提前比例，不自动认领收益。

入口为[全局性能图](index.html)，结论见[REPORT.md](REPORT.md)，断点见[STATUS.md](STATUS.md)。本次是配平诊断采集，原服务已经恢复。整个融合搜索尚未完成。

默认首页现为[同窗详细性能账单](../performance-breakdown-171/REPORT.md)：用同一次阶段标签采样列空档细分、CPU具体操作、全部79计算/辅助类与5通信类；可查完整shape签名并导出CSV。旧balanced采样仍作为对照保留，不混用窗口。

79类型相加57.30%，并行去重后Computing为54.19%；[可加总核准](../performance-breakdown-171/RECONCILIATION.md)及[逐函数计时方案](../performance-breakdown-171/TIMING-PLAN.md)已提供。构建时还会读取同级performance-breakdown-171/reconciliation-summary.json，页面分别保留排行和互斥分账。

首页“时间花在哪里”先显示算子执行53%–55%、设备空档约38%、未重叠通信7%–9%，再列MoE/DFC等热点。首次采集尚未分清空档原因，后续阶段补采见下文；不能直接叫调度耗时。该分布来自每卡约1秒的稳态短窗口，不是整批16秒的完整分解。更详细的链路、16卡时间线和逐算子证据放在其余页签。

后续已完成[空档阶段补采](../idle-attribution-171/REPORT.md)：闭合循环中约96%的Free与prepare input重叠，原服务已恢复；图中新增“空档在做什么”。旧样本绝对时间与新增阶段样本分开记账，不直接相加。纯EngineCore调度仍未直接采集。

数据按以下顺序读取。

1. findings.json：负载、参考吞吐、分组窗口热点、配平检查和恢复证明。
2. balanced-analysis.json：32份worker、2份前端的逐卡计时、区间交集和同次Host→Device关联。
3. kernel-full-inventory.json：early全部88类、steady84类算子的逐卡时间、调用分位数、首例shape/dtype/format和PMU。
4. activity-timeline.json：复用现有trace，512格共同时间轴；非原始微秒级trace。
5. steady-host-gaps.json：代表卡EP0/EP8的CPU区间与Free交集；同时发生不等于因果。

原始trace留在launch.json所列171自有目录。这里的JSON和单文件HTML均可离线复用。balanced-reference-0/1/2.json、balanced-early.json、balanced-steady.json保存请求与指标核验，receipt.json和restored-smoke.json保存采集及恢复证据。每卡首个shape样例不代表该类型所有shape。

本机复建HTML只读上述证据，不运行模型。

```sh
python3 audit/balanced-profile-171/build_global_view.py
```

build_global_view.py使用同目录的chain-template.html、chain-view-data.json、plain-summary.html、global-view-fragment.html、idle-view-fragment.html、global-view.css/global-view.js，以及performance-fragment.html、performance-view.css/performance-view.js；载入同级idle-attribution-171/findings.json和performance-breakdown-171/findings.json。若在GitHub工作台的evidence/balanced-profile-171目录恢复，执行同目录脚本会输出index.html。view-validation.json记录本地交互校验。export_activity_timeline.py和export_full_kernel_inventory.py只在已有trace目录做CPU复算；collect_remote.py属于已完成采集的记录，不是接手续跑命令。

后续先归因两组共有的Free与目标DFC内部成本，再按完整交付边界核算净收益。报告中的稳态窗口占比不等于完整请求占比。旧测试使用8条不同输入，本轮4条镜像，不能从新旧吞吐差直接认定DP路由收益。
