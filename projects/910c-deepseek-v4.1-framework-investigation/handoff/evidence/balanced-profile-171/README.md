# 171 框架到算子性能数据包

入口为[全局性能图](index.html)，结论见[REPORT.md](REPORT.md)，断点见[STATUS.md](STATUS.md)。本次是配平诊断采集，原服务已经恢复。整个融合搜索尚未完成。

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

build_global_view.py使用同目录的chain-template.html、chain-view-data.json、global-view-fragment.html、global-view.css和global-view.js。view-validation.json记录本地交互校验。export_activity_timeline.py和export_full_kernel_inventory.py只在已有trace目录做CPU复算；collect_remote.py属于已完成采集的记录，不是接手续跑命令。

后续先归因两组共有的Free与目标DFC内部成本，再按完整交付边界核算净收益。报告中的稳态窗口占比不等于完整请求占比。旧测试使用8条不同输入，本轮4条镜像，不能从新旧吞吐差直接认定DP路由收益。
