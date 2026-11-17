# G43：完整模型中的融合算子证据

G41 首轮正式冷热测试完成后，G33 相同 P/D 实例另跑独立诊断输入：16×32768→1024、C16。该诊断不是正式性能结果。两端原生 profiler 各完成八个 rank，CPU 后处理验证全部 16 个 rank 的真实设备计算事件，且均记录到 720 次 `_sinkhorn`。源代码、输入和部署身份由各 rank 报告与冻结回执关联。

此前 G30 相同诊断输入中，每个 rank 的逐元素 Div 为 28080 次；G43 所选设备算子表中没有该精确 Div 项。该观察支持完整模型实际使用了融合路径。不同 profile 的算子时长有重叠，不能把累计设备时间当作可移除墙钟时间，也不能据此把 G41 全部吞吐变化归因于融合。

P 原始证据共 1497 文件、5913435656 字节；D 共 1201 文件、1572587115 字节。原始 CPU/NPU trace 保留在任务证据目录，本分支提供逐文件 SHA256 清单、各 rank 紧凑分析与源代码身份。HTTP stop 返回值不作为完成依据：验收检查了原生八 rank 完成记录、文件稳定性、原始文件 SHA，以及独立 CPU 分析退出码 0。

为执行用户新增的双模型分配，G43 冻结和后处理完成后，旧 GLM G33 两端已正常退出；GLM 后续 SGLang 迁至 GLM 专用节点，新的服务 epoch 需要重新验证与预热。历史 G41 结果的部署上下文不改变。

- [16 rank 融合执行摘要](../evidence/G43_all16_fused_kernel_receipt.json)
- [P rank 汇总](../profiles/G43/p/review/role_p.json)
- [D rank 汇总](../profiles/G43/d/review/role_d.json)
- [P 原始文件清单](../profiles/G43/p/raw_stability_p.json)
- [D 原始文件清单](../profiles/G43/d/raw_stability_d.json)
- [诊断请求审计](../profiles/G43/d/workload_terminal_audit.json)

四档最终同口径比较与最终差异地图仍未完成，GLM-OPT-0002 的最终收益裁决保持 INCONCLUSIVE。
