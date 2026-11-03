# 当前交接

- 项目：DeepSeek V4.1 Ascend算子优化检查与分析逻辑归档。
- 当前：原任务v180记录为分析验收完成；42候选仍为条件式，未实施优化。
- 当前任务：按用户要求维护每个候选的逻辑分析、验证状态和待办。入口为[逐条进展](progress/README.md)；历史归档已提交，不从旧记录自动启动实验。
- 当前候选：[F06 同输入行域Q_A/KV联合投影](progress/F06.md)。已核对裁行、已有share_quant及CV交错，并复用Q3/KV24盲拼约2.67倍计算反例；同域联合投影仍待实现、正确性和收益验证，其他变体不关闭。
- [F05 Q路径NormQuant](progress/F05.md)保留源链/舍入/多消费者及分支重算讨论；[F20 MoE TopK与条件分支](progress/F20.md)已作用户点名局部复核。转下一条讲解不表示前面方案实施或验收通过。
- [F04 NormCast及Router输入](progress/F04.md)已记录用户表示了解并转F05；已有融合不重复实施，可选新衔接和低成本试验仍待执行。
- F04量级筛查：171 v166 NPU-only所选回放内11520次整NormCast平均2.913µs；旧166小M约3µs、大M约33–35µs，批次不作前后比较。纯转换未拆分计时；用户认可可尝试，当前仅完成只读定价，倾向低成本/大输入验证，不启动重工程或NPU实验。
- [F03 层内Post→Pre](progress/F03.md)已记录用户回复了解并转F04；R其他读者及舍入保留，新接口/正确性/收益仍未验证。跨层其他子方案未关闭。
- [F02 PRENORM](progress/F02.md)已记录用户确认“把后面的归一化接入HcPre”的方案边界。该理解确认不表示实现/正确性/性能通过，其他F02子方案未关闭。
- [F01 EARLY-Y](progress/F01.md)保留源码依赖已澄清、可实现性/正确性/收益待验证的状态；转第二条不关闭F01，也不改变其独立复核未启动的事实。
- 首读：progress/README.md、当前候选页、REUSE_AND_DEDUP.md；再按需要查REPORT.md、CANDIDATES.md和证据。不要通读历史。
- 独立复核：用户已请求GPT-6.1/xhigh；当前临时侧聊禁止子代理，尚未启动。任务恢复包在[progress/tasks/f01-early-y/](progress/tasks/f01-early-y/README.md)。
- 事实缓存：knowledge/facts.jsonl；问题状态：knowledge/questions.json。
- 原始profile/trace/DB不在仓库；以后有新数据，再建立新的run/cohort，不覆盖历史结论。
- 下次若开始实施：由用户选定具体候选和案例，先查已有事实/反例/条件，仅对改变决策的缺证据取证。不能因归档存在旧授权或next_action自动操作服务器。
- 更新记录：修改progress/candidates.json、向events.jsonl追加决策增量，再运行tools/render_progress.py。每次实质证据、纠正、失败或下一步变化都提交；重复阅读/讲解不记为技术进展。
