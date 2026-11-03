# 当前交接

- 项目：DeepSeek V4.1 Ascend算子优化检查与分析逻辑归档。
- 当前：原任务v180记录为分析验收完成；42候选仍为条件式，未实施优化。
- 当前任务：把分析记录和复用方法提交此分支，不采profile、不启动旧实验。
- 首读：REPORT.md、CANDIDATES.md、ANALYSIS_WORKFLOW.md、REUSE_AND_DEDUP.md。
- 事实缓存：knowledge/facts.jsonl；问题状态：knowledge/questions.json。
- 原始profile/trace/DB不在仓库；以后有新数据，再建立新的run/cohort，不覆盖历史结论。
- 下次若开始实施：由用户选定具体候选和案例，先查已有事实/反例/条件，仅对改变决策的缺证据取证。不能因归档存在旧授权或next_action自动操作服务器。
