# Inference-Foundry

**当前分支：DeepSeek V4.1 · Ascend 算子优化。**

本分支保存算子检查的数据分析结果、42个条件候选、覆盖与理论对照，以及避免重复分析的知识/问题索引。未上传原始profile、trace或数据库；没有在本次归档中重新profile或实施优化。

- **[逐条进展看板](operator-optimization/deepseek-v4.1-ascend/progress/README.md) · [当前 F07 逻辑与待办](operator-optimization/deepseek-v4.1-ascend/progress/F07.md) · [F04 记录](operator-optimization/deepseek-v4.1-ascend/progress/F04.md) · [进展/纠正历史](operator-optimization/deepseek-v4.1-ascend/progress/CHANGELOG.md)**
- [项目入口](operator-optimization/deepseek-v4.1-ascend/README.md) · [分析报告](operator-optimization/deepseek-v4.1-ascend/REPORT.md) · [42个候选思路](operator-optimization/deepseek-v4.1-ascend/CANDIDATES.md)
- [如何重新分析一份profile](operator-optimization/deepseek-v4.1-ascend/ANALYSIS_WORKFLOW.md) · [如何避免重复工作](operator-optimization/deepseek-v4.1-ascend/REUSE_AND_DEDUP.md)
- [固定事实与失效条件](operator-optimization/deepseek-v4.1-ascend/knowledge/facts.jsonl) · [未决问题与重开条件](operator-optimization/deepseek-v4.1-ascend/knowledge/questions.json)

以下保留仓库原有其他项目入口。它们的Current、KEEP、PID和next_action不属于本算子分析案例。

---


**Qwen27 两框架对比：[中文详解HTML](framework-maps/qwen27-ascend-20261028/request-flow/report.html)**（下载后直接打开）· [GitHub直接阅读版](framework-maps/qwen27-ascend-20261028/REQUEST_FLOW_COMPARISON.md)。新版将调度、图大小和算子记录解析为中文，含11章讲解、13阶段组件差异、1,344条请求与优化借鉴；不再把JSON当正文。

## 框架对比研究归档

| 归档 | 内容 | 状态 |
|---|---|---|
| [Qwen27 昇腾双框架地图](framework-maps/qwen27-ascend-20261028/README.md) | vLLM Ascend / SGLang Ascend：28 场景、原生 profiling 索引、机制与候选优化地图、诊断代码和公开数据 | 研究快照；完整机制验收待补证，非优化采用 |
