# Inference-Foundry

**Qwen27 两框架逐请求对比：[组件、流程、机制、源码与实测耗时分解](framework-maps/qwen27-ascend-20261028/REQUEST_FLOW_COMPARISON.md)。** 含 1,344 条请求端点、逐步 chunk/graph、算子通信窗口及输出至 EOF 的诊断链路；未采全的内部时间明确标示。

## 框架对比研究归档

| 归档 | 内容 | 状态 |
|---|---|---|
| [Qwen27 昇腾双框架地图](framework-maps/qwen27-ascend-20261028/README.md) | vLLM Ascend / SGLang Ascend：28 场景、原生 profiling 索引、机制与候选优化地图、诊断代码和公开数据 | 研究快照；完整机制验收待补证，非优化采用 |
