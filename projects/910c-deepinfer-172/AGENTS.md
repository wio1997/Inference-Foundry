# DSpark current scope — 2026-10-27 correction

- 用户明确要求官方 DSpark：学习式草稿、目标验证及置信度调度；提示查找不能替代。此前提示查找实验的完成结论撤回。
- 当前按用户条件PAUSED：172有外部训练占用；固定5指定两轮通过，变长调度未最终验收。旧文本中的“当前”“完成”“不再测试”仅对应历史提示查找实验，不约束新获授权的 DSpark 工作。
- 本次仅172，16逻辑设备，DP2 TP8 EP16。所有模型、配置、日志、缓存和证据放在 /data/tiankuan/wio。
- 简单机械工作交给本机 ZCode 配置的 DeepSeek；主控负责复杂设计、修改与独立验收。不向170发送消息，不停止其他任务。
- 仅本次172 DeepInfer privileged 容器已明确授权；权重和驱动只读，仅Engram CPU卸载，主干CPU prefetch关闭。
- 冻结 vLLM 与旧实验数据不覆盖；历史目录内的现场描述仅适用于其对应版本。
- 认证仅从当前Mac钥匙串直接送入SSH，不打印、不落盘、不提交凭证。公共网站用API/HTTP/Web Search，不打开浏览器。
- 同一AISBench 32768输入/1024输出/C8/两轮/89.990234%真实KV命中；校验16个完整输出，报告真实接受输出TPS及TTFT。
- 不把模型加载、接口就绪、置信头执行或正确性通过等同于性能达标。当前状态见STATUS及optimizations/20261027-dspark。

- 当前无本项目NPU服务。候选14只准备未启动；恢复先确认172无外部占用，不停止他人任务，不盲目重跑依赖已退出worker的旧transition。源码、数据和恢复点见STATUS及optimizations/20261027-dspark。
