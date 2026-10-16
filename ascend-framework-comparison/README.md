# 两个独立模型项目的导航

GLM-5.3与GLM-5.3-Flash是两个独立项目、两个独立对话。各自在同一模型内使用vLLM作参考/对照并适配或优化SGLang；两模型之间不比较成绩，不共享可变合同、状态或验收流程。

| 项目 | 工作 | 唯一接手入口 |
|---|---|---|
| [GLM-5.3-Flash](glm53-flash/README.md) | Flash文本/混合状态/1P1D适配，后多模态 | [独立START_PROMPT](glm53-flash/START_PROMPT.md) |
| [原GLM-5.3](glm53/README.md) | 原模型2P2D基线与SGLang优化，C8单独实验 | [独立START_PROMPT](glm53/START_PROMPT.md) |

本目录只导航和保存历史批次证据，**不是两项目的共同HANDOFF/实施方案/性能合同**。进入一个项目后默认只读该项目目录。各自维护AGENTS/HANDOFF/RESOURCES/PLAN/BENCHMARK_CONTRACT/STATE、config/scripts/runs/results，配置、模型路径、锁、Meta/provider及成绩独立。

两个对话各用GPT6.1 Sol High；本机ZCode DeepSeek执行要求分别写在各自EXECUTION_CONTRACT，不建立跨对话任务依赖或主控审批链。物理争用作为各自测量条件如实检查，不形成模型之间的逻辑依赖，也不授权操作对方服务。没有新建/消息其他对话、没有恢复定时。

旧tracks入口保留重定向，旧混合内容见[拆分前快照](reference/20261017/README.md)。旧4P2D[归档](archive/20261016/README.md)不变；旧资源批次JSON是历史验收，不等于新模型功能或实时服务状态。权重/镜像二进制和凭证不入Git。
