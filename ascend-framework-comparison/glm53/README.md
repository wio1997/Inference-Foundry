# 原 GLM-5.3 独立项目

原模型两套2P2D：先新SDMA vLLM基线，再SGLang同模型/精度基线与profile驱动优化；W8A8C8另列实验。 本目录完整自洽；另一个模型对话不属于本项目。

| 框架 | P | D |
|---|---|---|
| vllm_ascend | 30,31 | 43,45 |
| sglang | 34,35 | 36,41 |

模型：`glm_moe_dsa` / `GlmMoeDsaForCausalLM`，本地入口`/data/wio/models/GLM-5.3-w8a8`，193文件、764,240,829,290bytes。

先读[接手提示](START_PROMPT.md)、[当前交接](HANDOFF.md)、[资源与模型身份](RESOURCES.json)和[资产证据](ASSETS.md)。实施用[本项目方案](PLAN.md)，性能用[本项目合同](BENCHMARK_CONTRACT.md)，实际操作遵循[本项目ZCode规则](EXECUTION_CONTRACT.md)。本项目独立维护config/scripts/runs/results，配置及成绩均不得从另一模型带入。

同一份基础镜像可以分别固定到两模型，但权重、加载器、状态布局、PD语义和资格验证分别完成；没有跨模型性能比较。资源准备已有证据，新模型服务/性能是否已被其他接手者推进，必须先读现场真实进程和本目录最新STATE。
