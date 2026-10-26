# GLM-5.3-Flash 独立项目

先建立Flash的vLLM文本/PD参考，再完成SGLang Flash混合状态/KDA与1P1D适配，后单图/多模态和性能。 本目录完整自洽；另一个模型对话不属于本项目。

| 框架 | P | D |
|---|---|---|
| vllm_ascend | 14 | 15 |
| sglang | 46 | 47 |

模型：`glm5_next` / `Glm5NextForConditionalGeneration`，本地入口`/data/wio/models/GLM-5.3-Flash-w8a8`，97文件、332,970,562,139bytes。

先读[接手提示](START_PROMPT.md)、[当前交接](HANDOFF.md)、[资源与模型身份](RESOURCES.json)和[资产证据](ASSETS.md)。实施用[本项目方案](PLAN.md)，性能用[本项目合同](BENCHMARK_CONTRACT.md)，实际操作遵循[本项目ZCode规则](EXECUTION_CONTRACT.md)。本项目独立维护config/scripts/runs/results，配置及成绩均不得从另一模型带入。

同一份基础镜像可以分别固定到两模型，但权重、加载器、状态布局、PD语义和资格验证分别完成；没有跨模型性能比较。资源准备已有证据，新模型服务/性能是否已被其他接手者推进，必须先读现场真实进程和本目录最新STATE。
