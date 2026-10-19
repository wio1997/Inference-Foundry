## 当前状态：2026-10-19 L22 完成，用户要求暂停

两边 C32 诊断和原生 trace 采集结束；当前不再启动实验。详见 [本轮总结](diagnostics/model-profile/C32-L22-pause-summary-20261019.md) 与 [当前交接](HANDOFF.md)。正式性能仍为历史 vLLM 530.8125 TPS /19.1527 ms、当前 SGLang L17 411.9606 TPS /37.9931 ms；没有新候选收益。下方旧状态为历史。

## 当前状态 — 2026-10-18 L08 已拒绝，L04 恢复通过

MemCache 已先核对对应 Ascend/框架/SDK 部署文档，见[版本与部署核对](results/memcache-docs-L08-reconciliation-v4.json)。L08原生MoE候选遇到207001内存分配失败，未进入功能或性能测试；已恢复同源L04并通过四台原生启动验收。没有本轮新增外部请求或性能收益。[关闭证据](results/C32-L08-loop-close-v2.json)和[最新恢复包](RECOVERY_LATEST.md)是当前事实；下文旧CURRENT标题均为历史。仅C32、普通W8A8、全P→D SDMA。

# 原 GLM-5.3 独立项目

原模型只用普通W8A8权重，两套2P2D：先新SDMA vLLM基线，再SGLang同权重/精度基线与profile驱动优化。 本目录完整自洽；另一个模型对话不属于本项目。

| 框架 | P | D |
|---|---|---|
| vllm_ascend | 30,31 | 43,45 |
| sglang | 34,35 | 36,41 |

模型：`glm_moe_dsa` / `GlmMoeDsaForCausalLM`，本地入口`/data/wio/models/GLM-5.3-w8a8`，193文件、764,240,829,290bytes。

先读[接手提示](START_PROMPT.md)、[当前交接](HANDOFF.md)、[资源与模型身份](RESOURCES.json)和[资产证据](ASSETS.md)。实施用[本项目方案](PLAN.md)，性能用[本项目合同](BENCHMARK_CONTRACT.md)，实际操作遵循[本项目ZCode规则](EXECUTION_CONTRACT.md)。本项目独立维护config/scripts/runs/results，配置及成绩均不得从另一模型带入。

同一份基础镜像可以分别固定到两模型，但权重、加载器、状态布局、PD语义和资格验证分别完成；没有跨模型性能比较。资源准备已有证据，新模型服务/性能是否已被其他接手者推进，必须先读现场真实进程和本目录最新STATE。
