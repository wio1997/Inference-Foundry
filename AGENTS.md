# 当前分支范围

本分支 `sglang-deepseek-v41flash-w4a8-910b-pd` 维护 SGLang DeepSeek-V4.1 W4A8 / 910B PD。

- 新的 vLLM 源码改动、部署、测量及 profile 写入 `vllm-deepseek-v41flash-w4a8-910b-pd`，跨框架比较使用明确 commit/证据链接。
- 继承的跨框架源码和组件为固定历史背景，保持原始来源与验收边界；不通过改名将其视作本框架结果。
- 固定模型与镜像、实际硬件、负载、原始输出/校验链应随结论留证。公开副本不得包含凭据、真实服务器地址或私有路径。
- 离线档案检查：`python3 scripts/validate_deployment_evidence.py`、`python3 scripts/validate_components.py`。离线通过不代表完整服务、模型质量或性能通过。
