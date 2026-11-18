# 当前分支范围

本分支 `vllm-deepseek-v41flash-w4a8-910b-pd` 维护用户指定 vLLM-Ascend v0.32.0rc1 / DeepSeek-V4.1 W4A8 / 910B PD。

- SGLang 改动、部署、测量及profile写入 `sglang-deepseek-v41flash-w4a8-910b-pd`，跨框架仅用固定commit/证据链接。
- 共享权重/数据用 metadata/framework_scope.json 的权威指针。原有组件、失败和源码保持来源SHA，不将其升级为完整模型/PD/质量/性能通过。
- 用户指定的0.32参考属于机制参考；本任务镜像、四档精确输入/C16/1024、完整target/draft图和独立充分预热要求保持。
- 公开产物脱敏，不包含真实服务器地址、私有路径或凭据；大型原始trace和权重不进入Git。
- 离线档案核验：`python3 scripts/validate_vllm_archive.py`。

- 2026-11-18用户更新：当前目标镜像为 `quay.io/ascend/vllm-ascend:v0.32.0rc1`。旧0.30源码、组件与G527配置仅保留历史来源，不继续按其部署，也不把旧组件门槛直接认作0.32通过；当前digest/状态见 `metadata/current_version_requirement.json`。
