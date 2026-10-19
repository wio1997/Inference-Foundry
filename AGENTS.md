# Inference Foundry — AGENTS

先按任务目标选择作用域，避免加载其他模型的规则和历史：

- GLM-5.3 W8A8C8 / vLLM-Ascend 优化：默认读 [glm5-3-vllm/AGENTS.md](glm5-3-vllm/AGENTS.md) 和其 [HANDOFF](glm5-3-vllm/HANDOFF.md)。为本线修改共享代码时也采用本线规则；不继承其他研究线的 Current、部署任务或设备许可。
- SGLang 研究与部署资料：从 [glm5-3-sglang/README.md](glm5-3-sglang/README.md) 按需取证；明确继续该线时再读其 AGENTS/HANDOFF。已有部署和测试记录保留，不作为新 vLLM 线的实测结果。
- 旧 GLM 研究：仅在明确继续旧线时读 [glm5-3/AGENTS.md](glm5-3/AGENTS.md) 和其 [HANDOFF](glm5-3/HANDOFF.md)。新线按需引用旧证据，不把旧线状态当当前事实。
- DeepSeek任务：读[原AGENTS全文](docs/deepseek/AGENTS.md)，再恢复根目录HANDOFF/当前Task。原规则原样保留，其中原根目录相对引用仍按仓库根解析。
- 跨模型方法：按问题查FOUNDRY_METHOD、performance_knowledge及相关证据，不默认通读。

所有任务遵循用户当前目标和约束。公共网站使用HTTP/API/Web Search；需要浏览器时先说明限制并取得用户许可。不要把别的任务的Current、PID、KEEP或next_action当作本任务事实。
