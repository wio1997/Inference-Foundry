# Inference Foundry — AGENTS

先按任务目标选择作用域，避免加载其他模型的规则和历史：

- GLM-5.3-Flash独立适配项目：只从[Flash规则](ascend-framework-comparison/glm53-flash/AGENTS.md)与[Flash交接](ascend-framework-comparison/glm53-flash/HANDOFF.md)接手。
- 原GLM-5.3独立比较/优化项目（含W8A8/C8）：只从[原模型规则](ascend-framework-comparison/glm53/AGENTS.md)与[原模型交接](ascend-framework-comparison/glm53/HANDOFF.md)接手。两模型对话没有任务/合同/状态依赖，父级比较目录仅导航。

- 历史GLM-5.3 / SGLang研究：仅明确继续旧研究分支时读 [glm5-3-sglang/AGENTS.md](glm5-3-sglang/AGENTS.md) 和其 [HANDOFF](glm5-3-sglang/HANDOFF.md)。为本线修改共享代码时也采用本线规则；不继承旧 GLM / DeepSeek 的 Current、部署任务或许可。
- 旧 GLM 研究：仅在明确继续旧线时读 [glm5-3/AGENTS.md](glm5-3/AGENTS.md) 和其 [HANDOFF](glm5-3/HANDOFF.md)。新线按需引用旧证据，不把旧线状态当当前事实。
- DeepSeek任务：读[原AGENTS全文](docs/deepseek/AGENTS.md)，再恢复根目录HANDOFF/当前Task。原规则原样保留，其中原根目录相对引用仍按仓库根解析。
- 跨模型方法：按问题查FOUNDRY_METHOD、performance_knowledge及相关证据，不默认通读。

所有任务遵循用户当前目标和约束。公共网站使用HTTP/API/Web Search；需要浏览器时先说明限制并取得用户许可。不要把别的任务的Current、PID、KEEP或next_action当作本任务事实。
