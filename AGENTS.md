# Inference Foundry — AGENTS

先按任务目标选择作用域，避免加载其他模型的规则和历史：

- GLM任务：按[新会话恢复顺序](glm5-3/docs/START_NEW_CHAT.md#恢复顺序)先核验Git，再读[glm5-3/AGENTS.md](glm5-3/AGENTS.md)，经唯一导航与产品地图恢复；不通读runtime或全部Run。本仓库中为GLM修改共享代码时也采用该任务规则；局部规则覆盖旧DeepSeek身份、模型分工和kernel许可。
- DeepSeek任务：读[原AGENTS全文](docs/deepseek/AGENTS.md)，再恢复根目录HANDOFF/当前Task。原规则原样保留，其中原根目录相对引用仍按仓库根解析。
- 跨模型方法：按问题查FOUNDRY_METHOD、performance_knowledge及相关证据，不默认通读。

所有任务遵循用户当前目标和约束。公共网站使用HTTP/API/Web Search；需要浏览器时先说明限制并取得用户许可。不要把别的任务的Current、PID、KEEP或next_action当作本任务事实。
