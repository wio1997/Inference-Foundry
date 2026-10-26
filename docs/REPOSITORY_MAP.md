# 仓库整理记录 · 2026-10-26

本次将“当前任务”和“历史研究”分开，使根目录只有一个当前入口。

```text
README.md
AGENTS.md
docs/REPOSITORY_MAP.md
projects/910c-deepseek-v4.1-flash/
  README.md
  STATUS.json
  config/       # 固定模型、镜像身份与端口
  scripts/      # 预检、启动、服务命令和 smoke 验证
  evidence/     # 小型现场证据
  runs/         # 本轮状态及验证摘要
archive/legacy-20261019/
  ...          # 整理前根目录的完整内容与原有目录结构
```

| 历史内容 | 归档入口 |
|---|---|
| GLM-5.3 / Flash 的独立项目 | [ascend-framework-comparison](../archive/legacy-20261019/ascend-framework-comparison/README.md) |
| GLM / SGLang 研究 | [glm5-3-sglang](../archive/legacy-20261019/glm5-3-sglang/README.md) |
| 旧 GLM 研究 | [glm5-3](../archive/legacy-20261019/glm5-3/README.md) |
| DeepSeek 原规则和研究 | [历史根目录](../archive/legacy-20261019/README.md)、[规则](../archive/legacy-20261019/docs/deepseek/AGENTS.md) |
| 跨模型方法 | [FOUNDRY_METHOD](../archive/legacy-20261019/FOUNDRY_METHOD.md) |
| 调优候选与知识 | [tuning-brainstorm](../archive/legacy-20261019/tuning-brainstorm/README.md)、[performance_knowledge](../archive/legacy-20261019/performance_knowledge/) |

归档整棵旧 Git tree（`b049efd56394f2afc0fa6ad3bebac36140998c0e`），未删减或重写历史文件。
原目录内部相对链接保持原有关系；历史文档中的硬编码绝对路径或旧分支 URL 仍指向历史位置。
本分支不修改 main 或原比较分支，也不把历史精度、性能或部署资格自动继承到当前模型。
