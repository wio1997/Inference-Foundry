# GLM 推理性能研究

目标是在给定业务范围和延迟要求下，提高 GLM-5.3 / Ascend 910C 的有效总吞吐，并判断剩余性能空间。优先复用 SGLang 的工程实现，同时独立判断模型依赖与执行方案。

## 入口

| 文件 | 职责 |
|---|---|
| [AGENTS](AGENTS.md) | 少量协作规则与设备操作边界 |
| [MISSION](MISSION.md) | 目标、探索空间与成功标准 |
| [HANDOFF](HANDOFF.md) | 当前事实、未知项与下一决定 |

[PLAN](PLAN.md)、[RECORDING](RECORDING.md)、[REUSE](REUSE.md) 是按需参考；普通修改不要求通读。新会话可使用[恢复提示](docs/START_NEW_CHAT.md)。

## 已有工具与证据

- [历史审计](docs/RESEARCH_RESET.md)、[20261007 补丁包](patches/20261007/README.md)：保留已有成果与适用边界。
- [真实片段草案](studies/GLM-PLAN-0001.md)：需要缩小实验时可参考，不是必做任务。
- [execution_lab 示例与使用说明](examples/execution_lab/README.md)：固定操作集合的资源约束估计和 CPU 回放。它是可选 toy 工具，没有真实 GLM / Ascend 性能结论，也不要求继续扩建。

工具的本地验证（无模型加载或设备调用；仅在相关修改时需要）：

```sh
python3 glm5-3/execution_lab/cli.py demo
python3 -m unittest discover -s glm5-3/tests -p 'test_*.py' -v
```

原入口保存在 [legacy](legacy/README.md)；当前入口的修订可在 Git 历史查看，不再复制一套规则归档。
