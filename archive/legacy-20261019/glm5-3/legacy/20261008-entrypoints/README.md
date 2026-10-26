# GLM-5.3 Extreme

2026-10-07 用户最新指令：先发布 MTP 和已有调试结果 patch，**两台机器暂时不做代码调优和测试**。当前只做本地包装/离线检查和 GitHub 同步；不安装候选、不重启/改服务、不提交模型请求或 NPU Run。已验证 H6/H5 研究栈保留，MTP 限域设备正确但未获重复收益。恢复设备调优/测试须由用户另行明确要求，历史 next_action 不构成恢复授权。交付说明：[20261007 patch 包](patches/20261007/README.md)。本次没有服务器动作，完整 API/正式 80K600/93% SLA/最终 Current 尚未验收。

[独立MTP patch](patches/20261007/H11-mtp-graph.patch) · [完整zip](patches/GLM53-patches-20261007.zip) · [各patch依赖、应用/回退、验证与性能说明](patches/20261007/README.md)。共13个独立patch和2个原字节溯源替代版；全部本地应用/回退/hash/AST校验通过。AISBench实际helper与phase_runner依赖同步到本仓库，正式条件80000输入/600输出/93%prefix/93%声明KV命中；声明和实测分账。

当前目标为 **GLM-5.3 W8A8 标准 P/D 分离**，在两台910C与现有正确算子下研究框架执行浪费及完整PD性能。权重路径为 `/data/tiankuan/wio/GLM-5.3-w8a8`，服务模型名 `glm-53`；历史版本记录不作为5.3验收基线。

**Agent启动只读[AGENTS](AGENTS.md)+[HANDOFF](HANDOFF.md)+活动点/Git状态。**

| 入口 | 按需用途 |
|---|---|
| [MISSION](MISSION.md) / [PLAN](PLAN.md) | 目标/合同、极限标准与逼近逻辑 |
| [RECORDING](RECORDING.md) / [优化点索引](records/points.jsonl) | Git提交、Run证据及裁决 |
| [REUSE](REUSE.md) / [复用索引](records/reuse.jsonl) | 旧成果适用性、失败原因与去重 |
| [研究索引](docs/README.md) | GLM特性、源码、prof、历史迁移与模型成本 |
| [新对话启动](docs/START_NEW_CHAT.md) | 短交接指令、模拟演练与实际环境接入 |

交给执行Agent的短prompt：

```text
在glm5-3/先读AGENTS.md和HANDOFF.md。当前用户暂停两台机器的代码调优和测试，只进行本地patch/证据交付；从patches/20261007/README.md查源码基线和结论。未获用户明确恢复指令，不按历史next_action启动设备实验或改服务。
```

当前发布已包含上述patch和已有实验结论。旧恢复资料保留其历史口径；本次状态以顶部暂停指令和patch说明为准。
