# 重采集 MC2：16 rank、40 层的条件比较

从已保存的 16 rank 提取结果，独立对照 sidecar hierarchy：16/16 rank、1,280/1,280 个 conditional target layer scope 与 PID、层号、digest、call_id 匹配，每个 scope 有一条通过提取器筛选并唯一关联的 MC2 任务。每个 digest 的 40 层各有相同的 8 rank，每个 rank 在该 digest/层只有一个已保留匹配；本次四组共 160 个 digest/层组。每 rank 导出中 `flow_finish_invalid_ts` 为零。

在保存的早期 23 份 metadata、验证文件哈希后，对四个 digest 逐一计数：每个 digest 均为一个成功的 scheduler schedule 和八个 runner begin，**仅限已采集窗口**；digest 是内容摘要，不保证在未采集时间段或其他服务中唯一，也不等于请求唯一身份。EP8、EP13 各有一个重复 flow-start ID 被提取器跳过；“每个 scope 一条已保留匹配”不证明原 trace 不存在其他未能归属的任务。

## digest 1b826883…：层 14 与层 39

比例按 `sum(aic_total_cycles × 显示的 aic_mac_ratio) / sum(aic_total_cycles)` 计算；各 rank 是同组不同任务，不能把它们时长相加当墙钟。

| 字段 | 层 14 | 层 39 |
|---|---:|---:|
| rank | EP8–15 | EP8–15 |
| 周期加权的显示 MAC 比例 | 0.4891% | 8.8831% |
| 单 rank 显示比例范围 | 0.1%–1.1% | 5.0%–16.4% |
| 单条任务时长最小–最大 | 160.365416–160.422455 ms | 8.217616–8.290414 ms |
| 单条任务时长中位数 | 160.377028 ms | 8.2284155 ms |
| 同组内 8 个时长的最大减最小 | 0.057039 ms | 0.072798 ms |

层 14 与层 39 的中位数相差 152.1486125 ms（中位数之比约 19.49），只是两个已选任务集合的描述；不证明每 rank 时间差来自同一机制，不代表端到端时延、整卡吞吐或设备利用率。这种大差异值得进一步检查实际执行路径、trace/时钟语义和同次任务数据，但当前不足以指定原因。

## 每个 digest 的最低层

按上述周期加权显示比例，仅在每个 digest 自身的已保留 40 层中排名：

| digest 前缀 | 对应 rank | 最低层 | 周期加权显示比例 |
|---|---|---:|---:|
| 1b826883 | EP8–15 | 14 | 0.4891% |
| 7ddf7ae9 | EP0–7 | 14 | 1.6268% |
| 8c18c293 | EP8–15 | 0 | 0.1372% |
| db852bff | EP0–7 | 15 | 2.7863% |

各组未证明输入工作量相同，不能由跨组的低高值判断谁更快、归纳全局最低，或推断根因。这里的显示 ratio 来源于导出字段，有限显示精度、舍入和 AIC 分母的版本语义未证明；不应把算术结果中更多小数位当成测量精度。周期加权 MAC 比例也不是整个芯片的利用率；AIV、通信、其他 stream 和设备状态不能从中反推。

该 flow 关联表明 CPU scope 到被保留设备任务的关联，尚不能证明完整 tensor 数据依赖或请求关键路径。即便 EP8 层14 形状含 INT32[24]，每专家实际行数、有效 mask、maxOutputSize 和 `R` 仍 unknown，不能从 ratio 或 1,023 行形状反算运算量或理论性能差距。

## 复算

[recheck_recapture_mc2_16rank.py](recheck_recapture_mc2_16rank.py)（SHA-256 `17e4f56344ad2c5bab967d57564211ca6854c5b429d139ec3bbb0b62961c1462`）纯 CPU 读取已保存提取结果和固定 sidecar，检查 scope、rank、digest、层、单一任务及正周期，输出 [mc2-16rank-recheck.json](../../diagnostic-20261029/mc2-16rank-recheck.json)（SHA-256 `4c95034a61a5c91c4922a61fd9668e40944c08c20623177ab98a6d6af671a6f9`）。原提取 JSON SHA-256 `16d9e17986f59c4caeea1d0b1157f2d839b8344df0fc7c91e1b3105b612448a3`；sidecar hierarchy SHA-256 `e4ffaf3cacaeebddb6beb249d9f48d76a3800dc5f02528cdc2ae46e6c3a8a023`。未触 NPU、未启动新服务。
## 两个物理行数下的 40 层跨组中位数对照

复用同一次重型采样的四个已选调度摘要；分别对物理首输入行数 1023 和 643，按层比较 EP8–15 的八条 MC2 记录中位数与 EP0–7 的八条记录中位数。每个层和每个行数组分别核验完整名称、类型、输入输出 shape/dtype/format、核心类型及 block 配置 16/16 一致。两者摘要和卡组不同，不是同卡配对或相同工作量对照。

| 导出首输入行数 | 层数 | EP8–15 中位数较长的层数 | 逐层中位数比最小 / 中位 / 最大 |
|---|---:|---:|---|
| 1023 | 40 | 33 | 0.990 / 1.531 / 8.528 |
| 643 | 40 | 40 | 1.174 / 1.456 / 10.331 |

这只是两个摘要中逐层导出时长的组中位数对照；各中位数可能来自不同卡、组中位数的中位数也不是单任务总体中位数。签名相同不保证有效 token、专家分配、实际调用实现、通信组、层输入权重或运行条件一致，无法判定设备故障、某个硬件组较慢、通信或路由是根因，也不能推断线上吞吐或可实现收益。层 14 靠近配置的 Engram 位置，但时间关系并不证明资源竞争；高值可能受到采样扰动。

离线复算 `python3 agents/diagnosis/compare_mc2_group_layer_medians.py`：输入 SHA-256 `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`；脚本 SHA-256 `165e757acc9f656465acd0e3b7581186759bb8d941cd5e862270bd2b7bee617a`；按层聚合结果 `agents/diagnosis/mc2-group-layer-median-comparison.json` SHA-256 `09e15ab3f96a8f98f4f12beae833fbf0bff12d744ab02b0dd7d5a9ac2232aea0`。不运行 NPU 或修改目标实现。
