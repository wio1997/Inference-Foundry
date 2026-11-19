# GLM 256K 冷轮执行器工作目录错误与修正

**责任和范围：这是本次接手执行器选择只读工作目录造成的客户端启动错误，不是已证实的模型失败；该轮未产生有效性能结果。** 本文记录一次 G584 数据的冷轮事故，不改写此前首轮失败、已完成的历史测量或历史 claim。时间下文均为 2026-11-18 北京时间（UTC+8），不同主机时钟未校准。

## 1. 实际时间线和终态

| 时间 | 观察与范围 |
|---|---|
| 21:35:06 | 冷轮控制器启动；启动本身不证明请求已发送。 |
| 21:36:10 | G584 数据的唯一 claim 被独占创建，状态保留 `possibly_submitted_even_if_controller_interrupted`。 |
| 21:36 后 | 执行器把容器只读任务根设置为进程 cwd。客户端日志记录 `OSError: [Errno 30] Read-only file system`，配置临时文件的创建/清理失败。 |
| 23:01 左右 | D 侧只读快照中冷轮控制器及客户端进程仍在，未发现性能明细；P 路由所选日志无匹配的 completion POST。 |
| 23:05:49 | 在实际 G576 客户端的独立可写 cwd 上执行 CPU-only 的 mmengine `Config.dump` 合成配置烟测，记录退出码 0；**没有运行完整 AISBench 或发送推理请求**。该时间早于旧轮终态。 |
| 23:06:11 | 原冷轮按既有超时自然结束，终态记录退出码 124；控制器记录 `performance_validated=false`、`hot_submitted=false`，性能明细列表为空。 |
| 23:10–23:13 | 只读终态及进程复核：控制器和外层子进程不存活，G576 活动进程仅余 sleep；另一个 monitor 为僵尸状态。这个历史快照不保证之后的状态。 |

D 审计在终态后未发现 `*_details.jsonl` 原始性能明细；这不是断言不存在所有诊断文件或日志。历史 P 路由检查器只匹配 completion 接口的 POST，从 21:36 起扫描；后一次检查的最后已选日志时间为 **23:06:07**，匹配 POST 为 0。它不是所有路由/网络路径的证明，且两机时钟未校准，不能由此绝对断言零发送。旧 claim 因此继续保留为 **possibly-submitted**，不重放该 claim 或把它当作成功/失败的有效 benchmark。

## 2. 错误机制与修正边界

源端只读抽取的 AISBench `local.py` 在 `_launch` 中先基于 `os.getcwd()` 构造相对临时配置路径，再调用 `task.cfg.dump`；随后才会执行任务子进程。实际 CLI 错误记录与 source traceback 均指向只读文件系统；源码在 dump 前后还有清理逻辑，尾部 `os.remove` 同样报 EROFS，可能遮盖前一异常。这个证据支持错误发生在客户端准备任务阶段，而不是由模型推理结果导致；它**不排除其他路径曾发送网络请求**。

本机修复候选将每轮执行 cwd 绑定到独立可写子目录，并校验该 cwd 与同轮工作目录、BFCL 目录及挂载关系。可写 cwd 的真实 `Config.dump` 烟测只验证该狭窄文件操作；**修复后的完整 AISBench 启动、实际参数、服务请求和冷/热测量仍需后续独立真实试验验收**。源码身份和历史收据不等于该候选已在线部署或已经成功。

## 3. 来源与脱敏

[脱敏证据索引](../evidence/cold_cwd_incident_035.json)仅保留状态、时间、数值及 SHA；不收录私有 IP/路径、完整命令、输入输出或原始日志。下面哈希是先前只读采样中的源文件或日志哈希，不是此次重新访问远端：

| 原始对象的语义 | SHA-256 | 证据界限 |
|---|---|---|
| cold CLI 日志 | `78a94058942aa72bb5ed3ac571e19060e2e935782c2bd291488a6867c99317fd` | 日志有 EROFS 和 traceback，不等同完整推理输出。 |
| cold 命令记录 | `8d6e07f52c7e7052cb9115be32e42efc5345b2210036b5d5ae0af0adb2113dce` | 既有源端记录了只读 cwd；本仓不复制命令原文。 |
| 冷轮 terminal 记录 | `6d344db9b4fccec32d2a52390adc72f7b2af998e7047e65d3b6b7d28e54fe462` | exit 124、无 hot；独立原始测量与质量验收没有通过。 |
| AISBench `local.py` | `b80e07686c26669b4a7646164697a9d3e79420c48889654428311ea7ca343309` | 固定源的静态顺序：配置 dump 第 249 行，subprocess 第 260 行。 |
| P 路由日志 | `66e6e27a76725a9193175af3896b049c14dd9495399a3ca2e023da9d96959840` | 所选窗口和匹配规则下未观察到 completion POST；不覆盖所有网络路径。 |
| D 服务日志 | `3db89572f73ceb67f707618976b27d918184b356397abef195df03da08a0c942` | 历史快照的身份哈希；不能直接推出模型错误或性能。 |

事故发生时封存的执行器和控制器源 SHA、事故快照 SHA、数据/配置 SHA、P/D 健康与终态范围也记录在证据索引中。`Config.dump` 烟测的摘要是**先前 sentinel 输出的人手转录**，记录源脚本和合成配置 SHA，不是本仓内可独立重放的原始现场结果。

已有 [256K 原首轮路由失败](028-G316-first256-cold-router-failure.md)和 [256K→32 未计分诊断](033-256k-diagnostic-stage-evidence.md)属于不同运行；本次 G584 事故不替代或改写这些记录。

## 4. 离线核验

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/test_incident_035.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_archive.py
```

[回归测试](../scripts/test_incident_035.py)检查历史状态、hash 格式、源端报告的静态先后顺序、claim 不重放和烟测边界，并以合成错误状态验证拒绝逻辑。这只核验脱敏记录的内部一致性，不读取受控原始日志、不重新连接机器，也不证明跨机顺序或完整网络请求范围。
