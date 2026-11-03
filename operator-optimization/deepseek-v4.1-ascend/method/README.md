# 算子融合机会与 NPU 利用率诊断包 v2

以 v1 的理论上界方法为基线，主流程调整为：先扫描两两、三个、更多算子融合与拆分重组机会，记录覆盖和停止原因；再按模块分析 NPU 利用率，找到同口径下最低的模块，固定进行实际性能与理论极限对比，再结合差距分析原因。当前阶段只找机会、定位问题，暂不实现或设计解决方案。

候选结构只在分析图中表达；利用率对应已有实际实现。理论极限对比是固定环节，缺少参数时保留符号或缺口；缺少硬件参数不阻止扫描语义上的融合机会。此包提供 Agent 工作说明、模板和复算工具，尚未包含自动图改写器或 profiler 数据解析器。

本包通用，不包含用户的权重、W8A8 调查材料、服务器配置、连接方式、凭据或真实性能数据。教学数字全部虚构。

## 怎么使用

1. 将整个 ZIP 解压到一个新的独立目录，不要只复制 AGENTS.md。
2. 让执行模型读取 [AGENTS.md](AGENTS.md)。把 [START_HERE.txt](START_HERE.txt) 整段发给它，补充目标图/模块范围、源码版本、shape/dtype、硬件和已有 trace 路径及可读范围。未知字段可以不填。
3. 模型在 runs/新任务名/ 创建独立任务，复制 templates/ 中的文件，并将 workflow-state.json 命名为任务 STATE.json。先冻结原图，再扫声明范围内的融合/拆分重组机会；记录条件候选、覆盖缺口和停止原因后进入模块诊断。
4. 最后看 REPORT.md、fusion-scan.json、module-analysis.json 和 records.json。报告明确最低利用率模块、指标口径和原因证据；无数据时保留未知。固定交付性能与理论极限对照及 bounds-input.json，有数值复算时附原始结果。

ZCode、GLM 或 DeepSeek 可以执行。主控负责固定任务与审核；执行模型一次完成一个小任务，如检索一项资料、提取 shape、整理一条调用链、复算一个模型。不要给它一句“无限优化直到最快”。

## 文件对应什么用途

| 文件 | 用途 |
|---|---|
| [AGENTS.md](AGENTS.md) | 模型必须遵守的目标、边界与禁止项 |
| [START_HERE.txt](START_HERE.txt) | 可直接复制给执行模型的启动话术 |
| [PROTOCOL.md](PROTOCOL.md) | 每一步做什么、留下什么产物 |
| [FUSION_SCAN.md](FUSION_SCAN.md) | 两两/三算子/多算子/拆分重组的扫描、候选与停止规则 |
| [MODULE_DIAGNOSIS.md](MODULE_DIAGNOSIS.md) | 模块利用率定义、最低者排名及原因定位 |
| [PROFILING.md](PROFILING.md) | 短窗口采集、模块标记、离线聚合和大文件分层读取 |
| [OPERATOR_MODEL.md](OPERATOR_MODEL.md) | 必要计算、访存、依赖、争用、流水与融合的建模方法 |
| [REVIEW.md](REVIEW.md) | 检查扫描覆盖、利用率归属、原因证据与上界推导 |
| [CHANGELOG.md](CHANGELOG.md) | 本次相对 v1 的流程变化和范围 |
| templates/ | 合同、候选/证据记录、扫描覆盖、模块分析、任务状态和报告 |
| examples/ | 完全虚构的复算示例和固定 tile 的适用范围 |
| reference/ | 保留的理想上界理论、框架与自主补证说明；执行顺序以当前 PROTOCOL 为准 |
| tools/ | 无第三方依赖的 Python 3.9+ 复算和打包工具 |

## 先用教学例子检查使用方式

在解压目录执行：

```sh
python3 tools/calculate_bounds.py examples/toy-input.json
```

应得到三个阶段的时间下界 2、3、4 秒，以及吞吐上界 50、100/3、25 个结果/秒。它们只属于虚构问题，不能套用到真实芯片。工具只检查输入结构与精确有理数计算，不检查物理事实或必要性证明。

自己的任务复算：

```sh
python3 tools/calculate_bounds.py runs/你的任务/bounds-input.json > runs/你的任务/calculation.json
```

若缺参数，输出保留符号与已知部分的界。工具不会自动宣称结果已获理论证明。包装校验 PASS 也只表示文件可读、示例算术一致、归档哈希正确。
