# Qwen27 昇腾双框架对比地图：公开研究归档

**先下载：[面向读者的中文详解报告](request-flow/report.html)** · [GitHub直接阅读版](REQUEST_FLOW_COMPARISON.md) · [全部请求明细](request-flow/formal-requests.csv)。HTML为11章中文解读，将调度字段、图大小和算子记录转成流程说明、时间线及对比表；不再直接展示原始JSON。原 `request-flow/index.html` 入口也已替换为同一阅读版。

这是 2026-10-28 导出的研究快照，比较同一实际 Qwen27 W8A8 checkpoint 在 vLLM Ascend 与 SGLang Ascend 两套冻结部署中的表现。**地图与数据已经产出，完整机制验收仍未通过；本目录不是 Foundry 当前任务的 KEEP、优化采用或性能目标达成记录。**

## 从这里查看

| 内容 | 入口 | 范围 |
|---|---|---|
| 逐请求流程分解 | [直接阅读](REQUEST_FLOW_COMPARISON.md)、[交互页](request-flow/index.html) | 13 阶段组件比较、1,344 条正式请求、262 个诊断步、500 个算子窗口、原生阶段子集与 EOF 链路；缺测单列 |
| 原始技术快照 | [map/index.html](map/index.html) | 历史证据浏览器，保留原字节，含原始字段；阅读详解请用上方新版报告。原生事件明细入口关闭 |
| 完整文字报告 | [research/README.md](research/README.md) | 28 场景数据、消融、波动及适用边界；保留原研究的历史状态说明 |
| 采样与输出谱系 | [采样契约](research/SAMPLING_CONTRACT.md)、[输出谱系](research/SG_OUTPUT_LINEAGE.md) | 区分协议有效、argmax 路径、token 谱系和输出质量 |
| 诊断与地图代码 | [research/CODE.md](research/CODE.md)、[research/scripts/](research/scripts/) | 观察器、身份审计、后处理和地图构建代码；不是新推理优化 |
| 原始小表的公开副本 | [research/assets/](research/assets/) | 24 份 CSV；数字单元格保留，定位信息脱敏并规范换行 |
| 公开副本的逐文件校验 | [PUBLIC_EXPORT_MANIFEST.json](PUBLIC_EXPORT_MANIFEST.json) | 每文件公开 SHA、原始内容 SHA、变换状态、数字保留检查 |
| 导出与完整性工具 | [tools/](tools/) | 可复核公开文件、内嵌 CPU 分片、JSON 与 JavaScript 语法 |

阅读报告请下载后打开 `request-flow/report.html`，核心图表数据全部内嵌，不需要本地服务器或网络。`map/index.html` 作为原始技术快照保留，需要支持 gzip `DecompressionStream` 和 Web Crypto；不要把它当作新版阅读报告。GitHub 文件页本身不执行交互 HTML。

## 原理与已观察到的差异

实验节点为 aarch64、8×Ascend 910B1，每框架使用一组 TP4。正式矩阵为 7 个输入长度 × C1/5/10/16 × 两框架 × 每臂 N3，共 168 个正式波次，另有 56 个暖机波次；输出长度为 1024。主计分不启用 profiler，profiling 和诊断单列。

冻结部署的 28 个场景中，vLLM 的整波输出吞吐和端点 TPOT 均占优；SGLang 的请求 TTFT 中位数在 18/28 个场景占优，按均值为 15/28。例：512 输入/C1，vLLM/SGLang 吞吐为 53.3567/49.4354 tok/s，TTFT P50 为 0.274009/0.336138 s，端点 TPOT 为 18.4941/19.9239 ms。全部场景、定义和原始小表见文字报告。

这是两套部署组合的观测比较。CANN 9.2/9.0、卡组及 KV 配置存在混杂，**不能直接解释为纯框架净收益**。吞吐包含整波请求生命周期；端点 TPOT 不等于每个 token 的独立设备间隔。不同 rank 的累计通信时间不能相加为机器墙钟，HCCL 事件可能包含等待。

## 源码和可取用产物

精确上游 revision、文件与原始内容 SHA 由地图和研究快照给出。归档了 7 类诊断/交付组件，以及性能、消融、cache、GDN、HCCL、CPU lifecycle、采样和 token 谱系证据。Strata/DeepInfer 的来源、机制、前提和可移植性沿用已有研究，**不构成本机已验证的优化收益**。

诊断代码公开副本移除了私有路径。历史脚本中的输入 pin、源码 SHA 和运行记录描述的是当时的原始输入；脱敏后的文件会有不同字节，不能直接运行旧的固定路径命令或把旧 pin 当作公开文件校验值。当前公开字节以导出清单为准；真实服务/NPU复现需要另外配置依赖、模型及输入位置。

## 波动噪声

每臂每场景 N3，报告分别列出中位数、MAD、极差和配对结果。波内请求均值/P50与跨波统计分开。小差异须结合该场景波动解释，不能把单次诊断或格式检查通过写成性能改善。

## 结论与尚未闭合的部分

原始 204 个 rank 的导出、审计和归档完整性，以及最终地图/分发包的校验已完成。原版页面曾对两框架代表 rank 和最大分片进行 file/HTTP 选择核验；**不是逐个浏览 204 个 rank，也不是本公开脱敏页面的浏览器验收**。

仍待补证：

1. 同 payload 的短诊断输出差异及输出一致性解释；不同输出本身不自动证明质量失败。
2. 最后模型步、worker 产出、IPC、frontend、末帧到客户端 EOF 的完整因果链。
3. 跨 rank/stream 的完整关键路径、暴露通信及可归因的端到端优化空间。

地图中保留 `unknown`、未验证和候选状态；不能据此宣布全面优劣、正确性通过或任何新优化已采用。上述补证正在原任务中继续。原 `map/` 与 `research/` 固定为第 105 代已封存快照；新增 `request-flow/` 单列第 106–113 代已保存的局部诊断证据，逐项附来源 SHA 和边界，不将其升级为整体验收。新步表按原始顺序更正 SG mixed step5291 两个 RID 的 chunk 映射；原快照不改字节。

## 公开归档范围

本目录是新增的公开派生副本，不分配新的 R/Run 编号，不改变任何 Foundry 正在运行的任务、当前性能合同或正式结论。源研究里的编号只用于识别历史快照，不代表该结论在本仓重新验收。

服务器地址、绝对私有路径、私有仓库链接等定位信息已替换为稳定占位符；CSV 换行及日志/观察器的尾部空白已规范化。未分发权重、凭据、SSH transport、原始 SDK profile、原生 rank 事件分片和完整原版大归档。公开地图保留 204 个 rank 的索引/汇总，明确关闭未分发的原生事件加载；保留可脱敏的 CPU lifecycle 派生数据和内嵌分片。

原版完整分发包为 668585060 bytes，SHA-256 为 `7607725e3566b1abf24d66f2c4e57c1c9c871be7b89bc7877f064450e74f3796`。这是原版归档指纹，不是本公开副本的哈希，也不表示该包在本仓可下载。

本次校验记录见 [VALIDATION.md](VALIDATION.md)。
