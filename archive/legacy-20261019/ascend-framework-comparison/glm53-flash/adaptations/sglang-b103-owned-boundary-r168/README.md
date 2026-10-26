# Flash SGLang B103：R168 有界 PD 适配检查点

固定 c5cf 镜像、TP16/1P1D、eager、无 MTP/Graph/Radix，context4096、计算chunk128。overlay44个文件全部与实际 R168 preflight/import/hash绑定一致。相较R122只变4项：新增独立存储helper，修改indexer编排及两个ABI观察器；compression/tailstore/score整个kernel文件未变。

D 对12个实际输入建立独立紧凑缓冲，用原kernel压缩并立即打分，只提交实际写入的cache/tail行。原SDK注册区与owned scratch真实地址分别校验，不能伪装为同一指针；保留Req/Mamba/page所有权、数值/状态/SDMA字节及writer-first安全关停。

R168实测：4组真实completion reversal、8对PD/16HTTP200，完整IDs/text/EOS/repeats一致，selectedLP<=.25；2261755392B真实SDMA，32SDK关闭、320有序shutdown阶段、双方原生0/idle8。详见 qualification-summary.json。R166真实快照CPU控制另证明选择性提交保留并发邻居写和异常不部分提交。

边界：计算批次C1，只有两个D目标同时保留；不是任意服务、80K600、Graph/MTP/缓存/多模态/性能验证。全cache复制和CPU观察仅用于当前有界正确性定位，**不得作为性能实现**。它证明一组存储边界修复可用，尚未唯一识别原12个输入中的物理格式/alias/注册/生命周期根因。后续需将ND/所有权约束落到持久pool及小输入边界，消除每步全cache复制并重新验收。

保持当前目录不可变；新改动建新candidate/run。原始输入、业务文本、模型权重、tensor快照、日志archive均未发布。本包提供完整overlay与实际配置/SDK身份/证据摘要，受保护的测试输入只按原SHA在本地提供；不允许重放历史once launcher。
