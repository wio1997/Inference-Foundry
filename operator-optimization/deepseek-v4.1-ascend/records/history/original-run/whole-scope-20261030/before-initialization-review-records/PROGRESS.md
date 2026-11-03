# 整体优化机会分析进度

**整体未完成，目标active；尚无完整模块通过八类覆盖门槛。** 不给缺少分母依据的完成百分比。旧报告、历史complete回执及本次检查点PASS都不代表整体验收。

本轮完成的实质工作：

- 固定两库commit并保留566份去重源码，建立版本化站点索引；较早558文件索引原样保留。最新索引含7479类/函数、20291控制站点、39948调用、7772导入，全部只是定位底账。
- 建立26个暂定pipeline审阅面、260个变换规则槽和每模块8类门槛；范围仍扩展，不以四节点枚举限制完成范围。
- 对输出头/拒绝采样/DSpark的23个选定分支补充局部效果合同；区分源码事实、配置推导、实际观测和未知，没有关闭整模块。
- 核对同次16rank原CSV的hash和8类词表形状/dtype族，修复FLOAT/BF16 ArgMax混选；未以kernel时长之和宣称关键路径收益。
- 纠正DSpark导入路径误读；补真实Markov/Confidence头及采样参数源码。按固定启动/请求合同细化greedy、entropy/block、动态confidence、词表分片与A3 cache适用范围，历史resolved对象仍待证。
- 细化WS-VOCAB-01、WS-REJECT-01及按需求拆分bonus的WS-BONUS-01；补F27–F31跨位置/层/卡/迭代重组约束。候选存在重叠，未实现、未实测模型收益。

当前决定性缺口：

- 实际resolved模型/头/采样配置与历史同调用绑定、传递依赖和primitive语义闭包
- 全部相关分支及有依据的不适用排除、全部消费者/alias/状态/跨流跨卡跨层跨迭代边
- 完整请求关键路径、worker准备与device真实因果关系、通信组/消息/事件身份
- 同调用MoE专家有效行与容量、内部阶段；Indexer真实长度/mode；完整Engram host/UVA代价
- HBM/L2实际字节与适用硬件容量上限；可证资源账、候选重叠/冲突和范围内排除依据

下一步：继续M00/M01/M05的实际初始化与范围闭包：按真实import补generate/base、serve/engine及HF generation fallback，核实aux层ID映射、wrapper图像驻留与logits全消费者；继而补M04/M21/M24的版本化状态和同次因果边。并行概念槽不代表并行Agent；不启动目标优化实现或无目的重复采集。

读取顺序：[STATE](STATE.json) → [范围门槛](SCOPE-AND-GATES.md) → [模块台账](COVERAGE-LEDGER.md) → [输出头审阅](TAIL-HEAD-SAMPLER-REVIEW.md) → [DSpark合同](DSPARK-CONTRACT-REVIEW.md)。原始数据、失败版本、错误推断修订前记录和历史目标回执全部保留。

当前研究仅只读取证和离线分析；未修改目标代码、重启服务或发送新的推理/NPU负载。没有外部阻塞，可继续推进。

追加跨迭代实质审阅：已按冻结源码展开19条局部版本化状态边，覆盖拒绝输出、next token/count、seed、持久draft view、side-stream复制、CPU镜像和下一轮device修正。见[局部状态图](CROSS-ITERATION-REVIEW.md)。WS-NEXT-01为条件性重组，未测收益；全部消费者/真实事件仍OPEN，0个整模块关闭。
