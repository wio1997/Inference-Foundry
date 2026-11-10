# O投影原生资源与SP残差FFN交付判断

介绍：从实际 V41CP O 投影走到 SP 归约、四路舍入残差和 FFN 双输出，判断少算、片上交付和通信重叠。沿当前代码独立判断；旧 OA/OB shared-Cube 仅在执行边界匹配后作为参考。每个替换原语必须逐位一致。

到了哪里：当前 Python/CANN/SDK 源码及编译描述分开记录实际审查行段，空行段只表示 identity。142 个 safetensors header 含 43 对 OA/OB BF16 权重元数据，payload 读取 0。两代表 rank 各绑定 516 条源一致的 OA→OB→RS→HcPost→HcPre→RmsNormCast 链，其中目标 M24 各480条，草稿20行另列，不能混用。源码顺序与唯一类型语法不是实际指针、事件关联证明。原大 prefill 的全局 carrier M3352/6536/6544 与 local CP 行数不同；cache 的6540也不能直接拿作 OA 行数。

实际 OA 用 ND BF16 `TransposeBatchMatMul`，perms102/012/102、factor1；再走原 unquantized OB。OA 的 BF16 舍入是活跃边界。model 的输出缓冲为 fullM，opaque op 写完后由 decoder 做 SP RS；把返回 shard 的 MMRS API 原样塞入会同时破坏缓冲/行域合同并重复 RS。已有 Kimi BF16 adapter 仅 pure-prefill、`ai_cpu`，没有被当前普通 wo_b 路由选中。查到的 AIV V2/shuffle-K 源不能冒充这个 adapter 的实际 backend。OB 的 MatMulV2 BF16 已在原 trace 中出现；编译目录未找到匹配名不证明 BF16 不支持。

结果：整段 OA→FFN norm 的原目标窗包络为6.6681%/6.7856%。保留 OA/OB/RS，仅其它部分半省且每次付5us新增暴露成本，条件净省1.0770%/1.1009%窗口。Post→Pre→Norm 自身半省/h5为1.0014%/1.0207%。这是两个配平镜像负载短窗的敏感性，不是原 C8 请求预测；不套用另一839ms阶段采样的分母。全部144价格情景保留。HCCL aggregate/helper 与 AIV 实际区间不重复加价，未显式记录的 output copy 不算已测到0成本。

**已停止**：擦掉 OA BF16 舍入的普遍权重预合成。有限 FP32/BF16 RNE 公式例子 `x=[1,2^-8],Wa=[1,1],Wb=3`，先舍入 OA 后得 BF16 `0x4040`，预合成得`0x4041`。这是公式反例，不是 NPU/模型实测。实际尺寸预合成还增加到2.222倍逻辑 MAC，权重操作数40MiB对18MiB；不将逻辑流量等同 HBM/新增显存。

**已停止**：当前M24下，一个 AIC 独占完整1024列 OA 舍入 panel，再完成全部 OB 列的具体 schedule。读取旧 profile 主机侧 DB 时，首次误用容器路径得到空结果，已判 INVALID 并保留；修正后两个 DB 共6条 AICORE_FREQ 都是1800MHz。允许64KiB权重已在L0B的乐观模型，下界81.6356us，仍高于全部480对/每rank的63.38–74.779us。逐调用免费回退 oracle 信用0，无须重跑同一小M配方。原双 kernel 之间单独空档也仅11.74/11.299us每窗，gap-only小方案停止；不据此停止 persistent kernel 减少其它实际开销。

**待确认**：按 OB 的 N 分片，每个 owner 重算完整 OA，保留原 BF16 panel 后做原 K 顺序 OB。它与已停单owner几何不同，不需要跨AIC共享私有L1，也避免用新 OB 部分和改变关联顺序。24owners 的理想局部下界38.4us，但总逻辑权重操作数202MiB，达到该下界需约5.516TB/s聚合服务；不能把片上少搬当免费。h5理想净省1.2141%/1.2718%窗口，108个资源成本情景含负收益。仍缺实际 L2/fanout 服务、全部 buffer/flag、原 OA/OB 活跃 key 与 K 累积/转换顺序、逐原语位及新增成本。先保留具体缺口继续下一交付，不扫 NPU 参数。

SDK 允许2201的FP32 L0C→BF16 L1 NZ，noChannelSplit；这只是合法指令路径。OA panel 是 OB 的左操作数；权重 L1→L0B 约束不能换成左操作数带宽。[官方2201架构说明](https://asc.gitcode.com/guide/programming_guide/advanced_programming/hardware_implementation/architecture_spec/npu_arch_2201.html)支持私有L1、512KiB减256B可用、L0A/B64KiB、L0C128KiB及右操作数128B/cycle。旧模型4096MAC/cycle另列为参考假设，未由该网页核实。合法SDK、静态容量和候选成本均未证明原始 Fixpipe 位一致。

**待确认**：大prefill完整权重/原M下的多owner和交付，以及 SP→HcPost→HcPre→RmsNormCast 更大融合。HcPost 舍入后的四路残差仍供 FFN Post，RMS 的 BF16及FP32路由输入都活跃，不能只保留归一化输出。下一段沿实际 HcPre V3、HcPost、RmsNormCast OOT/native 追到归约、cast与所有读者，再价较大parent。普通负载、草稿/空行、FULL图及collective顺序仍须分别验收。

这些成本情景互为不同假设；O投影、HC提前交付以及既有QKV/head/frame候选有共同依赖和重叠范围，百分比不能直接相加。

**已确认可实现**：本轮新替换原语0。全链路未扫全，源码数量与具名缺口不作为穷尽证明。仅只读当前源码/旧profile/权重header与本机标量建模；新NPU工作、模型进口、请求、采样、服务动作、显式远端写入均0。

证据：`o-projection-native-{source-review,source-excerpts,frontier,catalog,dependencies}.json`、`o-projection-fixpipe-dependencies.json`、`o-projection-parent-price.json`、`o-projection-resource-ownership.json`、`o-projection-saved-clocks{,-a1-invalid}.json`。完整逐链和情景保留，不以摘要替代。
