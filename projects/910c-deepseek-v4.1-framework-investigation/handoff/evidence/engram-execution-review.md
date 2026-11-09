# Engram：两层内容生产、残差交付和代际

2026-11-09，loop-012。**Goal ACTIVE；这是一次限定源码前沿推进，不是全 Engram 或全链路穷尽证明。每个替换原语逐位一致。** 新服务请求、NPU 测量和远端文件写入均为 0。

## 介绍

从 `model.py:1565–1591` 两个实际残差消费者反推：

`V1 prompt lookback / V2 device history → request index → V1 slot write → 两层 hash → 本地共享 UVA lookup+decode → TP heads 收集 → SP 行 → BF16 Wkv → rotated FP32 gate → BF16 四路残差 → 本层 HcPre/Attention/HcPost`。

层号是零基 **1、14**。DSpark aux snapshot 在 Engram 修改残差**之前**。两层表和 Wkv 不同；相同 head ID 不证明整行 24-head tuple 相同，更不证明两层可以共用投影。

本轮边界由上述代码独立得到，再与历史成本、失败变体相遇。原 `S04`/`M14` 的覆盖标签和旧候选编号不作为本轮完成依据。

## 到了哪里

### 冻结身份和实际表示

[只读身份](engram-execution-identity.json) 对 **19 份本地/运行源码 SHA 全部匹配**，另外只读取回上游 `nvidia/engram.py`，只采用本分支导入的 `engram_head_shard_rank:44–55`；不把其 CUDA allocator/lookup 当 Ascend 实现。原脚本 SHA `c2972011…` 仍匹配：TP8/DP2，CPU offload/shared memory 均 true，DSpark 5，FULL_DECODE_ONLY，未显式启用 Engram overlap。`ascend_config.py:573` 在 A3 上默认 false。

| 交付 | 冻结事实 | 对边界的影响 |
| --- | --- | --- |
| Hash | max ngram=4、8 heads/order，共 24 列；compressed vocab=99092；层1/14有独立 multipliers/primes | 查表/投影缓存必须按有序完整 tuple 和代次区分；共享的是 token history，不是两层结果 |
| Codes/scale | 两表 INT8 `[384006168/384016682,256]`；checkpoint scale E8M0 `[rows,8]`，加载到 CPU FP32 | 既有 lookup 已融合 ownership、UVA gather、group32 decode 和 BF16 cast；不能再认领一次“首次融合” |
| head owner | 共享 DP 模式令 table.dp_size=1；TP8 每 rank 3 heads | 当前没有每步 DP hash gather / DP row AllToAll；这部分收益为 0。非共享分支另有义务 |
| Wkv | 两层 quant description 为 FLOAT；BF16 `[25600,6144]`；原实际 MatMul 输入/输出均 BF16 | 模型名 W8A8 不把此投影变成 INT8；不信用 7/8 投影计算消除 |
| Gate | Quarot map 指向 optional/quarot；文件 F32 `[5120,5120]`，loader 验证重复 block32；实际 profile 的 block buffer 经 BF16→FP32 cast | A3 无 ENGRAM_UNROTATED_GATE，不能直接接现成 A5 gate |
| 残差 | hidden `[M,4,5120]`；Wkv K/V 分成 20480/5120；eps=1e-20 | 保存 BF16 Wkv 舍入、FP32 逐操作及最后 BF16；残差后续仍有 HcPost 读者 |

`AscendUnquantizedLinearMethod.apply → unquantized_gemm → F.linear` 已读；加载后可能 `maybe_trans_nz`，本轮 typed profile 不鉴定当前物理 format，不能据逻辑 BF16 形状外推 vendor tiling 逐位不变。

### 实际生产、分支和末读

| 范围/分支 | 已沿代码读到的义务 | 本轮状态 |
| --- | --- | --- |
| V1 主路径 | runner `3169–3241` 从完整请求/物理 block table 取坐标；CPU lookback 只给 prompt，异步生成占位由 slot cache 回退；hash request search 一次供两层，先独立写 slot 后 hash 读 | 已扫源码；生成/回滚频率不从形状推断 |
| History | 当前 chunk → 有效 lookback window → slot fallback；image/dead 阻断后续 ngram，负位置 padding；重绑 KV cache 丢弃旧 history。V1写 slot 和结果 cache hit 是两回事 | 已扫源码；任何结果复用仍保留历史写/代次 |
| 无 Engram、未绑 cache、force_dummy、空请求/0 token | disabled 发空结果；共享表在不 hashing 时零输出/零 mask，不查表；非共享的 dummy 仍进共同 collective；0 token lookup不写 | 已扫源码；不能把“空 query”替代这些条件 |
| Hash→lookup | 两层一次 hash gather；切层 ID slice 后 `.contiguous()`；预零局部输出；lookup 对每个本地 head 都写 owned 或显式 0；UVA 指针表 CHUNK=2^22，ID int64 后算地址 | 已扫源码；只可删除被 lookup 完全覆写的当前区域预零，不能删 table 初始化/尾部 padding |
| 共享/非共享 DP | 当前共享映射按 TP shard 保留一次物理表，各 DP 查自己的 token；非共享 TP-major×EDP head owner，hash pad 到共同 slot，AllToAll 回所属 token、再 TP heads AG | 已扫源码；非共享分支不能由本地 cache hit 跳 collective；固定拓扑改变需新合同 |
| TP/SP | 当前先收集全部 token 的 heads，再按 `sp_shard` 零补齐/取行；SP的 mask先等 ready，再切行；lookup等各层 ready 后才SP切行 | 已扫源码；owner transpose可缩交付，但原 padding、head 顺序、静态地址和所有参加者保留 |
| V2 slotless/PCP | 读设备 all_token_ids 的 depth3 window；PCP 先对连续请求 hash 再选本地 token；FULL captured producer要求 slotless、无SP、PP1、table DP1/TP1 | 已扫源码；**TP8/SP 当前不满足 graph 内 producer 资格**，不能拿 V2 TP1 路径代替当前 prefix |
| 外部 event | FULL按 descriptor/bucket 用 ExternalEvent wait/reset；非FULL不 trace 外部wait；mask早发，每层 rows在lookup/collective/padding完成后 ready | 已扫源码；改两层 packet不能把层1无谓等层14，必须重新定价 deadline |
| 复用/退役 | prepare stream先等main（输入更新/上轮读者），record_stream保活；V2 finish含异常join/reset；captured上下文finally join；关闭UVA先同步再unregister，失败保留可重试的mapping/alias | 已扫源码；不能只按指针相等复用代码/scale或投影 |
| V1 显式启用 overlap 后 forward 异常 | prepare提交异常本身有 retire；所读 V1 `_model_forward.finally` 只release metadata executor，未见直接Engram retire；全仓 literal caller只在V2 finish/prepare异常/模型转发包装 | **待确认**：异常后输入复用前的完整join链；当前默认同步路径不借此宣称有bug，也不据V2外推V1 |
| reload/部分关闭 | shared leader只写一次并汇总错误；codes/scales分别close成功后才撤自己的alias；BF16/INT8/MXFP8/requant分支依源dtype；preflight必须在分配前检查同一index的成对scale | 已扫源码；不是运行中原子热更新协议；衍生cache另需完整代次 |

### 数学化简、共享和大融合的合法边界

原 gate 是 `original=FP32(hidden)·Rᵀ`，分别求 original/key 的 FP32 RMS，再按 **`(original * (FP32(q)*FP32(k))) * FP32(key)` 左结合** 求 dot；之后 rstd、dim^-0.5、abs/clamp/sqrt、signbit、sigmoid、mask、单独乘 V、单独加原 hidden、最后 BF16。

- **可复用常量/同次 cast**：每层 q*k FP32、FP32 rotation 可按权重代次准备；同次 hidden.float() 可供 rotation 和 residual，延长一次寿命。历史 native `ENGRAM-ROTATED-GATE` 的 M3/M15 full/mixed/IEEE fixture已逐位通过。它只证明所测原语重用，不证明全模型/其它shape。两层常量约164 KiB；原成本很小。
- **不接受纯代数放行**：把 channel_weight 预乘到 key、把旋转吸收到 Wkv/q/k、按 head 分解线性/Gram 范数、对 K 轴 partial 后相加，都会改变原舍入/归约或 BF16 Wkv 边界。没有每原语逐位依据，不能归入已确认；这不是“所有同目的实现不可能”的证明。
- **内容复用切口**：可存原 BF16 KV+key rstd，每行51216 B，或完整 widened KV+rstd102416 B。hidden rotation/RMS、左结合 dot、gate、mask与残差仍实时做。key须包含有序24列、表/code/scale代次、Wkv权重和数学几何/精度、epsilon、owner/bank代次。只知道相同 position 或单head命中不够；miss压缩改变M也不能默认逐位。
- **Wkv提前生产**：rows已就绪时可先做保持原 M/N/K/format 的 Wkv/key RMS，层14deadline比层1宽；它是原计算重排，不是删除计算。需计与早层 AIC/HBM/共享专家的争用、所加 bank、mask/ready 和最后reader；不能把全部 kernel时间信用成暴露净省。
- **gate→HcPre megakernel**：可在寄存器里把最终BF16残差喂给HC，但 `DecoderLayer.forward:895 residual=hidden_states` 随后的 Attention HcPost 仍要这份四路BF16。允许消掉重复cast/中间FP32面，不允许把最终残差整份write/read也一并算掉。要消这份持久化必须把Attention到HcPost一并纳入，新的生命周期/资源范围不能借小gate预算宣称完成。
- **mask false**：mask在非线性后才置0；仍保留 `0 * V → + hidden → BF16` 的IEEE结果，NaN与有符号零不等于直接passthrough。历史 native inactive passthrough已有8个位差；不复跑此原样配方。裁剪K/V投影还需原 Wkv各输出原语门槛、原M/N路径和额外launch/权重成本。

## 结果

### 匹配负载的成本与净价

[复算结果](engram-execution-frontier.json) 和[带类型的128窗口](engram-execution-windows-archive.json)：64个首tokenbody，两层各一Wkv+37 gate tasks，共4736 gate任务；8组首tokenWkv和2组当前captured-M3成本与独立 `PROFILE-UNIT-COSTS` 的 calls/min/median/max **10组全部一致**。当前112个decode period单列，不与首token相加。

| scoped 原工作全部免费 | C8原调度下最高请求 TTFT 条件信用 | 判断 |
| --- | ---: | --- |
| 两层Wkv | 0.615615% | 停止单独1%路线的小切分/替换预算；保留联合producer和容量目标 |
| 两层完整gate | 0.489683% | 停止同预算下单独gate小方案；真实保留数学后只会更少 |
| Wkv+key/value独立子链全部免费 | 0.710652% | 不为同预算、readiness不变的独立tuple cache收命中trace；更大交付/排程范围另算 |
| 常量+重复hidden cast | 0.037375% | 已有逐位可实现子动作，价值不足独立开发 |
| Wkv+完整gate+两边copy | 1.115388% | **不是全Engram上界**：不含lookup/TP/host；至少实现约89.65%信用才触1%且未计替代，余量很窄 |
| 上一项+紧邻HcPre | 1.207905% | 仍保留必要残差；全免费门槛约82.79%，不是megakernel预计收益 |

这些是原 observer cohort、原 EP 到达区间及请求顺序固定、给全TP组最乐观rank信用的模型；不是改调度的通用上界，也不是服务收益。1%只是已有排序参考，不是用户新增硬阈值。不把相互重叠的表格行相加。

TP owner delivery 的逻辑 payload有实质范围：M3每层每rank远端接收 **258048→32256 B**；M818为 **70361088→8795136 B**。原 full feature bank对应M818约80.41 MB，SP owner仅10.05 MB，但需要新的通信、重排和固定地址发布。旧匹配负载 pricer 在 tuned calibration情景下，C8 owner-only makespan条件信用 **6.69–38.87 ms / 18.59 s（约0.036–0.209%）**；default情景可为负；跨rank clock区间使更乐观端不稳。六普通请求的tuned owner-only约 **0.427–0.929%**，default约0.323–0.726%；T6与T24敏感性不能混为真实预测。联合input+gate+external提交的tuned情景可到0.968–2.073%，包含词embedding和host工作，属于更大input/iteration范围，不能记成Engram单项收益。

历史原样 overlap C8未建立稳定净益（相对前/后对照+0.718%/−1.077%，仅文本对齐不是原语逐位），原 staged prefix两组native配对host净省中位0.374/0.339 ms，均未过预冻结0.5575733 ms门槛。**停止这两个未改变配方的复跑**；不关闭改变交付/消除staging/联动Wkv的更大路线。

### 新发现：INT8 codes + 保留 E8M0 scale

现有 `storage_dtype!=BF16` 把 INT8 的 scale_dtype固定为 FP32；原checkpoint已经E8M0，无需重新量化codes。若将code类型与scale类型选择分离，UVA lookup读1字节scale并位解码到**原加载FP32位值**，可以保留原 `FP32 code * FP32 scale → BF16` 的交付。

两表合计768022850行，scale由 **24,576,731,200→6,144,182,800 B**，物理共享主机容量净少 **18,432,548,400 B = 17.16665 GiB**（未计小元数据）。DP共享只算一次；不是再按16rank乘。单个被请求的256维row逻辑 code+scale由288→264 B，最多是此逻辑payload的8.333%压缩；重复scale load、cache line、PCIe/SIO和UVA时延未测，不当作8.333%延迟收益。

[冻结Torch CPU穷举](engram-scale-representation-cpu.json) 的决定性差异：现有MXFP8 `_decode_e8m0` 对255输出 `0x7fc00000`，原加载E8M0→FP32是 **`0x7f800001`**。所以“直接复用现成decoder”不满足FP32位值合同，归入已停止的具体配方。修正255后，256个编码FP32位值0差，65536组signed-INT8×scale→BF16 CPU结果0差；这不是NPU替换验收，不能据CPU结果宣称整表已改好。

### 地图处置

| 状态 | 项目 | 具体结果/下一门槛 |
| --- | --- | --- |
| 已确认可实现 | 常量准备和同次hidden widening复用，限定旧native M3/M15 fixture | 复用已测逐位事实；价值低，不单独开发/外推其它shape |
| 已停止 | 独立gate、Wkv、readiness不变tuple-cache的同一1%预算小方案；原样overlap/staged-prefix；mask passthrough；直接沿用MXFP8 scale decoder的INT8新路线 | 分别有匹配条件预算、native价值失败、IEEE位差、编码255反例；停止范围不升级成全Engram否定 |
| 待确认 | **保留E8M0的INT8 UVA representation** | 已有确定容量差和穷举CPU位门槛；缺NPU原lookup逐原语位值（全256编码/INT8值、无owner/负ID/零行、chunk边界、NaN/次正规数）、成对加载/共享映射、reload/close和图地址；纯延迟目标还需匹配真实UVA净价，勿先跑C8 |
| 待确认 | TP head→SP owner直接交付，与input/iteration提交联合 | 原矩阵不改M/K/N；缺真实新transport+重排代价、producer deadline、全参加者/图地址/失败退役；原body预算不得剪该parent |
| 待确认 | 提前Wkv/key RMS、两层共享调度、post-Wkv内容复用 | 缺完整有序tuple可重复比例/共识协议、原M数值路径、原资源争用和bank代次。只为改变去留的联合方案补证，不为已低值的独立投影cache收集trace |
| 待确认 | rotated gate + HC大producer | 保存原rotation/归约/每原语结果与BF16残差末读；缺具体tile/资源布局和实际替换净价。A5现成kernel不是本机资格 |
| 待确认 | V1显式overlap异常退役 | 当前默认false，不阻塞扫描；缺forward异常后输入/lookup复用前的直接join链证据 |

**下一段**：从 `model.py:1559` 的 pre-Engram aux 消费者进入 `dspark.py`/spec proposer，沿真实5候选→target verify→接受/拒绝→cache/history与下一迭代读者，不按历史模块标签继承完成。输入/输出和跨迭代整体交付仍未独立闭合。以上待确认项保留具体缺口，先推进剩余真实路径。
