# DSpark：aux/context、三层decoder、五步反馈与迭代末读

2026-11-09，loop-013。**Goal ACTIVE；每个替换原语逐位一致；本轮不是全模块/全链路穷尽证明。** 新NPU测量、推理请求、远端文件写入均0。沿源码建立以下范围后，才匹配历史取证和profile。

## 介绍

`target层36/37/38入口HC均值（checkpoint IDs 37/38/39） → SP全局aux → 按本轮有效token选择/拼接 → main_proj+RMS → 三份独立context Wkv/RMS/RoPE/cache → 3个draft decoder → collapse/norm/base head → 五次[前驱token W1 → W2 → BF16 Add → argmax/概率选择 → ID写回] → 五候选交target → target接受/拒绝 → 有效count/bonus/request映射 → 下一轮context、KV/history`。模型循环在每层Engram调用之前采快照；本次aux层与Engram层1/14不重合。

先从最终读者看，固定greedy draft只需五个ID；中间全词表分数可以成为更大producer的内部交付。三份context cache才是aux/main的实际读者，不能以RoPE/slot小helper预算剪掉aux/main或已无读者行之前的工作。decoder里Attention、MoE、HC复用目标模型函数，但role、8个本地expert/top3、query行域、权重和状态单列。

## 到了哪里

### 冻结实现和遗漏依赖补齐

[21源SHA只读身份](dspark-execution-identity.json)全部匹配当前运行；保存的service脚本SHA仍为`c2972011…`，TP8/DP2/EP16、DSpark5、enforce_eager、async scheduling，未改部署。另追到初始snapshot未收录的真实 [patch源码manifest](dspark-runtime-source/patch-source-manifest.json)：`patch/platform/patch_speculative_config.py` SHA `afdb32f3…`。

该patch的`_normalize_deepseek_dspark_draft:111–176`在post_init中调用，将private draft config改成 **128 experts、top3、3层**；target raw config仍为384/top6。当前profile对应router`3×5120;128×5120` FP32和DFC`8×5120×4608 / 8×2304×5120`，与此一致。不把checkpoint上的`dspark_*`字段当作已经传到算子的证明。

冻结服务选项、源默认和typed调用共同限定：parallel drafting=true，anchor默认true，Q=5（B4实际20个query，不是24），draft_sample_method=greedy，无vocab映射、dynamic spec method默认None、finegrained lmhead TP=0。metadata中使用的B、padding及query范围仍按实际调用传递；K=0/skip/dummy等合法分支单列。**这不是读取当前worker Python对象的直接dump**。

| 原语/交付 | 实际路径 | 对融合的约束 |
| --- | --- | --- |
| Aux | 三个pre-layer四路HC均值→BF16；SP每个checkpoint各AG一次；propose按token_indices选取再cat | 保存checkpoint顺序、request/token行对应、负index/重复语义、捕获地址和pre-Engram代次。不能借同H维就把均值换成HC collapse |
| Main | BF16权重`[5120,15360]`，TP8输出列640，gather_output后全5120 RMS | GEMM前后改变M/列/K顺序都需原语逐位门槛；不能直接吸收RMS或按checkpoint partial BF16相加 |
| Context | 三份BF16 Wkv`[512,5120]`及独立512维norm、RoPE与cache | 输入可共享，权重/norm/cache不能互用；每层query cache写回是context的后继writer |
| Draft decoder | 三层延迟HC、SWA/CP、query projection/cache、OA/OB、RS、128-expert top3 DFC与shared、HC后置 | 无Engram，无C1/C2压缩源。相同函数不代表沿用target40层的shape/预算/4核数值资格 |
| Base head | checkpoint `mtp.2.head.weight`存F32，加载后实际profile是BF16 GEMM `20×5120→16160`，TP8 AG补全129280词表 | 原dtype由运行调用确认；不能用checkpoint F32或目标head权重代替draft head |
| Markov | W1 BF16`[129280,256]` nn.Embedding；W2 BF16同形，`disable_tp=True`；AscendVocabPlan优先REPLICATED | **五轮W2没有五次原TP AG可删**。base的一次AG和W2复制计算分别计价 |
| 五步选择 | base logits只算一次；第i个前驱ID决定W1/W2；W2输出BF16，加法BF16，ArgMax输入BF16，写下一ID | 原BF16 W2舍入和加法舍入保留；不能把FP32 GEMM accumulator直接加base来绕过BF16边界。五步不是独立batch |
| Confidence | source定义FP32 ReplicatedLinear；checkpoint该权重存BF16，加载dtype不能从header外推；只有dynamic_spec不为None才调用 | 当前默认未启用，不能认领删除一份已不存在的confidence工作 |

`LogitsProcessor._get_logits`和Ascend override的normal/finegrained/PCP三路已读；当前normal的gather由**lm_head.tp_size>1**决定，W2 tp_size=1确实跳过。greedy draft完整分数在五轮之后无概率读者；概率draft、映射和dynamic路径各有额外读者，不能沿用当前窄ID协议。

### 合法分支和状态顺序

| 条件/范围 | 已沿实际代码追到 | 本轮证据深度 |
| --- | --- | --- |
| K=0 | base `_propose:1053`直接返回[B,0]，不跑draft模型；上层仍必须用正确shape更新占位/旧prob状态 | 源码已扫，固定五候选profile不提供该分支价格 |
| 超模型长度/skip | runner先保留next-token/valid-count，DP2仍让DSpark dummy_run完整Q组参与模型collectives，再发全宽零draft、清prob和req IDs | 源码已扫；“本卡无真实draft”不代表能跳EP或复用旧IDs |
| Padded输入 | valid mask排除-1和越词表ID，统计count，从最后有效前缀取seed；discard用backup token。prepare_inputs_padded保留原行数，另生成rejected数与选择位置 | count和seed是整数融合候选，仍需保留backup/请求重排/异步事件；已有metadata路线不重复记收益 |
| 非padded | 删除rejected后重建query_start、token_indices、seq_lens；aux按这些indices选取 | 行域不同，不从padded prefix外推bit/性能 |
| Context/query坐标 | copy_expand Part1复制全部context，Part2用`seq_len-rejected`算**物理线性slot**；RoPE位置另由最后有效context位置+1算；零commit用本request首位置，不能读上一request | 整个Triton helper已读；不能把position ID当物理slot，不能同launch跨program无join读Part2产物 |
| 多KV group/DCP | 每group用自己的block table/kernel block size/slot buffer；DCP还算owner和local slot，非owner置-1 | 当前DCP1，但源分支已读；不能把第一个group的validity/slot转成全局资格 |
| Greedy draft | 五次argmax串行反馈；full-vocab map为identity；最终ID writer仍有cast/ViewCopy/CPU发布 | 类型/调用顺序和8组成本匹配；不把base logits的argmax提前到Markov之前 |
| Probabilistic/mapping | probability需经request temperature等采样并逐step保留probs；reduced vocab在这条autoregressive分支回退greedy；prob stack顺序[B,K,V] | 源已扫；不是冻结负载，不引用当前greedy的价格/窄packet资格 |
| Dynamic spec | confidence预测由scheduler path读取，未来K可改变；当前None不执行 | 源已扫，完整动态状态性能无本轮证据 |
| Target验证 | `AscendRejectionSampler`先bonus采样，再独立索引target logits、FP32 widening、processor/constraints；greedy第一mismatch输出target replacement，全部匹配再bonus，尾部PLACEHOLDER | Python主分支已扫；原NPU reject helper/所有processor/RNG内部与终端head联合将在下一段继续，未声称采样模块已闭合 |
| Random/entropy/block/synthetic | random需完整target softmax、uniform、recovered token，one-hot greedy draft不等于greedy target；block/entropy配置默认false；synthetic+block非法显式报错 | Python分支已扫，随机kernel及logprob读者仍保留，不拿argmax替代完整分布 |
| 异步状态 | valid-count D2H有event，前一请求行序先保存；下一轮按prev mapping重排，校正乐观num_computed_tokens；CPU token表在async时写占位-1，不能作为生成token真值喂Engram history | runner边界已读；accepted-count事件和valid-sampled-count事件不是同一代次；完整kernel/CPU scheduler/output消费在下一段追 |
| 退役/导出 | draft后才finalize KV connector；cache写者/窗口/后续请求重排有末读，propose入口清旧draft_probs/req IDs | 保留条件；当前无迁移协议性能资格，不以cache窗口128证明所有旧物理行已可释放 |

### 按整段交付形成的候选

1. **完整score producer→五个ID**：base保留词表shard，W2按owner/group做，局部执行原BF16 Add和选择，只发布精确(value,index)/ID。本来复制的W2若分成g≥2会新增**五次相互依赖**的选择通信；g=1保留完整M/N可以避免重切原语，但多rank重复计算减少不等于关键路径缩短8倍。原tie/NaN、padding、column顺序、base头和各层不同权重都必须保存。此parent先整体定价，再决定不做其小变体。
2. **同内容W2 bias cache**：key是前驱token和W1/W2/precision/形状/格式/owner代次；只能缓存bias，不含request-dependent base logits。相同前驱可能跨请求/迭代复用；不能按position重合当命中。完整概率/score读者仍需原列序。
3. **Aux→main→三cache的内容/行资格**：先由真实接受/拒绝和未来cache读者确定仍需的context行，再向前扩到aux均值/AG和main。非因果SWA当前可见context为`[max(0,P-128),P)`，P必须是**校正后的prefix**；同一未中断request的下一prefix不回退，是可复用的条件。仍要逐层确认物理slot代次、peer/connector/排队读者、query最后覆盖、无效slot和重复store语义；不得仅以“老位置”裁剪。未知acceptance前可考察保守需求union，不能将其冒充已证明的实际eligible集合。
4. **三层元数据/缓存writer融合**：同registry key/group和positions共享一份cos/sin、同group/block/validity共享slot转换；保留三份原Wkv/norm/RoPE及cache输出。RoPE已原地改tail，后cat只在ABI/alias与contiguity允许时删除。这个小范围不含aux/main，不用它的低值剪第3项。
5. **提前main/大提交/megakernel**：完整同M的aux packet收集后可在target sampler之前做main+RMS，后续接受/拒绝相关cache更新仍延后；移动只计重叠，不算删除GEMM。draft source强制eager不是硬件“不可能capture”的证明；更大三decoder+score提交/融合需要实际host/device依赖、相同原语和静态地址/事件/DP参与证明，旧staged prefix结果不能代替此parent。

## 结果

### 价格与停止范围

[当前复算](dspark-execution-frontier.json)从accepted CP+slot+CPUplan原件绑定112个score period，每个1个base head、5个W2/Add/ArgMax，**8组typed成本calls/min/median/max全部匹配**。另绑定完整历史C8的231 waves、3632个rank-score记录，保留原clock证书/请求顺序/next-target frontier。

| 原score段全部免费情景 | C8 TTFT P90条件信用 | C8 makespan条件信用 | 八请求最大比例 |
| --- | ---: | ---: | ---: |
| Base+五W2+收集/反馈/选择/发布整个bracket | 0.1851–0.1948% | 0.9262–0.9529% | TTFT 0.1988%；响应0.9602% |
| 只让五W2全部免费 | 0.0413–0.0432% | 0.1875–0.1933% | 响应0.1944% |
| 只让base GEMM免费 | 0.0488–0.0558% | 0.2251–0.2368% | 响应0.2388% |
| 保留base GEMM，其余score bracket全免费 | 0.1314–0.1422% | 0.6922–0.7239% | 响应0.7297% |

六普通请求使用原verification次数和匹配B1/M5旧score call的**情景**，full-score全免为0.4258–0.8027%，W2全免0.1278–0.2488%；不是ordinary真实cache/peer新测量。于是**只改score parent、在这些固定schedule预算下独立追1%**的owner分组、W2 cache、layout/add/select、再向GEMM epilogue细拆均停止补证。包含全部GEMM的parent尚低，不再为同一预算补前驱命中trace或重跑旧小prototype。1%是旧排序参考，非用户硬阈值；换负载、容量、改变更大提交依赖不继承此停止结论。

不能把同为112个period、同shape/count当同一价格：首次把独立clean trace对UNIT表，duration断言失败；确认UNIT表来自accepted CP+slot+CPUplan后改用对应原件，8组才全部一致。独立clean保留另外的summary：whole score中位742.05µs，保留原六GEMM时其余全免最高period占比0.6224%；accepted组有一个1.7106% outlier。**不跨cohort相加，不把outlier或host gap当内在通信耗时**。[资格纠正](dspark-cost-cohort-qualification.json)

context三layer writer内的旧匹配C8估算：整个三panel span全免约98.29–103.97ms/18.59s（最高0.5593%），三writer保留原必需原语后可删任务全免最高0.3311%，六普通请求这两者最高0.5622%/0.3264%。停止相同预算的独立小writer开发。**这不含aux/main，也不含三decoder或更大host提交**；host首到末提交span全额传播的更宽情景可到C8约1.98%、普通请求2.91%，没有证据可实现全额信用，保留host/device依赖缺口。

三draft decoder历史64个matched bracket、32个unpaired bracket分开：把全部typed compute+HCCL免费，原rank-local敏感性P90信用32.04–37.91ms；只让三个DFC免费为27.20–31.26ms。原P90为1491.85ms，约2%量级，所以**不以score/context小段低值终止decoder、EP等待和大提交范围**。这不是可实现收益；F21含内部通信，peer迟到可能只是上游等待，相同TP rank不能随意被每个component轮换以制造净省。

### 相遇后的历史证据复用与限定

[限定复用包](dspark-native-evidence-reuse.json)保存原件路径、SHA和范围，未重测：

- 真W1/W2无nonfinite，W1全部129280行逐位distinct，因此没有可合并的相同W1内容class。B2/B3/B4所测真权重前驱上，M1/同M置换/重复/selected columns都逐位通过；只确认这些fixture，不外推当前服务format或所有M/N。
- 旧bias-cache all-hit一步约省19.84µs；mixed prototype慢26.13–27.17µs，零classifier成本也需约56.8–57.8%全hit步骤抵消。旧scheduler记录只有345个placeholder、可用前驱ID为0，不能由此估命中。全W2免费预算已经低，所以不为同一独立方案补trace。
- aux三均值packet/strided mean和**同M**main投影在六个旧fixture逐位0差。改变M的两个fixture有4/2个位差，但这些是**synthetic projection weights**，只推翻通用“行代数可交换即逐位”等价，不声称真main权重/大prefill形状已经失败。真实大parent仍需自己的门槛。
- context裁剪的原反例保留：S=512、c=1时正确cutoff385，用乐观target end算390会丢5行；c=0时不是所有rejected context都有Q=5 query覆盖。已有反例不重跑，真正物理eligible集合仍待确认。

### 地图分类

| 状态 | 范围 | 本轮判断与剩余门槛 |
| --- | --- | --- |
| 已确认可实现（限定fixture） | 原同M aux packet/strided均值、限定前驱的原W2 bias复用 | 历史逐位事实保留；未宣称服务集成、其它shape/format或性能接受 |
| 已停止 | 固定C8/六普通情景下独立score parent追1%的各小变体；独立三context小writer同预算；未改变的mixed cache prototype | 先全parent全免费仍低或已有native负收益；不再做同预算命中trace/小kernel；停止不外推大提交、aux/main或其它负载 |
| 待确认 | aux/main→三cache的live-row/内容资格、同M提前生产/packet调度 | 缺原M/format逐位、真实物理slot/全reader union、图地址和争用净价。大prefill潜在宽交付未按小writer预算剪掉 |
| 待确认 | draft三decoder/EP/HC与完整提交或megakernel | 128/top3/8本地expert已绑定；尚缺每层peer就绪归因、可消工作方案、同负载净价和每原语门槛。不能继承target M416 shared4/8资格 |
| 待确认 | target head→verify→异步输出→下一次输入的完整consumer closure | 本轮已追Python接受/拒绝边界、校正prefix和部分request代次，下一段继续reject kernel、processor/logprob/RNG、CPU scheduler/async writer全部末读 |

**下一处**：保持全图优先，从target最后HC collapse/norm与真实选中head行进入验证/异步输出和下一迭代输入，闭合当前末端未扫边；保留上述有量级parent的具体门槛。DSpark标部分扫描/部分判断，不把21源/8成本组/具名缺口当成“所有融合已找齐”。
