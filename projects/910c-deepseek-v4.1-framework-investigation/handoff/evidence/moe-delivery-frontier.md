# D-M1工作断点：FFN双输出、真实DFC与共享分支

2026-11-09。**本段在扫，尚未闭合；每个替换原语必须逐位一致。** 本文不接管DSpark的8本地专家分支，它另有行域/拓扑；此处先核目标40层的实际384/EP16=24本地专家、TopK6。

## 介绍

从`RmsNormCast`的两个真实输出反推：FP32→内部gate→prepare/padding→sqrtsoftplus TopK6→DFC；原BF16→同一DFC及独立共享专家。最后两支组合→FFN HcPost，再交付下一层的residual/premix、Engram/aux等指定读者。不能只按norm或TopK小预算剪掉整个MoE。

## 到了哪里

- `DeepseekV41MoE.forward`把原FP32 norm输出作为`router_logits`传给内部router；`_prepare_router_input`只有在已precast `weight_fp32`时直接用它。专家和shared消费原BF16 norm舍入值。不能把BF16转回FP32充当原router输入。
- 当前配置`enable_shared_expert_dp=True`、SP、`mix_placement=False`、`multistream_overlap_shared_expert=False`：共享支为SP+权重复制，原shape直接在本地行计算；无shared输入TP AllGather/输出RS。不能根据generic shared TP分支多记一笔通信节省。
- 实际`enable_fused_mc2=1`回到自有`dispatch_ffn_combine`，不是把CANN MegaMoe的接口参数套进来。target专家权重INT8、FFN内部宽4608→2304→5120。完整MoE的缩放/ForeachAdd和残差还要追到实际数值/状态消费者；尚未逐项归约。
- 当前C8原窗口有三种不同的行数：例如DP0/carrier818，norm/router/shared M419，DFC M818；轻侧可norm/shared M3而DFC M816/817。DFC的padding/active mask与EP远端服务要保留，不能让共享支也按padded DFC M扩大。
- `moe_gating_top_k_hash`虽名字含Hash，normalizer的`num_hash_layers=0`只排除tid2eid查表。新核对的config类把root `vision_config.num_hidden_layers=32`展平为`vision_n_layers=32`，于是bias_vl仍存在，纯文本也走input_id对齐和sentinel判断再进入该算子；原profile正好有非空input_ids/bias_vl输入。不能直接删全部input_id依赖。[config源码](moe-runtime-source/deepseek_v41_config.py)及[只读SHA](moe-hf-defaults-readonly.json)。
- 原checkpoint routed_scaling_factor=1.5。新读到的Factory参数`apply_routed_scale_to_output`默认False，当前模型caller未传该关键字：本runtime pin把1.5交给router、给runner的output scale为1.0；模型收到Tensor也不再外乘。它与模型旁的“scale outside router”注释不一致，应以实际参数流为原数值基准。不能依据注释把系数从TopK概率搬到CDF BF16输出之后。完整因果链及profile时期依赖身份仍须记录，未修改模型。
- 新只读提取了此前manifest缺少的`FusedMoEFactory`和upstream `MoERunner`依赖，原件与SHA在[新依赖manifest](moe-runtime-source/manifest.json)；其它主要Python/C++范围来自冻结源码。新依赖是当前容器pin，不冒充已在原manifest覆盖。

## 结果：已经排掉的假节省与保留的大范围

### 1. 观测器历史写入不属于模型融合

原DFC后有约60–90µs的mask sum、cat及`[4096,26]`scatter链。曾先保留为“未归属”以免误删，现已匹配到`joint_runtime.State.observe:80–89`：24个receiver count、valid sender数和诊断epoch写历史银行。旧创建CPU栈和钉住源码给出来源，非推断其为EPLB。模型收益信用**0**，不为它做融合/异步移动/新NPU实验。结果见[原来源证明](v168-observer-provenance.json)。其它v168任务仍是带观测器的原cohort耗时；不能简单减去observer就称生产基线。

### 2. DFC前段Cube等待可作为shared Gate-Up的替代位置

- **介绍**：`GMM1`先等cumsum/数据flag，AIV做mask、routing quant、跨rank计数和收集；研究在同launch中先用Cube做共享Gate-Up，再接原GMM1。它与旧K03的GMM2后尾段是**同一份Up的替代放置**，不能加两次收益。
- **到了哪里**：源上等待位置明确（`dispatch_ffn_combine_kernel.hpp:218–230,450–451,793–844`），实际空隙未测。共享Up原生输出INT32，K5120，即便按INT8绝对值128，最大绝对整数和83,886,080仍小于2^31；原输入量化、NZ权重语义、完整INT32输出不变时，有比改变BF16归约更直接的逐位路线。后续浮点activation和Down仍保留原语，不凭整数结论越过它们。
- **结果**：新绑定2560个目标FFN窗口。将所有rank共享Up全部隐藏，每层授予16rank最大native时长、忽略所有EP等待和新开销，原carrier/请求顺序固定，C8最高条件信用仅**0.90620%**。因此停止该“只藏Up、C8单独达1%”路线的阶段补证；不会先量前段τ再重做旧尾段。这个停止不外推到decode/普通请求；那里的共享Up和可用窗口仍需对应负载数据。整个shared DQ/Up/activation/Down全部隐藏的同类乐观信用最高**2.24833%**，仍保留大范围，须检查AIV资源、原浮点舍入和实际阶段窗口。

### 3. 不预设routing量化重复了TopK6次

已读动态量化的full-load与gather实现；`scaleDim0=0`、dynamic quant的1H路径按原token归约/量化后把相同结果交付多个route。冻结host `BLOCK_NUM=20`，故实际参数为target E385/K5120/TopK6/AIV40/UB196352。离线执行未修改host tiler，在M1…8192内分为：1…592 full-load key21000、593…1018 gather/single-sort key11000、1019…8192 gather/multi-sort key11010。它是源码选择结果，不冒充加载二进制的实际tiling buffer。初版CPU适配器误用硬件48个AIV上限，原参数化结果和[纠正](host-routing-correction.json)保留，不用于实际core推断。

不能先把六次必要的route复制说成六份可删量化。边界仍有重复：源码每core/loop按route区间分工，M24/416/816/817/818的quant row计算数分别48/436/836/837/838，对应24/20/20/20/20份跨边界重复。M24原36个core中最慢core量化两行；unique-row owner可变24个core各一行，但每core route发送从4变6，不能仅据总工作减半预测时延减半。大行重分块同样要计入最慢core和复制负载。该局部变换没有独立时延证据，暂不排新测；与更大的norm/dispatch producer一同保留。与共享支共用完整INT8/scale还受两套实际量化配方逐位一致约束，旧浮点公式反例不能被“同为W8A8”覆盖。[正确40-AIV选择器与行域](host-routing-tiling-summary.json)。

### 4. 已有mega内部交付及分支，不重复登记为“新增融合”

原DFC同一launch已经包括mask、routing quant/排序、跨rank token计数/收集、GMM1、分段dequant/SwiGLU/quant、GMM2、combine、清理和unpermute。AIC先GMM1、等V2C后GMM2；AIV分段等待C2V，执行activation后全核同步再释放GMM2。不能将必要barrier当作随意可删等待，或把expert分段当作空闲Cube数量。

Combine的分支与routing selector不同：`M*6<=4096`走V2细块交付，即M≤682；M≥683走V1按expert回传。因此目标M24/416用V2，816/817/818用V1。当前完整组合有四个M范围，不能只查一个kernel名字。另须保留每expert空计数、maxOutputSize截断、local sender全false仍接收/计算远端专家、清理及跨rank结束同步。`ResetTokenPerExpert + CrossRankSync`之后才可重用peer arena；最终unpermute还读原expandedRowIdx及已经带router系数的probs。

更细的候选是按真实recv分布研究combine粒度/输出producer；当前M谓词也受DP padding影响，不能用local有效sender数替代recv工作量。缺口是同窗receiver counts与V1/V2实际epilogue/flag边界的联合净价和逐位门槛；不能从整个DFC数毫秒时长直接声称都能省。实际逐段epilogue实现仍待读，未准入新kernel。

### 5. full-load公共排序producer · 待确认，不能按48份总工作估速度

- **介绍**：`MoeV2FullLoadDynamicQuant::Process`在每个活跃AIV调用相同的`CopyIn/SortCompute`，每次SortCompute含两次全量排序；只有core0写expandedRowIdx，末core写counts。M24/416分别36/40个活跃AIV，存在公共排序计算。可研究一份原排序同时发布原permutation/counts，其它core提前量化，再按原route顺序发送。
- **到了哪里**：source `moe_v2_fullload_dynamic_quant.h:97–161,382–395`确认重复；但各core原来并行排序，共享后关键路径仍有一份排序，加上GM发布/同步和缓存，不能把core数当加速倍数。可能节省的是排序与量化的串行等待，且新增quant scratch/映射GM读者需计价。必须复用原排序/tie和每row量化运算，保留INT32及BF16中间边界。
- **结果**：当前C8只有carrier416用此full-load路线；816–818已经使用global-sort/gather，不能全40层×4carrier领取此收益。即使忽略其它传播，最早请求要达1%需该carrier平均每层净省约112.21µs；晚请求需约365µs/层。缺原排序/量化阶段的独立耗时与新发布成本，暂不给预计收益或排新全模型试验；若同scope阶段预算低于这些门槛，则该C8独立路线直接停止。decode/其它行域另用原负载门槛。它可与完整routing/shared producer联合研究，不与同一阶段的等待消除相加。

### 6. DFC固定20/40发核，剩余4/8承接共享支 · 待确认

- **介绍**：硬件有24个Cube/48个Vector，但冻结DFC host固定20个AIC组、40个AIV。研究用其余4/8资源承接整个shared DQ/Up/activation/Down，不只藏一份Up。它是独立资源并发方案，也可作为更大megakernel的明确分工。
- **到了哪里**：原完整shared全部隐藏的C8条件信用最高2.248%，有继续判断价值；无新增成本时，最有利请求仍须兑现约44.5%的整段信用。core数相加不证明共驻；必须核真实BlockDim/调度和带宽争用、固定形状逐位一致、不同rank的CDF占用窗口。不能直接沿用全核shared耗时；4/8资源可能显著拉长它。
- **结果**：TorchNPU实际安装源码有stream资源接口；[官方说明](https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu-set_stream_limit.md)仅对支持控核的Ascend C算子有效，并要求禁用private format。当前shared权重为NZ，且activation自定义算子的tiler支持尚未核实。因此下一步先核ND/NZ与每个原语的控核实现，随后才值得做固定4/8原语数值/时间/BlockDim门槛；不修改加载服务或全局设备配置，不以get_stream_limit返回值代替实际生效。

## 当前证据与下一步

逐调用压缩窗口（本地归档，SHA见验证）按task ID/类型/shape绑定norm、FP32 gate、TopK6、DFC及shared四阶段；可用Python zlib/json读取，[可读摘要](moe-delivery-summary.json)。未把它们当成全MoE穷尽证明。request预算对post-DFC工作使用忽略所有等待的serial max信用；**不套用当前层pre-EP arrival模型**，因为这些工作在该层EP到达之后。下一层及末端传播需另建模型。

下一步：继续读`prepare_finalize`的padding/mask及DFC quant各tiling分支，再走GMM1→activation→GMM2→combine/unpermute→共享支合并/缩放→HcPost，登记原输出舍入与状态读者。高值完整shared/DFC交叠、router+dispatch大producer保留具体缺口。还未扫描完这些主体，Goal保持ACTIVE；不启动新原样失败变体或低值Up-only阶段测量。

4/8方案的具体初版排程：保留gate/TopK原语和shape，在DFC开始事件之后让辅助流依次执行shared四原语，末端join后再走原ForeachAdd/HcPost；每个替换原语逐位门槛先过，且不改DFC的20/40常量、arena reset或跨rank结束同步。按匹配重侧的中位原时长，完整DFC窗口与原shared流水之比分别约7.47/9.23/9.32/12.40，表示可容纳的伸长预算，不证明这些时间全可并发，也不直接按core数推六倍耗时。[原始预算和缺口](shared-4cube-residency-budget.json)。
