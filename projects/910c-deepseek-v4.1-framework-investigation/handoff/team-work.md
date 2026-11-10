# 171 融合算子团队开发清单

## 2026-11-10 Engram成对owner与lookup整段停止边界

**配平数据补充**：旧loop036原件冻结后获知：配平复采已完成并恢复原服务。新集合为前4条prompt镜像，不能直接沿用旧8条请求比例；32worker窗口中steady lookup union中位约0.03034%/0.03029%。四个native算子族全部站点的松包络约5.009%/5.083%窗口，含其它target/draft，下一步需绑定实际indexK/longKV调用，不能作为候选收益。 [分负载补充](evidence/engram-storage-balanced-supplement.json)。

**Goal ACTIVE；loop036限定source/原生旧价PIVOTED，loop037 RUNNING。全链路未扫全，每个替换原语逐位一致。**

介绍：Engram checkpoint code/E8M0scale→成对加载→共享UVA注册owner/pointer bank→lookup/dequant与退役。到了哪里：12当前source/header pins、11非空ranges；3236Python字面caller搜索；2当前I8/E8M0header且payload读0；7248原native lookup/906TP8group与固定C8端点匹配。结果：weight回调成对写、scale回调no-op；910C flags为MAPPED|PINNED，normalclose有同步/注销失败保留owner，但热graph/reload与半初始化事务仍未证明。仅改dtype/开MXFP8/热复用旧ptr非法。先价整个lookup：全免且替代0，TTFT条件最多1.761056ms/.118045%，整批最多8.599814ms/.046259%；同旧1%固定延迟预算的compactscale/groupbroadcast/lookup-only开发停止，不再补小NPU实验。共享host请求payload差18,432,548,400B单列，非实际释放RAM或NPUKV容量；较大input-owner/WKV/iteration仍待确认。本轮新原语准入0，全链路未扫全，每个替换原语逐位一致。

[整段判断](evidence/Engram成对UVA寿命与整段lookup价值判断.md) · [源审阅](evidence/engram-storage-source-review.json) · [当前前沿](evidence/engram-storage-frontier.json) · [完整旧价与906组记录](evidence/engram-storage-price.json) · [原件/摘录范围](evidence/engram-storage-public-source-manifest.json)。未做新原语NPU验收，不再为已低值lookup小范围补证。

loop037 Run run-20261110T035955Z：indexK/longKV的RMS→inplacepartialRoPE→DynamicQuant→code/halfscale scatter实际原生交付。6当前source匹配；已读indexer.update_keys、AscendRMSNorm.forward_oot无residual与bias_loaded分支、DeviceOperator旋转/散写和ops导入。继续OOT注册/实际选中callee、TorchNPU/CANN/自定义kernel tile/舍入、slot/cache代次与末读。归入已有input-ready/query/SWA较大parent，条件全免2.1327%/half-h20us1.0449%不是小epilogue收益；先整段净价，不干扰配平profile控制器。 [新入口](evidence/cache-epilogue-entry.json)，Run仍PLANNED。此源码扫描无新NPU/模型/请求/profile/显式远端写入/服务动作，独立配平控制器未操作。


## 2026-11-10 原序L1与尾块联合判断、Engram下一入口

**Goal ACTIVE；loop035限定source/CPU/条件价格PIVOTED，loop036 RUNNING。全链路未扫全，每个替换原语逐位一致。**

介绍：selected IDs/可见长度→原V0物理pair/半区压紧→三槽GM panel→QK/PV真实L1交付。到了哪里：11实际source/限定ranges，333原V0 CPUfixture/46184行地址；12旧nativeQLI×PA情景与1239552活跃NZ坐标；54旧价复现、108联合与20指令成本情景。结果：910C ND2NZ srcDValue为uint16元素，物理行距128会溢出，保留64位base并按原顺序拆分；朴素逻辑序104例不符，7例活跃洞回退。发现两块L1保留末128行、第三块work的较大联合方向，需跨QK(i)/PV(i-1)任务及新flags。K512/.75参考/h5us/12B factor1、全尾合资格时最有利请求条件1.6663%，非P90/实测，负值保留；高指令数可转负。原生ND2NZ padding/LoadData/MMAD逐位、真实K/页代次/分组、预取与净价仍待确认。停止逻辑序直替、A5宽stride套910C、原7块全驻K512与旧stockGM prepack的具体配方。本轮新增原语0，全链路未扫全，每个替换原语逐位一致。

[整段判断](evidence/SMLA原序直达L1与尾块复用判断.md) · [源审阅](evidence/smla-direct-l1-source-review.json) · [待确认/已停止前沿](evidence/smla-direct-l1-frontier.json) · [全部价格](evidence/smla-direct-l1-price.json) · [原件与公开范围](evidence/smla-direct-l1-public-source-manifest.json)。CPU证明不作NPU原语逐位准入。

loop036 Run run-20261110T032750Z：Engram紧凑scale checkpoint→host/sharedUVA注册owner→device pointer bank→lookup/dequant→末读/close/reload。3当前source SHA匹配；已读storage/scale类型分配、I8与MXFP8加载分支和lookup入口。继续SharedUvaBuffer注册/失败回滚/各rank close、成对code/scale bank寿命和图稳定地址；保留原INT8×FP32→BF16与255 NaN位合同。只为会改变18.4325GB容量/大交付净价去留的参数补证；不干扰进行中的配平profile控制器。 [新入口](evidence/engram-storage-entry.json)，Run仍PLANNED，不当成已完成。此源码扫描无新NPU/模型/请求/profile/显式远端写入/服务动作；本机另一个已授权配平profile控制器未操作。



## 2026-11-10 SMLA原序计划继续深入

介绍：沿长度/掩码/平台/确定性→AICPU分核包→arch22 FA/FD与QK/PV实际读者，判selected-KV大交付。到了哪里：21实际源pin/18限定ranges；原标量323–1631的7个本机CPUfixture；1840原C8 native/2560窗/30typed组，6旧值精确复现、54付plan/协调成本情景。结果：packet已按batch ratio共享，900词已定义/124尾词未写；默认整行，q3 level3可生成6FA/3FD/48vec。更正旧“未传FLASH_DECODE”原因：入口会传，但host/key选择FD0；未继续消费的是BATCH_CONSISTENCY算术。停止flag-only安全跨S2和重复38层metadata信用；不是服务故障/NPU精度结论。12B计划/.75稀疏参考/h5us，K256条件最有利请求.727%–.875%，K512为1.101%–1.395%；后者继续深入。13负请求信用保留，均非实测/P90。新增原语准入0，全链路未扫全，每替换原语逐位一致。

loop035 Run run-20261110T023853Z：原序selected-KV64位地址+transfer计划→Cube QK/PV精确L1布局。已读Vector pair/large-stride fallback、Cube panel ND2NZ和DataCopyPA片段；继续vendor Nd2NzParams字段/stride/padding、PV布局/LoadData/flags、原ring与源/page/index代次，先找匹配旧搬运数据，再只补影响去留的门槛。保留原metadata/逻辑TopK及所有MMA/softmax/FP16-BF16边界。

[判断](evidence/SMLA分核包与原顺序交付判断.md) · [具体缺口](evidence/smla-schedule-frontier.json) · [下一入口](evidence/smla-direct-kv-entry.json)。


## 2026-11-10 workspace实际寿命与分核包前沿

介绍：从实际GetWorkspaceSize→Byte empty→rawptr/OpCommand→ACLNN/设备末读，判断容量、共享和大融合。到了哪里：安装TorchNPU声明commit绑定24参考源，15文件限定ranges；9安装header逐SHA一致，16worker初始expandable_segments:True；原rank0 DB无分配/释放或PYTORCH_API表。结果：原NPU empty走同流CachingAllocator；scope析构不等于设备完成/物理释放，queued闭包没有workspace Tensor lease；capture用private pool且replay不逐节点重跑empty。停止逐层容量累加、captured每步malloc收益与scope末读计法。分桶DFC/QLI缩容量支持更大融合仍待确认，缺全sender界/实际ACLNN-inner scratch/全流末读/物理峰值/逐位与净价。大DFC旧共同尾段全免条件P9020.09%–22.69%，阶段占25%/半省/h5us为2.458%–2.783%，容量本身不兑换成该收益。新增已确认原语0；全链路未扫全，每替换原语逐位一致。

loop034：SMLA metadata AICPU的1024×INT32分核包→arch22 FA/FD字段和workspace末读；8实际source SHA匹配，已读adapter35–89、AICPU Compute/Prepare25–110与packet布局，BalanceSchedule/GenMetadata、actual core attrs/合法determinism/空KV/ori_sparse及读者仍待扫。36/72只是源码packet容量常量，不代表实际launch核数；不按小metadata成本剪掉selectedKV/QK/PV大parent。

**范围更正**：三处审阅尾行标签超过EOF，已按实际文件长度更正；不存在行不计覆盖。[审阅v2](evidence/native-workspace-source-review-v2.json)与[原记录更正](evidence/native-workspace-source-review-correction.json)保留原Run证据。

[判断](evidence/原生workspace寿命与融合容量判断.md)；当前不接纳新的原语/性能基线。

## 2026-11-10 DFC原生交付与24/48资源更正；继续workspace实际分配

**介绍**：沿receiver行→GMM1 FP16→SwiGLU INT8/scale→GMM2 FP16→combine追真实交付、别名和同步，再价整段。

**到了哪里**：31实际source pin、所列ranges；16旧DB逐SHA及2,560原eager任务时间/globalId精确匹配，全部 `MIX_AIC / blockNum24 / mixBlockNum48`。8 typed成本和24专家计数逐窗绑定。旧“DFC固定20/40留4/8空核”解释撤回：40是内部routing tiler参数，wholeDFC从平台取资源。独立shared4/8的原语字节/发核事实保留，不能推共驻。

**结果**：已停止——依赖固定空闲4/8、当前arch2201直接L0C→UB/换TSCM名就删GM、仅compact4608/把既有别名当四bank节省的配方。C1/C2已同640MiB、A/activation已同320MiB；2171计数窗有潜在WAR见证，960MiB公式不是已释放显存，不乘层/卡。原FP16、2304维max、clamp/Exp/Div/RINT和scale末读都保留。

**待确认**：大DFC共同尾段全免条件P9020.090%–22.690%；若相关曝光阶段占25%、半省/h5us，为2.458%–2.783%。半省/h20us要同1%排序参照，阶段需占10.70%–12.09%。这些是固定原调度情景，非实测/预计；计数不等于阶段时间。第二V2C等待可研究移到真实group21消费者前，但需实际DMA/旗标/位一致与曝光。显式24→20让核若多花20%×Cube敏感阶段，并非免费空核；全局负收益已保留。六普通/完整decode与实际阶段时间仍缺。已确认可实现本轮新增0。

**下一处**：loop032有限PIVOTED，loop033 RUNNING；6实际source pin，已读 `GetWorkspaceSize→at::empty Byte→rawptr capture→OpCommand.SetCustomHandler/Run` 入口。继续queue/allocator/stream/capture/现有memory记录和末读；不凭tensor作用域断言use-after-free，也不凭声明容量断言物理释放。无新NPU、模型、请求、profile、显式远端写入或服务改动。全链路未扫全，每替换原语逐位一致。

[介绍、到了哪里、结果](evidence/DFC原生交付与24核资源更正.md) · [资源更正证据](evidence/dfc-resource-correction.json) · [净价前沿](evidence/dfc-gemm-tile-frontier.json) · [实际分配入口](evidence/native-workspace-entry.json)

## 2026-11-10 latent双缓存整段净价；继续DFC专家GMM内部

**介绍**：从四KV源输入追到FP32投影/pool/ring、BF16 latent、index-K量化双发布和同latent原地long-RoPE/发布，先按整段再判小配方。

**到了哪里**：256首token源窗，224发布，其中40窗query0仍更新cache；133个typed子集条件匹配。另112个完整decode诊断周期、448源窗、38条件；它们是candidate周期样本，非全C8请求/六普通。实际C2为token-aligned清零后pool、原独立RMS；pool读旧ring必须早于尾写，wk末读前不能在另一流原地RoPE。native RMS/quant/scatter内部未读，未用pin冒充覆盖。

**结果**：已停止——同独立1%预算的尾段（span全免条件P90 .2802%）、保留原GEMM的其它计算/间隙（.4856%）、仅挪早生产段理想重叠（.7990%）。完整root含cast/zero/span全免1.1518%，half/h20us .5545%；扩大到含保留query/SWA的输入ready联合parent为2.1327%、half/h20us1.0449%，仅待确认，须补真正ready/资源/位一致/通信净成本，不与旧C2/host/QLI/SMLA相加。graph样本root span全免周期最大.9972%，非请求收益。本轮“已确认可实现”新增0，无新NPU/请求/profile/远端写入/服务改动。

**下一处**：loop031有限PIVOTED，loop032 RUNNING；实际DFC GMM1接收端M/轮转core/两组C2V发布入口已读，10源hash匹配。继续GMM2→BlockMmad异步fixpipe/FP16→SwiGLU INT8+scale→combine真实消费者，先整段价。GMM1 logicalN与expertC strideK解释尚未核，不能直接认作可删padding；独立4/8原语不证明与20/40 DFC共驻。每替换原语逐位一致，全链路未扫全。

[介绍、到了哪里、结果](evidence/latent双缓存整段与保留GEMM净价判断.md) · [净价前沿](evidence/index-key-publication-frontier.json) · [下一处入口](evidence/dfc-gemm-tile-entry.json)

## 2026-11-10 QLI原生末读者、容量与有成本门槛；继续latent双交付

**介绍**：沿整段QLI继续host分核→Cube/Vec→token/candidate最终输出及PA物理读者，判断少算/少搬与资源再分配。每替换原语逐位一致。

**到了哪里**：18 native源核对，arch22主body、原Sort32/MrgSort接口、host parser/tiler主路径和非FD metadata分配已读列明ranges。16当前worker映射custom opapi，QLI JSON/.o f1dd0d93…相符、4key；磁盘/映射/initial env不冒充每调用device binary/tiling或loaded内存证明。

**结果**：已停止——910C所选metadata源不生成S2/LD归并payload，删除该数据信用0。待确认——末尾仍Reset/init/prefetch/SyncAll的空epilogue；声明workspace与显式region差额；保留原half链、candidate2048/pin/filler/tie次序的top512 accumulator及shared membership/大QLI。C24且host核数=kernel block数时，源声明99.0234375MiB、显式G32区域4.6875MiB；这不是实际resident可释放显存，不乘八，LibApi L/隐藏scratch/实际alias与高水位待核。

有成本筛查：若accumulator占原kernel时间25%、阶段时间删75%，e1/h0、5、20us每index source时条件P90为1.017481%/1.006756%/.974581%。h20us要过1%需f≥25.6246%(e1)或51.2491%(e.5)。阶段比例/曝光/新成本会改变去留；PMU比率不当阶段耗时，表非预计/因果/实测收益。旧byte-mask局部机制保留，旧batched-slot C8稳健门槛未过；不认领新服务收益。

下一条已经从实际caller读到：KV源2/8/14/20同latent→indexwk/RMS/RoPE/quant/key-scale两scatter，再原地longRoPE/第三scatter。跨流必须保护indexwk对preRoPE latent的末读；query的half-scale-first quant与key先quant后scalehalf也不能混用。loop031继续真实行量/C2完整位与整个多输出交付价格，再追native consumers。**Goal ACTIVE，全链路未扫全，无新primitive/性能基线准入。**

[QLI原生/资源与净价判断](evidence/QLI分核末读者与容量及净价判断.md) · [native覆盖/待确认](evidence/indexer-native-contract-frontier.json) · [下一条入口，未价全段](evidence/index-key-producer-entry.json) · [地图](evidence/source-frontier.json)。本轮只读，模型/NPU/请求/新profile/远端写入/服务变更0；未操作其它controller。


## 2026-11-10 启动交付已判，热Indexer整段继续native

**介绍**：从完整启动交付追四helper、workspace及MM容量，再按实际消费者定热Indexer融合范围。每个替换原语逐位一致。

**到了哪里**：四helper/early guard、workspace capture锁定及MM profile/KV预算已读；16rank旧启动wrapper .329–.378s（不相加）。纯文本启动仍profile8192 encoder tokens/8images。142直接safetensors头部匹配266视觉tensor、485268480 BF16元素，checkpoint925.576MiB；构造dtype约925.605MiB。张量payload读取0，非live allocator上界。热路径368组/50独立成本条件吻合，实际每卡rows≤1024，长度已融合。Python V3→native V2 adapter及arch22 half舍入、算后membership/filler、crosscore flags追至所列局部ranges；源pin不当加载库或全分支完成。

**结果**：已停止——冷开销删除和locked热增长对暖请求直接信用0；同1%独立预算下量化+chronological排序即使全免费，条件P90仅.594571%，额外大行sum在这些调用本就不存在。待确认、值得深入——QLI整段有利P9089.693940ms/6.012243%；删50%且每source付5us为44.686970ms/2.995397%，均非预计/实测收益。不能用这份小段价格剪掉大QLI，也不能相加其它候选。视觉参数单bank不足2240MiB/卡，联合容量仍需allocator/headroom/KV policy/逐位与请求价。

**逐位限制**：原FP16 query scale、weights×scale half、QK DEQF16+ReLU half、FP32 head sum/key scale次序保留；候选外真实位置可作不足top512填充，无条件压掉/改-1停止。chronological bank已在八index sources共享，每消费者重算不获免费信用。下一条继续host tiler/SplitCore、candidate pin/tie/merge、LD末输出、PA物理gather和当前priority/loaded库-key；随后核有成本净价。原ordinary/跨迭代cache与其它全链路缺口保留，**Goal ACTIVE、全链路未扫全**。

[启动/容量判断](evidence/启动预热与容量交付判断.md) · [热Indexer整段深入](evidence/热Indexer整段交付与逐位边界判断.md) · [覆盖地图](evidence/source-frontier.json)。本轮只读：模型/NPU/请求/新profile/远端写入/服务变更0，未操作其它controller。header-index失败及analyzer字段错误原样留证，无新原语/性能基线准入。


## 2026-11-10 实际mode更正与公共前缀原语：复用原Encoding，保留冷miss

**介绍**：沿输入renderer→native encode_batch→完整Encoding→每请求IDs的实际消费者查重复计算。每替换原语逐位一致，先价整段，再深入稳定前缀共享。

**到了哪里**：C8八任务在2.473533ms提交，单worker native466.209239ms/父方法487.110933ms，最后input492.603961ms。显式tokenizer-mode=deepseek_v41已沿两registry追完限定分派：tokenizer为V41 factory+cached HF，renderer却是DeepseekV4Renderer→BaseRenderer。**撤回“实际服务已有HfRenderer副本池、只调线程数即可安全并行”的解释**；HfRenderer不在这条路径，flag-only配方停止，显式独立backend/调度仍待确认。实际无pool CachedDSV41TokenizersBackend+原scalar方法重放8份IDs同hash；from_pretrained bootstrap/实际加载对象及wire尚未准入。Generic/pool回放只属反事实能力。初稿报告及旧JSON解释由actual-tokenizer-dispatch-frontier更正，原件保留。

**结果**：
- **已确认可实现（给定prefix的受限CPU原语）**：固定pipeline的117067字符/29385 IDs安全数字cut，不跨AddedToken；1256cases/269698 IDs含24fallback、Unicode/special/cap/error/list ownership，差异0。实际公共前缀117286字符；预选prefix是点证，不是在线发现/LRU实现。
- 进一步从首次原完整Encoding的字符token_to_chars和已有IDs提取prefix，8份完全相同，增量插入中位.182108ms/max.261111ms，避免另encode prefix。char117067≠UTF8byte117187，错误bytecut控制拒绝。局部9对old8 781.565971→保留首个完整miss+7hits166.072773ms；这是孤立CPU原语观测，不是实测请求收益。第一组另一cohort的759.538/79.447/额外build129.943ms原件单列，跨Run标不具可比性，不与旧profile466ms相加/替换。
- **待确认、值得开发：在线prefix cache**。缺一般prefix发现/插入、byte-LRU/epoch/实际backend指纹失效、完整bootstrap/wire、真实CPU关键暴露和六ordinary。每请求独立list，原text/arrival/cache_salt、validation与EngineCore/Engram/H2D消费者保留，无模型/H2D删除信用。
- **有成本粗价**：旧FIFO/其它父work/gap不变，冷miss保留第一job全部native，只给7hits删50%，每job付.5ms并首次插入.182108ms，最后input条件提前199.150528ms、firstinput延迟.682108ms。固定下游lag/未知request配对的P90情景−.682108～144.505989ms，非因果界/预计或实测收益；不能把input提前改名为batch收益，不叠加旧seed/metadata/parallel预算。
- **已停止**：依赖不存在pool的flag-only并行、裸共享可变backend、任意prefix拼接和错误bytecut。保留显式pool与在线cache的不同净成本/代次问题。

**下一段**：loop029已pin7源/5旧match，读到kernel_warmup wrapper，四个rejection_sampler/penalties/rms/indexer helper body仍未读；之后workspace/encoder_budget真实producer、消费者与cleanup。loop027只做mode前置守卫，不能记这些helper已扫。无新startup/model/profile/服务改动；全链路未扫全，不用函数/证据数量作完整性证明。

失败原件保留：AutoTokenizer尝试含模型/NPU包进口，未观测其设备初始化，不把“无显式计算请求”当进口无设备初始化证明；成功native点无Torch，成功scalar replay无torch_npu。未请求模型/device计算，owned cache路径核实不存在，无远端文件写入。配置/21进程initial-exec flags与native trace支持FASTOKENS关闭，但不当加载中Python对象证明。

[实际分派/原语/净价与缺口](evidence/实际tokenizer分派与prefix原语判断.md) · [受限边界证明](evidence/shared-prefix-tokenizer-proof.md) · [下一启动入口](evidence/startup-warmup-entry.json)

## 2026-11-10 文本/SSE整段与ByteLevel长prompt预热点证

**介绍**：沿真实FastIncrementalDetokenizer/stop/logprobs→RequestOutput/collector→completion SSE→Starlette UTF8/Uvicorn HTTP交付，先价整段，再深入首步长prompt重复decode。每替换原语逐位一致。

**到了哪里**：437834完整Python事件唯一ID及父边包含校验通过，父子区间并集去重；8个约10ms的native大首步合计84.266439ms。installed tokenizers0.22.2/ByteLevel JSON/binary已pin。候选保留原构造的完整ID转换，只在list prompt IDs最后有效token以ASCII结束且tokenizer不可变时选择空seed；其它边界回退。129280词表ID、两skip标志、多步UTF8/special/error等共1302780次逐步字节/None/异常比较无差异；576个原vLLM wrapper场景/2976次update也无差异。未完成UTF8盲删seed有反例，未放宽guard。None/prompt_embeds入口尚未准入，接入前必须绕过原型guard。

**结果**：
- **已确认可实现（限定CPU原语/原wrapper fixture）**：8份同输入CPU重建都命中，9对交错配对中位合计101.113973→4.124833ms，包含原构造和新增guard/空stream，非服务TTFT/NPU实测。原构造4.392ms不认领为已删。
- **待确认**：实际renderer IDs/kwargs/backend/不可变资格、完整SSE wire回归与真实关键暴露/新成本、六ordinary。旧profile native8全免是宽松固定schedule敏感性：最高84.266439ms/P90约5.86%/整批.482%；rho.5、每seed .1ms条件净41.33322ms，非预计收益，不能拿CPU点96.989ms替换旧cohort预算。
- **已停止**：JSON-only同C8独立1%预算，全免首tokenP90信用仅.255192ms，整批约.0851%；不剪whole output/seed/ordinary。整输出609.977ms并集有大量保留/重叠工作，不当实测可省。
- **下一段**：实际输入renderer→async tokenize/executor→HF native。已pin8源并读局部入口：真实调用tokenizer(prompt, kwargs)，有max_input_tokens+1截断、token IDs/embeds分支；8后台encode_batch共466.209ms尚不是event-loop阻塞或收益。之后四warmup helper内部/startup MM仍未扫完。

Goal ACTIVE。loop023/024 PIVOTED；第三Run原语对拍通过，前两host/container路径ERROR/INVALID保留，Task禁止含invalid Run的Loop标ACCEPTED。CPU词表加载3次，模型权重/NPU/服务请求/新profile/远端写入/服务改动0；未操作其它controller。源码范围/哈希/fixture计数均不当全链路穷尽证明。

[整段与价格](evidence/文本到SSE交付与长prompt预热判断.md) · [CPU定点/边界/净价缺口](evidence/ByteLevel长prompt预热定点判断.md) · [下一输入入口](evidence/input-renderer-entry.json)

## 2026-11-10 请求代次、stop与终止轮：保守host界与整段价格

介绍：从真实internal ID/归一化stop参数到abort双队列、旧future、worker tombstone与异常消费者，反推no-next-demand整段交付。到了哪里：18运行树源14旧hash匹配/4新取；原方法控制fixture验证external abort可作用多internal代、EOS/显式stop/min-token集合不同、GPU valid数不等于可见cap裁剪数，async空CPU valid列表也不等于无输出。默认随机后缀/opt-out是source条件，不当live值或唯一性证明。

结果：scheduler已有保守终止界，会抑制即使全拒绝也必然达到max_tokens的额外target；不给重复40层信用。同一post-schedule数学界可随immutable frame提供终止保守子集，不需新accepted-count D2H；实际资格频率还缺。原C8七终止波把所有GPU-facing尾部全免也仅32.152328ms/整批0.172948%、TTFT0，rho.5/每波.25ms只14.326164ms，停止同独立1%小预算。该边界是target采样/verification最后native→draft最后score，不能作为更大host/queued/EOF父区间上界。

待确认：no-next-demand DSpark/provider/output/queued-input联合，保留context/prefix cache活写者、真实peer专家服务、全DP/TP决定、zero placeholders/valid counts、原算术几何与前轮lease。调用_skip_drafting本身仍跑dummy body，不是整段删除。普通六负载用exact9144 client SHA eba6确认caps126/13/300/20/126/13、ignoreEOS True；18端点各arm保留，四名称跨arm输出hash不同，不作位一致/性能前后对。短code-single末交付松弛约60ms不等于本地native时长或净省，缺实际资格、native/CPU parent和新增fence/协议成本；不套C8低预算剪掉普通。

已确认可实现：本轮无新原语准入；控制fixture/review PASS不是NPU逐位或模型通过。Goal ACTIVE、每替换原语逐位一致、全链路未扫全；下一条loop-023追hot frontend detokenizer/stop-string、logprobs、RequestOutput到SSE字节与buffer/error消费者，之后startup/warmup/MM。NPU0、模型0、服务请求0、新profile0、远端写入0，未改服务/环境。

[本轮介绍、到了哪里、结果和证据](evidence/请求代次与终止轮交付判断.md)。

## 2026-11-10 Prefix-cache/退役：实际guard排除空GPU配方

介绍：沿request hash→多组prefix命中/touch→分配/发布→worker cache消费者→finished/abort/reset/复用，先核整段已有共享与真实工作。到了哪里：27运行树源19旧hash匹配/8新取；不变serve.sh/default和完成profile启动记录确认prefix caching开启。普通KV/EC transfer、generic权重预取与sparse KV offload未启用，Engram CPU表仍活跃。全16既有trace SHA/size不变；该capture没有专属cache/prepare-input标签，不能据此定CPU父区间为0。

结果：实际V1＋DSpark不初始化通用KV block zeroer；无Mamba、128对齐cached specs、circular32私有/noncacheable，使当前partial-hit CoW copy也不触发。删这两类现有GPU工作的local/request信用0（固定host），停止同配方；CPU ID收集/drain仍待大host范围定价。仅翻cascade/trailing-drop或通用changed-M hidden复用不能继承逐位一致；不剪新的layout/owner/shared-read机制。

待确认：typed physical-cache＋immutable BatchFrame退役，以及保留原row算术的跨请求selected-KV读取共享，分别与既有metadata-host、SMLA/QLI/物理cache范围重叠，不相加。缺整个engine CPU父区间、实际选中物理行交集/现有L2共享、owner代次/最后读者及净成本。原free_blocks fixture证明refcnt0仍可prefix命中且不可写；当前无KV consumer，defer_block_free=False，新早交付不能继承不存在的通用fence。

已确认可实现：本轮无新增原语准入。review/fixture PASS不是NPU或模型通过。NPU0、模型0、请求0、新profile0、远端写入0；Goal ACTIVE、每替换原语逐位一致、全链路未扫全。下一条loop-022沿normalized internal request ID/stop参数→abort旧future、同ID新代及异常消费者；4个warmup helper内部/startup MM缺口仍保留。

[本轮介绍、到了哪里、结果与证据](evidence/Prefix-cache与退役交付判断.md)。

## 2026-11-10 整inactive-DP与实际就绪：限定判断，继续全链路

介绍：沿engine本地无工作→target40＋draft3→peer专家与最后输出，先评估整轮、合成sender和host联合交付。到了哪里：171 hostname核对恢复，20运行树文件16旧hash匹配/4新取＋5依赖；原C8的43个idle F21入口全部早于active。独立新profile全16 CPU/native每轮40＋3关联完成，26个共同Gloo ordinal形成208对；实际Ascend target/draft共用INT32[2,DP]数据包，8个CPU pair并非单个16rank reduce。源范围/身份计数不证明全链路完成。

结果：原C8固定active到达/EP服务/资源的独立idle prefix/tail停止；252ms整轮的32ms非DFC补集不能直接算收益。新录制C8 Gloo backend-only全免P90仅5.586240ms/0.388793%，rho=.5、每阶段新增.1ms为1.393120ms/.096959%，停止同独立TTFT1%预算。DP1独有末尾约715ms等待在所有首token后，且跨同步profiler stop/export；不得将Free/Gloo总量归为生产收益。

待确认：全局无真实工作＋BatchFrame整轮快路、单侧真实EP service-only/资源重写。前者仅两DP代表marker与frontend-ready夹住的有利启动上界21.561858ms（P90 1.500668%、整批.123345%；half/h.1为10.580929ms），还免费抹掉下一真实准备，不是预计净收益。缺实际全16工作标志/TP一致、inbox与在途future/lease、协议代次/成本和六普通负载；后者缺eager M416接收计数与改sender行序/M的逐原语逐位证明。与已有metadata-host范围不相加。Engram全false mask直接returnhidden不具通用IEEE位等价，非新NPU失败。

已确认可实现：本轮无新增原语准入。NPU0、模型0、请求0、新profile0、远端写入0，未操作另一个已完成profile控制器。Goal ACTIVE、每替换原语逐位一致、全链路未扫全；下一条loop-021沿prefix/cache/provider/finished/error退役，先核冻结guard再追最后读者；冷启动warmup/启动MM缺口仍保留。

[完整介绍、到了哪里、结果及证据](evidence/整inactive-DP与Gloo就绪交付判断.md)。

## 2026-11-10 实际worker家族与有效配置更正

介绍：沿真实V1 profile/capture/dummy与消费者，见[本轮判断](evidence/worker-capture-review.md)。到了哪里：22份局部源范围、15旧identity匹配；3632参与窗分为3136真实Q>0、488有metadata但Q0、8个None-metadata dummy，SP/head/三层sender全部核对。结果：仅翻DSpark flag、按M3共图、仅翻shared flag均已停止；真实metadata图模板/dispatcher/代次与显式资源重写仍待确认，无新native准入。

**更正此前“shared已重叠”**：固定enable_fused_mc2=1会强制multistream_overlap_shared_expert=False，源码SHA与旧启动排除回执一致；可选helper不是已生效重叠，旧candidate样本0、无speedup。整段价格与已有原语资格保留，不能因此宣称共驻已支持。

下一条loop-020：整inactive-DP target40+draft，追真实跨卡expert服务、原mask和死输出再定价。原C8只一波idle，连真实peer整三body parent免费也仅P90 .647747%/整批.051980%，停止同小预算，不剪六普通/target40。新授权framework profile已独立完成，先绑定Gloo与target/draft就绪；不混入旧227波价格。Goal ACTIVE，每替换原语逐位一致。



## 2026-11-10 三层body限定进展；全链路仍进行中

介绍：沿真实三层DSpark HC→CP/SWA→TP恢复→EP16/top3/shared→下层与head输入，见[本轮完整判断](evidence/dspark-three-body-review.md)。到了哪里：原C8全部3696窗与11088 F21计数；227活跃波保留3632参与窗（3624自身active），原observer epoch为DP各自历史标签，错误断言与更正保留。结果：全parent免费有利P90 1.424587%/整批3.569201%；rho.5、h.1ms每波条件净.678778%/1.662497%，不是预计/实测，不与host/context/score相加。每替换原语逐位一致；本轮无新native准入。

待确认：同原语body graph/提交+shared/routed资源安排；具体切口、persistent metadata、wrapper同步与DFC末读已下钻。已停止：零sender跳EP、空expert新增GMM收益、三层同值合并及同小TTFT预算配方。已确认可实现：本轮没有新增候选准入。

下一条：loop-019真实V1 worker _dummy_run/profile_run/capture_model及继承caller，派生target/draft warmup、dummy、metadata/cache代次和collective顺序，之后继续prefix/provider/error剩余路径。当前171认证失败，本轮只用固定源码/旧profile，无当前运行身份核验、NPU或远端写入。


## 2026-11-10 当前：输入/context整段价格，继续三层draft body

**介绍**：从实际loader/runner/selector走到三cache与query embedding。**到了哪里**：15运行源；原C8 SQL3696窗（3624 active、453 TP8组），不是地址或覆盖率证明。draft MM预embedding被loader关闭，own embedding保留；padded aux全QSL arange与accepted/rejected/sample selector分开。**结果**：完整aux选行→main→metadata/context→embedding→第一draft HcPre全免，有利P90 1.549643%/整批1.460072%；rho.5/h.1ms每波条件净.741306%/.607932%，非预计/实测/准入。大host/context待确认，与metadata-host不叠加；main单独全免.304272%/.177883%停止同独立C8小预算。不存在的重复、跨模型共值、通用changed-M原配方均停止；旧反例native回执和7当前源pin核对，旧派生flow-index失配保留。

loop-017 PIVOTED仅限源码/价格判断，loop-018继续真正三层draft的noncausal SWA/CP、HC和128/top3/EP16、提交/graph合法性与megakernel范围。物理三cache末读、原M/tile逐位、联合新成本及完整host-parent六普通时间仍是具体缺口。Goal ACTIVE，全链路未扫全，每替换原语逐位一致；本轮NPU0、推理0、新profile0、远端写入0。

[介绍、到了哪里、结果与下一条交付](evidence/embedding-context-review.md)。


## 2026-11-10 当前：末层物理cache与原行映射，继续输入/embedding

**介绍**：保留body38完整aux39，body39沿writer/page/未来读者到topK/TP-SP返回。**到了哪里**：16运行源pin；slot3/group11、padded stride、非模128环；页按已处理提交prefix退役。3328单请求末行=CP7/旧local415，不能压成一行后取topK前缀。**结果**：尾128 writer且原prefix发布不变、新topK前缀且原restore不变两配方已停止；全原cache+显式旧行/原数值几何联合query/O/FFN仍待确认。56原C8窗口保留六cache原语旧时长的有利组件P90包络2.163059%；rho.5/h.5ms每wave条件净.947468%，非预计/实测/逐位准入，也不剪改变cache producer的大范围。混合波的全部验证行仍有读者。

loop-016 **PIVOTED仅限源码/价格判断**，loop-017继续文本输入/embedding与DSpark context producer；实际多请求QSL/device长度、SMLA/HC/GEMM/EP原几何、逐位实现/净成本/六普通缺口保留。Goal ACTIVE，全链路未扫全，每替换原语逐位一致。本轮NPU0、推理0、新profile0、远端写入0，服务不变。

[介绍、到了哪里、结果及下一条交付](evidence/final-cache-review.md)。


<!-- FINAL-PRODUCER-20261110-START -->
## 当前前沿：aux完整接线更正，继续body39（loop-016）

**Goal ACTIVE；全链路未扫全；每个替换原语逐位一致。更正并撤回：此前aux36/37/38与两个body可删的判断漏了NPU runner先加1，两层3.8115%/1.77169%情景不得再用于候选收益。**

- **介绍**：重新沿GPU setup virtual selector、NPU override、setter到model采集点追完整接线，保留原错误记录和失效说明。
- **到了哪里**：冻结ID37/38/39，上游无eagle/dflash覆盖，NPU选择38/39/40，setter排序去重不换ID，model再判断layer_idx+1；真实aux入口为37/38/39。body38全行输出mean供aux39，只有body39在最后采集之后。这是源码/冻结配置推导，不冒充live对象dump；缺Docker INFO日志未被当成分支证据。
- **结果**：原两body裁剪已停止/撤回，原Run不可变记录由新Run及superseding knowledge明确声明旧解释失效。当前body39候选仍待确认；56个real eager窗口下inclusive最有利DP全免包络P90 2.32864%，h=.5ms/wave、rho.5情景净1.03026%，rho.25仅.44810%，负收益保留。非实测或原语准入。metadata72窗末读结论不变，producer_done/alias/native/graph/error仍待确认。
- **下一处**：body39 SWA128物理writer/page/slot/未来读者、selected-query到topK36原CP-local映射；保留KV20/topK36与body38完整aux39。38 HCPost→完整aux-mean物化只能另研、须算全aux行，不能认领38整层计算删除。新NPU0、推理0、新profile0、远端写入0。

[更正与当前判断](evidence/metadata-tail-and-final-producer.md) · [完整接线](evidence/aux-config-call-chain.json) · [当前body39价格](evidence/final-producer-entry.json) · [撤回档案](evidence/withdrawn-two-layer-price.json)

<!-- FINAL-PRODUCER-20261110-END -->

<!-- METADATA-HOST-20261110-START -->
## 上一段：metadata与host联合交付（loop-015已PIVOTED）

**Goal ACTIVE；loop-014 PIVOTED（限定结论），loop-015 PIVOTED（限定结论）；全链路未扫全。每个替换原语逐位一致。**

- **介绍**：从target DP协调、global/local metadata，经实际DSpark query到producer/consumer，先核早知字段与整段净值。
- **到了哪里**：13源SHA；DP为8组[r,r+8]CPU Gloo pair。实际DSpark override返回B×Q（冻结Q5），不能复用target24桶；GPU reject后的length/slot仍晚知。CP同代batch可共享，但global/local域及每资源physical state分开。V1每次host submit、FULL用external event、finally release，不能借V2全task finish证明V1末读。
- **结果**：旧CP/CPU-plan renderer精确重放，21项profile输入hash核对；已有方案不重复认领，旧文本验收不满足逐位门槛。同代BatchFrame+group组装/发布**待确认，值得深入**。原样shape-only返回缓存/mini graph已停止；device-only小预算不剪较大host方案。
- **净模型**：保留两个parent builder、每组return、first producer。旧B4诊断组件毛预算1.689–2.490ms；f=.5/rho=.5/h=.25ms时条件净.79357ms/迭代，C8整批.969%、P90.266%；rho1时1.83714ms、2.243%/.616%。这是有利情景而非matched实测。无暴露的负收益保留。原C8串行敏感性到整批1%需净.81898ms/wave，P90需2.98371ms；首输出一次另列，不多认异步下一波。
- **下一处**：BatchFrame observer/alias/closure和V1 release到实际末读，再决定具体grouped实现；DP packet保留empty/dummy/K0/input-fits/error序列及Gloo净成本缺口。其后回末层attention/FFN row-demand和剩余输入/embedding/Engram。新NPU0、推理0、新profile0、远端写入0，无部署变化。

[完整判断](evidence/metadata-host-review.md) · [具体联合协议](evidence/metadata-batch-frame-protocol.md) · [有成本净模型与请求门槛](evidence/metadata-host-net-price.json) · [源码与缺口前沿](evidence/metadata-host-frontier.json)

<!-- METADATA-HOST-20261110-END -->

<!-- INPUT-STAGING-20261110-START -->
## 上一段：输入原保护与条件 dead-backup（loop-014已PIVOTED）

**Goal ACTIVE；loop-014 PIVOTED（限定结论）；每个替换原语逐位一致。**

- **介绍**：继续沿物理 CPU 源追 H2D 末读，判断提前输出真正需要的代次边界。
- **到了哪里**：21份源码身份核对。继承的 synchronize_input_prep 在 target 准备前等待、结束后记录，覆盖 block table、zeroer 和 target CPU 源；已知晚到持久源为 proposer backup INT32[4]（每卡16B），CP qsl/suffix 为本次新结果。device metadata 另有 stage/join/reuse。
- **结果**：无 discard greedy 下每行至少一个有效ID，backup臂条件无活读；可仅消去 fallback fill/H2D，作为提前输出协议深入，仍属**待确认**。原 C8 回复 rank0/8 的2641/2677条日志不含 discard 数/all_greedy 标志，受限分支的实际资格未证。原7–15ms仅重叠组件预算，未扣新协议成本，不能全额授予此分支。原样 early-copy 补丁继续**已停止**；扩展复用栅栏或 live-source snapshot/ring 亦待确认，需支付 lost host overlap / event / backpressure；不能预设全部输入双份。
- **下一段**：共享 metadata / host submission 的较大交付联合定价，保留本段全caller/委托末读、实际资格和native逐位缺口即可继续。随后回末层attention/FFN producer。不重复自动快照实验。
- **环境**：新NPU0、推理请求0、新profile0、远端写入0，无部署变更。AST遍历顺序reader错误保留，修正后仅review通过，未认领实现或性能准入。

[完整判断](evidence/input-staging-review.md) · [行段与所有权前沿](evidence/staging-execution-frontier.json) · [协议修订](evidence/early-output-generation-protocol.md)

<!-- INPUT-STAGING-20261110-END -->

**上一轮更新（2026-11-10北京，UTC Run2026-11-09）**：28源/72旧窗口，只9个TP0回复等待。条件short TTFT3.395%、P90.5373%；native pinned H2D否定自动快照，原样early-copy补丁已停止。值得深入的是提前输出连同输入代次lease，当前待确认完整源/末读、回压净成本和native集成。a1 setup错误保留、a2匹配16worker且独立代次控制byte-identical；无模型加载/推理请求/部署变化。Goal ACTIVE。

[具体判断和下一步](evidence/sampler-output-review.md)

**本轮更新**：终端18源与真实读者/下轮状态已展开，DSpark dead-bank有具体未部署补丁（320MiB/卡）；native资格待确认。完整Post/head/verify独立1%预算低，结束对应小方案；更大末层与host-device交付仍待确认。旧compact原语位差不以最终readout一致豁免。历史reader失配保留error，精确Git blob及原receipt已核。Goal ACTIVE，下一处完整sampler/RNG/block-verify。

[判断与下一步](evidence/terminal-execution-review.md)

**当前增量（Goal ACTIVE）**：DSpark真实21源/128-top3-3层，五轮replicated W2和BF16 Add边界已核。整score原请求预算先全免仍低，只停止独立小路线，不为它再做cache trace。aux/main live-row、三decoder/EP和大提交仍有具体缺口。loop-014开始terminal全部读者、verify/async/scheduler/下一输入；MTP bank源码容量320MiB/rank尚未证明可删。

[开发门槛/三类处置](evidence/dspark-execution-review.md) · [下一处](evidence/terminal-entry-frontier.json)

**前次Engram增量（历史快照）**：Engram两层consumer→UVA/TP-SP/hash/退役已推进。保留E8M0的新容量路线有18.43GB明确逻辑容量与编码255反例；修正CPU枚举通过，NPU/成对加载/代次缺口待确认。停止同预算小方案和未改变旧失败配方，保留owner/提前Wkv/HC大parent。当前loop-013扫DSpark的5次Markov串行反馈及target接受拒绝状态。

[Engram开发门槛/三类处置](evidence/engram-execution-review.md) · [DSpark入口](evidence/dspark-entry-frontier.json)

**前次SMLA增量**：SMLA按真实选中KV/partial大交付判断，direct或跨层panel+首partial保留具体K/速率/逐位与生命周期门槛；不要以首partial单项低值剪联合范围。shared M416以服务custom绑定a3为准，a2只有其独立进程资格，整个DFC共驻尚未测。[SMLA](evidence/smla-execution-review.md) · [服务shared原语](evidence/shared-4cube-service-gate-summary.json)。全图当前续扫Engram。

**MoE当前开发门槛**：完整shared4/8已有M416真实权重、正确BF16 scale逐位与actual BlockDim限定事实；下一开发缺口为真实DFC共驻/争用、其它M/40层逐位及图/格式设置。前缀裁剪/deferred返回融合需保留舍入和arena末读。原样空credit变体已有native价值失败，不重做。全链路独立扫描现进SMLA实际kernel。[原语门槛](evidence/shared-fourcube-primitive-gate.md) · [返回消费者](evidence/moe-return-consumers.md)。


**当前合同与断点**：每个替换原语逐位一致。C08已测Wkv切M与stock AGMM直接替换已停止，不重跑原配方；保留原算术的重排继续判断。D-A2 OA大行有量级，先核完整权重逐位和净增成本；下一独立前沿为SMLA真实选中cache与QK/PV消费者。[native/停止依据](evidence/attention-transport-review.md) · [输出交付/缺口](evidence/attention-output-review.md)


适用范围为冻结 DeepSeek‑V4.1‑Flash W8A8、CANN 9.2.0、910C。下面是建议分工，负责人实名和结果由团队认领时填写。编号沿用研究台账；F08、F03 来自历史 46 条 CSV。首批有两条原型线，四项限范围的排查，六项都可以独立领取。

| 顺序 | 工作项 | 起始状态 | 建议主责 | 认领人 | 结果或证据 |
|---|---|---|---|---|---|
| 1 | C09 SMLA 分页 KV 直接读取 | 待领原型 | Attention / Ascend C | — | — |
| 2 | C08 保留原数值接口的producer重排 | 待确认协议；旧切M/AGMM直接替换停止 | 并行通信，Attention 配合 | — | — |
| 3 | F08 大行 OA→OB 同核交付 | 待查 | Cube / GEMM | — | — |
| 4 | C11 QLI 按有效 key 压实评分 | 待查 | Indexer / 向量算子 | — | — |
| 5 | C07 mHC 主计算与工作区复用 | 待查 | 归约 / 数值精度 | — | — |
| 6 | F03 小批量 slot 批写扩形状 | 按普通对话目标领取 | Runtime / 图调度 | — | — |

[三行进度记录与上传方式](README.md#阶段说明与上传流程)。阶段结果写回同一优化点，工作单同步下一步。

## 现在开工

**C09 · SMLA。** 先从已归档的长提示请求选取真实有效 K 和分页顺序。在相同输入下比较原实现（Vector 把 KV 写入 `kvMergeGm_`，Cube 的 QK/PV 回读）与一个直接分页读的最小变体。覆盖正常与逆序物理页及回退路径，保持 mask、索引顺序和归约语义。第一份交付为可复现用例、输出对照，以及每个 shape 的原版/新版整核 µs、波动和净差。正确且净省超过重复测量波动，再做首块 PV 直写；若稀疏读降速吃光节省，关闭这条直读路线。17.78 ms 是直读加首块交付在 K256 下的条件请求估值。[代码和成本](evidence/S07.md)

**C08 · 保留原数值接口的producer重排。** 已测原样ND切M及stock AIV AGMM直接替换有位差，按用户合同停止。下一缺口为真实root/P2P协议及收集、窄发布、争用后的净价，或保留原归约的流水；不继承旧30.27ms，不重测失败配方。[当前判断](evidence/attention-transport-review.md)

## 排查后决定是否开发

| 工作项 | 第一份交付 | 转开发条件 / 停止条件 |
|---|---|---|
| **F08** 大行 OA→OB | 选一个归档大 prefill shape，给出 OA 四块输出怎样在同核交给需要 1024 通道的 OB，列出舍入、L1 占用、权重搬运及重叠后的净时间。 | 数据放得下、BF16 边界可保留且预算优于原配对时间，才写该 shape 的原型。M24 已有权重搬运下限 81.64 µs，超过原配对 67.00 µs，跳过。[OA/OB 原始判断](https://github.com/wio1997/Inference-Foundry/blob/2557e99b66657511e7eb0bd66d57cbeae6a1f346/projects/910c-deepseek-v4.1-framework-investigation/oa-ob-shared-cube.md) |
| **C11** QLI 按 key 压实 | 复用四个消费层的真实候选分布和当前共享成员基线，估出省掉的打分时间，扣除 gather、压实和次序恢复；记录并列、filler、非有限值怎么维持。 | 代表性 shape 有清楚的局部净收益才转原型。在历史 223 波情景下，四次 QLI 每波净省 0.834 ms 才约合整批 1%。不能用减少 50.09% MAC 代替耗时。仅重做成员共享就关闭。[QLI 分析](evidence/S07.md) |
| **C07** mHC | 找出 HcPre 中一块可删的主要工作，交付一份按 shape 的耗时分解和保持 FP32/BF16 边界的计算顺序。对照已有失败精度样例。 | 可删时间扣除替代计算后仍有明确局部余量，且通过现有数值反例，才转原型。历史 HcPre 合计约 2.846 ms/周期，局部净降 19.6% 约贡献该周期 1%。若只剩小 cast 或需复跑已有位差写法，停止。[mHC 依据](evidence/S06.md) |
| **F03** 小批量 slot | 在 B1 普通对话匹配已归档的 13/26 项 caller，列明哪一组 worker 决定图入口；先复用现有数据，必要时只补 13 项分支原调用的可删时间。 | 以两轮后续对话整段均快 1% 为目的时，26 项每步省 0.961 ms 的假设下，13 项还需净省约 0.880 ms/步。若连原分支可删上界都低于门槛，关闭本次扩形状。不重做 B4 的 26 项实现。[小批量数据](https://github.com/wio1997/Inference-Foundry/blob/2557e99b66657511e7eb0bd66d57cbeae6a1f346/projects/910c-deepseek-v4.1-framework-investigation/ordinary-input-and-slot.md) |

## 第二批：8 张确认单，覆盖另外 10 个机制

以下是待认领的排查工作。本批优先查 1–3；4–8 只在拿得到表中资料时投入。优先级是团队排期建议。每张先用已有记录交一份“命中次数 × 每次可净省时间”或局部微秒账。同一主机空档、同一张量和同一等待只计一次。1% 数字沿用原表的请求或周期分母，用于决定是否值得整网投入；算子局部价值另列。既无收益预算也无单个决定性未知时，暂存条件，不启动实现。

| 顺序 / 主责 | 路径与编号 | 首先确认什么 | 进入下一步的条件 |
|---|---|---|---|
| 1 · MoE + 调度 | 空闲 DP、末层或已退役 context 的无贡献行 C05 | 先查普通请求里 idle 波次、假 token 路由数和最后一个完成的 rank；远端发来的真实专家工作须保留。末层裁行另查原始 HC 快照的最后读者。 | 确有命中且少算发生在关键 rank，才原型化 mask 或裁行。若 100 次 idle，每次需净省约 1.859 ms 才到旧整批 1%；不把一个假路由当一次完整专家计算。[MoE](evidence/S08.md) |
| 2 · Engram / 缓存 | 跨层或跨步复用查表、WKV 与统计结果 C06 | 用已有请求按完整查询键、层/权重版本回放，统计可复用工作占比，再扣命中查询、失配和容量成本。 | 原 profile 中即使忽略新缓存开销，也需消去可缓存工作的 71.1%–96.3%（中位 75.4%）才过历史门槛。位置重复不能替代完整键命中。[Engram](evidence/S04.md) |
| 3 · GEMM / Router | Norm → Router GEMM → TopK C12 | 在真实动态路由中标出必要点积和可并入 GEMM 的中间写读。给出新 producer 的局部净省微秒，检查选中 ID、概率与舍入。 | 只有实质缩短主体才转原型。整段占参考周期约 1.88%，要贡献周期 1% 需净降约 53%；仅省 Cast 或 ID 打包已排除。[MoE](evidence/S08.md) |
| 4 · Attention + 通信 | OA 前移并直接生成发送布局 C10 | 查询新实现能覆盖的真实形状；核算每 rank 约 2560 MiB 权重副本对 KV 容量的影响，以及直接写 send 布局的新增成本。 | 旧方案服务命中为零。只有新方案可命中、保住 KV 容量且开销落在模型余量内才重开。与 F08 比较，同一中间结果只计一次。[Attention](evidence/S07.md) |
| 5 · Runtime / 主机性能 | 未变 block 表增量发布、请求元数据共同输出 C01/C02 | 一次对齐 host 提交、H2D 和图入口，分别标明哪些 copy 确实可跳、哪些地址和镜像必须保留，输出两条机制的去重预算。 | C01 旧整批还差 162.24 ms，C02 已定价的设备工作全免后还差 124.17 ms。只有可归因的新增关键时间能补差才继续。两项差额不能相加。[输入发布](evidence/S02.md) |
| 6 · Runtime / 服务 | target/draft 共用 DP 协调 C04；客户端 ready 与缓存退役分离 C15 | 分别找出“所有 DP 的计数何时确定”和“token 可发、缓存可回收各在何时”，检查取消请求、缓存页代际和仍在写的 tensor。 | C04 若 200 次命中，需每次净省约 0.930 ms 才到旧整批目标；C15 在普通独立代码请求需净省约 4.666 ms 才达该请求整段 1%。分别按匹配负载决策。[协调](evidence/S03.md) · [回环](evidence/S11.md) |
| 7 · DSpark / 跨层 | aux → main/norm → 三个 context KV 共同发布 C14 | 确认三路不同权重计算以外能删的重复读写，并扣掉转移 main 权重约 137.63 MB/rank、行重排和同步。 | 整个前缀原约 1.050 ms/同周期，需净省约 52.7% 才贡献该周期 1%。有足够净余量再画 kernel 接口；否则归档。[草稿路径](evidence/S10.md) |
| 8 · MoE / megakernel | shared 与 DFC、专家返回与多层草稿联合调度 C13；历史 F09/U08 | 要求明确一份此前没有计过的跨核写读或可重叠区间，连同占核、通信、返回顺序和等待成本报价。 | 与已有 megakernel 可行性任务共用结果。有新传递机制和扣成本后的正预算才做小原型。现有读者仍需全局写回、或只把等待当作可删工作，就停止。[MoE](evidence/S08.md) |

覆盖口径：原型 C08/C09，首批排查 C07/C11、历史 F08 和 F03（对应 C03 的小批量扩形状），本批覆盖余下 10 个 C 编号。历史 Sinkhorn 提前退出归入 C07 的数学检查，仅在已有数据支持提前达到所需精度时重开；SMLA 共享 KV 面板是 C09 的替代路线。固定文本不命中的视觉或随机分支沿用历史处置。完整依据见[最新机制表](mechanisms.csv)与[历史逐项处置](history-46.csv)。

## 团队验收与排期

首轮评审看每个真实 shape 的命中次数、原版/新版 µs 或有依据的净省区间、数值对照和证据位置。局部净省与请求收益分列。每个排查项只提交一次“转开发 / 暂缓 / 结束”判断；缺资料时写下会翻转决定的那一个量。原型稳定后再由性能负责人用同一运行版本、同一负载分别开关候选，比较首 token 与整批耗时。共享设备上的正式计时错开运行。

团队有一位算子开发时先领 C09；有通信负责人可同时开 C08；其余成员按专长领取排查项。原 slot 双输出、26 项批写、四层 QLI 成员共享沿用已有实现。Qb/index‑Q 单次动态量化共享、独立 cast/tiling、旧失败的 MoE 合并不单开任务。已有 megakernel 可行性任务独立维护，有相同交接时复用本表结果。[立项复核全表](history-46.csv) · [当前环境](README.md)

<!-- THREE-MEGA-PERFORMANCE-20261109-START -->
## 2026-11-09 三个megakernel性能补价

- **介绍**　K01/C08先建立同owner多kernel基线，K02/C09先建立direct/partial基线；K03/C13核可用尾部、Up吞吐、净成本与覆盖。原责任链及认领状态不变。
- **到了哪里**　三项均待确认，只有条件数值复算。未新增原型或设备实验。
- **结果**　K03 τ20/h5下ρ1→0.600ms，ρ0.9→0.486ms/周期；只20层即使Up全藏住也不足本周期1%。K01 50%暴露另扣5ms→25.272ms包络；K02等速direct留给OA净惩罚2.859ms。[公式、覆盖与端点门槛](evidence/three-megakernel-performance.md)。
<!-- THREE-MEGA-PERFORMANCE-20261109-END -->


## 2026-11-09 C08数值门槛更新

- **介绍**　C2逐调用和大producer的Wkv切M验收。
- **到了哪里**　C08仍待确认，原样ND切M的逐位等价配方已有反例；先审精度/同M方案，不能直接照旧估价认领部署。
- **结果**　1组真实权重ND比较有302个位差，整体后续NZ分支ERROR。C2单独全免C8信用≤0.636%；同M所有权方案重新计收集/窄发布成本，不继承旧计算省时。[证据与下一处](evidence/C2-deep-screen-and-Wkv-numerics.md)。
