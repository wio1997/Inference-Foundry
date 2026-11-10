# SMLA分核包与原顺序交付

2026-11-10，loop-034。**Goal ACTIVE，全链路未扫全；每个替换原语逐位一致。** 本轮只读171源/目录并在本机运行标量控制实验及旧profile复算；NPU、模型、服务请求、新profile和显式远端写入均0。

## 介绍

从持久metadata包反推完整交付：`长度/掩码/平台/确定性 → AICPU分核 → FA范围/可选FD描述 → arch22每核QK/softmax/PV → BF16 attention`。分核包虽小，却决定哪些query和KV由哪个核读、是否跨核归约，因此先核原producer和消费者，再判断selected-KV地址计划、跨层共享和大Attention融合。

## 到了哪里

[21份实际源身份及审阅范围](smla-schedule-source-review.json)：8份入口沿用同SHA，新增13依赖中9匹配旧件、4份实际op-api新取；171主机名、owned容器/镜像核对成功。AICPU实现25–1638及header相关范围已读，接上实际ACLNN attrs、Python按ratio共享和arch22消费者。其它host checker错误分支只核所列范围；不以源hash/本机fixture证明loaded二进制全部语义或实时确定性状态。

| 交付/条件 | 实际代码结论 | 对融合的影响 |
|---|---|---|
| caller共享 | `_publish_task`已经按同一batch的`smla:c0/c1/c2`共享；生成后copy到持久buffer，再按task/event发布 | 不存在新增“38层各省一次metadata生成”的预算；下一迭代仍有地址/代次和末读义务 |
| 输入和attrs | ACLNN对9项可选输入做检查/contiguous，从平台取Cube/Vector核数与Soc，读取`ACL_OPT_DETERMINISTIC==3`，传给AICPU | 36/72为包的容量上限，不是实际launch核数。只按B或指针复用会遗漏长度、mask、ratio、平台和确定性 |
| 包布局 | 分配1024个INT32；`SmlaMetadata`为36×9＋72×8＝900词；GenMetadata清零/写该struct | 后124词未被该函数定义。CPU sentinel见证保留，不把它称成已清零或可删的计算。完整1024词原语逐位资格尚未建立 |
| 默认分核 | ASCEND910下CSA G64每M块一query；dense SWA最多4query；ori_sparse SWA另设Mbase=G64 | 默认supportFd=false按整batch/整行分配，不进入按块/强制跨S2分配；DSpark不能套dense SWA的4query几何 |
| 真正有效K | AICPU没有selected ID或page table输入；CSA consumer另取causal界，再二分首个负TopK | metadata里的512容量/块数不是实际K512证明；物理地址计划最早接QLI结果或第一真实reader，而非仅改这个AICPU包 |
| FA/FD消费者 | CSA/SWA Init读FA起止，Process再读enable；选中arch22直接读取7类FA字段，不读FD描述/另两词 | 元数据解析结束并不退役仍被循环读取的query lengths、TopK、page table或KV。跨38层共享包仍等最后层/rank的packet末读 |
| 空KV/空query | 全空标量排程保留一个FA核；consumer保留原zero-output与preload排空；Python无本地query会zero持久包、不调此AICPU | 不能跳活cache producer或只因主输出为空删全部同步；跨流仍保存2轮flush、3个RunInfo槽及原flags/ring |
| 原workspace | host按3个KV merge buffer申请；C24/BF16条件为36MiB，arch22 FD staging为0源公式 | CSA Init末尾`offset +=4`之后没有后续区域使用；它不是第四bank或新增12MiB释放机会。物理峰值/释放沿用loop033缺口 |

### 必须更正的旧解释与level3资格

旧[SMLA报告](SMLA内部交付与候选判断.md)“arch22入口不向SMLAType传FLASH_DECODE”的**原因表述撤回**。入口确实传它；实际host `GET_TPL_TILING_KEY`首参数固定0，arch22模板选择也只编FD0，因此“本包FD0”的事实保留。未传入/消费的是`BATCH_CONSISTENCY`的算术语义。6个旧compiled key身份保留，不把编译key存在当逐调用选择或新FD资格。

[CPU控制实验](smla-schedule-cpu-a1/result.json)编译原323–1631标量函数体，vendor Tensor ABI用小fixture代替、仅公开private访问，直接调用ParamsInit/BalanceSchedule/GenMetadata。**不测试真实Compute/Prepare/ParamsCheck入口，不是NPU原语数值实验。**

| 显式fixture | FA核 | FD任务/FD vec | S2边界 |
|---|---:|---:|---|
| G64、q3、C2、默认 | 3 | 0 / 0 | 均0，整query |
| 同参数、level3 | 6 | 3 / 48 | 有非零边界，跨S2 |
| q24、默认/level3 | 各24 | 各0 / 0 | 此fixture未拆S2 |
| 空KV、q3、默认 | 1 | 0 / 0 | 保留空输出路径 |

因此“只打开level3就能安全跨核S2并保持同位”的配方停止。需要新完整producer/consumer归并及每原语位门槛；当前arch22 FD0不会因为packet带了FD描述就自动消费它。这是**源级分支资格反例，不是已观测服务故障**；未证明冻结服务实际处于level3。

同一组fully capped q3/C1-C2 fixture的900个已定义词完全相同，但未定义尾部、其它有效长度/窗口/平台/代次、全部1024词和NPU行为未验，不能据此接纳通用C1-C2缓存或新原语。

### 值得继续的大范围与有成本估价

具体方向是：保持原TopK、metadata、QK/PV MMA分块、softmax顺序、FP32 partial和BF16边界；在QLI输出或首reader发布**原实际gather顺序**的64位地址＋32位transfer描述，后续相同(KV源、index源、page表代次、query/mask/行域)使用。让Cube直接读选中KV，并联合首个非末FP32 partial交付；不认领已经同launch的QK-softmax-PV为新融合。

[54组有成本情景](smla-address-plan-paid-price.json)先精确重现6项旧parent值，再支付12B/条的首generation写、每层QK/PV两次plan读、长度字、1或8倍charged-byte情景和每调用h。原逻辑ID和metadata完整保留，不预扣旧索引读取。K512最大818query的plan payload约4.793MiB/rank；不是总pool/物理释放。

下表沿用原C8诊断1840 native/2560窗口、30 typed组、固定EP到达和乐观TP信用模型；毛交付50%曝光、2405.30GB/s为旧双向charged-byte参考，**不是新稀疏Cube实测速率**。结果是最有利请求的条件信用，不是预计净收益、请求P90或六普通结果。

| 有效K | 新稀疏速率/参考 | h/调用 | plan流量倍率 | 条件信用 |
|---:|---:|---:|---:|---:|
| 256 | .75 | 5us | 1 / 8 | .875% / .727% |
| 512 | .75 | 5us | 1 / 8 | 1.395% / 1.101% |

同h5us下达到1%的新稀疏速率门槛，K256约84.80%/100.22%，K512约63.42%/71.67%（分别对应倍率1/8）。54情景保留13项负的请求信用，未将回退归零。全部模型局限、原请求值和source SHA均保留。

这说明K256/.75这类已付成本的同独立1%子范围可结束；K512范围仍值得深入。**真正改变去留的是有效K、原序ND2NZ/DMA issue与charged plan速率**，不是继续拆小metadata零填充。不能用这张情景表裁掉更大host/QLI/SMLA联合parent。

## 结果

- **已停止**：新增38次packet重复生成的信用；只删包尾/FD字段/死offset的同小预算配方；flag-only level3安全跨S2；上述K256/.75/h5同独立1%情景。停止范围不覆盖大Attention、其它有效K/负载或新归并实现。
- **待确认，继续深入**：原序selected-KV计划→Cube直接L1交付＋首partial。缺原pair/大stride fallback、ND2NZ目的layout/尾padding/flags、真实有效K/稀疏issue率、所有源/页表代次、原语逐位和净价。下一loop继续这些实际consumer，原metadata暂保留。
- **待确认**：BatchFrame/host/capture联合、条件C1-C2共享、DSpark ori_sparse直接交付、level3选中时的完整归并。各自缺口见[前沿](smla-schedule-frontier.json)，不因本段有限源码和CPUfixture变成已完成。
- **已确认可实现**：本轮新增0。CPU控制见证与条件价格不作替换原语/服务性能准入。

下一入口：[selected-KV原序到L1](smla-direct-kv-entry.json)。继续从Vector的GetKeyGmOffset/CopyInKv、3-bank GM ring到Cube QK/PV原Nd2Nz目的布局、MTE/LoadData/flags；已有8个CPU地址轨迹仅作原序参考，不证明新ND2NZ数值或带宽。
