# QKV原生反量化与贯穿Attention的head布局判断

2026-11-10，loop038续扫。**全链路未扫全；每个替换原语逐位一致；新原语准入0。**

介绍：接SWA/Qb join固定其它到达的.65%–.66%窗口包络，扩大到真实Q/KV完整输入交付，继续量化方法、CANN整数/反量化，再把head分布延伸到SMLA和O投影消费者；不在同一小epilogue内反复补测。

到了哪里：30个实际source pin，15份非空行范围及2份compiled JSON的指定字段，见[源审阅](qkv-native-source-review.json)。7份Python生产者当前SHA重新匹配；142个safetensors header与247条投影相关shape/dtype、量化描述已读，权重payload0。两个代表rank各516完整Q/KV段与QA/KV/Qb、query RoPE、SMLA、inverseRoPE、transpose/A2A和O投影ready时刻绑定；新候选仅用其中480段24行target样本。见[完整价格原件](qkv-joint-delivery-price.json)。

结果：只改Qb、仍回到原CP query布局且SWA/cache到达不变的小范围没有join信用；较大范围将head分布保留到Attention输出，可能直接交付原TP O投影，值得继续。严格精度门槛下，整数累加安全不等于最终BF16安全；CANN原反量化分支必须保持。所有价格均为已有steady短窗的条件敏感性，**不是完整请求收益或性能预测**。

## 实际分支、精度与计算共享

- 当前text config为40目标层、3层DSpark、64 heads、head_dim512、QR1280、eps1e-20。源`init_attention_projections`在DSACP时将Wq_b建为ReplicatedLinear；Wq_a/WKV亦replicated，三者bias=False。KV源2/8/14/20，index源2/8/14/20/24/28/32/36。
- 量化描述中Wq_a/WKV为FLOAT，Wq_b为W8A8_DYNAMIC；selector→AscendLinearMethod→scheme已追。Wq_a/WKV实际native是BF16 MatMulV2；Qb是INT8×INT8 QuantBatchMatmulV3。不能因为整模叫W8A8便共享一个量化输入，或把三次投影当同一个量化核。
- checkpoint目标/草稿attn Wq_b共43份I8[32768,1280]，scale header为F32[32768,1]。加载目标参数按BF16创建scale，运行native确为BF16 scale；必须保留**已舍入的BF16 scale**，不能用checkpoint F32代替。原权重transpose→NZ，当前存储[1024,80,16,32]。
- 每代表rank516个Qb全为`aclnnQuantMatmulWeightNz_QuantBatchMatmulV3`，输入INT8/INT8/BF16/FLOAT，输出BF16，raw BlockNum22/MixBlockNum22。**22不是tilingKey22**。安装910_93/NZ/BF16-scale compiled JSON列25种key，不能从shape、核数或包存在推断实际key；无loaded binary逐调用证明。
- CANN pertoken/basic的INT32矩阵结果经AscendDequant(channel scale)，再FP32乘token scale、最后BF16 RINT。OPT预乘两种scale，再乘FP32-cast INT32结果。不得以实数相等或无溢出替代原次序；AscendDequant intrinsic也未证为任意手写FP32乘法。
- K1280下INT8部分和绝对界为20,971,520，小于INT32上限，能排除该无bias点积的整数溢出；但不是所有这些整数都能精确表示成FP32。保留原INT32→FP32与后续顺序仍必要。原RMS BF16→DQ、Qb BF16→FP32 trig RoPE→BF16均不能自动约掉。

本机CPU有限数反例明确否定两种浮点公式的普遍等价：[原件](qkv-dequant-order-cpu.json)。dot=13,926,403，BF16 channel scale `0x3b71`，token scale `0x3d060c18`（取BF16 max4.15625×FP32(1/127)）：先dot×channel再×token的末BF16为`0x44d2`，先channel×token再×dot为`0x44d1`。dot可用865个合法INT8乘积及415个零填充构成。**这是IEEE有限标量公式反例，不是CANN原生运行、实际模型已发生位差或完整norm/DQ输入证明。** 它说明active key与原epilogue是会改变可行性的参数。

## 先价完整交付，再决定扩大范围

从Wq_a输入ready到query RoPE完成，保留实际KV侧并行：

| 两代表rank的target24，共480段 | EP0 | EP8 |
|---|---:|---:|
| 整parent span全免 | 4.6780%窗口 | 4.6644%窗口 |
| 保留原QA/KV/Qb三GEMM及串行次序，其余工作/间隙全免 | 1.02454%窗口 | 1.02042%窗口 |
| Qb原调用中位 | 42.05µs | 41.509µs |
| 同次attention head restore A2A中位 | 14.11µs | 14.9595µs |
| 同次较大input allGather中位 | 16.81µs | 14.41µs |

原三GEMM全保留的小范围已接近松上界，不能据完整parent4.67%把三个必要GEMM都记作可删。它不是任意资源重排的绝对界，更不是原8条不同prompt全请求或六ordinary预算。

Qb只拆head后再加A2A恢复旧query布局，保留SWA固定到达时，原cache在全部516实例均晚于Qb。免费Qb也不能提前join；新增通信只有被遮住或延长等待。停止这个**两个当前窗、其它到达不变的小范围**，不剪掉较大SWA、资源竞争或后续布局。

较大候选：保留各rank原QR norm/DQ结果，收集INT8 codes和FP32 scale；Qb每rank只读4096 channels（8 heads）的原权重/已舍入scale，产全24 query×8 heads。query和selected IDs转成同代全局布局，SMLA输出直接是原O投影所需[24,8,512]，不再绕回CP [3,64,512]。

原`restore_tp_heads`实际就是view[3,8,8,512]→permute/contiguous→all_to_all_single→[24,8,512]。扩展consumer边界后，才有原transpose和A2A的可删方向；单独Qb方案需要的query A2A不再插入。不能只删通信、仍给消费者原CP布局。

Qb单rank原weight payload40MiB→分head的5MiB，差35MiB；这是**读取范围**，不是已测HBM流量或释放显存。保留完整weight作其它shape回退，可能仍占40MiB；额外拷贝5MiB shard还会增加占用。正确NZ channel slice/descriptor及alias寿命尚未证明，容量信用0。

## 付通信、metadata和新SMLA成本后的条件价格

完整[324个情景及逐段原件](qkv-joint-delivery-price.json)保留负数。新增QR codes+FP32scale的紧凑原始payload为每rank3852B、TP收集30816B；32B padding/pack/事件成本不能凭字节数忽略。旧同次245760B inputAG只作新小payload通信的.5/1/2倍**参考情景**，不是测过的新32KB耗时或上界。96个实际QLI producer各付一次global selected-ID收集，并要求后续读者复用同代bank；不是每层都重新收集。

下表令新Qb耗时为原1/4（敏感性，**不是35MiB差线性预测**），新QR/自有index收集各取旧较大inputAG参考，保留原queryRoPE→SMLA间整段必要工作，另每段付5µs pack/metadata/sync；不改变原SMLA耗时：

| 完整consumer边界 | EP0净省 | EP8净省 |
|---|---:|---:|
| head布局贯穿SMLA/O投影，SWA工作不变 | 11.549ms / 1.22836%窗 | 12.078ms / 1.29922%窗 |
| 同范围且SWA native工作半省 | 24.874ms / 2.64564%窗 | 25.314ms / 2.72296%窗 |
| SWA不变，新SMLA慢5% | 1.11011%窗 | 1.17933%窗 |
| SWA不变，新SMLA慢10% | .991851%窗 | 1.05943%窗 |

这些相互重叠的情景不能相加。SMLA慢一点就会改变去留，原语精度和新布局的Cube/Vector分核也可能改变。没有把“新kernel只用1/4耗时”或“pack只要5µs”当作已发生的事实；没有采新profile、执行NPU或发送模型请求。

## 下一处合法边界与具体缺口

待确认——保留原Qb epilogue的head分区贯穿Attention。优先从target24/common TP8每rank3有效query的共同资格入手。已有全16 inventory在EP7/15少36个quant族调用、72个RoPE族调用，与draft20尾部/零query源分支一致，但该family计数不代替逐调用映射。草稿20、尾行、空DP、prefill/ordinary及不同bucket必须保持原路径或另建全组一致资格，不能让某rank单独跳过新增collective。

当前global cache metadata builder明确`build_query_metadata=False`，不能把它直接喂给新的全局query SMLA。local metadata使用device seq length扣除query suffix，CPU镜像可能只是上界；causal、sink、ori/compressed selected-ID、接受/拒绝与cache页代次必须保持。原RoPE local视图本来就切自global表，重复cached gather会覆盖仍被引用的进程级buffer。增加全局query/逆RoPE布局前需核末读、source压缩表复用和图稳定地址。

下一段继续SMLA合法head8分支、原QK/PV MMAD与softmax归约顺序、全局query metadata/96次index bank收集及后读者、原TP O投影物理布局。编译对象key/原反量化次序、完整替换原语位对拍、新SMLA成本和真实关键TP/DP/请求曝光均为具体缺口。**已确认可实现本轮0；全链路未扫全。**
