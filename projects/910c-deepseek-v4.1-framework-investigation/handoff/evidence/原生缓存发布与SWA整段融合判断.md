# 原生缓存发布与SWA整段融合判断

2026-11-10，loop037。冻结W8A8、CANN9.2.0、910C。**全链路未扫全；每个替换原语必须逐位一致；本轮新增准入0。**

介绍：沿index-K代码/half-scale与long-KV双发布的实际native继续追踪，并按完整消费者交付扩大到每层SWA缓存发布。目标是少算、少交付和提前就绪；原有量化共享与主辅流重叠不重复计价。

到了哪里：39个当前source pin，29份非空审阅范围，见[精确行表](cache-epilogue-native-source-review.json)。源码包括worker注册、OOT dispatch、实际DSA caller、custom RoPE/ScatterSk的910C分支，以及已安装CANN RMS/DynamicQuant的部分分支。下载未读的10份只计身份，不计覆盖。两个配平steady代表rank EP0/EP8的已有CSV绑定48段十算子链、516段SWA三算子链；保存逐段行号、task ID、shape/dtype及价格情景，见[逐段原件](cache-epilogue-balanced-sites.json)。

结果：独立小index-K/long-KV epilogue仍停止；较大SWA交付值得继续查Qb重叠、join与实际原生位合同。**窗口上界不等于请求收益**，不拿本轮4条prompt镜像数据去除以旧8条不同输入的请求端点。

## 已扫的实际调用和合法边界

1. worker初始化调用`register_ascend_customop(vllm_config)`，RMSNorm映射AscendRMSNorm；A3 standard capability启用`custom_ops=[all]`。CustomOp有禁用回到native的分支，310P有独立override。当前source与RmsNorm原生shape一致，不当作逐调用Python对象或已加载二进制身份的证明。无residual的AscendRMSNorm调用`npu_rms_norm`、丢弃rstd、可另加`bias_loaded`；不能假定权重常数、bias永远为空或norm可约掉。
2. 实际`dsa_v41.scatter_cache_sk`直接调用`torch.ops._C_ascend.npu_scatter_nd_update_sk`，先`values.to(cache.dtype).contiguous()`。此前generic DeviceOperator散写范围不是这条实际callee；本轮明确更正，旧原件保留。
3. 910C特殊RoPE在已列shape/FP32 trig守卫下：单KV head推导key12，query广播推导key11。原核只修改末64维，保留前448；BF16输入转FP32，FP32 cos乘、交换分量sin乘、交错负号、FP32加，最后BF16 CAST_RINT。cos/sin已在每行跨heads广播，不能再计重复64份加载。不得通过BF16三角表、FMA或实数正交公式自动取得逐位准入。
4. RMS原生generic/single-row有原归约、sqrt/div与乘gamma、BF16 CAST_RINT；whole-reduce BF16路径显示不同向量序列和CAST_NONE。仅凭算式或API名字不能选择其中一种。**尚缺实际调用tiling key/编译对象绑定**，不能宣称改归约次序、融合RMS+DQ或省中间舍入合法。
5. DQ single-row key100：BF16提升、abs/max、FP32 `max*(1/127)`作scale、独立`127/max`乘原输入，再RINT整数→half→INT8；multi-row key10使用INT16中间。原key更新先量化code再将scale转FP16；query的先half-scale方案不同。RoPE位于key RMS和DQ之间，现成RMS-DQ融合名字不能作为替换证据。实际key、零/NaN/Inf/饱和与half-scale转换仍需绑定和逐位验证。
6. ScatterSk HP核已融合线性地址与散写，无SyncAll；按indices分核，重复目的地的跨核胜出不确定。HP检查的是最终linearIndex负数，不是每维负数；INT64大索引key30另按目的区间分核、逐维负数检查并顺序扫描。当前合法builder的双负sentinel可跳写，但不能据此声称任意负坐标语义相同。换分核之前须证明实际slot唯一或保留原合法行为。
7. SK wrapper传真实`var.strides()`。host只允许首轴扩大的stride，尾部连续；device按uint64 base处理layer-outerplane。当前INT32、17304×128总索引小于2^24，source推导key11，仍不是loadedkey证据。tiler launch所有48个vector cores，HP有用行owner为min(48,N)，当前N24/20；不能把空owner直接当可得吞吐。source workspace仍宣告sys＋两个对齐索引数组，HP并未用这两个数组；物理显存/allocator收益未证明，不按层累加。

## 同负载已有价格与去留

本轮配平参考已结束并恢复，16.0908秒是未profile请求中位；下表仅来自最多12迭代的steady窗口。EP0窗口940.19375ms，EP8为929.6495ms。选中类型内同流连续且native task ID连续；被排除的CSV任务类型是否介入另列，不冒充全runtime event因果链。

| 完整交付 | 每代表rank段数 | native全免占窗口 EP0 / EP8 | 半省、每段付5µs | 判断 |
|---|---:|---:|---:|---|
| norm512→WK128→keynorm→keyRoPE→DQ→code/scale发布→longRoPE/发布 | 48 | .431862% / .427015% | .190404% / .187691% | 原loop031独立小预算继续停止；保留更大parent |
| norm512→RoPE512→SWA cache512 | 516 | 3.079767% / 3.065219% | 1.265472% / 1.255085% | 待确认，继续查实际曝光和逐位资源 |

SWA包含480段24行、36段20行；数目与40层/3层各12次吻合，但**不靠数目单独给每段贴target/draft标签**。十算子链与SWA三算子实例互不重合，也没有覆盖完整query/GEMM/input-ready parent。原native span上界分别为.432938%/.428105%和3.082367%/3.067832%；并不证明间隙可省。

成本足以改变去留：SWA只省25%且每段付20µs时净值为−.327704%/−.343791%窗口；半省/h20µs仅.442238%/.422514%。因此3%不等于稳拿3%，也不据此启动开发。wall Duration和aiv_time分别保留；后者是PMU活动字段，缺失值计数单列，不能把两者差全部判为launch/等待或删除信用。

独立index尾段此前已用原C8全请求端点判过：保留原GEMM、其余全免仅.4856%，仅挪早理想重叠.7990%；本轮不重开同预算小试。更大source/input-ready/query/SWA旧parent的2.1327%全免与half/h20µs1.0449%只是原cohort条件值，不与新窗或其它QLI/SMLA范围相加。

## 有价值的融合范围与还缺什么

**待确认：保留原算术的SWA直接缓存发布。** 每行在UB保留norm BF16输出与RoPE最终BF16结果，只向目的cache交付一次完整512维，条件下可删2304B/行：norm GM写1024＋RoPE局部GM读写各128＋SK更新GM读1024。24行55,296B；当前516段合计28,200,960B/代表rank/window。RMS输入、gamma/cos/sin/indices读取、最终cache写仍保留；不是实际HBM交易量或内存峰值下降。rstd和中间norm若有其它可见读者，必须保留。

源`multistream_preprocess`已经将Qb Cube与KV norm/RoPE/cache Vector重叠，末尾main.wait_stream(aux)才给attention放行。实际收益取决于谁最后到join；不能按516段总时长直接兑换请求省时。只有query为空时走cache-only更新，仍必须发布cache；A5 packed写主流分支由硬件守卫排除当前910C资格，不删合法分支。

**待确认：raw latent的非原地RoPE→cache与indexWK并行。** 同一原latent先由indexWK读取，当前longRoPE原地改写，形成WAR。直接缓存发布若不改raw latent，可能放宽边界；仍须证明所有alias读者、source/ring代次、cache发布事件和净资源。旧“仅挪早”小范围仍停止，不重复计一份收益。

**已确认可实现：本轮无新增。** 尚缺实际RMS/DQ tiling/loadedbinary、逐原语位对拍、slot唯一与skip/stride、UB/register/flags代价、完整TP/DP关键曝光及六ordinary/真实prefill边界。每个缺口具体保留，不把“已读名称/已有代码”当作完成证明。

下一段按[SWA parent入口](swa-publication-parent-call-pins.json)沿全部已有native task（EP0/8各35,664行）与现有Host/runtime关联继续查join和actual reader，未发新请求或新profile。此次扫描只读当前源码和已保存CSV，远端显式写入、Torch/backend/NPU导入、模型加载、推理请求、采集、服务动作均0。所有源码下载与数量不构成全链路穷尽证明。
