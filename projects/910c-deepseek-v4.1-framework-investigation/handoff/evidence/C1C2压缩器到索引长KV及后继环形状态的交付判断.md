# 2026-11-11 C1/C2压缩器完整交付：双cache与环形状态，独立小方案停止

介绍：从C1/C2真实输入投影追到索引K/FP16 scale、长KV两路发布，及下一迭代FP32环形状态和QLI/SMLA实际读者。到了哪里：16当前源码身份、15原Python协议/有限控制；旧两代表卡96完整源交付，原C8全16卡历史分别匹配。结果：C2绕过Compressor.forward，零化/未完成行与整页状态都是合同；同预算独立后处理/池化写环/压缩历史小方案停止，更大输入/query/SWA/QLI/SMLA联合交付保留。每替换原语逐位一致，native新增准入0，全链路未闭合。

真实入口与角色：zero-based2/8/14是C2 KV+索引K源，20是C1 KV+索引K源。24/28/32/36虽生成新索引query，却复用20的K，没有额外K投影可删。C1走compressor.forward里的BF16 wkv→原RMS；C2在attention._write_compressed_source直接做一次共享hidden.float()、两个FP32 nn.Linear(wkv/wgate)、pool_projected与原RMS，不能从forward单入口宣布扫过C2。旧generic _proj_kernel/容差autotune和TOKEN_ALIGNED=False融合norm不是选中的替换原语。

完整可见输出：compressor_from_projected先验证dtype/shape/power2/contiguous/同device/bounds，再把全部token-aligned BF16 out归零；即使tokens/batches/maxquery为0也先零化。live非NPU在零化后报错。池化只在原完成行写BF16，其他行零；之后原RMS的所有行仍有后继调用，不能仅以未完成行不写cache就删除raw pooled scratch或改变输出。FP32 ring更新只覆盖每段最后min(used,32)行，其余字节不动，used0/skip不写状态；start0无需旧history但不等于清空整个ring。原wrapper同流先pool后update，再RMS。

状态/跨迭代边界：odd start第一组读上一even token，环形offset用全局物理page且先转INT64。start1/used32时，池化需要旧slot0、尾更新会把token32写到slot0；朴素跨program合核有WAR。exact-SHA旧证据已有“每request/feature tile先读完再写”的完整状态融合与数值对照，并要求无跨owner物理冲突，否则回退原全局pool→update路径；本轮不重跑。compact三even槽只在retreat≤5、四槽≤6等owner/初始化/单调前进/全部读者合同下成立，三槽retreat6已有反例。延后accept再提交只把候选前驱的生存期移到暂存，不会抹去信息；SWA仍可能决定总page分配量，逻辑字节变少不等于容量兑现。

实际A3元数据：c2_state builder的build_c2_metadata closure产生5行(start,used,outputbase,inputbase,globalpage)、complete mask、sourcepos与cos/sin。start=seq_len−原query_len，而非seq_len−截断used；used同时受actualreq/actualtokens/skip约束。sourcepos为有效odd completion−1，否则0，不能用当前odd位置的RoPE替代。A3走原Torch closure，独立c2_ring_metadata Triton fused helper只在A5 packed分支调用；不能把它列为A3已运行融合。旧任务/同cache组的准备可能共享，未绑定真实group前不按3个C2源重复计费。15控制执行原closure和wrapper的主体，验证padding/skip/缺表、零化/错误顺序和pool→update flags；native launch、RMS/exp、capture和dtype数值仍未验收。

两路cache与真实读者：原latent先交indexer.update_keys，完成BF16 Wk/norm、index partialRoPE、dynamicINT8、FP16 scale转换及K和scale两次SK写；再对latent本身做long partialRoPE并SK写长KV。必须保留这个次序，不能让in-place long RoPE污染index输入。C1直接复用已取query/cache cos/sin；C2用metadata的completion−1表，缺表才fallback。A3 SK保留实际page stride、cache dtype、values.contiguous和[-1,-1]跳行；三cache输出不保证平面连续、不同名字也不证明物理独立。source/index20和C2源交QLI；其他层复用selected_indices，后四个C1 query源只重算query/selection。QLI候选源/过滤、topk lengths和稳定输出及空cache填−1仍保留；SMLA等ATTENTION metadata后同时读原SWA与long cache。query0的CP rank仍可用全cache hidden更新源，不能按无本地query提前退出。

数学与共享：两次FP32投影可研究共同搬入hidden/权重布局，但hidden.float已共享，不能再收一遍收益；改N/tiling或按完成行改M均需独立逐位证据。ratio2 softmax加权可以写成sigmoid/lerp实数公式，但FP32操作序不可抹掉：同分数、v0=16.000001907348633/v1=−16，原两项乘.5再和为2^-20，lerp序为2^-19，最终BF16也不同。该有限IEEE反例只否定通用逐位恒等，不是模型实际值/native失败声明。池化BF16边界、原CANN RMS/动态量化/两个RoPE及所有scale/零行副作用都须保留。全M owner能少重复FLOPs，owner本身仍算完整M且加通信，不能宣称8倍源延迟收益；要连同query布置/资源竞争/原顺序输出交付定价。

整段先价：旧配平C8 EP0/8、globalM24/localM3中匹配96条完整源链(72C2/24C1)。投影→双cache及ring完成native均0.709665%窗口，保留原GEMM后其余0.481619%；单pool/zero/update为0.054702%。连所选QLI/SMLA读者native为1.652394%，起止span1.681389%，包含必须的索引/attention数学，不能全免。假设仅省非GEMM的50%、另付10us/源，净1.753–1.789ms/0.188616–0.190317%窗口；付20us降为0.136983–0.139264%。均是敏感性情景、不是实测合核；未知六普通曝光和请求信用为0。

原C8另算：已有全16卡首token固定DP arrival模型，完整source native全部免P90最多14.192124ms/0.951307%，保留GEMM则4.215676ms/0.282580%；112 decode周期完整source native全免最多0.994572%周期。独立三C2 publication bracket全免也≤0.708251%本地周期，包含必要工作。因此相同预算的小后处理/池化写环/compact-history配方沿旧决定停止，不追加原样NPU测试。source parent span固定arrival1.151833%和input-ready→source完成2.132705%首token宽包络含CPU空隙、query/SWA等必要工作；更大整段联合调度另有边界，不能套小family停止，也不与metadata/Engram/QLI收益相加。1%是历史独立优先筛选预算，不是用户新硬阈值；不把代表窗或固定arrival模型外推六普通/全请求。

结果分档：已停止——同旧预算独立source尾部/池化写环/compact小方案、擦掉中间舍入的通用lerp改写，以及已失败changed-M配方；旧C1 slot双writer是metadata，不能按C1名字混入此投影价。待确认——输入/原M-owner/query/SWA/QLI/SMLA联合交付：需真实全16rank ready/物理page与stream资源、保留原数值tile/全部输出、packet通信与新控制成本、六普通/请求曝光；若这些不会改变去留就不补测。已确认可实现——本轮0新增；旧全状态pool/update有限native实现资格按其guard独立保留，但性能决定仍为停止。既有metadata/gid12、add-HC/sharedDFC、Engram owner、Markov/host-context继续保留具体缺口。

下一条待扫：loop047下一Run：从Engram embedding.forward/lookup实际receiver继续，沿ngram/hash→CPU-UVA或device gather/dequant→embedding投影/门控与HC消费者，核冻结storage/scale布局及后继cache/state读者。先定义完整输入交付与跨TP owner/计算共享/布局/量化/megakernel，再按exact-SHA旧证据去重并匹配C8/六普通价；原样低值gate/copy/changed-M配方不重测。把原递归worklist的字段/native/closure边逐项归位，具名缺口和计数不代替全链路闭合。每替换原语逐位一致；本源码Run无新Torch/NPU/模型/profile/请求/服务/远端写入。
