# latent 双缓存整段与保留 GEMM 的净价判断

## 介绍

从四个 KV 源层 2/8/14/20 的真实输入，追到 C2 FP32 双投影与持久 ring、BF16 latent、index K 的投影/归一化/RoPE/INT8 codes 与 half scale 两次发布，再到同一 latent 的原地 long-RoPE 与 long-KV 发布。范围先含所有真实生产工作，再判断尾段、原 GEMM 不变的 epilogue 和提前执行，未按旧候选清单限定搜索。

每个替换原语必须逐位一致。当前没有替换实现、数值验收或新的性能基线。该段是有限范围判断，不能保证全链路或所有融合算法已穷尽。

## 到了哪里

1. 五份入口源核对后，又只读取回/核对六份依赖（三份补齐，三份旧 hash 一致）；审阅范围单列于 `index-key-publication-source-review.json`，pin 不等于审阅。实际 `compressor_from_projected` 是 token-aligned 输出，保留独立原 RMSNorm，**不是**通用 `compressor()` 中已合并 norm 的另一分支。
2. C2 先 `out.zero_()`，只在完成 pair 的原 token 行存 BF16 pooled；未完成/空 page 行仍为零。pool 残余读取 FP32 ring，**pool 必须先于 ring 更新**，否则本段尾写覆盖本段头的历史残余。ring 仅写最后 min(used_len,32) 行，页面地址先扩 INT64，null page 受 guard 保护。不能把投影 M 行误读为 M/2 个紧凑输出，也不能直接省清零后继续原全行 norm/wk。
3. index wk 读取的是 long-RoPE 前的 latent。原主流先提交整条 index-K 链，再原地旋转 latent；新跨流必须等 wk 真正末读，不能以 Python 返回或 launch 完成为另一流的读取完成。pool/ring 两个方向的状态代次、global slot 和未来请求也必须保留。
4. key 的 DynamicQuant 先产 INT8 codes 和 FP32 scale，再将 scale 转 half。query 则在除法/编码前先 half-scale 舍入。不能把二者当同一量化原语，不能实数约掉 BF16 norm/RoPE/两种 scale 边界。RMS OOT wrapper 调 `torch_npu.npu_rms_norm`；native RMS、DynamicQuant、scatter 的内部算法本轮**未读**。独立低预算配方结束后，不为它们追加设备点证；大 parent 若重开需补这道门槛。
5. `dsa_v41.py` 的实际 cache 读者已再追：index source 从 `index_k_source_prefix` 取 paged codes/scales→QLI→原按位置排序 bank；非 index source 复用该 bank。long cache 从 `long_kv_source_prefix` 取→SMLA 分页 QK/PV。源 2/8/14 的 long cache 各供六层，源20供20–39层；源20 key 还供20/24/28/32/36各自 QLI query。cache 跨请求/迭代仍活，某个本地 query0 或当前层结束均不意味着 producer 可删。物理地址、QLI/SMLA内部与退役门槛复用已有对应审阅，并不冒充本轮重新逐行核完。
6. 910C standard adaptor/INT8 profile 资格与 A5 packed/folded 路径分开。磁盘源/包或旧 native signature 是身份与接线证据，不是 loaded Python/逐调用 tiling/二进制等价证明。

### 实际行量与旧计时匹配

原首 token native 表：16 rank ×4 eager body ×4源层 =256窗，224窗发布缓存，其中40窗本地无 query、仍发布缓存；32窗没有缓存生产。DP0 的 carrier416/816/817/818 对应 global cache rows3328/6/6534/3346，DP1 对应0/6528/12/6540。carrier816/817的小行不能替代正常 decode carrier24。

每个有效尾段逐序绑定十个调用，完整 C2 root 再含输入 cast、两 FP32 GEMM、zero、pool/ring，C1 root 含原 BF16 Wkv。133个 shape/dtype/DP/carrier 条件独立匹配。相同 RMS/scatter signature 还有其它读者，因此只核对子集次数和 min/max 被全表覆盖，**不宣称全 signature 次数/中位数相等**。首稿只选 MatMulV2、以及随后漏 cast/zero 的价格均已修订，最终原件/哈希见 profile 与 notes。

另取现有 CPU-plan 诊断 candidate cohort 的112个完整 rank 周期（16 rank ×7），每周期四源共448窗，target graph 为 global24/local3；38组独立 typed 子集匹配。它只有周期样本，**不是**原227波请求完整重建、干净 before/after 或六普通负载证据；没有从 padding shape 推断实际 slot 内容/完成 pair 数。

## 结果

### 已停止：同一独立小预算下的尾段/保留原 GEMM 配方

原 C8 first-token 价格采用既有 DP 到达区间、每 DP 最大 rank advance、原 eager 前缀次序；N8 P90 按原最大 TTFT 计算。rho 表示假定被删/隐藏比例，h 为每源新增成本。它是**固定原调度的乐观条件敏感性**，不是因果硬界或预计/实测收益；parent span 包含间隙和保留工作，并非全可删 CPU。

| 范围 | 全部免费条件 P90 | rho=.5，h=20us/源 |
|---|---:|---:|
| latent norm 起的十调用原生尾段 | 2.515ms /0.1686% | 0.937ms /0.0628% |
| 尾段及其全部间隙 | 4.180ms /0.2802% | 1.770ms /0.1186% |
| 完整 root 原生计算，含所有 GEMM | 14.192ms /0.9513% | 6.776ms /0.4542% |
| 完整 root span，含间隙 | 17.184ms /1.1518% | 8.272ms /0.5545% |
| **保留原 GEMM，其余全部计算及间隙** | **7.244ms /0.4856%** | **3.302ms /0.2213%** |
| 只提前整个 root 且理想完全重叠，min(root span,输入到 root 延迟) | 11.919ms /0.7990% | 5.640ms /0.3780% |

所以独立 norm/RoPE/quant/三 scatter、保留 GEMM 的 epilogue，以及只挪早一次生产段，都不为同1% C8首 token预算继续拆小变体。这里停止的是这些限定配方，不是删除真实缓存工作，也不是给其它调度/普通负载或完整 Attention 判死刑。

decode 诊断样本中，四源尾段 span 全免占 rank 周期0.4469%–0.6099%（中位0.5679%）；完整 root span 全免0.7467%–0.9972%（中位0.9448%）；保留 GEMM 其余全免中位0.6345%、最大0.6756%。完整 root 假设半省再付20us/源，周期中位0.3283%、最大0.3512%。结束这些样本下的相同独立小配方，不将该比例乘全231波/全部请求。

### 待确认：更大的同源交付、共享计算与 Attention 联合

- 将边界前移到输入 AllGather 完成，直到 long 发布的整段全免条件 P902.1327%；半省且20us/源为1.0449%。**此跨度包含必须保留的 query/SWA 工作**，只能说明整个输入/源生产/QLI/SMLA 联合仍值得查，不能当简单 source-move 的收益。需把合法 metadata/source-RoPE ready、主辅流的真正末读、GEMM/Vector资源竞争、新发布成本和下游 query readiness 一起排程。与旧 C2、Attention输入、host和QLI/SMLA parent共用预算，不能相加。
- TP8 同源完整 M 的结果共享可少做总算术，但一个 owner 仍须完整原 GEMM，再付传输/共识/ready；不能把八份累计算术当八倍请求信用。按行分摊再 AllGather 会改 GEMM M/tiling/归约，需要自己的逐位证明；旧 Wkv changed-M 已有位差，不能直接承接。整段全原始 span 仅1.15%且付成本后余量薄，当前不为 standalone owner 小方案新跑模型。
- 直接更改表示、跨层 panel/地址计划复用或 megakernel 可与已有 QLI/SMLA 高值范围相接。具体缺口仍为实际有效K/页序、原位数值边界、消费者/代次/资源和付成本净暴露；本轮没有把这些名字登记成已扫描或已实现。
- 六普通负载与原C8完整 decode 的**逐源交付原生时序**尚未单独绑定；不得拿全局6/12行 eager 或112个 diagnostic bank24周期外推。已有低预算判断限定其 cohort；若该负载的整段预算变大再重开对应 native 门槛。

### 已确认可实现

本轮没有新增该类准入。源码可表达、独立 CPU 地址轨迹或乐观价格都不替代逐原语位一致与真正实现验收。

171 已核主机名；同 owned 容器/镜像。本轮仅只读 source/config、复算本地已有 profile；模型/NPU计算、服务请求、新profile、远端写入、服务改动均0，未操作其它profile controller。
