# D-A1深入判断：C2真实链、请求信用和Wkv换M反例

2026-11-09。全链路搜索仍在进行。本轮从冻结代码确定的D-A1交付范围继续，不以旧候选列表确定范围；旧profile只在两份FP32投影及消费者已经确定后用于定价。

## 实际消费者和合法分支

对原始v168/C8 prefill记录，将16 rank×4个目标body×3个C2源层2/8/14绑定到同rank的HcPre、RMS、宽AG、SWA Wkv/写入、FP32 Wkv/Wgate、pool/ring、long cache写入和SMLA。匹配使用typed shape、实际kernel、流、层序及连接编号，保留全部192组ID；不是运行时指针/别名证明，更不是全模型覆盖率。

| 真实分支 | 窗口数 | 判断 |
|---|---:|---|
| 本地Q及全局cache均有效 | 138 | 两份FP32投影在原主流上先于pool和long KV发布，后续本地SMLA等待该状态。 |
| 本地Q为空、全局cache仍有效 | 30 | 实际执行两份FP32投影及pool/ring/cache更新，没有本地SMLA。不能因为本地无Q就丢掉后续状态；跨迭代延迟发布需独立合同。 |
| 全局cache和本地Q均为空 | 24 | 没有这两份FP32工作；保留原空侧输出及EP服务，不能认领一次原本不存在的投影。 |

大形状有240次`MatMulV2`；global M6/12还有96次`MatMulV3`。原先只按MatMulV2提取的9.113ms是五组大形状的中位串行参考，不能替代逐调用时间线。现已将336次native duration与独立单位成本表逐签名/DP/行域交叉核对一致。

大prefill的SP和CP行域已核实：全局3328/3346/6528/6534/6540对应local 416/419/816/817/818，实际Q行数等于本rank区间与有效prefix的交集，尾部裁剪吻合。小M6/12实际SP local为3，保留旧桶/回退，不能使用ceil(actual/8)硬改接口。

FP32两核native并集中位分别为422.110、454.196、857.373、883.741、879.752µs/源层；小M6/12为43.130/44.650µs。其他stream与pair跨度的native重叠，在大形状中位29.7–87.0µs；这不是AIC/AIV阶段利用率，也不能据其余时间宣称资源空闲。原pool之后还有index-K、量化、cache发布、QLI等读者，SMLA的整段等待不能一起记为可删。

## 两投影单独优化：本轮结束其C8首token 1%立项路线

- **介绍**　只改变C2两份FP32投影的重复计算或实现，保持宽通信、其余计算/提交、EP专家服务及原调度关系固定；先给两投影全部免费，候选执行成本和新增同步均为0。
- **到了哪里**　**已停止，限定上述单独范围与C8的1%排序参考。** 1%是已有优先级参考，不是新设的用户精度/收益硬门槛。该决定不排除廉价的联合实现，也不覆盖更大owner通信方案或六普通请求。
- **结果**　按原EP到达差区间和每个请求的实际first-eager归属，四carrier条件信用为1.285/2.608/2.733/2.670ms；最晚请求累计9.296ms，各请求TTFT信用0.286%–0.636%。即使全免仍不到参考，无需为此单独补local-M耗时或扫描参数。只免费cache-only窗口的两投影，即便有利地将整个TP组都提前，原peer约束下信用仍为0。

折算公式为 `max(d,0)-max(d-a0,-a1)`，d使用同一归档的DP到达差区间，每组a给其rank中最大的可删耗时。这样有意过给partial-rank优化信用，再按原请求顺序累加。依赖稳定时钟偏移和既有集合成员；EP入口提前并不等于客户端TTFT必然同量提前。这是原调度条件下的乐观排序模型，不能当作实测、无条件端到端上界或所有算法的停止证明。

复算：`python3 analyze_c2_producer_windows.py --recompute-exposure C2-exposure-inputs.json --output recomputed.json`。原始绑定、全192组ID及限制见summary与本地完整窗口文件。

## 更大owner producer继续：数值门槛现在有实际反例

- **介绍**　较大D-A1仍可同时减少全宽AG、重复SWA投影、C1/C2源生产及部分物化；C2两投影的单独低值不能剪掉这个范围。原local-M方案先按位一致门槛作最小native诊断，决定是否值得继续原样实现，而不是直接测服务性能。
- **到了哪里**　**待确认，已发现原生数值反例。** 171冻结容器镜像已复核，真实checkpoint第0层SWA Wkv为BF16 `[512,5120]`。隔离进程使用同一真实权重、有限synthetic BF16输入，对比ND `F.linear`的full M3328与八份M416。实际`AscendUnquantizedLinearMethod.apply→unquantized_gemm`代码确实调用F.linear；但此诊断尚未证明已加载服务权重的物理format或全caller合同。
- **结果**　1,703,936个输出中302个bit不同，max abs=`0.000244140625`，normalized RMSE=`2.8127e-5`，全部finite。因此“逐行实数线性就能保证原样换M逐位相同”被否定。不能把旧30.27ms条件包络升格为可直接开发/部署的预计收益；该差异也不能直接推成模型输出退化或证明所有owner实现不可行。

诊断整体状态为ERROR：ND比较完成后，NZ29分支在`allow_internal_format=False`下触发GE编译缓存，试图访问只读`/root/Ascend`而失败；M6540/NZ等后续比较未执行。保留完整失败原件，不把进程退出码或部分结果写成PASS。另一次启动仅因容器cwd不属于已映射目录而在进入进程前退出，未执行NPU算子。没有为了补齐表格重跑未改变的ND配方。

运行使用现成RW映射中的独立子目录，主机为`/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/fusion-native-wkv-partition-20261109-a2`；容器对应`/data/tiankuan/wio/projects/910c-deepseek-v4.1-flash/fusion-native-wkv-partition-20261109-a2`。未改挂载、依赖、服务代码或部署，未提交模型请求和性能benchmark；事后health为200。Torch为2.10.0+cpu、torch_npu为2.10.0.post7。只完成1组ND native数值比较，不是服务或完整融合验收。

## 反例后的具体方案分支

1. **同M的producer所有权重排**：将原BF16全宽行仅汇到选定producer，保持full M/N/K、输入顺序、weight格式及原norm/RoPE；广播窄KV/C1 latent或C2两FP32 plane，其余rank执行原query与cache读者。这样避免已知的local-M变量，但root收集+窄发布协议、原精度和资源争用尚未验证。它保持producer全M计算耗时，不能再记旧local-M计算节省，通信净值必须重新定价。
2. **local-M且保留原数值顺序的kernel**：需要确认原full-M归约/tiling并复现，不能凭K相同认定位一致，也不能无成本继承原时价。
3. **允许误差的local-M路线**：只有找到既定精度验收依据、验证KV/RoPE/cache及后续真实负载结果才可采用；本轮不自行设置宽松阈值或把小RMSE当作模型验收。

三条仍属于D-A1/C08相关范围的方案分支，共享原通信与投影预算。下一步先查原框架实际ND/NZ策略、现有精度验收合同及同M producer通信原件；对会改变去留的缺参再定点补证。其余真实执行路径继续扫描，Goal与D-A1保持进行中。

[绑定摘要和192组链ID](C2-producer-summary.json) · [请求模型输入](C2-exposure-inputs.json) · [native原件及完整失败](native-wkv-a2-evidence.json) · [运行树身份](runtime-source-verification.json) · [复算脚本](analyze_c2_producer_windows.py) · [验证](C2-validation.json)
