# DFC 原生交付与 24 核资源更正

## 介绍

沿真正 DFC 内部的接收端行→INT8 GMM1→FP16→SwiGLU→INT8/FP32 scale→GMM2 FP16→combine/remote return，先看完整交付与共享缓冲，再匹配旧性能。每个替换原语必须逐位一致。本轮没有新算子实现、设备计算、模型/服务请求或新 profile。

**先更正资源前提**：旧报告把 `BLOCK_NUM=20`、内部 routing tiler 的 `aivNumInitRouting=40` 当作整个 DFC 固定20/40，这是错误解释。当前同 SHA 的 host tiler 从平台取得 `info.aivNum`，并用平台 AIC/AIV 计算 blockDim。又只读原16个数据库，逐个 SHA 与缓存一致；**全部2,560个原 eager DFC TASK_ROW、起止纳秒及 globalTaskId 匹配，均为 `MIX_AIC, blockNum=24, mixBlockNum=48`**。因此“当前 DFC 留出固定4/8核”前提撤回。旧 shared4/8 原语的字节一致结果与单独发核记录仍成立，不能由此推得与实际 DFC 的共驻。原始旧报告/已完成 Run 不覆写，当前地图和此文优先。

## 到了哪里

### 真实原生路径、合法条件与末读者

- 当前容器10份 DFC 源重核全部旧 SHA 一致；再取 Catlass/CANN9.2.0 依赖，22个请求中21个实际文件、一个猜测的 `arch/atlas_a2.hpp` 不存在。真正 AtlasA2 常量就在已读 `arch/arch.hpp`。具体审阅范围列于 `dfc-gemm-tile-source-review.json`；检索命中、pin、其它未选架构不是已审阅。先前 return/物理cache 审阅仅按相同 SHA 复用，不冒充本轮全量重读或 loaded binary/每调用tiling证明。
- 实际权重几何是 K5120/N4608，SwiGLU宽2304；target24个本地专家/top6，draft为8/top3，不能套默认2048。固定源中 `swiglu_limit=10`。每专家 `currentM` 来自**接收端累计计数**并按原65536容量裁剪；sender0仍可能做专家。空专家不做 MMA，但仍保留原旗标/代次。
- GMM1/2 都是 L1 tile128×256×512、L0 tile128×256×128；INT8 MMA 先累 INT32，最后按原 per-channel `VDEQF16` Fixpipe 成 FP16。`ENABLE_SHUFFLE_K=true`，起始K块随 AIC index 变；unit flag关闭。异步 preload需 `SynchronizeBlock` 后 `PIPE_FIX` 发布。新并行不能把“已排队”当 FP16 已可见。
- 原 AIC **先完成全部 GMM1，再等待第一组 V2C，然后 GMM2**。AIV 先收齐/准备输入，再等第一组 C2V，做前21专家激活；再等第二组 C2V 做末3专家。原 GMM2 在处理 group20前（zero-based）就消费第二组 V2C，尚未到需要第二组输入的 group21。保持原地址/行域时可研究把该等待推到 group21前，必须仍消费相同两次旗标，并核实际 DMA/flag/位一致与曝光。不是删除等待。
- SwiGLU 已把 FP16 widen、pertoken乘法、两个不同的 clamp、原 Exp/Add/Div/Mul、**整行2304维 ReduceMax**和量化放在同一 epilogue。输出 scale=`max/127`，codes用另一次`127/max`，随后 RINT→INT32→half→INT8；不能把除法与倒数乘约掉、局部N块独立量化、改成共享专家 INT32 activation或加一个新零值保护。零/Inf/NaN等原行为未过新原语位门槛。
- 激活 codes 供 GMM2所有20个输出N块读取，**scale2供后面的 combine**，不是 GMM2 launch 后就死。原 combineFP16→FP32→pertoken scale→BF16，以及 unpermute的 top6 顺序归约和 shared BF16 add 均保留。结束跨卡屏障早于最终unpermute读 offsetD，不把全arena提前退役。

### 已有别名和真正存储量

`ptrC == ptrC2`；GMM1逻辑宽4608但行stride5120，是与GMM2的5120宽FP16输出共用同一bank。`ptrA == ptrPermutedToken`，接收端5120宽INT8与2304宽激活也已共用一个bank。原65536容量下，两bank分别640MiB和320MiB，**已经是960MiB，并非四份相加**。

- GMM1 Fixpipe只存实际4608列，SwiGLU只读这4608列；512列stride空洞不是已发生的额外FP16写读。仅压紧GMM1的stride不能缩小仍由GMM2要求的640MiB bank，容量信用0。
- 相同stride使第一批GMM2写区与第二批尚未读的GMM1区在 `first21_rows*5120` 分界。改成compact4608会把分界前移，可能提前覆盖未读尾部。2560个真实计数窗中2171个满足一个明确的潜在WAR见证；例如rank0/carrier416/layer0，first21=1406、tail=1，compact tail从元素6478848起，GMM2第1265行便可碰到它，早于原group20起始行1399。它是地址/合法交错反例，不是已运行压紧版本的位差。
- 仅当前profile最大receiver18222不能作为所有合法请求的容量上界。若证明整个EP组每sender至多M行，安全全组界可用 `min(65536,16*M*topK)`；target M3时为288，两payload bank公式变成4.21875MiB。但图/ACL缓存、prefill大shape、跨流末读与allocator高水位尚未核，**不把955.78MiB直接算成可释放显存，不乘40或16**。这引出下一处实际 workspace 分配/graph缓存/退役交付。

### 数值与硬件接口判别

INT8直接积的绝对和分别不超过5120×128²=83,886,080和2304×128²=37,748,736，小于INT32上界。这只说明该整数部分不需按浮点Wkv的归约误差推理；不证明新的 DMA、Fixpipe FP16、perchannel/pertoken次序或整原语位一致。不能借它接纳已有 changed-M BF16失败方案。

已沿实际 V220接口追到 arch2201实现：LocalTensor目的UB分支调用的 `FixpipeL0C2UBImpl` 是 unsupported空实现；L1→UB的 DataCopy同样不实现；当前Catlass `copy_l0c_to_ub.hpp` 只为3510 include。UB→TSCM/L1接口则经 `AllocUB`、UB→GM及消息让Cube再读，仍有GM staging。**只改目的地/套TSCM名字并没有删掉这段GM交付**。其它新硬件接口/代码生成路径不在本次结论内。

所选源资源公式：L1实际约386KiB（静态容量断言按双scale保守计388KiB）、L0A32KiB、L0B64KiB、L0C128KiB，后两者已占满AtlasA2定义容量；Vector双stage基础UB108KiB，另有动态scale/scratch。这些是源公式，不是实测占用或共驻许可。完整128×4608 FP16 panel约1.125MiB，不能凭空留在当前L1余量内。

## 结果

### 待确认：整段排程、等待边界及资源重写

独立8个 shape/dtype/DP/carrier 条件的320次调用及min/median/max与 `PROFILE-UNIT-COSTS` 完全一致；2560个24专家计数窗逐层再与原native任务/receiver totals匹配。receiver rows为14–18222，中位1935，单专家最大9478。末3专家为空107窗；group20和末3同时有工作1546窗。这些计数不含阶段耗时。

按原跨rank时钟约束，取每层**最后DFC结束−最后DFC进入**的共同尾段；该段仍含计数/通信、GMM/向量、return与等待，不能直接称作GMM时间，也没有拿最早到达rank的大段等待当可删计算。160个层×carrier依原 eager/request顺序累计。各层时钟极值可独立保守相加，不保证全局同时达到；所有价格是固定外部调度的乐观条件情景，非因果/预计/实测请求收益。

| 假设 | 条件 P90 信用 |
|---|---:|
| 全共同尾段免费 | 299.719–338.509ms /20.090%–22.690% |
| 可优化阶段占25%，省一半，新增5us/层 | 36.665–41.514ms /2.458%–2.783% |
| 阶段占10%，省一半，新增5us/层 | 14.186–16.125ms /0.951%–1.081% |
| 阶段占10%，省一半，新增20us/层 | 11.786–13.725ms /0.790%–0.920% |

半省/h5us要过同1%排序参照，相关且暴露的阶段需占共同尾段约9.29%–10.49%；h20us需10.70%–12.09%。阶段占比、实际重叠与新增控制/权重重读成本确会改变去留。只保留这些具体缺口；无新阶段profile，未把整体DFC时间或PMU比例当激活时间。

待确认方向：保留原位数值和原别名下调整21/3批次/每expert发布、将第二V2C等待移到真实消费边界、按原INT32→FP16边界做行tile流水/分工、在付DFC代价后给shared留核或设计更大联合kernel。等待单项的缺时量是 `min(原group20处暴露的第二V2C等待, 原group20 GMM2时间)` 减成本；不是1546个计数窗就证明值得实现。M3 graph/六普通缺匹配阶段时序，不外推本次prefill价格。

### 已停止的具体配方

- 当前原配置“DFC20/40所以剩余4/8可直接放shared”：资源前提被2560个24/48实录否定。跨kernel非独占/让核/重写仍另列，不能当停止整个shared parent。
- 只改L0C→UB或改用当前TSCM API就宣称消去FP16/INT8 GM交付：当前所选接口不支持，或仍经GM。
- 仅将C1压紧4608、把别名四bank当新容量节省、取消整行量化/中间FP16或scale末读者：容量信用0或破坏现有交付/逐位合同。
- FP16发布+读取按每receiver行18,432B作纯字节情景：旧独立双向copy参照2405.30GB/s、全免P90 .5416%，半省/h5us .2172%。在500GB/s假设则全免2.6056%、半省/h5us1.2492%。**该参照不是DFC有效HBM带宽或硬上界**，且当前直接接口配方不合法；只结束“现成接口已省字节”的小配方，不以高带宽假设剪其它排程/布局或整个DFC。

让核成本不能藏起来：若假定24→20使共同尾段中Cube敏感部分f多花20%，同时每层付5us，完全隐藏shared在f=.25时条件P90约1.014%；仅隐藏一半为−.087%。f=.5时完全隐藏也约−.120%，一半约−1.221%。这是另一个假设模型，没有声称实际线性慢20%；**负收益已保留**，没有按层裁掉亏损以冒充全局净省。

### 已确认可实现

本轮新增0；原shared4/8仅独立原语限定资格保留，实际共驻/净收益不准入。下一处转 native workspace的真实分配者、缓存、图捕获与最后读者，先查有没有独立960MiB/QLI空间及何时可复用，再讨论容量能支持何种新融合边界。

只读核对171主机名及同owned容器/镜像；DB在host存在、容器内未挂该路径，未将容器内missing当作原记录丢失。首个launch字段校验误把connection_id当global_task_id，错误reader/说明保留，修正列7后重新读取验证通过。无NPU/模型/请求/新profile/远端写入/服务改动，未操作其它controller。
