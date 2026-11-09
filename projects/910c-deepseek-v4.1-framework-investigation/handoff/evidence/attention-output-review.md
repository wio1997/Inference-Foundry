# D-A2：Attention结果到FFN归一化交付

2026-11-09。用户要求每个替换原语逐位一致。本文是一个新读者范围的实际排查；形成改法后才匹配历史OA研究，不以旧表作为搜索目录。全链路仍未闭合。

## 介绍：范围由真实数值/状态读者决定

`SMLA output → inverse RoPE → pad/permute → CP AllToAll → suffix padding → OA → OB → SP ReduceScatter → Attention HcPost → FFN HcPre → RmsNormCast`。

| 交付 | 实际消费者/必须保留值 | 融合边界含义 |
|---|---|---|
| SMLA BF16结果 | 逆query RoPE，随后只有输出投影路径读取 | 可研究SMLA epilogue直接产send布局，但必须先保留原SMLA BF16舍入，再完全复现旋转计算；不能把FP32 accumulator直接旋转当等价。 |
| `[local query,64,512]` | 8路head owner交换后得到`[global padded rows,8,512]` | padding、rank-major行序及local query为空仍交换的合同不能丢。整侧无投影另留零输出/RS路径。 |
| OA输出 `[rows,1,1024]` BF16 | BF16 OB矩阵乘，其后SP归约 | 实际OA和OB均非INT8；不存在可直接复用的OB动态量化或scale。OA BF16中间舍入是独立精度门槛。 |
| OB partial `[rows,5120]` | TP ReduceScatter，后接本地mHC Post | 可以研究OB尾写RS输入和通信流水，不能省略各rank partial、原归约顺序或改成提前四路残差。 |
| Attention Post四路残差 | FFN Pre现在读，FFN Post之后还要读 | 跨Post/Pre融合仍须全局保存原rounded BF16四路残差，不能宣称整份写回消失。 |
| FFN norm BF16与FP32双输出 | BF16给专家/共享分支，FP32给router | 不能只留BF16再随意重建FP32；更大消费者范围进入MoE段判断。attn_pre/ffn_pre系数的交错代际也保留。 |

实际后端选择：`ops/dsa.py → DSAAttention → AscendDSASWABackend → enable_dsa_cp → AscendDSACPImpl._forward_o_proj`。已在当前冻结容器只读核对这些依赖及CVLinear/util的hash与本地归档一致：[源码验证](attention-output-source-verification.json)。主入口/CP wrapper前轮已核对。相关行：`dsa_v41.py:613–683,690–716`、`dsa_v41_cp.py:209–225`、`dsa_cp.py:77–91,1657–1682,1754–1799`、`dsa_v1.py:262–272`、`model.py:760–763,879–928`、`sequence_parallel.py:71–101`。

## 到了哪里：数据流与条件价格已绑定，主方案继续深入

独立绑定2560个原窗口，区分1840有本地query、400无本地query但仍投影、320整侧无投影（不作覆盖率分母）；OA/OB的14个DP/carrier typed分组调用数及中位耗时与独立`PROFILE-UNIT-COSTS.json`全部一致。有效cache M与OA M分开：cache6534/3346/6540分别对应OA6536/3352/6544；不能拿cache的shape给OA定价。

### D-A2-O：query owner先OA，再窄交换 · 待确认

- **介绍**：每个query owner用全部8组OA权重，先得到8×1024 BF16结果，再按head owner交换。OB及RS保留原地，原OA BF16中间值不能略过。payload由65536降至16384 B/query；原OA/OB不能简单预乘矩阵代替，因为OA的BF16舍入仍需逐位保留，且K4096→rank1024→H5120的低秩分解本来减少MAC。
- **到了哪里**：值得深入的主方案，尚未准入实现。须额外常驻2240MiB OA权重/rank，并保留原KV容量和全负载峰值余量；逐层临时gather权重每rank接收56MiB，比当前local M≤818时可删的payload还大，不能把按需权重gather当免费。源码下stock OA只能输出`perm_y=(1,0,2)`，send排布需要额外pack或新kernel。
- **结果**：把整个原AllToAll按字节缩为1/4且新成本0，原EP/其它工作固定，条件TTFT信用最高6.217%；原通信含到达等待，不能把它视为可兑现估价。更窄的场景锚用同cohort/同payload最快通信时长，扣小消息startup代理，再按3/4缩减：新增成本h=0/50/100/150µs/层，最高信用约2.485%/1.938%/1.391%/0.923%。这些是可证伪的条件场景，未测字节敏感比例、新OA和pack成本，不是硬上界或预计收益。净增成本跨100–150µs会改变此排序，故缺口具体为大行416/818、40层完整真实权重集的逐位一致门槛与OA+pack增量，随后才值得定点核窄交换。不能热测单权重后外推40层。

形成该方案后匹配历史：`OA-OWNER-COST.json`在decode M3/G8的完整40层原语已逐位通过，但stock OA+pack周期3.003ms，相比1.993ms增加1.010ms，超过该decode预算；这条旧小行实现维持停止。其`perm_y=(0,1,2)`直写变体已被API拒绝，不重试。M3的数值或耗时不自动迁移到本次M416/818；大行及SMLA/逆RoPE/OA联合producer是不同的待确认范围。当前不新增该旧配方实验。

### D-A2-E：只合逆RoPE与send打包 · 已停止独立1%路线

- **介绍**：保留投影、通信、HC及所有其它工作，将原逆RoPE和rank-major contiguous打包合入producer。
- **到了哪里**：全部相关native工作免费、各TP组领取最大rank信用，原EP/外部调度固定。
- **结果**：8请求最高条件TTFT信用0.72287%，低于既有1%排序参考；不为同一预算下的微小变体单独补测。可以作为OA/SMLA更大producer的一部分，不能据此剪掉通信或SMLA主体。

### D-A2-B：OA/OB共享Cube或megakernel · 待确认

- **介绍**：保留OA BF16舍入及OB原K归约，在同核承接完整1024宽OA panel给OB，尝试少一次GM写/读及部分提交/输出拷贝。两次矩阵乘使用同一Cube，不把MAC直接相加当可重叠收益。
- **到了哪里**：匹配`OA-OB-SHARED-CUBE-SCREEN.json`已有本地panel与L1→L0B约束模型。原M24的2560 native pair全不高于该单owner局部下界，既有decode局部路线维持停止；不能把全段小行结论搬到大prefill。
- **结果**：旧大行条件模型的有效资源率门槛约85.5%–89.3%，保留BF16中间量、全1024列归约及原TP归约。它允许删间隙/物化，未证明kernel实现或逐位一致。下一缺口是候选L1/operand路由与分块能否保留原归约、以及现有带宽/争用是否能接近该门槛；不重复已失败的group-one MatMul替代。与前移OA共享工作，不相加收益。

### D-A2-H：Post/Pre/norm联合producer及更大MoE入口 · 待确认

- **介绍**：按行/块生产rounded四路Attention残差，既写回供FFN Post，又供FFN Pre；继续融合其collapsed BF16输出的归一化，同时交付原BF16/FP32双输出。
- **到了哪里**：整段native全免的条件信用最高3.317%，只能用于保留搜索，不是可实现增益。不能把合法HcPost/HcPre的全部MAC和归一化删除。保持原语舍入时，常规物化消除只约每row少61440B：一次四路读取40960B，collapsed输出一次写/读20480B；四路写回及后续双输出仍要保留。
- **结果**：更大的FFN入口可能把router投影、dispatch量化和shared expert读取并入producer，不能现在用一个norm小预算剪掉。继续从`DeepseekV41MoE.forward`走到真实DFC/共享分支；局部HC联合kernel的缺口是原HcPre分块/归约与AIC/AIV跨阶段handoff，须逐位一致，不能用分配律跳过BF16舍入或重排归约。

## 可复算与未扫边界

完整typed窗口（7MiB归档原件SHA见验证，原task ID/签名保留本机）；[便携输入](attention-output-portable.json)供价格复算；[条件价格](attention-output-value.json)、[验证](attention-output-validation.json)。14组成本核对、portable复算及16个AGMM rank-case原件完整性检查通过，均不表示整个模型/候选通过。

```sh
python3 analyze_attention_output_delivery.py --recompute --inputs attention-output-portable.json --output /tmp/attention-output-recomputed.json
```

这里完成的是这条交付的读者/所列变换首判与高值范围定位；SMLA内部QK/PV和分页直读、跨layer共享索引的完整变换域、mHC内部实现仍未穷尽。其余输入、Engram、MoE、终端、DSpark和回环的旧记录仍需独立审查，不能自动继承闭合。
