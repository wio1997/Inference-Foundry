# 模块排查：耗时和计算活动分开看

## 真实端到端：暂时不能按模块排占比

无 profiler 基线为 32K 输入、1K 输出、并发 8、前缀命中约 89.99%，两轮合计约 **519.84 输出 token/s**（`evidence/baseline-result.json`）。这是整个服务的吞吐量，不能直接拆成单个模块用时。现有 trace 是有 profiler 的采样，覆盖不全，跨 rank 和不同 stream 的工作会重叠，也缺准确的完整请求依赖／关键路径。**因此目前不知道哪个模块占真实端到端时间最多，也没有可信的模块端到端占比排序。**

## 可以按相同样本窗口比较的观察

下表来自 `captured-layer-attribution/interior-robustness.json`，40 个已归属 target 解码层，每层都是 16 rank × 3 个内部采样轮次，共 48 个窗口。时间是窗口跨样本平均值，**不是端到端关键路径耗时**；MAC 指标是已归属 AIC 周期加权流水活动代理，不是整卡利用率。

| 模块 | 平均窗口耗时 | MAC 活动代理 | 观察范围 |
|---|---:|---:|---|
| target 第 0 层 | 1419.59 μs | 3.640% | 在这 40 层中窗口均值最长、代理最低 |
| target 第 28 层 | 1184.41 μs | 8.068% | 在这 40 层中窗口均值第二长 |
| target 第 24 层 | 1173.89 μs | 7.913% | 窗口均值第三长 |
| target 第 32 层 | 1169.14 μs | 8.112% | 窗口均值第四长 |
| target 第 36 层 | 1167.51 μs | 8.134% | 窗口均值第五长 |
| target 第 16 层 | 743.94 μs | 7.579% | 代理第二低，但窗口均值不是第二长 |
| target 第 20 层 | 1027.92 μs | 7.595% | 代理第三低；不是窗口均值第三长 |

第 0 层样本中 MC2 平均 **942.266 μs**，是该层值得追踪的部分；其采样周期分母占比和窗口时长比都不是端到端占比。证据：`evidence/captured-layer0-engine-diagnosis.json`。另一组 15 rank 的严格草稿样本，三个 draft block 合并的同主流 MAC 代理约 **2.015%**，但只覆盖 162/288 个候选，块身份、形状和采样均与 target 不同，不能直接排成全框架最低模块。证据：`evidence/draft-module-diagnosis.json`。

## 怎样排查优先级

1. **活动低的名单**：按同类模块、同一形状和同一采样口径的流水代理排序；它告诉我们哪里可能有活动不足，不能独立判断影响或原因。不同阶段／形状和不同采样分开列。
2. **耗时的名单**：先按同类调用的单次窗口时长和调用频率分别列出；如能建立完整请求图，再按真正暴露在关键路径上的时间排序。并行 rank 的时长、重叠通信、不同模块窗口不得直接相加为端到端时间。
3. **优先交叉核查**：第 0 层同时有低活动代理和较长窗口，是当前优先诊断对象；第 28、24、32、36 层虽不是活动最低，窗口较长，仍需要查依赖和是否位于关键路径。草稿、Engram、prefill、调度、通信等待和输出头仍有归属缺口，不能因未排名就忽略。
4. **只有更多证据才能定因或评估收益**：须补请求 ID、图重放／通信关联、模块依赖与可重叠区间，并证明数据覆盖代表真实负载；独立统计未计入排队、profiling 扰动等误差前，不给可节省时间或真实提速。

无需新 NPU 负载即可复算所列 40 层窗口排序：

```sh
python3 -c "import json; x=json.load(open('captured-layer-attribution/interior-robustness.json'))['interior_all_40_ranking']; print([(r['layer'],r['mean_layer_window_us'],r['cycle_weighted_reported_mac_ratio']) for r in sorted(x,key=lambda r:-r['mean_layer_window_us'])[:8]])"
```

## 分 DP 复核：40 个 target 层的流水代理（已有采样）

使用 `captured-layer-attribution/module-call-metrics.json` 已归属的 model 49 样本，限制每张卡各自的采样序号 1、2、3，每层 **DP0 八卡×三窗 = 24 窗，DP1 同为 24 窗**。四个 HC 标记及角色约束归属层窗口；窗口内收集同 model、完全包含的记录并剔除名称含 metadata 的行，可包含多个 stream。DP0、DP1 是冻结启动配置和 rank 标签的分组；逐卡采样序号并非已证明相同请求或同一次跨 DP 执行。

| 分组与指标 | 40 层中最低的三层及其代理 | 第 0 层代理 |
|---|---|---:|
| DP0，AIV Vec | 第 27 层 0.961%，第 1 层 0.972%，第 0 层 0.988% | 0.988% |
| DP1，AIV Vec | 第 0 层 0.290%，第 1 层 0.928%，第 27 层 0.995% | 0.290% |
| DP0，AIC MAC | 第 10 层 7.430%，第 20 层 7.513%，第 16 层 7.776% | 8.718% |
| DP1，AIC MAC | 第 0 层 2.330%，第 16 层 7.388%，第 20 层 7.675% | 2.330% |

16 卡 pooled 的 AIV Vec 代理中第 0 层为 0.445%，但不能据此说每张卡都最低。对 48 个独立 `(rank, 采样序号)` 组分别在 40 层内求最低：AIV Vec 和 AIC MAC 均有 **24/48 组**在第 0 层最低；这 24 组全属于 DP1，DP0 的 24 组没有一组是第 0 层最低。这里的“独立组”只是计算分组，**不是统计学上的独立实验**。第 0 层平均窗口 DP0 约 815.224 μs、DP1 约 2023.949 μs，是各卡本地耗时的样本均值，不能相减断言通信等待或可节省时间。

**指标口径。** AIV 代理为对所选 CSV 记录按 `aiv_total_cycles` 加权的 `aiv_vec_ratio`：`Σ(cycles×ratio)/Σ(cycles)`；AIC MAC 类似但用独立的 AIC 字段和分母。`captured-layer-engine-proxies/summary.json` 对 40 层所有正周期行均报告 ratio 缺失或无效数量为 0；AIV 共 18192 行周期为零、负数或缺失，按该摘要统计无法进一步分辨，故均排除，不当作零活动。正周期行的零 ratio 参与分母。ratio 为 CSV 已舍入字段，可能影响细微排名。

单行可能同时携带 AIC/AIV 数据，两者分母不同，流水可以并发；不同记录可能跨流重叠。现有归属脚本确保同一 CSV 行不被重复分给两个层窗口，但没有证明 profiler 的不同记录或 MIX 计数不会反映同一物理活动；不能将两代理相加、视为整芯片利用率或推算等待/空闲比例。不同层的模型角色、形状、路由与工作量也可能不同，分组差异不证明 DP 是原因。

草稿只严格归属 162/288 候选窗口，Engram 候选处在目标层间、含 host/UVA/其他流，attention、MoE 与层窗口又相互嵌套；因此这些范围无法凭当前聚合构成统一的全框架模块利用率排名。需要同一请求/重放的层、draft、Engram、CPU/NPU 事件归属和统一计数语义，才有条件继续合并；未知部分不是零。

本机复算：`python3 agents/diagnosis/recheck_target_dp_aiv.py`。脚本以 `Decimal` 从已冻结每窗分子和分母汇总 40 层 × 两个 DP，并与独立 `captured-layer-engine-proxies/summary.json` 的 16 卡结果逐层逐指标一致性核验、检查正周期缺失。全量 120 个 DP/整体层结果及输入 SHA-256 在 `target-dp-aiv-comparison.json`；结果 SHA-256 `d71312f60fe9bae30835af7c20d98c0190f56b365b86f262a09afdf438e45d96`，脚本 SHA-256 `7687690c9ea9b31731e9c975bfdd20d3b46426cb4d8405fe1c088880c2eaeafd`。没有操作设备或修改实现。

### 第 0 层：已有逐窗分组对照

离线将 48 个第 0 层窗口与 MC2 时间线按相同 rank、每卡内部采样号连接，逐条核对 layer 起点、MC2 起止、model 49 和物理输入首维 `M=3`；每 DP 有 24 窗，均有 38 条已归属 CSV 行。

| 项目（各组 24 窗的本卡中位数） | DP0 | DP1 |
|---|---:|---:|
| 层窗口耗时 | 829.474 μs | 2141.848 μs |
| 其中 MC2 任务耗时 | 353.973 μs | 1661.877 μs |
| 整窗 AIC 报告周期分母 | 18954872.5 | 76912171.5 |
| 整窗 AIV 报告周期分母 | 42961348 | 155964477 |
| 已归属任务时间并集 / 窗口 | 97.88% | 99.17% |

MC2 在两组中均为同名融合任务，物理首维相同，但有效 token、专家分布、cache 与请求阶段未确认相同；其他任务仍可能重叠。周期分母与长窗口同时增大提供了解释 pooled 权重差异的线索；其内部到底是计算、通信、等待或资源争用尚不能从现有字段拆出。时间并集只证明导出记录覆盖，不等于设备忙碌率，非中位数的各列不是同一个代表样本，也不是可节省时间。

复算 `python3 agents/diagnosis/recheck_layer0_dp_existing.py`；结果 `layer0-dp-existing.json` SHA-256 `251e8c60b7855b5037bbf19d857830a0321af7220d536185659aaacf0197b366`，脚本 SHA-256 `4f6cd6c31c09213f6369c53f779b75e3d31fbaee3857cd9b15dec7567e677f03`。已准备仅查询冻结 CSV 的 `query_layer0_dp_type_counters.py` 和 staging 容器 wrapper，待主控安排 ZCode；目的是按任务类型检查周期分母归属，不在现有数据未查询时假称该分母主要来自哪个任务。

### stage159：第 0 层代理的任务类型分解

ZCode `operator-ceiling169-v2-159-layer0-dp` 退出码 0、`model_verified=true`、模型 `deepseek-flash`。只读查询对 16 份冻结 CSV 的 SHA 逐一校验后，复现每个窗口 38 条归属记录、每窗唯一的主流 `DispatchFFNCombine`。48 条 MC2 与已有同 rank 的 task ID、时长、主流标识一致，全部报告相同输入形状：`[3,5120];[24,5120,4608];[24,2304,5120];[3,6];[110592];[122880];[3,6];[3]`。这些是物理张量 shape；没有记录对应内容或有效行数。

| DP / 计数口径 | MC2 在第 0 层该引擎总周期分母中占比 | MC2 周期加权代理 | 其他已归属任务合并的周期加权代理 |
|---|---:|---:|---:|
| DP0 / AIC MAC | 55.47% | 8.664% | 8.784% |
| DP1 / AIC MAC | 88.51% | 1.492% | 8.785% |
| DP0 / AIV Vec | 69.30% | 0.469% | 2.161% |
| DP1 / AIV Vec | 91.20% | 0.109% | 2.159% |

每组 24 窗；按类型累加的 AIC/AIV 分子、分母与此前按窗独立汇总结果逐项一致。**这是加权统计分解**：MC2 在 DP1 报告了更大的周期分母、较低的对应代理，因而主导了这组汇总值的差异；不证明 MC2 内部的通信、计算、等待或专家路由是哪一项造成差异，也不意味着 DP 身份本身是原因。其他任务的合并代理接近只是当前样本的观察，不能证明所有对应任务和工作量都相同。AIC 与 AIV 不同分母、MIX 和多流可能重叠；不能把不同任务的报告周期或持续时间加成墙钟时间、设备空闲或可恢复收益。

核对内容的最短复算：读取 `layer0-dp-type-counters.json` 中按 DP/type 的 `aic_cycles`、`aic_mac_cycle_proxy`、`aiv_cycles`、`aiv_vec_cycle_proxy`，与 `target-dp-aiv-comparison.json` 中 DP 分组 layer0 的同名分子/分母相加比较；MC2 占比是其周期分母除以该组同引擎总分母。原始结果 SHA-256 `0482575955229199d9b2c25e1cff2590d82a27d8a91f89e0d8f3418e7be37fad`；查询脚本 `query_layer0_dp_type_counters.py` SHA-256 `4055e04a78c3a00fe223c4ae82090c49024d2ac7e95cfa7bfbada8c25837cc6b`，wrapper `run_layer0_dp_type_counters.py` SHA-256 `ea857e6a7fb086ef8ab3f8db90f47e46c68085a70a9485e61c1a88988e5d87e9`。每窗的 MC2 task ID、shape 和计数及各类型聚合均保存在结果文件，所有 16 份 CSV 的 hash 也已记录。

**动态路由缺口。** 冻结 `dispatch_ffn_combine.h` 的接口包含 `expertIdGM`、`xActiveMaskGM`、`expertTokenNums`；`dispatch_ffn_combine_kernel.hpp` 还读取按专家累计的 token 计数并决定 GMM 的 `currentM`。这说明正确解释需要实际有效 mask、每 rank/每专家分配及处理规模。当前所查 CSV 和 stage159 只保存这些输入的 shape/dtype 与计数器，没有提供上述 tensor 数值，已有单数据库动态字段关键词扫描也没有覆盖所有可能导出，因此其值 **UNKNOWN**；模型配置的 top-6 和相同 `[3,6]` 不能代替每次专家分布。最小补证为同一次 kernel/replay 的匿名标识、实际有效 token 数、按目标 EP rank 和专家聚合的 token 数及容量/丢弃情况，连同与 MC2 任务的可靠关联；已有导出若含此类统计应先只读查询，不需采集 token 内容。

进一步的离线复核：`python3 agents/diagnosis/recheck_layer0_type_decomposition.py`，逐样本匹配并比较每个分组的 MC2 与其他任务的原始分子、分母。该脚本 SHA-256 `6cfa849adf9a753aa28441fa225ca57e5b1004dd9217939584bc86d68c7a2866`，输出 `layer0-type-decomposition-verified.json` SHA-256 `a8cce2d7a08ad87b4ec2ba43738928ad21680e27ee4b87d6672aee5d8995a862`；结果经二次执行逐字节一致。

### 40 层对照：第 0 层的 MC2 耗时差是否普遍

复用本机冻结 `module-call-metrics.json`、`rank_layer0_timeline.csv` 和 `rank_target_calls.csv`，按 rank、单卡 source-attributed 图采样序号、40 层序号关联 MC2 任务；对每条核对 model 49、主 stream、任务完整包含在层窗口，物理输入首维均为 3。采样序号 1/2/3 在每卡对应已记录的 forward ordinal 1/6/7，不能据此认定跨 DP 是相同请求。共匹配 **1920/1920 条**，每层每轮 DP0、DP1 均各 8 卡；第 0 层 48 条与 stage159 的 task ID、stream 和耗时逐条相符。

| 单卡采样序号 | 第 0 层 DP0 中位数 | 第 0 层 DP1 中位数 | 第 0 层组中位数之差 | 其他 39 层组中位数之差的范围 | 其他 39 层差值中位数 |
|---|---:|---:|---:|---:|---:|
| 1 | 273.245 μs | 2031.189 μs | +1757.944 μs | −11.963～+8.780 μs | −3.413 μs |
| 2 | 392.173 μs | 942.921 μs | +550.749 μs | −12.100～+15.010 μs | −1.221 μs |
| 3 | 353.973 μs | 1661.876 μs | +1307.903 μs | −10.919～+17.756 μs | +0.200 μs |

每列时间均先在本卡算任务持续时间，再取相应 8 卡中位数；上表差值是 **DP1 组中位数减 DP0 组中位数**，不是逐 TP 配对差的中位数，也没有跨卡时间戳相减。在这三个采样轮次中，大幅组间差异集中在第 0 层，而不是所有层普遍有同样差距。该观察不证明 DP 身份、通信等待、专家路由或其他任一因素是原因；同一物理 M 也不能证明有效 token、路由和请求一致。数据不是独立实验或端到端关键路径，不能换算可恢复收益。

复算：`python3 agents/diagnosis/recheck_40layer_mc2_dp.py`。脚本 SHA-256 `f46c8ca1413c01d4ec7340b5dea04f45ab77a72cf8c4995249b53a01fdc619b7`；输出 `40layer-mc2-dp-comparison.json` SHA-256 `cf0962177d27f703143704a44e9030835aa1d4c86a19c35bcf12236c14fe9f3b`，含 120 个层/轮比较、覆盖计数、原始输入及脚本 hash。既有逐任务索引带 AIC MAC 比率但不含 AIV 周期；本次已有耗时对照足以回答“大差距是否各层共有”的有限问题，未额外查询 40 层周期，也未操作 NPU。

### 第 0 层相对其他层：源码与配置结构

以下层号都使用源码的 **0 起始**编号；是冻结代码及配置的结构结论，不是对 48 次运行的张量值或因果结论。

- **初始输入与 mHC：** `model.py:1519,1551–1556` 使用 input embedding 或 `inputs_embeds` 作为首层输入、复制 HC 维度，并初始化 `pre_mix` 为 `[1,0,...]`。`model.py:1592` 每层返回的 `hidden_states,pre_mix` 进入下一层；所以第 0 层处于序列入口，第 1 层及之后使用前层输出。这不说明该入口本身造成了 DP 组耗时差。
- **Engram 位置：** 冻结配置 `engram_layer_ids=[1,14]`；`model.py:837–842,1565–1592` 构造指定层 Engram，并在进入相应 decoder 前、满足运行条件时执行。因此第 0 层没有自己的 Engram 注入，第 1 和 14 层才是配置的注入位置。启动脚本提供 `--engram-config` 的 CPU offload；`engram/common.py:31–34` 的开关取决于层列表。但 `model.py:1519–1523,1242–1259,1392–1486` 允许此前准备 lookup 并与主计算异步重叠；不应把“0 层无注入”扩展为“0 层不可能与 Engram producer 竞争资源”。本次现有 trace 没有证明此类竞争或 event 配对。
- **Attention 角色：** `model.py:510–549,691–740` 按 `compress_ratios` 和 source 层选择模块。配置前两层 ratio 均为 0，2–19 层为 2，20–39 层为 1；layer 0/1 均不在 `kv_source_layer_ids=[2,8,14,20]` 或 `index_source_layer_ids=[2,8,14,20,24,28,32,36]`。因此 0 与 1 的配置均为无压缩角色，而不具有 2 及后续指定源层的 compressor/indexer 构造；不同于 1 层前有条件 Engram。静态角色不能代替运行时具体 cache 内容与负载。
- **MoE 权重与路由：** `model.py:151–209,236–260,811–817` 各 decoder 构造同一 `DeepseekV41MoE` 类，传入各层 prefix；共同配置有 384 routed experts、top-k 6、1 shared expert、`sqrtsoftplus`。不同层的参数命名空间不同；独立保存的 **172 参考**权重索引也列出各层的独立专家键，但它尚未证明与 169 当次 checkpoint 完全一致，更不证明各层实际权重值不同、输入专家分布相同或实际硬件执行策略相同。
- **hash-router 不可仅由名称判断：** `model.py:210–234` 用 `layer_idx < config.num_hash_layers and not is_draft_layer` 选择是否创建 `tid2eid`；原始冻结 text config 没有 `num_hash_layers`。相同 baseline/stage 哈希的 `utils.py:168–172` 仅在**配置对象没有该属性**时补默认 0；解析类可能已提供属性，当前未绑定其历史运行值。因此不能单凭 JSON 缺字段断言第 0 层或其他层的 router 选了哪个分支。`fused_topk_router.py:159–179,204–209` 的算子名含 hash 即使 `tid2eid=None` 仍可调用，不能仅凭 profiler 名字判定查表。

同样的配置被两组采用并不能证明两组实际输入或专家路由相同。已观察到的第 0 层 MC2 组间差异是采样现象，当前这些结构差异只是待验证的上下文，不确认解释。

证据哈希：`evidence/model-config.json` `43d3b2bc55626893d59275b32e04068c0fc916f5c382eb91de73bdbade9c88f7`；冻结 `model.py` `101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91`；`agents/bounds/sampler-patch-snapshot/adapt_patch.py`（容器 `vllm_ascend/utils.py`）`5f88fd83bcba539c3d24324ed104469914254989996342b27bb36a326e3f8458`；`evidence/frozen-serve.sh` `c297201185b5676254af4d7693279f89bfa60f9c76eb340058a78ecf730b7bba`；172 参考权重索引 `c32977056a3e8bb17d2330d566020f49ff65facb022c25035b8d3fcf5d2af3a2`。本轮只读源码与配置，没有触卡。

进一步检查 172 参考 `quant_model_weights.safetensors.index.json` 的全部 143324 个 `weight_map` 键，大小写不敏感的 `tid2eid`、`token_to_expert`、`hash` 均无命中；第 0/1 层各有 `ffn.gate.weight`、`ffn.gate.bias`、`ffn.gate.bias_vl` 键。此结果仅指参考索引，不覆盖 169 的实际 checkpoint 或其他命名，而且 `model.py:220–230` 以零初始化 hash table，缺键本身不足以证明历史实例没有创建该参数。当前仍不能确认历史 `num_hash_layers` 与路由分支。

为避免无界检索，准备 `query_layer0_hash_config.py`（SHA-256 `a81fb4ed64da796aa0404e31854f167ad32e9608752a4106aa5304e4609f2683`）及双容器 wrapper `run_layer0_hash_config.py`（SHA-256 `11ec2e0156f4da195d90df0752350a9edc83db9a3cf5392171614ca6fb6fe77a`）。仅检查冻结启动脚本模型目录下的配置及参考索引同名文件，按已安装包根限定 DeepSeek 配置源、记录类定义/目标字段及哈希；不可见就标 UNKNOWN，不读取模型权重 payload 或执行 NPU。待主控安排 ZCode。

### stage161：169 配置与 checkpoint 索引的只读核验

ZCode `operator-ceiling169-v2-161-hash-config` 退出码 0、`model_verified=true`、模型为 `deepseek-flash`。baseline 和 stage 两容器均能读取冻结启动脚本指定模型目录的 `config.json` 和 `quant_model_weights.safetensors.index.json`；两边当前数据及包源列表/hash 一致。当前 config 的 SHA-256 `43d3b2bc55626893d59275b32e04068c0fc916f5c382eb91de73bdbade9c88f7` 与本任务冻结配置相同，根对象和 text_config 中均无显式 `num_hash_layers`。当前 169 索引 SHA-256 `c32977056a3e8bb17d2330d566020f49ff65facb022c25035b8d3fcf5d2af3a2` 与保存的 172 参考索引一致，143324 个键中 `tid2eid`、`token_to_expert`、`hash` 名称匹配数均为 0，第 0/1 层各有 gate weight、bias 和 bias_vl 键。

当前磁盘索引的哈希一致**不等于**已经证明历史服务启动时索引同一版本或所有权重 payload 完全一致，也不证明历史实例没有从别的入口获得/初始化路由表。已安装 `/vllm-workspace/vllm/vllm/transformers_utils/configs/deepseek_v41.py` 中找到 `DeepseekV41Config`，继承 `PretrainedConfig`，本文件没有 `num_hash_layers` 字面定义；没有读取该类的完整初始化和基类语义，更未捕获历史对象。另一份 Transformers 的 `DeepseekV4Config` 出现 `default_num_hash_layers=3`，它是不同类，不能当作 V4.1 实例的默认值。冻结 vLLM-Ascend `utils.py:168–172` 只有在实际对象无该属性时才设置 0；**第 0 层历史 hash 路由分支仍为 UNKNOWN**，不能由空索引、同名 hash 算子或不同配置类推断。

结果 [layer0-hash-config-inventory.json](layer0-hash-config-inventory.json) SHA-256 `cded93891b3b67ab93cbfd652ce7bc1a78202339103168531fd23cf96670abd8`，含两容器路径、源文件哈希和直接字段；没有读取权重 payload 或运行 NPU。下一步若需闭环，仅需读与当前镜像 hash 匹配的 V4.1 配置类、基类及配置映射/构造路径，并寻找历史实例摘要；若无法证明历史对象未被运行时修改，应保留条件结论。

### stage162：V4.1 配置构造和路由默认值的静态链

ZCode `operator-ceiling169-v2-162-v41-config` 退出码 0、`model_verified=true`；两容器读取的五个文件逐一哈希一致，本地将结果 JSON 内五份源码重新计算 SHA-256，均与记录值一致。结合 stage161 与冻结配置，可以建立以下**当前源码和指定输入条件下**的链：

1. `config.py:157,283–285,320–339` 把 `model_type=deepseek_v41` 注册到 `DeepseekV41Config` 后调用 `AutoConfig.from_pretrained`；类地址由 `configs/__init__.py:39` 指定。若没有更改 model_type 的覆盖且注册/解析按该源码正常执行，应构造该类。`DeepseekV41Config` 不是另一份 Transformers 的 `DeepseekV4Config`，后者的 3 不能作为本类的默认值。
2. `deepseek_v41.py:20–42` 消费 `text_config` 并将其中字段摊平到该配置对象，没有创建独立 text_config 属性。冻结配置的 root 与 text_config 均没有 `num_hash_layers`；相应输入也未提供 `decoder`、`generator`、`text_encoder` 或 `per_layer_config`。已快照的 `PreTrainedConfig` 没有 `num_hash_layers` 字面定义，`attribute_map` 默认空，`__post_init__:326–338` 将剩余输入字段设置到对象；`get_text_config:1299–1324` 在没有有效候选时返回自身。因而在额外继承、补丁或覆盖未注入相关属性的条件下，`hf_text_config` 为同一摊平对象。
3. `model.py`（vLLM config）`:628–641` 接收 HF 配置并生成 `hf_text_config`；允许 `hf_overrides` 和字典覆盖，`transformers_utils/config.py:867–872` 也可应用覆盖。冻结启动脚本未显式传入覆盖，但静态脚本无法排除代码或运行期其他入口修改对象。
4. Ascend `model.py:931` 将 `model_config.hf_config` 传入 `normalize_deepseek_v41_config`；`utils.py:168–172` 仅在 `not hasattr(config, "num_hash_layers")` 时补 0；其后 `model.py:210` 的层路由条件为 `layer_idx < config.num_hash_layers and not is_draft_layer`。因此若正常路径构造且直到归一化前该属性始终不存在，则会设成 0，目标层不会选择 tid2eid hash 路由。这是条件推导，不是历史活体对象快照；默认 0 也不能解释 DP 间耗时差。

**尚未闭合：** 当前源码静态路径及当前磁盘 config 不证明历史进程使用相同文件版本，也不证明 Python 父类/其他运行期补丁、额外覆盖或对象修改未添加属性。若需历史确定值，需对应历史启动时配置对象的可信快照或日志（包含配置类、`num_hash_layers`、来源和时间），否则保持 UNKNOWN；不要为此读取权重 payload 或启动 NPU。

结果 `layer0-v41-config-sources.json` SHA-256 `c117604cf6413fe837810f931beb980a0030d4b36f29ff015a3aa7237e652296`；核心源码 `deepseek_v41.py` `5e7052bca7ce79dfa801a3bff20fa2c287ec15d82cf999d003f309b30bf6c1f0`、`config.py` `aaaf215c9166f85656b7491fca3c1abd48d395d084973e7ca8c3e73188b6b2e8`、Transformers `configuration_utils.py` `23ca0a0c0dd1591ccdce5ced8dee09997f0656304f46b85fed725ade85fe25c9`，另外两份哈希在结果 JSON。没有执行模型或设备工作。

### stage163：第 0 层 MC2 的活动代理比较

复用 stage159 的 48 条记录，每条将 `aic_total_cycles` 分别乘 `aic_mac_ratio` 和 `aic_mte2_ratio`，再按每轮 DP 各 8 卡取中位数；这只是导出周期字段乘无量纲比率得到的 **cycle 口径代理**。三轮 DP1/DP0 的组中位数之比，时长分别为 `7.434 / 2.404 / 4.695`，MAC 代理为 `1.284 / 1.001 / 1.124`，MTE2 代理为 `1.112 / 1.019 / 1.118`。这只能说明本次三轮选定样本的组中位数代理比值低于时长比值；非同一工作量或同一请求的前后增长，不足以确定导致组间差异的原因或可恢复时间。

[厂商 MindStudio 字段说明](https://github.com/Ascend/msinsight/blob/efb28234772093b0cabc3c538e71375c957540b2/docs/zh/user_guide/system_tuning.md)描述 AIC MTE2 为 DDR/L2 到 L1 的搬运流水及其占比；但当前采集版本的指标实现、MIX 多核聚合与文档对应关系尚未闭合。代理不是物理搬运字节数、带宽、MAC 运算次数或等同停等时间，不能由其推断饱和、等待或流水阶段因果。导出 ratio 有限小数，舍入方法及采集误差上限未知；活动少时相对量化影响可能明显。

脚本与 JSON 经离线执行逐字节匹配，48 条额外核对 `(rank,run)` 唯一及 timeline 起止时间与同 rank/ordinal 的目标调用精确相符；依赖已保存 stage159 记录、没有重新读取远端 CSV。逐项比值、限制及误差示例见 [LAYER0-MC2-DP-RECOVERY.md](LAYER0-MC2-DP-RECOVERY.md)。复算脚本 `recheck_layer0_engine_work_proxies.py` SHA-256 `d2dda3b9195ef1da2aae83a969ab0ba3dc00ffd80547276b471da31e35fecc1a`；结果 `layer0-engine-work-proxies.json` SHA-256 `9738b889bdf3f8d418f5bb1e9c2bd83c4c07f7c32e36e62c9c1c08482258f860`；厂商说明本地快照 SHA-256 `4c431dac256f6bb69927b721d49316f2716e2b1fc2228924d8d8a84bff5d9fa3`。未触卡。

### 可比较范围内最低的已观察对象：15 rank 主流 AIC MAC 代理

同一历史 profile 中，严格 draft 候选与 target 40 层可以在**有限的统一统计口径**下并列：仅选同一组 15 个 rank 的主流 CSV task，对每个对象的正 `aic_total_cycles` 行计算 `Σ(aic_total_cycles×aic_mac_ratio)/Σ(aic_total_cycles)`。这里“统一分母”是指**分母类型和排除规则相同**，不是 41 个组使用同一个数值分母或具有同样工作量。它是 AIC MAC 活动代理，不是整个 NPU 利用率、MFU、等待比例或端到端时间占比。

| 已观察对象 | 覆盖 | 周期加权 MAC 代理 |
|---|---|---:|
| 严格归属的 3 个 draft block **池化组** | 162 个窗口，15 rank | **2.0151%** |
| target 第 0 层 | 45 个窗口，同 15 rank | **3.7515%** |
| target 第 16 层 | 45 个窗口，同 15 rank | 7.4701% |

在这 41 个**限定分组**的 pooled 排名中，draft 池化组最低，第 0 层是 40 个 target 组中最低。不能将池化的结果归给某一个 draft 层，也不能扩展为完整框架中最低的单模块。针对每个 rank 分别比较其 draft 池化值与其 40 层各自的最低值：**13/15 个 rank**的 draft 更低；DP0 为 8/8、DP1 为 5/7。另 2 个 DP1 rank 不满足，且逐 rank 计算只是分组描述，不能推成独立实验或 DP 导致差异。

**覆盖和选择偏差：** draft 的 162 个窗口仅占 288 个候选中的严格样本；126 个候选因对应 attention 区间未检出 SparseFlashMla 而排除，并未证明是零 query，EP15 的 18 个候选全部未严格归属。15 个 rank 贡献窗口数分别落在 3、6、9、12、15，卡数依次为 1、2、2、7、3；各 target 层在这些 rank 上各有 3 个窗口。两种对象的窗口、模块组成、batch/shape 与请求均可能不同，不能称为同一次请求的配对比较。区间内 **仅主流**参与代理；可归属的辅助流被排除；重复 MIX/多核计数、MTE/Vector 等其他资源和实际整机空闲尚未确定。draft 共排除 3081 条 AIC 周期非正/缺失的记录，target 第 0 层排除 765 条；它们可能是其他引擎的任务，不能视作零活动。另有无效正周期 ratio 时 draft 候选被剔除、target 路径检查拒绝；CSV 的 ratio 是有限精度导出。

此前 `engram-boundary-controls.json` 的 144 个区间、3650 条事件未保存 AIC 周期字段；stage226 已从校验 SHA 的冻结 CSV 有界补查同主流、完整包含的周期字段。得到的是**层间区间代理**，不是 Engram 单模块；区间含多种任务、其他流未计入，权重身份尚未绑定，不能据此得出全框架最低模块排名。含同 15 rank 的结果及零分母处理见 `ENGRAM-BOUNDARY-EVIDENCE.md` 的 stage226 节。不能将不同比率的周期分母相加当墙钟，或将草稿主流结果与此前 target 全可归属流的 3.640% 视作相同采集口径。

复算：任务根目录执行 `python3 agents/diagnosis/recheck_draft_target_mainstream.py`。该脚本核验输入 SHA、逐 rank draft 162 个非重叠窗口的分子/分母相加、40 层各 15 rank×3 窗覆盖及原 41 组排序；逐 rank 和 pooled 排序以 Decimal 分子/分母转成精确有理数比较，并与汇总表存储 ratio 排序断言一致。**精确复算不能恢复原始任务 CSV 的有限 ratio 精度**。脚本 SHA-256 `f72e5091b818bf68b9265bc6cf9db067673fa7138e66cac0d63e4dfe0783fa33`；输入 `module_metrics.csv` `9e96d35b57fe4652da6e955e3a0f1318f8638b5094f42b12c9c2299c3b43fe23`，`draft_call_metrics.json` `0e6f0cb4d810fdd9f7ec99f7b9a7dbe2123a7db7e5a927115e721b8fd8982c48`，`engram-boundary-controls.json` `b88c60f030b6994934aed5cfd37a50b5ab2a0e23060149a27c66415ea4d4e765`。脚本只重算已存汇总，并不能补证原始 CSV 在历史采集时的 hash、请求身份或 profiler 无扰动。本轮未访问设备。


上述逐 rank 复核由本机 ZCode stage `operator-ceiling169-v2-182-draft-target-rank` 再次执行，退出码 0、`model_verified=true`、模型 `deepseek-flash`。结果 `draft-target-mainstream-rank-recheck.json` SHA-256 `50e14758177f00e22ef8f3bfea0b049ad8cc8df5c06ab080045618bb166b2ade`；脚本 `recheck_draft_target_mainstream.py` SHA-256 `f72e5091b818bf68b9265bc6cf9db067673fa7138e66cac0d63e4dfe0783fa33`，无设备操作。精确有理数仅消除复算排序舍入，不能恢复原 CSV 计数器比率已丢失的精度。

### 草稿池化组为何报告了低 MAC 代理：已存任务的同口径分解

对上一节同一组 15 rank、主流、正 AIC 周期的记录，将已经落入窗口的 MC2 任务与其他主流任务按 `Σ周期×ratio / Σ周期` 分解。草稿为严格 162 窗、target 第 0 层为 45 窗；每个草稿窗口的 MC2 通过 rank 和起止时间核对落在该窗口内，target 的 45 个 MC2 与目标窗口的 rank/run 和主流一致。两个对象是不同工作与样本，不是同一次请求的配对。

| 池化对象 | MC2 占已计 AIC 周期分母 | MC2 MAC 代理 | 其他主流任务合并 MAC 代理 | 整窗 MAC 代理 |
|---|---:|---:|---:|---:|
| draft 三块池化组 | 78.03% | 0.5755% | 7.1287% | 2.0151% |
| target 第 0 层 | 81.52% | 2.6285% | 8.7043% | 3.7515% |

例如 draft 的分子、分母分别为 `125654060.574 / 6235486615`；其中 MC2 是 `28003072.879 / 4865653418`，余项由相减得到。表格解释的是**导出周期分母的加权构成**：被记录的 MC2 周期占比较大且其报告 MAC ratio 较低，因此它对所选草稿组低代理值贡献较大。这不能证明 MC2 的实际乘加次数少、硬件空闲、等待通信是原因，或能节省多少业务时长；其他任务包括多种算子，非 MC2 行可能跨资源并发、MIX 计数和 CSV 舍入规则未完全闭合，且辅助流未覆盖。草稿和 target 样本不等、形状与实际 token/专家路由未知；不要把各组数字相减视为相同工作量的差距。

严格 162 个草稿窗口的 MC2 任务时长之和 / 窗口时长之和为 48.47%；这是**多卡多窗口的时长和之比**，不是端到端占比，不等于 78.03% 的周期分母比例，也不是等待时间。分子/分母及对照复算 `python3 agents/diagnosis/recheck_draft_target_mc2_decomposition.py`；脚本 SHA-256 `c0ac77ca0d84b2ab257ca9d0a6bb9a015adb02f017f3668cf0268ebd4fc33aec`，输出列出五份输入文件的 SHA。它核验草稿 162 个窗口逐项匹配、目标 45 个 rank/run/主流匹配及分子/分母包含关系；draft 汇总最初读取原始 CSV 时未逐文件核验 hash，不能补称已做。本轮仅 CPU 离线复算，未触 NPU。

## 严格草稿与同 15 卡 target 0 的主流分母：按算子类型拆分（stage183）

复用严格草稿 162 个窗口和同 15 张卡的 target 第 0 层 45 个窗口，沿用 v2 统计规则：同 model、主 stream、窗口完整包含、排除 metadata；仅有正 `aic_total_cycles` 且 ratio 有效的 CSV 行进入分子/分母。每个草稿窗口的分子、分母及跳过行数和原结果一致；target 对每 rank 的 3 窗合计也与原结果一致。下面的“分母占比”是该类型报告周期之和除以该组报告周期之和；“类型代理”是该类型的周期加权 MAC ratio：

| CSV 类型 | 草稿分母占比 / 类型代理 | target 0 分母占比 / 类型代理 |
|---|---:|---:|
| DispatchFFNCombine | 78.032% / 0.576% | 81.517% / 2.629% |
| QuantBatchMatmulV3 | 7.858% / 10.067% | 7.237% / 9.853% |
| HcPre | 5.196% / 5.538% | 4.527% / 5.726% |
| MatMulV2 | 3.890% / 8.133% | 2.560% / 11.961% |
| SparseFlashMla | 2.271% / 0.767% | 1.546% / 0.800% |
| MatMulV3 | 1.508% / 3.325% | 1.489% / 13.770% |
| TransposeBatchMatMul | 1.245% / 8.300% | 1.123% / 10.051% |

这 7 类构成所统计的全部正 AIC 周期分母，分别恢复草稿 **2.0151%** 与 target **3.7515%** 的原加权代理。草稿 MC2 的高分母权重与低报告比例是其总代理低的主要**算术来源**；它不是证明 MC2 内等待、通信或任何因素是性能根因。非 MC2 中按报告分母最大的类型为 QuantBatchMatmulV3（7.858%），类型代理较高，单靠本表不显示它解释了整体低代理。下一步要区分 MC2 计算规模与等待，需要此前未采到的同次有效 token/专家分布、融合内部阶段或可绑定的同步数据；不应凭这个加权分解推测原因或可节省时间。

草稿 1782 条正周期行、3081 条无正周期/缺失行；target 495 条正周期行、765 条无正周期/缺失行。未进入 AIC 分母的行仍可能有 AIV、通信或其他活动，不是没有工作。算子类型是 CSV 标签；原统计中的重复或重叠行按原口径保留，不是已去重的物理忙碌周期。162/288 草稿只是严格子集、三块混合；target 与草稿窗口数、形状、输入和请求均不同。所有比例为采样导出字段代理，不是整卡利用率、MFU 或端到端耗时。

结果 [draft-target-operator-decomposition.json](draft-target-operator-decomposition.json) SHA-256 `7384031bcfa37219d347f535ae5cf8cf7a099d4f89565867af7288de9d795197`；逐草稿窗与原 162 条对齐、逐 target rank 与原 15 条合计对齐、分类分子分母和行数均独立重算；草稿 MC2 汇总亦与 `draft-mc2-analysis/strict_draft_calls.csv` 一致。ZCode stage `operator-ceiling169-v2-183-draft-operator-decomposition` exit 0、`model_verified=true`、模型 `deepseek-flash`；只读历史 CSV，未运行 NPU。

## 统一 40 层首尾跨度的已选任务比较（stage197）

对已有 16 rank × 3 轮 target 模型采样，**每卡每轮分别**使用第 0 层主流窗口起点到第 39 层窗口终点的跨度作为分母。核验各组 40 个 MC2、8 个 Indexer 和 2 个 Engram 形状相符的矩阵任务归属相同 model 49、同一主流；MC2/Indexer 分别位于对应层窗内、两条矩阵位于层间，50 个所选任务区间逐组两两不重叠。重新汇总得到的 40 层窗、MC2、Indexer 时长与此前数据精确一致。

| 轮次 | MC2 中位占比 | Indexer 中位占比 | 两条矩阵候选中位占比 | 三类任务合计占比中位数 |
|---|---:|---:|---:|---:|
| 1 | 35.89% | 0.52% | 1.69% | 38.12% |
| 2 | 35.88% | 6.65% | 1.50% | 44.02% |
| 3 | 36.30% | 6.62% | 1.48% | 44.39% |

三轮各 16/16 卡的 MC2 合计均大于另外两类；Indexer 在第 1 轮的 16/16 卡均小于两条矩阵候选合计，第 2、3 轮均大于它。三个中位数不能直接相加得到合计中位数。这里仅比较选定任务，**不是全算子排序**；剩余跨度含其他任务、间隔及未导出活动，不是设备空闲。其他流、CPU 或其他 rank 可并行，候选矩阵的 Engram 权重身份未绑定；采样轮次未证是同一请求，不等于基线端到端占比，也不能推实际可节省时间。

CPU 脚本 [recheck_span_cost_comparison.py](recheck_span_cost_comparison.py) SHA-256 `fccedfcf18c92450f899bd2362ef8ffd3703fdd08c92b207c8358a673cfbd03b`；结果 [target-span-cost-comparison.json](target-span-cost-comparison.json) SHA-256 `e2c2ab268a3d8f09b83b721ce835244ea3ce8ae3ba0c59174bcd4052c836b840`，逐组分母、各类型合计、剩余及完整输入 SHA 均在结果中。ZCode stage `operator-ceiling169-v2-197-target-span-comparison` exit 0、`model_verified=true`；与本地 CPU 复算的 JSON 完全相同，未运行 NPU。

## 40 层跨度内主流已导出任务类型排名（stage199）

在 stage197 的 **16 卡 × 3 轮、model 49、每卡各自的主流及第 0 层起点至第 39 层终点**跨度内，扩展统计 CSV 中的任务类型；共观察到 38 个类型标签。逐类型先截断至各自跨度，再取时间区间并集；每轮取 16 个卡内占比的中位数。排除空类型、名称含 metadata 和零时长事件；本次所选 model/stream 未出现这类排除记录或无法归属的坏时间（计数均为零）。每卡跨度独立计时，没有跨卡时间线求和。

| 已导出类型 | 第 1 轮 | 第 2 轮 | 第 3 轮 |
|---|---:|---:|---:|
| DispatchFFNCombine（MC2） | 35.89% | 35.88% | 36.30% |
| QuantBatchMatmulV3 | 11.48% | 9.96% | 9.92% |
| HcPre | 8.78% | 7.66% | 7.61% |
| VllmAscendQuantLightningIndexerV2 | 0.52% | 6.65% | 6.62% |
| TransposeBatchMatMul | 6.56% | 5.66% | 5.64% |
| MatMulV2 | 5.91% | 5.15% | 5.12% |

这里按展示方便列出前几项及 Indexer，**不是表内从上到下的每轮完整排序**：例如第 1 轮 Indexer 排在表中数项之后。所有 48 个卡轮次中，MC2 均为当前选定主流已导出类型里占比最大的标签；这些类型标签可包含多个不同形状及调用位置，不等同模型模块。48 组均有 40 条 MC2、8 条 Indexer，与 stage197 对应类型耗时精确一致；每组另有 89 条 MatMulV2，stage197 所选的两条候选矩阵任务时长不大于 MatMulV2 类型整体并集，不能把这两条当作整个 MatMulV2，更不能仅凭形状认定其 Engram 权重身份。

全部类型的主流已导出区间并集占各卡跨度的中位数为 **88.08%、90.36%、90.35%**。48 组的“各类型并集之和减全部类型并集”均为 0，说明**本次选中记录的不同类型区间没有观察到重叠**；各类型内部即使存在重叠，已通过并集处理。此结果不说明其他流、其他模型或未导出设备活动无重叠；未被选中记录覆盖的跨度不能称为设备空闲。覆盖率不是芯片利用率，时长排名也不证明根因、关键路径、可实现提速、全框架/所有请求排名，采样轮次未证明相同请求或工作量。

证据 [target-fullspan-types.json](target-fullspan-types.json) SHA-256 `dcbcf052d24b9ecffd4e644960487c5291ad4b4feaaf5399f9d0502875193495`；只读查询脚本 [query_target_fullspan_types.py](query_target_fullspan_types.py) SHA-256 `76882ba389fb43aefc567c994d050a3fe264deed2d315eb130a447ef60cb7182`；封装脚本 [run_target_fullspan_types.py](run_target_fullspan_types.py) SHA-256 `4579d88bb0398984b05f64799d7d24d8a484644fc5baeb861057dc8738601ae6`。查询核对三个冻结输入和 16 个原 CSV SHA；逐组对照 stage197 的 MC2 和 Indexer；无坏时间、无归属不明记录。ZCode stage `operator-ceiling169-v2-199-target-fullspan-types` exit 0、`model_verified=true`、`deepseek-flash`，仅只读历史 CSV，未运行 NPU。`COMPLETE` 仅表示脚本规定的检查通过，不代表采集覆盖全设备或全请求。

## 基线整轮草稿计数与按位置接受计数（离线复算）

现有 **未开启 profiler 的两轮 C8 基线**各完成 8 个请求，每轮服务计数生成 8192 token。`measure1` 的 `spec_decode_num_draft_tokens_total` 差值为 8755，`spec_decode_num_accepted_tokens_total` 差值为 6447，前者作为分母的聚合接受比例为 73.64%；`spec_decode_num_drafts_total` 差值 1751，聚合平均每次 draft 接受 3.682 个。`measure2` 分别为 8730、6456、73.95%、1746、3.698 个。计数原始文件是 `/Users/wio/work/169-vllm-ascend-reproduce-172/benchmark-20261028/raw-results/results/measure1/progress.json` 与同级 `measure2/progress.json`；这些是服务指标采样差值，不是逐请求记录，也不是 draft 的独立执行时间。

在这两轮各自的 `metrics-before.txt` / `metrics-after.txt` 中，指标 `vllm:spec_decode_num_accepted_tokens_per_pos_total` 的 HELP 文本为 Accepted tokens per draft position；按 `engine=0,1` 分别计算前后差值并相加，position 0–4 的计数依次为 measure1 `[1751, 1647, 1407, 954, 688]`，measure2 `[1743, 1598, 1289, 1040, 786]`。各位置计数之和分别等于该轮总 accepted 计数；这是总量一致性检查，并不证明两个抓取点内每个请求都各出现一次、每个位置有相同数量的候选，或给出任何单请求的接受率。缺少逐位置草稿候选数及请求 ID 等关联，不能把位置计数除以 draft 次数当作“该位置有候选时的条件接受率”；按当前源码语义可条件解释为接受至少该数量草稿 token 的 draft 占比，详见下节。

复算：运行 `python3 agents/diagnosis/spec_position_accounting.py`；脚本固定读取上述两轮原始文件，检查完整的 2 engine × 5 position、计数非负和总数一致，并输出原始文件 SHA-256。原有汇总也可查 `agents/diagnosis/spec-counter-accounting.json`。客户端各请求详情包含请求输入输出 token 和时间点引用，但该证据未提供草稿提议/接受 token 的逐请求绑定。两轮完整批次分别耗时 15.6825 和 15.8347 秒，包含 prefill、target、draft、调度等，不能视为草稿模型耗时；本分析未建立与已采样草稿窗口的时间或请求对应关系，因此无法从这些数确定草稿瓶颈、节省时间或精度。

### 草稿接受长度的条件分布（stage206–208）

stage206 按安装包 RECORD 查找源码，但安装包版本为 `0.30.0+empty`，目标源码选择数为 0；这不是指标不存在。stage207 只读当前容器固定 `/vllm-workspace/vllm/vllm/v1` 下 metrics/spec_decode/core/engine/sample 的 73 个 Python 文件；`spec_decode/metrics.py:41–49` 每次 observe 将 `num_drafts` 加 1，并将 `i < num_accepted_tokens` 的位置计数各加 1；`:266–281` 将这些累计到对应 engine 的 Prometheus 计数器。源码 SHA-256 `35e0a709042471559937f1a1979a721af9b910476efaa8b6e29865bace985afe`；摘录与范围见 `spec-counter-workspace-semantics.json`。`scheduler.py:1980–2001,2798–2814` 显示当前路径，但不证明历史对象当时调用或源码没有在运行期间变化。

**在历史基线使用相同累加逻辑、两类指标抓取窗口一致且 counter 未重置的条件下**，可由相邻位置相减得到以下分布：

| 接受草稿 token 数 | 第 1 轮次数 | 第 2 轮次数 |
|---|---:|---:|
| 0 | 0 | 3 |
| 1 | 104 | 145 |
| 2 | 240 | 309 |
| 3 | 453 | 249 |
| 4 | 266 | 254 |
| 5 | 688 | 786 |

因此条件下全 5 个接受占所有 draft 调用的 **39.29% / 45.02%**。这不是第 5 个 token 在被提议时的条件接受率：还需要对应位置的实际提议数。计数是每轮各引擎汇总，并非逐请求数据；不与 profiler 窗口绑定，也不说明精度、draft 运行时间或可节省时间。计数之和和加权接受总数一致只验证算术，不能单独证明历史源码或采集时刻一致。源码内建位置计数除以 drafts 的口径可称为“接受至少该数量的比例”，不能混同为条件成功率。

本机 ZCode stage206/207/208 均 exit 0、`model_verified=true`；阶段 206 文件选择为空，其余分别只读当前容器源码和对已存计数做 CPU 算术复算，未操作设备。复算 `python3 agents/diagnosis/spec_length_distribution.py`，结果保存在 `spec-length-conditional.json`。

stage211 只读比较基线容器与 stage 容器当前 `spec_decode/metrics.py`：二者同镜像、文件 SHA-256 都是 `35e0a709042471559937f1a1979a721af9b910476efaa8b6e29865bace985afe`，目标文件无覆盖挂载。它加强了当前两容器源码对应的证据，但不证明基线运行时实际加载的代码对象或无动态 patch，也不能证明指标抓取时刻完全一致。结果见 `spec-metrics-container-compare.json`；ZCode stage211 exit 0、`model_verified=true`，未触设备。

## 草稿候选筛选的敏感性：零条与一条导出 SMLA

对已存 v5 的 288 个候选窗口按 `attention_count_allstreams` 分组，沿用此前的**主流、正 AIC 周期加权 MAC 活动代理**。这不是新采样：以 rank 和起止时间核验 162 个一条 SMLA 的窗口与 v2 严格组完全一致，且逐条核对原有主流、model 及分子分母。

| 已导出 SMLA 数 | 候选窗口 | rank 数 | 合并分子 / 分母 | 代理 |
|---|---:|---:|---:|---:|
| 0 条，未通过严格归属条件 | 126 | 16 | 57835979.389 / 9394668276 | 0.6156% |
| 1 条，严格组 | 162 | 15 | 125654060.574 / 6235486615 | 2.0151% |
| 合并，仅敏感性分析 | 288 | 16 | 183490039.963 / 15630154891 | 1.1739% |

两类均出现的 15 个 rank 中，12 个的零条组代理较低，而 EP0、EP1、EP2 三个反向；每组包含的窗口数、工作量、周期分母不同，15 个比较也不是请求级配对。零条组中 EP0–2 只有 9 个窗口、占该组周期分母约 0.789%，因此池化结果不能代表各卡同向变化。EP15 的全部 18 个候选均在零条组；没有严格组可以比较。

**含义与边界：**126 个窗口虽然满足原候选的 top3、8 个本地专家及 4 个 HC 标记规则，缺少 SparseFlashMla 使其无法通过原严格分类；“零条”只描述指定窗口内未检出该名称，不证明零 query、没有 attention、属于特定草稿层，也不能确定实际运行分支。池化 1.1739% 只是以放宽分类作敏感性，不应替代严谨的 2.0151% 草稿组，不能作为整卡利用率、性能收益或缺失条件的因果证据。此差异说明选择样本对结果有实质影响，后续优先寻找缺失窗口的调用身份和真实 query/工作量，而非仅根据池化值排序解释原因。计数器有限精度、多流、资源范围和采集不完整限制不变；0 条/1 条组分别有 2097/3081 条无正周期或缺失周期行未进入分母，这些行不是空闲。

只读离线复算：`python3 agents/diagnosis/recheck_draft_attention_selection.py`；输入哈希分别 `6473445dda048c157cf7e451bf959b973c5138a05e0272172a5ea98cfef21923`（v5）和 `0e6f0cb4d810fdd9f7ec99f7b9a7dbe2123a7db7e5a927115e721b8fd8982c48`（v2）；脚本 SHA-256 `710a79af2ea5c9a90ddcc5800d3cbd4772e2475f1f9e2f2736eeea011620f427`；保存输出 [draft-attention-selection-sensitivity.json](draft-attention-selection-sensitivity.json) SHA-256 `5affcbfbd1ff7ce725064e85f05506d0db0382f1a14d5cbc0e54e5e5bb0176b0`。ZCode stage210 exit 0、model_verified=true，输出与已保存 JSON 相同。这验证离线输入和聚合计算，不修复原始采集归属缺口；未操作 NPU。

### 零条导出 SMLA 候选的主流 AIC 分母构成

针对上表未严格归属的 126 条窗口，在已保存的 16 rank 历史 `kernel_details.csv` 内，使用相同 v5 主流、model、窗口边界、排除 metadata、正周期和有限 MAC ratio 规则，按导出 Type 拆分。逐窗口核对原分子、分母、跳过条数完全一致，且每个窗口有且仅有一个正周期 `DispatchFFNCombine`，Task ID 与原窗口记录相同。CSV 哈希绑定此前 [indexer-attention-paired-runs.json](indexer-attention-paired-runs.json) 保存的 16 rank 文件哈希；这是当前文件与此前证据的一致性，不证明历史采样时无丢失或无扰动。严格组复用此前 stage183 的同口径拆分，15 个共同 rank 的 CSV 哈希一致；两组窗口和工作量不同，且零条组多一个 rank，不是配对对照。

| 候选组 | 窗口/rank | MC2 的 AIC 周期分母占比 | MC2 活动代理 | 其他类型合并代理 | 窗口整体代理 |
|---|---:|---:|---:|---:|---:|
| 零条 SMLA，未严格归属 | 126 / 16 | 93.7308% | 0.1924% | 6.9435% | 0.6156% |
| 一条 SMLA，严格归属 | 162 / 15 | 78.0317% | 0.5755% | 7.1287% | 2.0151% |

这里“占比”是**导出的 AIC 正周期数值分母之和**的构成，不是设备时间占比、单卡利用率或请求耗时。MC2 是融合通信和计算的导出算子类型，其计数器低不能证明通信等待、矩阵工作量小或任何其他根因。差值只解释这两个样本池中代理值的算术构成，不说明把 MC2 改快能带来多少 TPS；其余类型、两组请求、负载、数量和窗口均不同，且遗漏其他 stream、无正周期任务和可能重复的导出记录。未严格归属组不能作为已确认草稿层的全局排名。

复算 `python3 agents/diagnosis/recheck_draft_zero_type_comparison.py`，其 SHA-256 `1ca843bba5a8a7236ff52eab76e4ff6f70c7913ea8b5225195acbba31b669df9`。新结果 [draft-zero-attention-types.json](draft-zero-attention-types.json) SHA-256 `9a1a75cd23ebfc4a1001f2eabd92a06d794a976225254ca4d98745e7ed9919e5`；[查询](query_draft_zero_attention_types.py) SHA-256 `b48b4a7d939e587c29fb9434c8385c305add41be075a412c0bff60298212a02a`，包装脚本 SHA-256 `88b1c55fa01bcce8726725a1177606673bb7efdfa2f0b72a63a27f317187c72c`。本机 ZCode stage213 和 stage215 均 exit 0、model_verified=true；仅对冻结历史文件进行 CPU 只读查询及复算，未操作 NPU。

### 既有草稿候选窗口：同流 MC2 时间占比（stage216）

在原冻结的 288 个候选窗口中，通过同 rank、model、stream、task ID 和窗口完整包含关系，与哈希核验过的 16 份历史 `kernel_details.csv` 中 `DispatchFFNCombine` 记录逐窗唯一关联。两组都是**MC2 时长之和 / 各窗口时长之和**，不是窗口比例的算术平均；不按跨卡时钟重叠推导墙钟。

| 样本组 | 窗口 / rank | MC2 时长总和 | 窗口时长总和 | 时间和之比 | 各窗 MC2 中位时长 | 各窗窗口中位时长 |
|---|---:|---:|---:|---:|---:|---:|
| 严格归属 | 162 / 15 | 123619.837 µs | 255046.793 µs | 48.4695% | 181.186 µs | 723.376 µs |
| 未严格归属（无已导出 SparseFlashMla） | 126 / 16 | 210983.602 µs | 288918.251 µs | 73.0254% | 172.197 µs | 709.906 µs |

严格组比例与既有 `evidence/draft-module-diagnosis.json` 的 `strict_moe_component.sum_mc2_time_over_sum_bracket_time` 一致。独立逐条复核 288 窗与 `eager-draft-mainstream-attribution-v5/draft_call_metrics.json` 的键和分组，核验区间包含、时长和每组及每 rank 求和；逐 rank 数值和每条记录见下列 JSON。两组在共同的 15 rank 中，10 rank 的未归属组时间和之比更高、5 rank 更低；这仅为样本构成描述，**不同组不是相同请求或等工作量配对**。未归属组不能由缺少导出 attention 推断本地 query 为空、没有 attention 或已确定草稿身份。组内长尾使总时长和之比与每窗中位数不直接对应；两个中位数也不代表同一窗口。

这两个比例均不是整机、跨流、端到端或请求关键路径的耗时占比，不是可消除的时间，更不能视为通信等待或根因。尤其 73.0254% 的时长比例与该组 93.7308% 的 **AIC 周期分母占比**口径不同，不应混用；其他流、请求和 rank 的活动可能重叠且未在这里归因。

证据 [draft-mc2-window-time.json](draft-mc2-window-time.json) SHA-256 `0d3a7e8cf68545937c45f69f20bfce84d5fab8e39040fa2a7baf9f3869445d07`；查询 [query_draft_mc2_window_time.py](query_draft_mc2_window_time.py)、封装 [run_draft_mc2_window_time.py](run_draft_mc2_window_time.py)，输入及源码脚本哈希保存在结果中。ZCode stage216 exit 0、`model_verified=true`；只读既有 CSV，没有启动 NPU 或修改推理代码。

### 同卡、同物理输入首维下仍有长短 MC2 记录

离线关联前述 288 条窗口与 [16rank-draft-alltoall-distribution.json](16rank-draft-alltoall-distribution.json)：以 rank、窗口开始时间和主 stream 唯一匹配 288/288 条；前者再以 rank、model、stream、task ID 和窗口起止与 v5 逐条唯一匹配。原 bracket 查询要求同流内 `DispatchFFNCombine` 唯一，并保存该行的输入形状；bracket 摘要本身不含 MC2 task ID，不能称其直接按 task ID 关联。复算脚本、输入 SHA 和输出见 [recheck_draft_mc2_shape_rank.py](recheck_draft_mc2_shape_rank.py) 和 [draft-mc2-shape-rank-recheck.json](draft-mc2-shape-rank-recheck.json)。

导出 MC2 输入第一张量首维为 2 的 96 条窗口中，**16 个 EP 每个均有时长低于 300 µs 和不低于 1000 µs 的记录**。因此固定这批记录的卡号和物理输入首维，时长仍显著变化；不能仅用这两个字段解释差异。阈值用于描述观察，未预设其性能含义。该统计合并了 63 条严格归属和 33 条未严格归属候选，未归属不代表草稿身份已经确认，也不是设备或请求的总体分布。

物理输入首维不等于有效 token 或各专家接收行数；样本没有绑定相同请求、相同有效行、专家路由、通信状态或相同工作量，也未确定根因。特别是不能由此排除动态工作量或卡的影响，不能将跨 rank 记录视作同一 collective。仅复用历史汇总做本机 CPU 离线计算，未触碰 NPU。

### 已选窗口的 rank 汇总权重敏感性

复用 `draft-target-operator-decomposition.json` 的逐窗已存分子、分母和原筛选规则，分别对两组同 15 个 rank，先各自汇总该 rank 正 AIC 周期任务再计算代理，再取 15 个 rank 的等权平均：严格 draft 162 窗得到 **2.7830%**，而全部周期加权池化为 **2.0151%**；target 第 0 层 45 窗分别为 **5.7023%** 与 **3.7515%**。两种统计都来自同一批窗口，差别来自各 rank 的周期分母权重；不能将任一种当作全芯片 NPU 利用率，也不能用等权平均纠正没有覆盖的窗口、不同工作量或未记录的流。

另外以同一 rank 成对剔除的方式重复池化 15 次：草稿组为 **1.9694%–2.0994%**，target 第 0 层为 **3.5960%–3.9476%**；每次都仍是草稿组较低。该有限样本排序对删除单一 rank 稳定，但这不是对 126 个未归属窗口、缺失 rank、其他模块和全部请求的结论。草稿组 1782 条正周期行、3081 条非正或缺失周期行；target 第 0 层相应为 495、765 条。排除行可能包含其他引擎的工作，无法从这些计数推断空闲或设备活动比例。

复算命令 `python3 agents/diagnosis/recheck_sample_rank_weight_sensitivity.py` 只读已保存逐窗 JSON，并校验输入 SHA-256 `7384031bcfa37219d347f535ae5cf8cf7a099d4f89565867af7288de9d795197`、15 个共同 rank、样本数量及去重键。脚本 SHA-256 `6be2917dc94413b75f911f254a417b2349dd629d540fae7ed4c299b6e1f922d5`，存档输出 `sample-rank-weight-sensitivity.json` SHA-256 `0222aeac17bdcaa3de3308a6e4d16239e7de1cbd5b06038a2f9f0ac6b9b6a4c7`。结果只是汇总权重的敏感性，不是纠正后的真值；有限精度、任务跨流/MIX 以及采集完整性的既有限制仍然适用。

## target 第 0 层的双引擎排除行与排序（只读复核）

复核 `captured-layer-engine-proxies/summary.json` 的 16 rank、48 窗口、第 0 层同一集合，共 1824 条已归属记录。生成脚本逐行分别按 AIC 和 AIV 的周期字段判定：AIC 非正/缺失 1248 条，AIV 非正/缺失 432 条，两种记录属于同一 1824 条集合；因此集合差至少 `1248−432=816` 条（占全部记录 44.74%）AIC 非正/缺失但 AIV 周期为正，且 AIV 正周期对应 ratio 无缺失。不能仅凭这两个汇总数确定交集全部内容或排除原因，更不能把这些行认作计算空闲或计算耗时。

同一组第 16 层 AIC MAC 代理 **7.5786%** 低于第 27 层 **8.4209%**，AIV Vector 代理却是 **1.1372%** 高于第 27 层的 **0.9777%**。各自使用不同引擎周期分母加权；这给出排序不一致的实际反例，不代表整个 NPU 哪层最低、哪层更慢，也不能相加。这里是 16 rank 48 窗，与严格草稿对照的 15 rank 45 窗不同，不能混用两组百分比。

输入 SHA-256 `010499ffc68cd21e042407cf075b339736c5f2d547dd538547b9eef261a5ed9f`；生成逻辑 `tools/analyze_layer_engine_proxies.py` SHA `82637d71b7d51cbbd4861b90387e5f5290c9ced743aa43dbb0b54cb91c036c48`。两类 ratio 的分母各有对应资源解释，华为 [CANN 9.1 指标说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/devaids/optool/docs/zh/user_guide/msopprof_user_guide.md) 提供 AIC/AIV 周期和活动比例的类比，但与本次 CANN 9.2 exporter 的版本、聚合及零值语义未绑定。对话中已通过 CPU 检查这三个层项与生成逻辑，未启动 NPU。

本机 ZCode stage262 `deepseek-flash`、exit 0、model_verified=true 复算 `recheck_aic_aiv_sampling_scope.py`；[结果](aic-aiv-sampling-scope.json) SHA-256 `bb30ebc39afd1ff11328542eeb1817d84abfb0d1e4880566c8501f9dbf6cd84f`，与本机独立执行标准输出一致。只复核汇总算术和排序，不验证原始采集完整性或 exporter 语义。

### 全部流下 AIC/AIV 交集的精确归属（stage263–266）

按原脚本规则重放冻结的 16 个 CSV：仅选 model 49、窗口完全包含并排除 metadata，不过滤主流。每卡 120 个窗口的归属行数，以及第 0 层 48 窗 1824 行的 AIC/AIV 分子、分母与非正或缺失计数均与原汇总一致。以窗口的主流 ID 标记，得到以下记录数：

| AIC 周期 | AIV 周期 | 总行数 | 主流 | 其他流 |
| --- | --- | ---: | ---: | ---: |
| 非正或缺失 | 正 | 1104 | 816 | 288 |
| 正 | 非正或缺失 | 288 | 240 | 48 |
| 正 | 正 | 288 | 288 | 0 |
| 非正或缺失 | 非正或缺失 | 144 | 0 | 144 |

因此此前仅由集合差得到的 816 下界现精确为 1104，其中 816 行属于标记主流。该组类型包括 HcPost 96、DynamicQuant 96、DequantSwigluQuant 48、RmsNorm 144、InplacePartialRotaryMul 144，以及三种通信任务各 48 行。它们是导出记录数，不是物理调用数、等待时间、整芯片利用率或收益；正周期和 ratio 是导出值，非正或缺失不能认定物理设备空闲。不同流可能重叠，Exporter 采集范围及混合核重复计数仍未知。按 rank 内完整行及 model/stream/task/起止键复核，无重复额外记录；但不同记录仍可能映射同一物理活动。

原查询 stage263–265 因 TextIOWrapper detach 后退出上下文引发 ValueError，失败且无有效输出；修复资源生命周期后 stage266 成功。ZCode deepseek-flash、exit 0、model_verified=true。查询脚本 SHA d99a37b5998ab7c8a8ac7a317be1ef4785c20b6b61c6fc6af241dfed88526ff9，wrapper SHA f98056cd67b396abfd15d96811f4b334ce71e1530f4b4a4e10e39f8b16a09f5c，结果 [layer0-aic-aiv-intersection.json](layer0-aic-aiv-intersection.json) SHA 0393893f08cae3421eb385727e8c658a769a1da9b5311dfec6de6784cc709d94。仅只读历史文件，无 NPU 负载。

### stage266：第 0 层 AIV 代理的算术构成和已有重组边界

对上述同一份 `layer0-aic-aiv-intersection.json` 按 type 汇总 **AIV 周期为正且对应 vec ratio 有效** 的条目，周期和分子使用 JSON 保存的十进制字符串：全组分母 `4508918621`，分子 `20082171.015`，周期加权代理 **0.44539%**。主要构成如下，分母占比是已导出条目的周期权重，**不是耗时占比或硬件份额**。

| 导出 type | 正 AIV 周期行数 | AIV 分母占全组 | 该 type 的加权 vec 代理 | 主流 / 其他流行数 |
| --- | ---: | ---: | ---: | ---: |
| DispatchFFNCombine | 48 | 86.3182% | 0.1736% | 48 / 0 |
| HcPre | 96 | 4.4595% | 1.9050% | 96 / 0 |
| QuantBatchMatmulV3 | 96 | 2.3200% | 1.8381% | 96 / 0 |
| ForeachAddList | 48 | 1.5853% | 0.4579% | 48 / 0 |
| SparseFlashMla | 48 | 1.3141% | 1.4154% | 48 / 0 |
| Transpose | 48 | 0.8544% | 0.3000% | 48 / 0 |
| hcom_alltoall_ / hcom_allGather_ / hcom_reduceScatter_ | 各 48 | 0.7396% / 0.6724% / 0.1250% | 0.1531% / 0.2066% / 0.2131% | 各 0 / 48 |

`DispatchFFNCombine` 的 AIV 分子为 `6757946.981`、分母 `3892017873`；其余所有类型合计的同口径加权代理为 **2.15987%**。因此在这个固定样本及计数规则下，低的整体 AIV vec 代理在算术上受 MC2 的高权重和低 vec 比例主导；剔除它是一个**统计对照**，不是假定它可以被删除或预测可节省的时间。其余类型不论计数大小，不能以 vec 比例直接判断执行是否低效、正在等待，或计算是否是该模块的目标工作。

类别与归属限制：DispatchFFNCombine 是已识别的 MIX 复合任务；HcPre、QuantBatchMatmulV3 和 SparseFlashMla 也在本样本中同时出现 AIC 与 AIV 正周期字段，不可当作纯 Vector；其他条目 AIC 字段非正或缺失也**不证明**其硬件路径必为纯 Vector。通信条目的 AIV 正周期行与同名非正周期行均出现在其他流，可能是不同导出视图；对具体通信未证明独立物理事件及与 MC2 内通信的关系，不能把它们与内部工作相加。样本同时保留同一 model、窗口内的其他流，窗口内邻近并不证明生产消费或同一请求链；字段不完整、重复物理活动、采样/导出聚合与历史二进制绑定仍有限制。

可用于继续核查而非新增假定收益的既有边界：**F21** 已分析 MC2 内 dispatch、GMM、activation 和 combine 的现存交错与分批重排，必须保留行域、scale、双缓冲别名、跨核和跨 rank 发布；仅凭低 vec 代理不能定性某处等待。**F03** 已从冻结 HC 源码指出有外部 `pre_mix` 时 `y` 不依赖本次投影及系数，可研究与系数计算分支拆分并提前交付，但要保留其他 HC 输出、舍入、共享 scratch 和同步；采样 shape 与该分支相容，不证明 tensor 身份或历史 binary。`ForeachAddList` 不能仅按名称认作 F23 的 `muls_add_triton`，主流记录本身不证明合流到 HC；逐 tensor/事件绑定仍是判断边界是否真的存在的缺口。上述候选均已登记，不将已实现的复合算子重复称为新融合。

ZCode stage267 `deepseek-flash`、exit 0、model_verified=true，复算 `recheck_layer0_aiv_mc2_contribution.py`；[结果](layer0-aiv-mc2-contribution.json) SHA-256 `c417f8e821d89e9d181462203dde3e7e180360a15edd1cb07e05e146fb7358c5`，与本机复算标准输出一致。这只核对聚合算术，不验证采集完整性或计算路径。

### 厂商同名字段口径与版本边界

华为 [CANN 9.1.0-beta1 op_summary 字段说明](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta1/devaids/Profiling/atlasprofiling_16_0067.html) 对 A2/A3 将 total_cycles 分为 AIC 与 AIV，总周期为 task 各 block 的周期累加；vec/mac ratio 是对应类型指令周期占 total cycle 的比例。若本次导出使用同义口径且覆盖与精度条件成立，则本报告加权值是所选记录按报告 block-cycle 加权的数值，并不是墙钟时间占比、实际操作数或芯片利用率。若周期分母变化，低 ratio 不单独证明指令工作减少或等待增多。

该文档说 task_time 为 l0/off 时不提供相应 PMU，部分长算子指标可能显示 N/A；它不确定本次任何零值原因。其 aicore_time/aiv_time 是按各 block 同时开始且执行时长相等的假设给出的理论时间，与另一本 msinsight 文档的简短字段表述不完全相同，不能把它们直接等同于实际内核纯执行时间。本次历史版本是 CANN 9.2、导出为 kernel_details.csv，而页面是 9.1 beta 的 op_summary；缺少对应版本导出实现绑定，本段仅为条件解释，不改变已报告的历史数据或得出新的因果结论。

### kernel_details 指标来源与 Python 导出链（stage268–272）

ZCode 在固定当前诊断与 staging 容器核验 ID、镜像和源码路径无覆盖挂载，并只读 profiler analysis 两个固定目录内 849 个 Python/SQL 文件，五个精确指标名共命中 5 个文件；未查编译文件。随后只读取命中 parser 及其 import 文件的有界源码。`_kernel_view_parser.py:21–33,35–40,56–78` 从 CANN OP_SUMMARY 文件读取，按选择的输入头投影输出 `kernel_details.csv`；`_op_summary_bean.py:13–19` 按头名从输入字典获取值；`_common_bean.py:20–21` 原样保存该字典；`_file_manager.py:34–53,75–89` 使用 csv.DictReader 和 csv.writer 读取并写出。对于输入中存在、输出保留且不重命名的 ratio/cycles 字段，这段当前 Python 路径传递字段字符串，没有对字段重新计算或数值舍入；CSV 转义与头映射不保证文件字节完全相同。部分未选字段可被省略，缺键返回空，step ID 也可另外添加。

这加强了厂商 op_summary 字段口径对当前导出路径的相关性；但读取的源码文件身份是当前首次观察，虽然两容器当前镜像和挂载核验通过，仍未证明历史采集时实际加载了同一份代码，也未证明 CANN 上游如何计算 block-cycle、精度及舍入、MIX 计数去重或采集覆盖。CANN 9.1 beta 文档与历史自报 9.2 版本差异仍在，不能把低 ratio 改判为等待/空闲。

固定源码审计结果：[目录索引](ratio-export-source-inventory.json) SHA b9eaeb31ed0daff3f5509d2e6cd3d93402023c2b0c44bf12845b263a87363583；[parser 摘录](ratio-export-source-excerpt.json) SHA 90d336a2aaf1904a92b15099d5bc84f08b489159d5fc36730f13b14c2a522195；[bean 摘录](ratio-bean-source-excerpt.json) SHA 157648763dd9cd52cffe883aaabf35759f3b745d9f1ac65fee0c6ec7fc3c488b；[I/O 摘录](ratio-io-source-excerpt.json) SHA a7535fc7fb177aab1611e75ab505255ce35ad54fcad34e47d690adf80f1d9740。ZCode stages 268–272 均 exit 0、model_verified=true、deepseek-flash，只读源码，无 NPU 工作。

### 指标草稿长度的饱和条件推导（2026-10-30）

本节补充上面的条件接受分布；不是新增采样，也不要求新的目录或推理实现变更。已存 `spec-counter-workspace-semantics.json` 中的 `spec_decode/metrics.py:33–49` 摘录表明：当 `SpecDecodingStats.new(num_spec_tokens=5)` 执行时，逐位置数组长度为 5；每次 `observe_draft(k,a)` 增加 draft 次数和 k 的总数，再按 `range(k)` 访问逐位置草稿数组。对应源码 SHA-256 为 `35e0a709042471559937f1a1979a721af9b910476efaa8b6e29865bace985afe`，摘录文件 SHA-256 为 `913531db3764658a7721336a0eef492e601aa9db1b4522ce008f8930b7af035a`。同一源码 `:266–281` 将这些统计累计到 engine 计数器；scheduler `:1980–2001,2798–2814` 将经过 invalid-spec 调整的计数传入 observe，故 k 是**指标记录的草稿长度**，不自动等同实际有效 token 数。

条件假设必须同时成立：历史被计数的所有调用实际使用上述相同代码和长度 5 的对象；每次计入发布的 observe 均正常完成，整数 k 没有被其他路径改写；两个累计指标的前后差分恰好覆盖同一组完整调用和相同引擎范围、无 counter 重置或抓取错位；统计未被其他进程或运行时 patch 注入。历史逐调用代码对象、抓取一致性和逐请求标识未直接捕获，这些仍是条件而非已证历史事实。正常完成且列表长度为 5 时，任何 `k>5` 都会访问越界，因此所有被计调用的 `k_i≤5`。这个上界由成功调用及数组长度得到，不从平均值猜测。

已存服务计数差分两轮分别为 `(N,Σk)=(1751,8755)` 与 `(1746,8730)`。各自满足 `Σk=5N`。因每项 `k_i≤5`，所有 `5−k_i` 非负且其总和为零，故在上述条件下**每项均有 `k_i=5`**。这是饱和等式的逻辑推论；并不由平均值单独成立。如果某个长度允许大于 5，长度 4 与 6 的平均同样可以是 5。如果 observe 中途失败仍计入且被发布、配置长度改变、不同 counter 时间窗不一致或统计被重写，结论均需重证。

同样条件下，每个位置 0–4 在每轮都对应 N 个**由该指标语义计数的候选位置**。位置接受计数来自 `metrics-before.txt` 与 `metrics-after.txt` 逐 engine 差分相加：

| 轮次 | draft 计数 N | 接受位置 0–4 | 各位置接受数 / N |
|---|---:|---|---|
| measure1 | 1751 | 1751, 1647, 1407, 954, 688 | 100.0000%, 94.0605%, 80.3541%, 54.4832%, 39.2918% |
| measure2 | 1746 | 1743, 1598, 1289, 1040, 786 | 99.8282%, 91.5235%, 73.8259%, 59.5647%, 45.0172% |

它们是按框架指标计数、两轮聚合的各位置数除以本轮 N；不是逐请求概率、以“前一位置已接受”为条件的成功率，也不证明位置内容、padding 或数值正确。条件下，原先“各位置被指标计数的候选数未知”的结论可收窄为各 N 个；没有闭合的条件仍不能省略。它不把一次 observe 认作一次 NPU 执行或不可并行的步骤，也不能用计数推出理论延迟、端到端 TPS 上界或允许融合、预计算和重排的所有实现的分组限制。

只读复算 `python3 agents/diagnosis/spec_position_accounting.py`。脚本 SHA-256 `370438b395722c1e2c0a8e7df3aa36340e16fe9a4cd58e075b85e89e9cb87bb2`。measure1 的 metrics-before / metrics-after / progress SHA-256 依次为 `d9dacde0fbdd5d82a4f37b2f4bfde5501f5a5109b54dc7c70b753f3dd50c7df8`、`e1bfaffa59f9cdfe989f908c90b0e6e7c678c4a63d69377d78014b8fb8d1ea6e`、`6f963b20d3e223976fc872649a5004d25f7cfe16a53689f98548e34ed63bd38c`；measure2 依次为 `df4760abb2e74ecaf8f89e3abbe7f4345d77349d5b69b53eaa5ec0426d29e5b0`、`9db394fab194048e381262e1fefa741f92aac2d8c73fa083c3c95f88a10dd1ec`、`793cd4039c138571d57d7655e8a04b11e4ca921ce8156ec236e15cea91b56844`。本次未访问远端或启动 NPU 工作。

### 接受长度分布的条件还原（2026-10-30）

沿用紧前一节的历史代码绑定、完整计数、同 cohort、无重置与非原子抓取不造成错位等条件，另要求每次记录的接受数 `a` 为 0–5 的整数，所有位置计数均来自相同完整 observe 集合。所保存 `spec_decode/metrics.py:45–47` 对每次调用为 `i < a` 的位置增 1；保存的 scheduler 摘录 `:1983–1985` 以 `max(...,0)` 构造 `num_accepted`，历史实际调用仍须绑定。记总次数 N、位置 i 的接受数 C_i，则 `C_i = count(a ≥ i+1)`，所以精确接受长度频数为 `H_0=N−C_0`、`H_j=C_(j−1)−C_j`（1≤j≤4）、`H_5=C_4`。这仅依赖源码中的逐前缀计数语义及同组计数，不需要假定五个位置的接受事件相互独立。

在以上条件下，measure1 的接受长度 0、1、2、3、4、5 次数依次为 **0、104、240、453、266、688**；measure2 为 **3、145、309、249、254、786**。每轮各频数和分别为 1751、1746，按长度加权之和分别为 6447、6456，与已存接受总计一致；完整接受 5 个的比例约 39.29% / 45.02%，平均接受长度约 3.6819 / 3.6976。位置计数相减只还原所覆盖指标调用集合的分布；不能据此判断具体请求、首个拒绝后的补充 token、每次 kernel 或框架重组后的执行次数，更不能当作模型精度或潜在提速。

同窗两个 engine 的生成 token counter 增量各为 4096，总计 8192；草稿次数 N 加接受数分别为 **8198** 和 **8202**。因此，直接把这两个聚合窗口中的 `N + accepted` 当成恰等于生成 token counter 的恒等式，与保存数据不符。指标更新时点、对象集合和实际补充 token 关系未绑定，不能把 6/10 的差额认作实际多生成、丢 token、错误或某个具体调度开销；也不能从这个差值计算理论界。

由已保存的固定哈希快照与 `spec_position_accounting.py` 复算后，可用下列纯 CPU 算术交叉检查；该示例本身不验证历史源码、指标发布时点或采集完整性：

```python
for n, p, accepted in [(1751, [1751,1647,1407,954,688], 6447),
                       (1746, [1743,1598,1289,1040,786], 6456)]:
    edges = [n] + p + [0]
    h = [edges[i] - edges[i+1] for i in range(6)]
    assert min(h) >= 0 and sum(h) == n
    assert sum(i * count for i, count in enumerate(h)) == sum(p) == accepted
    print(h)
```
