# 169 vLLM-Ascend：融合扫描、低活动模块和条件理论界

**历史阶段报告**：较新发现与更正以 `CURRENT-FINDINGS.md` 为准。此前利用跨卡时间戳判断先后和等待的说法已撤回；时钟误差上界与同次 collective 身份均未证明。

更新：2026-10-28T09:01:25.803644+00:00。结论范围是已有证据的发现与定位；没有修改模型、框架或 kernel，没有实施融合或调参。本轮仅 CPU 离线分析既有 16 rank trace，GPT 主控，本机 ZCode DeepSeek 执行机械任务。

## 当前发现

已验证归属的 40 个 target 解码层中，第 0 层的 CSV 报告 AIC 周期加权 MAC 活动指标最低。去掉采样边缘后为 **3.6402%**；全采样口径为 3.0315%。该指标是报告字段构成的代理量，**不是整卡利用率、MFU 或相对物理峰值的算力效率**。

第 0 层内部 MC2 的 48 次调用平均 942.266 微秒，其他 39 层对应均值的中位数为 303.045 微秒。此前对跨卡时间线作出的先到和等待推断已撤回：跨设备时钟误差上界和同次 collective 身份未知，内部等待、可消除时间及端到端影响均未直接证明。不能据此认定底层 INT8 计算慢。

## 基线与数据范围

无 profiler 基线保持为 32K 输入、1K 输出、并发 8、实际前缀命中 89.990234375%；两轮聚合 **519.8440965 输出 token/s**。基线与有 profiler 的短样本分开，不用采样耗时替代基线。源版本、镜像、数值语义见 contract.json；原始导出 CSV 哈希见 evidence/TRACE-EXPORT-SHA256.json，原始大 trace 保留在 169 本任务原目录。

全量聚合覆盖 16 rank、481548 行 kernel CSV。通过 mHC 次序、每层 1 MC2/1 SparseFlashMla、压缩 KV 与 index source 角色等独立不变量，从捕获图恢复 4160 次 target decoder 调用；16 rank 全部通过。主排名使用每 rank/model 的 runs1、2、3，共 1920 次调用、每层 48 次，先固定采样规则再比较。全采样与内部样本的最低层身份一致。

| 已测模块 | MAC 报告代理指标 | 平均模块窗口/微秒 | 样本 |
|---|---:|---:|---:|
| 第0层 | 3.6402% | 1419.587 | 48 |
| 第16层 | 7.5786% | 743.945 | 48 |
| 第20层 | 7.5951% | 1027.922 | 48 |
| 第10层 | 7.6970% | 733.662 | 48 |
| 第9层 | 7.8349% | 774.998 | 48 |

分子为 `sum(aic_total_cycles*aic_mac_ratio)`，分母为 `sum(aic_total_cycles)`，仅纳入已归属 kernel；不是百分比简单平均。PMU 字段审计未发现正周期但比例缺失/非法的行。缺周期的任务没有被当成具有已知 PMU 的零活动任务。完整分子/分母、各 rank 结果和全部 40 层排序在 module-analysis.json、captured-layer-attribution/ 中。

覆盖缺口：eager target 完整阶段语义及 720 个未见 attention 窗口、3 个 eager DSpark block 的独立身份与完整活动、Engram host-UVA NPU lookup 的完整归属及投影门控/通信/CPU 工作、root/head/metadata 未形成同口径全框架排名。这里只能称“已测 40 个 target decoder 中最低”，不能称整个推理框架最低。模块窗口的跨 rank 平均值不是全局关键路径时间。

## 融合扫描：做了什么，停在哪里

冻结 G0，G1 补齐延后消费的 mHC 系数跨层依赖。12 个源码模板、221 个模板节点，扫描两两、三个、四个相连节点和共享输入分支，枚举原版 928 个结构项；原覆盖表对跨模块扫描的标注不准确，现已用 G1 复合图另做有限跨模块结构枚举；第二轮同图同规则新增 0。928 是结构项数量，不是已证明可实现或能提速的融合数量。

对 31 个源码边界检查项复核：原版登记 27 个有条件候选，3 个已实现的融合（mHC pre、RMSNormCast、SparseFlashMla），1 个 metadata 路径未知。复审发现部分图缺输出和状态输入，且 DSpark 的部分 runner 绑定未经证明；现在依照修正版记录将相应项标记为 UNKNOWN。候选图在 candidates/，原始检查在 evidence/FUSION-SWEEP.md，覆盖/停止在 fusion-scan.json，重叠关系在 records.json；旧图保留供追溯，不构成消费者完整性或语义等价证明。已实现融合不重复算收益；相互重叠的候选不能把节省量相加。

具体例子 F05：Q 归一化输出 qr 在 8 个 index source 层还供给 indexer，融合 norm+quant 不能直接删除 qr 写出。其余 32 层没有该数值消费者时，qr 的一写一读最多为 5120*M 字节的有条件移除机会。BF16 舍入、scale 归约、两条分支及原有多流重叠必须保留。详见 evidence/QR-consumers-and-fusion-review.md。

当前融合阶段标记 `PAUSED_WITH_GAPS`，指完整 primitive/alias/proposer/metadata 图和语义证明仍有缺口；**不表示整个任务暂停，也不表示已经证明再无可融合项**。不能把有限结构复扫包装成全模型融合穷尽。

## 理想上界与实际对照

先允许理想融合和重叠，再对已证明必要的窗口工作及对应硬容量建界。一般情形须计入窗口短时突发或在途信用：例如 `T≥max((F8−β8)_+/P8, (D_HBM−β_HBM)_+/B_HBM, (D_link−β_link)_+/B_link, L_causal,...)`；仅当下界 `L>0` 且输出 Q 的窗口相同时有 `U≤Q/L`。只有额外证明各信用均为零时才能简化为原来的工作量除以容量。共享资源只在不重复计数时累计工作；当前同步、Python 转换和所选 tile 的开销不自动进入不可避免下界。详见 evidence/IDEAL-MODEL.md。

官方标称公式与 profiler 设备元数据给出条件分母 353.8944 TMAC/s/逻辑设备；9382 适用性、1800 MHz 频率上限、标称公式构成各短窗容量上限及相关初始信用为零均仍是前提。以下独立 Wq_b 是所有 M 行输出需保留、所计 MAC 全由声明的 INT8 Cube 容量池执行的标准 dense 算法域，M 不等于返回 token 数；观测样本属于该窄域仍有待核实。

补充审计：Q 投影的现有 1920 条汇总只筛选部分输入形状、类型及层窗口；并未逐条验证输出形状、NZ 逻辑布局、tensor 身份或实际二进制，也未在查询脚本内对每份 CSV 逐一验 hash。该汇总仅是条件线索，不能作为执行对象及工作量已绑定的证据。见 `agents/bounds/int8_ceiling_certificate.md`。

正式实测/界比值均为 UNKNOWN：样本属于下面的受限实现域尚未核实。下表仅展示假设样本满足这些条件时的算术敏感性，不是已验证比较或可达提速。

| 对象 | 实测中位数/微秒 | 条件计算下界/微秒 | 假设同域的 T/L 算术 |
|---|---:|---:|---:|
| Wq_b_rank0_M3 | 41.919 | 0.355556 | 117.897 |
| Wq_b_rank0_M416 | 156.557 | 49.303704 | 3.175 |
| Wq_b_rank0_M1 | 40.399 | 0.118519 | 340.867 |
| Wq_b_rank0_M2 | 40.439 | 0.237037 | 170.602 |
| Wq_b_rank0_M817 | 242.215 | 96.82963 | 2.501 |
| Wq_b_rank0_M419 | 151.817 | 49.659259 | 3.057 |

对选定 target INT8 工作，用 16 设备容量得到的 **116294.745 输出 token/s** 只是极松的窄域条件计算上界：除容量前提外，要求每个未命中/生成位置对每个请求独立计算、不发生额外跨请求/位置复用，且所有计数 MAC 都由该 INT8 Cube 池执行；观测基线是否完全属于该域尚未证实。这个数忽略 attention、BF16/FP32、draft/reject、访存、通信和调度。不能把它当“本机实际极限预测”，不能声称基线存在 223 倍可提速空间。

本次已保存 7 个独立数值对象的精确输入与计算器原始输出；40 层模块和原版候选也有对照记录，缺失项保留符号/null。允许跨请求共享或用其他计算资源完成等价计算的宽域没有得到数值上界。HBM/L2/互连容量上限、必要字节和有效路由计数未证，**本机无条件、不可超越的数值吞吐上界仍未知**。没有用最快实测速率代替硬上限。

## 第 0 层为什么低：支持到哪里

MC2 源码已融合 dispatch、GMM1、SwiGLU/量化、GMM2、combine，并有逐 expert/跨核心 flag 与跨 rank readiness 轮询；它不是一个纯 INT8 GEMM。源码见 evidence/MC2-known-dependencies.md。

**更正：**此前根据跨卡绝对时间戳和序号推断对应调用、谁先到及谁等待，缺少跨设备时钟误差上界和同次 collective 标识，已全部撤回；相关旧数字保留于 `evidence/pre-independent-review/REPORT.md`，不得作为当前结论。同卡的首层起点到 MC2 起点局部持续时间，在各组各轮的中位数约 378–395 微秒；仅凭它不能确定层外准备、输入或通信原因。图 `MODEL_EXECUTE` / `MODEL_WAIT_COMPLETE` 的时间包围和连接键也不足以证明请求身份或纯等待。详见 `agents/diagnosis/LAYER0-MC2-DP-RECOVERY.md`。

小 M/专家负载不均/访存与当前分块策略可同时贡献；未拿到专家有效 token 分布、内部发布时间与实际容量，不能精确分摊。去掉边缘后异常仍在，削弱“纯采样首尾效应”的解释，但不排除 profiler 对调度的扰动。基线端到端可消除时间为 UNKNOWN。

## 复核与恢复

先读 STATE.json、contract.json、profile-summary.json 和本报告；再按索引读取最多 200 行/32 KiB 的局部证据。完整聚合保留，不把整个大 trace 塞入模型。tools/ 保存 CPU 分析脚本，bounds/ 与 calculations/ 保存精确数值复算；calculator 仅检查算术，前提由主控审阅。ZCode 模型验证和执行失败/成功凭据在 zcode/ 与 evidence/ZCODE-EXECUTION-INDEX.json。

本轮分析记录同步到 `/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028`。不操作 172，不修改目标框架/模型，不发送新推理负载。

此前目标工具曾持续返回无原因的 blocked 与旧时间戳，期间离线工作继续执行。最新复查已返回 active，目标模式状态恢复；此前状态异常的原因仍未知。详见 evidence/GOAL-STATE-ANOMALY.json。

## 本轮复审补充

原 MHC 模板遗漏了无 Engram 修改时 `ffn_post → next_pre` 的条件残差边；原 F23 未列出 MoE 结果给后续 hc_post，F29 未列出持久 KV 状态，index source 模板漏了历史索引缓存/元数据。DSpark `main_norm → block0` 直接边没有源码证据，已从修订模板移除并保留 runner 绑定未知。修正模板为 graphs/source-templates-G2.json；F03/F23/F29 的修正版边界图保存在 candidates/，仍未证明所有消费者或语义等价。

昇腾官方 [MindStudio Insight 文档](https://github.com/Ascend/msinsight/blob/efb28234772093b0cabc3c538e71375c957540b2/docs/zh/user_guide/system_tuning.md) 将 `aic_mac_ratio` 描述为 MAC 耗时占总耗时的比例，将 `aic_total_cycles` 描述为 AI Core 执行周期；原文件已按 commit 和 Git blob 哈希保存。本文按报告字段加权得到的是代理指标，采样窗口、内核归属、每核聚合及未覆盖模块的限制仍然存在。


## 后续只读追查：DSpark 链路和前向间隔

冻结基线容器与 CPU stage 使用相同镜像；5 个关键源码文件在两容器和本地快照中的 SHA-256 一致（见 `evidence/dspark-source-provenance.json`）。对 DSpark 的 K>0、上下文非空且 slot 有效的分支，源码显示 proposer 先将 target 辅助状态经 `combine_hidden_states` 投影归一化，放入上下文缓冲区，再调用 `precompute_and_store_context_kv` 写入各草稿层缓存，随后调用草稿模型。修正版 `graphs/source-templates-G3.json` 将它表示为通过上下文缓冲和持久 KV 状态的依赖，不能简化为投影输出直接作为 block0 激活；也不因此证明跨所有层的整阶段等待不可避免。F27/F28 重新记为条件结构候选，F30 的验证/接收绑定仍未知。

在 3 组内部样本中，按各 rank 自己的时钟计算，从上一 target 第 39 层 MC2 结束到下一 target 第 0 层开始，DP0 比 DP1 分别多约 1.764、0.540 和 1.311 毫秒。前后两端使用同一 rank 的时间差，因此固定时钟偏移不改变这个差值；跨 DP 的同轮配对仍依赖 offset0 假设。其中第二组的上一轮是 eager 路径，第 39 层身份依赖调用序号推断，证据较弱。区间内已导出 kernel 的并集与未覆盖时间不能分别等同于本任务设备忙碌和设备空闲；不同 stream 的通信、计算时间也不能简单相加推断原因。

最初的间隔扫描给可能跨窗的 kernel 设了 50ms 回看范围，后续用所有候选区间的结束时间前缀最大值修正；5 组统计未变化。v2 文件一度误写入本地 v1 目录，现已从未修改的远端副本恢复 v1 并保存两个版本及审计记录。原始 trace 未受影响。


## 后续只读追查：eager 草稿层的低活动样本

此前 40 层排名只覆盖捕获的 target decoder。根据 8 个本地专家、top-3 路由、四个 mHC 标记和一条 SparseFlashMla 组成的约束，从两段前向间隔中严格归属到 162 次 eager 草稿层调用，覆盖 15 个 rank；另 126 次不满足 SparseFlashMla 约束，其中一个 rank 全部未被严格归属。三个草稿 block 暂时只能合并，不知道每次属于哪一个。

用**主流上的同一 MAC 周期代理指标**，这 162 个草稿调用合并约 2.0151%，对应的相同 15 个 rank 的 target 第 0 层约 3.7515%。这提示草稿部分也可能是低活动区域，但调用数、输入形状和模块组成不相同，严格样本存在明显选择偏差；不能称它为全框架最低模块，也不能把它与此前包含更多流的 3.6402% 直接混为同一指标。

在这 162 次严格样本里，MC2 的 MAC 代理值约 0.5755%；MC2 平均约 763.085 微秒、中位数约 181.186 微秒，占这些模块窗口耗时和的约 48.47%。这是采样和，不是端到端关键路径；无法据此区分通信等待、小矩阵、专家负载不均、访存及 profiler 扰动。

源码允许本地 query 数为零时跳过 SparseFlashMla，同时全局 KV 准备仍可能执行。抽样窗口与此一致，因此单独做了条件敏感性统计：{'EMPTY_LOCAL_QUERY_SOURCE_CONSISTENT_NOT_DIRECTLY_OBSERVED': 126, 'ONE_SPARSE_FLASH_MLA': 162}，共 288/288 次，16 rank 草稿合并指标约 1.1739%。其中空 query 只是通过窗口形态推断；实际每次 query 数、跨流事件的因果归属未直接记录。剩余缺口和完整分子分母见 `evidence/draft-module-diagnosis.json`。实测不能覆盖的模型模块仍保留 UNKNOWN。


## 后续只读追查：图提交和目标层到达

对 104 次捕获的第 0 层调用，以前 10ms 为候选范围，99 次有唯一 `aclmdlRIExecuteAsync` 事件；其连接 ID 在导出数据库中均关联同一个 modelId 的一条 `MODEL_EXECUTE` 和一条 `MODEL_WAIT_COMPLETE`，后者唯一覆盖该层已验证的窗口。5 次未匹配均在首个捕获轮次，未强行归属。连接支持图提交与模型任务的对应；它不是请求身份或时钟同步的形式证明，也不能把 `MODEL_WAIT_COMPLETE` 直接解释成纯计算或纯等待。

在三个两组 DP 都有 16 rank 完整连接的内部样本中，候选提交与层入口的 DP0 减 DP1 时间差如下，单位为微秒：

| 条件配对序号 | 提交 API 差 | MODEL_EXECUTE 差 | 第 0 层入口差 |
|---|---:|---:|---:|
| 1 | 1348.473 | 1345.423 | 1761.344 |
| 6 | 545.720 | 552.796 | 541.276 |
| 7 | 1239.724 | 1240.816 | 1305.782 |

这些数需要 offset0 跨 DP 配对和可接受的跨时钟误差作为前提。它们支持“时间差在进入第 0 层之前就已出现”的观察，并提示图提交时序是需要追查的一段；第一个样本中入口差比提交差还大约 0.413ms。因缺少精确请求身份、时钟误差界及上游数据依赖，不能断言调度器、Engram、通信或前序草稿计算中的某一项造成了差异，更不能把差值当作可消除的端到端时间。


## 源码复核：Engram 资源与 DSpark 逐步依赖（G4）

Engram 哈希调用 NPU Triton 内核；主机卸载的 UVA 分支也由 NPU 内核读取映射的主机表，CPU 参考路径是另一分支。此前将 hash 和 lookup 统一标成 CPU 不正确。G4 补上位置、请求边界、lookback、掩码、条件 slot cache、表、gate 输入和输出的边界；缓存 alias 和全部消费者仍未证明。

基线 DSpark 配置 5 个投机 token。源码逐步以 token 生成 Markov embedding/bias，给当步 logits 加 bias 后采样，结果成为下一步输入；G3 的 logits 直接连接 accept 是不成立的简化。冻结启动脚本未设置 dynamic_spec，默认 method=None，confidence 调度分支在该配置下关闭，除非有其他覆盖；scheduler 内部绑定尚未验证。G4 独立保留五步和条件分支，未声称整个运行图或融合等价已完备。结构候选仅是有限图枚举，不等于可实现或有收益。详细证据见 evidence/engram-dspark-g4-review.json。


## 时钟元数据与条件输出约束

对 16 个 rank 的采集锚点和当前镜像 profiler 源码做了可复现核对：device wall/raw 截距观察跨度 9.520 微秒，host 为 4.760 微秒，host counter 截距为 0.100 微秒。这些是不同时间采集的元数据跨度，不能视为跨设备事件同步误差上界；原先跨 DP 配对及时间差仍是有条件观察。源码快照、哈希、逐 rank 数据与复现脚本见 `evidence/clock-source-and-anchor-audit.json` 和 `tools/capture_clock_metadata.py`。

两轮固定输出各 8192 token，其文本 UTF-8 载荷分别为 32628 / 33024 字节。若指定接口必须在计时内交付这些确定文本、窗口各前缀的交付量满足 `Q(t) ≤ β + B t` 且必要字节数为 D，则有 `T ≥ max(0,D−β)/B`；仅在 `β=0` 另有证明时，才能简化为 `T ≥ D/B` 和 `U ≤ 8192 B/D`。β 应覆盖计时开始时的在途或突发信用，B 是该接口的硬容量，不自动等于网卡带宽。压缩、复用、β 和 B 均缺证据，因此仅是条件符号界，不能推广为模型无条件数字上界。`evidence/fixed-output-conditional-bound.json` 为保留的旧版证据，保存的无 β 公式仅限零信用假设，不能当作一般界；更新推导见 `agents/bounds/baseline-client-count-provisional.md`。

更宽实现域若能证明某统一工作量 F 必须完成，且所有资源池对同一工作单位分别具有硬速率上限 C_i，则放松资源约束得到 `T >= F/sum(C_i)`；计算、访存、通信如可重叠，已有有效下界取最大值，不可机械相加。当前 F、不同资源可比速率及共享容量约束缺乏证明，不能把已有指定稠密 INT8 MAC 数代入当作普遍必要工作量。


## 16 rank 的同形状 INT8 矩阵算子

用已验证的 target 层内部窗口、相同 model ID 和 stream 关联，在 16 个 rank 找到 1920 个 M=3、INT8×INT8 输入、BF16 输出的同形状矩阵算子样本。整体中位耗时 42.000 微秒，各 rank 的中位值范围约 41.090–43.259 微秒；分组原始聚合见 `q-projection-allranks/`。严格 draft 样本和不能归属的同形状调用分开保留。

沿用先前明确列出的标称计算容量和独立 dense、全输出、Cube-only 条件，计算项的理想时间约 0.356 微秒，若样本确属该条件模型所限定的域，中位时间与条件下界的算术比约 118.1；同域尚未证实，正式比值为 UNKNOWN。这不是可达到的加速比，也不是框架整体上界。该组 AIC MAC 和 MTE2 计数器代理约为 10.6% 和 72.9%；不同流水可以重叠，不能相加或单凭 MTE2 值判定 HBM 带宽已饱和。见 `evidence/q-projection-16rank-diagnosis.json`。


## 融合图再次复核（G5）

独立源码复核后，G5 增加 Engram 的 hash gather、按层切片、dead mask 数据依赖，并把数据边与 ready 事件的顺序约束分开。DSpark 增加 seed 与 query 输入的共同来源及各自生成过程、条件 SP gather、采样索引选择、概率采样路径的状态副作用和概率输出。相关 DP/SP 通信是否执行取决于实际分支，不因图里出现就认定已运行。G5 仍是条件源代码图，不代表语义穷尽或融合获益；细节及历史版本见 `evidence/engram-dspark-g5-review.json`。


## eager target 与 Engram 的补充归因

已有 trace 中提取了 16 个 rank、2560 个 eager target MoE 窗口：1840 个有同 stream 的 attention，并与压缩输入及 indexer 层角色一致；720 个没有观察到 attention，具体分支未知。对每个 rank 穷举 40 个层序号偏移，仅偏移 0 与可检查的角色一致。这加强了层序号推测，但不等同于 profiler 显式层标签，也不能由物理 M 推断 prefill。

例如物理 M=834 的 40 个角色一致层组，每组 16 个调用；按相同 stream 的 AIC 周期加权 MAC 计数代理，推测第 0 层约 16.61%，在该组最低。它仅是该形状和采样范围的排名，不是整个框架的最低利用率；窗口平均约 11.19 毫秒也不能直接当成端到端暴露时间。完整数据与 720 个未归属窗口保存在 `eager-target-attribution-v2/`。

另外通过精确任务名及 TASK→CANN launch 关联，识别 16 个 rank 共 640 个 Engram hash/lookup 内核：请求索引、缓存写入、哈希各 128 次，host-UVA 查询 256 次。它们的采样中位耗时分别约 6.059、13.069、127.127、12.160 微秒；采样混合形状及阶段，且可能互相重叠，不能将中位数相加或当作完整 Engram 模块耗时。TP gather、事件等待和用户可见的关键路径仍未归属。详见 `engram-kernel-profile/summary.json`。


## 第 0 层：Cube 和 Vector 流水的进一步对照

已归属 target 的 48 个内部采样窗口中，第 0 层 AIC MAC 计数代理为 3.64%，AIV Vector 代理为 0.45%，AIV Scalar 代理为 10.44%。不同流水的分母、覆盖和可重叠关系不同，不能相加或当成整卡利用率。窗口内已导出任务的时间并集覆盖约 98.76%，但 kernel 区间内可能等待，因此不表示 NPU 一直有效工作。

同一组样本中，融合的 DispatchFFNCombine 占第 0 层已归属 AIC 周期分母约 81.73%，其 MAC 代理约 2.49%；其他已归属 kernel 的 MAC 代理约 8.78%。这说明该代理的低值受到 MC2 样本显著影响，不能据此推断跨 DP 到达先后或等待；仍无法仅据此分离通信等待、工作量不均或其他内部因素，也不是端到端耗时占比。原始分子、分母和其它层对照见 `evidence/captured-layer0-engine-diagnosis.json`。


## 频率证据修正

前述 1800 MHz 是 profiler 的设备元数据字段；尚未证明它是在基线运行时逐时采样的实际频率，也未证明它是 9382 的物理最高频率。华为文档说明其 MFU 工具使用设备信息和数据类型估算理论峰值；另一份华为文档区分标称频率与当前频率，但不能据此直接确定本设备 profiler 字段的对应关系。有关来源和适用范围见 `evidence/frequency-field-review.json`。现有数值仍是有条件模型，不是不可超越的已证物理上界。


## 未见 attention 的窗口和 Engram 通信候选

720 个未见 attention 的 eager target 窗口中，按同一 model/stream 的第一 attention 子窗口扫描，400 个出现 `ZerosLike` 且形状末尾为 `[64,512]`，320 个没有该形状；两组均未看到指定输出形状的 QuantBatchMatmulV3。一个补零样本可用连接 ID 精确追到 `vllm::dsa_v41_forward` 的 CPU 调用，但调用的输入形状未记录。现象与源码中的空 query 或补齐分支相容，不能据此确定任一窗口的 query 长度或将 720 个窗口归为同一原因。见 `evidence/eager-no-attention-patterns.json` 和 `evidence/eager-empty-query-example.json`。

一个 Engram host-UVA lookup 样本与随后同线程 all-gather 的启动链均可各自精确追踪；通信元素数与 lookup 输出形状一致，设备时间相隔 0.960 微秒。它是值得继续核对的候选，但没有 tensor ID 和通信组与 TP 组的映射，不能证明二者直接依赖，也不能把时间差当作等待或可掩盖时长。见 `evidence/engram-gather-candidate-example.json`。


## 最终交付口径和复现入口

融合图的逐节点、逐边结构覆盖台账在 `coverage/`。G1 的跨请求反馈边 `verify_accept → inputs` 明确排除在单轮 2–4 节点枚举之外；F31 有条件 metadata 局部图已保存，消费者和任务等待仍未知。`python3 tools/verify_fusion_enumeration_readonly.py` 可只读重算 G1 跨模块及 G2/G5 已声明的有限集合；它只验证结构枚举，不验证语义等价或全图穷尽。

现有受限域的七项正式 T/L 与 L/T 保留 UNKNOWN；历史数值另存 `evidence/conditional-ratio-sensitivity.json`，仅表示假设输入及实现确属该域时的算术。16 个实际使用的 profiler DB、时间锚点及 EP0 trace 的原路径和哈希在 `evidence/used-raw-source-sha256.json`；kernel CSV 哈希在 `evidence/TRACE-EXPORT-SHA256.json`。未被分析引用的其他导出文件并未全部计算哈希。
