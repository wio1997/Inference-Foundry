# F27：target 辅助状态到 DSpark main projection 的条件流水

## 范围及可恢复结论

针对冻结源码进行静态分析；**不是已实现融合，也不是已证明运行时命中、确切收益或性能上界**。与 F26 target 尾部 residual collapse 不同，这里讨论采集多层 target 中间状态、草稿侧合并投影与归一化。样本中真实 tensor、graph 调用身份和各事件的依赖尚不明确。

## 数据流、消费者和已有操作

- `sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py:1555–1565,1603–1608`：若 `layer.layer_idx+1` 在 `aux_hidden_state_layers`，在该层及其可能的 Engram **执行之前**，对当时的 hidden_states 沿第 1 维求 mean；sequence parallel 时再 all-gather 并切到 `full_num_tokens`，加入列表。随后 target 正常推进；最终该列表作为返回值之一输出。相同 hidden_states 还为 target 后续层和其他路径使用，不能因为草稿侧融合删除原 target 消费。runner 对列表的组装和其他可能消费者不在所查源码中，未知；也未在历史运行中证明每个列表项地址、拼接和层号。
- `evidence/model-config.json` 给出 `hc_mult=4`、`hidden_size=5120`、`dspark_target_layer_ids=[37,38,39]`。**层号更正**：此前“直接安装 [37,38,39] 时进入零基层 36、37、38”仅是人为假设，不能把配置值直接当已安装值。已保存 runner 源码的 DSpark fallback 会把 draft config 中的 ID 各加 1，再传给 model；在 superclass 没有优先返回其他列表、draft config 确为 [37,38,39] 且该初始化路径实际执行时，安装的是 [38,39,40]，分别在进入零基层 37、38、39 前采样。历史 draft 实例、覆盖配置和绑定仍未知，见下节。
- `sources/.../models/deepseek_v41/dspark.py:150–155,177–203,281–282`：按配置设置 `main_proj`，逻辑输入宽度 `3×5120=15360`、输出宽度 5120、无 bias，`ColumnParallelLinear` 指定 `gather_output=True`，其结果交 `main_norm`。这是源码中两个接口的组合调用；未发现该处宣称已有 main_proj+norm 的单设备内核，实际编译结果未知。量化配置只在 packed-cache ops 可用且 checkpoint quant_method 为 fp8 时通过条件传入；这条静态条件不能取代真实实例权重 dtype、布局或 kernel 路径。
- `sources/.../spec_decode/llm_base_proposer.py:1063–1078` 在方法为 dspark 等时先调用 combine，再调用 `set_inputs_first_pass`。`dspark_proposer.py:258–259` 把按 context 数裁切的结果写入持久 buffer；`dflash_proposer.py:301–319` 后续调用 precompute 并读取对应 context hidden 和 position；`dspark.py:284–309` 继续各层 KV 投影、归一化、位置变换及 cache 写入。缓存生命周期、有效 token 选择、后续轮更新次序都必须保留，不能仅保留一个局部中间结果。

## 具体可重组候选及不可省略条件

1. **提前部分投影**：若 runner 证实草稿投影输入按上述三个 target 状态在 feature 维拼接，则实数代数上对 `W=[W0 W1 W2]` 有 `W[u0;u1;u2]=W0u0+W1u1+W2u2`。每个 `ui` 产生并就绪后，具备相应权重时可以探索按行及输出列分块计算部分投影，安排与后续 target 层重叠。仍需等全部三个对应位置的部分结果完成，才能得到完整投影；target 最后两层、通信及投影间的资源争用可抵消重叠，不因代数可分就断言变快。权重分片、token 排序、跨 rank 数据可用时刻、事件和多个消费者的读写生命周期都必须核实。
2. **严守浮点与均值**：三次短 K 矩阵乘再相加，可能改变原长 K GEMM 的归约顺序及中间 dtype/舍入；原 `mean` 的输出和 dtype 也不可无条件省去。实数恒等不保证位级等价，例如 FP32 顺序的 `(2^80 + -2^80) + 1 = 1`，而 `2^80 + (-2^80 + 1) = 0`。这只说明一般重排存在反例，不证明该具体 GEMM 的实际归约树或精度差。必须确认原 kernel 累加/输出类型与精度验收要求；若必须位级一致，则需要证明具体实现的归约和舍入等价。
3. **投影→norm 分块**：`gather_output=True` 表示该投影接口要求输出聚集；如模型并行确将输出 feature 分片，完整 RMSNorm 的每个 token 分母依赖全部 feature 的平方和。可探索本地部分平方和与跨 rank 汇总、再完成对应输出，但应保留全宽分母、epsilon、权重、原精度与舍入，并保证下游获得其要求的完整张量。**不能把每个 rank 的局部 RMS 当全局 RMS**：实数例子 `[1,3]`、epsilon 为 0、单位权重时，全宽结果 `[1/√5,3/√5]`，分别只对单元素计算则为 `[1,1]`。示例用于说明依赖，非运行时输出。通信方式、是否比原 gather 更快、归约精度与确切接口语义尚未证明。
4. **mean→部分投影融合**也可作为候选：按需形成某行某块 mean 后送投影，避免不必要的中间物化须保留原 mean 的舍入与对外可观察的列表；在实际存在多个消费者或返回该 tensor 时，不能简单删除它。源码层表达式独立不等于已发生独立 HBM 读写；一个融合操作具体能省多少字节和启动时间未知。

按逻辑尺寸，不含 padding、仅计算一行，完整矩阵有 5120×15360 = 78,643,200 个乘加；若输出恰好均分 8 个 rank，则每 rank 对应 9,830,400 个乘加。只有在权重实际为 BF16、按该尺寸无额外填充时总逻辑权重 150 MiB；这不是实际显存读量或带宽下界，也不能用于声称利用率或优化收益。

## 后续证据

已准备 `query_f27_dspark_gap.py` 和只读 wrapper `run_f27_dspark_gap.py`：固定查询 EP8、layer 39 run 2 标记之后至下一 target layer 0 起点的已导出 kernel CSV，扫描形状包含 15360 的记录及选定近邻；校验 CSV 冻结 SHA，限制输出，不执行 NPU、不读取模型权重数据。匹配形状和时间不能证明同一模型调用、请求或数据地址；没有匹配也不排除编译融合、形状缺失及其他 rank。stage172 已由本机 ZCode `deepseek-flash` 执行，退出码 0、`model_verified=true`；输出 `f27-dspark-gap-evidence.json` SHA-256 `d68ee49164b4db836705c08f026e939772974e72e54fc4b59eafd5458aa9f19b`。EP8 选定窗口内，同一 stream/model 有 BF16 `ConcatD` `[24,5120]` 三份→`[24,15360]`（相对末层标记约 786.704–795.184 μs）及 `MatMulV2` `[24,15360]×[640,15360]`→`[24,640]`（约 795.204–824.244 μs）。形状与 15360 输入、TP8 输出分片的 DSpark main projection 相容；但无 tensor 地址、请求、运行实例或调用栈绑定，不证明这是该投影、也不证明 ConcatD 的输出被该 MatMul 消费。另有两条同一区间的 allGather 样式记录，可能是同一通信的不同导出视图，不可重复计数；通信内容和事件依赖未知。相邻和未匹配项不证明收益、同步关系或任何设备没有活动。只读已有 CSV，无设备操作。


## Runner 数据链与层号条件校正（既存源码复核）

`agents/bounds/metadata-path-snapshot/ascend_runner.py:751–765,787–802,4427–4444`：PP 最末 rank 且 `use_dspark()` 时设置 `use_aux_hidden_state_outputs=True`；PP=1 时按该值决定是否配置 aux。子类 `_get_eagle3_aux_layers_from_config` 首先调用父类：有非空值便直接返回；没有时，若 draft `hf_config`（或其 `text_config`）的 `dspark_target_layer_ids` 非空，才返回每项加 1 的 tuple。父类快照 `vllm_runner.py:5518–5551` 还可优先从 `eagle_aux_hidden_state_layer_ids`、dflash/eagle 配置返回列表。故若 draft 的三个值确为 `[37,38,39]` 且父类无优先配置，则传给 `set_aux_hidden_state_layers` 的是 `[38,39,40]`；`model.py:1560` 使用 `layer.layer_idx + 1` 比较，并在运行该层之前取 mean，所以对应零基层 **37、38、39** 的入口。只是静态条件推导：现有 `evidence/model-config.json` 是冻结模型配置，不单独证明历史 draft 配置对象、子类/父类实际版本与覆写、以及安装结果。

`ascend_runner.py:2063–2091`：在 `use_aux_hidden_state_outputs` 下，metadata 缺失时对列表各项用同一 `[:num_scheduled_tokens]` 截取并 `torch.cat(...,dim=-1)`；有 metadata 时先按实际分支准备 `token_indices`，对列表各项按相同索引取行再按末维拼接；`:2097–2110` 将此 tensor 交给 drafter `_propose`。冻结 `llm_base_proposer.py:1049–1074` 对非零 K 且 method=dspark 时调用 `combine_hidden_states`，而 `dspark.py:280–282` 的该方法是 `main_norm(main_proj(aux_hidden_states))`。这闭合了**该源码分支下**“逐层 aux → 同一索引选行 → concat → main projection → norm”的条件调用关系；K=0 会提前返回、其他分支与运行时实际数据仍需核实。不能将 stage172 形状相符的任务升级为已绑定该张量或具体调用，不能假设每一行均为有效 token。

来源哈希：`ascend_runner.py` `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`、父类 `vllm_runner.py` `87c29d08c0bbf66993d8b984811e7325e35e6436242a773feb55ec88f16b2c51`，见 `agents/bounds/stage-095-metadata-output-review.md` 的双容器核验；`model.py` `101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91`。没有运行设备。


### setter 及 PP cache 的静态核查（stage173–174）

两次只读查询由本机 ZCode `deepseek-flash` 执行，退出码均为 0、`model_verified=true`。staging 容器 `vllm/model_executor/models/interfaces.py` SHA-256 `5cc70b37e1dc715fe6ea2710678e8a4446ec223d154a7c31a53f0a05242c64b4`：`EagleModelMixin._set_aux_hidden_state_layers` 将传入层号去重排序后存为 `aux_hidden_state_layers`，再清空缓存并调用 `_cache_aux_pp_layout`；后者在未初始化或 PP 小于 2 时返回，在多 PP 时只更新其他缓存属性，所见方法不改写层号。对三个互异值 `[38,39,40]`，上述已核查方法在正常执行条件下保留这三个值。setter 提取 `f27-aux-setter-evidence.json` SHA-256 `3dd283959f281dfc3c4c25f43f0546ab610f9950d12d9c90792bff0b29af71f4`；layout 提取 `f27-aux-layout-evidence.json` SHA-256 `f0f4401bc269a91adea38f22d005c1c5b28d2d8317317e01a601976f3d31c4ab`。这只读取当前 staging 文件，尚未证明历史服务使用同一 setter 文件或历史配置没有优先覆盖，故整个层号结论仍是条件推导。无模型 import、NPU 操作或服务修改。

## 最后三个目标层入口的既有导出记录（stage230）

在固定的 16 rank × 3 轮中，按各卡自身时间轴检查 36→37、37→38、38→39 三个目标层间窗口，以此前选定的紧邻 35→36 窗口作描述性对照。窗口仍是源码角色规则推断的边界，并非框架的完整模块标记；对照并未控制层语义或请求，不能用它证明因果关系。每轮下表数值分别为 16 张卡的中位数，单位微秒：

| 边界 | 轮次 | 窗口 | 主流 `ReduceMean` | 另一数字流 `hcom_allGather_` |
|---|---:|---:|---:|---:|
| 36→37 | 1 / 2 / 3 | 34.930 / 34.920 / 35.699 | 15.710 / 15.080 / 15.680 | 16.070 / 14.900 / 15.290 |
| 37→38 | 1 / 2 / 3 | 35.479 / 33.380 / 35.209 | 14.980 / 14.980 / 15.000 | 15.510 / 13.440 / 15.420 |
| 38→39 | 1 / 2 / 3 | 35.300 / 35.240 / 35.889 | 14.910 / 14.950 / 14.750 | 15.630 / 15.640 / 16.589 |

144 个所选目标窗口中，每窗恰有一条 model 49 的主流 `ReduceMean`，导出输入 BF16 `[3,4,5120]`、axis INT64 `1`，输出 BF16 `[3,5120]`；每窗另有一条 model 49 的其他数字流 `hcom_allGather_` 和一条 stream `N/A` 的同名行。后两行在每窗起止时间完全相同，可能是同次通信的不同导出视图，**不可相加**。144 窗中 `ReduceMean` 的导出结束均早于另一个数字流 `hcom_allGather_` 的导出开始，间隔范围 **1.22–2.02 μs**，均完整位于窗口内；这种先后关系不证明任务之间存在数据依赖。所有导出行的并集覆盖每窗约 **84.08%–89.40%**，未覆盖部分不是设备空闲证据。对照的 48 窗没有与之相交的 CSV 正时长导出行，窗口中位数依三轮分别为 1.190、1.260、1.270 μs；这并不能排除未导出活动。

所见均值的 shape、三个目标层入口位置、随后 allGather 名称，与前文源码的条件 aux 取均值和 sequence parallel gather 相容；当前没有 tensor 身份、历史实例配置、运行分支或 allGather 数据/请求身份，不能把它们确定为 DSpark 辅助状态，也不能确认通信成本、资源争用、关键路径或可兑现收益。`hcom_` 导出行可能是汇总视图，时长不能直接当纯链路传输时间。样本不是跨 rank 同一请求的证明，也不代表端到端耗时。

复核输入 `agents/diagnosis/dspark-aux-boundaries.json` SHA-256 `13b5584a6b2e93ddf7a7b19da37364b1212e54a875c7a1720ceb80831af0655e`；固定只读查询 `agents/diagnosis/query_dspark_aux_boundaries.py` SHA-256 `f7edc54eef2b52b4b34c125a366af349c8203217ccc83b400eda10efe57c739c`。查询校验既有 16 份 CSV 哈希、窗口清单和同卡层/流边界；聚合时对每窗独立核对各类型数量、完整包含、流归属和另两行时间一致。由已验证成功的 ZCode stage230 获取既有导出数据，无新 NPU 工作或目标代码修改。

### 历史启动日志与诊断实例的身份边界

本机旧基准保存的 `benchmark-20261028/raw-results/service-log-tail.txt`（SHA-256 `64bfa3d8523ad86e1c466fcca166cdcfe1dc4a3676573fc51b65279a69aa8aba`，与该目录 `RESULT-SHA256.json` 一致）第 41 行有启动警告，称 sequence parallel 已启用并移除 graph batch sizes `[1,2,4]`；第 100 行 EngineCore_DP0 的启动配置记录 `method=dspark`、TP8、PP1、DP2。`guard-measure1.json` 记录基准容器 ID 为 `b00f4d2018e3d495d876b6e396ba9579e78b47a97111b99c27d6678e31039092`，StartedAt 为 `2026-10-27T17:45:23.4119648Z`；这支持对旧基准启动记录的描述，不等于采样时的逐算子运行分支。

stage230 使用的历史 trace 来自**另一个诊断容器**：`analysis/operator-ceiling-20261028/contract.json` 说明旧服务没有配置 profiler，另建诊断实例；`diagnostic-status.json` 记录诊断容器 ID `238c1fae630aff68ef3a7c77d4a4fdd26e55019b35cd2f8e2ae3a3c776d182f3`、StartedAt `2026-10-28T01:49:32.86049606Z`，另有独立 trace 导出清单。`evidence/baseline-restored-verified.json` 中恢复的服务虽仍为容器 ID `b00f4d...`，StartedAt 已是 `2026-10-28T02:04:13.1426007Z`；相同容器 ID 不能代替进程连续性。不要把旧基准日志的 SP 结论直接套给诊断进程。

本次定向核对本机 `diagnostic-status.json` 所保存的 6000 字 `log_tail`、诊断启动元数据和既有 F27/并行配置审查；该保存日志范围内没有发现实际 aux 层列表、resolved SP 值或将这三条 `ReduceMean` 绑定到 DSpark 的记录。这不证明诊断容器的原始完整日志没有这些信息。既有 layer0 graph/CANN 关联检查也不是对这三个 `ReduceMean` 的绑定，不能当作其否定证据。诊断实例实际 aux 配置、SP 分支、张量与请求身份仍未知。

可复核文件 SHA-256：`diagnostic-status.json` `2fae8e06814b6b6cc6f68174e6dbde528b7374cca4423d1f2a97ceec9ab12e26`；`contract.json` `a023a02cff84670d8946786e696892ddc410255f5cbb2e3720a870a6f3d49fa5`；`guard-measure1.json` `49d4513bfca2aeaf3b62c6536941ff4e2b777ee08d6238ba4239302b63c027e8`；`baseline-restored-verified.json` `30f3ad1664f6e332c4fcb78fd4e191ff4af58fabd0c219ee73f15df0015f5803`。本次仅查本地既存资料。

### `ReduceMean` 的图级关联与仍缺的身份

复用本机三个已有快照，按 **相同 rank、model 49 和时间包含**将 144 条末层候选 `ReduceMean` 与已关联 `MODEL_EXECUTE`、`MODEL_WAIT_COMPLETE`、`aclmdlRIExecuteAsync` 的图级记录连接。144/144 条各自均被唯一一个 `MODEL_WAIT_COMPLETE` 区间包含，run 1/2/3 分别对应各自 rank 的图 ordinal 1/6/7（每组 48 条）；每个选中记录已有一个关联的 `aclmdlRIExecuteAsync`。这比仅有层窗位置多出一层**图级包围线索**，但包含关系本身不证明图节点来源、模型 Python 调用、tensor 生产消费或请求身份；ordinal 不证明跨卡同一次请求，`MODEL_WAIT_COMPLETE` 时长也不代表纯等待。三个快照用的是既有同卡导出时间口径，连接不跨卡相减。

可运行 `python3 agents/diagnosis/recheck_dspark_aux_graph_enclosure.py` 复算：它校验三个输入哈希，核对每条主流/模型/窗口、同卡父区间唯一性及图 API 连接，逐项输出 144 条。结果 `agents/diagnosis/dspark-aux-graph-enclosure.json` SHA-256 `52ad195df1c7e471b6a0dcf75ef7f1e251b0330d29e51e5ef55d4985f699d775`，脚本 SHA-256 `5b8ec127320ce8a4fc55aa63ca36f77f8898c7b7e2f01d4db4c6e9efb43c3b8f`；输入分别为窗口快照 `d98aac57fbf07885e5bfe405e9f50c2e27bc22d4d5aba27e80b95e1aecf4de5a`、图记录 `ecb1ba603e52cf3cf6722f4b8cb0c1aad3b295c596f4f60c7fc047ecff6ef925`、stage230 结果 `13b5584a6b2e93ddf7a7b19da37364b1212e54a875c7a1720ceb80831af0655e`。仅本机 CPU 读取数据。

已保存诊断工件的文件清单提供 profiler DB、trace JSON、kernel/operator CSV、通信导出和 profiler 元数据；清单不是全部文件内容的检索，也不是完整计算图。既有 EP0 profiler DB schema 快照显示 `TASK` 与 `COMPUTE_TASK_INFO` 字段包含 task/model/stream、shape、dtype 等，未见显式 tensor 地址或 Python 来源列；`PYTORCH_API` 虽有 callchainId/shape，但上述父区间连接不等于将图内节点连接到某条 Python API。既有连接 ID 重用、历史日志截断等限制仍在，尚未检索本次目标的所有原始 trace 表或其他未索引工件，不能声称其中必定没有来源信息。

若要把当前候选升级为已绑定的 DSpark aux 操作，最小证据是诊断实例实际安装的 aux 层列表及 SP 状态，以及可验证地对应这次 rank/graph/node 的算子调用来源或输入输出 buffer/数据流身份；若要判断通信依赖，还需该次 allGather 与 mean 结果的对应和事件依赖。可先查看已有、带来源与版本索引的诊断完整启动日志或已保存图/节点映射（如实际存在），现有文件清单尚不能证明这些工件存在；否则在现行不启动新 NPU 的限制下保持未知，不通过时间相邻或形状推断。

复算由本机 ZCode `deepseek-flash` 的 stage232 执行：exit 0、model_verified=true，标准输出 SHA-256 与已保存结果 `52ad195df1c7e471b6a0dcf75ef7f1e251b0330d29e51e5ef55d4985f699d775` 一致。无 NPU 操作。

## 三个辅助状态的通信合批：条件性候选（2026-10-30）

冻结 `model.py:1555–1565` 的 SP 条件路径，在选中层入口分别对各自 hidden 做 `mean(dim=1)`、分别调用 `sp_all_gather`，切到 `full_num_tokens` 后按层次加入 `aux_hidden_states`；在 forward 返回时又把列表作为输出之一。**三个**仅在冻结配置的三个 DSpark target layer ID 经前文说明的 runner 初始化路径安装、无覆盖，且这些层属于本次执行等条件下成立；历史实例的安装列表、SP 和具体 gather 数据身份未绑定。已有 144 个 `ReduceMean`/allGather 相容窗口不能反过来证明这些实际通信就是源码中的三个状态或属于同一请求。

一个不同于前述“各状态提前部分投影”的结构候选：每次选中层到达时先按原 mean 的数值规则保留本地结果，待三个本地结果就绪后在 feature 维合并，进行一次更宽的 token 维 gather，再按 feature 切回三个列表项并分别保留原来的有效 token 截断。若真实通信在 rank 间给各状态使用相同且可证明的 token 归属、rank 顺序和 padding，且三个本地行域相同，记每 rank 的局部状态为 `A_r,B_r,C_r`，理想地可由 `G(concat_feature(A_r,B_r,C_r))` 的三段列恢复 `G(A_r),G(B_r),G(C_r)`。这是条件关系，不证明当前 SP wrapper 的具体布局、当前接口支持变长或编译图能直接采用。各项 dtype、原 mean 已发生的舍入、结果列表顺序、完整 shape 和实际外部消费者必须一致；不能因草稿侧随后可能拼接就删除原列表项或改变 forward 的其他可见输出。

**顺序反例：**如果某个替代方案改用 token 维拼接，并且仅作假设 gather 把各 rank 的发送块按 rank 顺序拼接，两 rank 各一个 token，且每 rank 先按 A/B/C 拼接时，接收数据会得到 `[a0,b0,c0,a1,b1,c1]`。直接把它均分成三段，会得到 `[a0,b0]`、`[c0,a1]`、`[b1,c1]`，而原三个结果应为 `[a0,a1]`、`[b0,b1]`、`[c0,c1]`。所以合并后需要正确地依据真实 rank/token 布局重排并分别裁切；即使输出形状恰好相同也不够。例子是对一种**假设顺序**的数学反证，不是对本机 HCCL 实际排序的断言。本机 ZCode 对这个两 rank 例子独立复核，结果见 `zcode/f27-gather-pack-example.zcode.json`。

同步和生命周期构成另一条硬边界：第一、第二个 mean 源自更早层的 hidden，必须在原定义的那一刻取得并保存，不能等到第三层直接重新读取被后续层更新的 hidden；临时 buffer 在 gather 完成之前不可覆盖。合并后发布结果须等待全部三个输入和通信完成，并确保每个列表项在自身全部消费者读取前有效。源代码只在所查函数结束时返回列表，**尚未穷尽 hooks、共享缓冲及运行时其他消费者，不能据此断言前两项无人能提前消费**。而且三次 collective 原本位于不同层之间；若延后合并，各参与 rank 必须一致改动 collective 的参与者、次序和消息契约，尤其要核实这些层之间与 attention、MoE 或其他路径同组的 collective。某些 rank 延迟而另一些保留原顺序会破坏通信协议；即便全 rank 一致，也可能丢失原本的并行交叠或增加等待，不能按三次调用减为一次来推算性能收益。

若每个有效行的三个 mean 都是 BF16 `[5120]`，合并后的内容本身仍是 **3×5120×2 = 30,720 B/行** 的逻辑数据，与三项分别表示的总元素数相同；这不是实际链路传输量、必要 HBM 流量或可能节省的字节。打包、重排、padding、buffer 驻留、collective 算法与原可能的 overlap 均可能产生新成本。具体可证伪条件包括：真实运行在任意 rank 未安装相同三层、行域/token 顺序不一致、某列表项有需要更早完成的消费者、或 collective 的顺序无法一致改动；这些条件一旦出现，不能直接使用上面的合批结构。代码历史版本、运行参数与实际事件还未绑定，不修改实现、无新增 NPU 操作。

证据：冻结 `model.py` SHA `101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91`；`evidence/model-config.json` SHA `43d3b2bc55626893d59275b32e04068c0fc916f5c382eb91de73bdbade9c88f7`；前述同批物理窗口结果 `agents/diagnosis/dspark-aux-boundaries.json` SHA `13b5584a6b2e93ddf7a7b19da37364b1212e54a875c7a1720ceb80831af0655e`，其限制仍按 F27 前文执行。

## 局部 token 与局部输出权重的二维分片：只算对角块一般不够

这一节收紧前述“提前部分投影”和“合批 SP gather”的同一 F27 候选，不另计收益。冻结 `model.py:1559–1565` 在所选 target 层取本 rank 的 HC mean，SP 分支再按 token 维 gather；runner 快照 `ascend_runner.py:2063–2091` 按相同 token 行选择并把各 aux 最后一维拼接；冻结 `dspark.py:192–201,281–282` 用 `ColumnParallelLinear(..., gather_output=True)` 执行 `main_proj`，然后做 `main_norm`。所存 EP8 的 BF16 `[24,15360] × [640,15360] → [24,640]` 与 TP=8 时每 rank 5120/8=640 个输出分量相容，但 **shape 和启动请求不证明**该任务、输入、权重、SP 与 draft 的 TP communicator、token 归属以及运行路径已经相互绑定。

在以下明确的假设下：各 rank 的 SP 状态拥有互不相同的有效 token 行；draft `main_proj` 的各 rank 分别持有互不相同的输出 feature 权重分片；两种分片参与者及 rank 对应关系已知，且每个被选中 token 最终均需要所有输出 feature，那么所需结果为 `Y[token_rank, weight_rank] = X[token_rank] @ W[weight_rank]^T` 的二维块。若各 rank 只用本地 token 与本地权重计算，接着仅交换这些计算结果，只产生 `Y[0,0],Y[1,1],...` 的对角块，不一般地决定其余块。全宽 RMSNorm 之后还需要完整行的信息及符合原调用的输出，不能把缺失块当作零。

可复算的两 rank 反例：`x0=(1,2), x1=(3,4)`，两输出分片的权重分别为 `w0=(1,0), w1=(0,1)`，则完整结果为 `[[1,2],[3,4]]`，本地对角计算仅得 `[1,4]`。另一组 `x0=(1,9), x1=(8,4)` 的对角结果仍是 `[1,4]`，但完整结果是 `[[1,9],[8,4]]`；即使交换对角结果和已知权重，接收方也无法区分这两组缺失的非对角值。该抽象例采用整数精确计算，只反驳不附带额外信息的通用简化，不假设这些权重或 token 是 169 checkpoint 的实际值：

```bash
python3 - <<'PY'
w = [(1, 0), (0, 1)]
for x in [[(1, 2), (3, 4)], [(1, 9), (8, 4)]]:
    dot = lambda a, b: sum(p*q for p, q in zip(a, b))
    print('full:', [[dot(xi, wj) for wj in w] for xi in x],
          'diagonal:', [dot(x[i], w[i]) for i in range(2)])
PY
```

因此，对该条件分片结构的重组需取得能计算或重建非对角块的足够信息，例如按需传输已选 token 的状态、按条件复制/传输权重分片、交换可证明充分的部分表示、或改变计算与输出所有权；这些只是候选，不意味着必须采用原两个 all-gather，也不构成字节或耗时下界。若所有 rank 已复制权重、行本就复制、仅需部分输出、权重具有已证的特殊相关性、或其他合法预计算提供了足够信息，上述假设不成立或结论可能收紧。逐层产生的三个 aux、选行顺序、rank 分组、通信/投影的 completion、浮点归约与中间舍入、原列表及其它消费者仍须满足 F27 先前合同；这里没有证实历史运行或收益，也未触设备。

本轮重新核对 SHA-256：`sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py` `101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91`；同目录 `dspark.py` `fcee256905928917154d8dac94d6611bb83d02da441e8aa154a525ef51e6a601`；`agents/bounds/metadata-path-snapshot/ascend_runner.py` `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`；既有 `agents/fusion/f27-dspark-gap-evidence.json` `d68ee49164b4db836705c08f026e939772974e72e54fc4b59eafd5458aa9f19b`。
