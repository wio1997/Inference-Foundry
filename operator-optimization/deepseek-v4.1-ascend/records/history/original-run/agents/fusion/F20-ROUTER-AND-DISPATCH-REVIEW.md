# F20：router GEMM → scoring / expert 选择 → dispatch 边界

这是 F20 内部的新审计；先前 `F20-NORM-ROUTER-CONSUMERS.md` 只覆盖 **F04 → F20** 的 norm 双输出连接。本报告不代表 F20 全部底层指令和所有分支已覆盖，也不估算收益。

ZCode `operator-ceiling169-v2-112-f20-router` 退出码 0、verified `deepseek-flash`。`f20-router-provenance.json` 验证冻结基线容器和 CPU stage 的镜像、路径挂载及三个 router 源码哈希一致；本地三文件哈希已复核。已捕获层窗口 `f20-target-window-summary.json` 中 40 层各有 48 条归属的 `MoeGatingTopKHash` 和 48 条 `DispatchFFNCombine`；这是按旧窗口和名称汇总，不能从同窗次数推出张量一一对应，也不能把各条都视为独立请求。

## 生产者、配置和分支

冻结 `model.py:176–179,210–259,282–298`：gate 是 replicated linear，设置 `precast_fp32_weight`；外部 router 分支使用 `hidden_states_fp32` 或 `hidden_states.float()` 做 `F.linear` 产生 logits，内部 router 分支则将 router 输入与 expert hidden 和 input_ids 交由 experts 内部处理。该源码并不单独证明当前实际实例、逐次 dtype 或 GEMM 已和后续步骤合并。

模型保存配置 `text_config` 有 384 routed experts、每 token 选 6、`scoring_func=sqrtsoftplus`、`norm_topk_prob=true`、routed scaling 1.5。实例内部传递的参数和真实分支仍须由运行时对象/trace 确认。`model.py:210–234` 仅对满足 `layer_idx < config.num_hash_layers` 且不是 draft layer 的层构造 `tid2eid`；未查明 runtime `num_hash_layers` 或每层实际 table，不能从名字断言 40 层均按固定 token 哈希选专家。

冻结 `router_factory.py:15–33,56–118` 在满足接口检查且非 custom routing/310P compatibility 时建立 `AscendFusedTopKRouter`；另有 grouped fallback。即使 factory 选择 fused 类，`fused_topk_router.py:122–154` 还依据 shape、vision bias 和 group 等条件可能走 fallback；因此静态配置或同名 task 不等于每步对象的实际分支。

## 已存在的复合路径及其边界

`fused_topk_router.py:156–223` 的 sqrtsoftplus 条件分支发出单一 `_C_ascend.moe_gating_top_k_hash` 调用，输入包括 logits、top-k 数、可选文本/视觉 bias、可选 input_ids/tid2eid、group 参数、scaling、epsilon 等，返回 topk weights 与 IDs，并按接口约定转换结果 dtype。source 调用指定 `norm_type=2`、`renorm=0`；旁注说该 hash 自定义 op 当前只接受 0。**不能据此把 `renorm=0` 解释成模型必然没有权重归一化，也不能从 Python 参数单独推断算子内部所有数学步骤、舍入或适配了哪些精度。** 真实数学语义、bias/权重顺序与完整实现需要核对所用已编译 op 或对应冻结源及 contract；当前不以“归一化缺失”定性。

重要：同一个 `moe_gating_top_k_hash` 调用在 `tid2eid is None` 时也存在（`:177–179,204–210`），因此 profiler 名称带 Hash 不代表该层一定执行 token→expert 表查询。`tid2eid` 或 vision bias 存在时，router 还会按通信类型 gather 或 pad/split input_ids，对未对齐 token 做 SP chunk，并将 -1 替换为 0（`:159–176`）；选择和打包必须保持这些不同的 row 域和掩码。

冻结 `grouped_topk_router.py:80–128` 的 **fallback** 是显式 `softplus(...).sqrt()`、条件分组和 correction bias、top-k、权重归一化/scaling；vision fallback `fused_topk_router.py:32–74` 区分 image/text 路由并在必要时从未加 bias 的 score 取权重。这些 fallback 的具体公式不是 hash 自定义算子的已证内部实现，不可混用或据其中一条证明其它分支 bit-exact。

`moe_comm_method.py:563–608` 对 fused-MC2 的 dispatch 分支，把 `fused_experts_input.topk_ids` 作为 `expert_idx`，`topk_weights` 转 FP32 后作为 `probs`，并与 hidden、quant 权重/scale、EP group 和 mask 交给 `dispatch_ffn_combine`；它已经是复合调用，不能再次把其内部 dispatch、expert FFN 和 combine 当成分别未融合的独立操作。router 输出到该结构之间的 EPLB 映射、padding、具体 packing 及全部消费者在当前快照尚未闭合；不能仅根据名字和局部字段认定完全直接传递或当前 packing 已/未被融合。

## 可继续调查的条件重组

- **GEMM→router 跨边界**：可研究按 token/专家列分块算 gate logits，再将评分和候选合并到 router。每个 token 的正确前 K 项依赖所有可能进入最终选择的 expert 列，group 筛选、bias、哈希表/vision 分支与权重归一化都须分别保持；可以分块求局部候选并正确合并，不可假设一个不完整 tile 就知道全局选择。数值上需保持 GEMM 输出 dtype、rounding、scoring、选择 tie/NaN、原/偏置 score 的区别，以及返回权重和 ID，不能只求 expert ID。
- **table 分支优化**：如果某个实际调用证明由 tid2eid 决定选择，可能研究只计算所选专家所需 scoring，而非先得完整 logits；但须先取得冻结 hash op 语义，证明 vision 行、bias、分组、需要的归一化/其他消费者及输入对齐满足该变换，且不破坏现有 dispatch 和采样结果。当前仅是条件问题，**不证明这次计算可以省去**。
- **与 dispatch 分块交接**：当一个 token 的最终 ID、权重及输入行就绪并且 EP/HCCL 接口保证相应块可见、对齐与通信顺序后，可能让该块进入路由发送；代码未证明 MC2 提供 tile-ready 接口或支持 token 级独立发射。共享 expert、EPLB、buffer 生命周期、各 rank 集体顺序和资源冲突要保留。不能将现有 `dispatch_ffn_combine` 再重复计为一个新融合机会。

对“sqrtsoftplus 实数单调所以先 top-k logits 一定等价”，`f20-monotone-rounding-witness.json` 仅给一个严格的 IEEE FP32 sqrt 反例：相邻不同输入可舍入为相同 sqrt 结果；实际 softplus/custom op、tie 规则和真实 logits 尚未证，该反例只排除**不附加条件的数学推断**，不声称当前模型实际会碰到它。所有新的设计需要自己的精度证明和实际信息，现阶段不改代码或设备。

## 草稿 gating→MC2 窗口的适用性复核（stage190/192）

上述分块交接是 F20 既有条件候选，不因草稿采样再次登记为新的融合。stage190 在 EP1 的三个固定窗口通过 task 的完整字段、connection 和 connection-set 将两端分别关联到名称类别匹配的 CANN/PyTorch 候选，但这不能证明两端逐次 tensor 身份、实际 router 对象、分支或其间 producer-consumer 依赖；PyTorch 候选 `sequenceNumber=-1` 且 `callchainId=null`。因此采样不能证明这里发生了确定的全局路由同步或跨块可直接交接，也不能证明 F20 之外新增融合机会。

stage192 在上述三个 CANN 同线程候选区间中分别发现 1 条名称属于 `sync` 类的短 API 记录（0.400、0.410、0.380 µs，connection 分别 102086、102407、102728）；stage193 已将三条完整名称核实为 `aclrtGetHardwareSyncAddr`；名称不证明它在等待或执行设备同步，不能归因为整个 200 µs 区间的成因。CANN 和 PyTorch 表的区间不应相减，短 API 的持续时间也不是等待或设备空闲时间。数据依据见 [stage190/192 草稿诊断](../diagnosis/DRAFT-EP7-QUERY-METADATA.md) 和 [stage192 逐行摘要](../diagnosis/draft-mc2-gap-host-intervals.json)。

对当前所存源码，router 的 `moe_gating_top_k_hash` 调用已把若干选择步骤放在一个复合操作里，`dispatch_ffn_combine` 又把 dispatch、专家 FFN 与 combine 封装到另一个复合操作中；仍可研究 F20 的 token 分块交接，但先要绑定 router 输出及 MC2 输入实际张量和中间变换，确定 EPLB、padding、掩码、其他消费者、collective 顺序以及内存可见性。F21 已记录内部已有按专家组 dispatch/GMM1 重叠，不能重复计为尚未融合的独立机会；MC2 历史二进制与保存源码仍未绑定。


## 草稿 gating 到 MC2 的 Python 条件路径（源码复核）

补充同一已保存源码快照中的中间路径，消除“router 后直接等于 MC2 入参”的歧义；不是对历史三次运行的分支断言。`evidence/precision-paths-20261028/snapshot-initial.json` 中各源码 SHA-256 与本地文件匹配：`routed_experts.py` 为 `f9db74836ae210a541432b923118696c7a15be9a10d5776834e1d3b64ce11d77`，`w8a8_dynamic.py` 为 `a67601ee596d9aa4ae72ea3638f67ff7236cd8df884564bcd19f33653a1e0910`，`moe_comm_method.py` 为 `7040f691355bd9acb3c668e698de3f7d3854edbc22e3f979f9b03d99800364d7`。它们与对应历史 Python 实例及编译对象的逐次绑定仍未证。

`routed_experts.py:638–691,731–751` 中，在 `_select_experts` 的 router 返回之后，`log2phy is not None` 时对 ID 做映射索引；`mix_placement` 时生成共享专家 ID/权重并拼接；`enable_force_eplb` 或 profile run 的负载均衡分支可能改写 ID。随后 `forward_impl` 将这些输出传给 `quant_method.apply`。这不是说上述分支在本次采样开启，也不是说操作必然发生主机等待。相应 `w8a8_dynamic.py:203–237` 组装 hidden、topk、mask 等调用 `moe_comm_method.fused_experts`；`moe_comm_method.py:569–605` 的 `enable_fused_mc2 == 1` 且非 MegaMoE 分支先 `torch.empty_like(hidden_states)`，再将 `topk_ids` 与 `topk_weights.to(torch.float32)` 传入 `dispatch_ffn_combine`。router 侧返回值也有 `.to`，调用可能因 dtype 相同不复制；实际 dtype 和后端实现未绑定。

可检验假说是该间隔含 Python 分支、输出缓冲准备或下发工作；stage193 三次 `empty_like` 外层分别约 10.66、10.93、10.31 µs，位置和源码相容，但无调用栈、tensor 身份和实际分配行为，不能把它直接归给该行；也不能把这些 API 的嵌套时间累加，更无法由此解释全部 200 µs 设备记录缺口或证明主机同步。要区分需先利用现有 trace 的调用栈、内存事件、dtype/shape、线程状态与相关 ID 验证绑定及采集完整性；若材料不含这些证据则保持未知，不能用同名/相邻时间代替。

## token ID 补齐的已存样本与跨层复用条件

沿用已保存 profiler 聚合 `target-layer-all-kernels-16rank.jsonl`，限定同一摘要 `7ddf7ae98a00…`、EP0–7、每卡 L0–39 各一 scope。每个 scope 均唯一关联一条 `aclnnConstantPadNd_PadV3AiCpu_PadV3` 和一条随后同流 `MoeGatingTopKHash`，两类各自的输入输出 shape、dtype、format、核心及 block 字段跨 320 scope 一致。前者记录为 AI_CPU：INT64 `[595]` 补齐到 INT64 `[643]`，另有 INT32 padding 参数；后者包含 INT64 `[643]` 输入及 FLOAT `[643,384]`，输出含 FLOAT/INT32 `[643,6]`。所选 pad 记录耗时 min/median/max **50.579 / 103.978875 / 305.774 µs**，gate 为 **28.93925 / 31.5195 / 43.03925 µs**，逐 scope 320/320 pad 的记录时长大于 gate。pad 结束至 gate 开始的记录间隔 min/median/max 为 **7.49925 / 8.199875 / 8362.640 µs**；其中存在 Equal、Fill、Select 等同流记录，间隔不是空闲或等待的证明。

每卡选中的 40 条 pad 在自身设备时间线中按层序且区间不相交，其记录区间并集分别为 **3.927–4.958 ms**；不同卡不可相加推断整轮耗时。它只是一次带 profiler 的选定算子时长，不是线上关键路径、CPU 全部利用率、实际 token 内容相同或可节省时间。EP0–7 的另一摘要各 40 scope、EP8–15 的两个摘要各 40 scope 未找到相同签名的已选任务；这不等于它们实际没有补齐或成本为零，采集与关联有缺口。

调用栈绑定：EP0 L0 scope `736-51`，flow `1793241333374500830` 已由既有提取器的 host flow → device X → kernel CSV 唯一关联 stream 45 / task 45438。只读该 rank 的历史 `ascend_pytorch_profiler_0.db`，在提交线程与 host 起点精确匹配得到唯一 `aclnnConstantPadNd` API（connectionId 496472、callchainId 1198793），栈依次包括 `torch.nn.functional.pad` → `prepare_finalize.py:338 pad_and_split_input_ids` → `moe_comm_method.py:327` → `fused_topk_router.py:169` → target `model.py:1592`。进一步只读 EP0–7 的各自 profiler SQLite，对上述 320 条分别按本 rank 的 host 线程与起点精确查找，**320/320 唯一匹配 API，且每条栈都含上述四个路由/模型帧**；每 rank 40 个 callchainId 互不相同。这支持把选中记录归于该路由 ID 准备提交路径，而非仅凭 shape 猜测。host 与 device 时钟没有相减；采集时 Python 文件/对象 hash、pad 的 storage 与 gate 输入身份仍未绑定，也不证明其余 960 scope 的行为。

冻结 `model.py:1541–1551,1592` 中 input_ids 在条件 SP 分片后传入各 target 层；冻结 `fused_topk_router.py:159–176,204–209` 的 sqrtsoftplus 且 tid2eid 或 vision bias 存在时，ID 转 INT64、按通信方式 gather 或 pad/split、必要时再按 TP chunk、将 -1 映射 0 后交 gating；`prepare_finalize.py:327–347` 的不同分支对 padded 数量及 TP 处理不同。可研究在同一 forward 中复用已准备 IDs 或与上游行布局构造合并，但必须先证明各层的 ID 值/行域、通信组与 TP rank、`num_tokens`、padding、特殊值处理、输出 buffer 生命周期、写入和跨流事件均相同；同形状不够。各层可能有不同 tid2eid/vision bias、门控分数及路由，不能借此复用 TopK 或跳过 gating。算子名含 Hash 也不证明所有层都采用 hash 表。本次只是候选，未实现、未评估收益。

CPU 复算 `python3 agents/diagnosis/analyze_int64_pad_gating.py`：输入 SHA-256 `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`、脚本 `4a3f481548081bbff25964b283017343a920c21b87dbc84571c09d0a724d0d9a`、输出 `int64-pad-gating-scope.json` 为 `49e4028342df962d0724c7f95aac4537b4804f7227049ffa42bf22040829c218`。320 条栈只读复算脚本 `agents/diagnosis/query_int64_pad_stacks_320.py` SHA-256 `da7108d4c62f9c8e9b5bb06601281ffc8d95606fb055e407a261a796fe235b5d`：在专用 stage 容器传 `--task-root`，按 JSONL 的 rank 目录以 SQLite `mode=ro` 分批检查 API 与栈，不读 tensor 值、不操作 NPU。原先单条复算脚本 `query_int64_pad_stack.py` 保留。冻结 `prepare_finalize.py` SHA-256 `dff4fac292200fc06f60aef48505f1ee38314ea5adb53560b325901db308eb24`。

### 先按行合并补零与特殊值映射，再审查跨层复用

在冻结 `replace_allreduce=True` 且 pad 后不再触发额外 TP chunk 的分支，若 pad 的中间结果没有其他必须保留的观察者，那么对长度 N 的 INT64 输入 x、末尾补 P−N 个零，随后 `where(value==-1,0,value)`，可在数学上合并成单个按行表达式：`y[i]=0` 当 `i≥N` 或 `x[i]==-1`；否则 `y[i]=x[i]`。例如 `[-1,7,-2]` 扩至 5 行得到 `[0,7,-2,0,0]`；不得把其他负数、图像 sentinel 或合法 token 0 一律改写。这是精确的整数值等式，不证明后端实际可融合或更快；需保留 output dtype、stride、空输入、图缓冲地址及所有消费者。320 条栈行号均与 `replace_allreduce=True` 下的 `:338` 相容，但历史加载源码、tensor 身份及额外 chunk 条件尚未绑定。

`-1`、真实 token 0 和补出来的 0 在映射后可能变成同一个 0，不能仅靠 prepared IDs 反推出原来哪些行有效；如果消费者还需要原始行/掩码，必须保留独立 mask 或经过证明等价的状态。条件下跨层复用 prepared IDs 还需同一 forward 内输入值版本、分支、`num_tokens`、padded 总量、TP 组/rank、行映射与 buffer 生命周期均相同。shape 相同不足以证明：在另一个源码分支中输入 `[10,11,12]`、`num_tokens=3`、TP=2/rank=1，pad 到 P=4 后该 rank 得 `[12,0]`，pad 到 P=5 得 `[0,0]`；两个结果形状都是 `[2]`。这是构造反例，不声称历史 P 实际变化。各层 logits/专家表仍不同，不可据此复用路由结果。

### 已选任务时间线与异常间隔的关联

在上述 320 scope 的每个 pad 结束与 gate 开始之间，逐个筛选同设备、同 stream 且已关联的 scope 内任务，**320/320** 恰有按顺序排列的 Equal、Fill、Select 各一条；其记录区间并集 min/median/max 为 **7.37975 / 7.999875 / 8.740 µs**。进一步以每个任务的 host 起点、线程与 API 名称精确查各自 rank 的 SQLite：Equal、Fill、Select、gate 四类各 **320/320** 唯一匹配，前三类调用栈均含 `fused_topk_router.py:176`，gate 均含 `:204`；此前 pad 自身 320 条绑定见上文。它们支持所选调用与源码的 `where`/gate 位置对应，但历史源码 SHA、标量值与 tensor storage 仍未绑定，不能仅据此断言完整 tensor 边或数值。其余 scope、未匹配任务和同步事件未被覆盖，间隔的剩余部分不能叫做设备闲置。

以同 digest、同 rank、同层和同 scope 连接 pad→gate 间隔与同 scope 的 MC2：在事后选定 `>100 µs` 描述阈值下，EP3 的 40 层均超过阈值且其 MC2 均是同层八卡中最短；EP2 的 39 层超过阈值且其 MC2 40 层均为第二短，第 28 层是间隔 **8.54025 µs** 的例外；其余六卡均没有超过阈值的 scope。这是一次 profiler 采样内的共同出现，未比较跨卡绝对时间，不能推断 gap 造成 MC2 变短或通信被掩盖；真实路由量、内部阶段、未匹配事件、历史二进制及 profiler 扰动均未知。

CPU 复算 `python3 agents/diagnosis/audit_pad_where_gap.py`（脚本 SHA-256 `e0685e70f245e95b0bd2ef2198301bfee57ad125a333816071166336325e4802`，输出 `pad-where-gap.json` SHA-256 `822f20c03e7711793e4e196d36437d5f75b3523404b2879be97b0280371fd446`）及 `python3 agents/diagnosis/audit_pad_gap_mc2_rank.py`（脚本 `c50c278be56d68dea247b56c74e85fdfafe9ce389468eea1c187c2ab1d2eb858`，输出 `pad-gap-mc2-rank.json` `64703ddfb13a711016d61bde5d5dc2483bbfdcb746b0c288555e58bbbb523c5b`）。输入哈希在脚本中校验；未启动 NPU。

对四类 host 栈的只读复算，在专用 stage 容器运行 `python3 agents/diagnosis/query_pad_where_gate_stacks.py --task-root <本任务目录>`；脚本 SHA-256 `25d8394f9bc668cccf24eaa5418d85ed6bb61588e8c8291a1343cf0af567cb66`，校验三份现有输入哈希并查询八个 rank SQLite 的 `PYTORCH_API`、`PYTORCH_CALLCHAINS`。每类结果 320 唯一 API、320 目标栈。此查询不读 tensor 数值，也不将 host 与 device 绝对时刻相减。

### 两个长间隔与一个短例外的完整已导出 TASK 查询

仅选同一摘要下 EP2/L15、EP3/L15、EP2/L28，以每卡数据库内 pad 的 TASK 结束和 gate 的 TASK 开始定义半开区间。按 deviceId、stream、task 和开始时刻对端点各唯一匹配；在对应 device 的**所有 stream**查询 TASK 交集。三个窗口分别为 **5731.545、6550.460、8.540 µs**，每个都只查到此前的同流 Equal、Fill、Select 三条。例如 EP2/L15 三条分别在 pad 结束后约 **1.702、3.225、5.688 ms** 才开始，EP3/L15 约为 **1.911、4.088、6.513 ms**；EP2/L28 三条紧邻窗口起点。查询对应 `COMMUNICATION_OP` 区间、所在库 `COMMUNICATION_TASK_INFO` 时间点和非零时长 MSTX 区间均无匹配；后两表没有 deviceId，MSTX 点事件不在此筛选内。数据库是现存导出，并不保证包含全部设备活动、内部通信或其他进程；跨轨时钟关系未证明，空结果**不能证明设备闲置、无通信或给出根因**。

复现 `python3 agents/diagnosis/query_pad_gap_full_tasks.py --task-root <本任务目录>`，脚本 SHA-256 `fa59f71c0fabf52c32c444d59ebe623c7b626a9dfa4306505337d02d8f31298c`；DB 只读。EP2/L28 的 trace 记 8.54025 µs，数据库以整 ns 计算为 8.540 µs，不应把不同精度强行当作矛盾。两份历史 DB 在本次首轮查询前后分别观察 SHA-256 `2f51a3a608eaf21fbadbb98d54e79032bc23638da76cd80344e9b5c363add7c9`、`084b8718888e33271435ec79d7573c95ca1615f3e7484c807ea24590fad234c4`；这只证明该次读取稳定，不证明过去运行时二进制或数据的身份。

### 对已存 A3 路由源码的进一步条件重组

冻结 `moe_gating_top_k_hash_without_group.h:218–227,326–336` 的 `IsImageRow` 与 `SelectExpertIdxByHash`，各自读取 input ID 后均将 `-1` 替换为 0；该文件中 `inputIdsGm_` 的数据读取只出现在这两处。因映射 `f(x)=0 if x==-1 else x` 满足 `f(f(x))=f(x)`，如果该源码对应的 without-group kernel 确实是唯一需要观察清理后 ID 的消费者，且保留 padding、INT64 布局、有效 mask、输入生命周期和异常行为，那么可以研究把 Python 侧 `where` 的映射留给这两个读取点执行，而不单独物化其输出。是否存在其他消费者、不同分支/二进制、图重放和真实 tensor alias 都未绑定；现有 host 栈和 task 名称不足以证明可在历史服务直接删除操作，也不能保证收益。源码 SHA-256 `f18d63bda201eb407a3861da0f5189d45d439b6630abf52058a6eaa0420d8fdb`。此项与前述 pad+where 及跨层复用是重叠候选，不可把收益叠加。


### 发射线程的 tiling 记录：三个样本

在已导出数据库中，按 device TASK 的 connection ID 唯一找到各自对应的 CANN launch。五个 launch 位于同一发射线程；其间 Equal、Fill、SelectV2 的 tiling 记录分别按同线程、名称和区间嵌套关系落在对应 aclnn 与 PyTorch Dequeue 内，且结束早于各自 launch。tiling 记录自身没有直接与 TASK 的 flow 绑定，因此该关联依赖上述名称与嵌套，不代表 tensor 身份或请求关键路径已确认。主线程提交与发射线程是不同线程。

| 样本 | 设备 pad 结束→gate 开始间隔 µs | 发射线程 pad→gate launch 起点跨度 µs | Equal / Fill / SelectV2 tiling 记录 µs |
|---|---:|---:|---|
| EP2 L15 | 5731.545 | 5824.690 | 1752.390 / 1482.560 / 2421.360 |
| EP3 L15 | 6550.460 | 6715.070 | 2032.560 / 2134.730 / 2375.560 |
| EP2 L28 | 8.540 | 3732.190 | 1145.820 / 1135.900 / 1297.490 |

每列只是各自时间轴内部的间隔；尚未证明主机与设备的时钟对齐，不能跨列相减算排队。L28 是重要反例：短设备间隔也伴随毫秒级 tiling 记录，因此不能仅凭这些记录认定 tiling 造成长设备间隔，也不能排除它在其他条件下的影响。区间长度可能包含 profiler 扰动、线程阻塞或调度时间，不代表全部为有效 CPU 计算，也不是无 profiler 的线上耗时。后续可查当前导出中 enqueue/dequeue 关联和可证实的同步记录，或转到其他独立路径；必要的新采集需先取得授权。

只读复算：在专用 stage 容器执行 `python3 agents/diagnosis/query_pad_launch_tiling.py --task-root <本任务目录>`。脚本 SHA-256 `914983f509dec3573af7025ef308d6658d5db9c12e33b81b4e4a41d2d504ca24`；对五个 TASK、launch 的连接及三个 Dequeue/aclnn/tiling 的同线程嵌套均做唯一性和顺序断言，不启动 NPU。

### 条件分支与输入连续化的额外约束

冻结 A3 tiling 源中 softplus 或 vision bias 条件要求 group_count=1；在非 950、source 与二进制匹配、input_ids 为 INT64 且 tid2eid 未提供的条件下，表 dtype 默认 INT32，对应 host 数值 key 6。该 key 在 host 命名为 `TILING_KEY_WITHOUT_GROUP_3`，在 kernel 中数值相同但命名为 `TILING_KEY_WITHOUT_GROUP_4`，对应 `WithoutGroup<DTYPE_X,int64_t,int32_t>`；名称不同不能据此判断键不匹配。现有轨迹尚未证明实际属性、可选参数和历史二进制，不能断言实际选中 key 6。源码 SHA-256：host `839a3f156fa5a74d792297f002ae77d604aef00c64745c7d125c7be505aef842`，kernel `db68b6b7c94d8a66246ced81c8ea65a31f0e84955fb3c9dd66e7554bf4566563`。

算子声明对 input_ids 使用 `AutoContiguous()`（冻结 `moe_gating_top_k_hash_def.cpp:52–66`）。绕过 where 后若输入变成非连续 view，运行时可能引入连续化转换；原 where 输出还可能提供独立 buffer 快照，取消它必须核对输入别名、异步写入、图重放、事件与生命周期。源码声明本身不证明此次发生转换，也不证明重组可省时。


### 同 scope 与 MC2 的时长规模对照（2026-10-30）

连接 `agents/diagnosis/int64-pad-gating-scope.json` 的 320 条 pad 与 `agents/diagnosis/pad-gap-mc2-rank.json` 中按相同摘要、rank、layer、scope 归属的 320 条 MC2：每组都唯一匹配，逐组 `pad duration / MC2 duration` 的中位数约 **0.772%**，范围 **0.301%–9.376%**；所选 MC2 时长中位数约 **11.654 ms**。EP3 的 40 组比例中位约 **4.346%**，其余 rank 的中位约 **0.559%–0.967%**。这是同一带 profiler 的选定样本里的单任务时长规模比较，不是全模型占比，不能将跨流/跨卡时长求和，也不证明路径依赖、延迟暴露或 fusion 的收益。两类任务的 shape、引擎和含义不同；profile 开销、实际 ID 张量身份与服务请求均未闭合。

因此就这一组数据的时间规模而言，继续调查 MC2 的有效工作量和内部等待更优先；ID 补齐/映射融合保留为边界清楚的次级候选，尤其不能因为其单条比 gating 长就认定它是 MoE 或端到端主瓶颈。是否处于关键路径需补事件依赖和请求关联；其他负载的排序可能不同。

本地只读复算（两份文件在仓库已有，未读取 tensor 内容）：

```sh
python3 - <<'EOF'
import json, statistics
p=json.load(open('agents/diagnosis/int64-pad-gating-scope.json'))['rows']
q=json.load(open('agents/diagnosis/pad-gap-mc2-rank.json'))['rows']
k=lambda x:(x['ep'],x['layer'],x['scope'])
d={k(x):x for x in q}
assert len(p)==len(q)==len(d)==320 and len({k(x) for x in p})==320
r=[100*float(x['pad']['duration_us'])/float(d[k(x)]['mc2_duration_us']) for x in p]
print(min(r), statistics.median(r), max(r))
EOF
```
