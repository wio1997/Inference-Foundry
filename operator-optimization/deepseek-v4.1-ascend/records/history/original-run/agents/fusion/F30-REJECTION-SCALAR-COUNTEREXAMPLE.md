# F30：单个草稿 token 概率不足以重建拒绝后的恢复采样

## 源码范围与运行绑定

现有快照 `sources/vllm-workspace/vllm-ascend/vllm_ascend/sample/rejection_sampler.py` SHA-256 `bb1d39de851540c16c2dd2b16df60c60a1fc4957cedb88bdce9417efc0d55eda`，函数 `rejection_sample` 约 558–580 行在非 all-greedy 分支计算 target softmax 和 uniform，调用 `sample_recovered_tokens`；后者约 730–795 行为请求生成整行指数噪声 q，再按条件选择 Triton 或 PyTorch fallback。源码中调用未向其传 `use_block_verify`，默认 False。PyTorch 非 ngram fallback 约 1103–1184 行对每行逐词表计算 `max(0, target_probs - draft_probs)`，除同请求指数噪声 q，再 argmax。这里只是已存源码条件行为，不证明历史服务执行了该文件、fallback 或概率验证；Triton 实现及二进制身份未绑定，不由名称推断其舍入或实现。冻结启动脚本请求 DSpark K=5，未显式指定 draft/rejection 采样方法；resolved config 和服务实际接收的采样 metadata 未绑定。已保存客户端构造了 temperature=0 请求，详细范围见文末。

## 可复算反例

令词表 `{0,1,2}`，草稿 token 为 0，target 概率 `p=[0.2,0.45,0.35]`，两种草稿分布分别为 `d1=[0.4,0.5,0.1]`、`d2=[0.4,0.1,0.5]`。两种分布对选中 token 的概率同为 0.4，target 该 token 概率均为 0.2。若使用通常 acceptance ratio 且随机值 0.8，则两种情形都拒绝（0.8 > 0.2/0.4）；这仅为反例约定，不是历史记录的随机值。

按照上面已存 fallback 的恢复公式，`max(0,p-d1)=[0,0,0.25]`，`max(0,p-d2)=[0,0.35,0]`。使用同一组有限正的噪声（例如 `[1,1,1]`）仍分别选出 token 2 和 token 1。因此，在该源码路径和逐结果等价要求下，仅保留草稿 token ID 与它的概率不能恢复该结果；至少还需有关其余词项的信息或一个经过证明可支持同等恢复采样的替代表示。反例不证明必须物化完整向量，也不排除融合、压缩或等价重新计算。

在任务根目录执行 `python3 agents/fusion/f30_rejection_counterexample.py` 可用 CPU 的 Decimal 算术复算此构造；它不是历史输入，也没有执行设备或声明输出分布等价。

## 对 F30 候选的影响

分块产生或使用 draft_probs 的候选必须将被拒绝位置和请求、target 分布、draft 分布、噪声对应关系与 RNG 状态绑定，并覆盖实际比较、零值、浮点和累积概率规则；不能仅保存一次采样的标量并丢弃其余信息。仅在全部 greedy、实际消费者无需概率，或采用另外证明等价的接口时，才可另行讨论更少的状态。历史是否进入概率路径、所用 verify 实现、请求映射、跨轮生命周期及收益依旧 UNKNOWN；无需重复扫描算子名称或形状，需既存 resolved config、采样元数据和调用/张量关联才可收窄。只读已有材料，无 NPU 工作或目标实现修改。

## 恢复采样的共享随机向量与融合边界

同一冻结文件 `:730–795` 的 `sample_recovered_tokens` 先创建 FP32 `q[batch_size,vocab_size]` 并对整矩阵执行 `exponential_()`；对 `sampling_metadata.generators` 中每个 `(i,generator)`，另外创建同词表长度的 `temp_q`、调用其 `exponential_(generator=generator)`，然后 `where(has_draft_mask[i], temp_q, q[i])` 写回第 i 行。注意：即使某个有独立 generator 的请求 `num_draft_tokens[i]==0`，源码仍先调用该 generator 的随机生成，再用 mask 决定是否选用 `temp_q`。这个调用次序是源码事实；实际耗时、该情形是否发生、设备随机数实现以及 generator 状态的精确推进量尚未核实。

在 `HAS_TRITON=False` 且 `use_block_verify=False` 的 PyTorch fallback，`:1140–1187` 通过 `cu_num_draft_tokens` 将每个扁平化 draft token 位置映射到 request 行，`q[token_to_batch]` 因而让同一请求的所有位置使用同一行 q。每位置以 `max(target_probs-draft_probs,0)` 或 ngram 修改后的 target 作为逐词表残差，除以 q 后 argmax，得到 `[num_tokens]` 恢复候选；零和无限 q 有专门处理。调用者在非全 greedy 分支先计算 FP32 target softmax 与 uniform，再生成全部恢复候选，最后进入验证逻辑（`:558–585`）。q 由一个 request 的多个候选位置共享，不能当成每个位置独立的新随机张量。

一个纯算术反例：设同一请求两个位置的 target 都为 `[0.4,0.4,0.2]`，draft 都为 `[0.2,0.2,0.6]`，则两处残差均为 `[0.2,0.2,0]`。若共用 q=`[0.2,1,1]`，两处 argmax 都是 token 0；若重组时第二个位置另用 q=`[1,0.2,1]`，第二处变成 token 1。两行 q 均有限正，第一候选向量因此不同。此例仅是 fallback 的有限输入反例，不代表实际请求、实际随机值或历史 Triton kernel；验证器可能只消费其中一个恢复候选，因此不能单靠候选数组的差异断言最终请求输出或输出分布必然不同。

在相同的玩具残差下，还可区分单位置分布与**联合分布**。假设噪声各项相互独立、连续且前两项同分布并为正；每个位置输出 0 当且仅当 `q[0] < q[1]`，否则输出 1，平局概率为零。两个位置共用一行 q 时，两个候选必相同；如果两个位置的 q 行彼此独立，则各位置的 0/1 概率均为 1/2，两个候选不同的概率为 1/2。两种方案的单位置边际分布相同，但候选联合分布不同。该推导限定理想独立连续随机变量，不能直接套用具体设备伪随机数、请求、验证器选取规则或最终服务输出；是否允许改变联合行为由所要求的输出与随机状态契约决定。

可研究按词表块将残差、除 q、局部 argmax 与跨块归约组合，或将其与后续验证衔接，但须保持 request→位置映射、同请求 q 的复用、argmax tie 与零/无限处理，及实际接口要求的随机 generator 消耗次序和状态。改变当前全量预生成或对零 draft 行的 generator 调用，可能改变后续使用同一 generator 的结果；需先明确是否要求逐 seed、后续状态或仅输出分布等价，再相应证明，不能以“不影响本轮选中的候选”为充分理由删除 RNG 调用。块内只保留当前最大项本身不要求物化整行，但生成 q 的算法、随机流映射及并列次序必须相容；历史 kernel 的实际内部实现、该条件分支是否运行以及收益均未知。无新 NPU 工作、无目标代码修改。

## 已存温度与分支证据的限定（2026-10-30）

此次有界复核找到比“启动脚本未指定”更具体的**请求构造证据**：原始带 profiler 的 2026-10-28 采集客户端 `analysis/operator-ceiling-20261028/profile_client.py:21` 构造请求时显式 `temperature=0`；其 `diagnostic/runtime/profile-client-results/progress.json` 显示 measure1 的 profile start/stop 都返回 200，客户端稍后因 `KeyError(data_id)` 结束，保存的计数对应八个完成请求。这是客户端源码与进度记录，不是逐个已导出 task 的请求映射或服务器最终对象观测。后续独立的 2026-10-29 profiler 重采集客户端 `diagnostic-20261029/profile_client.py:34–36,70–89` 也构造 `temperature=0`；这次重采集及其任务不能与无 profiler 的性能基线等同。基线的冻结 `benchmark-20261028/raw-results/measure1.py:48`、`measure2.py:48` 分别配置 `temperature=0`，正式 payload 构造限定条件见 `agents/bounds/stage-101-local-benchmark-client-review.md`。

已存同镜像的 `completion_protocol.py:316–333,370–377` 仅当请求 temperature 为 `None` 才取服务默认值，非 `None` 的 0 传给 `SamplingParams.from_optional`；`sampling_params.py:27,533–537,781–786` 对低于 `1e-5` 的 temperature 设置 greedy 参数并将 `sampling_type` 归为 GREEDY。**条件推导**：如果历史实际使用这些源码和该请求体、中间未改写，且同一采样批的所有有效请求均为这种 greedy 请求，且运行时将其正确汇总为 `sampling_metadata.all_greedy=True`，则所存 Ascend `rejection_sampler.py:508–554` 的 greedy 分支返回，不会进入后面的 `:556–585` 非 all-greedy 恢复 q 路径。这个链条中的源码与输入相容，**没有直接观测**每个历史 batch 的成员、实际 `all_greedy`、环境及调用版本，因此不能断言 EP7/EP8 的指定任务已命中或完全没有命中恢复 q。

`all_greedy` 是 batch 条件：与随机请求混批、dummy 或其他来源的 metadata 不能仅凭八条客户端请求的温度排除。保存的 `metadata-path-snapshot/vllm_runner.py:6278–6281,6348–6371` 中 dummy sampler 构造 `temperature=0.5, all_greedy=False` 并在条件满足时调用 rejection sampler，随后还构造 all-greedy metadata；这是源码路径而非证据表明某个所选 profiler 事件是 dummy。另在已存 Ascend sampler `:481–491`，`synthetic_mode` 在非 all-random 的 greedy 路也会生成 uniform 随机数；因此即便已证明 `all_greedy=True`，也只能排除其后非 all-greedy 的 q 恢复路径，不能笼统断言没有任何随机操作。

限定搜索：只查了上述两份指定 profiler 客户端、两份基线测量配置、保存的服务日志尾部 `benchmark-20261028/raw-results/service-log-tail.txt` 与现有关联报告。该日志尾部在固定字符串 `Rejection sampling path`、`RejectionSampler config`、`all_greedy`、`all_random`、`has_triton` 下无命中；`debug_once` 本来就可能受日志级别与采集范围影响，此结果不证明日志或实际执行中不存在这些信息。EP7/EP8 已导出 `rejection_greedy_sample_triton` 仅证明命名 task 出现过，不绑定该 sampler、调用、request 或运行时 `HAS_TRITON`。真正闭合需同次请求/批次与目标 rejection task 关联、真实 sampling metadata、synthetic/verify 模式及加载文件/二进制；不重复扫描同名任务。源码 SHA-256：28 日客户端 `48b0111a80edd2d616d0d41e56384d57709b095196bce7ce5bf7a3bcf9fd7434`；29 日客户端 `e3528495c48ae02f114dc3943263ba67c6494f795b67db4bfef9079408d9c2a3`；协议 `5f7930635fd3924826015211032fb298206b27c7f04cd6cd2224bb128e4a3a50`；SamplingParams `f21efc98bf8ce8aab88eb67adb4207285e509a167d976e14d3e2e8282e024bea`；Ascend sampler `bb1d39de851540c16c2dd2b16df60c60a1fc4957cedb88bdce9417efc0d55eda`。本次没有新设备工作、没有修改推理代码。
