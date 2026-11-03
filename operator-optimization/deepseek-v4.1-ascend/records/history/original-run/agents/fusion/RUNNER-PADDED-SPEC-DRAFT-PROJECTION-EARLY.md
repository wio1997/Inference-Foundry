# target 采样与 DSpark main projection 的 padded-batch 调度边界

## 结论和范围

冻结 runner 中，**padded speculative batch 的 target hidden 行选择本身不依赖接收/拒绝结果**：它保留所有输入行，将被拒绝的行当 padding，另外计算应采样的行和 rejected 数。这提供一个明确的条件调度候选：当 DSpark 真要运行且 aux hidden 已完整就绪时，可研究提前执行相同行集合的 aux concat → main projection → norm，并与 target `_sample`/验证并行；draft seed、被拒 token 数和由其派生的 metadata 仍须等采样结果。此候选与 F27 的“按 target 各层出现顺序提前部分 GEMM”不同：这里保留一个完整输入投影的逻辑 shape，移动它相对 sampler 的执行时点；两者重合时不能重复算收益。

以下只读 `agents/bounds/metadata-path-snapshot/ascend_runner.py`、`sources/.../spec_decode/{llm_base_proposer,dspark_proposer}.py` 和已有记录；没有设备操作、实现或真实 event 绑定，也没有证明耗时收益或性能上界。实际历史是否选择 padded-batch、K、采样、input fits、进程源码与这些快照一致均未绑定。

## 直接源码依赖

1. `ascend_runner.py:2785–2813` 的 `sample_tokens` 首先从已完成的 `execute_model_state` 取出 `logits`、`spec_decode_metadata`、common metadata、`aux_hidden_states` 等，再按可选 grammar 处理 logits，然后 `_sample(logits,spec_decode_metadata)`。随后判断 draft 输入是否可用，安排 `propose_draft_token_ids`。非 PP early 分支在 `_bookkeeping_sync` 之后以 GPU 的 sampler token 调用 padded drafter；`early_pp_padded_drafter` 有不同路径，源码注释提示 padded 模式输入不必等待 CPU bookkeeping，但**实际代码调度仍应以对应路径为准**。
2. `ascend_runner.py:2010–2040,2063–2099`：padded batch 从 sampled tokens 计算 `next_token_ids` 和 `valid_sampled_tokens_count`；`spec_decode_metadata is None` 时，按 `num_scheduled_tokens` 切 aux 列表后拼接。存在 spec metadata 时调用 `prepare_inputs_padded`，再按返回 `token_indices` 选择 aux 各项并 concat，然后交 `_propose`。
3. `llm_base_proposer.py:2389–2444,2488`：padded helper 先依 sampled count、每请求 draft 数和 query_start 计算 `token_indices_to_sample`、`num_rejected_tokens_gpu`，**但** `token_indices = self.arange[:total_num_tokens]`，其中 `total_num_tokens = query_start_loc_cpu[-1].item()`；文档明确保留 rejected 输入、稍后过滤。这是与 sampled count 无关的顺序 `0..T-1` 行集合（前提是原 arange、aux 切片和 scheduler 顺序契约成立），而不是将每个请求只保留已接受行。由此在 padded 分支可先取相同 T 行并执行完整 concat/projection，数值上无需因 accepted 数未知而多算原逻辑未选的行；直接 slice 与索引可能改变 copy/alias、连续性、内核选择和输出地址，行为等价仍须证明。
4. `llm_base_proposer.py:1037–1078` 对 `K=0` 提前返回；非零时 `combine_hidden_states` 后才 `set_inputs_first_pass`。`models/.../dspark.py:192–203,281–282` 的 combine 为 `ColumnParallelLinear(...gather_output=True)` 和完整 hidden 宽 RMSNorm。`dspark_proposer.py:243–259,270–318` 仍要用真实 next token、rejected 数、target positions、block table 与 group slot 信息，写 context hidden buffer，扩展 query/slot 并修正 seq_lens；不能仅因为 hidden 投影可前移就将完整 draft forward 推到采样前。
5. `ascend_runner.py:1876–1896` 已有 accepted count D2H 的辅助流 copy：辅助流先等待产生 count 的 default stream，非阻塞 copy 后记录事件；async spec 条件下还保存 GPU count。它是已有 overlap，不能把它记录为新提出的优化；代码与注释不直接证明 copy 恰好隐藏或适用于本样本。

## 正确性、图和并行限制

- **guard 必须与原流程相同**：需先确定 `drafter` 实际为 DSpark、辅助状态输出已配置、`input_fits_in_drafter`、scheduled K 非零、正确的 PP/DP/TP 分支及对应 `_propose` 真会执行。空请求、skip drafting、dummy/profile、取消请求和异常处理不可因提前计算而产生额外可观察副作用；是否有守卫能在 `_sample` 前稳定决定需核实。
- **不能推测接受结果**：例如一请求原有 3 个 draft、有效 sampled count 为 2，源码 fallback 的 rejected 为 `3+1-2=2`，`token_indices_to_sample = query_start_loc[1]-1-2`；假设全部接受会错选 2 行并使下游 effective seq_lens/slot 不同。非 padded 模式调用 `prepare_inputs`，其选行依赖 CPU sampled tokens，不能沿用上述 arange 结论。多请求、padding、discard 和 `spec_decode_metadata is None` 必须分开核对。
- **数据版本与跨流事件**：aux 列表均需同一 batch/轮次、正确 PP 最终阶段、各个行与 scheduled position 一致，明确 target 写完成及 aux 缓冲直到提前投影读完不可被下一轮复用。main projection 的输出必须一直保留至对应 `set_inputs_first_pass` 接收的行；不能用下一轮覆盖的稳定 buffer 或重新排序混合请求。采样/拒绝结果、CPU bookkeeping、graph 和 target 缓存可能与下一步并行，所有参与 stream 的事件必须保证其各自的正确版本。
- **数值与通信**：最稳妥候选是在相同行数和行序上调用原 concat、main_proj、norm，仍须验证索引转 slice 的地址/stride、量化或 GEMM 的 shape 与 dtype、输出舍入、norm epsilon、异步上下文及图捕获完全一致。不能拿 F27 把权重分块重排后的实数等价当逐位证明。`gather_output=True` 可能调用 TP collective：所有参与 rank 的 collective 顺序和 stream 必须与 target/logits 侧通信相容，不能让不同 rank 因采样/guard 差异走不同 collective 顺序；可见顺序与 event 需按实际运行实例验证。
- **没有时间结论**：sampler 与 main projection 都可能占用 Cube、Vector、MTE、通信或 host 执行资源；事件和 TP 通信即使数据独立也可能抢资源。现有 stage172 的 concat/projection 相容形状无法标明实际调用/请求或对应 sampler，也没提供两个流和关键路径证据，因此不能据其计算收益。需要先定位同一请求/轮次的真实路径、有效 token 和数据依赖，缺证保持 UNKNOWN，不额外做无目的测试。

## 收紧：采样前哪些 guard 能从已保存源码确定

本次只复核静态控制流；下列是源码条件，不是对该历史服务执行分支的断言。

1. **可取得的候选检查时点**：`ascend_runner.py:2629–2677` target forward 后将 aux、`scheduler_output`、spec metadata 和 common metadata 一起放入 `execute_model_state`，随后还执行可能的 `deferred_state_corrections_fn`，`execute_model` 返回后才进入 `sample_tokens`。在 `sample_tokens:2785–2813` 解包 state、完成已有 deferred correction 之后、`_sample` 之前，Python 级可取得这些对象；但 `_model_forward` 返回或 Python 解包只说明提交次序，跨 stream 前移投影仍要基于完成事件保护设备写与 alias 生命期。若在 forward 返回**之前**更早发射，则不能忽略尚未执行的 deferred correction 和 aux 就绪，不属于已证范围。
2. **K 已在 sampler 前确定，但不得用 K 替换现有 fits 条件**：`vllm_scheduler.py:1391–1396,1427` 将本轮 `num_spec_tokens_to_schedule` 写入 scheduler output，state 解包时已有；`_propose:1046–1056` 为零时跳过 combine。`ascend_runner.py:2680–2701` 的现有 `_input_fits_in_drafter` 则使用 `common_attn_metadata.max_seq_len + self._num_drafter_query_tokens() <= effective_drafter_max_model_len`，其中 DSpark 取实例 `num_query_per_req`；`dspark_proposer.py:39–43` 在构造时按 `sample_from_anchor` 和该时配置决定此宽度。假设实例宽度=5、本轮调度 K=1、max_seq_len=96、limit=100，原条件 `96+5<=100` 为 false；若错用本轮 K 判断成 `96+1<=100` 就错误执行。该例是源码条件反例，不表示本样本这些值出现。
3. **fits=true 不能单独当“本 rank 有 drafter”**：当 common metadata 为 None，fits false；但 drafter 为 None 时，函数**明确返回 true**，注释称非末 PP rank 不应用局部 drafter 限制。故非末 PP rank 的 `True` 并不授权其调用 `combine_hidden_states`；还需确认实际拥有 DSpark drafter、相应 PP rank、aux 列表及 TP group，并在所有 rank 上遵守匹配的 collective 次序。`ascend_runner.py:751–765,4427–4444` 及 F27 已说明 aux 是否启用与 PP 模式相关；本机快照并不证明 169 历史 runtime PP/aux 对象身份。
4. **T 的来源有两条路径，不能混用**：spec metadata 存在且 padded 时，`prepare_inputs_padded` 令 `T=common.query_start_loc_cpu[-1]`、`token_indices=arange[:T]`，数值独立当前 sampled count；spec metadata 为 None 时，runner 直接 `h[:scheduler_output.total_num_scheduled_tokens]`。两者是否始终相等未证明，前移时必须按实际分支取 T；还需 T 在全部 aux/ arange 界内、行排列及当轮 metadata 之后不变。这里证明的是值和索引的静态表达式，不证明原运行是否实际走这一分支、arange buffer 在那一时刻没有被外部写入、或 tensor storage 等价。
5. **K=0 时已有 concat 与无 projection 要区分**：runner 在调用 `_propose` **之前**即选择和 concat aux，因此当前源码的某些 K=0 分支仍可能完成上述 concat；K=0 的早返回只说明 `_propose` 内的 `combine_hidden_states` 不执行。不能把“原路径什么工作都没有”当反例，也不能提前执行原本不会执行的 main projection。
6. **bookkeeping 有真实状态写**：`_sample:2973–2991` 调用 `input_batch.update_async_output_token_ids()`；`_bookkeeping_sync:3028–3120` 会回退被丢弃请求的 RNG offset、解析采样、更新 `input_batch.num_tokens_no_spec` 和每请求输出。随后 `prepare_next_token_ids_padded:2204–2220` 的 backup token 读取正使用这些 request/input batch 状态。虽然 aux 投影表达式未显式读取 sampled token 或 backup token，**不能把整段 propose/next-token 准备一起移到 `_bookkeeping_sync` 之前**；不同 PP 路径调用时序还需分别保留。`_sample` 和 bookkeeping 没有直接接收 common metadata，但其内部/外部共享对象与 hook 是否会改变 `.max_seq_len`、`query_start_loc_cpu` 或 aux alias，单靠函数签名不能排除。要前移 fits 的结果，需证明其读取的各对象在原检查点与前移检查点一致，不能把静态可能性写成已证无条件安全。

明确尚缺：历史运行的 `disable_padded_drafter_batch`、`input_fits` 值、K、spec metadata None 与否、aux 的 storage 地址/版本、同一 TP group 所有 rank 的执行顺序，以及 cross-stream ready/final-consumer event。目前这些均不能从 profile 的 kernel shape 还原；上述候选尚不具备可直接落地的完整证据。
