# Target rejection：gather、转换和 raw 副本的条件性重组

范围：现有 169 冻结版本的 target rejection sampler，属于输出/验证路径；与 F30 的 **draft** Markov bias→argmax、F26 最终 HC collapse 属于不同边界。源码级发现，不声称历史请求实际执行到这一分支、可节省某字节的 HBM 访问或形成性能收益。

## 已核对的证据

- 两个既有容器当前文件逐字节一致：Ascend `sources/vllm-workspace/vllm-ascend/vllm_ascend/sample/rejection_sampler.py` SHA256 `bb1d39de851540c16c2dd2b16df60c60a1fc4957cedb88bdce9417efc0d55eda`；上游 `sources/vllm-workspace/vllm/vllm/v1/sample/rejection_sampler.py` SHA256 `0bb9d224887ccb1355e16199005cc23a4afc6fa75156eca419851b7247d99750`。这是本次只读两个容器当前文件比对，并不单独证明历史 trace 运行时加载的代码/二进制。
- Ascend `:183–218` 先 tensor-index 分别抽 bonus 与 target：该源码明确说明两个索引结果为新存储；target 转 FP32，再在 `not is_processed_logprobs_mode` 时 clone，随后原位/可能返回新对象地处理 logits。`:240–249` 按是否 processed 选择 target 或 raw 用于可选 logprobs。`:224–236` 无论模式都将 `raw_target_logits` 作为 `ori_target_logits` 传给 rejection；`:434–464` 若 enable_entropy_verify 则对该原始 logits 做 FP32 softmax，甚至该计算出现在后续 greedy 早返回之前；`:508–554` 按处理后的 logits 做 argmax 与 greedy verification。 `:94–144,321–375` 展示 penalty、allowed-token mask、bad words、logits processor、thinking budget 及 temperature/top-k/top-p 条件处理，可更改结果或产生状态可见的效果。
- 现有基线客户端记录 `agents/diagnosis/BASELINE-CLIENT-LATENCY-DISTRIBUTION.md:23` 两轮 generation kwargs 含 temperature=0、ignore_eos=true；这不是 target rejection 当次 `sampling_metadata.all_greedy`、处理器配置、logprobs、entropy_verify 或 tensor 对象身份的运行时捕获。冻结启动和客户端负载不能单独证明当次路径，已有 trace 也没有对此分析建立逐请求 task/对象绑定。

## 候选与保留条件

1. **条件多输出 gather+cast**：将 target 的按索引取行、FP32 转换和所需的 raw/processed 副本初始化合并为一次消费输入的逻辑操作，以两个独立输出或经证明只读的共享版本提供旧接口需要的值。原代码若输入已是 FP32，`.to(float32)` 也可能复用 gather 的对象；若其他 dtype 则通常转换，故不能把每个 Python 操作都断言为一次设备拷贝。index 顺序、重复行、边界、输入 dtype 到 FP32 转换、空行、结果 stride/contiguous、NaN/Inf、索引合法性和异常行为须保留；bonus 分支以及原 logits 原始存储不可被意外修改。原采样 API 与任何异步消费者可能持有上述对象，候选不能只凭 Python 源码同名断言地址或生存期安全。是否已有编译器/图优化折叠未知。
2. **原值不再有消费者时研究免 clone**：仅当这一具体调用所有 raw 值消费者均已排除、返回值和后续被观察的 alias/版本语义不受影响时，考虑让 processed 流从 target gather 的新存储原位处理，而不额外 clone。一个可检查的充分前提组合是 `max_num_logprobs is None`、`enable_entropy_verify=False`，且已核实当前源码里 `ori_target_logits` 没有其他隐式读者、hook/并发 observer，bonus 原始输入不因 alias 被改；其余 processors 仍逐一按原顺序执行。两标志只是必要的可见证据，不能用客户端没有传 logprobs 推定生效配置，也不能推定 graph 中 clone 的真实下发与耗时。
3. 若 raw logprobs 或 entropy 需要原值，必须保留原始 FP32 数值/读取时序；可以探索一次 gather 同时生成两个独立结果，但不保证减少物理读写或更快。若输入中的该行后续在其它流写入，或相同 index 多次出现，仍须保证冻结时刻、不同目的行各自修改的语义。

**严格否定不带条件的删 raw 副本**：限定 `not is_processed_logprobs_mode` 的现有 clone 分支，以一行原始 logits `[1,2]` 为例，allowed-token mask 把第二个词原位设成 `-inf`。原始行的 softmax 第二词概率是 `e²/(e+e²)>0`，改后对应概率为 0。若 `ori_target_logits` 错误别名为处理后的 tensor，`enable_entropy_verify=True` 时 `:461–462` 得到不同的原始概率；如果请求 raw logprobs，`:240–249` 也读取错误值。这个反例仅说明源码允许条件下错误，不宣称基线出现此 mask 或该次配置打开 entropy。在 processed mode，源码没有这次 clone，传入 `ori_target_logits` 的对象可能受到原位处理，不能声称所有模式下它都保留原值。原始 logits 的独立性还有并发、数值/状态契约，不能由数学近似代替。

## 尚缺

需要既有 trace 或服务日志中可关联同一个 request、真实 sampler 类/调用、resolved metadata 和 entropy/logprobs 配置、索引/源 shape、图节点与目标设备任务的证据；当前没有这些就不排序收益，不把本候选计入已确认热点，也不额外启动 NPU。若以后解除资源限制，仅针对这一缺口设计最小取证，并分别检查修改前后的全部可观察输出和 RNG 状态；本阶段不修改实现。
