# 输出头—跨卡词表—采样/拒绝：新增范围的语义审阅

状态：**局部条件合同已审阅，整体模块和运行绑定仍未闭合。** 这份记录补旧F26/F30的具体缺口，不把词表接口存在或形状相容当作该模块全部完成。

## 1. 本次实际补齐的依据

两批只读快照在 stage 容器核对 vLLM `ced6857…`、vLLM-Ascend `8fde6d4…` 且 tracked clean，保留 558 个去重源码文件。其Python源已建条件/调用底账，全部初始未审阅；C++只建文件索引。不存在的导入路径和运行对象绑定另外记录。

关键调用链：

1. 模型注册表把 `DeepseekV41ForCausalLM` 指到 **VL wrapper**，wrapper 的 `compute_logits` 再调用 language model；不能仅按架构名称跳过wrapper/vision条件。
2. language model 的 `compute_logits` 调用 `LogitsProcessor(lm_head, hidden_states)`。V1 worker在选择 `hidden_states[logits_indices]` 后生成logits，再通过 `_sample` 进入 `AscendSampler` 或 `AscendRejectionSampler`。
3. `utils.register_ascend_customop` 的注册表包含 `LogitsProcessor→AscendLogitsProcessor` 和 `ParallelLMHead→AscendParallelLMHead`。这是源码选路，实际实例仍须结合配置/运行证据确认。
4. Ascend logits processor 区分 REPLICATED、STANDARD、FINE_GRAINED、PCP_X_TP：普通TP重建词表；fine-grained先汇集hidden rows，再all-to-all为局部token恢复完整词表；PCP路径有两个组的词表gather。新方案不能套用一个全局TP组替代这些不同所有权。
5. 目标拒绝路径把bonus行与target行分别取出，后者转FP32并在特定模式下clone，然后经过processors、sampling constraints和greedy/random验证。bonus sampler强制请求整行logits作为内部结果，外部是否需要logprobs是另一条件。

源码入口（完整SHA以快照manifest及源级台账为准）：

- [模型注册表](sampler-source-supplement/frozen-source-snapshot/vllm-ascend/vllm_ascend/models/__init__.py)，`register_model`。
- [Ascend词表和logits覆写](sampler-source-supplement/frozen-source-snapshot/vllm-ascend/vllm_ascend/ops/vocab_parallel_embedding.py)，`_resolve_vocab_parallel_plan`、`AscendLogitsProcessor`、`lmhead_all_to_all`。
- [上游logits合同及局部选择接口](frozen-source-snapshot/vllm/vllm/model_executor/layers/logits_processor.py)，`forward/_get_logits/get_top_tokens/get_top_k_tokens`。
- [V1 worker](frozen-source-snapshot/vllm-ascend/vllm_ascend/worker/model_runner_v1.py)，`execute_model/_sample/_get_nans_in_logits`。
- [Ascend拒绝路径](sampler-source-supplement/frozen-source-snapshot/vllm-ascend/vllm_ascend/sample/rejection_sampler.py)，`AscendRejectionSampler.forward` 与模块级 `rejection_sample`。
- [Sampler](frozen-source-snapshot/vllm/vllm/v1/sample/sampler.py)，`forward/sample`；[Ascend greedy拒绝kernel](sampler-source-supplement/frozen-source-snapshot/vllm-ascend/vllm_ascend/ops/triton/reject_sample.py)。

## 2. 同次采集中的物化与计算族

读取原官方后续窗全部16rank CSV，并逐文件核对原SHA。下列是rank0的例子；16rank汇总另见[完整导出](head-sampler-exports.json)与[形状+dtype摘要](head-sampler-family-summary.json)。不同重复调用不是独立实验。

| rank0形状/类型相容组 | 次数 | 单条中位 µs | 能证明及不能证明 |
|---|---:|---:|---|
| BF16 MatMul `[24,5120]×[16160,5120]ᵀ` | 12 | 153.087 | 与目标头接口相容；权重/调用身份尚非直接张量绑定 |
| BF16 MatMul `[20,5120]×[16160,5120]ᵀ` | 12 | 185.827 | 与草稿基础词表接口相容 |
| BF16 MatMul `[4,256]×[129280,256]ᵀ` | 60 | 24.350 | 与复制Markov头相容；5倍次数与K=5相容 |
| BF16 Index `[24,129280]→[20,129280]` | 12 | 68.609 | 与验证行取出相容；尚未绑定同次Python索引对象 |
| FLOAT TensorMove `[20,129280]` | 12 | 12.710 | 与raw-target副本相容；不等于可节省12.710µs端到端 |
| FLOAT ArgMax `[20,129280]→[20]` | 12 | 19.739 | 验证argmax相容组 |
| FLOAT ArgMax `[4,129280]→[4]` | 12 | 16.980 | bonus相容组；不同于BF16 Markov ArgMax的60次 |
| `rejection_greedy_sample_triton`，输出 `[4,6]` | 12 | 2.760 | 已有前缀比较/拒绝复合任务，不能重新登记为尚未融合 |

同形状 `[4,129280]` 的 FLOAT 与 BF16 ArgMax属于两组。首次仅按shape选择时触发二义性断言，失败记录保留；修订后显式包含dtype，没有任意取一组。词表维度匹配也会选中Engram hash元数据，因此不能把所有匹配条目都算成词表头成本。

## 3. WS-VOCAB-01：词表生产、局部选择与跨卡通信重组

### 已有机制

上游 `LogitsProcessor.get_top_tokens` **已经**提供各rank局部max后汇总 `(value,global_index)` 的接口；`get_top_k_tokens` 也存在。它们不是本次新发明的kernel。可见的V4.1 model/V1 worker/proposer路径仍以 `compute_logits` 和完整logits为合同，是否存在动态替换/其他入口需要继续闭合。

### 具体候选与条件证明

在STANDARD TP、所有rank处理相同token行、有效词表被不重叠分片完整覆盖、所有会影响argmax的processor都已等价作用、完整logits没有其它必需消费者的条件下，可研究让目标/草稿头交付局部最优与必要状态，再做全局归约供greedy/验证消费，减少完整词表重建及中间物化。

有限数值下，若原argmax可表示为一致的 `(score, tie_order)` 全序，则每个分片保留局部最优，再对各局部最优归约，等价于对完整词表归约。原实现的tie/NaN规则必须保持，不能自行指定另一规则。已存在接口用FP32携带global ID；129280词表的整数ID在其精确整数范围内，但任何padding、附加词表、LoRA或不同重排都需另审。

**没有证据支持**先对原始logits取少量候选，再只给这些候选加会改变排名的bias/penalty。例：`L=[10,9]`、`bias=[0,2]`，基础top1与处理后top1不同。应将所需处理放在局部选择之前，或证明被排除词项的上界；不能直接套现有get_top_tokens绕过采样处理。

fine-grained/PCP模式的token行和通信组不同，不能直接使用STANDARD方案。还需保留bonus行、target验证行、采样输出广播及下一轮request/token对应。随机拒绝、熵验证、全logprobs、指定token logprobs、自定义processor以及NaN诊断，都可能需要更多信息，必须逐项处理；不能因为temperature=0就删掉全部logits消费者。

逻辑信息量仅作条件账：R行、TP=P、词表V的完整logits有 `R×V` 个分数；局部最优对有 `2×R×P` 个标量。它不是HCCL实际字节/必要通信下界；当前gather/all-to-all、dtype和复制策略不同，不给通信提速倍率。头矩阵计算本身未因此减少。

状态：**CONDITIONAL，实际分片模式、完整消费者和端到端暴露未闭合。** 对应M16/M17/M20/M23，跨算子及跨卡边界；与旧F26/F30有重叠。

## 4. WS-REJECT-01：拒绝路径的原值副本与bonus物化

### 4.1 target FP32 clone 的局部消费者审查

`AscendRejectionSampler.forward` 中：

1. `raw_target_logits = logits[target_logits_indices]` 使用高级索引，产生与原logits分离的存储；随后转FP32。此行为与[PyTorch 2.10索引/view合同](https://docs.pytorch.org/docs/2.10/tensor_view.html)一致，不能把它误当作普通切片view。
2. 在非processed-logprobs模式下 `target_logits=raw_target_logits.clone()`，保留raw副本；processed分支本来就不clone。
3. raw副本的后续源内消费者为：传给 `rejection_sample(..., ori_target_logits=raw_target_logits)`，以及条件logprobs提取。模块级rejection函数仅在entropy开且原logits非None时对该参数softmax；后续entropy分支消费该结果。
4. `_get_logprobs_tensors` 只在 `sampling_metadata.max_num_logprobs is not None` 时执行。

因此，在**entropy关闭、max_num_logprobs为None、普通推理tensor且无额外可观察alias/回调**的合同下，raw值保留不再被该函数的后续数值路径使用。可以登记“让processor直接使用已独立取出的FP32 target存储”的条件性重组。它保留原完整logits，改变的是内部raw快照；没有改变processor、温度、拒绝、RNG或输出顺序的数学必要性。

这是函数局部的条件消费者论证，**不是已证明本次运行满足全部guard**，也不覆盖entropy/logprobs开启的其它分支。若原值在任何扩展consumer中可见，必须保留或等价重算。没有实现或改目标代码。

### 4.2 bonus采样的内部整行结果

拒绝forward调用bonus sampler时强制 `max_num_logprobs=-1`，以raw/processed logits模式取得整行结果。基础Sampler在raw_logits模式下会保留FP32副本；该结果仅在外层条件logprobs分支被读取，bonus token IDs则始终需要。若外层不请求logprobs且没有指定token概率或扩展消费者，可研究只交付bonus token和真正需要的状态，消除强制整行结果。这与target clone是两个不同物化点，不能用同一条TensorMove记录重复计收益。

条件逻辑账：N个验证行、R个bonus行、V词表，FP32副本分别含 `4NV`、`4RV` 字节。仅当实际独立copy读写成立时，对应copy的逻辑读写是两倍；这些不是实测HBM流量，也不是时间下界。源码里的两次 `.to(float32)` 是否实际产生两个独立转换，须结合具体输入dtype、buffer及编译路径验证；不能据调用次数直接算流量。

状态：**CONDITIONAL_LOCAL_CONSUMER_REVIEW**。target clone的条件原值消费者已更具体；runtime guard、接口分派和实际copy身份仍待证。该候选不要求全部greedy，适用域与WS-VOCAB-01不同。

## 5. 已有greedy验证与跨迭代约束

Ascend kernel已按请求比较draft token与target argmax，在第一次不匹配处写target token并停止接受后续draft；全接受时追加bonus。它是已有融合边界。进一步组合argmax和prefix比较时，仍须保留完整词表选择、每请求草稿长度、padding/sentinel、synthetic路径及输出 `[R,K+1]` 的未写区域初始化。

这些输出决定accepted计数、KV状态、下次query位置及DSpark seed。不能把下一轮输入准备任意提前到接受结果产生前；可以提前的、与结果无关的部分需单独列出状态版本和最后读/首次写事件。该跨迭代合同尚未完整绑定，不能据本次词表审阅关闭M24。

## 6. 余下的关键证据

- 当前服务未公布 `/server_info`，只读OpenAPI检查已记录；不假定已读取resolved worker对象。
- 需核实实际vocab plan、LP/LMHead实例、sampler guard及所有非默认processor；支持的源码/启动配置推导与直接观测分别标注。
- 需把同次head→gather/transpose→bonus/target索引→cast/copy→argmax/reject按事件/调用身份绑定；当前完整shape+dtype及12/60计数仅是较强相容证据。
- DSpark引用的 `vllm.models.qwen3_dspark` 在所查新路径不存在，但tracked legacy `model_executor/models/qwen3_dspark.py` 存在。导入重定向机制还未闭合，不把“找到legacy文件”当运行对象证据。
- 仍需head quant方法、实际layout/scale、真实通信量与端到端暴露；不得把局部kernel时长相加为保证收益。

**结论：**这一范围新增两个具体条件性重组，并补到了此前缺少的Ascend LP覆写、bonus/raw副本消费者及实际词表形状族。M16/M17/M20/M23/M24仍有未分析/未绑定范围，整体目标继续active。
