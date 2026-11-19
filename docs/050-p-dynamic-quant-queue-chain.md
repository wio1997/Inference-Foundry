# P 动态量化：CPU/队列/设备关联候选和阶段刻度

## 范围与等级

本页只针对 [目标 DynamicQuantV2 单个已记录 shape/dtype/format](048-p-dynamic-quant-subgroups.md) 的 SG 与 vLLM P rank0、两次独立32K非计分诊断；不跨框架匹配请求/step。持续时长和指标均是该窗口的局部事件统计，8卡不是独立实验；CANN/torch_npu版本、tensor内容/stride及实际工作量未配对。原trace整事件在授权主机内存中解析，只使用白名单字段作关联；发布的统计不含原始ID、绝对时间或请求内容。

[此前设备CSV→trace候选](049-p-quant-csv-trace-candidates.md)的复合键唯一后，在每个trace独立检验，不把不同ID命名空间混合：

1. SG 120/120、vLLM 399/399：每条设备候选的 `connection_id` 在本目标集均唯一；同名字段关联1个 `Node@launch`，其 `item_id` 等于设备kernel名称。
2. 每个节点的记录 `Thread Id` 与唯一 `dequeue` 的 `tid` 数值相同，节点时间区间被该 `dequeue` 区间包含；匹配 `dequeue` 名称包含DynamicQuant。它们的 `correlation_id` 在各自trace中分别关联唯一 `enqueue` 和唯一 `dequeue`。
3. 对该唯一`enqueue`，两框架全部候选均有一个 `enqueue_to_dequeue` s 端点位于同 PID/tid 的enqueue区间、一个 f 端点位于同 PID/tid 的dequeue区间。**这是记录的候选关系，不等于跨线程调度语义或请求因果已校准。**
4. 每个enqueue都被**多个**同PID/tid、名称含动态量化的 `cpu_op` 区间包含。按事后“最短包含区间”选出的标签为 `aclnnDynamicQuantV2`，同时也出现 `npu::npu_dynamic_quant` 标签；最短区间不是已证明的直接调用父子关系，trace标签亦不能单独证明当前runtime源码、模型层或tensor语义。

上面数值是目标事件加权的候选计数，不是所有量化事件或跨框架相同事件；只用 `args.connection_id` 查 CPU op 时0个直接匹配，**不能说没有CPU op**，因为候选链使用了不同的 `correlation_id` 与明确的 `s/f` 事件及时间包含。

## 队列标记刻度：不是TTFT拆分

下表按同一候选事件所属的事后Duration阈值分组，单位µs；三个量分别是 trace 上的 enqueue span、dequeue span、以及 `dequeue开始 − enqueue结束`。最后一列是两个局部标记的间隔，**不是NPU空闲时间、真实请求排队或可加到TTFT的损失**；可能与其他计算/通信重叠。两个框架采集工作量与epoch不同，不能直接相减为框架开销。

|P rank0窗口|n|enqueue span p50|dequeue span p50|标记间隔 p50|标记间隔 p90|负间隔|
|---|---:|---:|---:|---:|---:|---:|
|SG ≤100µs|40|11.67|19.50|4.775|6.791|0|
|SG >100µs|80|3.66|28.51|434.25|523.242|0|
|vLLM ≤100µs|399|8.59|22.74|3.14|67836.104|1|

SG慢组的局部标记间隔明显高于快组；vLLM此形状p50小但p90有长尾，且有1个负值，不能只报中位数或静默去掉负值。**这定位到量化候选的队列标记差异和设备事件差异，不证明谁导致谁，也不证明其解释32K端到端TTFT差。** 100µs为事后阈值，快慢组邻近算子上下文不同。

## 缺口和验证

下一步若要精确指出模型层和端到端根因，需能验证该enqueue所属logical layer/forward、真实输入布局/数据、request→batch映射和图capture/replay，且在同软件栈和同工作量下复测并验证质量与TTFT。现有SG capture `record_shapes=False`、`with_stack=False`；模型层级标签和逐请求映射未证实。禁止把其他trace的连接重数结果推到该目标样本之外，或把标记跨度直接相加为关键路径。

证据来自 [连接多重性](../evidence/p_flow_050/connection.json)、[节点/出队包含](../evidence/p_flow_050/nesting.json)、[CPU/flow候选](../evidence/p_flow_050/cpu_queue.json)、[白名单名称分类](../evidence/p_flow_050/cpu_names.json)、[队列时间分布](../evidence/p_flow_050/queue_timing.json)。源trace每次扫描均SHA/稳定性校验，SG为G240冻结源，vLLM为G148派生分析副本，不是原始冻结CSV/trace。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_p_flow_050.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_archive.py
```

[离线复核](../scripts/verify_p_flow_050.py)验证5份源SHA、每阶段分母、0/1/多候选与示例分位；不验证调用语义、输出质量或端到端收益。
