# 32K P 重型算子：逐 CPU step 标记候选分布

## 结论先行

对 SGLang G240 与 vLLM G148 各自 P rank 0 设备CSV和同源trace，分别选取11、10种完整设备事件名，使用独立唯一的 `设备CSV复合键 → Node@launch connection → dequeue线程/区间 → enqueue correlation和s/f端点 → 同PID/tid CPU step区间` 候选检查；所有零个或多个候选保留为未知，不按临近时间强行匹配。两框架采集窗口、CANN和运行配置不同，step或请求没有配对。

**SGLang rank 0单次诊断中，每个EXTEND标记的关联CPU投递均含KDA 34次、SparseFlashAttention 11次、indexer 11次、MoE dispatch 42次、普通GMM 84次以及SwiGLU量化42次。** 每个DRAFT标记分别含0、1、1、1、2、1次。这是所选设备名经候选链关联的CPU投递次数，不是层号、设备关键路径、逐请求耗时或因果证明。设备事件可能在标记外完成；AICPU、NPU和不同stream事件可重叠。

模型静态配置是45层（34 linear attention、11 sparse attention；3 dense、42 sparse），与该EXTEND的34、11、42次观测**数值一致，但不是单事件到层的证明**。已归档静态配置在[模型身份](../code/glm53_g32/nextn_checkpoint_identity.json)；模型源码按layer分派是静态候选，图重放和layer_id→事件关系未实证。NextN是独立draft路径，不能把DRAFT一条事件直接说成target第46层。

## Indexer在本窗口随记录的 c_sk 增长

选取8个 EXTEND，均记录 `bs=1,toks=8192`；各组中有11个同名indexer目标事件。`p50`为组内排序线性插值，两个值是**同一次诊断中两个标记**，不是独立实验。`c_sk`是标记中的CPU侧长度聚合字段，不是物理KV占用实测。

|记录 c_sk|标记内11个indexer设备事件p50，两次（ms）|同标记内11个SparseAttention p50，两次（ms）|同标记内34个KDA p50，两次（ms）|
|---:|---:|---:|---:|
|8192|3.078 / 3.087|24.180 / 24.162|4.023 / 3.992|
|16384|10.838 / 10.832|24.292 / 24.342|3.999 / 4.005|
|24576|13.926 / 13.924|24.350 / 24.399|4.000 / 3.997|
|32768|21.672 / 21.679|24.390 / 24.372|4.004 / 3.994|

这是**同一诊断窗口内部的条件观察**：所选indexer事件p50随标记 `c_sk` 从8192至32768明显增大，同表中所选SparseAttention和KDA p50相对稳定。条件和设备输入形状随上下文变化，不是单变量消融，不可将7倍变化定性为算法复杂度或端到端因果。EXTEND CPU marker范围约1363–1606ms，部分时间可等待或异步；不能把其差值判给indexer，也不能直接加设备事件总时长。

## 双框架边界

|选定设备名类别|SG rank0 CPU标记分组|vLLM rank0同类名字事件|说明|
|---|---:|---:|---|
|SparseFlashAttention|88 EXTEND + 8 DRAFT|112|设备shape与工作量未配对|
|lightning indexer score|88 EXTEND + 8 DRAFT|96|真实输入shape不同，单事件变化不等于框架因果|
|KDA主核|272 EXTEND + 0 DRAFT|272|两框架核名不同，SG设备shape缺失；vLLM另有transpose事件|
|普通 GMM|672 EXTEND + 16 DRAFT|360|两边shape和路由数据不一致|
|融合 GMM+SwiGLU+quant|SG该名称未选|360|vLLM发生了融合名称事件；不能推断同层/同token比较|

vLLM所选10个事件名里，未出现SG式 `step[...]` 标记，不得将对应设备算子赋值为零或伪配SG的EXTEND/DRAFT。四条vLLM事件无法在Node时间范围找到唯一dequeue：DequantSwigluQuant 1、DynamicQuantV2 1、QuantMatmul 2；这些保留未知。**没有相同forward/层/请求配对，所以不能据本表把32K正式TTFT差额归因某算子。**

## 可复核材料

[阶段覆盖](../evidence/p_heavy_052/stage.json)和[逐标记算子统计](../evidence/p_heavy_052/step_detail.json)为只读脱敏聚合；SG G240源按冻结索引SHA/大小/mtime验证，vLLM G148派生CSV/trace与先前及本轮分析回执一致，但不是已认证的历史冻结原始CSV。分析两次，每次两worker并行、idle/resource gate、低优先级、不发推理请求；第二次最大采样CPU约1.24%，I/O PSI不可用不等于无影响。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_p_heavy_052.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_archive.py
```

[复核脚本](../scripts/verify_p_heavy_052.py)核对两次SHA、候选分母和16个CPU标记与[量化分析](051-p-quant-cpu-step.md)的一致性及每标记算子计数。它不提供请求配对、层映射、调度因果或可复现性能提升。
