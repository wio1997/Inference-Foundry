# P/D 逐记录签名：分位方向与反例

## 适用范围和结论

本页从既有单次32K、C16、1024输出的 P/D 双框架**非计分诊断**脱敏摘要离线重算，不重新采集或推理。P侧14个签名各覆盖8个rank，每签名按精确算子名及记录的输入/输出shape、dtype、format配对；D侧15个签名每个覆盖8个rank，记录的输入/输出shape和dtype相同，但**format、stride未知**。一个rank内两框架事件数、数据、逻辑step、请求不配对；8卡属于同一次采集，不是8个独立重复。两框架epoch、CANN/torch_npu及服务配置不同，不能将分位差归因为框架或实现。

P DynamicQuant 的 `8192,4096` 签名是一个待验证线索，但**不是“DynamicQuant在SG普遍较慢”**：下表另外3个签名在该窗口的各rank p50方向相反。用户应优先查看单签名与完整分布，不应按名称概括所有形状。

## P 侧：同记录格式的4组量化

各行为rank0单事件分位和该rank样本数；右侧方向是8个rank分别计算的SG减vLLM p50，不是8次独立实验。所有四组输入BF16、ND，输出INT8/FLOAT、ND;ND。

|输入shape|SG n / p50 (µs)|vLLM n / p50 (µs)|SG n / p90 (µs)|vLLM n / p90 (µs)|8个rank p50方向|
|---|---:|---:|---:|---:|---|
|1024,4096|344 / 22.821|336 / 24.300|344 / 24.095|336 / 24.771|SG 8/8更低|
|8192,1536|120 / 41.231|84 / 42.311|120 / 42.013|84 / 42.581|SG 8/8更低|
|8192,2048|96 / 50.591|84 / 51.352|96 / 52.421|84 / 52.829|SG 8/8更低|
|8192,4096|120 / 122.043|399 / 90.402|120 / 124.290|399 / 90.901|SG 8/8更高|

`8192,4096` 里SG还混合快/慢组：每rank 40次≤100µs、80次>100µs；见[逐事件子群](048-p-dynamic-quant-subgroups.md)及[CPU候选链](051-p-quant-cpu-step.md)。阈值事后选择，不能把慢组直接解释为内存瓶颈。以上所有记录字段相同也不能证明实际buffer stride、数据、层/step或二进制实现相同，不能将分位差乘事件数推TTFT收益。

## P 侧：其他签名与分位方向反转

|记录签名|rank0 SG / vLLM n|rank0 p50 SG / vLLM (µs)|rank0 p90 SG / vLLM (µs)|8rank方向|
|---|---:|---:|---:|---|
|RmsNorm `8192,512;512`|96 / 84|28.811 / 24.141|30.280 / 28.524|p50、p90均SG 8/8更高|
|MatMulV2 `8192,128;1024,128`|544 / 476|24.520 / 20.660|30.661 / 26.131|p50、p90均SG 8/8更高|
|RmsNorm `8192,4096;4096`|96 / 28|140.943 / 147.373|156.423 / 148.775|p50 SG 8/8更低，p90 SG 8/8更高|
|RmsNorm `1024,4096;4096`|664 / 630|26.481 / 29.261|29.374 / 29.620|p50、p90均SG 8/8更低|

例如RmsNorm `8192,4096` 的逐rank p50差中位数为−15.005µs，而逐rank p90差中位数为+7.636µs。它们是**各rank差的中位数**，不是两组总体中位数相减；也不是置信区间。样本量不等与分布尾部差异使单个总体p50可能掩盖方向变化。

## D 侧：同记录shape/dtype但格式未配对

|记录签名|rank0 SG / vLLM n|rank0 p50 SG / vLLM (µs)|rank0 p90 SG / vLLM (µs)|8rank方向|
|---|---:|---:|---:|---|
|MoeGatingTopK `4,288;288`|8 / 360|5.220 / 4.680|5.346 / 5.580|p50 SG 8/8更高，p90 4/8更高|
|DynamicQuantV2 `4,2048`|8 / 112|3.010 / 3.121|3.132 / 3.680|p50、p90均SG 8/8更低|
|DynamicQuantV2 `4,4096`|8 / 496|3.900 / 3.960|3.946 / 4.161|p50、p90均SG 8/8更低|

D TopK每个SG rank只有8条，不能称统计稳定或外推每token耗时。D indexer和SparseAttention虽在各rank显示SG事件p50更高，但前者shape未知、后者记录shape不同，不属于本表的同签名配对；AllGather名称/字段也不足以同签名配对。不得把这些局部事件方向当作端到端根因。

## 数据与重算

- [P汇总](../evidence/p_signatures_055/summary.json) SHA `6e80be36fd8abf23b9e605de257dd8415f63c1fa65d621d1abb9ecfb35534dfd`；依据[原格式与字段分布](../evidence/p_formats_047/distributions.json.gz)及[格式范围](047-p-quant-format-counters.md)。
- [D汇总](../evidence/p_signatures_055/d_summary.json) SHA `2dc9153fefdc58e0c42362145dfc5183edef5d36ec56c6ad7daba9ccdabd9553`；依据[D匹配分布](../evidence/d_shape_040/matched_distributions.json)及[比较限制](040-d-physical-shape-distributions.md)。
- [P重算器](../scripts/summarize_p_signatures_055.py)、[D重算器](../scripts/summarize_d_signatures_055.py)仅消费已归档摘要，不输出stream号、请求ID或输入内容；重算不等于新测量或模型正确性验证。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/summarize_p_signatures_055.py --check
PYTHONDONTWRITEBYTECODE=1 python3 scripts/summarize_d_signatures_055.py --check
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_archive.py
```
