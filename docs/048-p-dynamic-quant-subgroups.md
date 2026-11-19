# P侧动态量化：同记录签名内的快慢子群

## 新观察，先看限定

从 [相同记录shape/dtype/format的P诊断](047-p-quant-format-counters.md)出发，按照每个rank自己的事件开始时间顺序重读16份设备 CSV，并把与目标签名相同的事件按 **100µs** 切分。阈值是在观察已有范围后选择的描述性分界，不是预先注册的假设、真实阶段或算子语义分类。只读分析，没有新推理；SG源与 G240 冻结SHA/大小/mtime一致，vLLM G148派生CSV与既有扫描及本次双读SHA一致，不是原始冻结 CSV SHA。

目标是 `aclnnDynamicQuantV2_DynamicQuant_DynamicQuant`：输入 `8192,4096`、BF16/ND，输出 `8192,4096;8192`、INT8/FLOAT/ND;ND；shape/format字段相同不等于stride、数据、layer或step相同。两框架采集为不同epoch、不同CANN与torch_npu版本，rank不是独立重复。

|每rank单窗口数量|≤100µs|>100µs|
|---|---:|---:|
|SGLang|40 / 120|80 / 120|
|vLLM|399 / 399|0 / 399|

SGLang rank0 快组/慢组 Duration p50 分别 **89.032 / 123.0025µs**，在同一事件组内各自统计的 CSV `aiv_mte2_time` p50 分别 **14.573 / 94.6795µs**，`aiv_vec_time` p50 分别 **74.362 / 74.622µs**。这是**事件级联合分组**，比把各自边际p50相减更有信息；仍不证明MTE2导致慢时长，也不能将不同字段相减或相加。字段/版本语义未经校准，不推断带宽饱和。SG每rank按本地时间独立排序，5个等频段×24事件；vLLM分7段×57事件。SG rank0五段p50均约120–123µs，说明高分位现象不只局限一个序号段；各段不是跨rank、跨框架或同请求的同步步骤。

## 同一记录stream的相邻算子

以下是按**同一 CSV stream 的事件开始时间**排序后的直接前后邻居；不证明数据依赖、执行因果、层身份或调用栈。各rank的计数/相邻名称一致，但不能视为8次重复。

|框架与子群|每rank数|前一记录事件及输入shape|后一记录事件及输入shape|
|---|---:|---|---|
|SG ≤100µs|24|Add `8192,4096;4096`|QuantBatchMatmulV3 `8192,4096;96,256,16,32;3072;8192`|
|SG ≤100µs|16|Add `8192,4096;4096`|QuantBatchMatmulV3 `8192,4096;64,256,16,32;2048;8192`|
|SG >100µs|80|Add `1024,4096;4096`|QuantBatchMatmulV3 `8192,4096;64,256,16,32;2048;8192`|
|vLLM ≤100µs|315|TensorMove `1024,4096`|QuantBatchMatmulV3 `8192,4096;16,256,16,32;512`|
|vLLM ≤100µs|77|RmsNorm `1024,4096;4096`|QuantBatchMatmulV3 `8192,4096;64,256,16,32;2048;8192`|
|vLLM ≤100µs|7|RmsNorm `8192,4096;4096`|QuantBatchMatmulV3 `8192,4096;64,256,16,32;2048;8192`|

**因此此前单一总体p50混合了不同相邻执行上下文。** 同一后一算子记录shape下也出现SG快/慢组，但无法据此确定前一Add是量化输入的生产者或其shape决定慢因；其他stream可能有重叠事件。同stream本地开始/结束间隔不能理解成客户端等待、依赖链或端到端贡献。还需关联真实调用API、layer/forward、图重放、stride与张量内容，再以同CANN、同输入的可控实验和端到端验证。

## 可重算证据

- [分段和联合子群统计](../evidence/p_quant_048/temporal.json.gz) 解压 SHA `7bf47e03562583b86d925471b392872028654cbcb52fc2743c8f01515d917aae`。
- [同stream相邻上下文](../evidence/p_quant_048/neighbors.json.gz) 解压 SHA `5b1996a40250ddcfed8d6414517c86da29f1381826b872227fd4b6addcae4dc7`；各自只记录局部差值，不含请求文本或绝对时间。
- [离线验证](../scripts/verify_p_quant_048.py)核对16 rank、阈值分组、分段、邻居计数与示例数字；不声称因果、优化收益或输出正确性。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_p_quant_048.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_archive.py
```
