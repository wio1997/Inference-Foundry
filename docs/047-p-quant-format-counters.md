# 32K P：DynamicQuantV2 的记录格式与设备字段

## 对照范围

在 [P侧112组记录 shape/dtype 分布](046-p-shape-dtype-distributions.md)基础上，从原16份 P设备 CSV 追加只读提取输入/输出格式字符串及数值字段；16 worker 并行、未发推理。112组均可找到相同的记录格式字符串，逐组次数与此前分位数逐字节数值一致；无效或未知格式不当通配符。SGLang源对 G240 冻结 SHA/大小/mtime；vLLM源为 G148 派生 CSV，与先前扫描及本次双读一致而非历史冻结原始 CSV SHA。资源门核过，运行期采样 CPU 8.44%、16 worker 活跃；I/O PSI 不可用，不能宣称 I/O 未受影响。

这只把**已记录的格式**加入比较，不证明 strides、对齐、tensor 数据、执行层/步骤、实际工作量或时钟/频率相等；两个框架为不同 epoch，CANN9.1/post6 对 CANN9.2/post7。设备字段可能重叠，p50不能相加或相减构成路径拆解。

## DynamicQuantV2 观察

rank0 的完整记录字段为：输入 `8192,4096` / `DT_BF16` / `ND`；输出 `8192,4096;8192` / `INT8;FLOAT` / `ND;ND`。SG 120事件、vLLM 399事件，各自只在一个本地 stream；两个 stream ID 不可跨框架当同一条流。

|rank0设备 CSV 字段|SG p50|vLLM p50|性质|
|---|---:|---:|---|
|Duration(us)|122.0425|90.4020|单设备事件时长，不等于请求时间。|
|aiv_time(us)|117.0320|86.7340|CSV原始设备字段，版本语义需校验。|
|aiv_vec_time(us)|74.6100|74.3720|分组中位数接近，不等于两边全部计算相同。|
|aiv_mte2_time(us)|94.1335|15.4430|分组中位数差异较大；不是实测带宽饱和率。|
|aiv_mte3_time(us)|46.3195|10.8400|独立指标，可能与其他pipe重叠。|
|Wait Time(us)|2.2305|0.6190|原字段，不等于客户端排队。|
|Block Num|48|48|记录值相同不等于实际线程工作量相同。|

该 signature 的8 rank 先分别算 p50 再求中位数：SG / vLLM 的 `aiv_vec_time` 为约 **74.619 / 74.372µs**，`aiv_mte2_time` 为约 **95.191 / 15.839µs**，`aiv_mte3_time` 为约 **49.095 / 10.847µs**。同轮8卡不是8次独立试验；各指标分位数也不一定来自同一条事件。厂商 [msopprof 字段文档](https://github.com/Ascend/msopprof/blob/ce5a1158cebf0ba3b9f8a48793304660c22ece36/docs/zh/user_guide/msopprof_performance_data.md)将 MTE2/MTE3 描述为搬入/搬出指令相关时间；它不是本次两个 CANN 版本的逐字段校准。**只能将搬运相关字段列为优先待检线索，不能认定内存带宽瓶颈、工具版本贡献或 TTFT 根因。**

静态源码表明 SG 的普通 W8A8 线性层和 MoE 均有动态量化接口候选；vLLM 也有不同路径的动态量化候选。仅凭设备事件名不能确定当前调用来自哪一层、是MoE还是dense，或证明两边相同二进制实现。下一步应建立调用/forward链，采同版本同shape/format/stride/数据的算子试验，再验证端到端；不直接将小事件乘以次数推算收益。

## 数据与复核

[脱敏格式与指标分布](../evidence/p_formats_047/distributions.json.gz) 解压 SHA `9ae081169cd00b9cebaa9bf2b922232c1def10d71db44ef5562da79946e15b2f`，含8 rank各14对签名、设备字段有效/缺失数、stream和单事件分位数；不包含原始请求文本。原 [shape/dtype 数据](../evidence/p_joint_046/distributions.json)保留未覆盖及旧结果。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_p_format_047.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_archive.py
```

[离线检查](../scripts/verify_p_format_047.py)核对全部112组与旧统计一致、format和目标计数、指标覆盖及示例数字；不验证工具字段语义、推理正确性或因果收益。
