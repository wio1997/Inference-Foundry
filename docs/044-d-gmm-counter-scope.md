# D 侧 GMM：主 stream 的数值字段与解释边界

本页复用[已归档的 G240/G148 CSV 数值分布](043-d-gmm-ordinal-counters.md)，未重新运行模型或改写数据。32K/C16/1024，D 侧为单次**未计分**诊断；rank0 和其他 rank 有相同记录的物理字段，不意味着同一专家输入、有效 token、layer、请求或 step。

## 主 stream 中位数

每个 SG rank 每个记录的 GMM shape 均有 360 个事件，其中主 stream 各有336次。以下是主 stream **各字段分别计算的 p50**，单位按 CSV 字段名标记为 µs；单事件同一组字段之间也可能重叠，不可把中位数加减当墙钟。

|记录的输入缓冲|字段|rank0 p50|rank1–7 p50范围|
|---|---|---:|---:|
|8192×2048|aic_mac_time(us)|7.207|1.441–1.441|
|8192×2048|aic_mte1_time(us)|19.8915|3.980–3.986|
|8192×2048|aic_mte2_time(us)|51.911|7.298–8.5955|
|8192×2048|aicore_time(us)|61.1915|16.8435–19.091|
|8192×2048|Wait Time(us)|1.200|1.100–1.130|
|8192×4096|aic_mac_time(us)|14.414|2.883–2.883|
|8192×4096|aic_mte1_time(us)|39.803|7.965–7.971|
|8192×4096|aic_mte2_time(us)|104.0625|17.6315–19.1325|
|8192×4096|aicore_time(us)|116.9635|29.329–32.3715|
|8192×4096|Wait Time(us)|2.899|2.8435–2.889|

可说的是：在每组主 stream 中，rank0 的 `aic_mac_time` 和 `aic_mte1_time` 记录值也约为其他 rank 的五倍，差异不局限于总 Duration 或 CSV Wait 字段。**不可据此证明实际计算量是五倍、设备坏或计算/内存带宽成为系统瓶颈**：记录字段的工具语义尚未独立校准，且不同 rank 的专家 token 分布、group-list、权重和执行顺序未匹配。单组采集也没有设备 A/A 噪声验证。

## 值得优先获取的因果证据

已有归档的 [SGLang W8A8 MoE 源码快照](../evidence/profile_mechanism_G287/SG_source/sglang/srt/hardware_backend/npu/quantization/moe_methods.py)显示 `NPUW8A8Int8MoEMethod.apply` 把 `expert_tokens` 传给 [GroupedMatmul](../evidence/profile_mechanism_G287/SG_source/sglang/srt/hardware_backend/npu/moe/matmul.py)，后者作为 `group_list` 参数传给 `npu_grouped_matmul`。这只是静态源码线索；**未新验证该函数在这些 G240 事件上逐次命中，也未记录 group_list 的值**。先验证现场版本身份、同一层/step 对应与 actual expert token，再考虑是否存在工作量差异或设备效率差异。若要采集，仅保留按专家聚合计数/散列且不记录请求内容，并先验证图重放、同步及探针开销；未授权、不隔离时不执行新推理。

离线复算：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_d_gmm_044.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_archive.py
```

[核验脚本](../scripts/verify_d_gmm_044.py)固定源数据解压 SHA、16 rank、每个主 stream 的336次数和数值有效性；它不提供端到端收益上限或因果结论。
