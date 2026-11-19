# GLM 32K D：GMM 按记录的物理 shape 分组

**这是单次 32K 输入、1024 输出、C16 未计分诊断的设备事件分布，不是跨框架 A/B、关键路径或根因。** 按完整 kernel 名、设备 CSV 中记录的输入/输出 shape 字面字段与 dtype 分组；各 rank 均来自同一次采集，不是独立重复试验。

## 方法与来源

从已核对的16份 D 端 SGLang G240 和 vLLM G148 派生 `kernel_details.csv` 只读提取目标 GMM，每框架各8 rank，16个 worker 并行。每 rank 选定的2个签名各360次、合计32组；每份源 CSV 的 SHA 和稳定性匹配原收据，逐签名实际事件数与[全kernel摘要](039-all-rank-kernel-distributions.md)一致。SGLang 对 G240 冻结 CSV SHA/大小/mtime；vLLM 对此前派生副本双读 SHA，**不是历史冻结原始 CSV SHA**。未发送推理；I/O PSI 不可用，不能宣称无 I/O 影响。

[脱敏分布](../evidence/d_gmm_042/distributions.json) SHA `e055132d8459602867055c2bbb8e37d3c8e661a96ae3338e1fcf0ff9e20d806b`，记录每签名次数、单事件分位数与 stream 分布。物理 shape 字段含空槽，不能从字段推断所有 tensor shape 或有效 expert token；`sum` 也不能当墙钟。

## SGLang D：分开两种记录形状

下表为同名普通 GroupedMatmul 单事件 p50，单位 µs；每个数字来自该 rank、该记录 shape 的 360 次事件：

|rank|输入缓冲 `8192,2048`|输入缓冲 `8192,4096`|
|---:|---:|---:|
|0|67.9315|126.0520|
|1|26.2200|42.2305|
|2|24.9980|39.8370|
|3|24.0400|37.7610|
|4|24.5805|38.9510|
|5|24.6810|37.6810|
|6|24.4600|37.2100|
|7|23.9410|37.8305|

rank0 在这两个记录字段各自的 p50 都高于其他 rank，而不只是把两种 shape 混合后造成的表象。每个签名的主体 stream 都有 336/360 次，但各 rank 的 expert group-list、有效行、层、并发与实际数值尚未配对；相同缓冲 shape/dtype 不意味着相同工作。当前只能标为**单窗口排查线索**，不能称设备故障或端到端瓶颈。

vLLM 普通 GMM 记录的输入缓冲首维为32，另有 GMM+SwiGLU+量化融合 kernel，事件类型、记录 shape 与 SG 不同。不能将其 p50 或累计时长直接相减解释 SG/vLLM 的 TTFT 差。

## 核验

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_d_gmm_042.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_archive.py
```

[核验脚本](../scripts/verify_d_gmm_042.py)核对16rank、32组计数、源 SHA/稳定性、时长排序及参考数值；不重新访问机器或证明请求级因果。后续需要按 layer/step、实际专家有效负载、group-list 与 stream 依赖建立配对，再做受控干预。
