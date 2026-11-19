# 32K P 侧：相同记录 shape / dtype 的事件分布

**范围：原有一次非计分 32K/C16/1024 Profile 的 P 侧，非端到端 A/B。** 从 [P 全量输入字段交集](041-p-recorded-shape-overlap.md)中选4个精确 kernel、每 rank 15个已记录相同输入 shape 字符串；16个只读 worker 并行重读两框架各8份 CSV，按实际输出 shape、输入和输出 dtype 分组，未事先假定这些字段相同。SGLang 的 CSV 与 G240 冻结 SHA/大小/mtime 一致；vLLM G148 是派生 CSV，与此前扫描和本次双读 SHA 一致，**不是历史冻结原始 CSV SHA**。未发推理，采集时 I/O PSI 不可用。

8 rank 共120个输入字段组，其中112个找到可用的相同输出 shape、输入 dtype 和输出 dtype 的联合记录字段对，其余8组无完整配对；每 rank 14 对。未知字段不充当通配符，不推断其为零。**输入/输出格式、stride、具体数据、算子调用角色、请求和逻辑 step 尚未匹配**，样本数不同且框架采用独立 epoch、不同 CANN/torch_npu 版本；以下不是单变量实现对比或收益归因。设备事件可并发重叠，累计时间不是墙钟。

## 观察：一个候选及一个反例

|rank0 P侧记录签名|SG单事件p50（次数）|vLLM单事件p50（次数）|解释|
|---|---:|---:|---|
|DynamicQuantV2，输入 `8192,4096` BF16，输出 `8192,4096;8192` INT8/FLOAT|122.0425µs（120）|90.4020µs（399）|本窗口约高35%；相同记录 shape/dtype 不代表同输入值、格式或同一步，不可外推 TTFT。|
|QuantBatchMatmulV3，输入 `8192,2048;128,128,16,32;4096;8192`|288.496µs（96）|286.646µs（84）|这一个签名的中位数接近，不能说所有 W8A8 矩阵乘都慢。|

DynamicQuantV2 的上述签名在两框架的8个 rank 均观察到。各 rank 先分别算 p50，再取8个 p50 的中位数：SG **123.306µs**，vLLM **91.072µs**。8卡是同一采集的 rank，不是8次独立实验；先取中位数再比较也不是置信区间或收益上限。

[脱敏逐签名统计](../evidence/p_joint_046/distributions.json) SHA `0fdc279b5181476aee41502ae43376440ac4b6d87810b754a86bd7628881d7a3`，保留两边次数、单事件p50/p90/p99和stream计数，以及全部120组覆盖统计。源 P 全量输入字段汇总 SHA `e1b4e1faef1c460d7688c65f1ea69f5177ae81594e3938d1ef9b2f0362aa27c7`。

## 还需什么才能指向原因

记录同一逻辑 forward 内的调用身份及完整 buffer format/stride、有效数据和运行时版本，比较同一 DynamicQuantV2 工作量，并核对它在请求关键路径上的次数/依赖。若仍有差，再在同版本可控环境中拆分算子实现、数据布局与输入分布；最后独立验证正确性及端到端 TTFT/吞吐。不可把每核 p50 差直接乘次数当可节省时间，也不应将此候选当作 32K/64K E2E 的已证根因。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_p_joint_046.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_archive.py
```

[核验脚本](../scripts/verify_p_joint_046.py)检查固定 SHA、8rank×15组覆盖、联合签名与分位数顺序及参考数字；不重跑模型，也不验证输出格式、数据内容或语义相等。
