# GLM 32K D：GMM 序列分段与 CSV 计数器

**范围：同一份 SGLang G240 32K/C16/1024 未计分诊断，D 侧八个 rank；不是相同 step 的逐卡实验，也不是性能根因。** 本文在[逐 rank 记录 shape 分布](042-d-gmm-by-recorded-shape.md)上，分别检查高耗时是否只集中于部分事件，以及 profiler CSV 中的原始数值字段。

## 独立序列分段

对每个 rank 的两种普通 GroupedMatmul 记录输入缓冲字段分别按该 rank 的事件 `Start Time(us)` 升序，360 次分成6段、每段60次。各 rank 和 shape 独立排序，**段号不是跨 rank 同步时刻、同一层或同一请求**；完整字段含空 tensor 槽，不能推断专家有效负载。冻结 CSV 的 SHA/大小/mtime 和8 rank×2 shape×360事件均核对。8 rank 同一次诊断，不是8次独立重复。

|记录输入缓冲|rank0 六段 p50 范围|rank1–7 各自六段 p50 范围|
|---|---:|---:|
|`8192,2048`|65.862–70.051µs|21.561–27.661µs|
|`8192,4096`|123.052–129.452µs|33.261–46.127µs|

对于每个 shape、每个序列段，rank0 的 p50 和 p90 均高于 rank1–7 的对应独立序列段（每项42/42比较）。这支持“高分布遍布该次采集序列而非集中在某一段”的观察；它**不排除不同 expert group-list/活跃 token、层、并发、调度和图语义**，不能据此认定卡故障、时间同步、关键路径或 TTFT 根因。

## profiler 原始字段

另一次只读解析16份 D 侧两框架 CSV，为逐签名统计设备 CSV 的数值字段，计数与源摘要一致。以 SGLang `8192,2048` 为例，每 rank 对应360事件：rank0 的 `aicore_time(us)` p50=60.3285µs，rank2–7 约17.1925–18.4635µs；CSV `Wait Time(us)` p50 分别约1.200µs与1.100–1.120µs。两组 `Block Num` p50 均为24。`aic_mte2_ratio`、`cube_utilization(%)` 等字段也有值，可从[脱敏汇总](../evidence/d_gmm_043/counters.json.gz)查看。

这些是 CSV **原字段的单事件分布**；字段单位/计数器语义尚未对照同版工具实测校准，单独字段分位数不能直接相减构成纯计算/排队时间。不能只依据 `aic_mte2_ratio` 高断定 HBM 带宽瓶颈、硬件故障或端到端差距。两次读取均只读原始 CSV、不发模型请求；现场 I/O PSI 不可用，分段工作太快未取得运行期资源样本，不能声称已测全程 I/O/CPU 峰值。

## 来源与复核

- [时间序列统计](../evidence/d_gmm_043/temporal.json.gz)解压 SHA `53a1cb239d6d9ae56380bc2c802bc7d878d4d633e17c7749ee05bdfccb9bc6f0`；仅保留每段相对偏移和分位，未发布原始绝对时钟或请求文本。
- [计数器统计](../evidence/d_gmm_043/counters.json.gz)解压 SHA `e50dead51a02affaff6870e9a4eb6a64896cb290dc0567f8957d93828e78f811`；每框架8 rank、每 rank 2签名，缺失数值记录为 unavailable 而不是0。vLLM 源为既有派生 CSV 与先前双读一致，非历史冻结 CSV 索引。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_d_gmm_043.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_archive.py
```

[离线校验](../scripts/verify_d_gmm_043.py)核对8 rank、96个序列段、计数器来源、覆盖数及上述数值；它不连接设备、不重跑模型。下一步需同一逻辑 step 的 expert group-list、有效 token、层与相应 stream 依赖，才可区分工作量和设备执行差异，并用受控 A/B 判定收益。
