# GLM 32K：P/D 两框架全 rank 设备 kernel 分布

本条记录现有 32K 输入、1024 输出、C16 的**单次非计分诊断 Profile**的设备 CSV 全名聚合；它不是新的性能实验，也不构成同请求、同逻辑 step 的因果归因。两框架 P/D 各 8 rank，共 32 份 CSV；各 rank 的设备事件时长可跨 stream 重叠，汇总 `sum` 不能当请求或端到端墙钟。

## 覆盖、来源与复算

|侧|rank|原 CSV 记录行数|rank × 精确 kernel 名分组|来源边界|
|---|---:|---:|---:|---|
|P|16|755447|31890|SGLang 对 G240 冻结索引 SHA/大小/mtime；vLLM G148 派生 CSV 对先前扫描与两次读取，**无历史冻结 CSV SHA**。|
|D|16|556800|15040|SGLang 对 G240 冻结索引；vLLM G148 派生 CSV 对此前双读收据，**不是原始二进制或冻结 CSV 索引**。|

[P 侧聚合 gzip](../evidence/all_ops_038/p_all_kernels.json.gz)解压 SHA 为 `3b5a99bf6882e2c8c13f1c9e5540528ccff429758e2509117ad5d9b2430b6622`；[D 侧脱敏聚合 gzip](../evidence/all_ops_038/d_all_kernels_sanitized.json.gz)解压 SHA 为 `596ce5ef9a04d8c9f6fce073dcfe15d4a40e5ee96c92380cacdfe8ea192a65e8`。D 原始本地聚合 SHA 为 `0003b1e4be8cadb2a097c1489318cff5b5780a84303a35f3a8dd956bd4230732`；脱敏副本将非白名单 dtype 整项归为 `unknown` 并合并计数，**不能从脱敏副本逐字节还原原始 D 聚合**。固定 `unknown` 是缺失 stream/type 哨兵，不是原始非数字身份。

每组包含次数、单事件 duration 的 min/p50/p90/p99/max 与累计和，分位数按 `(n−1)×p` 线性插值；shape 表示设备 CSV 所记录的物理缓冲区维度，不等于有效 token 数。P dtype 计数合并输入和输出列，可能超过事件数；D 的输入、输出 dtype 分列。D 同时提供 stream 与 shape/dtype/stream 组合计数。离线检查各 rank 的分组计数回加等于 CSV 行数，并验证 D 组合与边际计数守恒。

D 第三次采集的远端数据与本地结果曾已写出，但外层后处理在计算输出 SHA 时报错退出；此后对现有文件、全部 16 rank、源 SHA 与计数做了独立复核，**不得写成该 wrapper 的退出码为 0**。前两次读取失败记录留在本机私有工作区，不覆盖；没有发推理请求。

## 可观察的差异与不可推出的结论

- 在 D 的一个独立窗口中，SGLang 的普通 GMM 每 rank 720 次，rank0 p50 为 81.692µs、rank2–7 为约 26.59–28.72µs；所列缓冲区 shape 两组均为 8192 行，但 expert group-list、有效行和 step 未配对。这是需要进一步验证的负载/长尾线索，不是已确认的热点因果。
- P rank0 的 SGLang GMM 和独立 SwiGLU 量化，与 vLLM 的普通 GMM 和 GMM+SwiGLU+quant 融合事件不同；事件计数和输入 shape 也不同，不能将两边总时长相减当成融合收益。
- 这份 CSV 聚合没有完整 request→CPU→enqueue→device 依赖链；少数相同名称/shape 不能证明相同工作量。已发布的 [P rank0 局部时间轴](036-p-rank0-observed-timeline.md)和 [D rank0 局部时间轴](037-d-rank0-observed-timeline.md)分别独立选窗归零，也不是同一个请求。

## 离线检查

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_all_ops_038.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/test_all_ops_038.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_archive.py
```

[核验脚本](../scripts/verify_all_ops_038.py)检查 gzip 解压 SHA、32 rank 身份、事件计数、数值和隐私白名单；[D 脱敏实现](../scripts/sanitize_all_ops_038.py)保存归一化规则。离线检查不重新连接设备、不证明端到端性能、质量或优化收益。
