# GLM 32K P：全量记录的物理输入 shape 字段交集

本分析使用既有 32K、C16、1024输出、**未计分诊断**的 P 侧两框架各8 rank [全设备kernel聚合](039-all-rank-kernel-distributions.md)，逐 rank 按设备 CSV `Name` 的**完整字面值**相同，再比较已记录的完整输入 shape 字段字面字符串。两框架独立采集；同编号 rank 不代表同请求或同步执行。不存在的字段、脱敏或未记录字段不当作零。

## 汇总与重点算子

|范围|每 rank / 汇总观察|可支持的范围|
|---|---|---|
|全部 rank×同名 kernel|8 rank 相加 3339 组；其中336组至少有一个相同的已记录字段字符串|这是 rank×kernel 分组数，不是 336 个独立算子或试验；dtype 和输出 shape 未配对。|
|普通 GroupedMatmul|SG 每 rank 688 次、vLLM 360 次；8/8 rank 的完整已记录输入字段字符串交集均为0|每个rank两边记录的字段串含空 tensor 槽 `;;`；不能从不相等的串断言所有 tensor 形状已知或语义无交集。|
|lightning indexer|两边每 rank 各96次；SG 记录4个、vLLM记录14个完整字段串，8/8 rank交集为0|仅比字面字段；query 维可能相同但KV/batch字段不同，不能按名字推断同工作量。|
|SparseFlashAttention|SG 每 rank 96次、vLLM 112次；8/8 rank 字面字段交集为0|vLLM 的字段含空槽；两边独立窗口，不能由此推导耗时因果。|
|KDA|SG `chunk_kda_fwd` 每 rank 272 次的输入 shape 全未记录；vLLM KDA 名称不同且有记录|无法判断语义 shape 相同或不同，不能将未记录当0。|

P 聚合的 dtype 计数把输入/输出列混在一起，**不是按 shape 的联合 dtype**；相同的输入物理字段仍不证明相同 dtype、输出、active token、expert group-list、请求、step 或内容。一个字段出现分号分隔的空项可能是可选输入或尾部分隔符，不能解释为零维或必然有未记录tensor。内核 p50 属于该核完整窗口内的所有事件，不是某个 shape 的专属 p50。跨 stream 求和不等于关键路径。

[脱敏交集摘要](../evidence/p_shapes_041/fullshape_summary.json.gz)解压 SHA `e1b4e1faef1c460d7688c65f1ea69f5177ae81594e3938d1ef9b2f0362aa27c7`；P 全 kernel 聚合解压 SHA `3b5a99bf6882e2c8c13f1c9e5540528ccff429758e2509117ad5d9b2430b6622`。SGLang 设备 CSV 的 SHA、大小、mtime 对冻结索引；vLLM 是 G148 派生分析副本，CSV 对此前扫描与两次读取稳定，并无历史冻结 CSV SHA。原始大 trace 不入仓。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_p_shapes_041.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_archive.py
```

[离线核验](../scripts/verify_p_shapes_041.py)检查解压 SHA、rank 覆盖、重点事件的字面字段交集与空槽边界；它不读取设备、不证明这几个算子导致首 token 差距，亦不能代替请求级 stage 与数值/性能 A/B 验证。
