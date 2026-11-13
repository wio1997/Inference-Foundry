> 当前归档共1056条逐字可恢复请求记录；其中G42新增192条、G43有效32K初测新增48条，含预热。G43后续中止批次未计分。见[003条目](003-g42-g43/README.md)。下文816条/两轮等描述对应其历史归档范围。

# 数据与证据

**G37补充：**本分支已加入2026-11-13的SGLang G37完整四档192条、同机原生vLLM完整四档192条及原生32K一次重复48条。与下述早期两轮合计816条请求，共享同一组192条实际输入。恢复方式不变；所有新详情文件同样按原始SHA逐字验证。入口见[G37记录](002-g37-markov/README.md)。下文“两轮”指早期G29/G30。

## 两轮数据

- `20261112T124145Z`：G29，原生 QKV 融合开关为 0 的已完成基线。
- `20261112T134338Z`：G30，原生 QKV 融合开关为 1 的最终配置。

两轮均有四档 warm16/formal32，合计 **384 条成功请求记录**。两轮复用同一组 192 条实际输入；同轮预热、正式与不同档输入互不重复。每档实际输入为 32768/65536/131072/262144，输出均为 1024 token，实测最大并发 16；有限请求在启动和排空时，平均并发低于 16。G30 四档平均并发依次为 14.6346、14.7244、15.1613、15.8056。

输入为本模型 tokenizer 下严格计数的合成词表文本。生成脚本、词表筛选方法和随机种子已保留；直接使用已归档输入能避免 tokenizer 或生成器版本变化。见 [数据 manifest](../datasets/manifest.json)、[生成脚本](../benchmark/generate_datasets.py) 和 [实际 AISBench 配置](../configs)。生成脚本模板已修正为最终实测的 retry1 和端口 18167，原先 retry0/高位端口不作为复现默认值。

## 无损恢复

八个输入文件以 `.jsonl.gz` 保存。逐请求原始 JSONL 中的长输入改为对该输入库的引用，其原始 JSON 前缀和后缀逐字保留：

```text
原始行 = json_prefix + JSON编码(指定dataset、row的question) + json_suffix
```

这样无需多次保存完全相同的长输入，仍可还原原始行的空格、顺序、输出、UUID、时序引用及换行。模型预测中生成的示例网络地址属于原始输出，保留原字节；实际服务器地址和连接信息未放入数据。

```bash
python3 scripts/unpack_data.py --with-results --verify-only
python3 scripts/unpack_data.py --with-results
```

前者只在内存恢复并核对 SHA；后者写出原始 `datasets/*.jsonl` 和 `results/**/_details.jsonl`。已存在文件只有哈希相同才复用，不覆盖变更后的文件。原始详情文件恢复后即落在官方聚合结果的 evidence 字段所引用位置。格式与每个原始文件 SHA 在 [data-archive-manifest.json](../evidence/data-archive-manifest.json)。

## 保留的测量内容

- 两轮每阶段的 CLI 命令和日志、官方 JSON/CSV、HTML 图、逐请求输入/预测及 SQLite 时序库。
- 每档服务快照、flush 记录和 NPU 进程归属采样，证明当时没有外部 NPU 作业重叠。
- 独立复核结果、token 使用量、最大并发、正式/预热成功计数与原始文件哈希。
- 最终部署审计中的 ImageID、源码 commit、31 个运行时文件 SHA、Engram DRAM 映射、SSD 设备链及客户端身份。

除长输入去重和压缩外，逐请求记录可逐字恢复；文本日志和部署记录中的内部路径、服务器地址被替换为公开占位符。原文件与公开副本的哈希分别记录于 [publication-provenance.json](../evidence/publication-provenance.json)。字段中的历史绝对路径是脱敏后的记录，不保证是当前 checkout 下的实际位置。

## 指标口径

总输出吞吐 = 32 × 1024 / 正式阶段耗时。TTFT、TPOT、E2EL 使用 AISBench 官方 CSV 的 Average / P99；排队和资源等待包含于该端到端口径。预热不计入正式吞吐。`--max-num-workers 1` 是客户端任务 worker 设置，C16 由模型配置和实际并发检查证明。

本次归档检查验证字节完整性、补丁、语法和指标算术，不是新性能实验，也不是质量评估。每档仅一次固定 32 请求测量，没有重复臂、同臂漂移或跨批次噪声标定。
