# 32K / 1K · AISBench · C8 · 90% prefix hit

2026-10-26，172 的现有服务完成两轮，每轮 8 请求、8 并发。16/16 请求成功，每条实际输入 32768、输出 1024 token。服务最终 HTTP 200，16 个逻辑设备均有 worker，无重启、无容器 OOM。

| 轮次 | 成功 | 总输出吞吐 token/s | 平均 TTFT s | 平均 TPOT ms | 平均端到端 s | 实际命中率 |
|---|---|---|---|---|---|---|
| 1 | 8/8 | 48.0556 | 4.3693 | 161.9 | 170.0136 | 89.99023438% |
| 2 | 8/8 | 50.6070 | 0.9377 | 156.9 | 161.4925 | 89.99023438% |

总输出吞吐是整机 8 个并发请求的输出 token 总数除以 AISBench 正式测量时长；不把输入 token 加入输出吞吐。两轮合并为 **49.2983 token/s**。TTFT 和端到端为每请求均值，TPOT 为生成阶段每 token 的平均间隔。

配置为 DP2×TP8、EP16，16 逻辑设备 / 8 物理卡，vLLM-Ascend `v0.30.0rc2-a3`。max_model_len=34816、max_num_batched_tokens=2048、max_num_seqs=4/DP、block_size=128；Engram CPU offload 与 DP shared memory 开启，主干 CPU prefetch 关闭，eager、async、prefix caching 开启。

使用 [AISBench 固定源码及 prefix_cache 插件](https://github.com/AISBench/benchmark/tree/7bcb6440da11bef020943b2af9ed7c1ac8362306/plugins/prefix_cache)，插件版本 0.5.0。客户端独立 venv，通过 PYTHONPATH 运行原始源码；自定义客户端适配层让 AutoTokenizer 调用服务使用的 deepseek_v41 tokenizer，未修改服务包。stream=true、temperature=0、ignore_eos=true，输出固定 1024，retry=1 在本版本表示共一次 HTTP 尝试。

GSM8K train 问题作为合成长度素材，每轮前缀带不同的唯一命名空间。正式请求前分别预热两个 DP，预检与预热均不计入正式指标，AISBench num_warmups=0。每轮 8×32768=262144 个输入 token，实际 cache hits=235904，命中率为 **89.990234375%**；与按 128-token block 取整后的理论值一致。两个 DP 每轮各处理 4 个正式请求。

当前服务的 reset_prefix_cache 接口返回 404，因此保留 reset_url=null、assume_empty_cache=true，插件明确记录 ASSUME_EMPTY_CACHE 警告。每轮使用新前缀避免复用上一轮同名缓存，实际命中率由正式负载前后服务计数器差值独立验证。没有把配置中的 90% 当作实测值。

逐请求记录、原始 CSV/JSON、服务指标快照及 SHA256 验收均保存在私有 wio 项目目录；公开仓库仅保存本报告与脱敏汇总。两轮属于小样本性能基线，不代表准确率、持续负载或最大吞吐。与此前 C1/C2 基线的数据、客户端和采样不同，不能据此单独量化缓存收益。

机器可读结果见 [aisbench-summary.json](aisbench-summary.json)。
