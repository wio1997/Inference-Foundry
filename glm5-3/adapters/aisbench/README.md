# Task-local AISBench adapter

Current workload (user update, 2026-10-07): input **80K = 80,000 tokens**, output **0.6K = 600 tokens**, shared input prefix **93%**. `config.py` defaults and new formal commands use `--input_len 80000 --output_len 600 --repeat_rate 0.93`; the SLO analyser defaults to `--expect_output_len 600`. CLI arguments still override defaults. Historical frozen commands/results retain their original lengths and must be analysed with their original expected output length explicitly.

The 93% value describes shared prompt content. The user separately specifies **93% KV cache hit rate** as the workload condition. Record the actual tokenized request length after chat templating, prefix warmup and observed cache reuse in the frozen Run; `HitRateCollector` measures native HBM and external prefix-cache counters separately. Do not fill the measured result with the declared 93% value or equate D external KV loading with P prefix-cache reuse. These defaults do not declare a tested concurrency or stable service capacity.

完整AISBench驱动位于[研究分支的适配器目录](https://github.com/wio1997/Inference-Foundry/tree/glm5-3-autonomous-20261001/glm5-3/adapters/aisbench)。

