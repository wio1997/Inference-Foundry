# DeepInfer 与 vLLM Ascend 同期性能对照

测试时间：2026 年 10 月 30 日。服务器：172。两种服务先后运行于同一服务器。

## 测试条件

- 模型：DeepSeek-V4.1-Flash W8A8；16 个逻辑设备，DP2 × TP8，EP16；Engram CPU 卸载，主干 CPU prefetch 关闭。
- AISBench：32,768 输入 token、1,024 输出 token、并发 8，逐配置两轮，确定性生成并忽略 EOS。
- 每轮前执行独立推理预热和冷前缀构建。正式请求前对相同文本执行分词预热，未增加推理计数。
- 每轮正式请求总计查询 262,144 个输入 token，实际缓存命中 235,904 个，即 89.99%。两侧每轮 8 组输入 token 集合一致。

## 结果

| 实现 | 轮次 | 平均 TTFT（ms） | 平均 TPOT（ms） | 输出吞吐（token/s） |
| --- | ---: | ---: | ---: | ---: |
| DeepInfer | 1 | 671.2 | 12.3 | 608.90 |
| DeepInfer | 2 | 587.7 | 12.2 | 605.91 |
| vLLM Ascend | 1 | 2509.0 | 14.6 | 462.53 |
| vLLM Ascend | 2 | 784.3 | 14.0 | 525.21 |

两侧每轮 8 条请求均成功返回 1,024 个输出 token，完整输出文本与冻结参考一致。DeepInfer 的输出 token ID 逐项与历史冻结参考一致；vLLM 的本次客户端未记录逐请求输出 token ID。

按相同轮次比较，DeepInfer 第一、二轮的输出吞吐分别高 31.6% 和 15.4%，TPOT 分别低 15.8% 和 12.9%。本结果仅适用于本次固定负载和测试设置。

## 边界与待查项

- vLLM 两轮 TTFT 从 2509.0 ms 降至 784.3 ms，差异原因尚未定位。不能将两轮平均值视为稳定表现，也不能据此推断造成差异的具体环节。
- DeepInfer 正式请求使用已验证的分词缓存命中；vLLM 正式请求前已调用相同文本的纯分词接口，但尚未证明其内部复用分词结果。两侧端到端 TTFT 因此仍存在分词路径差异。
- vLLM 的缓存命中来自计数器窗口汇总，未取得逐请求缓存命中和 DP 路由。已观察到图捕获，但这本身不能证明正式请求的图重放。
- 两种服务先后而非同时运行；测试锁仅约束本次测试，不能完全排除外部客户端访问。

## 证据

DeepInfer 独立核验：`/data/tiankuan/wio/910c-deepinfer-172/fair-comparison-20261028/completion/fair-now-deepinfer-20261030/independent-fair-now-deepinfer-review-20261030.json`。

vLLM 两轮原始数据和独立核验：`/data/tiankuan/wio/910c-deepinfer-172/fair-now-20261030/completion/vllm1/` 与 `.../completion/vllm2/`。每轮均包含 `independent-audit.json` 和 `tokenizer-independent-audit.json`。原始数据、配置、日志及执行状态与对应核验报告一并保留。

