# vLLM 0.32首轮64K：冷测与低命中重放

同一原生vLLM-Ascend0.32.0rc1 epoch630，双机各8张910B1/aarch64、TP8/EP8/DP1/PCP1、DSPARK5，P/D memoryutil .88/.80，未重启模型或改并行配置。具体身份和配置沿用[32K记录](011-native032-first-32768-cold-hot.md)。

[父级数值与资格权威](../evidence/native64_first_G708/parent_assessment.json)保存原始五项指标、数据/配置SHA、计数、预热和编译审计。首次冷测32请求成功，随后原样重放同一32条；两轮均精确65536输入、1024输出，实测峰值并发16。

| 指标 | 首次冷测 | 同输入首次重放 |
|---|---:|---:|
| 平均TTFT（秒） | 51.5090 | 51.6286 |
| 平均TPOT（毫秒） | 9.3000 | 9.2000 |
| 输入吞吐（token/s） | 13623.1101 | 13667.0119 |
| 输出吞吐（token/s） | 212.8611 | 213.5471 |
| 总吞吐（token/s） | 13835.9712 | 13880.5590 |

冷测P/D本地前缀命中均为0。重放P为0、D为50432 tokens，仅约2.4%的D输入token；因此该轮保留为低命中重放，**未验证整批处于充分缓存的热态**。输出吞吐差约0.32%，没有足够证据声称性能改善。不能把名为hot的执行阶段直接视为全缓存命中。

五轮独立预热均16/16成功。最后两轮duration变化1.3977%、TTFT变化0.3935%，满足原先duration/TTFT不超过5%的稳定条件；TPOT变化10.98%作为已披露guardrail，未临时改变标准。D最后两轮及正式两轮编译artifact无变化；P在warmup3至5以及warmup5至正式两轮后，534项大小/mtime并集均无变化。各阶段独立P/D prompt/generation计数与请求匹配。

32条输入的工作集2097152 tokens超过启动日志报告的P1291720/D1812610逻辑容量，低命中与容量/淘汰相符；尚无控制实验或profile证明具体因果。后续[单条长上下文检查](013-native032-long-context-selected-quality.md)显示64K单请求立即重放可以命中，不能据此反推批量低命中的全部原因。

首轮原始输入、prediction/timepoints及失败历史保留受控存储；公开[原始SHA与派生视图说明](../evidence/native64_first_G708/raw_details_index.json)明确删除大input字段的行只是派生视图。没有重发首轮以提高命中。128K/256K矩阵、SGLang同口径质量/性能、双方all-rank profile、差异地图和优化收益仍未完成。
