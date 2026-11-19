# vLLM 0.32首轮正式结果：32K / C16 / 1024

这是同一910B1/aarch64 P/D双机、各8卡、原生vLLM-Ascend0.32.0rc1 epoch630的一档结果。固定镜像d5492ecf，TP8/EP8/DP1/PCP1、DSPARK5、P eager、D target/draft FULL_DECODE_ONLY意图，P/D memoryutil .88/.80。D fused_mc2实际关闭shared-expert overlap；不能声称两者同时开启。actual graph replay仍需profile。

下表从[唯一数值权威/父级验收](../evidence/native32_first_G691/parent_acceptance.json)生成。冷测使用32条此前未提交的独立前缀，热测原样重放同一数据和配置；两轮均32/32成功，实际峰值并发16，每条精确32768输入、1024输出。

| 指标 | 首次冷测 | 同输入热重放 |
|---|---:|---:|
| 平均TTFT（秒） | 22.2493 | 2.2438 |
| 平均TPOT（毫秒） | 10.5000 | 13.7000 |
| 输入吞吐（token/s） | 12923.8068 | 26400.6749 |
| 输出吞吐（token/s） | 403.8690 | 825.0211 |
| 总吞吐（token/s） | 13327.6758 | 27225.6960 |

## 验收证据与边界

正式请求前完成5个不同完整输入/C16/1024预热，每轮16/16。最后两轮duration变化0.6855%、TTFT变化0.0724%，D两轮编译缓存无变化；P在warmup3至5的534文件元数据并集无变化。正式cold/hot各自D缓存无变化，P从warmup5后至完整正式pair结束的534文件亦无变化。这是实际持久编译artifact大小/mtime证据，不等于证明所有临时内存编译绝不存在。

冷测两端local prefix-hit计数delta0；热测P/D各1044480 token。P/D逐阶段prompt/generation计数也与32条请求匹配。P按每条L-1 prompt、1 generation记账，不能直接将该计数与另一框架的物理计算量等同；端到端五项指标使用AISBench最终server usage。热测TPOT高于冷测，吞吐/TTFT变化不能表述为所有指标同时改善。

旧只读cwd启动失败保留，未提交模型请求；修复后首轮真实数据从未重发。原始manifest、冷/热相同数据及配置SHA、预热恢复记录、编译和cache计数审计均归档。完整原始details（含输入、prediction、timepoints）保留在受控证据包；[原始SHA索引与派生视图说明](../evidence/native32_first_G691/raw_details_index.json)区分完整原始文件与删除大input字段后的公开派生行，未删除失败或截断样本来取得有效结果。

本档不证明64K/128K/256K，不证明SGLang对齐或任何跨框架收益，不替代all-rank P/D profile/因果地图。完整GSM8K的1270/1319是独立质量结果，已在009记录；已知raw token-ID UTF8/greedy问题保留原失败，不借本档性能通过抹去。

从冷测与重放的原始 SQLite 时间点分别重建 32 条请求区间。两阶段并发峰值均为 16，首条完成前均只有 16 条请求开始。第 17 条分别在首条完成后 2.51 毫秒和 1.96 毫秒记录开始。这些时间点位于 AISBench 客户端时钟。 [请求并发复算](../evidence/native32_first_G691/derived/formal_C16_HTTP_admission_G1408.json)
