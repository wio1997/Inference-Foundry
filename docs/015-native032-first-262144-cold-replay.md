# vLLM 0.32首轮256K：冷测与零本地命中重放

固定原生epoch630，双机各8张910B1/aarch64，镜像d5492ecf，TP8/EP8/DP1/PCP1、DSPARK5，P/D memoryutil .88/.80、chunks8192/1024，P eager、D target/draft FULL_DECODE_ONLY配置。其余版本、权重、量化和不能等价的配置沿用[32K记录](011-native032-first-32768-cold-hot.md)，没有为本档更换模型或依赖。

[唯一数值与资格权威](../evidence/native256_first_G765/parent_assessment.json)确认首次cold32和同输入首次replay32全部有效，每条精确262144输入、1024输出，实际峰值C16。两轮原始输入/配置SHA相同，首轮数据未重发以改变结果。

| 指标 | 首次冷测 | 同输入首次重放 |
|---|---:|---:|
| 平均TTFT（秒） | 256.0294 | 256.0062 |
| 平均TPOT（毫秒） | 8.5000 | 8.5000 |
| 输入吞吐（token/s） | 12204.9498 | 12209.3867 |
| 输出吞吐（token/s） | 47.6756 | 47.6929 |
| 总吞吐（token/s） | 12252.6254 | 12257.0796 |

两轮P/D本地prefix-hit均为0；这里没有建立充分缓存热态，也没有证实性能收益。输出吞吐差约0.04%只作为该首次pair的观测。模型使用原生Mooncake P→D传输，本地prefix-hit为0不代表未进行外部KV传输。

正式提交前完成5轮独立完整输入/C16/1024预热，80/80有效。最后两轮duration变化0.1834%、TTFT变化0.03464%、TPOT变化1.1494%，满足原先duration/TTFT稳定条件；D最后两轮编译artifact无变化。P在warmup3至5、warmup5至完整正式pair后的534项大小/mtime并集相同，D正式每轮的711项也相同。这是持久compiler artifact证据，未声称排除一切瞬时内存编译。

父级从原始server snapshots独立复算：每轮P prompt8388576/generation32，D prompt8388608/generation32768，恰与32条请求匹配。P按L-1/1记账，不能把它直接等同于另一框架的物理工作量。[原始SQLite独立复算](../evidence/native256_first_G765/independent_SQLite_timing_G751.json)核对了全部请求区间、C16、TTFT/TPOT/E2EL、总耗时及吞吐，与报表精度一致。计时与server-usage契约沿用[128K口径说明](014-native032-first-131072-cold-replay.md)，不把SSE事件数当生成token数。

完整原始输入、prediction、SQLite和时间点保留受控原始包；[公开原始SHA与派生视图说明](../evidence/native256_first_G765/raw_details_index.json)区分完整原始文件和省略大input的派生行。工作集8388608输入tokens超过启动报告的本地逻辑容量；这与容量/淘汰一致，未通过控制实验确定具体原因。[单条256K语义/重放检查](013-native032-long-context-selected-quality.md)的通过范围独立，不替代本档C16缓存状态。

至此原生vLLM四档首次pair的原始指标范围已验收；SGLang同口径质量与四档矩阵、双方all-rank P/D profile、工作量地图和优化收益仍未完成。已知raw-ID UTF8和129-ID greedy问题保留，不借四档计数通过宣称整体功能/质量无缺陷。
