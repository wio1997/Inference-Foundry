# vLLM 0.32首轮128K：冷测与零本地命中重放

同一原生epoch630，双机各8张910B1/aarch64，固定0.32.0rc1镜像d5492ecf，TP8/EP8/DP1/PCP1、DSPARK5、P/D .88/.80、P eager、D target/draft图配置。没有为本档重启模型、换依赖或调整并行。详细配置沿用[32K记录](011-native032-first-32768-cold-hot.md)。D fused_mc2实际排斥shared-expert overlap，图捕获仍不替代实际replay profile。

[父级数值与资格权威](../evidence/native128_first_G748/parent_assessment.json)确认首次冷测32条及同输入首次重放32条全部有效，每条精确131072输入、1024输出，实际峰值并发16。原始first ledger及输入未重发以提高命中。

| 指标 | 首次冷测 | 同输入首次重放 |
|---|---:|---:|
| 平均TTFT（秒） | 113.7588 | 113.6995 |
| 平均TPOT（毫秒） | 8.4000 | 8.5000 |
| 输入吞吐（token/s） | 13303.5797 | 13317.6858 |
| 输出吞吐（token/s） | 103.9342 | 104.0444 |
| 总吞吐（token/s） | 13407.5139 | 13421.7302 |

两轮P/D本地prefix-hit计数增量均为0，因此这里是零本地命中的原样重放，未验证整批处于充分缓存的热态。输出吞吐差约0.11%，不构成已证实的性能收益。API cached_tokens填0也不是命中依据；结论来自两端独立原始prefix计数器。P每条L-1 prompt/1 generation、D每条L prompt/1024 generation的计数均与32条请求匹配；这些是框架记账，不能直接当物理计算量跨框架比较。

五轮独立完整预热均16/16有效。最后两轮duration变化0.02873%、TTFT变化0.00623%、TPOT变化0%，D最后两轮编译缓存无变化。P在warmup3至5的534文件完全相同，warmup5至正式两轮后的534文件也无变化；D正式cold/replay各自711文件的大小/mtime并集相同。这证明实际持久compiler artifact未变，不等于排除所有瞬时内存编译。

## 独立原始时间复算

[SQLite复算结果](../evidence/native128_first_G748/independent_SQLite_timing_G751.json)从完整raw details引用的SQLite NumPy float64数组逐条重算32请求区间、实测C16、首响应延迟、TPOT、E2EL、总耗时和吞吐，与AISBench报表精度一致；原DB的SHA也已记录。[标准库复算脚本](../code/native_metrics_G751/raw_sqlite_timing_audit_g751.py)不依赖NumPy或模型环境，不会生成负载、修改原始文件或反序列化pickle。它在分析客户端读取服务器原始测量，不是客户端性能仿真。

实际固定CPU客户端的[输出类](../evidence/actual_AISBench_timing_G750/models/output.py)、[HTTP计时实现](../evidence/actual_AISBench_timing_G750/models/api_models/base_api.py)、[统计实现](../evidence/actual_AISBench_timing_G750/calculators/base_perf_metric_calculator.py)和[source SHA](../evidence/actual_AISBench_timing_G750/source_index.json)已归档。TTFT沿用AISBench首次非空SSE data-event的时间点口径；它在解析内容前记时，不能据此声称每个事件对应一个生成token，或等于首个非空文本token到达。

原始输入、prediction、SQLite和完整时间点保留在受控证据包；公开[原始SHA与派生视图](../evidence/native128_first_G748/raw_details_index.json)明确省略大input的行只是派生视图。整批4194304输入token超过启动报告的P1291720/D1812610逻辑容量，零命中与容量/淘汰相符，但没有控制实验/profile证明具体因果。[单条131072检查](013-native032-long-context-selected-quality.md)已证明选定语义和立即重放可用，不能用它推定C16批量缓存状态。

256K已进入独立预热，尚无正式结果。SGLang同口径完整质量/四档矩阵、双方真实all-rank P/D profile、逻辑工作量差异地图和优化端到端收益仍未完成。现有raw-ID UTF8和129-ID greedy问题保留；本档不提供整体功能/质量通过或跨框架收益结论。
