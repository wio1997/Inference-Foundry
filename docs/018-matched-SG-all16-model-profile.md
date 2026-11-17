# 同机SGLang与G4输入匹配的十六rank模型profile

G240在同一G181服务上完成16个32768→1024、C16诊断请求，输入文件SHA与vLLM G4 G138完全一致。该诊断开启profiler，时延和吞吐仅保留原始记录，不参与正式性能收益比较。P先观察多个真实排队请求后，与D并行配置原生按阶段8步采集；不会把idle或PREBUILT占位当作模型步骤。

P/D各8个rank均输出原生Profiling done。随后对原始文件全树执行两次SHA/大小/mtime扫描并在CPU解析后再次核对，原始字节未改动。namespace PID、实际NPU进程、物理卡和profiler_info rank逐项一致；解析trace里的硬件进程标签及op_statistic设备号再次确认卡号。16个rank均存在真实硬件X事件和明确MatMul/attention等模型核：P每rank8880个明确模型事件，D每rank9312个。模型事件计数使用相同保守正则条件，不将拷贝或硬件元数据当作模型计算。

采集窗口P可见bs1、8192-token extend及draft步骤，D可见bs1、4-query verify步骤；C16是客户端并发，不声称每个采集步骤batch16。后续必须按工作量、具体shape和相同逻辑阶段比较，而不是把两个框架的整段profile总时长直接当作收益。

- [父级验收与输入SHA](../evidence/matched_SG_G181/profile_G240/G258_matched_SG_16rank_profile_acceptance.json)
- [P八rank模型设备证据](../evidence/matched_SG_G181/profile_G240/CPU_p/review_g252/role.json)
- [D八rank模型设备证据](../evidence/matched_SG_G181/profile_G240/CPU_d/review_g252/role.json)
- [P rank0硬件与CPU步骤窗口](../evidence/matched_SG_G181/profile_G240/CPU_p/review_g252/rank_0.json)
- [D rank0硬件与CPU步骤窗口](../evidence/matched_SG_G181/profile_G240/CPU_d/review_g252/rank_0.json)
- [匹配输入和原始控制manifest](../evidence/matched_SG_G181/profile_G240/control/assets_manifest.json)
- [诊断Run和原始usage/五项指标/SQLite](../records/runs.json)

显式stop_profile在原生自动完成后两端返回500，原始响应保留；验收依靠8rank完成日志、不可变原始树及正模型设备事件，不以500或文件存在单独判定成功。首次G248离线分析未把原生字符串时间戳转为数值，CPU求区间时失败；G252显式数值解析，复用同一原始trace，不重采样。失败分析器、日志及原始文件索引全部保留。

原始二进制和全trace继续在服务器任务目录，Foundry保存逐文件SHA、进程/卡身份、解析报告、控制记录与代码。算子duration求和有重叠，WAIT/SDMA不能直接当作端到端损失；跨主机时钟尚未校准。32K正式cold/hot已完成，完整三档剩余矩阵、因果时间地图和高潜力优化验证仍未完成。
