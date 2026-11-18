# 原生0.32：16-rank原始采集和实际逻辑工作量

epoch727已完成P/D各8rank原始采集，并从全部16rank的原始CPU记录提取真实调度注释；设备解析和跨框架对比仍未完成。镜像、95文件权重、依赖及数学配置沿用[profiling实例说明](017-native032-profile-epoch-readiness.md)。新16条诊断输入与448条原预热/正式前缀无冲突，实际服务端/tokenize逐条count与token-ID SHA和CPU冻结值一致；这一步没有模型生成或缓存预热。

正式采集前的[五轮80请求父级验收](../evidence/profile_capture_G804/parent_all5_warming.json)确认每条32768输入/1024输出、每轮实际C16。最后两轮duration变化0.57284%、TTFT0.17345%、TPOT2.65487%；D编译artifact并集无变化，P独立402项也完全相同。原epoch630首次正式批次未重发。专用warming脚本始终warmup_only，正式ledger继续保留。

[首次诊断原始验收](../evidence/profile_capture_G804/parent_diagnostic_raw.json)确认16/16成功、C16和精确32768/1024。它是带instrumentation的诊断负载，不替代四档首次baseline，不报告优化收益。P/D原始采集约6.16GB/26.59GB，全部保留受控存储；[P索引](../evidence/profile_capture_G804/raw_capture/prefill_raw_capture_index.json)和[D索引](../evidence/profile_capture_G804/raw_capture/decode_raw_capture_index.json)仅列文件bytes/mtime，明确没有冒充逐文件SHA。请求、prediction和SQLite的[原始SHA索引](../evidence/profile_capture_G804/raw_workload_index.json)区分完整文件与省略大input的派生行。

## 真实调度工作量

以下是[全部P rank汇总](../evidence/profile_capture_G804/logical/prefill/all8_summary.json)与[全部D rank汇总](../evidence/profile_capture_G804/logical/decode/all8_summary.json)的生成视图。CPU注释直接来自原始torch.op_range中的实际JSON，各侧八rank去除rank号后的完整序列SHA一致，因此公开一份canonical序列和八rank源SHA，不重复保存八份相同逻辑数据。

| 范围 | P | D |
|---|---:|---:|
| 捕获的scheduler调用（含零query） | 198 | 913 |
| 零query调用 | 133 | 7 |
| 实际scheduled query tokens合计 | 524272 | 21450 |
| 最大单步scheduled query | 8188 | 36 |
| 最大调度微批次请求数 | 2 | 6 |
| metadata中的draft proposal slots | 0 | 17875 |

这里每侧是同一全局调度工作，不把TP八rank复制相乘当用户token数。客户端C16不等于每个device step都batch16。P总query恰为16×(32768−1)，其中65个非零query步骤；其余133次零query不能先验认定为无效NPU计算。D多为每请求6个scheduled query，是否以及如何padding必须从实际graph/device证据确认，不能从配置96或proposal数直接推断。

[独立server计数器差值](../evidence/profile_capture_G804/actual_spec_counter_deltas.json)显示本地prefix-hit两端均0，D外部prefix-hit为524272，P generation16/D generation16384。D实际draft calls3559、draft tokens17795、accepted tokens12864；这些与metadata的proposal slots不是同一计数口径，差值原因尚未验证。不能把proposals当accepted、把accepted数直接当最终返回token，或将本地hit0解释为没有P→D外部KV传输。

## 设备解析状态和保留的失败

两端worker内置SDK回调都报告daemon进程不能解析，因此HTTP200和完整请求仅证明采集动作结束。离线操作先冻结原始rank文件SHA，再建立独立reflink或普通副本；SDK只在副本写入，结束后重查原始SHA。未缩短或重发NPU负载来绕过解析成本。

[P rank0设备检查](../evidence/profile_capture_G804/P0_parsed_device_review.json)已得到225813条kernel_details记录，device0，Model ID均4294967295，与P eager一致；解析trace中198个scheduler注释与原始CPU记录一致。完整trace约1.624GB、设备DB约531MB，仍留受控存储。P其余7rank已在两个独立8CPU槽并行解析，尚未据此标all-rank设备通过。

[D首次离线失败](../evidence/profile_capture_G804/offline/D_first_offline_timeout.json)明确记录4CPU/600秒解析以124超时，只有约713MB的部分CPU trace，缺op_statistic，原始数据SHA未变。尝试扩核时原解析已退出，实际没有改变线程；该失败保留。后续在新副本上启动16CPU/1800秒D rank0解析，尚未声明通过。

[当前状态](../metadata/profile_epoch_G727.json)保持all-rank设备、实际D target/draft replay、完整逻辑与物理任务映射未验收。busy union包含WAIT，不能称计算利用率；重叠duration不能直接相加当墙钟收益；P/D时钟未校准，不直接相减。仍需完成所有设备窗口、SGLang同题集质量及四档矩阵、匹配profile、算子/调度/通信/长尾/无效工作地图和正确性加端到端收益验证。整个Goal尚未完成。
