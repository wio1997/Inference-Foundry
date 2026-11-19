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

[P rank0设备检查](../evidence/profile_capture_G804/P0_parsed_device_review.json)已得到225813条kernel_details记录，device0，Model ID均4294967295，与P eager一致；解析trace中198个scheduler注释与原始CPU记录一致。完整trace约1.624GB、设备DB约531MB，仍留受控存储。P其余7rank随后在两个独立8CPU槽完成解析。[八rank设备与CPU交叉验收](../evidence/profile_capture_G804/all8_P_device_CPU_review.json)已逐库确认device0–7映射、非零实际compute tasks、198个CPU调度注释及每rank524272个scheduled query，且原始SHA保持一致。每rank实际compute-task行数均为219768；统计不相乘当全局用户tokens。D全八rank设备及完整跨框架profile仍未验收。

[D首次离线失败](../evidence/profile_capture_G804/offline/D_first_offline_timeout.json)明确记录4CPU/600秒解析以124超时，只有约713MB的部分CPU trace，缺op_statistic，原始数据SHA未变。尝试扩核时原解析已退出，实际没有改变线程；该失败保留。后续在新副本上启动16CPU/1800秒D rank0解析，尚未声明通过。

[当前状态](../metadata/profile_epoch_G727.json)保持all-rank设备、实际D target/draft replay、完整逻辑与物理任务映射未验收。busy union包含WAIT，不能称计算利用率；重叠duration不能直接相加当墙钟收益；P/D时钟未校准，不直接相减。仍需完成所有设备窗口、SGLang同题集质量及四档矩阵、匹配profile、算子/调度/通信/长尾/无效工作地图和正确性加端到端收益验证。整个Goal尚未完成。

## 后续D0与CPU解析状态

完整D0的Text/Db设备和CPU覆盖现已确认；最新失败、独立CPU解析路径与尚未完成项见[019](019-native032-full-D0-and-CPU-offline-analysis.md)。本文前述D解析等待描述保留为此前阶段记录，当前状态以019及metadata/profile_epoch_G727.json为准。

## Prefill 量化索引算子

G804 的 Prefill rank 0 原始采集离线导出中，`VllmAscendQuantLightningIndexerV2` 出现 520 次，设备任务累计 1.3505 秒。另有 520 次 query 量化和 520 次索引整理。固定镜像的实际模型源码调用 `npu_quant_lightning_indexer_v3`，上面的 V2 是设备统计表中的算子名称。设备任务时间可能重叠。 [设备算子统计](../evidence/profile_capture_G804/derived/P0_indexer_operator_aggregate_G1395.json) [镜像原始索引源码](../code/fixed_native_attention_G1130/models/deepseek_v41/indexer.py)

同一份 `kernel_details.csv` 的逐任务记录确认，520 次融合索引调用的前两项输入均为 INT8，对应源码中的量化 query 和分页 key。第一项输出均为 `[query_rows, 1, 512]` 的 INT32 索引。8184 或 8188 行的 256 次调用合计 1.1687 秒，占该算子累计设备任务时间的 86.5%。 [逐任务形状及源文件 SHA](../evidence/profile_capture_G804/derived/P0_indexer_operand_shapes_G1418.json)

设备任务与调度步骤已通过 `TASK.connectionId`、`CANN_API`、`CONNECTION_IDS` 和 `PYTORCH_API` 逐条关联。CANN launch 与调度标记位于不同线程，桥接后的 `aclnnVllmAscendQuantLightningIndexerV2` 调用与调度标记位于同一线程、且完整落在标记区间内。520 个设备任务均有唯一映射。65 个非零 query 步骤各有 8 次任务，其中 4 次 query 行数与该步调度量一致，另外 4 次使用 128–256 行。133 个零 query 步骤均无此算子任务。逐层归属及 SGLang 侧同等工作量的时耗仍待验证。 [逐步映射](../evidence/profile_capture_G804/derived/P0_indexer_scheduler_bridge_G1419.json) [提取器](../code/profile_capture_G804/native_P0_indexer_scheduler_bridge.py) [CPU 调度注释](../evidence/profile_capture_G804/logical/prefill/canonical_annotations.json)

同一 G804 采集的八个 Prefill rank 原始算子统计已复核，rank 0 文件摘要与此前单卡读取一致。每 rank 都有 520 次融合索引、520 次 query 量化、520 次索引整理和 2,795 次 SparseFlashMla 任务。融合索引的累计任务耗时在 1.3498–1.3512 秒，占各 rank 算子任务耗时总和的 4.49%–4.63%。SparseFlashMla 对应 7.5530–7.5563 秒及 25.11%–25.87%。任务可重叠，这组数字用于选择同工况对照项，SGLang 当前实际算子归因仍待匹配 trace。 [八 rank 实测](../evidence/profile_capture_G804/derived/all8_P_indexer_operator_statistics_G1451.json) [提取器](../code/profile_capture_G804/audit_native_P_indexer_allrank_g1451.py)

## Prefill 后段层的尾窗工作量

G727 的实际启动日志确认，原生 0.32 对第 21–39 层（从 0 编号）启用 128-token 的 Prefill 尾窗。固定源码先对每条符合条件的请求保留至多 128 行，再以紧凑 batch 执行后段层。按原始调度注释复算，G804 的 65 个非零步骤和 80 个请求分段全部与实际设备小形状吻合，包括两个请求且各自保留行数不对称的步骤。 [实际日志摘要与逐步复核](../evidence/profile_capture_G804/derived/P0_late_prefill_tail_replay_G1462.json) [核对代码](../code/profile_capture_G804/audit_G804_P0_late_prefill_replay_g1462.py) [固定源码](../code/fixed_native_attention_G1130/models/deepseek_v41/decoder_replay_state.py)

这 65 步的完整 query 共 524,272 行，尾窗共 9,847 行。Prefill rank 0 的融合索引在完整形状和尾窗形状各有 260 次，设备任务累计分别为 1.170 秒和 0.180 秒。按源码层界推导，21 层走完整形状、19 层走尾窗，对应约 1,119.7 万 query 行层数，40 层全长反事实为 2,097.1 万，几何量相差 46.61%。任务缺少独立层编号，行层数也无法直接换算 FLOPs 或墙钟。跨框架分析需核实 SGLang 是否在相同层边界保留等效工作量，才能解释同一输入长度下的设备耗时差异。 [实际任务形状](../evidence/profile_capture_G804/derived/P0_indexer_operand_shapes_G1418.json) [固定层边界源码](../code/fixed_native_attention_G1130/models/deepseek_v41/decoder_replay_layers.py)
