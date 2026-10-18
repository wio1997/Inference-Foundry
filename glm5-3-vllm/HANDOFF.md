# 当前交接 — 2026-10-18 本机复核与官方固定提交补源

整体研究尚未完成；本轮继续完成聚焦的本机状态核验，并获授权从Mac经HTTP补取官方固定提交的最少源码。没有SSH、部署、抢占/续租或在线压测。此前P11/D12退出0、proxy关闭，原32/33精确UID预留已删除且复核不存在；这些旧CID/API/holder不是live资源。交付渠道为现有分支和draft PR2，普通push已获授权，不合并、不新增设备窗口。

当前候选收窄为“把D初始有效query从最后prompt推进到P已知首输出y1”。**有条件可行，尚未证明设备状态齐全；不接受保持原MTP5即可省掉完整模型迭代的推断。** 原生P元数据及D cold external hits支持声明接收N，D接收晋升会为重采样把computed=N回退N−1；采纳y1为output history后可保持computed=N、总tokens=N+1。但NewRequestData无初始output IDs，worker新建CachedRequestState固定output_token_ids=[]，必须显式交接该状态，不能只加scheduler计数。

原生MTP5首批padding仍为6行/请求，C4可为24行，推进query不直接删去这批行数。**首批6→6也不证明整请求绝不可能少一次target forward；最终须按同预算/同采样完整请求的实际target前向次数判断。** 固定提交补源把路径追到AscendEagleProposer→AscendSpecDecodeBaseProposer，draft架构DeepseekV32MTPModel注册为AscendDeepSeekMTP。存档glm_moe_dsa配置只有1个nextn层，K5复用该层，不是五套层历史。首轮左移token并保持target位置/hidden支持如下源码路径推断：若P首轮确实运行并正确转移，尾部MTP行N−1由y1+h(prompt最后)形成，推进后D用新hidden与y2在N继续。此处不是现场内容验证；无需先认定必须传P raw hidden，但完整MTP前缀仍须有效。此前12个新增文件425191字节仅固定提交参考；本轮另补executor/mixin两个文件49693字节，同样未镜像SHA配对。原runner等已配对来源继续单列。不能由aggregate finished或P/D各101条目推出最后prompt行、全部C8/Indexer/MTP状态有效；完整handshake、D块表和最后行内容仍未存档。旧D会重算该行，既有正确性不能单独验证跳过它。

CPU验证层级要精确：完整原生Request导入因缺兼容torch失败，msgspec/vLLM也缺。只执行固定SHA的原类定义/方法段及8条真实存档token IDs；构造器、SamplingParams、真实KV manager、InputBatch/GPU/MTP没有验证。有限反例还确认：推进后不能保留v1“核对并抑制重复y1”协议，新D首结果为y2时会误报mismatch或误删；原4f42984代码块与真实存档参考ID20→198的检查已复现。把外部P token当普通Async完成会将0placeholder减为−1；在KV晋升回调内结束max1会被外层重新设WAITING。原生append/check_stop的计数检查不等于资源回收证明，不用自洽模拟冒充原生全链路。

报告口径更新：TPOT=(最后token−首token)/(输出数−1)，首token提前且后续不动会机械抬高它。原v1八组TTFT倍率0.80080、第二/最后到达1.00424/1.00491、首二gap2.46130，第二至末区间倍率1.00897；不能把平均TPOT1.12897叫稳定decode变慢13%。八组等权均值ΔTPOT1.48368ms，按首/末时刻分，首token前移贡献1.29091ms（87.0%）、末时刻0.19277ms；按首二gap/后段分，则为1.30817ms（88.2%）和0.17550ms。88.2%包含第二token时刻变化，不能与87.0%混称；后段仍受不同D实例、续写、batch/传输影响。draft3183→3240是不减工作证据，不等同target kernel数。

负phase已找到固定提交源码机制：提前输出被frontend当首次prefill输出，记录first_token_ts并置is_prefilling=False；异步KV接收时尚未记录SCHEDULED，随后真正调度才填scheduled_ts，完成时first_token_ts−scheduled_ts成为负。原方法的定义级显式时间fixture已得到−0.25秒，普通顺序+0.25秒；没有运行完整frontend事件或exporter，新stats源码也未镜像SHA配对。16/16原负增量保留，不能写成修复通过。后续需分开发布与计算phase事件，不能截零或伪造调度时刻。

下一步审阅材料在[最小取证准备](diagnostics/p-first-state-v1/README.md)：两份原生基线配置SHA、cold C1 N=256/257精确输入/边界位置、P K1/D K5、P/D min1与输出1/2、18文件无NPU身份检查脚本、既有启动/采集命令模板及保全退出范围已准备。**设备探针尚未就绪，不应据此申请新窗口**：缺实际cache views/group块表与writer位置的配对，以及D成功接收后、finished入队前的行/完整前缀摘要适配器；没有强写泛用C8切片或修改kernel。18项同源fixture、空/错SHA及输入错位/缺项等Mac检查通过，只证明源码/方案完整性，未实现或验证设备捕获拒绝器。建议先在新的SSH授权内用既有collector无NPU image模式配对源码身份，再决定承担两端边界关联与布局适配的成本；当前不SSH、不申请/续租或跑设备。

新增边界复核：Mooncake wait_for_save为空，不能把finalize称作MTP写完成fence。已匹配的Async输出在构造时copy stream等待当前计算stream，get_output在返回前等待已有event；新补固定executor在KV aggregator存在时收齐各worker并解包Async输出。实际MTP首轮是否执行、writer stream及NPU event映射、全部响应rank和TE同步完成契约仍须取证；原生wait前pending不等于race，不能新增同步掩盖缺依赖。D样本必须在原生重算N−1之前取得，与所选P rank/请求/逻辑位置配对，非零或少量行相等不能证明完整前缀。

若适配器将来就绪，新的有限窗口才考虑两台各16die、一次加载、仅两个输入的功能性取证。历史完整启动→API ready P11约626.893秒、D12约636.642秒，而非仅load-start后的201/213秒；预计含取证/保全18–22分钟、建议最大25分钟、15分钟未ready即止，不续期，均非新资源申请。只停止fresh launch receipt内自己的实例，精确UID释放新holder；旧lease/CID/D13/硬编码stop脚本不能复用。若状态齐全且后续获授权，再进入history交接、max1/取消及完整请求前向次数实验；不由旧trace预算新组织收益，不建平台。

本轮入口：[具体可行性判断](analysis/p-first-state-feasibility-20261018.md)，证据在evidence/20261018-p-first-state-audit/，重点fixed-source-followup-v1/source-and-state-findings.json与stats-definition-result.json；前轮native-contract-run-v3/result.json明确full_native_import=false。此前原始接收计数、完整时序与源码索引均保留。实际镜像源码归档在glm53-execution-evidence-backup/20261018-offline-state-audit/source；新增参考源码在同级20261018-fixed-commit-followup，逐文件SHA已重新核验。代码为tools/offline_p_first_state_audit.py与compare_pd_publication.py。

官方MTP/已有overlap/graph两种合成负载持续2.29224×仍归官方基线完善，不作原创机制收益。发布v1的240/240 formal、925/925同arrival、有限数值/复用通过及不默认采用结论保留；原12算术仍11/12，完整精度、生产SLO/容量未验收，不同D实例/续写差异未消除。原现场结论、SHA备份与资源退出见evidence/20261017-official-pd-optimization/p-first-publication-verdict-v1.json及finite-window-closed-summary-v1.json。不主动向其他聊天发消息，主控直接读本文件。
