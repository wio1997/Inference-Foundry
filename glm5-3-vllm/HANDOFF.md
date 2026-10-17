# 当前交接 — 2026-10-18 本机复核与官方固定提交补源

整体研究尚未完成；本轮继续完成聚焦的本机状态核验，并获授权从Mac经HTTP补取官方固定提交的最少源码。没有SSH、部署、抢占/续租或在线压测。此前P11/D12退出0、proxy关闭，原32/33精确UID预留已删除且复核不存在；这些旧CID/API/holder不是live资源。交付渠道为现有分支和draft PR2，普通push已获授权，不合并、不新增设备窗口。

当前候选收窄为“把D初始有效query从最后prompt推进到P已知首输出y1”。**有条件可行，尚未证明设备状态齐全；不接受保持原MTP5即可省掉完整模型迭代的推断。** 原生P元数据及D cold external hits支持声明接收N，D接收晋升会为重采样把computed=N回退N−1；采纳y1为output history后可保持computed=N、总tokens=N+1。但NewRequestData无初始output IDs，worker新建CachedRequestState固定output_token_ids=[]，必须显式交接该状态，不能只加scheduler计数。

原生MTP5首批padding仍为6行/请求，C4可为24行，推进query不直接减少这批行数或drafter工作。固定提交补源把路径追到AscendEagleProposer→AscendSpecDecodeBaseProposer，draft架构DeepseekV32MTPModel注册为AscendDeepSeekMTP。存档glm_moe_dsa配置只有1个nextn层，K5复用该层，不是五套层历史。首轮左移token并保持target位置/hidden：P尾部MTP行N−1由y1+h(prompt最后)形成，推进后D用新hidden与y2在N继续，因此无需先认定必须传P raw hidden，但完整MTP前缀仍须有效。12个新增文件425191字节均仅固定提交参考，没有现场逐文件SHA配对或proposer对象类型记录；原runner等已配对来源继续单列。源注册/映射按实际层group/page/scale描述缓存，不能由aggregate finished或P/D各101条目推出最后prompt行、全部C8/Indexer/MTP状态有效；完整运行时handshake、D块表和最后行内容仍未存档。旧D会重算该行，既有正确性不能单独验证跳过它。

CPU验证层级要精确：完整原生Request导入因缺兼容torch失败，msgspec/vLLM也缺。只执行固定SHA的原类定义/方法段及8条真实存档token IDs；构造器、SamplingParams、真实KV manager、InputBatch/GPU/MTP没有验证。有限反例还确认：推进后不能保留v1“核对并抑制重复y1”协议，新D首结果为y2时会误报mismatch或误删；原4f42984代码块与真实存档参考ID20→198的检查已复现。把外部P token当普通Async完成会将0placeholder减为−1；在KV晋升回调内结束max1会被外层重新设WAITING。原生append/check_stop的计数检查不等于资源回收证明，不用自洽模拟冒充原生全链路。

报告口径更新：TPOT=(最后token−首token)/(输出数−1)，首token提前且后续不动会机械抬高它。原v1八组TTFT倍率0.80080、第二/最后到达1.00424/1.00491、首二gap2.46130，第二至末区间倍率1.00897；不能把平均TPOT1.12897叫稳定decode变慢13%。八组等权均值ΔTPOT1.48368ms，按首/末时刻分，首token前移贡献1.29091ms（87.0%）、末时刻0.19277ms；按首二gap/后段分，则为1.30817ms（88.2%）和0.17550ms。88.2%包含第二token时刻变化，不能与87.0%混称；后段仍受不同D实例、续写、batch/传输影响。draft3183→3240是不减工作证据，不等同target kernel数。

负phase已找到固定提交源码机制：提前输出被frontend当首次prefill输出，记录first_token_ts并置is_prefilling=False；异步KV接收时尚未记录SCHEDULED，随后真正调度才填scheduled_ts，完成时first_token_ts−scheduled_ts成为负。原方法的定义级显式时间fixture已得到−0.25秒，普通顺序+0.25秒；没有运行完整frontend事件或exporter，新stats源码也未镜像SHA配对。16/16原负增量保留，不能写成修复通过。后续需分开发布与计算phase事件，不能截零或伪造调度时刻。

下一最小设备判别有条件值得做，但仅作说明，未申请/安排：新的允许窗口内一次加载复用，冷C1 N=256/257先捕获所有target C8/Indexer/MTP层metadata及最后prompt行有效性；缺行/前缀即否定直接推进。若齐全，显式送入初始output history=[y1]且computed=N、prompt=N、预算64，检查实际前两个D步骤的输入ID/位置、有效/补齐行、hidden来源、max1/64及取消归属，再判断第二token和完整请求是否真少步骤。P强制min_tokens1与D0的采样等价性仍需验证。不要由旧trace推新组织成本；不立即写/部署未经状态证明的原型，不建通用平台。

本轮入口：[具体可行性判断](analysis/p-first-state-feasibility-20261018.md)，证据在evidence/20261018-p-first-state-audit/，重点fixed-source-followup-v1/source-and-state-findings.json与stats-definition-result.json；前轮native-contract-run-v3/result.json明确full_native_import=false。此前原始接收计数、完整时序与源码索引均保留。实际镜像源码归档在glm53-execution-evidence-backup/20261018-offline-state-audit/source；新增参考源码在同级20261018-fixed-commit-followup，逐文件SHA已重新核验。代码为tools/offline_p_first_state_audit.py与compare_pd_publication.py。

官方MTP/已有overlap/graph两种合成负载持续2.29224×仍归官方基线完善，不作原创机制收益。发布v1的240/240 formal、925/925同arrival、有限数值/复用通过及不默认采用结论保留；原12算术仍11/12，完整精度、生产SLO/容量未验收，不同D实例/续写差异未消除。原现场结论、SHA备份与资源退出见evidence/20261017-official-pd-optimization/p-first-publication-verdict-v1.json及finite-window-closed-summary-v1.json。不主动向其他聊天发消息，主控直接读本文件。
