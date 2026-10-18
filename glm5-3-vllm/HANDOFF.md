# 当前交接 — 2026-10-18 两节点源码身份实核与尾行探针实现

整体研究尚未完成。本轮已通过既有授权恢复SSH，先核对访问pod→node→hostname，再对32/33固定实际镜像各跑一次CPU-only18文件SHA检查，均18/18匹配，约0.317/0.267秒。network none/read-only、CPU1/内存1G，无NPU/驱动/权重挂载、resource_lease=null，下载SHA核验完成。**更正：前轮Mac-only只是当轮范围，不代表人类撤销SSH/只读授权，不需要重新申请SSH。** 本轮未新建/续期设备预留、加载模型或运行NPU；普通push/更新现有draft PR2授权保留。旧P11/D12/CID/holder仍只作退出归档，不复用、不据此判断现在设备空闲。

当前候选收窄为“把D初始有效query从最后prompt推进到P已知首输出y1”。**有条件可行，尚未证明设备状态齐全；不接受保持原MTP5即可省掉完整模型迭代的推断。** 原生P元数据及D cold external hits支持声明接收N，D接收晋升会为重采样把computed=N回退N−1；采纳y1为output history后可保持computed=N、总tokens=N+1。但NewRequestData无初始output IDs，worker新建CachedRequestState固定output_token_ids=[]，必须显式交接该状态，不能只加scheduler计数。

原生MTP5首批padding仍6行/请求，C4可24行；**首批6→6不证明整请求绝不可能少一次target forward，最终须按同预算/同采样完整请求实际前向次数判断。** 源码路径追到AscendEagleProposer→AscendSpecDecodeBaseProposer，DeepseekV32MTPModel注册为AscendDeepSeekMTP。glm_moe_dsa有1个nextn层，K5复用该层。若P首轮确实执行并正确转移，MTP尾行N−1应由y1+h(prompt最后)形成，推进后D用新hidden与y2在N继续。现在受检18文件（含factory/config/proposer/model/executor/mixin等）已有两台实际镜像逐文件SHA配对，名单外文件如frontend stats仍只为固定提交参考。**实际proposer对象、调用/slot/完成依赖和尾行内容尚未NPU观察**；不能从源码或aggregate finished/101条目推出物理状态齐全。旧D会重算该行，既有正确性不能单独验证跳过它。

CPU验证层级要精确：完整原生Request导入因缺兼容torch失败，msgspec/vLLM也缺。只执行固定SHA的原类定义/方法段及8条真实存档token IDs；构造器、SamplingParams、真实KV manager、InputBatch/GPU/MTP没有验证。有限反例还确认：推进后不能保留v1“核对并抑制重复y1”协议，新D首结果为y2时会误报mismatch或误删；原4f42984代码块与真实存档参考ID20→198的检查已复现。把外部P token当普通Async完成会将0placeholder减为−1；在KV晋升回调内结束max1会被外层重新设WAITING。原生append/check_stop的计数检查不等于资源回收证明，不用自洽模拟冒充原生全链路。

报告口径更新：TPOT=(最后token−首token)/(输出数−1)，首token提前且后续不动会机械抬高它。原v1八组TTFT倍率0.80080、第二/最后到达1.00424/1.00491、首二gap2.46130，第二至末区间倍率1.00897；不能把平均TPOT1.12897叫稳定decode变慢13%。八组等权均值ΔTPOT1.48368ms，按首/末时刻分，首token前移贡献1.29091ms（87.0%）、末时刻0.19277ms；按首二gap/后段分，则为1.30817ms（88.2%）和0.17550ms。88.2%包含第二token时刻变化，不能与87.0%混称；后段仍受不同D实例、续写、batch/传输影响。draft3183→3240是不减工作证据，不等同target kernel数。

负phase已找到固定提交源码机制：提前输出被frontend当首次prefill输出，记录first_token_ts并置is_prefilling=False；异步KV接收时尚未记录SCHEDULED，随后真正调度才填scheduled_ts，完成时first_token_ts−scheduled_ts成为负。原方法的定义级显式时间fixture已得到−0.25秒，普通顺序+0.25秒；没有运行完整frontend事件或exporter，新stats源码也未镜像SHA配对。16/16原负增量保留，不能写成修复通过。后续需分开发布与计算phase事件，不能截零或伪造调度时刻。

下一步已收束为[实现的窄范围尾行探针](runtime-patches/state-tail-probe-v1/README.md)，不再停留在方案。单份worker overlay默认关闭，原生函数体AST保留；P读取前先保留原native event等待、核对writer/constructor stream和target→proposer→copied hidden尾部、移位token/position及每层真实slot；D在原生READ/futures返回后、finished入队前按实际P rank/layer/block配对取样，早于D重算。仅支持本模型packed C8与split Indexer K/scale实际view约定，记录group/spec/shape/stride/alias，取N−2/N−1，不新增kernel/clone/全局同步或传输协议。客户端固定cold C1 N256/257、P K1/D K5、P/D min1与输出1/2；若P16个rank的MTP调用/hidden/完成依据缺失，在发D前停止。成功只能说末两行匹配，不证明完整前缀；先用能提前否定候选的小检查，不把全前缀证明塞进首探针。

边界依据：Mooncake wait_for_save为空；原生Async构造时copy stream等待当前stream，get_output等待已有event；实际镜像executor在KV aggregator存在时收齐各worker并解包Async输出。另补3份官方固定提交参考源码确认aggregator初始化仅以scheduler connector存在为条件，factory无producer分支；这3份尚未镜像SHA配对，实际仍以16rank记录为门槛。探针记录原等待前/后event状态，先确认覆盖MTP writer才CPU读回；原等待前pending不等于race，不新增等待补造就绪依据。D使用原生同步READ API完成契约，仍不是kernel-completion trace。输入缓冲若已被下一正工作复用、缺slot/component/完成依据或布局不支持，记录refused/incomplete，不以任意非零或纯相等字节认定有效。

本机已实现并检查构建器、hash路径stager、fresh lease计划生成器、两请求客户端、行配对器和own-CID保全停止脚本，复用既有collect_node/start_capacity/backup_artifacts。Mac验证使用真实NumPy storage/stride、匹配源码接口和合成配对记录，包含空/错位/错源rank/块/组件、无writer/完成依据、UID替换和过期lifetime拒绝；**不是完整原生或设备回放通过**。构建与实际模板路径在state-tail-probe-v1/prepared-artifacts.json。review-plan仍是未绑定fresh lease的非launch schema；有新窗口及两枚匹配Pod收据后生成可执行plan，starter重查精确UID/剩余时间。当前没有申请/执行新NPU窗口。

预计下一次新窗口两台各16die、一次加载、只这两个原生功能输入。完整历史启动→API ready P11约626.893秒、D12约636.642秒，而非仅load-start后201/213秒；含取证/保全预计18–22分钟、hard max25分钟、15分钟未ready即止，不续期，保留原900秒启动下限和≥5分钟清理余量。finish只停本次精确CID并保存原件，下载SHA核验后按精确新UID释放holder；旧lease/CID/D13/硬编码stop不复用。若尾行判别通过，完整前缀、history交接、max1/取消和同预算整请求前向次数仍为后续问题，不自动扩大本窗口。

本轮证据入口：evidence/20261018-p-first-state-audit/image-identity-live-v1/summary.json（实际两台18/18）与state-tail-probe-v1/（构建/本机检查/模板），代码runtime-patches/state-tail-probe-v1/。此前fixed-source-followup、probe-preparation、native-contract、原始接收计数和时序原件均保留；过往only-reference/未实现状态是当时快照，当前以本HANDOFF与最新收据为准。Mac固定源码归档仍在glm53-execution-evidence-backup/20261018-offline-state-audit/source及20261018-fixed-commit-followup。

官方MTP/已有overlap/graph两种合成负载持续2.29224×仍归官方基线完善，不作原创机制收益。发布v1的240/240 formal、925/925同arrival、有限数值/复用通过及不默认采用结论保留；原12算术仍11/12，完整精度、生产SLO/容量未验收，不同D实例/续写差异未消除。原现场结论、SHA备份与资源退出见evidence/20261017-official-pd-optimization/p-first-publication-verdict-v1.json及finite-window-closed-summary-v1.json。不主动向其他聊天发消息，主控直接读本文件。
