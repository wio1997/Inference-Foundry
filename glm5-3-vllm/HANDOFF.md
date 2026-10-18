# 当前交接 — 2026-10-18 query-only无净收益，继续草稿种子交接

最新决定：尾行全rank7488匹配后，query-only已实际执行。smoke4/4、max1回退4/4、原两输入64输出ABBA8/8逐token通过；2输出dispatch2→1，但原64输出仍13/14次不减。干净C1 ABBA24/24逐token通过的两道算术输入，TTFT倍率1.01566、last1.00282、TPOT0.99977，无净收益，当前原型不采用。四路24题数值均24/24，但PD control/adopt完整64序列仅14/24一致，不能宣称严格通用逐token等价。

8次客户端关闭后的8次立即复用通过；4次token后关闭每rank只提交3次target，4次headers后关闭0次。abort counter增量0且无显式abort日志，不宣称8次后端abort计数通过。D回idle/缓存0，P仍0.0031847缓存占用，P完全清理尚未证明。原件均在新snapshot-backups.json及work-summary/performance-summary-c1/quality-matrix-summary.json；模型仍运行，D trace已用adopt-trace/disabled关闭。

下一机制尚在准备：P MTP1已计算一个未验证y2草稿；D初批当前仍是[-1]*5。计划每请求显式导出已有草稿：pinned CPU D2H排在P writer stream、原Async构造的copy-stream wait之前，复用原event后读取，不加全局等待/clone/计算kernel。复用现有ModelRunnerOutput.spec_token_ids，KVOutputAggregator核对16rank，P控制metadata附未验证草稿；D history仍只含y1、computed=N，草稿填入原5占位首位，由原生验证器验证。P K1/D K5不变，不把草稿当确认输出。new DTO bool让P控制请求没有导出拷贝。新D2H需实际SDMA证据。

旧holder的PID1同为sleep7200，单独延长activeDeadline不会延长该进程。下一次必要代码重载时，先精确停止自有P/D CID、核实空闲，再按精确旧UID释放/重建新holder；新holder可用sleep infinity加有限activeDeadline，才能后续合理续期。当前P90分钟model watchdog注意到期，持续执行授权有效，不在阶段/提交边界等用户。下文“已启动未ready”等是较早快照，以本段和最新收据为准。

最新人类持续执行授权经主控转达：仅在172.16.12.32/33推进实质研究，核实归属后可自行创建、续期或重建本任务合理的有限窗口，尽量复用模型，保留watchdog与精确CID清理，不抢占/停止他人作业、不扩大服务器范围。此前仅Mac、不得新窗口、必须再次授权、25分钟且不得续期限制已撤销；阶段结束/失败/提交不等于目标完成，不因此停工。保持原模型W8A8C8/int8 KV/Indexer与给定算子，不新增算子，全部SDMA；控制HTTP/SSH另记。普通push/更新draft PR2授权保留；不主动跨聊天发消息，主控读本文件。

**实际尾行判别已通过**：原生P/D cold C1 N256/257两请求输出/usage/首ID通过。两端各16rank实际AscendEagleProposer/runnable命中；P全部MTP调用1、target/source/copied hidden tail相符、原native event完成依据齐全。按真实P rank/layer/block及component比对共7488条末两行，mismatches/incomplete/缺调用为空。只证明取样末两行，不证明完整前缀、query优化或性能。原件已Mac SHA核验，evidence/20261018-state-tail-device-v1/{comparison.json,snapshot-backups.json}；大原件在glm53-execution-evidence-backup/20261018-state-tail-device-v1/。P保持加载，旧D已按精确CID停止以装载新原型，退出收据D-replaced/。

**query前移原型已实现并在新D启动，未ready/未功能验证**：runtime-patches/p-first-adopt-v1/。build-v3 identity695977bbbda5...，在原KV完成后把y1真实append到history，保持computed=N，显式NewRequestData.initial_output_token_ids经RPC送worker，CachedRequestState从N查询y1。全局GLM53_P_FIRST_ADOPT=1且每请求kv_transfer_params._glm53_adopt_first=True才生效；先限N256/257、greedy/min1/ignore_eos、max>1，max1/unsupported保持原生。y1仅prepend到首个正常D输出wire，native历史/stop/placeholder只处理新D samples，不沿用v1重复抑制，不做提前发布。实际_target_model_forward入口的host-only记录用于整请求dispatch次数，非kernel数/设备时长；adopt-trace/disabled可停日志后测性能。先同一个D实例per-request control/adopt配对，不能因代码存在就宣称正确或提速。

**实际CPU检查通过**：evidence/20261018-p-first-adopt-device-v1/cpu-check-summary.json。固定实际镜像、无NPU设备/权重，完整导入原生Request/SamplingParams/CachedRequestState；真实MessageQueue.enqueue/recv用socket recorder保留初始history及独立copy。没有真实IPC/完整Scheduler初始化、KV manager或InputBatch/GPU回放。失败原件保留：只读tmp不可用、直接typed MsgpackDecoder不适用、MessageQueue接口假设纠正；最终native源码实际SHAf3e7a3...另存native-message-queue-source.json，不使用Mac其他checkout代替实际接口。下一验证仍以设备请求/真实trace为准。

**当前资源/进程**：两枚本任务holder各16die、7200秒，允许续期。32 UID5a039510-7d08-4b5f-a011-05c240975f61，P CID54ede546246bfb87642aa5717e9a11a29f42f4cd1c8443f10490556aa0eda823，P root=/data/wio/glm5-3-vllm-execution/20261018-state-tail/tail-c1-live1-20261018/node32（保留未重启）。33 UID932cf1d6-cb2c-4dd2-9741-426ed0c9242f，旧D4e1ec...已停；新D CID5e457720adb117e100a8cdca6e1f5bbee1936a3d46f044b50ae3a62dcbc22723，root=/data/wio/glm5-3-vllm-execution/20261018-state-tail/adopt-live1-20261018/node33，已有launch receipt在evidence/20261018-p-first-adopt-device-v1/D-launch/。新D模型watchdog为plan中剩余有限lifetime（<=5400秒），复用owned holder，无新租约；900秒ready门槛仅针对未ready实例，不误用已复用P的总存活时长。旧P11/D12/D13/holder均不复用。SSHcontrol=/tmp/glm53-active-20261018.sock，既有私有配置恢复，勿打印凭据。

**正在运行**：本机startup watcher /tmp/glm53-adopt-watch-live1.py（exec session40574），保存到adopt evidence/startup-watch-v2.jsonl及startup-outcome-v2.json。P已ready，新D正在启动。待both_ready后用run_pairs.py先max2 control/adopt，再64 token ABBA同实例全序列比较、max1fallback/取消复用；取实际D adopt-trace与scheduler adoption日志，核对首次query位置N/token y1/history1及全部rank完整请求dispatch次数。若失败保全原件分析修复，不停工等用户。自动请求driver /tmp/glm53-adopt-requests-live1.py（exec session19902）已排队，等待both_ready后依次smoke2、max1、64 ABBA、quality24，失败即停该批并保全分析。

时钟：Mac/API服务器与32/33节点墙钟不一致（Pod creation晚于node startTime数分钟）。旧初watch错误用Mac now减host launch，已停并留原件；当前watch用各节点自身elapsed，事件依赖用进程单调时间。fresh lease/starter按creation/startTime较早值对Mac计算，保守留清理余量。此次原生双端共同观测ready约648秒，各节点自身启动顺序仍支持约10.5分钟，不直接用跨机器时刻相减。模型timeout为进程相对时间。

研究限制保留：已知ID不等于KV已算，computed=N+1错误；只有显式history/DTO/worker交接才可前移。MTP5首批6行不决定整请求target forward数；必须实际计数。当前完整前缀未独立证明，末两行通过仅支持尝试有限正确性原型。max1目前明确走native fallback，不在KV晋升内结束而被外层复活；取消/复用仍需新原型验证。冷/热、C4与复杂采样未自动扩展。

历史官方基线及发布v1：MTP/已有overlap/graph持续2115/2115、输出率208.67/165.03对无MTP84.45/77.61，2.2922×归官方基线完善。v1正式240/240、同到达925/925，TTFT .8008但第二/最后基本不变，首二gap2.4613，不默认采用。第二至末区间1.00897，不能把平均TPOT1.129叫稳定decode慢13%；负phase16/16保留，固定源码算式解释不等于完整frontend修复。原算术11/12，有限正确性不等于完整精度/SLO/容量验收。详情在analysis/official-pd-optimization-20261017.md、p-first-state-feasibility-20261018.md，后者“尚未实现”是原快照，当前以本HANDOFF/新收据为准。

PR2 https://github.com/wio1997/Inference-Foundry/pull/2 仍draft，已推cc508eeb；本轮授权/窗口调整、设备证据和新原型仍在工作树，待实际验证后普通提交/push，不自动合并。不动主控维护的analysis/performance-envelope-20261017/。一次探针、一次提交或有限窗口不是原研究目标完成。
