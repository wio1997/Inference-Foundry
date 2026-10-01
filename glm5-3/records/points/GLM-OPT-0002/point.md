# GLM-OPT-0002 — 完整请求部署与动态placement

ACTIVE；性能INCONCLUSIVE；Current未提升，达标稳定容量和全可行域上下界unknown。两台都持完整GLM模型，native合法本地执行已证实；研究完整请求部署与PD兼容及动态调度，KV/MTP/Graph/采样/提交合同由native保留，不改算子计算或精度。

实现：透明raw请求/响应router、单worker精确lease、drain/代际readd/取消回收、native HTTP及流内错误隔离、开放到达loadgen。输出计数依据usage+finish+DONE；reasoning/content均可作为真实首输出，不以SSE chunk当token。Router trace默认关闭、opaque lease/body SHA/首字节/真实首输出；显式测试header ID解决相同body哈希碰撞。PD overlay独立未替换stock8000。

Run4/6/7/10/13为测试或driver INVALID，原始证据保留：greedy n2错误成功预期；P JSON shell多括号；readiness旧日志；flag tuple key；event(name,name=)冲突。Run13已完成有效P控制单列复用，未把整Run当成功或重跑旧队列。

Run5/9/11/12完整功能子合同验证：mixed open arrivals、stream/nonstream、chat/completion、合法n2温度.7、expected400、active drain/readd、cancel后lease/native idle。Run12新9推理1552输出加PD81932→256有效，10请求1808输出。之前capability_summary旧硬编码P eager/MTP1文案以实际argv/manifest修正，不改raw。

瞬时显存：
- Run8两端Graph/MTP3/batch16384/gmu.92，D mixed在MTP merged draft DCP recv520MiB OOM；三D请求HTTP200后SSE500无usage/finish零信用。REJECT具体配置/负载。
- Run9 D恢复128后子合同有效，但P PD draft DCP gather contiguous1.13GiB OOM。
- Run11 P gmu.87/batch16384启动后KV1.96GiB，后续draft recv1.01GiB仍OOM（NPU3 free1005.91MiB）。gmu headroom单独调整不够。归约v2纠正cold/Run9标签而保留原hash和真实数据。
- Run12只P恢复batch4096/gmu.87：KV3.17GiB/296902tokens，capability及PD通过。D保留Graph/MTP3/batch128/gmu.92/KV6.64GiB。有限合法合同通过不证明完整容量。

Run14复用P13控制，新D81932→2048控制和六动态请求9088输出有效，wave84.131token/s。P控制首输出1.644s/TPOT20.509ms；D控制8.119s/20.588ms。D预填充预算128，medium期间短请求等待首输出6.562s，支持phase机制假说。

Run15 A/B/A与Run16 B/A/B，同config/source/body/sampling/arrivals，全部36请求54528新输出有效，终态idle/lease0，gateway退出；每波9088输出。opt-in prefill_aware先pending-first-output lease数，再原active/完整校准score；默认active_count不变。短请求arrival10时P active2/pending0、D active2/pending1：A去D首输出6.663–6.778s，B去P0.684–0.736s，三次各自复现。这是局部QoS机制，不是native prefill完成断言；unknown压缩/超限/非流保持pending至release，可能集中decoder。
总吞吐收益未复现：Run15 A86.287/91.063、B93.961；Run16 A96.920、B80.200/81.984。无preemption，各波native cache命中与MTP acceptance均原counter归约；不同acceptance/尾程速度不能随意归因代码。两Run INCONCLUSIVE，无KEEP；有限2048/4096输出不能替代61440正式合同。

35 CPU合同真实zcode --prompt/Result/bridge通过，PREFILL-CONTRACTS-20261001T1244Z/test.stderr5216B SHA62b420cc382e6522a40017f3cc23a1e73cd599568498188ec1f070edc0283a13。29原合同包含真实Run8 SSE错误所有byte boundaries、原样wire、避故障/zero-credit；新增phase/默认count/伪造与旧代际lease/首reasoning与tool/text/malformed observer。CPU不是性能KEEP；原failed Job不覆盖。

当前由Run21已完成的严格full证据推进Run22：只D原生prefill预算8192、P21预算4096保留。新seed匹配prefix/tail原body和native cache/prefill/usage/idle，以及PD/finite mixed瞬时安全决定后续。Dynamic-K原生接口已CPU确认但短K会丢失当前maxK decode Graph键，需覆盖执行成本；metadata/DP2/PP仍候选，无固定扫描队列。

Run17有限D4096/.87验证：旧D128/.92停止、P12保留。新17推理12944输出、native cold81932→2048/capability/PD/dynamic均有效，Dynamic long0 TTFT1.6794s（旧约8s）、medium1.2971s（旧13s）、short1 .8231s（旧6.7s），wave80.0026token/s未证明吞吐提升。D KV3.16–3.17GiB/296902tokens，权重47.2182GiB、activation.35、non-torch2.58、Graph.86；runtime fit仅该有限合同。

Run18在任何推理前reset端点404，driver INVALID，零新输出/原生无失败。Run19新正式4条81932→61440/closed c2已终态INVALID，保留现役缓存+原生counter观测；两副本前缀预热有效。新增可选raw wire audit及有界离线usage/finish/DONE/全部attempt复验；49 CPU合同通过，raw/observer成本在E2E内。DP warmup重复prefix原有索引/计数缺口已修正且DP1不变。没有正式重复KEEP、stable capacity或全可行域界。

Run19新增功能证据：四native usage/DONE原流均结束且无OOM，三条严格length通过，一D条无tools仍解析出316479字符函数名和321629B末帧/tool_calls。bounded observer unknown、strict验收失败；三条仅部分184320输出，全部generated245760不成为完整有效TPS。原合同不放宽；AISBench CLI0/success4不能替代native功能，Current不升。Run3旧证据没有该原wire，新增合同下不得事后认证；保留历史计数/SLA口径。已terminal/dead、lease0/native idle/gateway退出。

CPU原型glm47_contract保持原生工具能力，缺tools或choice none启用native skip_tool_parsing保留文本，避免未请求工具识别/缓冲；提供tools走原native，非vendor patch/模型算子变更。真实tokenizer复现stock无tools误识别与tools-none文本丢失，候选流/非流/ID与字符分片及316k文本均通过；auto工具JSON仍正确。Run20仅D重载K5+插件，P12 K3保留；真实API/PD/动态成本验证已完成（Run20），组合部署差异不伪装isolated gain。K3固定601.659s窗口P/D第三接受.729/.962支持研究深度，但新增草稿/验证/通信成本和动态低acceptance效应未知，无容量界/KEEP。

Run20 finite：21新推理13037输出、实际工具API/PD/capability/dynamic与终态清空均有效。D K5 KV296902/3.16–3.18GiB、Graph.92实测fit。Dynamic78.4769 vs旧80.0026无总体收益；冷同body2048 TPOT22.814vs25.451ms只是单次且normalized channel hash不同，不作精度质量/same-output因果评价。Cold K5 draft+accepted2051而actual2048，裁剪3不计信用；dynamic draft计sequence不是device round。Run21新正式合同：仅P重载同K5/插件、D20保留，P工具API/PD与双prefix预热已过，四完整61440 E2E按Run19原严格判据执行；所有收益与容量待真实结果，Current不升。


- Run21 completed/dead：四条81932→61440严格原生usage/length/DONE/error/unknown/全部attempt/hash重放均通过，245760有效输出；full CLI2454.551585s/100.124194token/s。TTFT(first SSE)P50 6003.388667ms、未舍入TPOT P50 18.006747585ms，原CSV判据仍FAIL。四样本/单正式轮INCONCLUSIVE，无重复KEEP/稳定容量。
- Run21 full-only sample6→final：各副本queries163864/hits143360，external queries20504/hits0，generation122880/preemption0；每请求平均71680命中、10252需计算。P/D native prefill两条合计11.227994/11.403356s，queue合计44.232/34.512微秒；native TTFT合计11.711228/11.878820s，gateway真实首输出5.930–6.018s。原生host计时支持prefill gap，不是device kernel归因。P drafts29886/accepted92999，overshoot5不计输出；D26359/96521/overshoot0；dynamic count为sequence而非engine round。
- **活动Run22**：唯一controller PID2066773、boot6d9cf06f-a02d-49ef-b7fe-cc456a7b2025/start261953222，20项源码/spec SHA2ba81b1c1cbd8e705f778b67f16172fcd629f8f21956639adad3beb48e89c0cd冻结。Run21终态/dead后仅D20 PID3900500精确停止且旧整log已保存；新D PID3307628预期budget8192/.87/K5/Graph FULL_DECODE_ONLY，startup/readiness待完成；P21 PID3646900/原身份、budget4096/.87/K5保持健康。真实zcode launch-v2通过；原launch Job因work_type协议校验在CLI/controller之前拒绝，原记录保留。
- Run22问题：试减少10252未命中token的prefill chunk数，同时测native KV/显存及mixed负载代价；原算子实现不变，不把budget提高当免费收益。新datasetseed20261002的73740→1和两个81932→2048 body分别发两副本，逐pair原body hash一致/usage长度/原生before-after counters/idle；requestseed omitted/native1024。旧cold/PD/六dynamic原body按原诊断seed复用；不恢复旧队列、不计旧API/cancel/n2为新结果。当前尚未完成，Current无KEEP。
- 只读monitor PID2111841，MONITOR-RUN22-20261001T1701Z，真实zcode/Result/bridge，正常采样45s、无推理/服务动作；原Run21 monitor已终止。
