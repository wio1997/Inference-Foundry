# GLM-OPT-0002 — 完整请求部署与动态placement

状态ACTIVE；性能INCONCLUSIVE，无正式KEEP/稳定容量或可行域极限证书。

当前1P1D两端持完整模型，P长期闲置；Run3 c2完整长输出仍未达TTFT/TPOT SLO，c1诊断TPOT也不稳。研究在现有两台内利用已合法驻留模型的完整请求部署，native engine保留KV/MTP/Graph/提交和sampling状态，不改算子计算或精度。

代码：完整请求透明router、唯一worker placement lease、native状态/raw payload、一次回收、drain/代际/故障隔离；未知成本用active count，全部候选有速率提示才比较估算秒。开放到达loadgen把计划到达纳入延迟，usage/finish/DONE必须齐全，negative/cancel不算有效推理；包含native reasoning/可见text首输出。可关闭的opaque router trace关联lease、body SHA、目标代际、header/firstbytes/release，不改变payload。Stock PD阶段overlay是独立观测候选，未替换现役8000。

真实Run：

- Run4：两端本地334→128均有效；mixed四条成功720token/native400正确。错误测试将greedy temperature0/n2预期成功，导致INVALID；原生SamplingParams要求greedy n1，不能归为router失败。D reasoning首输出测量遗漏已修正后续代码。
- Run5：复用Run4有效本地P/D请求，新7推理请求1296token及expected400、独立cancel；variable open arrivals、stream/nonstream、chat/completion、合法n2 temperature.7、active drain、generation readd通过。Cancel lease释放后native P running1→0约观察1s，P/D waiting/running均0，gateway8002退出。仅功能诊断，完整rare branch/稳定服务仍未知。
- Run5 P-only32输出TPOT239.775ms，D drain512 TPOT21.879ms；不同配置/输入，不是代码收益。证据支持让P由eager/MTP1转为Graph/MTP3，D提高prefill预算以支持完整长输入服务，再检验PD兼容。
- Run6：候选两端Graph/MTP3、batch16384。P Bash JSON默认展开多括号，CLI解析失败，未加载P；D已启动。核验controller boot/startticks后取消，保留INVALID，没有误判硬件或模型。
- Run7：P literal JSON已通过CPU实际shell argv展开验证；接管D、实际启动P。Readiness误读启动前旧P日志提前退出，INVALID；没有中断两端模型。
- Run8：核验两端实际PID/boot/startticks和raw proc argv，接管既有Graph/MTP3启动，不重复加载；等待健康后执行相同native capability及旧PD81932→256兼容E2E。最新state为准，尚未宣称通过。

源码/存储依据见[source_notes](source_notes.md)：DP2TP8EP16的专家跨DP/TP分片语义成立，不用checkpoint/8否定，也不证明runtime fit。PP默认39/39从共享索引层开始，IntermediateTensors不传topk buffer；38/40是待验证候选，不是正确性结论。旧7月TP8DP2 profile为线索，版本/量化/关联D缺失，不继承数值。

24 CPU合同测试经真实zcode --prompt/Result/bridge通过；TRACE-CONTRACTS-20261001T1040Z/test.stderr3706B SHA13c846203bd7edefb93d5c4b364521dbdfcaa52c57e0351b37582318e2e7caec。真实E2E是独立裁决，fixture不是性能证据。ROUTER-TEST-0937 Result及REDUCE-RUN3-1022失败均保留，未改raw或悄悄提升Current。

未知：跨配置/负载的达标稳定容量、可信上下界、对齐配置后的Graph/MTP/缓存/observer成本及完整native rare branches。路线按Run8及后续动态请求证据决定，不预置固定步骤或审批门槛。
