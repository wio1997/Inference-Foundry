# GLM — HANDOFF

2026-10-01持续自主执行。研究、开发、执行/raw在服务器，Mac仅SSH入口；真实仓库166:/data/tiankuan/wio/Inference-Foundry，研究分支glm5-3-autonomous-20261001。源码/Run按GitHub记录和server hash恢复，禁止恢复旧队列。

- **Current**：没有正式重复KEEP，稳定容量/上下界unknown。Run3新的c2正式长测4/4请求，每条81932实际输入/61440输出有效；TTFT(first SSE) P50 6.0588s、TPOT P50 22.9ms，未达4s/18ms。[Run3](records/points/GLM-OPT-0001/runs/GLM-RUN-0003/summary.md)。四样本不能证明P99/稳定服务。
- **测量修订**：Run1/2旧parser漏delta.reasoning，把较晚的正文content当TTFT。Raw不改，新first_output_reduction按原SSE离线重算；Run1真正首输出34.509s/TPOT29.961ms，Run2 c1首输出2.324/2.222s、TPOT24.615/21.927ms左右（精确值看文件）。不能再引用14.625ms为Run1 first-token TPOT。AISBench已解析reasoning；Run3 first-SSE SLA失败仍成立。
- **活动点**：[GLM-OPT-0002](records/points/GLM-OPT-0002/point.md)。完整请求router/lease/drain、开放到达loadgen、native兼容PD阶段overlay及optional opaque routing trace已开发；24 CPU合同通过。未改模型/算子计算，不做精度/量化比较。
- **真实功能**：[Run5](records/points/GLM-OPT-0002/runs/GLM-RUN-0005/summary.md)新7推理请求1296有效token、expected400及独立cancel通过；复用Run4有效本地P/D128+128。mixed lengths/open arrivals、stream/nonstream、chat/completion、合法n2(.7 temperature)、active drain/代际readd、取消后lease与native idle均验证。仅诊断，不提升Current。P eager/MTP1完整decode慢（P-only32 TPOT约239.8ms），D graph/MTP3约21.9ms，不当同负载代码收益。
- **无效证据保留**：Run4错误预期greedy/n2成功，原生合法拒400；Run6 P JSON默认展开多括号导致CLI失败，P未加载，D启动；Run7 readiness读旧P错误日志早退，实际修正P已启动。均INVALID，不归为模型/硬件失败；source/state/log保留。CLI命令包装为明确Bash，禁止Zcode改写Result；失败Job不覆盖。
- **当前执行**：Run8接管Run7实际P与Run6实际D，核验boot/PID/startticks、raw proc argv（ps文本不能shlex解析JSON）；两端目标TP16/EP16/DCP16、Graph FULL_DECODE_ONLY/MTP3、batch16384，保留producer/consumer及native工具/reasoning API。无额外重载。controller PID3185709，boot6d9cf06f-a02d-49ef-b7fe-cc456a7b2025/start259814786；状态以Run8/state.json、task controller-owner.json为准。只读monitor PID3213349，Job MONITOR-RUN8-20261001T1101Z，每45秒读P/D日志。PID必须核验身份，不能按文档盲操作。
- **执行合同**：Run8独占两个task锁，等待两端健康→identity→有界本地/完整路由/drain/cancel E2E→旧PD81932→256兼容→finalize；阶段spec/source冻结，不改其运行依赖。当前D已健康、P仍初始化，未宣称Run8通过。Controller失败/owner消失不重放，按确切phase/模型身份裁决。
- **研究空间**：[source_notes](records/points/GLM-OPT-0002/source_notes.md)证明DP2TP8EP16专家跨DP/TP分片，不用checkpoint/8否定；runtime fit/性能未测。默认PP39边界有topk buffer跨stage依赖，38/40只是待验证候选。两完整实例合法执行路径已证实，下一依据Run8功能/性能及动态负载选择高价值实验，不固定P/D比例、shape或扫描步骤。

现场deploy:/data/tiankuan/wio/glm52-pd/deploy；glm52-single；P166:9081，D167:9900，PD proxy166:8000，temporary full router8002由Run启动/退出。连接/操作沿用现场HANDOFF/SSH_AND_OPERATIONS/PD_START_AND_TEST_GUIDE；localhost/内网HTTP显式绕过宿主机代理。Legacy lifecycle可绕过新锁；仅唯一controller排程本任务资源。原始日志/权重/凭据不提交Git。新controller允许接管已核验现役模型，不把旧未知阶段当可重放队列。
