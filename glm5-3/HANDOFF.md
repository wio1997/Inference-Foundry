# GLM — HANDOFF

2026-10-01持续自主执行。研究、开发、执行/raw在服务器，Mac仅SSH入口；真实仓库166:/data/tiankuan/wio/Inference-Foundry，研究分支glm5-3-autonomous-20261001。源码/Run按GitHub记录和server hash恢复，禁止恢复旧队列。

- **Current**：没有正式重复KEEP，稳定容量/上下界unknown。Run3新的c2正式长测4/4请求，每条81932实际输入/61440输出有效；TTFT(first SSE) P50 6.0588s、TPOT P50 22.9ms，未达4s/18ms。[Run3](records/points/GLM-OPT-0001/runs/GLM-RUN-0003/summary.md)。四样本不能证明P99/稳定服务。
- **测量修订**：Run1/2旧parser漏delta.reasoning，把较晚的正文content当TTFT。Raw不改，新first_output_reduction按原SSE离线重算；Run1真正首输出34.509s/TPOT29.961ms，Run2 c1首输出2.324/2.222s、TPOT24.615/21.927ms左右（精确值看文件）。不能再引用14.625ms为Run1 first-token TPOT。AISBench已解析reasoning；Run3 first-SSE SLA失败仍成立。
- **活动点**：[GLM-OPT-0002](records/points/GLM-OPT-0002/point.md)。完整请求router/lease/drain、开放到达loadgen、native兼容PD阶段overlay及optional opaque routing trace已开发；24 CPU合同通过。未改模型/算子计算，不做精度/量化比较。
- **真实功能**：[Run5](records/points/GLM-OPT-0002/runs/GLM-RUN-0005/summary.md)新7推理请求1296有效token、expected400及独立cancel通过；复用Run4有效本地P/D128+128。mixed lengths/open arrivals、stream/nonstream、chat/completion、合法n2(.7 temperature)、active drain/代际readd、取消后lease与native idle均验证。仅诊断，不提升Current。P eager/MTP1完整decode慢（P-only32 TPOT约239.8ms），D graph/MTP3约21.9ms，不当同负载代码收益。
- **无效证据保留**：Run4错误预期greedy/n2成功，原生合法拒400；Run6 P JSON默认展开多括号导致CLI失败，P未加载，D启动；Run7 readiness读旧P错误日志早退，实际修正P已启动。均INVALID，不归为模型/硬件失败；source/state/log保留。CLI命令包装为明确Bash，禁止Zcode改写Result；失败Job不覆盖。
- **Run8真实失败**：[Run8](records/points/GLM-OPT-0002/runs/GLM-RUN-0008/summary.md)两端Graph/MTP3/budget16384/gmu0.92本地334→128均通过，但合法mixed的三个D请求HTTP200后原生SSE500/DONE、无usage/finish，零有效token；P两个mixed请求128+16及expected400通过。D_failure.log4918/5012/5014首次NPU14申请520MiB DCP recv buffer OOM，位于MTP merged draft；随后多rank OOM/EngineDeadError。Canonical verdict REJECT此配置/负载；未改capability_summary原generic INVALID。不是非法测试合同或硬件容量极限。
- **活动Run9**：唯一controller PID36744，boot6d9cf06f-a02d-49ef-b7fe-cc456a7b2025/start259973164；只读monitor PID79322，Job MONITOR-RUN9-20261001T1129Z。Run8 controller已核验terminal/dead；P保留Run7原owner，idle后仅恢复D到原生batch128、Graph/MTP3/gmu0.92，source/hash核验。Run9 state.json/task controller-owner为准，不能按文档PID盲操作；截至11:38Z两端健康、identity完成、capability正在真实执行。然后按已冻结spec执行PD81932→256兼容，不重放旧队列。cache条件P温热/D重启，不是隔离代码速度比较。
- **代码缺口**：Run8 router原样转发错误但只按HTTP>=500/transport标记backend_failure，原生HTTP200 SSE500没触发cooldown。隔离candidate目录正在实现有界增量SSE错误观察，保持所有wire bytes、不重试、标记failed replica；loadgen显式native_error且即使后续usage/finish也不计有效输出。Run9冻结依赖不改；候选CPU合同与真实native部署分开，测试通过不代表E2E/KEEP。
- **归约诚信**：Zcode REDUCE-OOM8原Result finding scope未通过bridge，CLI0及原大归约保留；GPT定向核验首OOM/全部请求，CURATE-OOM8命令Job真实通过，canonical reduction带GPT audit，不将chunk数当token或target eager当draft eager。现场原始文件不改写。

- **研究空间**：[source_notes](records/points/GLM-OPT-0002/source_notes.md)证明DP2TP8EP16专家跨DP/TP分片，不用checkpoint/8否定；runtime fit/性能未测。默认PP39边界有topk buffer跨stage依赖，38/40只是待验证候选。两完整实例合法执行路径已证实，下一依据Run8功能/性能及动态负载选择高价值实验，不固定P/D比例、shape或扫描步骤。

现场deploy:/data/tiankuan/wio/glm52-pd/deploy；glm52-single；P166:9081，D167:9900，PD proxy166:8000，temporary full router8002由Run启动/退出。连接/操作沿用现场HANDOFF/SSH_AND_OPERATIONS/PD_START_AND_TEST_GUIDE；localhost/内网HTTP显式绕过宿主机代理。Legacy lifecycle可绕过新锁；仅唯一controller排程本任务资源。原始日志/权重/凭据不提交Git。新controller允许接管已核验现役模型，不把旧未知阶段当可重放队列。
