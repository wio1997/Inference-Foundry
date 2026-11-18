# SGLang epoch832：恢复与匹配验收入口

本轮继续固定daily A2镜像7da963869ee5，在同两台910B1/aarch64节点恢复P/D。P沿用603、D沿用501的数学实现和原生启动参数，[准备清单](../evidence/full_model_G832/preparation_manifest.json)逐文件确认Python math SHA及argv相同；变更仅为owned生命周期控制、run目录和profiler输出环境变量。准备文件中的actual_deployed=false是准备时状态，实际状态见[当前metadata](../metadata/full_model_G832.json)。

[P实际启动](../evidence/full_model_G832/P_actual_start.json)已完成，并在[现场只读检查](../evidence/full_model_G832/P_actual_ready_and_step_span_source.json)确认controller仍存活、/ready=200。P启动中warmup线程在监听端口建立前出现connection-refused，随后服务实际就绪；该警告不据此抹除，也没有为此重启实例。D目前只有SHA核验完成的payload，尚未启动；原生vLLM D保留等待独立CPU解析对照。

TP8/EP8/PP1、P chunk8192/eager、D chunk1024/full target+draft graph、mem .88/.80、maxseq16、ctx263296、DSPARK5 draft加root、prefix/overlap保持。SG的BF16 KV/FP4 index与native vLLM A2的BF16 KV/INT8 index加FP16 scale不等价；CANN和torch_npu组合也不同，后续对比须披露。历史112组件及7个完整PD选定case通过不替代本epoch完整质量或性能验收。

固定镜像原生profiling已支持detailed_annotations，从CPU长度镜像构造extend/decode/target-verify的query/KV聚合，source SHA与选定源码见现场证据。计划直接使用该入口，尚无新实际trace。不能以source支持或/start_profile返回200宣称所有rank设备窗口成功。

后续顺序是受控释放owned native D、启动SG D，核对双端/ready、源码、参数、8卡owner与图执行，再用同一官方1319题/提示/严格grader/预算完成质量对照。之后32K/64K/128K/256K均以5轮独立充分预热和首cold/replay原始请求验收；诊断profile复用native冻结的16条数据。任何未通过或未执行项继续保持未验证，完整跨框架差异地图与端到端收益尚未完成。

## D实际启动与同题集准备

后续已完成原生vLLM全量D0在无NPU设备CPU路径的覆盖对照，受控释放原生D八卡后，[SG D832实际启动](../evidence/full_model_G832/D_actual_start.json)。这是启动记录，尚未凭此声称D就绪、实际图重放或完整PD通过。

[质量请求worker AST检查](../evidence/full_model_G832/matched_quality_worker_AST_check.json)确认SG与native完整请求worker一致：原始1319题、同提示、T0/seed0/max2048、C16、thinking=false和无重试。继续引用同一官方冻结数据/严格grader原件；[已部署源码](../evidence/full_model_G832/matched_quality_source_staged.json)只更换框架URL、输出目录和镜像标识。[固定镜像聊天源码](../evidence/full_model_G832/fixed_image_chat_thinking_contract.json)也确认chat_template_kwargs的thinking开关。源码匹配不等于实际模型质量通过，必须完成双端就绪与原始请求/响应的全量评分。
