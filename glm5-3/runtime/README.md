# GLM runtime原型

现役源码与版本依赖只从[CURRENT_PRODUCT_MAP](../CURRENT_PRODUCT_MAP.md)进入，部署/安装源与证据位置查[RECOVERY_INDEX](../RECOVERY_INDEX.md)。下文原型、Run5/Run28、端口及启动说明保留为历史知识，不能用来推断当前部署或直接启动服务；当前现场按HANDOFF→manifest→state/process只读核验。

checkpoint158之后按[AGENTS](../AGENTS.md)研究，本文旧Run/启动状态/待验证动作仅作历史证据，不能机械继承。`replica_gateway.py` / `placement.py`及prefill-aware、prefill-work、work-seconds对完整请求副本的调度属于Full Replica / Complete-request Placement：各副本完整Prefill+Decode，不能称标准PD优化。标准PD为Request→P Prefill→KV Transfer→D Decode→Output。历史路线进产品须新裁决；Sol亲读代码/diff/profile/raw，经Correctness、matched A/B与真实完整E2E重复收益才计代码KEEP/提升Current。模块增加而Current不升或API/driver/compatibility支线长期占主线须Goal Review；历史实现/Run/raw/裁决保持原样。

当前native engine仍拥有KV、MTP、Graph、sampling与输出提交状态。原型在两台现有910C上研究完整API服务、PD阶段及动态请求排程；Run5有限native capability已通过；尚无正式KEEP或稳定服务容量结论。

- `controller.py`与`phase_runner.py`：task双锁、boot/PID/start ticks、心跳、argv子进程与真实退出。阶段来源被hash锁定；未知中断不重放。直接legacy lifecycle仍可绕过锁。
- `glm_gateway.py`：按SHA锁定当前stock PD proxy，保留其完整接口与行为。默认trace关闭；GLM_TRACE_PATH启用首字节/结束、P请求、D流与实例释放观测。不能把proxy首字节当模型首token。
- `replica_gateway.py`与`placement.py`：完整请求透明转发、native状态/raw响应、一次lease回收、drain、代际、暂时故障隔离。一个调度worker；无需改变engine权重或KV。未校准以active count排程，全部候选有decode速率提示才比较估算秒数；提示不代表容量证据。
- `loadgen.py`：保持body不变，独立开放到达任务，计划到达是延迟起点。SSE必须完成且usage/finish齐全；negative不计有效推理。nonstream TTFT与multi-choice TPOT不臆造；有限窗口不证明稳定容量。
- `checkpoint_inventory.py`：只读safetensors文件头，归约存储字节及header hash。不读或hash权重payload，不证明运行内存适配。
- `monitor_run.py`：读取controller身份/阶段与现有D日志，不发推理请求，不重启，不重放。

使用task资源的真实脚本经唯一controller执行；CPU合成合同测试可以与正式Run并行。现场端口8000是现役PD，8001/8002只是候选端口，不能因代码存在就声称已经部署。宿主机代理不参与内网任务HTTP；curl显式noproxy，httpx trust_env=False。原始SSE、日志、配置与Result保存在服务器Run/Job目录，Git只保存代码与裁决引用。

完整请求路由先由`scripts/capability_e2e.py`在controller所有权下做有界真实验证，包括P/D本地生成、chat/completion stream/nonstream、n2、native400、混合长度/开放到达、动态drain及cancel后lease和native running/waiting归零。该脚本未证明所有tools/多模态/稀有native分支，也不评估模型精度。失败保留Run并按实际原因裁决；禁止将原型或synthetic fixture当性能KEEP。

Native SSE `delta.reasoning`纳入首输出，和`reasoning_content/content`同样处理。HTTP错误原body以base64留raw，不仅保存hash，便于定位合法拒绝。可选GLM_ROUTER_TRACE_PATH只观测opaque lease/目标代际/header/firstbytes/release，不解析或重写输出；firstbytes不是token级计时。Run1/2早期遗漏已按原SSE离线修订。

## Opt-in coupled DP control metadata prototype

coupled_dp_worker.CoupledMetadataWorker uses the native NPUWorker and installs instance-bound, source-guarded metadata methods from coupled_dp_metadata.py. Only Graph-padding predicates change: graph padding follows the mode synchronized across DP ranks; SP/oproj/embedding/draft requirements stay native. Default worker selection is untouched, vendor/operator/model implementations are not rewritten. The GLM v1 DP prototype fails on unsupported runner, architecture, or source drift. Native global token maximum, communication/masks and draft padding remain required. Run28 uses a versioned plugin directory and explicit --worker-cls;32 worker receipts and bounded mixed/native E2E are pending. CPU contracts are source/dispatcher evidence only, not NPU correctness or speedup.
