# SGLang gen67 公共 Event 诊断组件

本包保留实际 gen67 的最小诊断源码子集及 SHA 绑定，提供 CPU 检查、raw/native 身份审计和历史扩展构建入口。它用于解释已采集的短诊断，不是推理优化或完整服务部署包。实际环境为 aarch64、910B1、TP4、CANN9.0.0、torch2.10.0+cpu、torch_npu2.10.0/git94f8a8e6；完整镜像、权重、输入和原始 profiling 另有证据入口。跨 CANN/卡组对照不能归因于单一框架。

`source-bindings.json` 逐件绑定任务内原件、复制件与 SHA。10 个运行模块与实际 gen67 部署清单逐件相等；动态加载的 `sg_layer_event_anchor.py` 也在其中。静态依赖扫描包括条件分支，不能证明实际命中。旧的非公共 Event 模式需要的 `sg_layer_task_anchor.py` 不在包中，只描述 `Q27_SG_LAYER_PUBLIC_EVENTS=1` 路径。`bootstrap-reference/sitecustomize.py` 仅供查阅原启动逻辑，单独放在不自动导入的目录。它引用的完整兼容 overlay、权重 gate、其他 profiler hooks 不在最小包内；不要将该目录直接加到服务 PYTHONPATH。兼容模块清单仅在 `deployment/sglang-manifest.json` 保留。

在仓外新目录运行 CPU 检查：

```sh
python3 -I -B -S check_cpu.py --output /absolute/new/cpu-output
```

检查逐源 SHA、语法和静态任务模块依赖。6 个历史 Event/capture CPU 用例、3 个 gen67 清理用例、4 个正式请求筛选用例、4 个 raw 身份用例、3 个 native 筛选用例，共20例。历史 gen44 capture 用例早于自动清理线程，因此检查器仅在这些旧用例中 stub `arm_owned_graph_cleanup`；另3例调用实际 gen67 函数，以同步线程、torch/NPU/stream/executor/graph stub 核对成功顺序、身份错误和 drain 失败。它们不证明真实线程竞争、设备同步、模型输出或性能。检查器还执行原构建 driver 的 CPU 静态检查；不编译或导入真实 torch。

`runtime/q27_sg_probe.py` 在框架模块加载后安装 hooks；`install_finder()` 委托现有 loader，因此完整部署仍需正确的兼容 overlay 和冻结环境。公共 Event capture 在确切第三次 forward、确切 shape key 上记录64层×pre/post；回调结果在 capture 关闭并 drain 后读取。对象与 Event 引用随确切 graph 保留。raw 审计使用同 capture receipt、capture_stream_info 和唯一 EVENT_RECORD 节点绑定 Model/Stream/Task。native 审计通过显式 flow 找 replay 端点；无 flow、歧义和跨边界 rows 保留未知。端点区间不代表独占 kernel 或全 stream 关键路径，四 rank 不相加为墙钟。

正式选择器 `selection/filter_formal_replays.py` 使用客户端 requestID→manager RID 显式关联、work_key 和四 rank start/end pairs；不按事件顺序、时间或 token 数猜测。gen67 干净控制还依赖原 controller 的 health bypass 与 observer 静默5秒；该 controller 不在本包中，选择器自身不能建立这些前置事实。历史 gen50 startup/formal 重叠的限制继续保留。

清理只有在外部 controller 确认最后请求 EOF 与 idle 后，才写入带任务 ID 和确切 worker PID 的独立 `Q27_EVENT_CLEANUP_REQUEST`。运行模块检查这个请求身份后 shutdown update executor、drain device/outer events、reset 确切 graph、再次 drain、删除 graph/output 引用并释放 Event，最后记录不可继续服务。模块自身不证明 EOF/idle，也不授予写清理请求的权限；实际控制仍需原 performance 锁和精确归属验证。失败保留引用和 unknown，不能写成清理完成。

`historical-build/task_anchor.cpp` 与原构建回执绑定。原 `task_anchor_driver.py` 提供明确分开的 `build-only` 与 `run-toy`；本包仅描述 build-only：

```sh
python3 historical-build/task_anchor_driver.py build-only \
  --cpp historical-build/task_anchor.cpp --outdir /absolute/new/build-output
```

构建入口硬性核对原 CANN9.0头、torch_npu版本/git、C++ SHA，原编译参数为-O0/-g0/C++17、MAX_JOBS=1。本轮未重新构建或动态加载扩展。历史 SO SHA 为 `f1594bd83168014ba96cb0c2703fadfc19d947b0faaa76b75d17b4e6ded7a1bf`，probe 加载也校验这个完整 SHA；重建不保证字节相同，不能自行绕过此校验。原始回执、现存远端 SO 的只读 hash 和环境记录在包与阶段证据中。SO/原始 trace/transport/key/模型/服务 controller 均不入包。

raw 审计入口为 `raw-audit/audit_public_model_markers.py --help`，native 为 `native-audit/audit_sg67_formal_by_rank.py --help`。实际输入 schema、文件 SHA 和原生结果另见阶段27诊断证据包及 gen67/gen74 记录。当前实际短诊断两框架文本不相等，token/greedy/原性能矩阵输出等价及差异根因尚未闭合，优化未采用。代码可打包和 CPU 用例通过不能代替实际204门禁或 root 总体验收。
