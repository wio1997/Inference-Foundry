# vLLM / Ascend Profiler：多实例覆盖、版本边界与调度极限证据

研究日期：2026-10-01。仅阅读公开官方源码与文档；未访问用户服务、安装或远程机器，未运行 profiling/benchmark。用户所说的“0.27”尚未核验为具体 vLLM、vLLM-Ascend、torch_npu、CANN 组合。以下公开版本事实不能替代该核验。

## 结论

了解 vLLM 的 prof，并且 Ascend 有真正的 CPU + NPU 时间线采集路径。对于多机多卡 PD，合适的做法是：常驻服务启用采集能力，每次实验按拓扑清单控制所有 P/D 实例的小窗口，收齐各 rank 的设备时间线，再与请求、调度、KV transfer 和 MTP 的框架事件关联。单独一次 `/start_profile`、一张 rank 0 图、NPU 利用率或“连续几次没有提升”，都不能证明调度达到极限。

当前最快的验证路径不是每轮重启大模型，而是先核验已安装版本与控制覆盖，一次启用需要的观测点，再在已暖机、权重常驻的服务上做短窗口。公开 Ascend worker 的代码允许复用 profiler 对象再调用 start；实际 torch_npu 后端能否正确生成多次窗口，需要在用户安装上用非空 trace 与错误日志确认。

## 固定源码范围

| 代码 | 本次固定 revision | 本地只读副本 |
|---|---|---|
| vLLM v0.27.0 | `4bdc8a788d2e2ce9165d552b3d4d8b72604626bf` | `work/glm-official-source/vllm-0.27.0/` |
| vLLM main | `4c2d277643e217344056e1d2c42115d5f005912f` | `work/glm-official-source/vllm/` |
| vLLM-Ascend v0.27.1rc1 | `3b31886237ed65c435e0965b10e8002ff5769766` | `work/glm-official-source/vllm-ascend-0.27.1rc1/` |
| vLLM-Ascend main | `a8fcedb03d93e60efceddbfc912406f7fa491d57` | `work/glm-official-source/vllm-ascend/` |

v0.27.0 与 Ascend v0.27.1rc1 是分别核验的公开源码范围，不是在用户机器上确认过的兼容安装。以下链接均固定 revision。

## 四种“profile”不能混用

| 名称 | 实际用途 | 能否提供此次需要的完整调度证据 |
|---|---|---|
| `model_runner.profile_run()` / memory profiling | 启动时 dummy forward、编译/暖机、激活峰值与 KV 内存预算 | 不能；它不是用户负载的完整运行时间线。即显式指定 KV bytes，Ascend worker 仍会调用 profile_run。见 [worker.py:552–588](https://github.com/vllm-project/vllm-ascend/blob/3b31886237ed65c435e0965b10e8002ff5769766/vllm_ascend/worker/worker.py#L552-L588)。 |
| `profiling_chunk_predictor` / `profile_prefill_latency` | 测量 forward/chunk 成本并拟合 chunk latency 模型，服务于动态 chunk 决策 | 不是全局极限证明；计时前后调用 `torch.npu.synchronize()`，会改变原有异步重叠。见 [predictor:38–88](https://github.com/vllm-project/vllm-ascend/blob/3b31886237ed65c435e0965b10e8002ff5769766/vllm_ascend/core/profiling_chunk_predictor.py#L38-L88)、[worker.py:870](https://github.com/vllm-project/vllm-ascend/blob/3b31886237ed65c435e0965b10e8002ff5769766/vllm_ascend/worker/worker.py#L870)。 |
| vLLM `profiler="torch"`，Ascend `TorchNPUProfilerWrapper` | 真正 CPU/NPU operator、device task、相关时间线与分析数据 | 是底层执行证据的重要来源，但不会自动解释跨实例请求依赖和哪些等待可消除。见 [NPU wrapper:29–86](https://github.com/vllm-project/vllm-ascend/blob/3b31886237ed65c435e0965b10e8002ff5769766/vllm_ascend/profiler/torch_npu_profiler.py#L29-L86)。 |
| MS Service Profiler | 按 symbols 配置采集框架函数、请求、batch、KV cache 等事件；可选 ACL/MSPTI/Torch device 采集 | 更贴合调度研究；覆盖取决于实际版本与 symbols。不能假设默认配置覆盖 V2、自定义 scheduler、proxy、全部 connector/MTP 路径。见 [官方比较与指南](https://github.com/vllm-project/vllm-ascend/blob/3b31886237ed65c435e0965b10e8002ff5769766/docs/source/developer_guide/performance_and_debug/service_profiling_guide.md#L9-L24)。 |

## 0.27 的接口与 Ascend 配置实际行为

vLLM v0.27.0 已有 `--profiler-config`，HTTP `POST /start_profile` 和 `/stop_profile`。仅启动配置的 `profiler` 非空时注册这两个路由；没有配置时可能返回 404。HTTP 路由调用 Python EngineClient API，路由本身不转交 request body 里的 `profile_prefix`。[api_router.py:21–46](https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/entrypoints/serve/profile/api_router.py#L21-L46)

Ascend v0.27.1rc1 与上述 Ascend main 的 `torch_npu_profiler.py` 完全相同：

- `profiler="torch"` 实际调用 `torch_npu.profiler.profile`，hardcode CPU + NPU；只接受 torch，cuda/proton 不属于该实现。
- 设备采集固定 Level1、PipeUtilization，关闭 l2_cache、op_attr、record_op_args 等；不能据此宣称拥有全部内存带宽、shape 或资源需求计数。
- 传递 `torch_profiler_with_memory`；将 vLLM `torch_profiler_with_stack` 转为 torch_npu `with_modules`，而 torch_npu `with_stack=False`。源码说明后者有显著时间开销。
- 未传递 vLLM 的 warmup/active/wait schedule、record_shapes、with_flops、use_gzip、dump_cuda_time_total；不能照抄通用 CUDA profiling 参数并认定 Ascend 等效生效。delay/max 是父类 WorkerProfiler 管理的 worker step 控制。

以上行为均见 [NPU wrapper:37–86](https://github.com/vllm-project/vllm-ascend/blob/3b31886237ed65c435e0965b10e8002ff5769766/vllm_ascend/profiler/torch_npu_profiler.py#L37-L86)。首次诊断可用以下配置形状；路径必须按实例区分，数值和窗口需按实际服务核验：

```json
{
  "profiler": "torch",
  "torch_profiler_dir": "/experiment/run_id/instance_id",
  "torch_profiler_with_stack": false,
  "torch_profiler_with_memory": false,
  "ignore_frontend": true
}
```

此例关闭的是 vLLM AsyncLLM 自带 frontend CPU trace，以便先测 worker 开销；研究端到端调度仍需以框架事件补全 frontend/proxy。需要 frontend 段时单独开窗口，或使用匹配版本的 Service Profiler。v0.27.0 AsyncLLM 的 frontend 采集是 `torch.profiler` CPU-only；它不随 worker 的 delay/max 自动截断。[async_llm.py:181–201](https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/v1/engine/async_llm.py#L181-L201)、[config:80–95,136–143](https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/config/profiler.py#L80-L95)。

旧 `VLLM_TORCH_PROFILER_DIR` 环境变量配方不应作为 0.27 的默认方案。Ascend 固定版本指南明确记录 2026-01-19 mainline 弃用该变量；本次 v0.27.0 envs.py 亦未找到该变量注册。[指南:36–55](https://github.com/vllm-project/vllm-ascend/blob/3b31886237ed65c435e0965b10e8002ff5769766/docs/source/developer_guide/performance_and_debug/service_profiling_guide.md#L36-L55)、[v0.27.0 envs.py](https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/envs.py)

main 比 v0.27.0 新增 Proton 和 `torch_profiler_activities` 等字段；Ascend 当前 wrapper 仍只接受 torch，并 hardcode CPU/NPU。因此不能把 main 的 Proton、CUDA Graph attribution 或 activities 选择当作用户 Ascend 0.27 已支持的功能。[v0.27.0 config](https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/config/profiler.py#L16-L115)、[main config:16–88](https://github.com/vllm-project/vllm/blob/4c2d277643e217344056e1d2c42115d5f005912f/vllm/config/profiler.py#L16-L88)

另一个真实版本差异：v0.27.0 达到 max_iterations 后只停止后端、保留 active，须显式 stop 才能清理并接受下一次 start；main 已改为走 public stop 清理。Ascend worker 的 step 在 execute_model 前调用，计数也不能机械理解成“恰好 K 个完整设备迭代”。每次捕获均执行 stop 并核对实际 trace 区间。[v0.27.0 wrapper:83–138](https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/profiler/wrapper.py#L83-L138)、[main wrapper:124–136](https://github.com/vllm-project/vllm/blob/4c2d277643e217344056e1d2c42115d5f005912f/vllm/profiler/wrapper.py#L124-L136)、[Ascend worker:680–683](https://github.com/vllm-project/vllm-ascend/blob/3b31886237ed65c435e0965b10e8002ff5769766/vllm_ascend/worker/worker.py#L680-L683)

## 多 rank / 多 P、D 实例怎样覆盖

v0.27.0 的控制链是 `HTTP → AsyncLLM.start/stop_profile → EngineCoreClient utility → EngineCore.profile → executor.collective_rpc("profile") → Ascend worker.profile → torch_npu`。collective_rpc 的接口定义为所有 workers；因此一个 engine 所管理的 TP/PP/EP workers 都应参与，不是仅 rank 0。[AsyncLLM:925–935](https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/v1/engine/async_llm.py#L925-L935)、[Core:769–770](https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/v1/engine/core.py#L769-L770)、[Executor:153–172,256–257](https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/v1/executor/abstract.py#L153-L172)

| 部署边界 | 控制事实 | 控制器必须核对 |
|---|---|---|
| 一个 TP/PP/EP engine | executor broadcast 所有 workers；无 rank 0 gate | rank 清单、每 rank 的非空设备 trace、各 host 输出文件 |
| vLLM 内置 DPLB 管理多个 DP engines | DPLBAsyncMPClient 对 core_engines 并发 utility；普通 AsyncMPClient 只控制自己的 core_engine | 实际使用哪种 client、managed engines 清单；不能推广到外部任意集群 |
| PD 的独立 P/D API 实例 | 各自配置、各自 start/stop；普通 Ascend PD proxy 不转发这些接口 | 按 deployment manifest 控制所有参加实验的实例，而不是只调用入口 proxy |
| 官方 EPD example proxy | 对 encoders/prefillers/decoders/pds 并发 broadcast | 它逐服务捕获 error，外层依然可能 HTTP 200；必须看 results[*]，转发 timeout 固定 10 秒 |

DPLB 证据：[core_client.py:1125–1126,1165–1168](https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/v1/engine/core_client.py#L1125-L1168)、[1521–1530](https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/v1/engine/core_client.py#L1521-L1530)。PD 证据：[官方指南:117–139](https://github.com/vllm-project/vllm-ascend/blob/3b31886237ed65c435e0965b10e8002ff5769766/docs/source/developer_guide/performance_and_debug/service_profiling_guide.md#L117-L139)。EPD 真实转发与返回：[proxy:786–899](https://github.com/vllm-project/vllm-ascend/blob/3b31886237ed65c435e0965b10e8002ff5769766/examples/epd_disaggregated/epd_load_balance_proxy_layerwise_server_example.py#L786-L899)。

Ascend trace 名带 DP/PP/TP/DCP/EP/global rank；同一对象重启采集仍使用第一次创建的 trace name。P/D 是不同 engine，均可能包含 rank 0，不能靠 rank suffix 唯一定位集群角色；必须记录 `run_id / instance_id / P-or-D / host / process / device / all ranks / topology_epoch` 的映射。[worker:1063–1081](https://github.com/vllm-project/vllm-ascend/blob/3b31886237ed65c435e0965b10e8002ff5769766/vllm_ascend/worker/worker.py#L1063-L1081)、[rank suffix:696–732](https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/distributed/utils.py#L696-L732)

并发 start 不是全局原子时刻。跨机时间线需要 request/transfer/step IDs 和时钟对齐或因果链接；未校准的主机时间戳不能直接相减解释 PD latency。headless worker 由所属 engine RPC 控制，不应假设每台物理主机都有单独 HTTP profiling endpoint。

## 输出、成本和 Graph 盲区

官方 Ascend 指南要求分析生成的 `*ascend_pt`，列出 `trace_view.json`、kernel/operator_details、step_trace_time、CSV 与 DB 等。实际文件取决于 torch_npu/CANN/export 版本，控制器须检查存在、时间跨度、rank 覆盖及分析状态。HTTP 200 并不证明这些文件有效：WorkerProfiler 会把后端 start/stop 异常记为 warning。[输出指南:88–115](https://github.com/vllm-project/vllm-ascend/blob/3b31886237ed65c435e0965b10e8002ff5769766/docs/source/developer_guide/performance_and_debug/service_profiling_guide.md#L88-L115)、[wrapper:54–69](https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/profiler/wrapper.py#L54-L69)

对调度研究尤其要保留以下判断边界：

- 总 kernel 时间不是端到端 critical path。异步 stream、HCCL、PD transfer、CPU submit、KV 等待需要因果与重叠分析；HCCL duration 可能包含晚到 peer 的等待，单 rank 无法确定起因。
- Graph replay 不再逐算子执行 Python forward。框架/module scopes 可能集中在 metadata/update/replay，底层设备任务是否被完整展开须在实际后端 trace 验证。应补 `graph bucket、real/padded tokens、capture/replay/update、stream/event dependency` 的轻量事件；不能把缺少 Python 算子事件解释为没有执行。
- v0.27.0 `capture_torch_profiler` 的文档明确是 CUDA graph capture rank 0；本研究未确认该通用选项在 Ascend ACL Graph 的实际接线，不能把它当成全 rank 的 ACL replay 诊断方案。[config:69–72](https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/vllm/config/profiler.py#L69-L72)。Ascend V2 存在独立 fullgraph 参数更新/重放路径，见 [aclgraph_utils:174–234](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/worker/v2/aclgraph_utils.py#L174-L234)。
- Service Profiler 默认 symbols 仍显式列 V1 runner 的 execute/update/prepare 和 propose_draft。选择 V2 或专用 runtime 后，需要核对实际符号、补 planner/state/MTP/connector/proxy 事件，不能仅因工具开了就宣称覆盖全链路。[profiling_config.py:90–123](https://github.com/vllm-project/vllm-ascend/blob/3b31886237ed65c435e0965b10e8002ff5769766/vllm_ascend/profiling_config.py#L90-L123)。
- Profiler 启停、数据导出、stack/memory 采集都会增加成本。先用关闭 stack/memory 的短暖机窗口与低开销 marker；比较 prof OFF / ON 的 E2E、batch 与提交时序，确认观察器没有改变瓶颈。正式性能数字在重型 prof OFF 时重复测量。stop 导出可能明显耗时，不能把导出时间当推理性能；EPD 的 10 秒超时也不能直接认定后端失败。[vLLM 官方 profiling:3–8,34–38](https://github.com/vllm-project/vllm/blob/4bdc8a788d2e2ce9165d552b3d4d8b72604626bf/docs/contributing/profiling.md#L3-L8)。
- 已有另一条 msMonitor dynamic_profile 路径：Ascend worker 在 msmonitor_use_daemon 时执行 `dp.step()`；当前 TorchNPUProfilerWrapper 明确拒绝与其同时启用。切换观测方案时需核验当前 daemon/config，不能重复套 profiler。[worker:649–656](https://github.com/vllm-project/vllm-ascend/blob/3b31886237ed65c435e0965b10e8002ff5769766/vllm_ascend/worker/worker.py#L649-L656)、[wrapper:42–48](https://github.com/vllm-project/vllm-ascend/blob/3b31886237ed65c435e0965b10e8002ff5769766/vllm_ascend/profiler/torch_npu_profiler.py#L42-L48)。

MS Service Profiler 能在加载后通过配置 enable 控制，改变 symbols 则需要重启。最低成本策略是先设计足够通用的轻量事件字段、一次加载，然后按问题启用窗口；只在新增无法热装的代码/符号或改变资源拓扑时重新 bootstrap。详细 counters 按具体未决假设取用，而非每轮全量采集。[指南:157–175,228–266](https://github.com/vllm-project/vllm-ascend/blob/3b31886237ed65c435e0965b10e8002ff5769766/docs/source/developer_guide/performance_and_debug/service_profiling_guide.md#L157-L175)

## 怎么判断接近框架调度极限

以下是研究方法建议，不是源码已有的“自动判极限”能力，也不是固定 SOP：

1. 对动态输入/输出、到达过程、MTP acceptance、PD 拓扑与资源预算定义条件；这不是永久固定形状，而是给每次比较建立可解释的条件。任意输入、无限资源下不存在一个有意义的单一 TPS 极限。
2. 用全链路 IDs 和所有 rank 的 trace 建立执行 DAG：框架依赖、数据依赖、KV ownership/ready、Graph dispatch、MTP propose/verify/commit、采样提交、P/D transfer，以及 CPU/NPU/HCCL/网络资源占用。分别标注必须依赖、实现导致的同步、未证实依赖。
3. 固定当前算子/模型路径，校准可实现的成本包络，构造遵守必须依赖与资源容量的乐观排程下界 `T_lower`；保留最好已实现完整排程 `T_best`，形成 `T_lower ≤ T_opt ≤ T_best`。吞吐则用已实现值 ≤ 最优值 ≤ 乐观容量上界。不能把含 peer stall/争用的当前 duration 直接当不可减少成本；改变 batch/chunk/MTP 时成本也会改变。
4. 找上下界差距由哪些等待、padding、提交串行、rank imbalance、transfer stall 或资源冲突支配，再用反事实排程/针对性代码干预验证。观察到空隙只提供候选；消除空隙后是否增加有效 token/s 与尾延迟，必须实测。
5. 当最好可实现结果接近经过验证的乐观界、剩余差距主要被必需依赖与资源约束解释、不同代表性负载与候选拓扑仍重复成立，才报告“在这些条件下接近已测得的调度极限”，连同置信范围和未决差距。若界仍很宽，就继续研究；“一时没有新想法”不能作为极限证据。

拓扑探索与服务常驻不冲突：同一组通信/权重布局中的 planner、chunk、batch、MTP 等策略优先常驻切换；TP/EP/PP 或权重分片改变的候选往往需要重新建 worker/groups/cache/Graph。应把 bootstrap 成本与采集/稳态成本分开记，避免为了重复抓 trace 反复启动模型。

## 下一步最低信息

只需要一次获得：精确安装版本/commit 与 patches、当前启动参数、V1/V2 路径、torch_npu/CANN、实际 profiler 配置/msMonitor 状态、所有 P/D API 地址及所属 host/rank/device 清单、proxy/connector 的具体代码路径。先跑一个短窗口验证“控制和覆盖”，然后在已驻留服务上收一个代表性完整周期；无需先跑大量 benchmark，也无需预设固定输入输出、PD 比例或 MTP 深度。
