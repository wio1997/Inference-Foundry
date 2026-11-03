# MC2 无重放观测：当前可用证据与版本界限

本记录仅只读读取 169 自有 stage 容器的已保存 profile、安装工具帮助及公开文档；没有开启 profiler、启动 NPU 负载或修改内核。样本是 2026-10-29 单次**带 profiler 的 prefill 采集**，不是未插桩性能基线，不能代表全部层、全部工作负载。

## 同一任务里此前尚未展示的指标

从既有 16-rank MC2 关联结果中选出 digest `1b826883...`、EP8–15 的第 14、39 层，各八条。以 rank 文件、Name、Model ID、Stream ID、Task ID、Start Time 完整匹配各自原始 `kernel_details.csv`，每条恰好一行。数据、原文件 SHA 和全字段行保存在 `diagnostic-20261029/mc2-extra-pipe-fields-16-tasks.json`；可用 `python3 diagnostic-20261029/extract_mc2_extra_pipe_fields.py` 重新只读提取，脚本调用 169 自有 stage 容器，不触发设备工作。

| 已有原表字段（范围为 8 rank 中最小至最大） | 第 14 层 | 第 39 层 |
| --- | --- | --- |
| 任务 Duration | 160365.416–160422.455 μs | 8217.616–8290.414 μs |
| AIC MAC ratio | 0.1%–1.1% | 5.0%–16.4% |
| AIC MTE2 ratio | 0.3%–1.2% | 10.9%–21.2% |
| AIV vector ratio | 0.1%–0.2% | 1.2%–3.6% |
| AIV MTE2 ratio | 12.4%–12.7% | 2.7%–6.2% |
| AIV scalar ratio | 1.5%–1.7% | 2.3%–4.0% |

这些是工具给出的整个融合设备任务的聚合、按原始显示精度呈现；不同流水和不同核可能并发，不可把这些比率相加或以 1 减它们推导等待时间。文档对 AIV MTE2 的一般描述是从 DDR/L2 读取；本实现还可能涉及 peer 内存，因此这一字段无法在当前数据中区分正常数据读、通信、重复读取及等待。本组第 14 层与第 39 层的计数模式不同，但专家路由数量和真实内部轨迹缺失，不能认定两层工作量相同、内存或通信是根因。只能说 `aiv_mte2_ratio` 在低 MAC 样本中不是零，不能将低 MAC 解释为整个设备毫无活动。

CANN 相关字段一般解释见 [Ascend MindSpeed 文档中的 kernel_details 字段说明](https://www.hiascend.com/document/detail/en/MindSpeed/2610/LLM/MindSpeed_LLM/docs/en/pytorch/tuning/fsdp2_backend_performance_optimization.md)；其不是 169 实际 CANN 9.2 安装包的源码语义证明，特别是 aicore_time 与 Wait Time 暂不据此推出内部等待。

## CANN 9.2 本机帮助与厂商 9.2 beta 文档

只读运行现有 `msprof --help`，原输出 `diagnostic-20261029/msprof-application-help-9.2.txt`；其声明 `--aic-mode=sample-based` 可定时采样、`--aic-freq` 范围 1–100 Hz，`--instr-profiling` 可打开指令相关数据、`--instr-profiling-freq` 范围 300–30000 cycle。帮助显示接口存在，**不证明 169 SKU、当前 SHMEM 算子、多 rank 或任何数据关联能正常工作**。原有 `msprof op --help` 的输出 `diagnostic-20261029/msprof-op-help-9.2.txt` 明确 replay-mode 默认 kernel，候选只有 application/kernel/range；不能称其已证明无需重放。

华为 [CANN 9.2.0-beta.1 AI 任务性能采集说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta1/devaids/Profiling/atlasprofiling_16_0011.html) 对一般 `msprof --instr-profiling` 指的是 AIC/AIV 带宽和时延采样，并列出 Atlas A2/A3 等支持范围；文档同时写明需传入用户程序且为**单算子 API 执行场景**才有具体性能数据，并与 task-time、ai-core、aic-metrics、HCCL、runtime-api 等采集开关互斥。该文件版本为 9.2 beta 文档，不能自动等同当前 169 的实际版本和产品 SKU。它没有证明可在这个多 rank 整模型 SHMEM 内核上绑定每个 task 或映射 PC 到源码阶段；也不要将它和 `msprof op` 的 `TimelineDetail` 指令流水视作同一能力。

`sample-based` 在帮助所列最高 100 Hz 下，名义采样间隔 10 ms：第 14 层 160 ms 任务理论上有多个采样窗口，但窗口边界、时钟、偏差及多核/多任务重叠都未核验；第 39 层约 8.2 ms，**可能完全落在两个采样时刻之间**。采样不是逐 task 或逐源码阶段归因的证明；即使多 rank 同时存在样本，也无法凭接近的时间认定 same collective、同一任务内部通信或同步阶段。单独采集间不能仅因重复相同输入，就宣称为同一次 task。

## 最小安全取证命令及前提

已验证并可重复的离线命令为前述 `extract_mc2_extra_pipe_fields.py`：读取既存数据并按完整身份匹配原表，无 NPU 访问。输出只包含 task 摘要、指标及 hash；不读取权重或用户 prompt。

若将来进行已授权范围内的诊断采样，须先确认所有参与卡属于本任务、隔离环境已备好、没有冲突 profiler、对应输出目录在 `/data/tiankuan/wio/main` 下、基线服务可恢复，才可评估下面的**参数形状示例，不是可直接执行的已验证运行命令**：

```
msprof --output=<任务专属目录> --ai-core=on --aic-mode=sample-based --aic-freq=100 --aic-metrics=PipeUtilization <已审核的隔离多rank启动器及参数>
```

真实 launcher、父子进程采样范围、16 rank 设备映射、多卡 MC2/SHMEM 支持、与其他 profiler 的互斥、host/device 时钟、采样损失、统计粒度以及诊断与基线行为是否等价均未证实。应先用本机帮助、对应版本文档和无设备只读检查完成核验；即使命令成功，也只能把采样作为时间窗口级设备活动线索，不会自动获得内部阶段、PC、同 task identity 或通信耗时。不能在生产/基线服务上直接运行，也不能和现有 profiler 同卡叠加。更细阶段的可行方案仍需满足二进制身份、符号映射和工具能力约束；当前没有发现可靠的无需重放且无需改 kernel 的内部阶段观测方法。
