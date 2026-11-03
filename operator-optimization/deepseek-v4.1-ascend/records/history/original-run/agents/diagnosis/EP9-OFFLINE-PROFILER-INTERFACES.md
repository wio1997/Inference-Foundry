# EP9 既有采集：官方离线接口与通信分析结果

## 官方接口与当前边界

华为 [CANN 9.2.0-beta.1 PyTorch 离线解析文档](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta1/devaids/Profiling/atlasprofiling_16_0034.html) 列出 `torch_npu.profiler.profiler.analyse` 接口，可将已经落盘的 PyTorch profiler 数据解析为 timeline/CSV/数据库；并注明根据数据量限制解析进程数。该页面描述的是 beta 文档，不能仅凭版本名证明它与当次采集生成器完全一致。此接口生成结果和日志，**不能将它直接作用于需要保留的原始 profile 并称为只读**。

[同版本通信分析文档](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta1/devaids/Profiling/atlasprofiling_16_0027.html) 描述先使用 `msprof --export=on --clear=off`，再对 PROF 目录使用 `msprof --analyze` 或 `msprof.py analyze` 导出通信算子统计与矩阵；原始采集必须已有相应数据，`--clear` 可删除原 PROF 内的 sqlite 目录，不能对原件启用。文档列 A3 支持，但不能因此推断现有自定义 `HcclShmem` 融合算子的内部等待/数据被采集，也不能保证任意 raw slice 包含 expert tensor 值。

当前 stage 安装有 `tools/profiler/profiler_tool/analysis/msprof/msprof.py`（本次观察 SHA-256 `f110cac92111fb3bb36904422f6e66469d434f9dc13f57b41c892caa941d8bdb`）。本次 EP9 的 `PROF_.../analyze` 已有 `communication.json`、`communication_matrix.json` 和 `communication_analyzer.db`，同一 `PROF_.../mindstudio_profiler_output` 已有 `op_summary`、`task_time` 和 `msprof_*.json` 等。文件存在不代表已捕获或识别 MC2 内部阶段，也不证明历史生成器与现在安装解析器二进制版本完全一致。

未重新运行生成器：既有导出足以对明确的通信分析文件开展下述定向检查，重跑会在所指 PROF 目录写入结果；没有证据表明重跑可凭空恢复未采集的内部阶段或专家值。若将来有必要尝试且获得适用约束，可先复制数据到独立任务位置、核查版本并限制 CPU/资源，再比较输出；当前任务不启动新 NPU 负载。

EP3 的已保存 `tensor_info` 原始切片已经用当前安装 bean 解析出 shape、dtype、format、时间等描述符，但该格式契约没有元素值或直接 TASK 绑定，见 [tensor_info 契约](MC2-EP3-TENSORINFO-CONTRACT.md)。当前没有发现一个正式、已验证的解析接口能从这一批切片读出逐专家 tensor 元素或当前 MC2 内部分阶段。

## EP9 已有通信分析文件的定向检查

对 EP9 保存的 `PROF_.../analyze/communication.json` 只读解析（当前观察 SHA-256 `5640081e05af3d8ecfb33975da324f8f432a1695fdd410fc4020286dc71d1c5c`）。该文件共 **1850** 个顶层通信条目，每条都有可解析的 `Start Timestamp(us)` 和非负 `Elapse Time(ms)`；顶层名称中没有包含 `DispatchFFNCombine`、`mc2` 或 `expert_token_nums` 的条目。

将条目的导出起止值与 EP9 第 0 层选定 MC2 的导出设备区间（stream 47、task 46755、globalTaskId 4709，时长约 273.565 ms）做**纯数值区间比较**：0 条相交，最近的前一条末端与目标起点相距 377.979 µs，最近的后一条起点与目标终点相距 2105.759 µs。目标身份、profile 和 trace/DB 嵌套关系见 [EP9 精确记录](EP9-MC2-STAGE-AND-PMU-AVAILABILITY.md)。

该检查仅说明所读的这个分析文件没有在对应导出数值区间内列出通信条目；不同文件的时钟对齐、生成器、采集覆盖未被完整证明，不能用它证明实际没有通信、没有其他流工作、没有 shmem 内部传输或等待。通信记录作为外部统计也不能直接用来拆解融合核的 dispatch/GMM/combine 阶段，不能推断端到端瓶颈或在线 TPS。

## 复算

在现有 stage 容器运行 `python3 agents/diagnosis/query_ep9_mc2_existing_comm_analyze.py`，输出为 `ep9-mc2-existing-comm-analyze.json`；脚本对该文件做十进制解析并记录本次 SHA，限定比对一个目标区间，不读取模型、请求或 tensor payload。此次只读取现有文件，没有运行 NPU 或更改服务、目标实现及原始 profile。
