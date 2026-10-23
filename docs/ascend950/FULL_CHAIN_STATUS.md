# 32K C32 全链路分析状态

范围：两框架 × 普通 / 实际约90%命中，共四组；输入32768、输出128、C32、每轮64请求。以下代码已在原服务器参与采集或分析，发布整理没有新增性能测量。

| 事项 | 当前状态 |
| --- | --- |
| 四组八卡详细采集 | 已执行，32份卡级算子/API导出已下载 |
| CPU 调度、线程计数、原生工作量 | 已分析；部分直接阶段探针仍缺失 |
| 显存账本 | 64份快照已核对，保留原生/驱动余项；不声称完成物理 HBM bank 映射 |
| 图执行与重叠 | 远端32份分析已结束；最终 JSON 尚未下载 |
| 通信 | 标准 HCCL 实例已匹配；跨卡设备时间和事件边界尚未独立确认，不能据此认定慢卡 |
| profiler 对测量的影响 | 普通组有低开销对照；HOT 受详细采集明显扰动，仍需低干扰确认 |
| 原始材料回收 | vLLM普通组先导八卡原始包已在本地；四组详细原始目录未全部回收 |
| 最终因果归因 | 未完成：统一输出接口、缺失元数据及必要因果对照尚未执行 |

## 代码入口

实验根目录为 `reproduction/ascend950/performance-20261021`。

- `adapter/deepinfer_dp8_adapter_e33.py`、`tools/e33_di_telemetry.py`：DeepInfer 整轮采集和诊断。
- `tools/e33_vllm_telemetry.py`、`tools/e33_vllm_bootstrap/sitecustomize.py`：vLLM 原生阶段挂钩。
- `tools/e33_profile_control.py`：采集生命周期、排空及导出核验。
- `tools/e33_*_capture_sequence.py`：预热、U1 / P / U2 以及每个 HOT 窗口独立前缀填充。
- `tools/e33_native_census.py`：所有导出算子/API清单。
- `tools/e33_native_timeline.py`：图执行与不重叠时间分区。
- `tools/e33_clock_map.py`：原始时钟变换；主机验证不等于跨卡设备时钟已对齐。
- `tools/e33_memory_ledger.py`、`e33_cpu_ledger.py`、`e33_work_ledger.py`：显存、CPU 和工作量账本。
- `tools/e33_communication_audit.py`、`e33_cache_layout_audit.py`、`e33_response_boundary_audit.py`：通信实例、缓存实际布局与首包边界复核。

这些离线分析工具需要相应原始导出及诊断文件。多 GB 原始包和完整已安装框架源码不放入 Git；本仓库的脚本本身不能替代这些输入。`STATUS.json` 记录缺项，`COPY_PROVENANCE.json` 记录所发布的原代码哈希。

DeepInfer 异步投机事件中的 request 归属字段发现错位，未用于逐请求归因；整批计数另行交叉核对。跨流/跨卡累计时间不可直接相加，未解释时间和内存余项保持未解释。没有开展 NVIDIA→950 优化移植，也没有宣称某个局部算子解释了完整端到端差值。
