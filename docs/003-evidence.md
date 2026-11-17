# 原始数据、脱敏与复核

每个测量Run的 `raw/*.jsonl.xz` 是原始AISBench请求JSONL的无损XZ封装；解包后的SHA与来源原件一致。相邻JSON/CSV用于复算请求数、时长、吞吐和延迟。

结构化profile和环境证据只替换连接信息、宿主名称及私有绝对路径，实验量纲、形状、token计数和时长指标保持原值（连接端口改为占位符）。原始SHA与公开副本SHA在 [provenance](../provenance/artifacts.json) 对应。大NPU原件由私有证据系统保管，公开索引用 `private-evidence://` 逻辑定位，不公开服务器地址。

离线运行 `python3 scripts/validate_archive.py`；它只检查归档、源码语法和算术，不启动模型/NPU，也不会把验档成功解释为性能成功。

失败同样保留：G32缺少CANN缓存目录；首次AISBench的宿主绝对软链接不能在容器访问；三轮预热未达到稳定门槛。它们分别属于启动/客户端无效轮和未计分诊断，不能作为优化收益。

首批提交中的JSONL使用SQLite引用，曾缺少对应数据库；后续提交已补齐原件并完成独立复算。首批提交中的缺口说明仍保留在Git历史中。

补充：原SQLite时序现已按JSONL的 `db_name` 和 `__db_ref__` 逐一补齐。`scripts/timing_witness.py` 用标准库解析原始浮点数组，复算逐请求TTFT、TPOT、E2EL及均值、极值、中位数、P75/P90/P99，再与原CSV核对。流式事件可能一次返回多个token，事件数量不能代替服务器输出token计数。

原AISBench公式与数据库读写源码：[default_perf.py](../code/aisbench/default_perf.py)、[db_utils.py](../code/aisbench/db_utils.py)。核验覆盖640条请求、30个阶段的630个延迟统计量。它验证原始测量的可回溯性，不代表已完成最终性能对齐。

后续状态：[实际分配长度的60种编译预热](../evidence/allocator_compile_only_g40.json)已完成，但仍需真实完整负载证实没有新编译。[vLLM G4单条P/D协议诊断](../evidence/vllm_G4_PD_protocol_diagnostic.json)返回实际32768输入/32输出；它不计入AISBench性能。
