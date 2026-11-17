# 原始数据、脱敏与复核

每个测量Run的 `raw/*.jsonl.xz` 是原始AISBench请求JSONL的无损XZ封装；解包后的SHA与来源原件一致。相邻JSON/CSV用于复算请求数、时长、吞吐和延迟。

结构化profile和环境证据只替换连接信息、宿主名称及私有绝对路径，实验量纲、形状、token计数和时长指标保持原值（连接端口改为占位符）。原始SHA与公开副本SHA在 [provenance](../provenance/artifacts.json) 对应。大NPU原件由私有证据系统保管，公开索引用 `private-evidence://` 逻辑定位，不公开服务器地址。

离线运行 `python3 scripts/validate_archive.py`；它只检查归档、源码语法和算术，不启动模型/NPU，也不会把验档成功解释为性能成功。

失败同样保留：G32缺少CANN缓存目录；首次AISBench的宿主绝对软链接不能在容器访问；三轮预热未达到稳定门槛。它们分别属于启动/客户端无效轮和未计分诊断，不能作为优化收益。

原始JSONL中的 `time_points` 使用数据库引用；本次首批归档尚未包含原SQLite时序数据库。当前可复算请求数与批次吞吐，逐请求时序和延迟分位数尚不能仅凭本分支独立重建，需补归档时序证据。
