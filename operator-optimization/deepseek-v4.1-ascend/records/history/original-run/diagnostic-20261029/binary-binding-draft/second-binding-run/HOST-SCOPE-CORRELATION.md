# selective2 主机调用关联与缺口

2026-10-29。仅分析采集已停止后的 metadata 快照；没有运行新 NPU 负载。

`correlate_host_scopes.py` 在 169 自有 stage 容器低 CPU 优先级运行，输入和输出摘要见 `host-scope-correlation.json`，单机 ZCode 运行输出见 `host-scope-zcode-output.jsonl` 和复核输出 `host-scope-zcode-recheck.jsonl`。解析时文件前后 stat 一致，所有输入均以换行结尾，JSONL 无畸形行；这并不证明采集完全刷盘。原始 C 文件 10,349,082 字节，SHA-256 为 `3fc3a80b67cbfd08dba9d259bfba2c70ada1bb1414aa2c1772984e25be62bcc1`。

23 个 Python 文件共 2759 对完整 begin/end；C 有 9428 个 launch begin 和 9472 个 launch end。以同一 PID、TID、C 线程局部 call、函数和 stream 唯一配对且返回 0 的有 8030 对；2840 个调用键不完整或重复。16 个 C PID 均在 sidecar 文件名 PID 集内，但 8030 对所在的 16 个 PID/TID 与所有 Python scope 的 18 个 PID/TID 无交集，因此没有同线程完整层 scope 关联。Python `call_id` 和 C `call` 是不同编号，禁止按值关联。字段检查覆盖实际出现的事件；未出现的 graph 和回填字段不能经该数据验证。实际未观测到 graph / backfill，不代表没有运行。

每个 C 进程都曾报告 `state_incomplete=true`，日志中明示的 `dropped_since_last` 总数为 8499；没有 process-exit stats，不排除最后未报告的丢失。未见文件 cap 标记不等于完整。状态表存在失配、并发变更及待分类容量限制，禁止将缺席作为不存在，也不能把候选 by-entry 的 SHA 当成已绑定实际执行内核。

采集启动脚本显示 Python sidecar 与 `LD_PRELOAD` recorder 在同一诊断容器的各进程生效；C 用 `getpid()` 与 `SYS_gettid`，sidecar 用 `os.getpid()` 与 `threading.get_native_id()`。16 个数值 PID 重合支持同容器观察，但不独自证明进程生命周期唯一或采集时 namespace 未变化。当前 stage 的 Python 3.12.13 `time.get_clock_info('perf_counter')` 与 monotonic 均报告 `clock_gettime(CLOCK_MONOTONIC)`，与 C 源 `clock_gettime(CLOCK_MONOTONIC)` 相容；当前 stage 的检查不是对已经终止采集进程的独立时钟证明。故所有跨日志关系仍受 clock、PID/TID namespace 和生命周期契约约束。

未来若出现回填，必须使用 `original_tid` 和 `original_begin_ns`，不是写入事件的线程和时间；回填只有起点，不能证明完整 launch 都处于某层 scope。图 replay 的 `model_ri` 只提供模型级候选而非具体图节点和设备执行证明。同线程完整嵌套也仅是 host 区间关系，不证明数据依赖或因果；跨线程不能按时间接近猜测。

## 下一步

先确认原基线健康，再在 stage 单 rank、低优先级解析已采 raw，检查实际可用的数据和数据库导出；本次 profiler_info 的 `_export_type` 为 `text`，不能预先假定存在 PYTORCH_API/CONNECTION_IDS/CANN_API DB。若可获得队列数据，复用已冻结 `_fwk_api_db_parser.py` 规则：层 marker 与同线程 PyTorch 操作、其内 enqueue，按 correlation 找另一线程 dequeue，再关联其内 CANN launch 和唯一设备 flow；每一步检查范围、同名及 ID 复用、时钟转换和缺失。仅关联实际记录，不能把单个关联推广到全部请求或证明 tensor 身份。若 raw 无所需字段或转换不可证，保留 UNKNOWN，不再靠时间邻近推断。

已只读检查 stage 当前安装的 `profiler.py`、`_npu_profiler.py` 和 path manager：`analyse` 支持 `export_type='db'` 及 `max_process_number=1`；path manager 在传入目录本身包含 FWK 或 CANN 路径时只返回该目录。见同目录 `single-rank-analyse-source.txt`、`npu-profiler-path-source.txt` 和 `profiler-path-manager-source.txt`。解析要求输入目录可写，故应把一个 rank 的 raw 复制到任务目录独立副本并仅操作副本。上述是当前安装源码所表明的条件性接口能力，不保证 DB 包含所需表，或原始采集没有丢失；尚未执行解析。
