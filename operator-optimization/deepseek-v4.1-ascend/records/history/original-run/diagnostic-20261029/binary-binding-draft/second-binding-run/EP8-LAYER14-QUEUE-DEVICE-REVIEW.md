# EP8 第 14 层：从 Python scope 到设备任务的候选关联

2026-10-29，以第二次采集的 EP8 为对象，使用 task-owned 独立 raw 副本，通过 ZCode 在 stage 容器以单进程、低 CPU/IO 优先级执行 `analyse(..., max_process_number=1, export_type='db')`；复制后逐文件校验源与副本 SHA-256。未运行新 NPU 负载，未修改原始采集。执行脚本 `offline_ep8_db.py`、`run_offline_ep8_local.py`，运行输出 `offline-ep8-db.stdout`、`offline-ep8-zcode-output.jsonl`；副本及原始文件哈希在服务器任务目录 `offline-ep8-20261029/input-copy-manifest.json`。

Profiler 日志出现 **`SQLite Error: disk I/O error`**，尽管进程退出 0、`PRAGMA integrity_check=ok`，且数据库中存在 PYTORCH_API、CONNECTION_IDS、CANN_API、TASK、COMPUTE_TASK_INFO。检查时该文件系统空闲约 58.56 TB，指定 TMPDIR 存在，任务副本中小型日志未检索到该报错位置；无法确定错误发生于哪个解析子阶段或是否丢失相关记录。完整性检查只能证明该数据库结构可读，**不能证明导出完整**。覆盖检查见 `ep8-db-coverage-inspection.json`。

按照 sidecar 的精确 call_id 选中第 14 层两次 scope；在同一导出数据库中，各自只有一个匹配 marker，且其内各有一个 `_C_ascend::dispatch_ffn_combine` 和一个同线程 enqueue。依照已保存的 profiler parser 队列规则，enqueue 经同一 correlation ID 各关联一个同名 dequeue；两个 dequeue 都包含一条 CANN `launch` 记录和一条嵌套 `aclrtLaunchKernelWithHostArgs` 记录。每个 `launch` 的 connection 各自匹配到两条 TASK：

| 层 call_id | enqueue / dequeue 共用 correlation | CANN launch connection | device / stream / task | TASK global ID | TASK 的 compute 类型 |
| --- | --- | --- | --- | --- | --- |
| 746-20 | 1342576916 | 97412 | 8 / 47 / 47465 | 71786 / 2538 | 外层无 compute 信息；内层 `DispatchFFNCombine` |
| 746-65 | 1342581857 | 128152 | 8 / 47 / 50913 | 74630 / 5524 | 外层无 compute 信息；内层 `DispatchFFNCombine` |

每条 TASK 的完整时间、model、stream、task、globalTaskId 组合在本 DB 查询范围内各唯一；各组两行的 stream/task 相同，时间相近而起止略有差异，不能作为两次独立内核任务相加。两个 CANN launch 所在 DB 的 globalTid 与各自 dequeue 的 globalTid 一致，均为 3204045609673；Python marker/enqueue 为 3204045603562。已保存的同版本 PyTorch parser 使用 `(pid << 32) | tid`，条件性解码为 746/6857 与 746/746；CANN globalTid 编码及其与 C interposer 真实 tid 的跨工具对应尚未独立证明。明确连接记录支持该 DB 内候选链，不能单凭相同数字称为机器层因果或二进制绑定。

精确 marker、enqueue、dequeue、全部 8 条 CANN 记录、TASK 及条件限制见本机 `ep8-layer14-queue-run.stdout`，同一结果的完整 JSON 在服务器任务目录 `offline-ep8-20261029/ep8-layer14-queue-audit.json`。可复算 SQL/代码为 `trace_ep8_queue.py`，启动及转存为 `run_trace_ep8_queue_local.py`；DB 输入 SHA-256 `95b58de883ee272b7161bce7b213d15673c1925fd165bafbf4deacf7a6b74b23`，线程及数据库字段覆盖见 `ep8-db-coverage-inspection.json`。

## 显式 HostToDevice flow 补充

对同一 EP8 raw 另建独立副本，以单进程低优先级 text 导出；原始复制 manifest 确认源中没有预先存在的 DB 或 trace。text 导出也记录 `SQLite Error: disk I/O error`，完整性仍未证。对 537,146,298 字节 trace 的 793,799 个事件做流式扫描，分别以两条 CANN `launch` connection 左移 32 位为**候选** flow ID；每个候选在扫描到的 s/t/f 事件中各恰有两个 `HostToDevice` 记录，分别为 s/f。s 时间精确等于对应 DB CANN `launch` 开始，f 精确等于对应内层 TASK 开始；按相同设备起点、stream 47 和 task 47465 或 50913 各匹配一条名称 `DispatchFFNCombine` 的设备 X 记录。具体记录、全 trace 哈希与筛选规则见 `ep8-layer14-flow-audit.json` 和 `trace_ep8_flow.py`。

两条 host flow 的 trace tid 均为 6857，与按已存 PyTorch parser 规则拆出的 dequeue 线程号一致；trace 显示 pid 为 764159，属于导出显示标识，不能直接当容器 PID。DB 和 text 来自同一 raw 的两个独立离线导出，connection 左移 32 的关系是本次数据中核验过的候选，不是已证普适契约；只使用导出记录和明确 ID 匹配，没有把跨线程时间接近当成关联。

这些是该采样中 EP8 两次范围内的发现，并非 16 卡全覆盖；queue correlation、connection 的复用范围、数据/tensor 和请求级因果、真实二进制、核内通信等待和真实硬件上界仍未知。DB 中使用 epoch 量级时间，而 C recorder 使用 monotonic，**本结论没有拿两者做直接时间比较**。
