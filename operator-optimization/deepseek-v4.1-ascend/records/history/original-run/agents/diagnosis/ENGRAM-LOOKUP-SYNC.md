# EP8：Engram lookup 与层间事件的关联边界

## 结论

已有 trace 在 EP8 的第二、第三个选定图执行之前，分别记录到两次 `_engram_host_uva_gather_dequant_kernel`，并在六个目标层间隔记录到其他 stream 的 `EVENT WAIT`。**当前不能把这些记录连成“对应本轮 lookup → ready event → 主流等待 → MatMul”的因果链**；更不能据此把大矩阵算子的耗时归因于 lookup 等待。第一轮在规定范围内没有这四个精确命名 kernel 的记录，不能证明它没有执行。

目标为同一 rank、同一 device 时间坐标上的三次图执行、各自第 0→1 与 13→14 层间隔；与模型源码结构一致的 MatMul 是强候选，并非已证明的 tensor / Python 源码身份。由 graph replay 的任务 connection ID 复用，不能按 ID 或时间邻近单独配对调用或依赖。

## 采集覆盖与截断

- stage118 的 EP8 DB 确有 `TASK`、`COMPUTE_TASK_INFO`、`CANN_API`、`PYTORCH_API` 和通信任务表，字符串字典包含四个精确 Engram/hash kernel 名。三窗口各自涉及到的 `EVENT RECORD` task 行计数为 318、269、269，`EVENT WAIT` 为 190、191、191；统计包含每个图开始前 10 ms，且以时间重叠选择任务，可能包含窗口开始前已开始的长任务，不是请求数或不同 event 对象数。
- stage118 候选明细三次都被 300 行上限截断，不能用来排除目标名称。stage120 将四个精确 kernel 名与六个间隔的同步事件分别查询，均未截断。`%SYNC%` 同时选到了 `MEMCPY_ASYNC`，这些 memcpy 行不是同步 event，未用作 wait 证据。
- `TASK` 记录 stream、model、task、connection 和时间，但该表没有 event 对象 ID；schema 中未见可据此直接配对 record 和 wait 的关联字段。字典或 task 中有部分事件不代表 profiler 记录了所有同步，未观察到的事件不能推断运行时不存在。

## 同一 rank 的具体结果

以每次 `MODEL_WAIT_COMPLETE` 的 TASK 起点为零点，在图前固定回看 10 ms，并查询到该轮第二个大 MatMul 结束。这个 10 ms 是**搜索范围，不是请求标识**；此处只比较 EP8 device 记录，不比较其他卡或 host 时钟。

| 图执行 | 四个精确命名 kernel 在窗口中 | 两次 host-UVA gather 结束相对于图起点 |
|---|---|---|
| 1 | 0 条 | 未知，可能在窗口外或未导出 |
| 2 | req-index 1、cache-write 1、hash-ids 1、gather 2 | 提前 974.22、487.67 µs |
| 3 | req-index 1、cache-write 1、hash-ids 1、gather 2 | 提前 928.32、464.41 µs |

第 2、3 次这五个任务都在 stream 47、modelId 4294967295；目标 MatMul 在 stream 89、modelId 49。不能只根据数量和时间证明它们属于同一 batch、对应层号、记录了哪个 event，或真正供给该 MatMul。

六个层间隔内均有 stream 87 和 88 的 `EVENT WAIT` 行完整覆盖间隔；这些任务 modelId 49、connectionId 184330，与 MatMul graph connection 相同，而该 connection 在大量图任务中复用。六个间隔在 stream 89 中均**没有导出的 `EVENT WAIT` 重叠 task 行**。这是有边界的 trace 观察，不排除别的同步形式、未导出的等待、间隔前的依赖，或间接资源阻塞；也不证明 stream 87/88 的等待导致主流慢。

源码的 `prepare_engram` 在发布 mask 后可进行 hash、lookup、通信和输出填充，最终 record 每层 ready event；但不参与 lookup 的分支也会清零并 record。因此 event 即使可识别，也须结合运行分支和 buffer 版本，才能确认产生了真实 lookup。源码消费者 `wait_lookup` 同样不能单凭源码证明本次采样走了相应分支或等待了多久。

## 断链与最小补充证据

1. **请求/迭代归属缺口：** 尚无同轮 request 或 batch ID 与这两个 gather task、模型图实例的可靠对应。需可复核的图 replay / 请求标识，及 lookup 输出 buffer 的标识与版本；窗口内先后顺序不能代替。
2. **同步对象配对缺口：** 需从当前采集的其他原始记录或已有 profiler 导出中找到 event handle / record-wait 配对与版本语义。若不存在，只能标为未知，不能将邻近 record 和 wait 配对。
3. **数据消费者缺口：** 需能将 lookup 写入 buffer 与候选 MatMul 的输入绑定，例如相应运行期张量或 buffer 地址、布局和调用关系；模型源码和 shape 仅为候选依据。
4. **等待和时长缺口：** 即使配对成功，还需要明确事件是否位于关键路径，不能将其他 stream 的长 wait 求和或直接加到主流耗时。优先检查既存原始采集是否保留标识和依赖信息；没有的话保留 UNKNOWN，不以零代替。

## stage121：原始 trace 的目标附近事件及 flow

对 EP8 `trace_view.json` 流式读取了 704,318 项，限定 stage120 的 10 个 producer 任务、12 个 stream 87/88 的 wait、以及 6 个候选 MatMul 的起点各 ±2 µs。28 个目标都未触发输出截断；每个目标在附近各找到一条与数据库起点及 stream 完全一致的 trace `X` 行。此对齐是这 28 个样本的观察，不是跨域时钟误差保证。

12 条目标 `EVENT WAIT` 的**筛选后输出**字段包括 `Model Id`、`Physic Stream Id`、`Task Id`、`Batch Id`、`Subtask Id`、`connection_id` 和 `Task Type`，筛选字段里没有 event 对象 ID。脚本为了不输出请求内容而过滤了原始 args 的其他键和字符串值，因而不能据此声称原始 args 一定没有 event 对象标识。选定 wait 起点没有同时间的已输出 flow 终点。

六个候选 MatMul 的起点各有一个 flow 终点，ID 均为 `791691321671680`，数值上等于 `184330 << 32`；该值跨三个图执行和两处层间隔复用，不能据此把它当成单个请求、单个 MatMul 或具体 lookup 的唯一 ID。10 个 producer 起点也存在 flow 终点；例如一个 gather 的 `680765201317888` 数值上等于其 connection `158503 << 32`。这些是数值和时间对齐现象，尚未查证本 trace flow 类别、始端、复用和版本语义；不能将其当成跨流 ready event 或 buffer 消费依赖。

冻结的 `sources/primary/frozen-image-clock/torch_cann_file_parser.py:188` 中，解析器对 `cat=HostToDevice` 的 `s/f` 流按 ID 组合，以终点 `pid/tid/ts` 查询 kernel；这是 host 到 device 的解析逻辑，而不是 ready event 的配对定义。stage121 安全输出未包含 flow category，不能证明上面采样到的 flow 就是该类。后续通过 [限定 flow 端点查询](inspect_ep8_selected_flow_endpoints.py) 检查了三个窗口的已选 ID，结果见下。

## stage123：确认 producer 的 host 到 device flow

对 stage121 同一 EP8 trace SHA 重新流式读取，按三轮预定窗口和预选 ID 查 flow：第 2、3 轮各 5 个 producer 的 connection 派生 ID，均有一条 `s` 与一条 `f`，两端类别均为 `HostToDevice`，每个 ID 在对应窗口共两条、没有截断。10 个 `f` 终点的时间和 stream 均与各自 producer task 起点和 stream 精确一致，与冻结解析器的 host→device 关联方法一致。具体是哪一层 lookup、属于哪个请求或产生哪个 buffer，依然未知；这些关联表达下发信息，不能证明 ready event 的配对或与 MatMul 的数据依赖。

共享 graph ID 在三轮窗口各有 581 条记录，每轮仅输出前 30 条，已明确截断；这 30 条均为 `HostToDevice` 的 `f`，但**不能将其类别推广到没有输出的 551 条，也不能在此结果中确认六个 MatMul 对应的 flow 终点类别**。每个 producer 按其开始时间构造的第二类 ID 各找到一条 `s` 和一条 `f`，类别不是 `HostToDevice`；脚本只输出其他类别的哈希，类别具体含义和是否与同步有关仍未知。筛选限于每轮指定的 10 ms 回看及图窗口，窗口外端点未检查，host 与 device 时间也未作差。

当前可复核的缺口仍是 producer 与本轮图的归属、实际 event 对象配对、以及 lookup buffer 与 MatMul 输入之间的关联。后续若查 flow 始端，须先从既有 exporter 源定义确定 flow 类别及 ID 语义，然后只针对发现的固定 ID 和本 rank 窗口采集安全字段；不能以相同 ID 的复用证明唯一依赖。

## 可复核文件

- [stage118 查询与输出](engram-producer-events.json)：SHA-256 `5796454585d70fb5d1a584016dcf3d5e312f40285c25c5642fcd4feffd75a3a3`；结果的 candidate 明细明确截断。
- [stage120 精确查询与输出](engram-producer-events-exact.json)：SHA-256 `ea31f7d421d10109a6cab2a14d13c084dda169761b4085177a658741821964cb`；数据库 SHA-256 `67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58`；六个 interval 的输入文件 SHA 均在 JSON 内。
- [stage120 查询脚本](query_engram_producer_events_exact.py)：SHA-256 `912b5662bdee0b2d1060a6c97d7ebc441368ba28695a4c2f1b8cb539bb578d72`。读取现有 EP8 SQLite，`mode=ro`、`query_only=ON`；由主控交 ZCode 在已有 CPU staging 容器运行，无新 NPU 负载。
- [stage121 trace 检查结果](ep8-trace-event-args.json)：SHA-256 `d2e9c8ee7ce178c106dbb02d7b3aae1bf59dc1f65206607e971ed65d156a7959`；读取 trace 大小 `150646621`，首次记录 SHA-256 `3abc8f27420c733c39810ec86e9144634a074c22c5b9fcff0e3b39929b63dc99`，没有已冻结的 EP8 trace hash 可独立比对。检查脚本 [inspect_ep8_trace_event_args.py](inspect_ep8_trace_event_args.py) SHA-256 `3a6032ec7e4379f83406c365c38c333f3691f04d03fd38b9f17bdc7ba65a6193`。`operator-ceiling169-v2-121-ep8-trace-ids.exit-code` 为 `0`，`.verified.json` 的 `model_verified` 为 `true`，`observed_models` 为 `deepseek-flash`。
- [stage123 flow 端点结果](ep8-selected-flow-endpoints.json)：SHA-256 `14e07d4311d679b26cc2262eaaa79652dc201b5fd55042b0f61045232245a8b2`，读取 trace SHA 与 stage121 相同；`operator-ceiling169-v2-123-ep8-flow.exit-code` 为 `0`，`.verified.json` 的 `model_verified` 为 `true`，`observed_models` 为 `deepseek-flash`。[脚本](inspect_ep8_selected_flow_endpoints.py) SHA-256 `526b69088307bf5410901e1a33033a7cb852eb18d8d299dd82d6dcba22a2a14a`。

执行记录：`operator-ceiling169-v2-120-engram-exact.exit-code` 为 `0`；`operator-ceiling169-v2-120-engram-exact.verified.json` 的 `model_verified` 为 `true`，`observed_models` 为 `deepseek-flash`。日志位于 `/Users/wio/work/173-sglang-matched-172-benchmark/`。
