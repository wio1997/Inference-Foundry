# Task 状态

- Task：`whole-opportunity-coverage`
- 标题：指定冻结场景的整体优化机会完整覆盖
- 算子：`DeepSeek V4.1 Flash W8A8 whole inference graph and state`
- 开发框架：`vLLM-Ascend`
- 状态：`ACTIVE`
- 阶段：`DEFINED`
- 活动 Loop：`loop-005`
- 已接受基线：`NONE`
- 证据成熟度：`E1_RUNNABLE`
- 用例数：`7`
- 当前知识条目：`1`
- 下一步：继续全部模块/条件分支和跨算子/层/卡/迭代原目标。上一轮v165真实共享profile与恢复是PROGRESS，本轮真实NPU-only采集、32DB/966674执行、64条Python-replay到native模型的关系和288个新native完成范围也是PROGRESS，整体未完成。直接读shared-profile-analysis-v166的device-only-evidence-review、ALL32-NPU-ONLY-RANKING、NATIVE-ENVELOPES、REPLAY-MODEL-BRIDGE与typed-communication-neighborhoods，不重复采集本批数据或重复旧CPU测试。原采集完整成功且8输出相同；控制器误将16个复用两次的profiler对象要求为32个，后置检查触发额外恢复。错误保留，原脚本与服务已恢复，当前没有保留可复用live profiler能力。不要因为该后置错误重跑成功Case。F21在新窗口累计任务工作时长占42.44%，仍为第一；旧68.28%和新窗口不能作生产加速比。优先复用现有F21原生协议/条件MAC合同，细分实算/通信/等待及小行数/专家负载策略；现有profile不能定案时只定点补算子代码或必要元数据。旧171 cohort已有64个同线程replay范围严格包含RuntimeAPI，再经官方connectionID绑定MODEL_EXECUTE/MODEL_WAIT_COMPLETE，105600图内计算执行唯一归属；只是本域graph身份/执行范围，不是每个捕获算子的源位置、参数或全部reader，也不回填169/166历史。新NPU-only cohort已有288个原生模型完成范围、475200图内计算执行，每范围40个DispatchFFNCombine；没有CPU/source绑定，不按model49同号跨进程或跨cohort串接。通信424个实际CPU栈邻域区分SP输入/最终HC、Engram和DSpark，按原F26/F32/F24等既有条件变体复用，避免重查已分析代码或新增笛卡尔配对。dtype已修正为HCCL专用enum，group_name_N仅本地别名，count非wire字节。针对本地M3/24expert/K6的形状，条件工作多面体给出同一EP16交换且所有rank<=3行时N<=48、assignment<=288、至少96空expert；不能把单rank形状套到混合prefill/decode交换，不能由空expert删除同步协议，不能当25%时间收益。保持35+7候选、26模块358行及G00-G14全范围；原9/1352、849PMU、16phase、3136归属、copy/control、数值/读者/物理容量与最低MFU仍open。当前授权覆盖自有容器推理/框架/探针操作，不重复询问。
- 更新时间：`2026-11-02T18:37:56Z`

## 最近 Loop

共 `5` 个 Loop；完整索引见 `loop-index.jsonl`。

| Loop | 状态 | 结论 | 决定/下一步 |
| --- | --- | --- | --- |
| `loop-001` | `ACCEPTED` | `ACCEPTED` | 这里只满足Loop001冻结的局部门槛（不是Task或全局Goal）：固定源码身份、模块/分支/状态ID底账，且新增初始化/尾部/MoE局部语义合同。573索引、47局部合同和19状态边均不是语义穷尽，0整模块关闭，整体目标未完成。 |
| `loop-002` | `PIVOTED` | `PIVOTED` | 合成graph helper与全负载参照有效，但模型取证入口的额外512行断言不成立为通用调用前提；源selector可无num_tokens检查进入custom FUSED。首轮未得到有效模型计数，原服务已恢复。修订形状域与ownership非原子采样处理后建立派生Loop。 |
| `loop-003` | `ACCEPTED` | `ACCEPTED` | 只满足本Loop有限观察合同：真实TensorList快照、同调用组守恒、实际容量65536与custom分支、各层/rank样本身份及恢复均已复核；正式C8样本87组覆盖43层。未覆盖全部迭代，40组graph的阶段未定，缺口继续保留；whole_modules_closed=0，整体Goal未完成。 |
| `loop-004` | `PIVOTED` | `PIVOTED` | Migration changes the active case/hardware to171. Loop004 preserves its frozen169/166 case domain and all scoped profile/cut evidence;full-scope completeness is not achieved. Continue in a derived171 evidence Loop,not by relabeling171 probe as166. |
| `loop-005` | `EVALUATING` | `PENDING` | 审查 Run run-20261102T183752Z 的证据，并判断是否需要更多 Run |

## 阻塞项

- 暂无
