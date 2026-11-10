# 框架到算子的性能账单

目的：用可对账的实际时间确定优先优化区域，再估算数学改写、融合、megakernel、调度等方案的净收益。函数慢、设备忙或设备空闲均不能单独证明原因。

## 顺序

1. **锁定实验。** 记录框架/模型/后端源码版本、硬件、拓扑、并行配置；固定输入输出长度、并发、路由、prefix cache、预热和采样位置。先测无profiler请求耗时与吞吐。保存输入/输出哈希、成功token数及负载配平证明。
2. **记录完整请求。** 从客户端提交到最终输出，用轻量标记保留接收/预处理、入队、组批、worker调用、结果回收和输出边界。异步请求须按request/batch ID连接；未打通的关联保留未知。
3. **只对热点窗打开详细profile。** CPU、提交线程、设备计算、collective、依赖事件分别保留。记录函数入口、源码行、PID/TID、父子关系、单调时钟、墙钟和线程CPU时间。原生kernel保留调用次数、shape/dtype/format、stream、task/flow与PMU。
4. **先对账，再归因。** 单线程父墙钟=子区间并集+自身区间；CPU与设备、多卡与多stream不相加。计算/通信重叠区只计一次，排队、等待、未知单列。报告包含时长与自身时长。线程CPU与墙钟之差可提示未在本线程执行CPU，不能直接命名为GIL、I/O或调度等待。
5. **按可暴露成本排序。** 先看设备空闲时上游在做什么、关键结果在等待什么，再看算子按实际调用点/shape的成本。按数据交付由大到小考虑计算消除/共享、重排、融合和megakernel；代价小的原语通常并入大区域，不单开开发项目。
6. **先算净价，再开发。** 同输入就绪与同外部结果就绪边界比较。Δt=T原−T候选；候选计替代计算、存储、通信、同步、争用与旁路交付。给局部净省、命中次数、关键路径可传递部分、条件和不确定性。乐观预算不足则停止同范围小方案。
7. **用无profile复核。** 通过正确性后，比较相同负载的端到端时间、吞吐和延迟分布；重新检查瓶颈转移。多个方案收益不直接相加。

## 本工具的边界

`sitecustomize.py`+`ft_hook.py`是**同步Python函数**的诊断标记器。通过独立PYTHONPATH，仅在诊断进程中对指定版本源码做内存AST变换；不写框架源文件，不检查张量值，不新增设备同步。保留签名、装饰器、返回值及异常传播。它会增加Host开销，必须与off/coarse/详细模式参考比较，不将详细profile当生产速度。

控制文件`$FRAMEWORK_TIMING_ROOT/control.json`：

```json
{"mode":"off|coarse|detailed","label":"run-name"}
```

adapter JSON指定模块SHA256、函数与粗粒度入口；SHA不符时拒绝加载。`coarse`只记录指定父阶段；`detailed`记录全部选中函数并添加`FT::` profiler范围。每个进程按JSONL输出；原始日志不存参数、请求正文或凭据。PID内ID用于父子关系，不能冒充跨进程batch ID。

当前已实接：固定版本vLLM Ascend、DeepSeek-V4.1-Flash/DSpark，910C/CANN9.2；实际结果与采样限制见`audit/framework-timing-171`。适配SGLang、DeepInfer时复用账本/采样/对账方法，**重新按真实版本确定入口和SHA**，不能照搬vLLM的函数名或耗时。异步coroutine须使用contextvars与显式request关联，不能直接用本标记器的thread-local父栈。

## 新框架需要绑定的入口

| 职责 | 需要看到的交付 | 核查方式 |
|---|---|---|
| API/预处理 | 请求接收→token就绪→入队 | 客户端时间+服务器request ID，分清tokenize/cache |
| 调度/组批 | 排队→batch生成→提交worker | 请求队列、调度函数、执行队列与batch代次 |
| worker输入 | 接收→状态/长度/位置/slot/metadata就绪 | 子函数范围+Host操作+H2D/同步事件 |
| 模型计算 | 输入就绪→目标/草稿结果就绪 | graph replay与eager区分，全部stream/卡 |
| 采样/回收 | logits→接受token→状态更新→发送 | 采样/校验函数、D2H、future等待 |
| 原生算子/通信 | 提交→设备开始→结束→消费者 | task/stream/flow、shape、PMU和拓扑 |

## 输出契约

每次实验保存`ENVIRONMENT / workload / client results / spans / trace inventory / source adapter / function ledger / operator ledger / findings / report / restore receipt`。报告首页必须先回答：**最大的时间块在哪里、已定位到什么层级、可调优成本的条件上限是多少、还缺哪一条会改变开发取舍的证据。**

时间字段：inclusive_wall、self_wall、thread_cpu、self_thread_cpu、device_compute、exposed_communication、device_free_overlap、request_critical_exposure。百分比必须注明分母。测得的成本、条件上限、预期净省、已兑现收益分别列出，不混写。

本次采集初版hook在off模式仍反复轮询控制文件，20.163s诊断轮不适合作为无探针基线。原版按SHA保存在`audit/framework-timing-171/captured-hook`；通用工具已改为关闭范围内只在外层读取控制，不在子函数重复读取。语义/异常/父子区间校验通过，该改进尚未在第二次NPU采集中重新测开销。

**完整timeline的主入口**：[交付图→多轨道→区间对账→覆盖验收→按价值下钻](TIMELINE.md)。该文明确完整性如何验收，哪些缺口必须保留，避免仅有函数排行榜却漏掉队列与异步交付。
