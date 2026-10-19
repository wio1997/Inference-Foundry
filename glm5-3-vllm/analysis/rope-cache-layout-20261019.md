# RoPE 只读表布局优化：真实 PD 的端到端收益

当前 GLM-5.3 W8A8C8 / TP16 P + TP16 D、原样官方 proxy、全 SDMA 通信路径的固定混合负载中，C4 吞吐从 **143.517 提升至 164.631 output tok/s（+14.71%）**。三轮 ABBA 分别为 +12.69%、+16.59%、+14.90%。C1 等权 TPOT 下降17.26%，完成延迟下降14.95%，TTFT下降3.87%。这是当前有限点的接受结果，不是最大持续 SLO 容量或所有业务质量结论。

| 指标 | 原生 | 连续只读缓存 | 变化 |
| --- | ---: | ---: | ---: |
| C4 output tok/s | 143.517 | 164.631 | +14.71% |
| C4 TTFT p95/ms | 1426.574 | 1308.674 | -8.26% |
| C4 完成延迟 p95/ms | 3639.342 | 3276.440 | -9.97% |
| C4 TPOT p95/ms | 24.068 | 21.451 | -10.87% |

原生 `_record_cos_and_sin_cache_interleaved` 发布的 cos/sin 是非连续 view。旧 rank0 trace/CSV 精确关联780个 `aclnnIndex_SliceAiCore_Slice`，每次将 BF16 `[1048576,128]` 物化为 `[1048576,64]`（约128MiB），然后才按少量 positions 取值，单个 Slice 中位359.872us。duration和可以重叠，未将其当作墙钟收益。

实现保持原生 `get_cos_and_sin_mla` 函数体，模型构造时把两张只读源表各保留一份连续副本，注册为同一原生 RoPE owner 的非持久 buffer，早于编译与图捕获。在真实 D 的 execute/sample scope 内临时选择源表，调用原生 getter 后恢复 globals。原有 `_cos_mla/_sin_mla` 对象、固定输出地址、模型/attention/KV 算子、精度和通信保持原有方式。源表、地址或 owner buffer 被替换时回退原生。源码仍显式开关；已在本次自有有限研究实例启用，observer关闭。

代价为 **256MiB/rank，D TP16合计4GiB**，实验中两种模式均持有这份额外内存。没有改变模型权重、W8A8C8 或 int8 KV/Indexer；P仍是原生 PIECEWISE[256,1024]/K1，D仍为原生 target FULL/K5，MTP whole graph保持原生关闭。

单NPU片段使用同shape/stride/dtype，完整源表与N1/4/6/18/24输出逐位一致，双表查找约0.76→0.05ms；它只作为候选价值证据。随后14项CPU guard/所有权/异常/指针测试、16条普通功能、8条C4功能、all16真实getter与源表布局、完整表逐位一致、固定图输出地址、26×2完整64-token、16条缓存复用均通过。26组完整输出在两模式间完全一致。正式24条C1与384条均衡C4全部预算和数值通过；C4原生本身仍存在序列变化，不将有限数学任务推广为一般质量接受。

正式测量是相同P/D、同输入256/257/1536/1537、64/128输出、四闭环slot每块各8条、case轮换、ABBA×3，observer/profile/export/相关传输全部结束。分块原生计数在请求完成后、计时区间外收集，两次稳定且generated精确匹配预算。两模式各生成18432 tokens；native draft5025/accepted13881，candidate draft5083/accepted13832。因此收益没有依赖更高接受率：generated/draft略低1.14%，draft计数吞吐高16.04%。这只是账面分解，不是完整GPU/host关键路径因果分解。C1两case的序列及draft/accepted/generated相同而延迟下降。

C4两个case的TTFT中位数稍高（N1536 +2.65%、N257 +5.23%），混合负载整体中位数/p95改善；不称每case TTFT均改善。取消验证10次client close、16次完整peer/reuse且P/D KV/running/waiting回零；明确backend abort日志/计数均0，不声称内部abort完成链已证明。

完整verdict及来源SHA位于 `evidence/20261018-p-first-seed-device-v2/recovery-v2/rope-cache-layout-formal-verdict-v1.json`。live启用收据为 `rope-cache-layout-live-enabled-v1.json`。P/D完整源与日志备份分别335225B/SHA7fe2d413…、1798682B/SHAc2982281…；当前CID、有限期限与回退位置见HANDOFF。此前MoE/SFA与P图桶的否定结果保留；RoPE改变了执行条件，后续组合候选须有新的供给/依赖假设，不把旧条件的否定当永久禁令。


后续短 trace 只采一条启用连续表的 P→D8 请求，16 rank 均停止后在 CPU 导出 rank 0。该 trace 中原百万行表 Slice 为 0；MTP MoE 的 325 个关联 compute task 从 host enqueue 到 device 执行至少提前 7.09 ms，中位 22.39 ms。25 次 core 的 host 提交结束到首个设备计算至少 6.64 ms。这里有 profiler 开销，不能当作未采样服务的关键路径证明；当前证据支持先放下盲目 MoE/SFA 组合重载。

下一项由新反常现象触发：target 图 5 次验证中出现 375 次 AICPU Cumsum，中位 44.32 μs。原生 AllGather dispatcher 请求 routing 的 count histogram，W8A8 GMM1 随后转换为 prefix，GMM2 又接受输入的 list type。已有 [Ascend routing 接口](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu-npu_moe_init_routing_v2.md)支持 type0 直接返回前缀和。正在以单 NPU 的 routing→prefix 与既有 int8 GMM 片段验证相同值/顺序与编译后成本；尚未改整模型或宣称服务收益。证据见 recovery-v2/rope-supply-analysis-v1.json、moe-count-pinned-source-v1.json。

成功窗口 P/D 均在 watchdog 到期后 exit124、OOMFalse，所有 NPU 空闲；已保留完整日志和 trace。启用状态是该有限研究窗口内的实际状态，不表示常驻生产服务。


后续 prefix 原型首版的整模型功能结果为原生 8/8、候选 6/8，候选续写异常，已关闭且未进入正式计时。all16 的实际选路通过与数值失败同时保留，二者不混淆。随后单 NPU 用原尺寸 NZ 权重串起 routing、融合 GMM1、GMM2、unpermute，8 个分布及 96 次交错图 replay 全部逐位一致；该片段仍不含完整 TP、shared overlap、attention/KV 与 FX 集成。当前保留已加载 P，仅 D 换入原生副图、MTP/target 分离对照与真实 core 成对比较，查清上下文依赖后再决定。RoPE +14.71% 的已验收结论不依赖这项后续原型。


D2隔离进一步表明：MTP-only两case完整64与native一致，16rank共32组真实MTP core成对输出bitwise；仅重复原生target计算的native-replica图却两case续写均异常。因此图构建本身有独立问题，未正式计时。源码中的`_int64_kv_slots`把int32→int64转换结果缓存在同一metadata动态属性上；同对象连续捕获时，后图可能直接引用前图Cast产物。对精确函数AST的CPU所有权模型重现了这一跨图依赖，仍不是D2内部缓存的设备观测。D3分别走原框架预热/捕获、重新构造metadata，并加variant warmup和有界输入地址记录；正在真实模型验收，尚无新收益。
