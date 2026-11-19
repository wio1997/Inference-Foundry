# P 侧选定事件的 trace 坐标重叠：不能把 sum 当墙钟

## 证据与定义

延续[11/10个设备名的关联](052-p-heavy-operators-per-step.md)，逐条使用各自 capture 的 rank0 设备CSV复合键关联 NPU0 trace 事件，再通过 Node、dequeue、enqueue、typed flow 和同线程CPU step标记作候选分组。仅当每个关系唯一才分入某个SG标记；零/多个候选保留未知。对于关联的设备事件，仅在**同一设备trace显示坐标**中按 `[start,end)` 求所选事件并集、按精确事件名求两两交集；未用CPU marker时间剪裁设备事件。

**这些是选中事件的 trace 坐标重叠，不是已校准的实际硬件并发或NPU利用率。** 所选SG 11个名称4920/59176条CSV事件、vLLM所选10个名称5564/35296条；比例只表示事件行筛选覆盖率，不是时间覆盖率。选定事件包括计算、通信以及AI_CPU名称，不是同种物理资源。未选事件的区间未知，单名称独占仅表示“相对这11个名称没有另一名称”，不表示真实设备独占。两个框架不是相同名称/采样窗口，不能比较其并集差作为TTFT差。

同源 host 与 device 的 profiler 元数据文件名和SHA见SG冻结索引，但当前未获取并验证其内容、转换公式与误差；**不能假设CPU/device时钟偏移为零，也不能声称完全不存在同步**。CPU marker只用于已核对的同线程enqueue关联，不表示关联设备区间落在marker内。

## SG 的局部示例：事件之和高于区间并集

第1个EXTEND候选（记录 `c_sk=32768`）选中的11个名称共596条设备事件：

|trace坐标统计|数值 (ms)|说明|
|---|---:|---|
|事件duration之和|1296.369|多名称/stream可重复覆盖|
|所选事件时间并集|954.550|仅被选中事件在设备trace显示坐标覆盖的时长；不裁到CPU区间|
|不同所选名称至少两个同时覆盖|341.819|不等于多层累加时长差的一般公式|
|所选事件最早开始至最晚结束|1572.832|含未选事件/空隙，不能称NPU空闲|

其他7个EXTEND的选定区间并集约739.622–951.696ms，不同名称重叠约325.581–347.670ms。**这只能证明所选trace坐标上不同事件名称的区间重复覆盖，不能证明产生或隐藏了多少真实墙钟开销，更不能从24.127秒TTFT差中扣除。**

对于该样本，两类最大的trace坐标交集分别为：

- `NotifyDispatchMc2AicpuKernel` ∩ `SparseFlashAttention`：约204.418ms；8个EXTEND中约194.687–204.418ms。
- `CamMoeCombineNormal` ∩ `CamMoeCombineNormalMc2AicpuKernel`：约114.164ms；8个EXTEND中约110.616–118.960ms。

区间交集只表明两类记录的显示坐标重合，不能单凭名称推断真正的异步重叠机制、两者互为父子、计算依赖、可消除时间或带宽瓶颈。事件之和−并集在一般三重以上重叠时与“至少两个名称同时覆盖”不是同一量；本次所选样本峰值不同名称数为2，但不可外推整个trace或其他采集。

## vLLM CPU range 搜索的负证据范围

这次扫描完整P rank0 trace的 `ph=X`，对六类预先固定名称/模式检查：`ProfilerStep#数字`、`execute_context_数字(数字)_generation_数字(数字)`、`modelRunnerExec`、`NPUModelRunner._model_forward`、`NPUModelRunner.execute_model`、`vllm.execute_model`；SG/vLLM各自都没有匹配项。这仅对这六类**精确筛选**成立，不证明所有可能命名的forward范围不存在、没有vLLM计算或没有其他标记，也不能凭此构造跨框架step配对。vLLM所选设备事件有4条dequeue候选未唯一匹配，继续保留未知。

## 复核与来源

[区间统计](../evidence/p_overlap_054/interval.json) SHA `b4b8fffa30f5036b84338db7749723a003ee709b0cd27ff76d61d02e7598fcae`，[两两重叠](../evidence/p_overlap_054/pair.json) SHA `39325b32c836d8803d7902b11d839dbdd6000d3cc301173ed3acfd6d77aac23c`；二次独立读取的候选分母、设备源SHA和标记信息与[052结果](../evidence/p_heavy_052/step_detail.json)一致。SG G240是冻结源；vLLM G148是派生副本、非历史冻结CSV。两次只读门控分析均不提交推理，I/O PSI不可用不表示没有影响。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_p_overlap_054.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_archive.py
```

[校验](../scripts/verify_p_overlap_054.py)检查两个源SHA、每阶段零/一/多分母、区间并集上下界、单名称与多名称覆盖守恒及两两交集上界；不验证跨时钟因果、层/请求映射或性能收益。
