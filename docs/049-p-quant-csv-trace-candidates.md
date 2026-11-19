# P 动态量化：CSV 到同源设备 trace 的候选对应

## 可证事实

对 [P DynamicQuantV2快慢组](048-p-dynamic-quant-subgroups.md)，只读重检每框架 P rank0 的设备 CSV 及 `trace_view.json`；两进程并行、资源门、低优先级，无新推理。仅筛选输入 `8192,4096` BF16/ND、输出 `8192,4096;8192` INT8/FLOAT/ND;ND 的 CSV 事件；trace 限定同采集的 NPU0 硬件 process 与精确 kernel 名。保持 SG冻结索引与vLLM派生CSV/trace来源的不同保证，不跨框架比较时间戳。

|rank0 筛选事件|SGLang|vLLM|
|---|---:|---:|
|CSV 目标签名|120|399|
|同 stream 与开始时间，trace 候选唯一|120|399|
|再要求 Task ID、Model ID 数值相同，且两边键均唯一|120|399|
|上面唯一候选中 duration 十进制数值完全相等|35|236|
|其余候选的 duration 最大绝对差|0.0005µs|0.0005µs|

时间使用源文件数值、JSON `parse_float=Decimal`，没有预设跨文件offset、没有取最近事件强配；ID仅用于内部计数，不输出原始ID或绝对时间。duration并非全部逐字节相等，绝对差上限 **0.0005µs** 与源数据表示精度一致的可能性值得核实，但这不是证明产生差异机制的版本文档或时钟校准。该复合键在本样本内唯一，**不意味着Task ID单独唯一，也不证明CPU调用、layer、forward、请求或跨框架相同step**。无法给出这些映射时须标 UNKNOWN。

同名设备 trace 事件全形状 SG为1024、vLLM为1105；CSV筛选只覆盖其中120/399，不应把它当完整模型的量化事件。SG rank0目标签名的120次设备事件duration算术和为13.392701ms，其中≤100µs的40次为3.557117ms、>100µs的80次为9.835584ms。它们是**一次非计分诊断窗口的事件和，不是墙钟、请求时长或优化收益**，不能拿它与另一轮正式32K cold的24.1272秒平均TTFT差相除或相减得份额、上限或根因。

## 仍缺的链路

现有小样本中 `connection_id` 可连接多种事件，SG样本同一 task/stream可有多个子事件，vLLM样本一个connection可关联32个不同任务；因此不能把ID独立当主键或强制一对一。后续可在同源trace内对上述设备候选查询connection关联的Node/flow事件的数量和类型，先保留0/1/多候选分母，再验证CPU enqueue、嵌套关系及graph replay是否真实连通。若没有可信layer/forward/request标记，应保持UNKNOWN，不能因设备侧复合键唯一而声称完成请求级归因。

## 可追溯性

- [完整数值精确候选计数](../evidence/p_trace_049/exact.json) SHA `14ed159b54f3180470adcb0cc890ed0a203e8cee950a3dc71652fbdd99720a78`；[stream/start/task/model候选计数](../evidence/p_trace_049/task_model.json) SHA `ec2071925278f61b579dea92c2cbc3965aa777f2a53dfe371e068db437fccd72`。
- SG设备CSV和trace与G240冻结SHA/size/mtime一致；vLLM G148派生分析副本与先前报告和本次读取一致，不能称原始冻结CSV/trace。
- 首次两进程远端只读分析已完成，但本地结果聚合对零慢组遗漏默认值导致 `KeyError`，未生成有效输出；保留旧claim。修复后重新运行独立只读分析，确认未发请求，旧失败未计入结果。
- [离线复核](../scripts/verify_p_trace_049.py)只核对来源SHA、候选数和时长差边界，不验证设备到CPU或请求归因。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_p_trace_049.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_archive.py
```
