# 当前适配与性能验证状态（2026-11-10）

DeepInfer原生完整模型接入、7物理卡上的8 die/TP8权重全驻留和单请求AISBench基线已实测。**C1对照已完成：DeepInfer为vLLM的0.675倍，至少1.5倍目标未达成；当前不代表生产批处理适配完成。**

## 已通过的范围

- 请求由DeepInfer LLMEngine、原生C++ scheduler与Worker执行完整45层GLM-5.3-Flash-W8A8，模型计算复用固定镜像中的Ascend模型模块、CANN和FLA-NPU；目标路径不通过vLLM引擎调度。
- TP8运行在8个活动die、7张物理卡上；全部非空模型参数位于NPU，NoopOffloader，无CPU权重卸载。不能把该结果称为8物理卡/TP16测试。
- 冻结工程24用例在该TP8路径通过24/24。此前TP4开发版本及干净源码重建版本均为23/24，且独立参考引擎24项最终输出相同。这是工程筛查，不是正式通用质量benchmark。
- 流式`/v1/completions`真实请求通过，串行请求的完成与状态释放已核验。默认reference后端仍只有一个活动模型请求；新增显式sdk-batch候选已通过原生双请求与槽位复用，但HTTP并发及C4/C8性能尚待验证，不能把默认HTTP排队当批处理。
- 源码`1b0101b41395e66335461feb884a5dfdc18a258b`完成AISBench正式C1测试：2次热身排除，32/32计分请求成功，0失败，每条1024输入/256输出token，输出吞吐**4.8031 token/s**。TTFT中位数261.8ms，TPOT中位数208.2ms；八个worker均exit0。

公开的测量配置和统计见[本次AISBench结果](validation/aisbench-native-tp8-c1-20261110/result.json)及[延迟统计](validation/aisbench-native-tp8-c1-20261110/latency.csv)。原始请求、token向量及SQLite时间戳证据保留在私有任务档案中，不公开服务器路径或凭据。

## 当前正在推进

相同8 die、TP8/DP1/EP关闭、eager、1024入/256出、32计分请求的C1对照已完成，双方32/32成功、0失败，热身2条排除。独立重算原始时间戳：

|指标|DeepInfer|vLLM|
|---|---:|---:|
|输出token/s|4.8031|7.1207|
|TTFT中位数ms|261.8331|172.2075|
|TPOT中位数ms|208.2030|139.7292|

DeepInfer/vLLM=0.6745233，尚未达标。详细数据见[对照统计](validation/aisbench-native-tp8-c1-20261110/comparison.json)。镜像实际版本为vLLM `0.30.0+empty`、vLLM-Ascend `0.30.0rc2`，不能将core版本冒充rc2。DeepInfer使用请求私有缓存、参考使用2GiB池，缓存内部管理方式不同；本轮仅C1，不能外推C4/C8。

后续批处理将优先复用固定SDK的ModelRunner及缓存组契约，保持DeepInfer拥有引擎与调度。未在硬件上验证的候选不作为默认性能收益。

## 未完成项与边界

1. 已取得C1完整对照且当前落后；至少1.5倍、争取2倍的目标尚未证明。
2. C4/C8真正连续批处理及其请求状态隔离、质量与性能验证。
3. 长上下文、正式大规模质量、生产服务能力及8物理卡/TP16验证。
4. 全部历史实验/CPU回归未因本轮硬件结果自动变成通过；本轮结论仅覆盖上述版本和负载。

权重及镜像二进制不随此仓库分发。未将框架状态、元数据或调用适配工作描述为新开发CANN底层算子。

## SDK计算单元的原生批处理候选

显式参数`--compute-backend sdk-batch --max-sequences 2`已在同组8die通过真实原生调度测试：实际批次2，分别输出E047、宁波；释放后第三请求再次输出E047，缓存所有权清空，8rank均exit0。证据见[批处理及槽位复用](validation/native-sdk-batch-smoke-20261110/result.json)。这只覆盖三个短工程请求，不是正式通用质量结论。

此候选复用固定SDK NPUWorker/ModelRunner与KVCacheManager，DeepInfer仍拥有LLMEngine、C++ scheduler、请求生命周期及输出回调。没有创建vLLM引擎、调度器或Executor。SDK调度输出数据结构仅作计算描述。默认reference后端保留；新的HTTP并发实现尚待实际验证，C4/C8和1.5倍收益均未宣称通过。

HTTP候选采用原生队列容量感知的入队与终止回收；SDK批处理模式把prefill尝试间隔由旧默认70/100个decode step设为1，使DP1每次迭代检查新请求。上述HTTP/调度策略正在独立硬件验证，不能将此前batch2烟测当作其全部通过。未修改模型公式、量化方式或采样规则。

HTTP硬件集成曾因数值token输入校验遗留旧runtime.backend字段失败，未进入性能计分；已改为SDK runtime的词表接口，并增加纯标准库的[入口回归检查](validation/native-sdk-batch-smoke-20261110/check_http_entry_contract.py)，覆盖实际Handler的token/text输入、非法输入和SSE结束帧。该检查不加载或模拟模型，也不能作为NPU/性能通过证据。修正版HTTP硬件验证仍在进行。
