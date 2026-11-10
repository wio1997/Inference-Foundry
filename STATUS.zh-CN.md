# 当前适配与性能验证状态（2026-11-10）

DeepInfer原生框架已完成GLM-5.3-Flash-W8A8的45层完整模型执行、8die权重全驻留、真实批处理与HTTP服务接入。**新图执行候选正式AISBench达到C4=104.2310、C8=197.0904输出token/s，两档各32成功0失败；相同图配置的vLLM在正式负载中发生CausalConv1d设备错误，整批对照无效，至少1.5倍框架收益尚未证明。此前eager同条件对照分别为参考的0.9033/0.9695倍。**

## 已验证的执行架构

请求经过DeepInfer LLMEngine、原生C++ scheduler、Worker及输出回调；计算和物理缓存复用固定版本的SDK NPUWorker/ModelRunner与KVCacheManager、Ascend模型模块、CANN/FLA-NPU。没有创建vLLM引擎、调度器或Executor；SDK SchedulerOutput仅作计算描述。

7张物理卡提供8个活动die，TP8/DP1/EP关闭；全部非空模型参数驻留NPU，NoopOffloader，无CPU权重卸载。这不是8物理卡/TP16验证。

默认reference组装后端保留。显式`--compute-backend sdk-batch --max-sequences 8`使用新的原生批处理接入；SDK DP1模式每次迭代检查prefill（间隔1/1），HTTP按原生队列容量入队并回收终止请求。该路径已验证真实模型批次8，不能与默认reference后端的串行HTTP排队混淆。

## 正式性能结果

分段图的有效参考已完成：同8die/TP8/DP1/EP关闭、PIECEWISE、compile mode0、capture[1,2,4,8]、`VLLM_USE_BREAKABLE_CUDAGRAPH=1`，vLLM C4=62.8418、C8=109.5219输出token/s，两档各32成功0失败，原始时间戳独立核验。DeepInfer同设置已通过两个并发工程guard，正式计分进行中。源码显式`--graph-decode --breakable-graph`，不改变默认eager路径。[分段图证据](validation/native-sdk-breakable-graph-20261110/result.json)。


新增显式`--graph-decode`复用SDK的`FULL_DECODE_ONLY`，capture sizes=[1,2,4,8]、compile mode=0。真实日志确认4/4图捕获完成并启用ACL replay；模型数学、量化、缓存状态协议不变。源码`f05f912bdbdc6c19009295853951732d3a83db39`，默认eager路径保留。

|图执行并发|DeepInfer输出token/s|TTFT中位ms|TPOT中位ms|同图vLLM|
|---|---:|---:|---:|---|
|C4|104.2310|489.1247|34.4641|设备错误，计分无效|
|C8|197.0904|737.8700|37.7958|设备错误，计分无效|

每档32/32成功、0失败，1024入/256出、2热身排除；原始SQLite时间戳独立复核。8rank exit0、owner清空、资源释放。详见[图执行证据](validation/native-sdk-decode-graph-20261110/result.json)。以下为保留的eager对照，不能把新图执行结果与旧eager参考相除宣布达标。

|后端与并发|DeepInfer输出token/s|vLLM输出token/s|DeepInfer/vLLM|
|---|---:|---:|---:|
|旧reference组装，C1|4.8031|7.1207|0.6745|
|SDK批处理，C4|23.5156|26.0326|0.9033|
|SDK批处理，C8|45.8135|47.2544|0.9695|

C4/C8每档32请求，全部成功、0失败；每条1024输入/256输出，2热身排除。原始SQLite时间戳按请求引用独立重算，并与AISBench全局输出吞吐一致。

|SDK并发|TTFT中位ms|TPOT中位ms|成功/失败|
|---|---:|---:|---:|
|4|487.4230|160.8599|32/0|
|8|733.2418|167.5923|32/0|

C4/C8源码为`77bbd9f89f07d7a93d8d143b3db41dc646409c3f`，共同配置为TP8/DP1/EP关闭、maxseq8、2GiB缓存池、eager、2048上下文/1792批token预算、chunked prefill、关闭prefix cache、贪心采样及固定seed。实际版本为vLLM core `0.30.0+empty`、vLLM-Ascend `0.30.0rc2`；不将core版本冒充rc2。

结果见[SDK C4/C8配置与统计](validation/aisbench-native-sdk-tp8-c4-c8-20261110/result.json)。两档共64正式请求、4热身，加2个工程guard，总70个HTTP请求正常结束；真实批次达到8，缓存owner清空，8rank均exit0，占用已释放。

C1来自旧源码`1b0101b41395e66335461feb884a5dfdc18a258b`和reference组装后端，maxseq1，使用请求私有缓存而参考为2GiB池；它与新版SDK C4/C8并非纯并发缩放。详见[C1原始统计](validation/aisbench-native-tp8-c1-20261110/result.json)、[C1对照](validation/aisbench-native-tp8-c1-20261110/comparison.json)。

## 正确性覆盖范围

- 旧TP8 reference组装路径通过冻结工程24用例24/24；这不是通用质量benchmark。
- SDK原生双请求分别输出E047、宁波，释放后第三请求再次输出E047；三请求、slot复用、无owner残留、8rank exit0见[生命周期实测](validation/native-sdk-batch-smoke-20261110/result.json)。
- SDK HTTP路径的两个并发工程guard同样正确输出E047、宁波。AISBench固定长度负载只测性能，不证明通用模型质量。
- 纯标准库[入口检查](validation/native-sdk-batch-smoke-20261110/check_http_entry_contract.py)覆盖真实Handler的输入校验和SSE结束帧，不加载/模拟模型，不作为NPU或性能证据。

## 剩余验收

1. 同条件C4/C8单轮对照已完成，每档双方32/32成功、0失败，原始时间戳、配置和数据SHA及实际设备映射已核验；详见[对照结果](validation/aisbench-native-sdk-tp8-c4-c8-20261110/comparison.json)。本轮不支持优于参考或1.5倍收益，亦不据单轮小差值宣称稳定的微小退化。
2. CPU/NPU采样已完成，rank0带剖析窗口显示主要主机提交间隔，缓存准备及采样仅约3ms/步；据此实现解码图候选并完成正式C4/C8。相同图配置的vLLM C4仅5成功27失败、生成长度不完整，CausalConv1d设备日志报非法地址；不能使用该批吞吐作为分母。下一步核查模型声明的可分段图边界与完整图捕获的兼容性，再取得有效同条件对照；不以graph/eager不匹配的比较充当收益。
3. 长上下文、正式大规模质量、生产服务范围及8物理卡/TP16不在当前已验证范围内。

原始请求、token向量、SQLite时间戳及版本清单保留在私有任务档案；不公开服务器地址、私有路径、凭据、权重或镜像二进制。框架元数据和生命周期适配未被描述为新开发CANN底层算子。
