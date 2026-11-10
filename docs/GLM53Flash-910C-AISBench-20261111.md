# DeepInfer × GLM-5.3-Flash × 910C：固定负载性能验收

本次输出吞吐至少1.5倍的目标已在声明范围内验证。最新原生候选g47、最终参考g49，源码`59c980223e3805df3f8c6e606ada69f16d1fff95`。两边均为8die/TP8/DP1/EP关闭，固定镜像与W8A8检查点，公共PIECEWISE配置相同；DeepInfer优化扩大KDA实际捕获覆盖。

|并发|DeepInfer token/s|新vLLM参考 token/s|新参考倍率|对两次较快参考的倍率|
|---|---:|---:|---:|---:|
|C4|107.4124|64.4161|1.6675|1.6675|
|C8|190.8478|108.1105|1.7653|1.7426|

每档32请求、1024输入/256输出、2热身排除、32成功0失败。64个完整输出与新参考一致，原始SQLite时间戳重算并与AISBench汇总一致。

|并发|首token中位ms：原生 / 参考|TPOT中位ms：原生 / 参考|
|---|---:|---:|
|C4|491.08 / 352.66|35.02 / 52.38|
|C8|742.45 / 518.30|39.11 / 55.67|

首token延迟和输出吞吐分别报告，不把吞吐倍率推广到所有延迟指标。该固定集合不替代正式通用质量评测，也不代表任意上下文/并发的性能。

## 验收与实现

DeepInfer LLMEngine → C++ scheduler → Worker → SDK计算描述 → 固定SDK NPUWorker/ModelRunner/KVCacheManager → CANN/FLA-NPU → 原生输出回调。没有创建vLLM EngineCore、请求Scheduler或Executor代理原生请求。

72个原生请求正常结束（4工程请求+4热身+64计分）；8rank均exit0、owner释放、物理占用回收。每rank34层×4批量捕获136次，重复close正常，非精确/小prefill安全回退实际命中。参考服务也完成两档计分并释放占用。

显式优化选项：`--compute-backend sdk-batch --max-sequences 8 --cache-bytes 2147483648 --max-context 2048 --max-step 1792 --graph-decode --breakable-graph --kda-decode-capture`。完整启动还需原生8rank进程组、只读模型目录与工作目录；部署命令和原始日志保留在任务证据中，不宣称任意机器一键干净复现。

## 版本与证据

- 镜像：`ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b`。
- 核心vLLM：`0.30.0+empty`；Ascend插件：`0.30.0rc2 g8fde6d4e3`。
- AISBench：`3.1.0 / 525ed1779f50a8ac068b8638771f633a151e769f`。
- 参数：BF16/W8A8，cache 2GiB/rank，context2048，token预算1792，async/prefix/offload关闭，capture[1,2,4,8]。
- 详细结果、版本、配置与原始证据哈希：[result.json](../validation/native-sdk-kda-capture-20261111/result.json)。
- 交互图解：[推理架构HTML](DeepInfer-vLLM-GLM53Flash-architecture.html)。

## 失败保留与范围

g43卷积叶子图在C8退化，已移除。g45缺捕获元数据，门禁阻止无效计分。g46吞吐已改善但重复关闭失败，未作为最终验收。g48启动参数预处理失败，未启动模型。上述记录保留；最终结论只用通过完整门禁的g47与新参考g49。
