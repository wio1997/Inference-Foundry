# 当前适配与公平性能验证状态（2026-11-11）

**独有优化相对同等优化vLLM至少提高30%的目标尚未完成。** 旧约1.7倍比较的KDA捕获条件不同，不能作为当前目标通过。

提交52c38617aba8779f4e6e2d458fd9791b9d5c00fa已完成原生共同能力和本轮正式测试。C4：DeepInfer104.5592、参考123.7807 token/s；C8：DeepInfer203.7050、参考196.8810 token/s。64条完整输出一致、原始计时重算通过、8rank exit0并释放占用。C4落后约15.5%，C8本批领先约3.5%；不声明稳定优势或独有收益。

首次正式运行的异步请求重入缺字段错误已修复并保留失败证据。FULL/MTP及其他8die拓扑尚未终审，原生桥额外内部logprob工作量也需解释；当前不能称所有共有优化已穷尽。

详见[源码、参数、指标和证据范围](docs/shared-features-20261111.md)。

---

## 历史声明与原始计时（保留，不作为当前目标验收）

# 当前适配与性能验证状态（2026-11-11）

本次固定负载的输出吞吐目标已通过完整门禁：原生g47与新参考g49，8die/TP8/DP1/EP关闭。

|并发|DeepInfer token/s|vLLM token/s|倍率|
|---|---:|---:|---:|
|C4|107.4124|64.4161|1.6675|
|C8|190.8478|108.1105|1.7653|

每档32/0，1024入/256出，64个完整输出与参考一致。72个原生请求完成，8rank exit0、owner与占用释放，源码9文件与`59c980223e3805df3f8c6e606ada69f16d1fff95`一致。

## 架构与优化

DeepInfer原生LLMEngine、C++ scheduler、Worker和回调持有请求生命周期；完整模型计算与物理缓存复用固定SDK NPUWorker/ModelRunner/KVCacheManager与CANN/FLA-NPU。不创建vLLM引擎、请求Scheduler或Executor。

显式`--graph-decode --breakable-graph --kda-decode-capture`，保留PIECEWISE公共配置。利用SDK metadata hook将原KDA纯decode捕获进根图；精确批量1/2/4/8回放，非精确/小prefill经原force_eager路径。数学、权重、量化不改；实际图覆盖与参考不同。默认未指定优化时保留原路径。

## 范围与取舍

首token延迟与输出吞吐分别报告，未声明全部延迟指标均提高1.5倍。当前仅覆盖固定32×1024/256、C4/C8；正式通用任务质量、全部长上下文和另一机器独立重建不在本次通过声明内。7物理卡提供8活动die，不能称8物理卡验收。

## 可复核产物

- [完整实测报告](docs/GLM53Flash-910C-AISBench-20261111.md)
- [结果/配置/原始证据哈希](validation/native-sdk-kda-capture-20261111/result.json)
- [交互架构图](docs/DeepInfer-vLLM-GLM53Flash-architecture.html)

## 历史保留

g43卷积叶子图C8退化，已移除；g45元数据门禁失败；g46计分改善但关闭异常；g48启动参数错误，均不冒充最终通过。旧eager及FULL/PIECEWISE混合成绩不作为本次倍率。
