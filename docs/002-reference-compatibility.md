# 已有910B适配记录的适用性

已读取用户指定[20261117复现记录](https://github.com/wio1997/Inference-Performance-Results/blob/ebfba945a0927598638e03a14fc975190659937b/results/Ascend-910B/DeepSeek-V4.1-Flash-1017-w4a8/20261117/reproduce.md)并固定仓库commit、Git blob和原文SHA。其config.json SHA与当前固定W4A8配置一致，但没有来源完整权重revision/逐张量SHA，因此不声称两套全部权重逐字节相同。

|维度|参考记录|本任务|
|---|---|---|
|镜像/依赖|vLLM0.32.0rc1、CANN9.2.1|固定0.30.0rc2、CANN9.2.0|
|硬件|910B3|910B1|
|并行|P DP2/TP1/PCP4，D DP8/TP1/PCP1，均EP|按同机对标优化；拓扑干预需明确对应关系和实际容量|
|图|target FULL_DECODE_ONLY，DSPARK draft enforce_eager=true|目标与草稿完整图均需实际验证|
|负载|16000目标输入、C4、16请求、50%重复prefix|四档精确输入、C16、首次cold32/hot32、1024输出|
|缓存|D关闭本地prefix，另有AscendStore外部池|保留原生正确性guard并验证实际外部prefix命中|

优先复用候选为P/D非对称DP/PCP划分、MultiConnector将P/D传输和外部池分离、Engram CPU offload/DP共享内存，以及模型特定tokenizer。先核对固定0.30源码/选项/张量与状态所有权，再决定所需接线或回迁；不更换固定镜像，不复制新版本二进制。参考的eager草稿、不同负载或历史缓存结果不替代当前验收。当前没有新的全模型性能数字。

- [逐项参数、来源SHA、可复用机制和不能照搬的部分](../metadata/reference_20261117_assessment.json)
- [固定0.30原生图路径与待验项](../evidence/native_graph_contract.json)
- [SGLang分支](https://github.com/wio1997/Inference-Foundry/tree/sglang-deepseek-v41flash-w4a8-910b-pd)

实际P模板显式设置`VLLM_USE_V2_MODEL_RUNNER=1`、P eager和原生`dspark`5预测，说明应优先核对当前固定镜像已有V2/DSPARK路径。模板Git blob与SHA已记录，未据此宣称固定0.30参数和PD状态协议已兼容。
