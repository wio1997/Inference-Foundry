# 171：从请求到函数、算子的性能诊断

本轮轻量记录：15.967999s，513.03输出token/s；8条请求、DP各4条、32K/1K、prefix口径与输出哈希核验通过。详细profile为17.632761s，不作生产速度。

完整性入口：[完整timeline逻辑](tool/TIMELINE.md) · [本轮覆盖账与明确缺口](COVERAGE.md)。

## 结论：应先优化哪里

详细稳态同窗：每卡平均1025.503ms，计算458.353ms，设备空档491.193ms，暴露通信75.427ms。输入准备对应空档477.800ms。

|具体函数，自身区间计一次|与输入准备空档重叠ms/卡|占输入准备空档|入口|
|---|---:|---:|---|
|AscendDSAV41CPMetadataBuilder.build|138.642|29.02%|/vllm-workspace/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_v41_cp.py:112|
|AscendDSAV41MetadataBuilder.build|118.491|24.80%|/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py:929|
|AscendDSACPMetadataBuilder._build_local_token_metadata|53.917|11.28%|/vllm-workspace/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py:1146|
|NPUModelRunner._build_attention_metadata|27.378|5.73%|/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:3449|
|_ReplicatedCacheMetadataBuilder.take_device_metadata_tasks|8.246|1.73%|/vllm-workspace/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_v41_cp.py:72|
|AscendDSAV41MetadataBuilder.build.build_c2_metadata|7.469|1.56%|/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py:1304|
|_ReplicatedCacheMetadataBuilder._build_global_metadata|6.532|1.37%|/vllm-workspace/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_v41_cp.py:87|
|GPUModelRunner._preprocess|5.036|1.05%|/vllm-workspace/vllm/vllm/v1/worker/gpu_model_runner.py:3513|
|DeviceMetadataExecutor.submit|3.944|0.83%|/vllm-workspace/vllm-ascend/vllm_ascend/worker/device_metadata.py:74|
|AscendDSAV41MetadataBuilder.build.build_smla_metadata|2.997|0.63%|/vllm-workspace/vllm-ascend/vllm_ascend/attention/dsa_v41.py:1171|

旧样本155.35ms无法在原日志恢复；本轮重新测得无cpu_op覆盖部分309.652ms，并按新增函数边界归属。两次采样绝对量分列，不能声称逐字节补回原155ms。函数自身时间仍可含Python、未标记的库调用和等待，不能等同纯Python执行。

新增FT标记边界与输入准备空档的交集为90.347ms/卡，已从具体函数自身贡献中单列扣除。该扣除不能回放不存在的无探针时序，仍不当生产收益。

设备三类活动与窗口的分类边界残差为0.530ms/卡，单列不命名成额外调度工作。

## 完整请求的调度器与worker

EngineCore结果等待与worker/设备执行并行。这一等待是等下游完成，不是独立新增的同等调度开销；线程CPU接近墙钟时也可能是在轮询。全部数字见engine-core.csv，子表可加总到各自记录到的step范围。
- EngineCore进程682：step 15587.046ms = 等结果15357.229 + 调度/更新172.025 + 提交34.557 + 其余23.234。调度单次中位0.5540ms。
- EngineCore进程697：step 15759.708ms = 等结果15526.157 + 调度/更新172.035 + 提交37.766 + 其余23.750。调度单次中位0.5533ms。

worker全请求表是16卡平均Host墙钟，可含异步设备提交与等待。包含关系不相加；不能将_model_forward的Host提交时间当整图设备执行时间。

## 设备热点：同一次新采样

|算子|任务区间ms/卡|占新稳态窗口|
|---|---:|---:|
|目标MoE路由专家 / DFC_target_top6|147.825|14.41%|
|量化矩阵乘 / QuantBatchMatmulV3|43.457|4.24%|
|矩阵乘 / MatMulV2|34.360|3.35%|
|mHC输入混合 / HcPre|33.368|3.25%|
|跨卡归约并广播 / hcom_allReduce_|32.418|3.16%|
|跨卡全收集 / hcom_allGather_|28.789|2.81%|
|稀疏注意力索引选择 / VllmAscendQuantLightningIndexerV2|27.259|2.66%|
|转置批量矩阵乘 / TransposeBatchMatMul|24.793|2.42%|
|稀疏注意力 / SparseFlashMla|22.042|2.15%|
|索引散写 / ScatterNdUpdateSk|18.445|1.80%|
|RMS归一化 / RmsNorm|12.005|1.17%|
|批量张量加法 / ForeachAddList|9.722|0.95%|
|矩阵乘V3 / MatMulV3|9.647|0.94%|
|跨卡全交换 / hcom_alltoall_|9.116|0.89%|

各算子有并行重叠，类型时长不能直接相加；Computing/Free/暴露通信按设备活动并集，通信任务区间另列。全量shape/dtype/format和图内分属见view-data.json。

## 调优价值与下一步

1. 输入/metadata：以真实CP构造和共同frame大段为首要Host候选；从最大重复计算与批量交付判断数学共享、融合与调度，净省扣掉替代工作、栅栏、存储与争用。停止单个微小Cast/where的独立研究。
2. 目标DFC：设备最大单项，拆权重读取、GMM、通信和同步；设备占比是成本上限而非可实现收益。
3. Attention/mHC/矩阵乘：按完整shape与真实读者绑定大段，考虑数学重写、布局贯穿及megakernel；通用Type不是单一开发任务。
所有新优化仍待确认。未做性能改写，不宣称吞吐已改善。两倍吞吐须在同请求交付边界将总时间减半，不能靠加和不同采样或局部比例证明。

## 测量边界与复用

本轮coarse诊断与既有真正无探针参考分列；off仍有控制文件轮询，20.163s不作无探针基线。单轮接近不证明零开销。
细函数与profiler时钟按长函数包围关系校准，16卡约36–42ppm漂移；最大所需包围残差2.012µs。FT范围成功匹配数量与未匹配数量按逐卡CSV保留。标记边界扣除是有时钟误差的诊断估算，不能重放无探针时序。见clock-alignment.csv与probe-costs.csv。
对照既有无探针批次，8组prompt SHA与完整输出SHA一致；这是本负载结果核验，不是所有分支的数值证明。见historical-output-check.json。
详细标记与profiler均会扰动Host。本轮报告是定位依据；任何收益需要正确性及真正无探针端到端复测。
API前处理的本次逐函数、跨进程自动batch关联、DFC内部、off-CPU具体原因均未完整测出，保留为明确缺口。不是整个框架已经无未知。
原始spans与16份trace保存在launch.json指向的171自有目录；源码SHA、全部标记、控制器/恢复回执、函数和算子CSV均保留。
复用入口：[通用方法与适配契约](tool/README.md)。SGLang/DeepInfer需绑定实际版本入口后验证，不照搬本框架数值。
