# 分页索引器实际失败与真实容量定位

G901候选未接入完整模型，完整质量/四档矩阵/跨框架地图和端到端收益仍未通过。延续固定daily A2、910B1/aarch64、CANN9.1.0/torch_npu2.10post6，P832保留，D832原容器和失败质量、短诊断均保留。以下是组件证据，不能升级为完整适配通过。

## 第一次：原生Triton finisher编译失败

[902实际终态和原始SHA索引](../evidence/paged_indexer_G921/G913_PAGED902_ACTUAL_FAILURE_JSON_parsed.json)确认stock finish_paged_indexer_topk的tl.sort在BiShengIR PlanMemory/cannot-align-axis失败，未完成一个完整case或图重放。各rank完整编译IR/traceback与日志留受控存储，公开仅保留异常首尾和原始SHA，避免大量IR挤入仓库。controller实际退出，八卡清空/owner released，没有模型部署。

G911保留同一native paged FP4 score kernel和capacity，改用原基线相同torch top-k与sort，仅以简单NPU Triton gather映射已选K个物理slot。它修复的是NPU编译边界，不改变FP4数学、排序规则或数值阈值。

## 第二次：基线TopKV2触发vector异常

[912实际组件记录](../evidence/paged_indexer_G921/G915_REVISED_PAGED_ACTUAL_JSON_parsed.json)在八rank分别通过ratio1的(1,1)、(6,65)、(16,513)、(80,4097)，FP32分数逐位相同、512个排序物理slot相同、输入未改。仅这32个实际case成立，ratio2和NPU图重放均未通过，不称整个候选有效。

下一(96,131073)形状在同步点报vector异常。[底层CANN日志](../evidence/paged_indexer_G921/G916_REVISED_PAGED_FAULT_JSON_parsed.json)明确fault kernel_name为stock TopKV2_...，不是paged FP4 kernel。旧probe没有在baseline返回后立即同步，故step字段仍显示candidate_scores；不能用该字段错误归因候选计算。新版probe在baseline score、TopKV2和candidate各阶段单独同步。此故障保留，不重试相同条件，也不放宽FP32/slot判据。组件controller已实际退出、八卡无进程、health8/8 OK和owner released。

## 真实模型容量与后续范围

[原D实际profile](../evidence/paged_indexer_G921/G917_ACTUAL_TOPK_GEOMETRY_JSON_parsed.json)中的FP4 FullTopK容量是96×263435（ratio1）和96×131717（ratio2），均512输出；131073是之前人为边界fixture，不是该服务实际图容量。不能把它的失败删除或伪装通过，也不能据此声称真实服务容量已验证。

[920单卡pilot实际启动](../evidence/paged_indexer_G921/G920_REAL_GEOMETRY_PILOT_STARTED_JSON_parsed.json)在NPU0、同不可变镜像重新定位真实两种更大容量，保留空/ragged/奇数小形状，另加单query全可见263435/131717长上下文，并要求两ratio变q/长度/请求表/packed K图重放对照。只做单卡是先降低重复设备异常的定位成本，完整八卡门仍必需；逐位分数、512个排序物理slot和输入不可变标准未变。该pilot结果此刻待审，不宣称成功或收益。

[D918准备清单](../evidence/paged_indexer_G921/sg_D_paged_preparation_g914_manifest.json)只在本地完成：同832原生argv/环境（仅profile路径更新），新增分页候选及每PID路径命中记录。score、top-k和mapper函数AST与组件源码相同。但组件未验收，明确禁止激活。待单卡与八卡完整门后，仍须新模型实际8rank装载、target/draft图、设备路径证据和完整1319题严格质量，再进行四档AISBench、最终匹配profile和端到端收益判断。
