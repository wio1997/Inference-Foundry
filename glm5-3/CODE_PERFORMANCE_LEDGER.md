# GLM代码性能账本

从checkpoint158之后建立（2026-10-06；父commit `e378d459e08e9f12300118e158bba16db6c4d04f`），只汇总后续代码候选的证据引用，不迁移或重写历史Run、raw evidence和裁决。记录规则见[RECORDING](RECORDING.md)，研究规则见[AGENTS](AGENTS.md)。

本阶段没有新增代码级性能 KEEP。

checkpoint158记录Current=None、性能KEEP无；本次仅更新规则，没有新性能实验或Current提升。当前最大可由代码消除的Gap仍需Sol直接研究源码和决定性证据后确认，不能把旧next_action预填成答案。

## 代码候选

| Code Optimization | Baseline | Patched | Gain | Correctness | Verdict |
| ----------------- | -------- | ------- | ---- | ----------- | ------- |

当前无后续代码候选条目；表格不是已有优化结果。每行用优化点ID/简短改动描述及code/diff引用，Baseline/Patched引用同合同matched Run和完整E2E指标/单位；Gain写公式、重复结果/波动和适用范围；Correctness链接验证；Verdict链接Sol裁决。缺matched A/B或完整E2E标INCONCLUSIVE，缺证据写unknown；真实执行无效标INVALID。不能以局部/内部TPS代填完整E2E收益。

对越小越好的完整wall，Gain=`Baseline / Patched - 1`表示吞吐提升（相同有效工作量）；若报告wall缩短则为`1 - Patched / Baseline`，明确口径。对达标稳定容量/有效吞吐，Gain=`Patched / Baseline - 1`。Baseline为0或unknown时不计算百分比。SLO与功能合同必须同时满足，有限包TPS不能替代稳定服务容量。

只有同口径、correctness保持、matched A/B与真实完整E2E重复成立的Product Gain可记代码级KEEP。真实完整E2E不足不得提升Current。Full Replica / Complete-request Placement与标准PD分别标记，历史调度知识是否进产品写新的裁决引用。

## 配置与部署收益

配置/部署结果记在对应优化点/新checkpoint并链接各自Run，明确`configuration`或`deployment`类别，不填入代码KEEP列。GMU/batch/token budget/KV/HCCL buffer/端口/并行比例/MTP深度/Graph开关/实例数量本身不算代码成果；配置随patch一起变化时，先隔离代码对照，否则混合收益INCONCLUSIVE，不冒领代码Gain。

## 重要checkpoint汇报

先报新增代码KEEP与每项E2E Gain，再报当前最大剩余Gap和下一最高价值代码问题。没有新增KEEP必须逐字写“本阶段没有新增代码级性能 KEEP。”未知Gap写unknown及缺失证据。Goal Review结果引用当前优化点或新checkpoint，不修改旧next_action/裁决，也不以Run数、功能测试数或Runtime模块数包装性能进度。
