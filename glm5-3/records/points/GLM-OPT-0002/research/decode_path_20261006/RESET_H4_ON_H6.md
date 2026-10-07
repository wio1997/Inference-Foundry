# H4 on retained H6 — Performance Research Reset

2026-10-07用户明确保留已验证patch作为active research stack，最终产品Current/KEEP另验。代码base bf07ebe2，规则GLM-RESEARCH-RULES-v2及本次用户覆盖；产品GLM-5.3 W8A8标准P→external KV→D，Current=None、正式API/SLA未完成。研究baseline为Run256/258已验证H6缓存，重新启用由Run259实际worker证据确认，不能把计划写成resident。P249固定；D16仅一次正常重载，布局、MTP K1、eager、现有正确算子不变。

当前最大已定位的可删区域是层间eager host供给。Run249 D15 post-combine finalize/shared局部暴露204.649ms；其中list gather/materialization117.132ms包含6080 copy暴露63.958ms，不能相加或直接缩放成profOFF gain。H4将同一TP gather直接写入新private output，删除split/list和16个output copy；保留512B group-common guard、fallback、不改变通信次数、rank顺序、unpad、输入/输出寿命、shared输入及模型数学。源码patch moe_gather.patch；精确candidate SHA26e52eef…，此前240 CPU/320 native byte cases REUSED。新H6仅缓存MC2 capability，不改变HCCL实现；实际padding含NaN逐字节门仍在每次B warm上复验。

Hypothesis（唯一active）：在H6 cache一直mode1的16个相同D workers中，删除finalize的list输出materialization继续降低host submission与真实PD延迟。H6机制已闭环，不重开其比较。H5及attention/prepare候选仅离线研究，不并行部署。

Distinguishing evidence：Run259，normal budget20 requests=4correctness+4warm+8 timed short+4 timed naturalEOS complete。短fixture2334/8、完整fixture58prompt/23自然EOS，gold IDs/content/外部KV/终止均不变；SSE group不是engine step或correctness invariant。记录既有Scheduler cumulative draft/accepted counters；同一重建native库mode1、同一gather selector源码，同worker A1/B1/A2/B2；每phase warm与all16 byte/cache/loaded-library witness在timing之外。无新profile、observer、扫描或大Run。仅异常允许一次H6 baseline owned恢复及warm；不无限重试。

Decision table（冻结）：完整4条cumulative signature一致、短8条signature一致；两个complete pair D wall saving均大于max(|A2-A1|,|B2-B1|) D wall drift；两对complete TPOT、实际PD wall均改善；两对short median TPOT改善，且correctness/all16 witnesses/同worker身份全过 → H4加入H6 active research stack，mode1保留。其余有效结果 → H4 INCONCLUSIVE/REJECT，mode0，仅保留H6。输出/bytes/ownership失败 → INVALID，不认可性能，安全撤回H4保留H6；外部owner异常不盲杀/重载。

完整PD wall与D wall分别报告，P drift分账；独立gain不与H6相加，最终stack对stock需重测。此小fixture判研究stack，不替代完整API、正式SLA、稳定性与完整stack产品验收。下一步继续保存trace/source拆attention/prepare/shared和提交边界，不因未得Current停止。
