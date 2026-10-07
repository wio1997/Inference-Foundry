# CURRENT_PERFORMANCE_STACK

2026-10-07按用户新裁决分开研究stack和最终产品stack。研究候选通过correctness、matched A/B/A/B及可重复完整PD E2E后保留，后续以旧stack对旧stack+patch比较；完整API/正式SLA未验收不构成撤回所有有效patch或停止研究的理由。历史正式promotion决定保留，下方历史“stack空”不能再作为当前研究策略。

| Active research patch | 代码 / 证据 | 独立 matched 完整PD收益 | 实际启用 |
| --- | --- | --- | --- |
| H6 MC2 capability cache | [两处缓存patch](records/points/GLM-OPT-0002/research/decode_path_20261006/mc2_capability.patch)、[Run256](records/points/GLM-OPT-0002/runs/GLM-RUN-0256/summary.md)、[Run258机制](records/points/GLM-OPT-0002/research/decode_path_20261006/NATIVE_BOUNDARY_DECISION.md) | 23token自然EOS，同worker/common库两对TPOT下降13.62/14.20%；Dwall下降14.36/14.09%，P drift分账 | Run262 terminal全16 cache/mapping/MC2 mode1；Run263继续固定H6 |
| H5 same-stream MoE event policy | [五文件patch](records/points/GLM-OPT-0002/research/decode_path_20261006/moe_event.patch)、[H6 baseline上的Run261](records/points/GLM-OPT-0002/runs/GLM-RUN-0261/summary.md) | 完整23token D TPOT下降6.77/7.04%、D wall下降6.98/6.94%；D saving0.314704/0.313984s超过0.017733s drift，P贡献分账 | Run262 terminal全16 None helper、event mode1；Run263继续固定H5 |

Run260 H4完整比较INCONCLUSIVE/off。Run261独立raw复核POSITIVE，研究栈[H6,H5]实际保留。Run262 H8 CPU/native all16 byte正确、同工作量，但两完整对照收益不重复，独立冻结INCONCLUSIVE，H8 off（[证据](records/points/GLM-OPT-0002/runs/GLM-RUN-0262/summary.md)）。Run263由唯一controller执行固定bucket2两请求的真实动态replay支持诊断，[Reset](records/points/GLM-OPT-0002/research/decode_path_20261006/RESET_BUCKET2_REPLAY_ON_H6_H5.md)；无性能或配置收益宣称。当前H6/H5保留，正式Current=None/PERF_KEEP表空；完整stack相对stock收益未测，百分比不相加。

# 历史产品 promotion 记录

2026-10-06目标为GLM-5.3 W8A8标准PD，权重`/data/tiankuan/wio/GLM-5.3-w8a8`、新服务名`glm-53`；上传与原生PD已完成核验。[checkpoint164](records/points/GLM-OPT-0002/research/decode_path_20261006/CHECKPOINT164.md)的H5最小event patch已通过176native及两对相同23token完整PD，D generation改善4.71/3.26%；[正式promotion裁决](records/points/GLM-OPT-0002/research/decode_path_20261006/H5_DECISION.md)仍INCONCLUSIVE/PARKED，标准动态workload/SLO未验收。H4也park，二者未叠加/未加入stack。当前P249/D253健康空闲，D stock mode0/原磁盘源码恢复。旧模型结果不自动迁移；Current=None，active PERF_KEEP为空。下方旧checkpoint为历史状态。

当前产品中正式PERF_KEEP且仍active的性能patch的唯一短索引；规则见[AGENTS](AGENTS.md)与[RECORDING](RECORDING.md)，各项验证引用[代码性能账本](CODE_PERFORMANCE_LEDGER.md)和原Run，不复制raw。

当前无active性能KEEP stack。checkpoint158记录Current=None/性能KEEP无；当时在规则commit `df7399c28791`上增量维护规则，没有性能Run、PERF_KEEP、产品代码改动或服务切换。规则commit不是产品Current的代码性能身份证明。

| Patch / Commit | 作用代码路径 | Matched baseline | 独立Gain | 当前是否启用 |
| -------------- | ------------ | ---------------- | -------- | ------------ |

表格为空，不将历史candidate、diagnostic、REJECT或功能/工程KEEP填入。以后每行只引用当前最终产品中已正式PERF_KEEP且仍active的patch/commit、路径、同口径matched baseline、独立E2E gain和启用状态/证据；未核验适用性不能假称active。

新PERF_KEEP需同时确认“旧active stack + 新patch”整体correctness与完整E2E成立；独立收益比较旧stack对旧stack+patch。累计Product Gain必须完整stack相对Stock/声明baseline重新测量，不能把+5%、+8%、+7%加成+20%。完整stack证据及产品代码commit/功能合同/workload/E2E/剩余Gap由Current权威记录引用，本页保持短。

架构变化时检查交互/回归，在新状态记录标记active / superseded / regressed及原因/证据，superseded/regressed从此active表退出；原历史KEEP裁决和源码证据不改。最终产品只计算当前active项，历史退出项保留在已有点/账本引用中，不扩展第二份历史数据库。

checkpoint159复核：H1 scoped REJECT、H2 INCONCLUSIVE/PARKED；诊断observer KEEP未进入产品。见[新Reset](records/points/GLM-OPT-0002/research/engine_commit_20261006/POST_REVIEW_RESET.md)。Current=None、active PERF_KEEP仍为空；无完整E2E对照，Gain unknown。
