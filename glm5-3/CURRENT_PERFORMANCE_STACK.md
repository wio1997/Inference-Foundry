# CURRENT_PERFORMANCE_STACK

checkpoint177: [MTP graph correctness and matched verdict](records/points/GLM-OPT-0002/research/decode_path_20261006/CHECKPOINT177.md). Run273 all16 replay/exact natural23 correctness PASS; Run274 cache-query driver defect INVALIDATED preserved; fixed Run275 valid A/B/A/B complete INCONCLUSIVE (D savings−2.313/−9.623ms). H11off, H6/H5/targetFULL retained, all16 healthy same workers, transition11. No new code performance KEEP. Current fullAPI/80K/600/93% open. One unchanged-mode two-request D-only frontier profile Run276 prepared/frozen, not started at this checkpoint; actual state decides.

checkpoint176: [MTP DCP metadata lifetime](records/points/GLM-OPT-0002/research/decode_path_20261006/CHECKPOINT176.md). Run272 capture passed but all16x2 replay guards fell back eager on one DCP local-table pointer; no complete/gain claim. H6/H5/mode0/all16 healthy retained. Source fix stages persistent table before dummy/real SFA builders; actual container CPU and independent final diff review passed. Sole Run273 started09:32:39Z spec47819c7c/52pins, four small correctness requests, still initializing at09:37:15Z. Matched draft terminal/cache/witness fixes CPU passed; no performance execution. No new code performance KEEP.

checkpoint175: [Actual MTP nested capture and single-outer patch](records/points/GLM-OPT-0002/research/decode_path_20261006/CHECKPOINT175.md). Run271 FAILED preserved, bounded H6/H5/FULL recovery closed. Final two-file single-outer/exact-Noop candidate passed CPU/container/independent review. Sole Run272 started09:12:10Z, specdf00a08d/54pins, four small PD requests; no device correctness/performance result yet. H6/H5 retained, H11 not in stack. Live status comes from Run272 state/active_epoch/guards. No new code performance KEEP.

checkpoint174：[真实API观察器与最小MTP图诊断](records/points/GLM-OPT-0002/research/decode_path_20261006/CHECKPOINT174.md)。Run269/270失败和各一次H6/H5/FULL恢复已closed；v4 actual context/proxy/production-run CPU及独立源码review通过。唯一Run271于08:47:04Z启动，speca7c3af23…/46pins，四固定小PD请求，尚无capture/性能结论。H6/H5保留，H11未入stack；现场以Run271 state/active_epoch/guards为准。AISBench cache隔离/GitHub同步完成。本阶段没有新增代码级性能 KEEP。

checkpoint173：H6/H5保留；Run268初始化失败后一次恢复已通过exact gold/all16/health/idle。唯一Run269验证修正版greedy K1 MTP Graph，H11未入性能stack。最新现场用Run269 state/active_epoch/guards；下方Run267等是历史。见[checkpoint173](records/points/GLM-OPT-0002/research/decode_path_20261006/CHECKPOINT173.md)。

2026-10-07按用户新裁决分开研究stack和最终产品stack。研究候选通过correctness、matched A/B/A/B及可重复完整PD E2E后保留，后续以旧stack对旧stack+patch比较；完整API/正式SLA未验收不构成撤回所有有效patch或停止研究的理由。历史正式promotion决定保留，下方历史“stack空”不能再作为当前研究策略。

| Active research patch | 代码 / 证据 | 独立 matched 完整PD收益 | 实际启用 |
| --- | --- | --- | --- |
| H6 MC2 capability cache | [两处缓存patch](records/points/GLM-OPT-0002/research/decode_path_20261006/mc2_capability.patch)、[Run256](records/points/GLM-OPT-0002/runs/GLM-RUN-0256/summary.md)、[Run258机制](records/points/GLM-OPT-0002/research/decode_path_20261006/NATIVE_BOUNDARY_DECISION.md) | 23token自然EOS，同worker/common库两对TPOT下降13.62/14.20%；Dwall下降14.36/14.09%，P drift分账 | Run267 applicability closure全16 cache/mapping/MC2 mode1，见retained_witness/guards_after |
| H5 same-stream MoE event policy | [五文件patch](records/points/GLM-OPT-0002/research/decode_path_20261006/moe_event.patch)、[H6 baseline上的Run261](records/points/GLM-OPT-0002/runs/GLM-RUN-0261/summary.md) | 完整23token D TPOT下降6.77/7.04%、D wall下降6.98/6.94%；D saving0.314704/0.313984s超过0.017733s drift，P贡献分账 | Run267 applicability closure全16 None helper、event mode1，pure KV/API修复驻留 |

Run260 H4完整比较INCONCLUSIVE/off。Run261独立raw复核POSITIVE，研究栈[H6,H5]实际保留。Run262 H8 CPU/native all16 byte正确、同工作量，但两完整对照收益不重复，独立冻结INCONCLUSIVE，H8 off（[证据](records/points/GLM-OPT-0002/runs/GLM-RUN-0262/summary.md)）。Run263同一请求rank5 KV传输失败使Graph数值归因不成立。Run264实测logical失败ID正确且全rank未执行模型，但流式API空error被guard丢弃。Run265 completed/phase0，独立raw核验canonical500及两个clean exact PD/全16动态FULL replay；[Run265结果](records/points/GLM-OPT-0002/runs/GLM-RUN-0265/diagnostic_reduced.json)确认H6/H5仍mode1/all16，pure KV/API修复驻留，Graph配置/观察器仅限诊断支持。KV/API修复不计性能收益，无Graph matched性能/配置收益宣称。[Run266](records/points/GLM-OPT-0002/runs/GLM-RUN-0266/summary.md) H9正确但完整saving均低于drift、不重复改善，INCONCLUSIVE/off，未入stack；独立terminal reconciliation completed/phase0。H10同步记账前greedy padded MTP提交源码/CPU通过，但Run267实际async=true而NOT_APPLICABLE，未启用mode1/无A/B；独立零请求closure与raw归约通过，H10off/未入stack。当前H6/H5保留，正式Current=None/PERF_KEEP表空；完整stack相对stock收益未测，百分比不相加。

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
