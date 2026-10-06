# CURRENT_PERFORMANCE_STACK

2026-10-06目标切换为GLM-5.3 W8A8标准PD，权重`/data/tiankuan/wio/GLM-5.3-w8a8`、新服务名`glm-53`；上传中，尚无5.3加载/correctness/完整E2E证据。旧模型结果不自动迁移；Current=None，active PERF_KEEP为空。下方checkpoint为历史状态，不是5.3验收。

当前产品中正式PERF_KEEP且仍active的性能patch的唯一短索引；规则见[AGENTS](AGENTS.md)与[RECORDING](RECORDING.md)，各项验证引用[代码性能账本](CODE_PERFORMANCE_LEDGER.md)和原Run，不复制raw。

当前无active性能KEEP stack。checkpoint158记录Current=None/性能KEEP无；当时在规则commit `df7399c28791`上增量维护规则，没有性能Run、PERF_KEEP、产品代码改动或服务切换。规则commit不是产品Current的代码性能身份证明。

| Patch / Commit | 作用代码路径 | Matched baseline | 独立Gain | 当前是否启用 |
| -------------- | ------------ | ---------------- | -------- | ------------ |

表格为空，不将历史candidate、diagnostic、REJECT或功能/工程KEEP填入。以后每行只引用当前最终产品中已正式PERF_KEEP且仍active的patch/commit、路径、同口径matched baseline、独立E2E gain和启用状态/证据；未核验适用性不能假称active。

新PERF_KEEP需同时确认“旧active stack + 新patch”整体correctness与完整E2E成立；独立收益比较旧stack对旧stack+patch。累计Product Gain必须完整stack相对Stock/声明baseline重新测量，不能把+5%、+8%、+7%加成+20%。完整stack证据及产品代码commit/功能合同/workload/E2E/剩余Gap由Current权威记录引用，本页保持短。

架构变化时检查交互/回归，在新状态记录标记active / superseded / regressed及原因/证据，superseded/regressed从此active表退出；原历史KEEP裁决和源码证据不改。最终产品只计算当前active项，历史退出项保留在已有点/账本引用中，不扩展第二份历史数据库。

checkpoint159复核：H1 scoped REJECT、H2 INCONCLUSIVE/PARKED；诊断observer KEEP未进入产品。见[新Reset](records/points/GLM-OPT-0002/research/engine_commit_20261006/POST_REVIEW_RESET.md)。Current=None、active PERF_KEEP仍为空；无完整E2E对照，Gain unknown。
