# Performance Research Reset166 — slow-rank host supply / MoE gather

身份: glm5-3-autonomous-20261001 / e36a52b4f93cb0d4884d985b4b6a54ea8156155b / GLM-RESEARCH-RULES-v2 / rule93133edc6567a2ecaac7754b3c9aab1537ca7fe3 / GLM-OPT-0002.
产品与Current: GLM-5.3 W8A8真实P→KV→D/MTP、动态完整服务、既有TTFT/TPOT SLA。Current=None，active PERF_KEEP为空；短8-token不是完整产品/SLA验收。
最大代码Gap: 现有Run249显示逐层慢rank host提交供给迟滞；profOFF265ms的精确分解未知。H3 KV晚观测已parked；唯一active H4为MoE list-gather逐片materialization是否可减少真实Decode host critical path。
决定性证据: CRITICAL_PATH.md、全部D16原trace/CANN连接；1901同collective中1896最晚device rank也是最晚host rank，median launch-end→device24.825us；D15窗口约902ms直接hostlate边；380 gather/6080逐片copy；实际prepare_finalize/torch_npu exact commit source；独立Astra源码/diff评审。
代码假说与patch: PrepareAndFinalizeWithAll2All.finalize原list all_gather改equal/aligned shard direct all_gather_into_tensor，保留fresh output和HCCL必要通信；其他shape退原path。不改kernel/math/MTP/布局。新增rank一致shape guard和必要contiguous成本。
复用: PK047 synchronous caller-stream join、PK050 decorator/underlying identity、PK126 host issue DAG纪律为方法/边界；旧DeepSeek性能不迁移。本次新增GLM5.3 TP16现有profile真实list unpack证据，不重采profile。
最小matched A/B: 最终guarded CPU4 rank先正确；仅D一次同配置重载，P不动；diagnostic shim在相同驻留进程选择stock/candidate，原生API无可用worker RPC。mode仅全局idle边界切换，各rank首次mode witness发生在warmup。A1/B1/A2/B2同8token请求重复，均profOFF/noProxy/freshsalt/相同MTP；每mode再一个自然结束完整简单请求用于语义/E2E检查。不做大容量Run、参数扫描或新profile。诊断shim两分支相同开销，不当产品代码部署。
Astra: YES；跨框架/通信源码归因，已有pd_challenger独立完成并写ASTRA_DECODE_REVIEW及ASTRA_MOE_GATHER_DIFF_REVIEW。

Hypothesis: 删除每MoE16输出copy和host提交，能够在同进程、同负载下减少Decode延迟，同时保持allrank数值和完整输出。
Distinguishing evidence: 最终CPU exact方法；实际HCCL/model原8token黄金IDs；4个mode各warmup1、计时3相同8token；A/B相同轨迹且两次重复；每mode完整自然结束语义输出；所有错误原样保存。
Decision table:
- correctness差异/原生异常/外部KV未全命中 -> REJECT，停止后续性能，恢复stock源码和mode。
- 两B均优于两A波动范围、token/MTP轨迹相同、完整E2E重复收益 -> 有界PERF_KEEP候选；完整API/标准SLA未验收仍不提升产品Current。
- 8token gain无稳定性或完整E2E未成立 -> INCONCLUSIVE/REJECT，不升级Current；继续离线辨因剩余host dispatch，禁止将profile inclusive time当预测收益。

本阶段没有新增代码级性能 KEEP。
