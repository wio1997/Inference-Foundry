# 参数优化实验计划

> 后续方案更新：MTP3组合及MTP5隔离测试已完成。以下保留早期候选及当时证据边界，其扫描顺序已停止执行；当前以[证据驱动调优方案](TUNING_PLAN.md)为准，不自动回退P或继续MTP/预算扫描。

状态：2026-10-08 基于已冻结的部署、运行结果和源代码审计制定；以下候选尚未完成性能 A/B。本轮未重启服务、未发新推理请求。当前 SSH 连接不可用，不把冻结配置当作本轮实时状态。

## 目标和证据

同时记录客户请求的输入、输出 token/s，保持约 80K 输入、600 输出、相同 prefix 热度及到达模型；在既定 TTFT/TPOT SLA 下比较吞吐。原始吞吐增高但 SLA 恶化，不认定为容量优化成功。计数见 [TOKEN_THROUGHPUT.md](TOKEN_THROUGHPUT.md)。

现有配置为 P DP4×TP16/EP64、D DP16×TP2/EP32；P 已经启用大 EP、FlashComm1、prefix cache、DSA CP、sparse LI C8 和 CPU 绑定。不能把重复开启这些功能当成新方案。D 为 MTP5、FULL_DECODE_ONLY，fused MC2 请求启用，但每层实际选用的算子路径未证明。

两个问题要分别处理：C16→C96 时，P 平均 queue 从 0.007 秒增加到 1.455 秒、prefill 从 0.807 秒增加到 2.477 秒，支持高并发 P 压力增加；D 每请求 decode 仍约 13.45–14.39 秒，低并发 TPOT 已超标。没有正式窗口 NPU/带宽/功耗采样，不能断言六台跑满，也不能仅凭 stage 均值判定所有请求的关键路径。

## 第一批：同拓扑、可归因的参数实验

| 顺序 | 参数与对照 | 要验证的机制 | 判据与约束 |
|---|---|---|---|
| D1 | D `num_speculative_tokens`：5→3→1→关闭；必要时补 2 | 少算草稿可能降低单步成本，更多草稿可能提高每步有效输出，寻找两者平衡 | MTP5 token 接受率约 62.4–63.0%，不足以推断最佳档。以客户输入/输出 TPS、TPOT/TTFT及每步有效 token 和 draft/verify 耗时共同判断。关闭时移除 speculative 配置，不填写非法的 0 草稿参数；验证输出与 KV 正确性 |
| P1 | P `max_num_batched_tokens`：8192→12288→16384；保持 `max_num_seqs=8` | 增加一轮可调度 token 容量，改善多请求拼批或切块 | 本负载 P 每请求剩余计算仅 5,644 token，8192 已可能容纳一个请求尾部；不是把预算翻倍就把单请求耗时减半。先核有效 chunked-prefill 配置，记录实际 scheduled tokens、P queue、TTFT P90/P99及内存 |
| P2 | 单独测 P `max_num_seqs`：8→16；预算仍 8192 | 判断请求槽位而非 token 预算是否限制调度 | 只有 engine 活动请求经常顶到 8、且存在可调度余量时才有收益。该参数是每个调度 engine 的上限，不是服务总并发。完成独立分支后再组合胜出的预算与 seq |
| D2 | D FlashComm1 off→on，保留同一 MTP 档 | 减少 TP 重复计算或相关通信开销 | D 审计为默认关闭且无启动覆盖；启动前再确认实际值和 GLM/w8a8 路径，证明功能实际生效。P 已开启，不重复测试“打开 P FlashComm” |
| D3 | 保持 MTP 档，核实实际 graph capture sizes；仅补真实 batch 分布需要的 bucket | 降低未覆盖形状或过量 padding 的成本 | capture size 计总 scheduled/padded token batch，不是 80K 上下文长度。MTP5 均匀 decode 的 q=6。实际 resolved list 未冻结，不能声称当前上限已证实，更不能盲目拉到上下文长度 |

24576 只在 P 实际 token batch 经常顶到 16384、且 P queue 主导时作为后续预算；D 的 4096 token 预算及每 engine 32 个请求上限没有被现有证据证明触顶，暂不盲增。

每档先做正确性和有限 matched 负载筛选，再对胜出配置跑较长 C16/C32/C64/C96；小样本筛选不替代最终 SLA 验收。MTP 档位改变会改变合法图形状，要记录由该参数引起的 graph config 差异，不能为了“保持图不变”强制使用错误 descriptor。

## 第二批：依赖明确后再做的实验

| 方向 | 候选 | 何时值得做／注意事项 |
|---|---|---|
| P 移除 MTP | P MTP1→不配置 MTP，D 固定 | Prefill producer 未必需要草稿；可能减少模型/工作区成本。必须先验首 token、KV token accounting、跨端接续与输出正确性，不能直接称其改善 D TPOT |
| P prefill MC2 | 保持预算 8192，独立试 `enable_prefill_mc2` | 先确认安装版、EP64、量化和动态容量支持。当前官方建议预算不超过 TP×512，TP16 对应 8192；不与首次 16384 预算实验同时开启。普通 MC2 与 fused MC2 不同，现有 EP64 不能照搬 EP≤32 的 fused 分支限制或方案 |
| Shared expert | `enable_shared_expert_dp`；多流 overlap 另做 | P 实际 shared-DP effective 值尚未冻结；不能因 FlashComm 开启就推断它开启。复制专家权重会改变 HBM 和计算分布。D shared-expert overlap 与当前 fused MC2 互斥，须作为“关 fused＋开 overlap”的替代路径对照，不能视为免费叠加 |
| D 调度均衡 | `enable_balance_scheduling` 候选 | 先看每 DP engine 的队列/运行请求/调度 token 分布。两台 D 请求数接近不证明 16 个 engine 内部均衡；没有不均衡证据就不排在 MTP 前面 |
| MoE 通信 | 默认算法与 hierarchy / fullmesh_v2 等兼容候选分别对照 | 先确认实际 MC2/mega-MoE/ALLTOALL 选择和 dispatch/combine 占比。hierarchy 可能引入同步与 padding；384 物理域不自动证明此算法更快。P、D 分开调 |
| CPU 调度与亲和 | 核对线程、IRQ、NUMA、host runqueue，随后小范围调整 | P 已有绑定完成日志；D 实际线程亲和未冻结。只在 host dispatch gap/CPU 争用证据支持时调整，不能直接解绑或凭空加 OMP 线程 |
| 内存与 context | 核对 KV blocks、峰值 workspace、真实最长输入合同 | P util 0.93 曾启动失败、0.90 成功，先维持 0.90；它不是 NPU 计算利用率。147456 context 不能为刷分擅改成仅支持 80K；若产品合同允许较短上限，再独立测其内存收益 |

## 图回退：独立代码修复，不冒充参数优化

此前 2P2D 诊断证明新请求真实 q=1 与 MTP5 均匀 q=6 descriptor 条件不匹配，可以触发全局 NONE；该证据不等于已经量化其在所有 4P2D run 中的占比。已做 FULL 参数探针仍被后端解析为 FULL_DECODE_ONLY，q=1 仍落到 NONE。单纯增加 capture size、写 FULL 或把 q=1 填充成 q=6，不构成已验证的安全解决方案。

需要独立实现/验证 q=1 capture/dispatch 支持，检查 attention、KV、MTP、跨 DP 同步语义，再做同负载修复前后对照；不能将旧诊断估算的收益直接减去新基线 TPOT。详见 [独立问题](../../issues/pd-q1-global-none/README.md)。

## 第三批：结构实验

同拓扑候选仍不能满足目标时，分别评估：

- P DP4×TP16→DP8×TP8，保持 EP64 与四台 P。增加独立请求组可能缓解 P 调度，但单请求 prefill、非专家权重 HBM、跨 DP 同步可能变差。需要 8 路服务/路由、connector metadata、KV 映射、CPU 与 warmup 覆盖重验。
- D DP16×TP2→DP8×TP4，保持 EP32 与两台 D。更多 TP 可能降低单请求计算时间，也会增加 TP 通信并减少 DP 请求组；只有 profile 支持单请求算力/带宽限制时优先考虑。

两侧拓扑不在同轮一起改；每个方案重新验证跨 TP KV 传输与全部 SLA。P0/P1 与新增 P2/P3 驱动版本不同，先记录并在维护窗口统一兼容环境；现有证据没有证明驱动差异造成性能问题。

## 实验记录与停止条件

每次保留配置 diff、版本/镜像/源码哈希、输入哈希及与历史输入的重叠检查、每节点缓存 counter 前后值、逐请求 usage/耗时、实际 HTTP 与 decode 并发、每 DP running/waiting、NPU 利用率/功耗/HBM和通信采样。warmup 不进入计时窗口，但需单独记录。不能重复小输入集把命中率刷到接近 100%。正式窗口按相同时钟标记采样，不能用启动或空闲期数值代替。

吞吐分输入与输出，另列缓存复用及 P 剩余计算；全程与原生稳态并报。稳态需满足窗口长度条件并延长测量、重复验证；当前 C96 稳态窗口不满足该条件。正确性失败、OOM、命中率不匹配、SLA/尾延迟恶化，均不能按成功优化合入。没有证据支持的开关停止扩展，避免穷举所有组合。

## 来源与适用边界

主要依据是本目录已发布的配置、性能、缓存及 q1 诊断摘要和私有冻结源审计。当前官方文档仅作候选兼容性参考：

- [Ascend additional config](https://docs.vllm.ai/projects/ascend/en/main/user_guide/configuration/additional_config.html)：prefill MC2 的容量建议和功能依赖；main 文档不替代安装版校验。
- [GLM5.3 官方参考](https://docs.vllm.ai/projects/ascend/en/v0.27.1rc1/tutorials/models/GLM5.3.html)：明确其参考验证范围为 colocated，PD 尚未验证；不能整套搬用到本测试。

私有源码、逐步 telemetry 不进入公开仓库；本文件仅公开聚合结论与实验方法。
