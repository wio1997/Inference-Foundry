# 配置机制与证据边界

本说明依据 D 当前安装版本源码和四组 N=384/C=64 单轮结果整理。它解释代码路径及可检验假设，不代替运行时事件分析，也不证明某配置最优。

## 代码路径

FlashComm1 通过 sequence-parallel 路径处理 TP token shard：调度 token 数按 TP 对齐，attention 在需要完整 token 序列时 gather Q/KV，模型执行后再汇集 hidden states。相关锚点：model_runner_v1.py 第 2829、2843 行附近；mla_v1.py 第 2062–2063 行附近。attention_v1.py 第 334 行附近记录了 padding 可能增加 dummy request 的布局处理。

shared expert DP 与 FlashComm1 是两个配置项；DP 默认值会跟随 FlashComm1，但还要求 EP 开启且 TP>1。启用后 shared-expert linear 不按 TP 切权重，而是复制权重；当 FlashComm1 token shard 与复制权重配合时，shared branch 不走 TP-sharded shared-expert 的 gather/pad-and-reduce 路径。源码还显示 shared-DP 开启时，标准 shared-expert TP all-reduce 在 ALLTOALL、MC2、FUSED_MC2 通信模式下会跳过；不能把这一点缩窄成 FlashComm1 开启且 shared-DP 关闭时的特殊 gather/reduce。锚点：ascend_config.py 第 130–134 行；linear_op.py 第 527–529 行；fused_moe.py 第 695–715、989–996 行附近。

shared-expert overlap 使用独立计算 stream，但等待点依赖执行分支。在当前 W8A8 分支，dynamic quant 可与 MoE gate（门控/路由 gate）并行：共享 stream 在 gate/up 量化矩阵乘前等待 after_routed_experts；等待 before_gmm2 事件后执行 activation/requant；down projection 量化矩阵乘前等待 before_combine；默认 stream 最后等待 shared stream。before_dispatch→part1（第 950–958 行附近）只属于 generic fallback，不能推广到 W8A8 路径。该实现提供重叠机会，不证明实际关键路径缩短。输入准备在默认 stream 上完成并记录事件，routed 返回对应事件供 shared stream 等待。锚点：fused_moe.py 第 856–914、950–963、989–1008 行附近。

fused MC2 与 shared-expert overlap 在当前配置实现中互斥：两者同时配置时，初始化逻辑会把 overlap 关闭并打印 warning。锚点：ascend_config.py 第 194–198 行附近。此代码确认配置互斥，不说明互斥的底层原因。另需区分请求配置与逐步通信路径：`enable_fused_mc2=1` 不保证每步使用 FUSED_MC2；当前 D 为 EP32，A3 选择逻辑会受 draft quant mismatch、CANN MegaMoE 选择及模型 shape 支持影响，插件可用但 shape 不支持时还可能降级。两份筛选日志未见该降级 warning，但本轮没有逐步采集通信路径。锚点：ascend_forward_context.py 第 414–430 行；ascend_config.py 第 200–207 行。

## 本轮观察

四组使用 P 预算 12,288、D MTP5、C64/N384。Baseline 为 FlashComm1/shared expert DP 关闭；候选1两项开启；候选2保留 baseline 布局并启用 shared-expert overlap、关闭 fused MC2。候选2和四格组合配置来自 sidecar；原 manifest 标签/UNASSIGNED 状态均保留并在来源清单说明。

候选1 native 稳态 TPOT p50 为 24.0095 ms（baseline 23.5695 ms，+1.87%），输出吞吐 1,910.43 tok/s（baseline 1,932.18，−1.13%）。候选2 TPOT p50 为 24.9428 ms（+5.83%），输出吞吐 1,862.78 tok/s（−3.59%）。这两项单轮筛选没有显示收益；没有重复运行，不能估计测量误差或统计显著性。

baseline、候选1、候选2 的 MTP accepted-plus-bonus 平均值分别为 4.042453、4.013442、4.012348；draft-token 接受率分别为 60.8491%、60.2688%、60.2470%。accepted-plus-bonus 观测相差约 0.7%；输入盐值、批形状和配置的影响尚未分离，因此不能把候选吞吐差异完全归因于开关。

## 解释限制

候选2同时关闭 fused MC2 并开启 overlap。若 baseline 实际使用 MC2，结果也可能包含失去 MC2 收益的影响；聚合指标没有 step/event 时序，无法把 MC2、overlap 窗口与 stream/event 成本拆开。因而不能把无收益单独归因于 overlap 窗口太短或 stream 成本。

TP=2 时，每个具体 scheduled-token count 的对齐最多补一个 token；但不能由平均每 engine 约 2.5 个 running request 推断 padding 经常发生或开销显著。均匀 MTP5 验证 q=6×request 为偶数；draft、q1 和 mixed batch 的奇偶与实际 padding 需要另有数据。本轮没有 q1 占比或逐步 padding 观测。

四格组合 native 稳态 output TPS 为 1,861.28（baseline 1,932.18，−3.67%），TPOT p50 为 23.8939 ms（baseline 23.5695 ms，+1.38%），native 六门 SLA 为 3/6。它是多个开关同时变化的单轮结果，不能归因于单个 flag。四组结果只说明本轮兼容性和观测值，不证明最优配置、q1 当前占比或实际 overlap 程度；C32/N1024 独立验证已完成，见 REPORT.md。

## 当前源码身份

C32/N1024 独立验证中，A 配置 native 稳态 TTFT P50 为 3.559s，TPOT P50 为 22.482ms，仅 TPOT P50 未通过，输出 1120.20 tok/s。较低并发下解码延迟仍超过 18ms，说明本轮不能把 SLA 问题仅归为 C64 的首 token 排队压力。但并发与 N 同时不同，不把差值当作严格匹配的配置收益，也不能据此拆分 CPU、图模式、通信、kernel 的耗时。

以下 SHA-256 和行号均针对 D 当前安装源码，不引用旧归档源码：

| 文件 | SHA-256 | 关键行 |
|---|---|---|
| ascend_config.py | adc7f68ddbb2cb2b20e5e7c310225d901c01f7edc71c1698d6f4fd01b1e134b4 | 130–134, 194–198 |
| ascend_forward_context.py | af4a6e79ffcf6803c2a2a41dd598618398f090ea3a97253642c340ea26295c9f | 414–430 |
| fused_moe.py | 7b47805d320882cd43a1a238b72c0618d49de2e4c2258cd1c9a46a1eeb2df1c6 | 695–715, 856–914, 950–963, 989–1008 |
| linear_op.py | 599f85b8dd4c13f72aa458e21ae7fa13e18548a645fd03b8a534a6b951e7ee02 | 527–529 |
| prepare_finalize.py | c467d0ed0a0d41525aa90008aafbfd26bdbb718b0e29db7de2bed9abe09cabdd | 386–387, 520–522 |
| utils.py | d56362aa59a94a25010dc967572e394f46735407ae73d51b09f9948eb89d830c | 861–898 |
| model_runner_v1.py | 31d749f3e223f98651899b3b85459d895d1237a57d20c36fe9f215a0a7fd49ed | 2829, 2843 |
| mla_v1.py | f28816c0e596446a6f9f15ca439df91cb3af655841430114cb8ae3131973b960 | 2062–2063 |
| attention_v1.py | 4ba903ee8afb23012e78870034b1280039ccf81edb41b8e16f00e6e5bbc4c076 | 334 |
