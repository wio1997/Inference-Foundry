# GLM — HANDOFF

2026-10-01持续自主执行；研究、代码、执行和raw在166服务器，Mac仅SSH入口。真实clone /data/tiankuan/wio/Inference-Foundry，分支glm5-3-autonomous-20261001。禁止恢复旧队列；唯一controller的controller-owner.json/state.json及PID/boot/startticks为现场权威。

- Current：没有正式重复KEEP，达标稳定容量/上下界unknown。Run3四条81932实际输入→61440输出均有效；TTFT(first SSE) P50 6.0588s、TPOT P50 22.9ms，未达4s/18ms。四样本不能证明P99或稳定服务。
- 测量修订：Run1/2旧parser漏delta.reasoning；raw不改，新first_output_reduction按真实首输出重算。Run1首输出34.509s/TPOT29.961ms；Run2 c1首输出2.324/2.222s、TPOT24.615/21.927ms左右；不能再引用Run1 14.625ms。Run3 first-SSE SLA失败仍成立。
- 活动点：[GLM-OPT-0002](records/points/GLM-OPT-0002/point.md)，完整请求透明router/lease/drain、native原始协议与采样状态、开放到达loadgen、可选opaque trace。未改模型或算子计算，不做精度/量化比较。35 CPU合同经真实zcode --prompt/Result/bridge通过；fixture不是KEEP。
- 功能证据：Run5、9、11、12验证mixed长度、stream/nonstream、chat/completion、合法n2(.7温度)、expected400、active drain/代际readd、独立cancel后lease/native idle。Run12新的9请求1552有效输出，PD81932→256也通过（首输出38.8902s，TPOT26.5554ms）；冷/缓存及部署差异不当代码收益。
- 原生失败：Run8两端batch16384/gmu.92本地128通过，合法D mixed触发MTP merged draft DCP recv520MiB OOM，HTTP200/SSE500/DONE无usage/finish，零有效token；Run9 P PD draft gather contiguous1.13GiB OOM；Run11 P gmu.87/batch16384启动KV1.96GiB/183021tokens，但PD后续draft DCP recv1.01GiB OOM（NPU3仅1005.91MiB free）。均REJECT具体配置/负载，不是硬件极限证书。Run11 reduction_v2保留原hash，纠正归约旧Run9标签和不成立的cold宣称。
- 当前驻留配置：P来自Run12，TP16/EP16/DCP16、DP1/PP1/PCP1，target FULL_DECODE_ONLY、MTP3/draft eager、batch4096/gmu.87；KV3.17GiB/296902tokens。D来自Run9，同Graph/MTP3、batch128/gmu.92，KV6.64GiB。现役argv/identity为准，配置入口[dual_resident_safe](configs/dual_resident_safe/README.md)。dual_graph的大budget候选已有OOM REJECT。
- Run13 driver INVALID：P81932→2048实际有效，随后event参数name冲突，D/gateway未运行；原有效P控制复用。Run14修正，新D控制81932→2048首输出8.119s/TPOT20.588ms，P旧控制1.644s/20.509ms。六动态请求9088有效输出，整轮84.131token/s；D短请求排在medium预填充后首输出6.562s。不是稳定容量/代码收益。
- Run15 A/B/A与Run16 B/A/B，共36有效请求54528新输出，无native preemption，所有lease释放、两端native idle、gateway8002退出。新增默认不变的opt-in prefill_aware：真实首输出把lease从pending phase转为decode，优先较少pending再原score；unknown非流/压缩/超限保守保留pending。两端active2时P pending0/D pending1，短请求A去D、B去P，首输出A三次6.663–6.778s、B三次0.684–0.736s，可重复局部延迟机制成立。总吞吐收益不成立：Run15 A86.287/91.063、B93.961；Run16 A96.920、B80.200/81.984。MTP acceptance漂移，缓存命中按原生counter观测；不把较快单波当KEEP。全部请求唯一X-Request-ID→body SHA→lease→首输出/释放可定位，避免相同长body哈希碰撞。
- SSE修复：有界增量观测HTTP200内native SSE500，原字节转发/不重试，触发5s fault cooldown；loadgen native_error禁止成功信用，即使后续usage/finish。29原合同含真实Run8错误每字节分割、原样转发和避故障；当前35含phase/stale-generation/malformed observer。不是永久健康监督器。
- 所有权：Run16 controller已completed 13:08:39Z。不要按文档PID操作；新执行只在锁/boot/startticks/健康idle确认后启动。本checkpoint之后的Run和现场动作看controller-owner/state，旧阶段不重放。
- 下一问题：D128的完整请求预填充约8s，medium期间阻塞短请求；P4096/.87已在PD和dynamic合法跑通。准备仅重启D至4096/.87，保持P/算子/Graph/MTP不变，真实冷长输入、capability、PD及动态请求裁决瞬时显存和延迟/KV容量权衡。可选phase策略局部QoS提升不能替代原生瓶颈研究或完整61440正式合同。

研究：[source_notes](records/points/GLM-OPT-0002/source_notes.md)。DP2TP8EP16专家跨DP/TP分片，不用checkpoint/8否定runtime fit；PP39共享topk跨stage依赖未传，38/40只是候选。MTP metadata后续draft物理padding与真实batch不一致的源级线索仍未实施，first-pass历史KV和多rank/Graph依赖须保留，部分Zcode源调查超时不冒充VALID。

连接：166 ssh -S /tmp/glm52-166.sock -o BatchMode=yes 910c-166；167原Mac socket失效，166→ssh -o BatchMode=yes root@172.16.10.167。deploy:/data/tiankuan/wio/glm52-pd/deploy；container glm52-single；P9081、D9900、stock PD proxy8000、task full-router8002仅Run内启动。内网HTTP显式绕过宿主proxy。现场HANDOFF、SSH_AND_OPERATIONS、PD_START_AND_TEST_GUIDE沿用。大raw/权重/凭据不提交Git；完整artifact_index在服务器，manifest记录路径/大小/hash。
