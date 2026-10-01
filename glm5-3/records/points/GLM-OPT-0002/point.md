# GLM-OPT-0002 — 完整请求部署与动态placement

ACTIVE；性能INCONCLUSIVE；Current未提升，达标稳定容量和全可行域上下界unknown。两台都持完整GLM模型，native合法本地执行已证实；研究完整请求部署与PD兼容及动态调度，KV/MTP/Graph/采样/提交合同由native保留，不改算子计算或精度。

实现：透明raw请求/响应router、单worker精确lease、drain/代际readd/取消回收、native HTTP及流内错误隔离、开放到达loadgen。输出计数依据usage+finish+DONE；reasoning/content均可作为真实首输出，不以SSE chunk当token。Router trace默认关闭、opaque lease/body SHA/首字节/真实首输出；显式测试header ID解决相同body哈希碰撞。PD overlay独立未替换stock8000。

Run4/6/7/10/13为测试或driver INVALID，原始证据保留：greedy n2错误成功预期；P JSON shell多括号；readiness旧日志；flag tuple key；event(name,name=)冲突。Run13已完成有效P控制单列复用，未把整Run当成功或重跑旧队列。

Run5/9/11/12完整功能子合同验证：mixed open arrivals、stream/nonstream、chat/completion、合法n2温度.7、expected400、active drain/readd、cancel后lease/native idle。Run12新9推理1552输出加PD81932→256有效，10请求1808输出。之前capability_summary旧硬编码P eager/MTP1文案以实际argv/manifest修正，不改raw。

瞬时显存：
- Run8两端Graph/MTP3/batch16384/gmu.92，D mixed在MTP merged draft DCP recv520MiB OOM；三D请求HTTP200后SSE500无usage/finish零信用。REJECT具体配置/负载。
- Run9 D恢复128后子合同有效，但P PD draft DCP gather contiguous1.13GiB OOM。
- Run11 P gmu.87/batch16384启动后KV1.96GiB，后续draft recv1.01GiB仍OOM（NPU3 free1005.91MiB）。gmu headroom单独调整不够。归约v2纠正cold/Run9标签而保留原hash和真实数据。
- Run12只P恢复batch4096/gmu.87：KV3.17GiB/296902tokens，capability及PD通过。D保留Graph/MTP3/batch128/gmu.92/KV6.64GiB。有限合法合同通过不证明完整容量。

Run14复用P13控制，新D81932→2048控制和六动态请求9088输出有效，wave84.131token/s。P控制首输出1.644s/TPOT20.509ms；D控制8.119s/20.588ms。D预填充预算128，medium期间短请求等待首输出6.562s，支持phase机制假说。

Run15 A/B/A与Run16 B/A/B，同config/source/body/sampling/arrivals，全部36请求54528新输出有效，终态idle/lease0，gateway退出；每波9088输出。opt-in prefill_aware先pending-first-output lease数，再原active/完整校准score；默认active_count不变。短请求arrival10时P active2/pending0、D active2/pending1：A去D首输出6.663–6.778s，B去P0.684–0.736s，三次各自复现。这是局部QoS机制，不是native prefill完成断言；unknown压缩/超限/非流保持pending至release，可能集中decoder。
总吞吐收益未复现：Run15 A86.287/91.063、B93.961；Run16 A96.920、B80.200/81.984。无preemption，各波native cache命中与MTP acceptance均原counter归约；不同acceptance/尾程速度不能随意归因代码。两Run INCONCLUSIVE，无KEEP；有限2048/4096输出不能替代61440正式合同。

35 CPU合同真实zcode --prompt/Result/bridge通过，PREFILL-CONTRACTS-20261001T1244Z/test.stderr5216B SHA62b420cc382e6522a40017f3cc23a1e73cd599568498188ec1f070edc0283a13。29原合同包含真实Run8 SSE错误所有byte boundaries、原样wire、避故障/zero-credit；新增phase/默认count/伪造与旧代际lease/首reasoning与tool/text/malformed observer。CPU不是性能KEEP；原failed Job不覆盖。

下一由证据决定：D4096/gmu.87原生预算与余量候选只重载D，先真实冷长输入及PD/功能/动态测量；保持P现役。达到可运行配置后恢复同口径完整61440长输出与动态可变负载的正式重复/容量研究。源码metadata padding、DP2TP8、PP等仍可选，不固定扫描步骤，不因缺全部复现项停止研究。研究条件见[source_notes](source_notes.md)，恢复入口见[HANDOFF](../../../HANDOFF.md)。
