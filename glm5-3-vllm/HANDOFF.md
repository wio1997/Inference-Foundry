# 当前交接 — 2026-10-18 SFA同步判别收敛，检查MTP纯FFN图片段

目标与授权见[MISSION](MISSION.md)、[AGENTS](AGENTS.md)。仅172.16.12.32/33，GLM5.3 W8A8C8、int8 KV/Indexer、已有算子、全SDMA、正式PD分离。可核归属后续建合理有限窗口，不停他人作业；不因阶段、失败、Ready或提交而等待用户。普通commit/push/draft PR2允许，不合并。不跨聊天发消息，不改analysis/performance-envelope-20261017/，不使用taskctl或另建记录系统。

## 当前现场与立即动作

- **14:18UTC左右更新：D已进入MTP纯FFN新窗口**。当前D CID **1bb3a19d7f6ed621b5fcf72189bbfa85c9a3ecb46753706ae5bf7cd1743a1ad2**，root=`/data/wio/glm5-3-vllm-execution/20261018-state-tail/mtp-ffn-graph-live1-20261018/node33`，build **6663eafa597ad5e62796f291590b493ac49e8bba7abfeaabdfcee722a38414a1**，GLM53_MTP_FFN_GRAPH_PROBE=1、其余helper0、marker默认absent。仅精确停止已SHA备份且空闲D05e98，确认NPU8×idle后reload；P d976与原样CPU8c8e均保持运行，holder未变。新D watchdog **3040秒**按当前P余量减180秒截限，不延长旧timer。14项CPU控制/缓冲区/回退/异常清理通过，**设备正确性/图捕获兼容性/收益尚待验**。启动watch exec23803，r/watch_mtp_ffn_graph.py，900秒从实际start计、失败仅清理本CID；stagecase exec57481。合同/源码/plan/reload receipt均r/mtp-ffn-graph-*。下述D05e98现场已转为历史，不要向旧D发请求。
- 当前P32 **d9762493f4e60c450b06d964bf6c7b9d5fbcfd5261e44086a86f2031ff7ec33f**；D33 **05e98c8425d8bb0db0c30425ae054f9e8ee7b747914a18defaa2f18d8aee8f2a**。root分别为`/data/wio/glm5-3-vllm-execution/20261018-state-tail/sfa-replay-order-live2-20261018/node32`与`node33`。同build **1575a17a8acc1d65f7da9fce1eadc0b005f6aec20383ede09836648c5ebf66bd**，P新helper0、D显式gate。P628.953秒/D645.482秒Ready。所有有限序列已完成，marker缺省native、诊断disabled；模型保持运行用于后续研究。
- 两模型硬watchdog **5400秒**，设备start约13:33/34UTC；节点时钟落后Mac/API约8–9分钟，32/33互差约50秒，测量用host elapsed。新holder均7200秒/16die、无controller，owner=codex-glm53-replay-order-live2-20261018：P名glm53-replay-order-32-live2-20261018，UID=fefe6a4e-e48f-4043-b180-e5afc0d162fb；D名glm53-replay-order-33-live2-20261018，UID=82fcb8d0-44cb-4666-8ed3-1a329f446023。不延长不可变timer；下一窗口按保守earliest creation/start减360秒。旧holder已按UID释放；首次新Pod的显式preemptionPolicy被admission拒绝，移除此字段并固定nodeName后成功，未抢占其他作业，原失败和修正保留。
- exec57386普通PD序列、82175取消检查均**rc0完成**，不要重跑。r/run_sfa_replay_order_live2_sequence.py与sfa-replay-order-live2-sequence-outcome.json。实际functional16/all16path、quality52、cached16且prefixhit递增、C4功能8、干净C1 24formal+4warmup、均衡C4 384formal+16warmup全部完成。balanced合同在测量前冻结，保持4slots×8req/block，仅每slot轮转四case、每slot相同768 output tokens，ABBA×3；不是参数扫描或最大SLO容量。
- 当前原样官方29880 CPU CID **8c8eac8d5d10bfdc993223fd0f2e4cad98091352f000656ffc8532e4e1362860**，source85b8982948530758be655d3448ff10fc3979e56622221feace560272a81f1469，父P=d976…。无NPU、watchdog为P余量减180秒。不恢复已收束29881/29882。旧live1 P=e812…/D=e202…/CPU120fe…已自然到期并SHA备份，r/sfa-replay-order-live1-{P,D}-backup.json；更旧P8df/D658也已停并备份。
- 当前动作：归约结果并快照当前模型而**不停止**，r/snapshot_sfa_replay_live2.py（P/D collector分别89709/29106）；CPU源码读取r/read_mtp_ffn_sources{,2}.py，首次已取6文件逐SHA，本轮source2 collector82788。下一假设只检查MTP纯FFN片段既有图可行性，保留attention/KV eager；尚无FFN新helper、build或设备改动。不能直接开不支持的整MTP图，也不新增算子。
- 两角色TP16/DP1/PP1、P K1、D K5、maxlen8192/maxseq4/maxbatch1024。**P实际PIECEWISE capture[256,1024]/sharedFalse，D原解码FULL图/原shared策略**，不可泛称图配置相同。image sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83；vllm ced6857…、Ascend5f828ad…。c8_enable_reshape_optim=False、mlapo_keep_prefill_weights=True、fused_mc2/dsa_cp/flashcomm1关闭。HCCL_INTER_HCCS_DISABLE=FALSE、INTRA_ROCE_ENABLE=0、INTRA_PCIE_ENABLE=1、OP_EXPANSION_MODE=AI_CPU。新P/D adoption和旧observer均0，D新observer在计时前disabled。

## 当前机制与验收边界

SFA源码update_graph_params为空，use_updatable_graph只接受AscendAttentionBackend而非AscendSFABackend；ACLGraphWrapper却对FULL回放执行通用host current_stream.synchronize。新helper仅在真实D、多输出、K5/TP16DP1PP1CP1/async/SPOff/nonhybridMLA、FULL、无device metadata executor、静态SFA、已有捕获entry、graph.tasks=[]、非ENPU/非draft/非capture时，复用原native静态回放尾部（debug地址检查、同graph.replay、同output handle、同logger），省去这一host wait。其余保持native。没有操纵ENPU flag，没有新device operator/copy/wait，也没有改精度、输入/状态/算子或图桶。

CPU26项guard/output identity/ContextVar异常清理通过，原NPU完整AST前缀与其他4overlay相同，ACL/SFA实际镜像源码SHA分别8e1c269…/9b94f67…已核。两次现场all16 cached ACL graph/静态SFA/host tasks0/eligible/selected逐请求通过。quality26×2两次均数值26/26、完整64序列26/26同；一般质量仍仅有限覆盖。合同r/sfa-replay-order-contract-v1.json，CPU/build/stage/source均同前缀。原始请求和完整IDs/text/usage保留，不能称native并发变化无害。

- live1 C1等权TTFT/last/TPOT分别×0.997989/0.997240/0.996952。原固定case/slot C4：114.052→120.036 tok/s，但两cycle分别×1.112078/0.989722，首native块异常慢，不能报稳定+5.25%；原件全留，r/sfa-replay-order-verdict-v1.json。
- live2均衡C4：native140.826→candidate142.191 tok/s，×1.009698；三cycle×0.997924/0.992483/1.039972。TTFT p95线性1496.565→1540.171ms，last p953886.226→3832.877，TPOT p9527.073→27.986；C1 TTFT/last/TPOT×1.000251/0.997741/0.996823。结论**INCONCLUSIVE、默认native、不追加此版benchmark**；不是服务联合收益。r/analyze_sfa_live2.py及sfa-replay-order-live2-verdict-v1.json。
- 取消：10条D128在首token后关闭，16条64复用/128同批peer完整、正确；P/D running/waiting/KV均回零。原脚本日志substring误把requestID内cancel计为取消，原件保持不改；去ID再单词匹配后**0条明确backend abort日志，abort counter delta0**，不能声明明确backend abort证明。指标length仅16、generation1418，但不足单独作取消内部完成链证明。修正源码+原结果复核r/sfa-replay-order-live2-cancel-correction-v1.json。用户已收到更正。

## 已完成的P填充判别

- 同题256/257交叉，P约130→218ms；原生profile证实257主干补到1024。两题数学尾部保持但无关内容/位置变化，所以先有交叉复现、后有物理shape证据。
- **整段切原生eager已否定**：26题每mode数值26/26，完整64序列18/26同；C1 24formal+4warmup，等权TTFT×1.037430、last×1.051912、meanTPOT×1.044587。C4 256formal+8warmup、短长输入64/128：118.831→110.133 output tok/s（×0.926806），全部预算/数值通过。已关闭，不再追加此版benchmark。见r/prefill-padding-verdict-v1.json。
- **图外attention裁剪是局部TTFT收益，未获联合服务接受**：保留PIECEWISE/MoE图与BatchDescriptor，仅原生attention metadata用真实n，已有zero_定义输出padding行。功能16、quality52、cache16、C4功能8均完整/数值通过；all16实参78层1024→257，长尾512/513，256/1024不变。质量完整64仅16/26同，历史native21/26、candidate15/26，严格一般质量未接受。
- 干净C1：N257 TTFT296.196→255.954ms（约-13.6%），last1504.264→1569.930ms（约+4.4%）。N256路径未变但native完整序列/TPOT自身变化，**不可把其last改善归新机制**。C4 256formal+8warmup：116.747→115.383 output tok/s（×0.988313）；TTFT p95线性1806.227→1558.975ms，last p95 4085.450→3820.067ms，但last中位2423.723→2452.523ms。默认关闭，见r/prefill-attention-verdict-v1.json。
- 普通PD物理profile已闭环：6条完整D64（2arm+N257 ABBA4），all16停止原件2383752936B，rank0导出1139197events、14561 CSV rows全部匹配P RPC窗口。native MlaProlog78×1024+MTP1×257，selected79×257；MoE Swiglu GMM仍75×8192+MTP1×2056；selected额外78个767×6144 ZerosLike，总duration约0.925ms。Prolog总duration约27.5→10.25ms、allreduceAicpu约112→82ms；这些可重叠，**不相加当wall分解**。rank0实际kernel与all16 native调用shape分开声明。
- 全profile已备份：r/prefill-attention-profile-backup.json，264587196B，SHA **38004d8bde2546e7b56787f33bfc41da158777487c49a6022d36d88ad37525f3**。首次单cat虽rc0却截断262656000B，未接受；保留坏副本，32个8MiB分块逐长/SHA及总SHA重取通过。rank0-small在B/recovery-v2/P-attention；export stderr仅CPU容器无法npu-smi info提示，trace/CSV有效。r/prefill-attention-rank0-summary-v1.json及small-files收据，不重跑设备整理。

## 继承结论、文件与工作方式

- P现有MTP1草稿交接/CPU提前发确认首token已完成有限64输出对照；quality24数值全过但一般完整序列不全同，且部分后续token/TPOT变慢，未默认接受。完整单token任务的D无用MTP与P局部完成分支已收束，收益不推广一般64/128。细节在[分析](analysis/p-first-seed-device-20261018.md)、对应verdict及此前Git/本机HANDOFF历史，不再加载全历史重跑。
- 原生C4及直接D也有完整序列变化；192份实际GPU hidden/logits与all16输入、positions、native output/API/argmax已对齐，部分hidden在LM head前已变；shared-expert overlap、无用MTP分别不是这些变化必要/唯一原因。不能宣称已定位CANN故障或普通舍入无害；也不永久阻断有边界的并发研究。
- MemCache C8 reshape挂起通过c8_enable_reshape_optim=False保留C8取得有限P→D store功能通过。后续人类已授权当前官方MooncakeConnectorV2+原样proxy全SDMA基线，不再因旧Store偏好阻断。官方Ascend Store文档与issue12533仅按原范围引用，不当GLM5.3全面适配证明。
- E=glm5-3-vllm/evidence/20261018-p-first-seed-device-v2；r=E/recovery-v2；B=/Users/wio/work/glm53-execution-evidence-backup/20261018-p-first-seed-device-v2。大raw/trace留B，Git保留源码、结构化证据及SHA收据。上一份26502B HANDOFF全文SHA备份见r/handoff-replay-order-history-backup.json；不要重放全部Run。
- 当前Git已推送head **a9c0c2e8b61f912be3cf056bbd4362a5cee6d1af**，draft [PR2](https://github.com/wio1997/Inference-Foundry/pull/2)已附；后续attention结果/新D源码及记录尚未提交。普通push用per-command HTTP/1.1已成功，无需force。PR body需在新结果齐后更新，不把待验写成收益。
- 简单CSV代码实际使用本机ZCode0.16.9，response modelId=deepseek-v4-flash-0731；外部CLI写权限被拒，主控提取最后代码提案、审阅纯本地stdlib范围后执行。详r/zcode-prefill-csv-v2/review.json；不能称CLI落盘成功、不改权限/凭据。设计、因果与验收由主控负责。
- SSH=/tmp/glm53-active-20261018.sock，root@61.241.77.34:60011；access pods32=sglang-node-access-32、33=sglang-tuning-access-33-20261013。不打印private connection配置。collect_node用host模式，**不加--npu-devices**（该旧入口硬编码过期lease）。exec_command session用write_stdin，functions.wait仅用于exec cell；不跨聊天回消息。公共网站仅HTTP/Web/API，不开浏览器。
