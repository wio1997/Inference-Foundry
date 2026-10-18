# P 草稿交接：已实测的局部收益与边界

保留 GLM5.3 W8A8C8、int8 KV/Indexer、既有算子和全 SDMA，P K1 / D K5。P 已计算的一个 MTP1 草稿经原完成事件、16rank一致校验和控制 metadata 交给 D，在已有五个 spec slot 中占一个，由原生拒绝验证器确认。D 历史只确认 y1，computed=N。默认请求保持 native，原型须每请求显式 opt-in。

## 实际结果

- 原两输入 N256/257 的 64-token ABBA 8/8 逐 token 通过。16rank 整请求 target 调用分别13→12、14→13，首个 D wire chunk 从1个 token 变3个；这是实际调用数，不是 kernel 数。
- 24道固定算术/提取用例三模式数值答案各24/24。seed/adopt 的完整64序列24/24一致，seed 比 adopt 在11题少一次 target 调用、13题不变。seed/control 完整序列仅14/24一致，严格通用逐token门槛未通过，不作为默认优化推广。
- 错误草稿1198/1013确实进入D初始spec槽，16rank记录一致；D返回正确198/13。单token native回退6/6通过。
- C4短请求ABBA共16/16逐token通过。完整64检查先在native control即有1/4与C1参考不一致，未提交seed完整批，不能据此判定seed回归或接受C4容量。

## 固定正确性用例上的 C1 配对

两道用例在上一轮计时前已按正确性固定，本轮三模式重新确认完整序列一致，未按当前时延挑选。独立预热后每用例/模式6次，ABCCBA顺序重复3遍；36/36逐token及usage通过，cold salt、同P/D实例、同采样/64输出。D诊断写盘关闭，所有profile导出及传输已结束；P新增copy与其日志成本仍包含在测量中。

| N | control TTFT ms | seed TTFT ms | control last ms | seed last ms | control平均TPOT ms | seed平均TPOT ms |
|---|---:|---:|---:|---:|---:|---:|
|256|203.355|204.115|1356.481|1302.268|18.290|17.426|
|257|290.277|290.877|1495.720|1442.303|19.139|18.277|

等用例权重的中位数倍率：seed/control TTFT1.002904、last0.962158、平均TPOT0.953850。last约下降3.8%、平均TPOT约下降4.6%，TTFT基本持平；不能称稳定逐token设备时间改善4.6%，也不是可持续容量/SLO或一般精度验收。query-only adopt/control last1.000031仍无净收益。

## 实验代理提前发布确认首token

独立CPU实验proxy29881保留官方29880原样入口。它可提前发布P已经确认的首token，D仍按原生seed验证与实际时间推进，代理只移除D首个重复token及ASCII文本前缀，输出usage保持64。不更写引擎时间戳。

同两道固定正确性用例、6次/模式/case，control/seed/publish-seed共36条完整token ID、拼接文本及usage通过。publish-seed/control中位数等权geomean：TTFT0.706434、第二token到达0.823348、last0.963221；平均TPOT1.014819、首二token间隔1.334426。TTFT约下降29.4%、last约下降3.7%，但首token提前使平均TPOT增加约1.5%，N256第16–39位置中位到达较晚，最坏+56.564ms。该原型不逐位置支配，也不作为默认部署/一般质量或容量结论。代理仅覆盖冻结的greedy/ignoreEOS/min1、stream+token IDs合同。

## C4 的参考失配与定位

继续做native control/control16条和matched control/seed ABBA16条，全部首数值答案正确，完整64序列均可变化；严格C1参考失配不单独证明seed回归。32条请求×16rank初始请求映射、确认首token与host调度状态通过，后续host history的-1占位不等于真实GPU历史。

增加原生Top5：PD control16条及绕过P/PD的native D16条，全部1024位置/组所选token是返回Top5的argmax。两组都出现同prompt不同64序列。共同前两输出后，N256候选22/99419相对logP可从+0.375变为-0.25，N257候选98886/16可从+0.375变为0。PD传输不是出现变化的必要条件；这不能证明它不存在额外问题、变化无害或完整质量合格。

将原prompt+共同前两输出作为输入，仅预测一个确认token：C1每case4次的Top5完全稳定；并发C4时Top5/选择仍可变。单个HTTP batched API虽共享history，却因同salt prefix复用而出现computed0与128、分开调度，不能当作相同物理batch对照。随后改为每请求独立salt、每cohort两个重点历史加两个8-token filler、总输入小于1024。16重点请求×16rank均确认computed0、完整258/259预填充、无spec、graph NONE；N257仍有1/8选16、7/8选98886。

下一步只捕获这两共同历史已存在的实际输入/位置、最终hidden和logits，确定首个可观察分歧边界。尚不能把候选排序变化归因于无害舍入、具体算子或状态错误。固定镜像已有batch_invariant_ops，但官方beta文档未列GLM W8A8C8支持；未盲目启用或安装新算子。

证据见recovery-v2/{first-proxy-formal,first-proxy-performance-summary.json,c4-control-repeat,c4-matched-pairs,c4-logprob-diagnosis.json,conditional-logits-summary.json,cold-cohorts-layout.json,divergence-contract-v1.json}及其SHA备份收据。

## 实际GPU输入、hidden与logits

12个重点请求×16rank=192份有界快照完整，无probe error；实际query IDs/positions与请求逐一匹配，native完成event、原生输出/API与完整logits argmax一致。final hidden为BF16[6144]，完整gathered logits为BF16[154880]，每请求跨16rank均一致。C1每case两次的hidden/logits逐字节相同；C4 N256的2/4、N257的3/4 final hidden已改变，raw logits也改变，差异至少发生在LM head之前。本批所选ID未变，不声称它直接捕获了先前发生的argmax翻转。

部分物理B1、同258/259输入shape与位置仍变化；最大hidden绝对差0.633789、logits1.460938。随后八条C1中第一条Top5不同于此前C1，后七条恢复一致。该现象不能直接归因普通批次舍入或确定的状态故障。原件及SHA见recovery-v2/divergence-{summary-v1,actual-input-layout,post-C1-summary,snapshot-backup}.json。

下一小规模判别针对当前已启用的shared-expert overlap现有分支：同实例ABBA开/关，只选择已有执行器布尔，记录16rank真实native forward调用，并对齐同样的输入/hidden/logits。SP-only初始化选择不同custom-op，因此拒绝运行时切换。若关闭后仍变化，排除它作为这些变化的唯一必要条件；即使关闭后稳定，也只定位到分支，尚不足以声称stream race或完整质量问题已修复。

## Shared-expert分支反证与当前入口

同实例on/off/off/on共48请求，32重点请求×16rank=512份snapshot齐全。每rank真实native target shared-forward75、MTP5次shared executor调用已核，开关实际生效。关闭重叠后C4仍有hidden/logits差异，C1-before/after的3/4对也不同；开启时4/4对不同。两case第一次C1-before跨模式hidden/logits完全相同。本轮否定该分支是这些差异的必要/唯一原因；不证明无害数值变化或其从不参与。

已核16rank恢复target_on75/drafter_on5、关闭诊断写盘后保留模型。有限窗口续建后，官方代理四条stream/nonstream和实验代理三模式六条均实际HTTP200、完整64 IDs/text/usage通过。新guard f24e7b80…只收紧合同；旧正式计时仍归source249ce0c…，本次不重作性能结论。见recovery-v2/shared-overlap-{summary-v1,verdict-v1,snapshot-backup}.json、live3-functional-summary.json。

## 删除完整单token任务的无用D草稿

使用原生`_skip_drafting`，仅D/exporter0、async TP16/DP1/PP1/K5、整批max1、显式marker时改变原生fits布尔；target采样与原生skip的accepted-count/zero-placeholder处理保持。48请求、32focus×16rank=512份snapshot对齐，cold full-prefill真实target shared-forward75保持、post-target MTP5→0。跳过仍有hidden/C1前后变化，故无用MTP不是该变化必要原因，不称故障修复。两distinct SFA mapping groups共1024组检查通过，但512份均省略更多groups，不能推广全KV内容或全部层状态。

36条完整数值单token任务通过（先按正确性从旧24题预选8题，另加1536/1537长输入），长输入实际1024+512/513两chunk；max2/max64原生保护共12条三模式完整IDs/text/usage及16rank拒绝skip通过。官方PD max1实际target仍FULL、N-1 computed、6scheduled/padded和五个-1 spec占位，未假定query1；初次错误路径断言已保留，Python shared日志封顶200不能推断缺失的GPU层数。

正式使用官方原样29880、同P/D、cold独立salt、四类D诊断写盘关闭、profile/export/传输结束。C1 ABBA×3每case/mode6次、24正式+4预热全部完整ID/text/usage通过：

| N | native TTFT/last ms | skip TTFT/last ms | skip/native |
|---|---:|---:|---:|
|256|199.117|186.143|0.934843|
|257|291.664|274.739|0.941970|

等权geomean0.938400，约下降6.16%。单token TTFT=last，TPOT未定义。

另冻固定C4 closed-loop4slots、每slot连续补充16次、64条/块、ABBA×3，每mode6块/384条，768formal+16warmup全通过。Native aggregate8.65605完整答案/s、TTFT median446.179/p95602.584ms；skip9.77816/s、421.502/459.586ms，即rate+12.96%、p95下降23.73%。双方在block边界idle后换marker，pipeline启动/排空时间计入块时间。Native各块7.29–9.33/s、skip9.76–9.79/s，原件全部保留；不是只选慢块，也不称最大SLO容量或一般64-token收益。结束已恢复native，四类observer关闭、模型保持Ready用于继续研究。证据recovery-v2/unused-draft-verdict-v1.json及其列明的完整metrics、contract、路径与SHA备份。

## 完整单token任务在P结束

独立CPU29882在原官方代理副本加入显式、严格max1合同的早期分支，其他原函数/类及handler剩余AST保持。P入队前两个remote标志均false，安装的原生Mooncake调度器不建立远端发送所有权，完成即原生释放；不伪造D ACK。P原生MTP1保持，转发真实完整P响应。默认仍native，该入口不能替代需要继续生成的一般PD请求。

同一实例native PD28条、P-local28条完整数值答案通过，含8个此前预选短题和两个长输入的stream/nonstream/C4。Local期间D成功完成delta0，末两engine running/waiting/KV_usage与代理request_num均0。8次client-close后8次完整token立即复用通过；P abort delta0，不称8次backend abort。

保护回退第一条64输出与旧C1参考失配，完整失败原件保留。随后16条native/local-header fallback ABBA中，N256 native自身也有完整64变化；N257 native两次相同而一次selected fallback不同。首两ID/长度/usage均通过，但严格一般64逐序列门槛仍未通过，不能归因为代理或断言无害。只在已验证的完整max1范围研究，不把回退一般质量接受伪装成通过。合同v2在正式计时前写明这一边界。

相同CPU29882、相同完整P/D资源，原PD开启已验证D max1 skip作较强baseline，P-local不调用D。C1 cold ABBA×3共24formal+4warmup全部完整IDs/text/usage通过：

| N | PD-skip TTFT/last ms | P-local TTFT/last ms | P-local/PD-skip |
|---|---:|---:|---:|
|256|189.023|130.264|0.689142|
|257|276.427|217.557|0.787033|

等权倍率0.736463，约下降26.35%；单token TTFT=last，TPOT未定义。四类D诊断写盘关闭、P trace env0、profile/export/传输结束。

固定C4 closed-loop4slots、64条/块、ABBA×3，768formal+16warmup全部完整IDs/text/usage通过。PD-skip aggregate7.90740完整答案/s、TTFT median505.661/p95636.178ms；P-local8.88680/s、444.027/634.068ms。rate倍率1.123857、p950.996683：完成率的有限点观察增加12.39%，p95基本持平。PD各块7.23–9.73/s、P-local7.31–9.97/s，波动明显，不称稳定容量增益。不能与上一个时段官方PD-skip9.77816/s直接连乘；两轮源/入口/运行条件明确区分。结束恢复native marker，代理计数也在块边界等待归零。

## P阶段的256/257边界诊断

为区分原两题不同内容的混淆，分别给两个原算术题构造256/257交叉输入：仅增一个无关空行token或删一处无关note冒号，最后32个数学输入token不变，detokenize验证问题尾行相同。每个输入先预热，按每题256/257/257/256×3，六次/长度/题、24formal+4warmup全部完整答案通过。P-local C1中，84题129.660→217.426ms（倍率1.676899），79题130.287→217.946ms（1.672812），等权1.674854。内容和后续位置也随编辑改变，这一一致现象尚不证明具体算子或状态原因。

下一步仅用现有P原生profiler捕获实际算子形状和成本，代表rank0分析、保留all16原始来源；不把profile请求延迟当正式计时，不直接盲改chunk阈值。新入口源码、保护失败、质量、取消、计时、SHA备份与边界合同均在recovery-v2/local-complete-*、prefill-boundary-*。当前运行身份与有限窗口见HANDOFF。

## SDMA 与一次修正

实际 P opt-in 导出在16rank均有 pinned CPU、同 writer/constructor stream 和 native event完成记录。profile采用本地P完成，不创建remote KV所有权；全部16rank捕获的80条D2H通过队列→ACL API connection_id→设备task逐条关联，均PCIE_DMA_SQE。节点内/跨节点KV数据路径继续用此前已核的原生全SDMA基线；本次P profile本身不替代完整PD路径验收。16rank离线导出最后rank15因CPU解析超时截断，原件保留，只重新导出该rank成功后接受完整JSON；原模型未重启。

第一次seed D请求在metadata校验前崩溃：错误访问AsyncScheduler.model_config。实际Scheduler持有vllm_config；修正为vllm_config.model_config.get_vocab_size()，补核原生构造函数实际属性后CPU10项通过。只重载D，保留P；首次失败、关闭及修复收据均保留。当时P使用初始build4ccffcc9a728且adoption关闭；后续必要续窗已换修正build51b0ba0c1c21，D判别版本的四个非NPU overlays保持字节一致。

## 入口与证据

当前官方原样 proxy 在 http://172.16.12.32:29880 ，P32:29481、D33:29482，model=glm5。官方proxy不传实验私有开关，保持native control；使用原SHA85b89829…，CPU容器无NPU设备，有限watchdog早于P模型。其cold N256/257、stream/nonstream四条64-token请求HTTP200、逐token一致。

主要证据目录：`evidence/20261018-p-first-seed-device-v2/`，修正后的功能、质量与配对计时见其`recovery-v2/`。完整profile/探针源与console原件在SHA备份收据指向的本机`glm53-execution-evidence-backup/20261018-p-first-seed-device-v2/`。不把单次失败、有限实验或本次提交当作整个研究目标完成。
# P 图补齐的后续判别（2026-10-18）

普通 PD 的 P 协议仍是 max_tokens=1、remote_decode=True，用户输出为完整64/128；下列结果不使用 P 局部完成替代 D。两个数学题各自256/257交叉输入，完整答案全部正确，P阶段均由约130ms增至218ms。原生profile中，N257的78个主干MlaProlog输入为1024行，另一个MTP为真实257行；N256均为256行。完整请求的MoE GMM token×top8相应为8192和2048。算子duration可重叠，不能直接相加当wall分解。原件全16rank已SHA备份，rank0导出/CSV摘要及来源在recovery-v2/prefill-boundary-*。

同实例显式选择原生force_eager消除P整体补齐，all16证实257→257、长尾512/513→实际长度，而256/1024保持图。26题每mode各26条完整输出均完成数值任务，完整64序列仅18/26跨mode一致；共享salt的实际cache复用和结束KV归零均另核。严格一般序列质量未接受，原生自身变化不被认定无害。

该直接eager方案被正式服务测量否定：原样官方29880、64输出、ABBA×3的24formal+4warmup，等权TTFT倍率1.037430、last1.051912、meanTPOT1.044587。四slot、短长输入64/128混合、ABBA×2的256formal+8warmup全部完整/数值通过，output118.831→110.133tok/s（0.926806）。虽然pooledTTFT及last中位数下降，吞吐变差，不能只取改善项。native/adaptive原始块和分位保留，见recovery-v2/prefill-padding-verdict-v1.json；已恢复关闭，不作为默认优化。

下一原型保留原捕获图与图桶，只把图外attention的原生metadata长度由graph padding改为实际token，填零输出padding行以定义后续图输入。原生SFA与ACLGraph源码和镜像SHA一致，24项CPUguard/buffer检查通过；现阶段尚无设备形状/质量/性能接受。此处记录假设与边界，不晋升为通用优化结论。当前现场见HANDOFF。
# 2026-10-18 后续执行机制判别

图外P attention裁剪已完成物理取证：主干MlaProlog从78×1024变为78×257，MoE仍75×8192，额外78次767×6144清零约0.925ms。C1 N257 TTFT约−13.6%，但last约+4.4%；固定C4吞吐×0.988313，不能作为联合服务收益，默认关闭。全16 raw profile已SHA备份，rank0 CSV14561行全部对应实际P RPC窗口；重叠kernel durations不当作wall时间分解。

D静态SFA回放中省去通用host synchronize，all16真实cached graph/host tasks0路径已核；两次26题×2均数值正确、完整64序列26/26同。第一次静态slot C4 aggregate+5.25%，但两个cycle分别+11.2%/−1.0%，首native块异常慢。预先冻结的均衡slot复核384formal+16warmup全部完整/数值正确：140.826→142.191 output tok/s（+0.97%），三cycle −0.21%/−0.75%/+4.00%；C1 TTFT基本不变、last/TPOT不足0.4%。p95 TTFT1496.565→1540.171ms、last3886.226→3832.877、TPOT27.073→27.986。判定INCONCLUSIVE，保持native，不追加此版benchmark，不宣称最大SLO容量。

取消验证10次首token后关闭、16个后续/同批完整响应均通过，P/D running/waiting/KV回零。原日志parser误把requestID内cancel算作取消，原件保留；去ID再按单词匹配后明确backend abort日志0，abort counter delta0，因此撤回明确backend abort证明。见recovery-v2/sfa-replay-order-live2-cancel-correction-v1.json。

下一原型仅捕获现有MTP纯FFN，不捕获attention/KV、不打开不支持的整MTP图、不改算子/精度。14项CPU控制测试通过；新D有限窗口保持P和官方proxy，设备功能/SDMA复制路径与正式收益尚未验收。合同、构建和后续状态见HANDOFF及recovery-v2/mtp-ffn-graph-*。


## 2026-10-19 MTP opaque MoE 图回放的端到端否定结果

原生MTP继承编译配置，NN实例forward被trace away；运行时vllm.moe_forward_shared会动态调用AscendMoERunner._forward_impl，限定唯一MTP实例可捕获相同原native core，保留attention/KV/target/外层combine。CPU14项及all16普通64/128选路通过，26×2完整64数值与完整序列均一致。native类别未全局修改、既有算子/精度不变。完整源码、先前全局包装启动失败、NN实例未生效两次否定均保留。

普通PD8旧profile的16rank×50精确新增copy全为SDMA_SQE。新C4物理覆盖16rank实际N6/18/24，624条中619条有直接SDMA关联，rank0剩5条CPU/API无device关联，不把样本覆盖写成逐条全证。profile/export/transfer/observer结束后，原样官方proxy跑固定C1和均衡C4 ABBA×3。C1 TTFT/last/TPOT等权比例0.994873/0.999484/0.999870；C4 native145.875→candidate145.263 tok/s，三cycle比例1.033896/0.955904/0.999086。384条正式请求及24条C1全部预算/数值正确；C4 TPOT p95从25.481升到27.288ms。结论：这一版本不作为服务优化接受，native默认，停止重复同版benchmark。详recovery-v2/mtp-moe-graph-formal-verdict-v1.json。

旧profile中opaque core CPU提交中位1956.52→161.315us，copy样本物理0.66us，但这是带profiler的异步提交成本，不是未profile关键路径成本；不能相加或由五次draft调用推接受token。端到端未兑现，尚无可支持的target验证/host供给空隙独立归因。取消仅确认10次client-close、16完整peer/reuse及KV回零，明确backend abort日志/计数均0。

下一步先做一次原生P图桶配置[256,512,1024]对照[256,1024]，D保留且native。旧N257补1024同时放大attention和MoE，只裁attention未获联合收益；先确认通用512桶能解决多少，再选择有效token贯穿图缓冲/attention/MoE的机制，配置收益不包装为自研。该一次P重启由原生update_config不支持compilation_config所必需，旧P/代理先精确SHA备份/停止，有限新watchdog不延长。跨窗口对照明确其限制。
