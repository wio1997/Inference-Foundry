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

同实例on/off/off/on共48请求，32重点请求×16rank=512份snapshot齐全。每rank真实native target75、MTP5次shared executor调用已核，开关实际生效。关闭重叠后C4仍有hidden/logits差异，C1-before/after的3/4对也不同；开启时4/4对不同。两case第一次C1-before跨模式hidden/logits完全相同。本轮否定该分支是这些差异的必要/唯一原因；不证明无害数值变化或其从不参与。

已核16rank恢复target_on75/drafter_on5、关闭诊断写盘后保留模型。有限窗口续建后，官方代理四条stream/nonstream和实验代理三模式六条均实际HTTP200、完整64 IDs/text/usage通过。新guard f24e7b80…只收紧合同；旧正式计时仍归source249ce0c…，本次不重作性能结论。见recovery-v2/shared-overlap-{summary-v1,verdict-v1,snapshot-backup}.json、live3-functional-summary.json。

下一候选核实原生max1仍做五步无后续输出的MTP，准备用现有完整skip分支进行有保护的判别与配对；尚无该候选故障或性能结论。

## SDMA 与一次修正

实际 P opt-in 导出在16rank均有 pinned CPU、同 writer/constructor stream 和 native event完成记录。profile采用本地P完成，不创建remote KV所有权；全部16rank捕获的80条D2H通过队列→ACL API connection_id→设备task逐条关联，均PCIE_DMA_SQE。节点内/跨节点KV数据路径继续用此前已核的原生全SDMA基线；本次P profile本身不替代完整PD路径验收。16rank离线导出最后rank15因CPU解析超时截断，原件保留，只重新导出该rank成功后接受完整JSON；原模型未重启。

第一次seed D请求在metadata校验前崩溃：错误访问AsyncScheduler.model_config。实际Scheduler持有vllm_config；修正为vllm_config.model_config.get_vocab_size()，补核原生构造函数实际属性后CPU10项通过。只重载D，保留P；首次失败、关闭及修复收据均保留。当前P使用初始build4ccffcc9a728的export，D使用修正build51b0ba0c1c21；二者其余四个overlay字节一致，P的adoption关闭，不执行该校验分支。

## 入口与证据

当前官方原样 proxy 在 http://172.16.12.32:29880 ，P32:29481、D33:29482，model=glm5。官方proxy不传实验私有开关，保持native control；使用原SHA85b89829…，CPU容器无NPU设备，有限watchdog早于P模型。其cold N256/257、stream/nonstream四条64-token请求HTTP200、逐token一致。

主要证据目录：`evidence/20261018-p-first-seed-device-v2/`，修正后的功能、质量与配对计时见其`recovery-v2/`。完整profile/探针源与console原件在SHA备份收据指向的本机`glm53-execution-evidence-backup/20261018-p-first-seed-device-v2/`。不把单次失败、有限实验或本次提交当作整个研究目标完成。
