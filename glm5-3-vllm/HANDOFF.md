# 当前交接 — 2026-10-18 原生hidden差异已取证，准备shared-expert同实例判别

目标/授权见[MISSION](MISSION.md)、[AGENTS](AGENTS.md)：GLM5.3 W8A8C8/int8 KV/Indexer、既有算子、全SDMA，仅172.16.12.32/33。可核归属后续建本任务合理有限窗口，不停他人作业，保留精确CID/watchdog。官方MooncakeConnectorV2+原样官方proxy已授权；MemCache旧偏好不阻断。阶段/失败/备份/入口恢复均不是任务完成，不因此等待新授权。不动analysis/performance-envelope-20261017/，不跨聊天发消息；普通commit/push/draft PR2更新允许，不合并。

## 当前现场

- P32新CID **4cb8a338da590f0d0fd624892b2af7a20884029c7418c05ada75f5dbf3bcb157**，root=/data/wio/glm5-3-vllm-execution/20261018-state-tail/seed-live3-20261018/node32，build51b0ba0c1c21，635秒已Ready，watch r/P-live3-startup-watch.jsonl。新holder glm53-seed-32-live3-20261018，UID=1f257348-8296-4fa5-8bc9-1791e1553b29，owner=codex-glm53-seed-live3-20261018。各16die，holder21600秒，模型14400秒，ready watchdog900秒。
- D33新CID **8217b0b16541016e558c985045f10a2b146e7646233f5ee869d62d9eb7ea0dfa**，root=/data/wio/glm5-3-vllm-execution/20261018-state-tail/shared-overlap-live1-20261018/node33，build47e8c3231076，648秒已Ready，watch r/shared-overlap-startup-watch.jsonl。复用holder glm53-divergence-33-live1-20261018，UID=e43c003b-ee2a-4f8b-a551-302c6017066a，owner=codex-glm53-divergence-20261018，16die/21600秒，新模型14400秒，ready watchdog900秒。当前原生overlap已还原on，tensor/policy/host trace均disabled，模型继续保留。
- P K1/D K5、TP16各16die、8K/maxseq4/maxbatch1024、原graph/overlap默认配置；HCCL_INTER_HCCS_DISABLE=FALSE、INTRA_ROCE_ENABLE=0、INTRA_PCIE_ENABLE=1、OP_EXPANSION_MODE=AI_CPU。固定image sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83。
- 原29880/29881代理与旧P均已保存并精确停止：旧P7200秒watchdog临近，卡空闲/UID条件释放旧holder后续建上述窗口。K8s拒绝增大activeDeadlineSeconds，失败原件r/divergence-renew-attempt.json；不用timer注入等手段绕过。D probe旧CID75831d…也已保存停止，holder保留用于新D。旧原件见r/{old-proxies-stop,P-window-stop,D-before-overlap-stop}-backup.json。新CPU代理已恢复：官方29880 CID=19dffcd43b09fe06e6c2e928d86341afd5f16332964d92fbd5a8b261808da4e9（source85b89829…原样），实验29881 CID=94e9c421d3cba926beb48ec927e01711d33cd08efa3ed10c4200a9afffd3bdf7（收紧guard sourcef24e7b80…）。**官方4条stream/nonstream及实验6条三模式请求均HTTP200、完整64 IDs/text/usage通过**，见r/live3-functional-summary.json；这是新窗口功能复核，不是新计时/容量结论。首次smoke因脚本写固定旧目录已存在，在发HTTP前退出，保留原件，改显式新输出目录后通过。
- SSH /tmp/glm53-active-20261018.sock；root@61.241.77.34:60011；pod32=sglang-node-access-32、33=sglang-tuning-access-33-20261013。私有配置仅用于恢复，不打印凭据。Mac/节点时钟不一致，使用各host elapsed，holder fresh lease取较早creation/start保守核余量。

## 已有证据与边界

证据e=evidence/20261018-p-first-seed-device-v2，r=e/recovery-v2；[分析](analysis/p-first-seed-device-20261018.md)。大原件及trace位于本机glm53-execution-evidence-backup/20261018-p-first-seed-device-v2，SHA备份收据在e/r。旧HANDOFF全文私有备份见r/handoff-history-backup.json。不重跑已验证证据，不新建另一套记录系统。

- P现有MTP1草稿pinned D2H排在原writer/Async依赖前，复用原event，无新增全局wait/clone/草稿算子。原spec_token_ids经16rank请求ID/确认token/草稿一致校验传metadata。D history仅确认y1、computed=N，未验证y2入原五槽首个，其余四个原占位，由原拒绝验证器确认；全局+每请求显式gate。单token/unsupported native回退。
- 原N256/257、64输出ABBA8/8逐token通过；16rank target调用13→12、14→13；wrong seed1198/1013实际入槽并返回正确198/13；max1回退6/6。quality24三模式首数值答案各24/24，seed/adopt完整64序列24/24相同，11题少一次target、13题不变；seed/control完整64仅14/24，严格一般逐token质量门槛未通过，不默认启用。八次client-close后八次立即复用通过，但abort delta0，不宣称八次backend abort/P完整清理证明。
- 固定正确性预选sum-three-4(N256,84)/sum-three-1(N257,79)，同实例cold C1、64输出、ABCCBA×3、每mode/case6次，36/36完整IDs/usage通过。D trace关闭、profile/导出/传输结束，P copy/log成本计入。seed/control中位数等权geomean TTFT1.002904、last0.962158、平均TPOT0.953850；query-only last1.000031无净收益。有限case，不是稳态单token设备时间或完整容量/SLO验收。
- 独立CPU代理publish-seed只提前发送P已确认ASCII首token，D仍原生seed验证/计时，代理去除D重复首token/text，usage64。smoke6与正式36 IDs/text/usage通过；publish/control TTFT0.706434、第二token0.823348、last0.963221，但平均TPOT1.014819、首二间隔1.334426。N256第16–39位置中位更晚，最大+56.564ms；不逐token支配，不默认或一般容量接受。正式source249ce0c…保存在旧proxy备份；当前源码f24e7b80…仅收紧参数合同（36保留请求通过、7种越界语义拒绝），新guard已部署并通过新窗口六条实际ID/text/usage复核（新计时未重复），不替换旧计时身份。
- 全16rank P profile80条新D2H均queue→ACL connection_id→PCIE_DMA_SQE关联，rank15离线CPU导出超时截断后仅重导rank15，全部有效；原模型未重启。是P本地完成profile，不代替完整PD路径；其余原生SDMA拓扑证据保留。见e/P-profile-all16-{repaired,backup.json}。
- C4 native control重复16与matched ABBA control/seed16均有64序列变化，首数值答案正确。32请求×16rank初始映射/y1/host调度状态通过；后续host -1占位不是实际GPU history。原生PD Top5对照16及完全绕过P/PD的D native16请求均有变化，全部1024位置/组的输出属于Top5 argmax。PD传输不是变化必要条件，不证明无其他故障或变化无害。严格C1参考失配不永久阻断并发研究。
- 同prompt+共同前两输出、native D max1：C1四次/case Top5稳定，C4可改变排序/选择。共享salt的单HTTP batch实际computed0/128并分开调度，保留反例，不当同物理batch证据。后续独立cold salt/两个focus+两个8token filler、总输入<1024：16focus×16rank均computed0、完整258/259预填充、无spec、graph NONE，N257仍1/8选16。见r/{conditional-logits-summary,cold-cohorts-layout}.json。
- **实际GPU边界已完成**：12focus×16rank=192份snapshot完整、无probe error；输入query IDs/positions与请求全部匹配，原生copy event及实际输出/API/全logit argmax全部对齐。final hidden BF16[6144]、完整gathered logits BF16[154880]，每请求16rank各自一致；C1两次/case hidden/logits相同。C4中N256 2/4、N257 3/4 final hidden已不同，raw logits也不同，故差异至少在LM head之前，不能仅归sampler/代理。部分实际B1、相同输入shape258/259仍变，最大hidden abs0.633789、logits1.460938；本轮输出ID未改变。C4后八条C1中第一条Top5不同，后七条恢复此前相同。不能草率称普通批次舍入或已锁定故障。见r/divergence-{summary-v1,actual-input-layout,post-C1-summary}.json、snapshot-backup.json；原件B/recovery-v2/D-divergence/artifacts/divergence-probe。
- 初次D错误访问AsyncScheduler.model_config导致崩溃，已改vllm_config.model_config.get_vocab_size()、补核实际构造属性CPU10项；失败保留。旧P4ccff版本adoption关闭，未执行错误分支；本次必要P续窗已换同修复版51b0。固定镜像已有batch_invariant_ops，未盲开新flag/装新算子；官方beta未给GLM W8A8C8保证。

## 当前判别 / 下一动作

shared-expert判别已结束：r/shared-overlap-contract-v1.json，summary-v1.json及verdict-v1.json。48请求/32focus/512份snapshot齐全，无probe error；16rank均实际target75、drafter5个native shared forward，on/off均生效。Off仍有C4 hidden变化，C1-before/after 3/4对不同；On4/4对不同。两case首次C1-before跨on/off hidden/logits相同。**否定该现有分支是这些变化的必要/唯一原因**，不证明它从不参与或变化无害，也不泛化为完整质量结论。SP-only已在源码/实际inventory排除。全部原件SHA在r/shared-overlap-snapshot-backup.json，B/recovery-v2/D-shared-overlap。末次16rank原生target_on75/drafter_on5、off marker不存在已核，然后禁用三类observer写盘并保留模型，见r/shared-overlap-restore。

下一判别：max_tokens=1的D仍实际运行5步MTP（上述native计数已证实），无第二个用户输出。这是可删除的候选冗余工作，也是假设中的post-sample状态影响来源，**尚未证明是故障原因或收益**。先核原生_input_fits_in_drafter/_skip_drafting：固定镜像已有保留sample/DP对齐/zero placeholders的完整skip分支。拟仅在D、DP1/PP1、整批sampling.max_tokens==1且显式实验gate时选择该原生分支；P已用草稿交接不能误跳过。原target采样在该决策之前，精度与算子实现不变；max2/64必须原生回退并实测。当前尚未修改/加载此候选。

复用当前模型先完成源码与合同；若为挂载所需helper必须重载，只保存/停止精确D CID、核空闲后在当前有限holder内刷新期限，不动P，不做timer/RPC注入绕过。目标是同实例native/skip ABBA，并追加实际SFA seq_lens/cum_query_lens/block_table/slot_mapping快照，区分无用草稿带来的后续状态与target attention mapping。先有限正确性/真实调用/状态判别，再做关闭诊断写盘的max1端到端配对；按结果继续，不因本次保存或提交等待用户。

PR2 https://github.com/wio1997/Inference-Foundry/pull/2 保持draft；源码/证据按普通commit/push保存在当前codex/glm5-3-vllm-optimization分支，提交身份以Git HEAD及PR head为准，不自动合并。阶段提交不等于任务完成，继续上述判别与性能工作。
