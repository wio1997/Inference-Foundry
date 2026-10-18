# 当前交接 — 2026-10-18 max1无用草稿与P完整结束路径实测，继续研究

目标/授权见[MISSION](MISSION.md)、[AGENTS](AGENTS.md)：GLM5.3 W8A8C8/int8 KV/Indexer、既有算子、全SDMA，仅172.16.12.32/33。可核归属后续建本任务合理有限窗口，不停他人作业，保留精确CID/watchdog。官方MooncakeConnectorV2+原样官方proxy已授权；MemCache旧偏好不阻断。阶段/失败/备份/入口恢复均不是任务完成，不因此等待新授权。不动analysis/performance-envelope-20261017/，不跨聊天发消息；普通commit/push/draft PR2更新允许，不合并。

## 当前现场

- P32新CID **4cb8a338da590f0d0fd624892b2af7a20884029c7418c05ada75f5dbf3bcb157**，root=/data/wio/glm5-3-vllm-execution/20261018-state-tail/seed-live3-20261018/node32，build51b0ba0c1c21，635秒已Ready，watch r/P-live3-startup-watch.jsonl。新holder glm53-seed-32-live3-20261018，UID=1f257348-8296-4fa5-8bc9-1791e1553b29，owner=codex-glm53-seed-live3-20261018。各16die，holder21600秒，模型14400秒，ready watchdog900秒。
- D33当前CID **6584fed1fd2712610c6840a6cc631914e6e4549958055a5b2175f22455c7426a**，root=/data/wio/glm5-3-vllm-execution/20261018-state-tail/unused-draft-live1-20261018/node33，build608c588c7b82，644.184秒已Ready，watch r/unused-draft-startup-watch.jsonl。复用holder glm53-divergence-33-live1-20261018，UID=e43c003b-ee2a-4f8b-a551-302c6017066a，owner=codex-glm53-divergence-20261018，16die/21600秒，新模型14400秒，ready watchdog900秒。上个shared-overlap CID8217b0…648秒Ready、全部判别/入口复核后，已SHA备份并精确停止；见r/D-before-unused-stop-backup.json。P仍保持Ready。
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

shared-expert判别已结束：r/shared-overlap-contract-v1.json，summary-v1.json及verdict-v1.json。48请求/32focus/512份snapshot齐全，无probe error；16rank均实际target shared-forward75、drafter5个native shared forward，on/off均生效。Off仍有C4 hidden变化，C1-before/after 3/4对不同；On4/4对不同。两case首次C1-before跨on/off hidden/logits相同。**否定该现有分支是这些变化的必要/唯一原因**，不证明它从不参与或变化无害，也不泛化为完整质量结论。SP-only已在源码/实际inventory排除。全部原件SHA在r/shared-overlap-snapshot-backup.json，B/recovery-v2/D-shared-overlap。末次16rank原生target_on75/drafter_on5、off marker不存在已核，然后禁用三类observer写盘并保留模型，见r/shared-overlap-restore。

当前判别已完成：max_tokens=1时使用原生`_skip_drafting`删除D无后续用户输出的MTP。保持D/exporter0、async、TP16/DP1/PP1/K5、整批max1与显式marker gate；P未改。48请求/32focus×16rank=512份snapshot齐全，实际冷full-prefill target shared-forward75保持、MTP5→0；输入/位置/native输出/API对齐，无probe error。跳过后hidden与C1-before/after仍变化，N257 skip仍1/8选择16；**否定无用MTP是这些变化的必要原因，不称故障修复**。两组SFA实际元数据512×2=1024组长度/slot→block/CPU镜像核对通过；512份均有超过两distinct groups而省略，不能扩大为全KV/全层一致。见r/unused-{draft-summary-v1,sfa-map-summary-v1}.json，SHA原件见unused-draft-snapshot-backup.json。

完整单token任务36/36通过（从此前24题按首token覆盖完整数值预选8题，另加1536/1537长输入，非全部24题）；长输入实际0/1024再1024/512或513、16rank均核。候选marker存在时，max2/max64各6条PD control/seed/publish-seed共12/12完整IDs/text/usage通过，16rank全部decline skip。见r/unused-functional-summary-v1.json及SHA备份。diag-active max64当次target22/24与旧clean13/14条件不同，r/unused-guard-max64-work-derived.json仅记录实际，不当干净性能回归/收益。

官方原样proxy的max1单token完整答案4/4先通过；all16实际decision与原生skip均只在skip模式执行。目标实际仍FULL graph、N-1 computed、6 scheduled/padded、5个-1 spec slots。第一次核对误设query1并要求已封顶200的Python shared counter，失败保留r/unused-PD-max1-path-proof；原件v2/v3及修正summary-v3.json证明实际路径，**不能从缺失Python日志推断PD GPU层/MTP计数**。合同v2在计时前已修正。

正式C1经原样官方29880、同P/D、cold、ABBA×3、每case/mode6次，24formal+4warmup全部完整ID/text/usage通过。四类D诊断写盘关闭、profile/export/传输均无、有限窗口余量已核。N256 TTFT/last中位199.117→186.143ms，N257 291.664→274.739ms；skip/native等权geomean0.938400（约-6.16%），HTTP finish0.938517。单token TPOT未定义，不能推广为64输出/一般容量收益。r/unused-PD-max1-performance。当前marker已恢复absent/native；4类诊断关闭、shared on，P/D均Ready。

C4已完成：官方PD预检8/8完整答案且128条request-rank join中原生skip按mode实际执行，host target每请求一次、每请求6scheduled、实际padded6或12。随后同实例固定closed-loop4slots×16次/块、ABBA×3，768formal+16warmup全部IDs/text/usage通过。Native384条/6块，aggregate8.65605完整答案/s、TTFT median446.179/p95602.584ms；skip384条/6块，9.77816/s、421.502/459.586ms；倍率rate1.129633（约+13.0%）、median0.944692、p950.762692。各块原件保留，native块rate7.29–9.33，skip9.76–9.79，不能只挑native慢块或据此称最大SLO容量。四trace关闭、无profile/export/传输，双方block间idle后换marker，结束已恢复native。合同r/unused-draft-C4-contract-v1.json、unused-PD-C4-{functional,paths,path-summary-v1.json,performance}。只是固定有限单token负载点，不推广一般64输出/最大容量。

完整单token任务的P局部完成：两个remote标志false时，安装的Ascend MooncakeConnectorScheduler原生不加入远端追踪，request_finished返回False,None，不创建_req_need_send/延迟释放/远端metadata。源sha4e476a50…及逻辑条件见r/native-local-completion-source。P原生MTP1仍执行、模型不改，不把未验证proposal当确认输出。

独立CPU29882从原官方proxy sha85b89829…构建，原函数/类AST保持，仅handler前两语句guard+新增helper，source=b28a9868185e76fd919fa93f71d77347c8292bdabfd41b297e3ce9fc2fbc1373。CID=cfa1eade67ed5dda23ab25e599a5f6c86dfcc2f0b179770886f812ce7b1a7579，root=P32root/local-complete-proxy-v1；启动时watchdog7540秒、早于P180秒，无NPU设备。官方29880/实验29881仍原样。必须X-GLM53-Route:p-complete且greedy/min1/ignoreEOS/max1/独立salt/IDs与长度256/257/1536/1537合同；其他走原生。暂不默认。

新路径完整答案功能56/56（native PD28+local P28，8预选短题+2长1536/1537，stream/nonstream+C4）通过；local期间D成功请求delta0，末两engine running/waiting/KV_usage与proxy request_num均0。8次body-sent/response-headers客户端close后8次完整token立即复用通过，P abort delta0、D success delta0，末资源计数归0；不称8次实际backend abort。见r/local-complete-{functional-partial-summary-v1.json,cancel}。

**保护回退的旧参考失配仍保留**：功能原件completed=false，因第一条64回退与旧C1参考不同而停止。随后同实例native/local-header fallback ABBA共16条均HTTP200、首两ID/长度/usage通过；N256/64 native自身两次也不同，N257/64 native两次相同而一次selected fallback不同。完整64严格逐序列接受未通过，不能把变化归因于新guard或宣称无害。原生fallback body不变、原handler AST保持、max2/64无local header、实际仍调用D；不据此泛化一般模型质量。合同v2在计时前明确：只测已验证的完整max1任务，研究入口不默认，原件不覆盖或伪装通过。见r/local-complete-{functional,protected-matched,contract-v2.json,build-manifest-v1.json,guard-check-v1.json}。

新C1按合同v2：同CPU29882，PD原生路径启用已验证D max1 skip作较强baseline；P-local两模式均保留同一完整P/D资源，marker两模式present，finally恢复absent。四类D trace关闭、P trace env0、无profile/export/传输、有限窗口余量已核。cold、ABBA×3每case/mode6次，24formal+4warmup完整ID/text/usage全过。N256 PD-skip189.023→P-local130.264ms，N257276.427→217.557ms；等权倍率0.736463（约-26.35%），HTTP finish0.736498。仅完整单token任务、TTFT=last、TPOT未定义；P-local没有D/KV传输，不能替代一般PD或称其通用性能。

新C4已完成r/local-complete-performance-c4：固定4slots×16次/块、ABBA×3，768formal+16warmup完整ID/text/usage全通过。PD-skip384条、aggregate7.90740完整答案/s、TTFT median505.661/p95636.178ms；P-local384条、8.88680/s、444.027/634.068ms。rate倍率1.123857、median0.878113、p950.996683，p95基本持平。PD各块7.23–9.73/s，P-local7.31–9.97/s，明显波动，不称稳定容量增益；与上轮官方PD-skip9.77816/s条件/时段不同，不连乘或冒充当前原生基线。两模式保留同一P/D资源，D skip marker均present，finally已恢复absent。双方与代理块边界idle，诊断关闭、profile/export/传输结束；不是最大SLO容量或一般64输出结论。

新CPU入口已做只读SHA/console/source snapshot，保持模型和代理的有限窗口；r/local-complete-snapshot。当前继续分解P阶段：先冻结两个算术题各N256/257交叉输入（只增一个无关空行或去掉一处无关note冒号，数学尾部不变），用原P局部完整结束路径校验答案并比较同题长度边界，再按实际native profiler能力捕获代表rank0的算子/输入shape成本。只有找到实际差异才选择机制，不把现有90ms差直接归因shape、算子、调度或盲改chunk参数。保持原模型/精度/算子与SDMA；此诊断不是参数扫描、通用性能或任务完成。不因提交/入口恢复等待用户。

交叉已完成：r/prefill-boundary-crossover，24formal+4warmup全部完整答案正确。84题N256129.660→N257217.426ms（1.676899），79题130.287→217.946ms（1.672812），等权1.674854。当前只证明两题均复现成本边界，内容/位置也改变，不能指定原因。

**当前离线解析中**：r/prefill-boundary-profile确认10条P原生局部完成请求（2arm+8active）完整答案均正确，原生stop_profile已返回，16rank原件均产出，总3298916524B，根=P32root/profile/。rank0叶=dp0_pp0_tp0_dcp0_ep0_rank0_240_20261018100827626_ascend_pt。门槛在P32root/prefill-boundary-profile-v1/result.json。当前原生TorchNPUProfilerWrapper忽略active_iterations、max_iterations0，Worker在execute前step，故第二arm已开始捕获；源码sha5242ee15…与base wrapper均保存，不能按active_iterations1误算捕获边界。

export_prefill_boundary.py正在host32归档all16 raw并离线CPU导出rank0，collector r/prefill-boundary-export-rank0，exec session82982（若仍运行用write_stdin查询）。CPU容器名glm53-boundary-p-rank0-export-v1，networknone、16CPU/64G、仅manager控制设备、timeout420；外层collector900秒。原P/D均保留，当前无正式计时/新请求。先取得导出结果和SHA备份，再用profile行记录的host32 monotonic/wall范围匹配kernel_details/trace输入shape、类型与区间；算子duration可重叠，不能直接相加称wall成本。失败仅修离线副本，不先重跑设备；后续计时必须确认全部profile/export/传输结束。

PR2 https://github.com/wio1997/Inference-Foundry/pull/2 保持draft；源码/证据按普通commit/push保存在当前codex/glm5-3-vllm-optimization分支，提交身份以Git HEAD及PR head为准，不自动合并。阶段提交不等于任务完成，继续上述判别与性能工作。
