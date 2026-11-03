"""Seal a complete output-cut comparison; retain missing runtime evidence."""
from pathlib import Path
from copy import deepcopy
from finalize_origin_coverage_v9 import N, O, read
from analyze_evidence_batches_v155 import pin
from persist_delta_checkpoint_v156 import persist_delta


def main():
    assert not(O/'REVIEW-CHECKPOINT-v157.json').exists()
    evidence=['WOA-local-versus-full-source-contract-v157.json','WOA-WOB-scoped-native-view-census-v157.json','output-chain-weight-and-delivery-plans-v157.json','output-WO_B-partial-round-witness-v157.json','WOA-evidence-route-status-v157.json']
    queue=deepcopy(read(O/'coverage-work-queue-v156.json'));queue.update(version=157,parent=pin(O/'coverage-work-queue-v156.json'),new_evidence=evidence)
    queue['work'][0].update(state='ONE_COMPLETE_OUTPUT_CUT_CONDITIONALLY_COMPARED_REUSE_PROOFS',action='Reuse the A/current,B/WO_Areplicated,C/both-weights+faithful-partial-replay output-cut comparison. Do not extend this cut with more unrelated pair or finite-case exercises. Prioritize an all-module critical-evidence acquisition plan grouped by source and acquisition point;complete other decisive saved-evidence checks where available.',evidence=evidence,remaining='Actual all-group weight values/residency/epochs/allocator/scratch/reader/nativeSUM/hardware capacity remain open. No checkpoint headers acquired. Original26module/358row inventory remains nonexhaustive.')
    queue['work'][2].update(state='CURRENT_STREAM_LIMITS_BOUND_NEED_DIFFERENT_EVIDENCE_SOURCE',action='Across all26modules,consolidate conclusion-changing evidence gaps into the smallest nonduplicated collection batches using existing prepared CPU/native-adapter readiness. Name exact evidence source,required scope,and acquisition point;no new NPU/model/context/replay is authorized. Archive-only login currently failed;do not repeatedly retry credentials or the same missing raw fields.',evidence=['raw-field-availability-and-reuse-v155.json','WOA-evidence-route-status-v157.json'],remaining='Address/length/workspace/alias/readers,actual binary/mathmode/physicalcapacity and unknown branch/phase epochs require stronger evidence;local arithmetic examples cannot close them.')
    action=('继续原完整目标与Loop004；先读v157摘要、覆盖工作队列和算子增量。v157已统一输出切面A当前headA2A+WO_A+WO_B+RS、B复制WO_A后缩小head交换但保留RS、C复制两级权重并在owner保持原partialBF16+SUM的参考图。'
            '直接逻辑交付模型中A=7*18432*sumM，B=7*(12288*sumM+8388608)，C=7*18874368；B/A=2/3+4096/(9MR)，C/A=1024/(MR)。额外独立BF16内容B56MiB/C126MiB每rank每层，不是实际分配。'
            '当前R2的B/A0.9395~0.9511、C/A0.6139~0.64仅为该参考模型，不能作为线流量/加速比。C不能直接替换成一个fullWO_B GEMM，构造BF16中间舍入例原结果1而合并129/128；原native SUM/round/order/NaN/error/ABI/读者等合同仍需证明。'
            '960个已有原生view绑定确认WO_A一组[1,4096,1024]BF16、WO_B[5120,1024]BF16；view不能证明全rank储存。V4.1CP调用默认local-group接口；通用full支持另有配置/is950/非decodeguard且gather两级权重，勿混同当前路径。'
            'v157暂停前的归档读取未取得checkpoint/header/opslinear/parameter新文件：缓存连接SSH0却CoreAPI资产错误，新身份尝试255；全部客户端已终结，恢复后仅本地分析。不重复登录，不把失败记作native失败或无权重证据。'
            '下一步转向全26模块的关键证据收敛与最小采集方案：按证据来源/采集时点复用v155可用性结论和已准备CPU/native-adapter接口，明确哪些原存档可补、哪些必须不同来源/新采集，完成可审阅的具体方案；不再围绕这一切面追加无关数学模型或候选全配对。'
            '没有新NPU/模型/context/capture/replay授权，先做本地/只读准备。原35+7候选、26模块358语义行、9/1352、849PMU、16phase、3136归属、G14及全部broader gaps保持open，整体/物理MFU/最低未成立。'
            '以DELTA_CHECKPOINT_V1中的实际路径和v155/156/157叠加证据恢复；未改动的候选/模块/分支/关系权威仍为v154，不能按版本名假定全量拷贝。')
    report='''本轮把“head AllToAll前移WO_A投影”的比较范围补到输出ReduceScatter，统一比较三种完整输出通信布局。v156的四分之一只以head交换的激活载荷为分母；当前还计入原WO_B的BF16 partial到owner的归约交付，避免把局部载荷收缩写成整段输出通信收益。

所有数字属于一个明确的直接源到目的逻辑交付参考：TP8、G=8M、每个非自身peer的原head载荷8192M字节，归约partial交付10240M字节；每层同一权重/布局代次只分发一次，复用调用的owner行数之和记为ΣM。它不是实际HCCL线流量、普适物理通信下界或时延模型。

| 方案 | 每rank、每同权重代次的逻辑交付量 | 额外独立BF16权重内容/层 | 当前M800～834、复用两次的新/旧 |
| --- | --- | --- | --- |
| A：当前head交换、两级局部投影、RS | 7×18432×ΣM | 0 | 1 |
| B：复制WO_A、缩小head交换、保留WO_B和RS | 7×(12288×ΣM+8388608) | 56MiB | 0.9395～0.9511 |
| C：复制两级权重、owner重现原partial和SUM | 7×18874368 | 126MiB | 0.6139～0.6400 |

因此，在同一参考交付图内，B的输出激活交付极限是A的三分之二，包含一次权重分发后当前两次复用约节省4.9%～6.0%。B优于A的严格整数条件仍是ΣM≥1366。C优于A需ΣM≥1025，C优于B需ΣM≥854；这些结果同时受独立权重内容预算、相同层/参数代次和全部数值/读者合同约束。内容预算不等于实际分配；加载来源、多播、缓存、scratch、排队读者、驻留寿命和不同网络算法需要对应的完整成本图。交付量减少不能直接证明加速。

C是保持每组原WO_A/BF16与WO_B/BF16 partial、再重现原SUM和舍入合同的条件参考，不能直接用一个全宽WO_B GEMM代替。一个精确的小BF16输入例中，两组未舍入partial分别为257/256和1/256；原分组BF16舍入后SUM再舍入得到1，先做全宽dot再舍入得到129/128。该反例否定无条件的全宽GEMM替换，保留忠实partial重现的可能；没有声称实际CANN加载算子已经发生此数值失败。新图仍需原native SUM顺序、精度、误差/NaN、捕获和各rank联合控制合同。

本轮复用了已有960个原生绑定，WO_A权重view均为[1,4096,1024] BF16（8MiB逻辑内容），WO_B为[5120,1024] BF16（10MiB），并有对应head交换和输出RS。它们证明这些操作数view的形状，不证明整个rank只存了这些权重。定点源码显示WO_A为ColumnParallelLinear，输出分片尺寸传给权重创建接口；标准legacy加载器按输出分片复制，v2仍是被委托的方法。选中的V4.1 CP路径以两个位置参数调用投影，使用默认局部分组；通用DSA的full模式有配置/is950/非decode条件，且会gather WO_A与WO_B，是另一张图。不能把通用支持当成当前各rank已经有完整权重，也不能据局部view断言没有其他完整分配。

checkpoint头部证据仍缺失。暂停前的初次只读读取在SSH返回0后得到非JSON输出，未形成缓存；随后只读身份核对捕获到堡垒机Core API资产错误，新连接身份尝试返回255。全部客户端已经终结，恢复后没有重试服务器。失败保留为连接/本地解析记录，不是native算子失败，也不是checkpoint不存在或权重不驻留的证据。

本轮的完整输出切面比较及相关数值、来源和预算合同已独立校验，未重跑未变更的旧算子证明。下一步优先跨全部模块收敛关键证据，按来源与采集时点合并最小补证方案，继续完成已有资料可支持的判断。35项模型候选、7项原框架候选、26模块358语义行的原范围保留；9类1352次操作数、849项PMU、16个phase图、3136项归属、G14及其他广泛缺口均未关闭。没有新独立性、原生融合、整模块或物理MFU/最低利用率完成声明，也没有新增设备、模型、context、capture或replay执行。
'''
    overlay=dict(version=157,base_operator_authority_checkpoint=154,retained_previous_overlays=[155,156],candidate_ids=['F14','F15','F16'],module_ids=['M11','M23'],evidence=evidence,scope='OneexistingcompleteoutputcommunicationcutwithsameM/Gtypedshapes and all previous guards;source-local-vs-generic-full weight interpretation plus joint activatedelivery/weightcontent/partialround models.',B_logical_activation_ratio_including_RS='2/3',C_requires_faithful_partial_round_SUM_replay=True,checkpoint_header_evidence_available=False,actual_all_weight_residency_or_physical_storage_closed=False,broad_candidate_gaps_preserved=True,new_pair_or_independence_proofs=0,new_native_or_whole_closures=0,critical_path_gain_us=None,overall_complete=False)
    gate=dict(status='COMPLETE_OUTPUT_CUT_WEIGHT_SOURCE_AND_NUMERIC_CONTRACT',questions=['DoescurrentlyselectedWO_Aprojectionrequestlocalorgenericfullgroups,andwhatdoesitsone-groupnativeviewprove?','Whatisjointactivation/weightdeliverycostwhenoutputReduceScatterisretainedorlocallyreproduced?','CanonefullWO_Bmatmulreplaceoriginalroundedgroup-partialsandSUMwithoutanewnumericproof?'],source_scope='Only frozen WO_A/B constructors,ColumnParallelLinear partition/copy/delegate interface,V4.1 CP projection call,and generic DSA output weight/view/full-switch helpers. Existing960 native frontier identities reused;no broad framework survey. Remote checkpoint/opslinear/parameter read failed and supplies no evidence.',target_operator_source_read=True,new_target_operator_source_read_after_resume=False,new_local_frozen_operator_source_read=True,no_model_operator_implementation=True,no_remote_Torch_CANN_NPU_model_or_replay=True,no_container_start_exec_or_attach=True,Pad_authorization='GRANTED_AND_CONSUMED',overall_complete=False)
    persist_delta(version=157,previous=156,prefix='output-weight-plan',changes=['Bind frozen local/default vs guardedgenericfull output weight source contracts and reuse960 native weight-view/RS identities without whole-storage inference.','Compare A/current,B/WO_Areplicated,C/both-weightfaithful-partial output graphs with one joint head+RS+weightdelivery account,logical-content budgets and exact reuse thresholds.','Independent BF16 round witness rejects unconditionalfullWO_Bdot substitution without rejecting faithful partial replay.','Retain failed157archive/identity attempts as missing evidence;pause/resume respected,all clients terminal,local-only resumed work;zero newpair/native/wholeproofs.'],report=report,action=action,remaining_delta='G157: Completeoutputcommunicationcutnowhasthreeconditionaljointdelivery/weightcontentplans. B retainsRSsoactivationlimit2/3ofAinthisreference;CneedsoriginalpartialBF16+SUM,notonefullWO_BGEMM. Checkpointheaders/actualall-weightresidency/nativepolicy/physicaltraffic-capacity remainunbound. Move next to all-module criticalevidence minimization without repeating localcutcases.',source_gate=gate,operator_overlay=overlay,queue=queue,jobs=[dict(kind='profile',status='error',description='Initial157readonlyarchive fetch produced no parsed evidence after SSH0;identity recheck captured bastionCoreAPI asset error;freshidentity client255. Allterminal before/respected pause,resumed local-only. No checkpoint/cache/newloader source/native or device failure proof.',artifacts=[O/'WOA-evidence-route-status-v157.json',O/'WOA-read-attempt-failure-v157.json',N/'rank157-identity-recheck.json',N/'original166-xinyufeng-recheck-v157.json',N/'read_woa_artifacts_v157a_failed.py',N/'probe_original166_xinyufeng_v157.py']),dict(kind='design-check',status='pass',description='Localoperator-onlysource/default/fullbranch audit and960reusednativeviews;complete A/B/Coutputdelivery/weightcontentbudget/strictthresholdmodels andindependent BF16partialround witness. PreflightPASS;no unchangedpriorproof replay/newpair/native/wholeclosure.',artifacts=[O/'output-weight-plan-preflight-v157.json',O/'output-weight-plan-inputs-v157.json']+[O/x for x in evidence])],scripts=[N/'analyze_output_weight_plan_v157.py',N/'verify_output_weight_plan_v157.py',N/'read_woa_artifacts_v157.py',Path(__file__)])


if __name__=='__main__':main()
