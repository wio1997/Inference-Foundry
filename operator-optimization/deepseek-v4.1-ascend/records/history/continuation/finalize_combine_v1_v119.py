"""Seal V1 operator/count evidence before the separately approved Pad diagnostic."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime,timezone
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v90 import persist_refinement

def main():
    assert not(O/'REVIEW-CHECKPOINT-v119.json').exists()
    assert read(O/'combine-v1-preflight-v119.json')['status']=='PASS'
    names=['combine-v1-inputs-v119.json','combine-v1-source-contract-v119.json','combine-v1-later-profile-scope-v119.json',
        'combine-v1-observer-peer-loads-v119.json','combine-v1-row-redistribution-proof-v119.json','combine-v1-composition-and-theory-gates-v119.json']
    detail=dict(evidence=names,later_profile_source_predicted_V1_executions=960,
        saved_observer_source_predicted_V1_receiver_records=640,raw_metadata_records_retrieved=4736,
        raw_compressed_bytes=1693174,actual_observer_expert_partition_cases=15360,logical_row_reference_cases=840,
        independent_margin_fixtures=300,independent_partition_fixtures=480,
        observer_expert_peer_joint_counts_recovered=True,later_profile_joint_counts_or_internal_phase_time_bound=False,
        new_pairs=0,new_groups=0,new_native_promotions=0,whole_requirement_closed=False)
    for base in ['candidate-relation-clauses','candidate-relation-coverage','candidate-pair-navigation','candidate-operator-footprints','operator-analysis-coverage-ledger']:
        d=deepcopy(read(O/(base+'-v118.json')));d['V1_peer_row_review_v119']=deepcopy(detail)
        if base=='candidate-relation-clauses':
            g=next(g for g in d['higher_order_groups']if g['id']=='HG73-MOE')
            g['V1_peer_row_review_v119']=dict(**detail,contract='Withinexpert rowredistribution keeps originalclipping,FP16/scale/BF16return and weightedunpermute. Itchangesworkerownership,notworkorpeerbytes. Keepflags/completion/alias/physicalstoreproof;do notdouble-countFINE/CUT/V2paddingcredits.')
        write(O/(base+'-v119.json'),d)
    cov=read(O/'candidate-relation-coverage-v119.json')
    obligations=deepcopy(read(O/'coverage-current-obligations-v118.json'))
    for g in obligations['obligations']:
        if g['id']=='G10':g['remaining']=f"Central{cov['scoped_conditional_pairs']} conditional relations/{cov['higher_order_groups']} groups leave{cov['remaining_pair_relations']} pairs to reconcile;"+g['remaining'].split(';',1)[1]
        if g['id']in ['G01','G02','G04','G05','G06','G07','G09','G13']:g['V1_peer_row_review_v119']=deepcopy(detail)
        if g['id']=='G04':g['remaining']=('RawC65536isboundto3200later targetrecords;1272otherlatercapacitiesremainunknown. '
            'Originalobserverjointsender/expert/receivercounts are now recoveredandconserved,including640source-predictedV1receiverrecords;notjoinedtolaterTASKtimings. '
            'Needmatchinglaterprofileperexpert-peerwork,loadedV1ABI/physicalwriteextents andactualallocator/capture/retirementlifetimes.')
        if g['id']=='G13':g['remaining']=('V1rowepilogueandfixedpeerownershipare nowexpanded;originalobserverjointcountsrecoveredfrom4736SHA-pinnedrawrecords,notmerelymarginbounds. '
            'Do notrereadgenericv56FINE/CUT, v116V2mask, v117descriptorcap orv118Engrammaskasnewanswers. '
            'ActualhistoricalV1branch/coremapping/phasetimes,matchinglaterprofilejointcounts,compiledstore/UB/eventownership andtypedVector/GM/linkservice remain namedquestions. '
            'UserjustauthorizedonePad44Bcapture;executeoncewithfreshidle/container/providercontrolsafterthischeckpoint,noModel/noReplay.')
    obligations.update(parent='coverage-current-obligations-v118.json',new_native_or_whole_closures=0,overall_complete=False)
    write(O/'coverage-current-obligations-v119.json',obligations)
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v118.json'))
    audit.update(previous_goal_turn='PROGRESS:v118Engramfalse-mask key/value demand andsplitcosts;postcheckPASS297',input_checkpoint=118,followup_checkpoint=119)
    audit['V1_peer_row_review_v119']=deepcopy(detail)
    for r in audit['requirements']:
        if r['id']in ['ALL_MODULE_CONDITIONAL_BRANCHES','CROSS_OPERATOR_LAYER_CARD_ITERATION','NATIVE_OPERATOR_REGIMES','THEORY_AND_MEASURED_COMPARISON','CRITICAL_EVIDENCE_AND_FINAL_CLOSURE']:
            r.setdefault('evidence',[]).extend(names)
    write(O/'whole-scope-requirement-audit-v119.json',audit)
    gap=('V1sourcepeer-ownedrowworkandoriginalobserverjointcountsarenowreviewed,butactualhistoricalcompiledbranch/coremapping/laterhostprofjointcounts/internalV1timearestillunbound. '
        'Withinexpertrowredistribution preserveskeptrow/source-scale-peeridentities,originalflag/Finalize/SyncAll/CrossRank/unpermute andFP16→FP32scale→BF16RINT. '
        'SameRrowDMAs forb1 andsamepeerpayload;prefix/function/ownership/peercontention/physicalstoreextents/UB-eventprivacy andnativeprecisionneedproof. '
        '48unitrowworkersisadeclaredmodel,notphysicalspeedup/MFU. OriginalobserverC65536cannotfill1272latercapacities;V2paddingandFINE/CUTcreditsnotadditive.')
    branches=deepcopy(next(m['branches']for m in read(O/'module-phase-branch-matrix-v118.json')['modules']if m['id']=='M12'))
    r=next(r for r in branches if r['id']=='M12-OP-21');r['evidence']=list(dict.fromkeys(r['evidence']+names));r['V1_peer_row_review_v119']=deepcopy(detail);r['gap']=gap
    write(O/'native-pad-authorization-v119.json',dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
        status='AUTHORIZED_ONCE_NOT_YET_EXECUTED',user_reply='允许本次 Pad 探针',
        question_call='call_e3996f654f444f9dbed47a176cf01c6d',
        scope='OneownBOOL20→24ConstantPadNd capture,44BapplicationIOplusruntimeoverhead;idleNPUonly,noModel/noReplay;queryworkspace/packetmetadata.',
        probe_plan='native-pad-probe-plan-v111.json',next_checkpoint=120,
        original_v109_probe_authorization_not_reused=True,new_device_work_in_v119=False))
    write(O/'PROFILE-CODE-QUESTION-GATE-v119.json',dict(status='V1_PROFILE_GEOMETRY_THEN_ONLY_ROW_EPILOGUE_PEER_MAPPING_AND_SAVED_JOINT_COUNTS',
        questions=read(O/names[0])['questions'],prior_answers='v56V1caller/fusedboundary/FINE/CUT andv116V2reused;noframeworkaudit.',
        remote_read='Exactlyone1.69MBpreviouslysavedcompressedmetadatafile;v17SHA d6e787...matches;4736records/296EP16groups. SSHclosed. No container/library/devicework.',
        historical_loaded_ABI_or_internal_branch_not_promoted=True,
        Pad_probe='Newuserauthorizationreceivedandrecorded;notexecutedyet,runonceasv120withfreshguards.',overall_complete=False))
    write(O/'combine-v1-validation-history-v119.json',dict(events=[
        'v118revalidatedPASS297. InitialDSpark/Indexerquestionshadprioranswers;noarbitraryemptycandidatepairenumerationornewcountclaim.',
        'v56alreadyreadV1callerandgenericFINE/CUT;cachedchildrowepiloguenotexpandedthere. Newreviewpinscount-formFP16→FP32→scale→BF16RINTandfixedpeerassignment,notnewkernel-levelfusion.',
        'Readlater7MC2regimes4472TASK:allblockNum24/mix48;960inputgeometriespredictV1,notactualinternalflag.48worker24*2mappingremainsdeclaredsource-modelpremise.',
        'Initialmargin-onlymodelsretaineduncertaintyofexpert-peerjointcounts. Inspectingoldv17derivationrevealedoriginalsender_expert_histograminSHA-pinneddecoded-records.gz;oneauthorizedread-onlySSHrefetch1693174Bendedsuccessfullyandconnectionclosed.',
        'OriginalSHA d6e787...matches4736records/296completeEP16groups,allcapacity65536/nooverflow. Reconstructexactexpert×peerjointfor640formalreceiverrecordswithsourceM*k>4096;allpeerandexpertmarginsconserve. Theseareoldobserverrecords,not960laterTASKorlatencyjoins.',
        'Computeoriginalunit-rowLfromjointcounts,notinventfrommargins. Two2×2tableswithsame[4,4]marginsgiveL8vs4;thewitnessjustifiestherawretrievalanddoesnotleavetheoldobserverjointunresolved.',
        'Withinexpertintervalcutsverify15360actualexpertpartitions/347667logicalsegments;840syntheticclippingrefsplusindependent300margin/480partitionfixtures. No compiledDMAextentproof.',
        'ReviewcaughtandcorrectedanoverbroadextraDMAstatementbeforepersistence:oldrowclassalreadycopiesone-row;retainingb1keepsRrowDMAs/eventsunchanged,onlylogicalpeerfunctionsegments/metadata/ownershipchange. BatchedDMAfragmentcostisseparate.',
        'Alllocalpreflight4724checkspass. Branchrows356/26andcentral134/26/461unchanged;oneexistingM12V1semanticrowrefined. No newnativeorwholeclosure.',
        'UserapprovedthepreviouslypendingonePad44Bdiagnosticduringthisanalysis. Approvalrecordedseparately;v119stillnoDevice. Nextv120mustfreshidle/container/CANN/providercheckandrunonce,noModel/noReplay;do notreaskorclaimstillpendingreply.'],overall_complete=False))
    remaining=[g['id']+': '+g['title']+'. '+g['remaining']for g in obligations['obligations']]
    remaining.append('v119V1rowepilogueand640originalobserverjointreceiverrecords reviewed;15360actualpartitionrefs/840synthetic, no laterprofilebranch/phasetimepromotion.356semanticrows26modules/134relations26groups461pendingunchanged. Pad44BnowAUTHORIZED_ONCE_NOT_YET_EXECUTED,nextv120freshguards.')
    report='''本轮展开MC2 CombineV1的返回行处理，复用v56已读的调用层和v116的V1/V2选择条件；新证据是V1逐行epilogue、peer固定分工，以及从原始观测恢复的expert×peer联合计数。没有新增整体融合边界，也没有把既有FINE/CUT方案重新计为新发现。

V1在每个local expert前等待原cross-core flag并进入SyncAll，按peer=coreIdx、步长coreNum分配工作。每个peer由一个worker负责它在该expert的全部返回行。逐行epilogue读取5120个FP16 C2值，转FP32、乘该行scale、按RINT转BF16，再写回peer。两阶段私有buffer布局为81920B；count形式的Cast/Muls不能套用V2的repeat128、外部false mask或8192 padding模型。完成后仍需Finalize、全AIV同步、CrossRankSync和原weighted unpermute。

后来的profile记录blockNum24/mixBlockNum48，960次输入几何按所选源码预测V1。以24×2的48 worker为声明模型、EP16时，至多16个worker能拥有peer行工作；这不是整算子的硬件利用率，也不是已测出的瓶颈。历史编译内核、get_block_num/subblock映射和原观测的具体launch仍未逐任务绑定。

原独立观测的汇总表提供peer总数和expert总数，但这两类边际不能决定每个expert的最大peer行数之和。例如[[4,0],[0,4]]与[[2,2],[2,2]]边际相同，单位行模型的该和分别为8和4。进一步查看旧v17推导脚本，发现原decoded-records.jsonl.gz保留sender_expert_histogram；已通过SSH只读取回1693174B文件，SHA与v17固定值d6e787...一致。4736条记录组成296个完整EP16组，原capacity均65536且没有计数溢出。SSH已关闭，无容器启动、库加载或设备工作。

在87个formal观测cohort中，640条接收rank记录符合源码M×topK>4096的V1条件。由发送方直方图精确恢复每个receiver的expert×peer联合计数，并逐项对齐原peer矩阵及receiver expert计数。按原分工，这些记录的正行peer owner数量为6..10，中位9；receiver总行数为13..14852，中位2549。它们来自observer-20261030T160315Z，不能与后来960次hostprof执行、耗时或phase作跨采集关联。原观测capacity也不能填补1272次后来执行的缺失字段。

可具体研究的改写是保留expert级就绪与clipping，按每个expert保留行的ordinal区间，把行平分给worker，再与peer前缀区间求交。每个cut继续使用原C2行、scale行和peer目标行；不改变expert/rank保留集合、舍入或weighted unpermute。原观测15360个expert分区、347667个逻辑片段，以及840组额外clipping参考均通过行覆盖核对，另有300组独立边际边界与480组独立行分区核对。它们证明逻辑行映射，不证明编译后DMA字节范围、UB/event私有性或跨卡完成协议。

保留逐行epilogue时，总R次行DMA、转换、Muls、scale引用和peer字节数不变；增加的是worker分工与可能更多的函数片段/metadata处理，不能声称已有整段DMA被进一步切碎。将多行合批是另一个选项，若保留三个buffer及双阶段，私有布局按81920×b增长；是否能放入UB、如何逐行scale、事件和mask如何处理仍需单独证据。新分工不缩小热peer的链路注入量，即使更多AIV参加，链路仍可能主导。

理论模型保留边界：在48个等速单位行worker、无限带宽、无额外片段/metadata开销、expert间同步的抽象域内，旧行服务量可由联合计数计算为sum_e max_d x[e,d]，均分构造的行服务量为sum_e ceil(n_e/48)。计数可用于比较排程，但不是微秒、物理速度上限或可达加速比。原固定peer源程序还有每worker H×S_d个转换/Muls的符号资源约束；重组后不能把该固定分工限制作为所有允许实现的下界。实际typed Vector/GM/L2/HBM/peer链路能力、就绪、开销和V1阶段时间仍未知。FINE片上handoff、expert cut和V2 padding的工作/接口收益不得重复相加。

本轮前置复核4724项通过，细化原M12-V1语义行，356条语义记录、26模块、134条件关系、26联合组、461对待核对不变；9类/1352次关键原生字段、849缺PMU、16未绑定phase图、3136含糊归属和0整模块关闭保持。用户刚明确批准单次44B Pad探针，授权已记录；v119未执行设备工作，下一步按新鲜空闲卡、容器和provider控制执行一次，禁止模型加载和replay。整体目标继续。
'''
    action=('继续原Goal，v119PROGRESS，verify_combine_v1_v119须PASS；先读combine-v1-summary/observer-peer-loads/row-proof/composition/current-obligations-v119。'
        'v118manifest58files/PASS297；v119preflight4724。V1childrowepilogueexpanded：onepeeroneAIVworkerwhenW>=EP，H5120每行F16→F32scale→BF16RINT，privateUB81920，countAPI不是V2repeat128。'
        '新只读取回原decoded-records.gz1693174B/d6e787...，4736records296EP16groups；原observer640sourcepredV1receiverranksjointE×peer从senderhist恢复，margins/C65536守恒。'
        '与后来profile960sourcepredV1（block24/mix48）严格分开，无耗时/phase/ordinal跨采集join。48workers是24*2声明映射非已绑定compiledcapacity；sourcepeer owners6..10中位9，非全kernel利用率。'
        'withinexpertordinalsplit保留clipping/source-scale-peer映射和flags/Finalize/allSync/CrossRank/unpermute；15360actualexpertpartitions347667logicalsegments+840refs+300margin/480independentpartitions。原RrowDMA/event count b1不变，newmetadata/functionsegments成本；batchedrowUB81920*b另域。'
        '单位行model exactoldL=sum_e max_peer x，new=sum_eceil(n_e/48)，不是physicaltime/speedup；不把旧fixedpeer限制强加所有新实现。samepeerbytes、FP16round、weightedSUM、alias/event/UBprivacy/physicalextent/typedcapacity必须保留。v56FINE/CUT/v116V2是prioranswer。'
        '当前356semanticrows26modules134relations26groups461pending，813/1297/46graph，2073/285598conditional9/1352pending849PMU16phase3136ambiguity0whole/noallMFU不变。'
        '关键新steering：用户明确回复“允许本次 Pad 探针”，native-pad-authorization-v119.json=AUTHORIZED_ONCE_NOT_YET_EXECUTED。立刻以v120单独执行既定v111脚本：先freshidle/process/container/CANN/providerSHAguards，只选空闲卡，一次BOOL20→24(valueTrue)capture44BappIO，无model/replay/opaqueGPUderef。原v109授权不可复用，但本次已获新授权，不要再问同一权限。'
        'v119所有read/local作业结束SSHclosed；下一probe另用新socket和唯一/tmp输出，有界超时并验证退出/NPU进程清理。TaskACTIVE Loop004EVALUATING GoalACTIVE，不改封存v119或重跑finalizer。')
    inputs=[Path(r['path'])for r in read(O/names[0])['inputs']]
    persist_refinement(version=119,previous=118,title='MC2V1 peer-owned return rows and recovered observer joint work',
        summary=dict(filename='combine-v1-summary-v119.json',previous_turn_classification='PROGRESS:v118Engrammaskdemand/splitcosts;PASS297',
            **detail,refined_candidates=1,refined_modules=1,branch_records=356,semantic_modules=26,
            relation_clauses=134,higher_order_groups=26,pending_central_pair_reconciliation=461,
            partial_graph_node_rows=813,partial_graph_edge_rows=1297,structural_pair_hints=46,
            scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
            new_device_work=False,implementation=False,native_test=False,actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,
            native_Pad_probe='AUTHORIZED_ONCE_NOT_YET_EXECUTED;nextv120freshguards',
            remote_job_status='OneSHA-pinnedsavedmetadataread1693174Bcompleted,SSHclosed;noContainer/Torch/NPU/model/context/capture/replayinV119. NewPadapprovalpendingexecution.'),
        changes=['Expand V1rowepilogue/countCast/peerassignment;retainalreadyfusedboundary andgenericFINE/CUTasprioranswers.',
            'Recoveroriginalobserverexpert-peerjointcountsfrom4736SHA-pinnedrawrecords;640source-predictedV1rankrecordsand15360actualexpertlogicalpartitionchecks,withoutlaterTASKtimingjoin.',
            'Modelwithinexpertrowredistributionwithunchangedwork/bytes/rounding/completion;keepphysicalcapacity/extents/nativephasegatesandallcoveragecountsopen.',
            'RecordnewexplicitonePadprobeauthorizationforimmediatenextv120execution;v119itselfhasnodevicework.'],
        candidate_updates={'F21':dict(review_v119=deepcopy(detail),conclusion_changing_gap=gap)},
        module_updates={'M12':dict(candidates=['F21'],finding='V1one-peer/one-worker rowpath andoriginalobserverexpert-peerjointworknowexplicit;withinexpertrowredistributionchangesownership,notsamepeerpayload orFP16/BF16boundaries. Laterprofileinternalbranch/timesstillunbound.',evidence=names,gap=gap)},
        semantic_branches={'M12':branches},remaining=remaining,action=action,
        jobs=[dict(kind='profile',status='pass',description='PinV1rowchildsourceandrefetchoneoldSHA-controlledmetadatafile;reconstruct640receiverjointcountswithoutjoininglater960TASKtimings.',artifacts=[O/n for n in names]+inputs),
            dict(kind='design-check',status='pass',description='15360actualexpertpartitions840syntheticrowrefs300margin480independentpartitionfixtures;preserveclip/scale/peer/round/events andrecordnewPadapproval.',artifacts=[O/'combine-v1-preflight-v119.json',O/'combine-v1-validation-history-v119.json',O/'coverage-current-obligations-v119.json',O/'whole-scope-requirement-audit-v119.json',O/'native-pad-authorization-v119.json'])],
        report=report,read_first=names+['coverage-current-obligations-v119.json','native-pad-authorization-v119.json'],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/'analyze_combine_v1_v119.py',N/'verify_combine_v1_v119.py',N/'read_v1_observer_joint166_v119.py',N/'persist_profile_refinement_v90.py'],
        executor_prefix='combine-v1',main_script=Path(__file__),source_read=True)

if __name__=='__main__':main()
