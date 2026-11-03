"""Persist raw-own-attribute evidence without expanding framework scope or claiming closure."""
from copy import deepcopy
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import N,O,R,read,write
from persist_profile_refinement_v90 import persist_refinement

def main():
    assert not(O/'REVIEW-CHECKPOINT-v98.json').exists()
    assert read(O/'raw-operator-analysis-preflight-v98.json')['status']=='PASS'
    names=['raw-profile-schema-and-args-v98.json','raw-operator-evidence-extraction-v98.json','raw-profile-capability-audit-v98.json',
      'MC2-recorded-attributes-and-executions-v98.json','remaining-operand-evidence-status-v98.json','MC2-attribute-to-ceiling-contract-v98.json','minimal-operator-evidence-plan-v98.json']
    gap=('RawprofileC=65536/worldSize16 isbound to3200targetM3executions;1272otherMC2executionsstilllackC. '
      'LoadedABI/writeextents,actualsame-callreceivercounts/all-rankbounds andallocator/capturelifetimesremainconditional. '
      'CapacityisnotprocessedGEMMrows;active-maskvaluesareunknown. SevenMemSetregimes1272executionslacknativeaddress/length;'
      'twoBOOL[24]regimes80executionshaveonlystaticnessmetadata,notfield/allreaderidentity. '
      'No universal numericphysicalceiling,actualspeedup,all-moduleMFU orwhole-goalclosurehasbeenproved.')
    for base,parent_key in [('candidate-relation-clauses','parent_relation_index'),('candidate-relation-coverage','parent'),('candidate-pair-navigation','parent')]:
        d=deepcopy(read(O/f'{base}-v97.json'));d[parent_key]=f'{base}-v97.json'
        if base=='candidate-relation-coverage':d['parent_relation_index']='candidate-relation-clauses-v98.json';d['scope']='124scopedconditionalpairs/25groups/35modelcandidates;471centralpairreconciliationsremain. Rawattributeevidencev98doesnotclosewholepairvariants.'
        if base=='candidate-pair-navigation':d['parent_relation_index']='candidate-relation-clauses-v98.json'
        d['raw_operator_evidence_v98']=dict(evidence=names,new_relations=0,new_groups=0,recorded_capacity_executions=3200,remaining_capacity_executions=1272,whole_pair_closed=False)
        d['overall_complete']=False;write(O/f'{base}-v98.json',d)
    fps=deepcopy(read(O/'candidate-operator-footprints-v97.json'));ix={r['id']:r for r in read(O/'candidate-record-index-v97.json')['model_candidates']}
    for r in fps['candidates']:
        p=R/'candidates'/(r['id']+'.json');h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==ix[r['id']]['master_sha256'];r['captured_master_sha256']=h;r['captured_master_version']=97
        if r['id']in ['F20','F21']:r['raw_operator_evidence_v98']=dict(evidence=names,new_operator_atoms=0,profile_recorded_C_bound=True,whole_candidate_legal=False)
    fps.update(parent='candidate-operator-footprints-v97.json',overall_complete=False);write(O/'candidate-operator-footprints-v98.json',fps)
    op=deepcopy(read(O/'operator-analysis-coverage-ledger-v97.json'));pendingids={r['regime']for r in read(O/'remaining-operand-evidence-status-v98.json')['pending_regimes']}
    for r in op['regimes']:
        if r['operator']=='DispatchFFNCombine':r['raw_own_attribute_evidence_v98']=dict(evidence=names[1:4]+[names[5]],promote_state=False,recorded_C_binding_uses_exact_TASK_identity_not_role_alone=True)
        if r['regime_id']in pendingids:r['raw_operand_gap_recheck_v98']=dict(evidence=[names[1],names[2],names[4]],state_unchanged=True,missing_is_not_zero=True)
    op['raw_operator_evidence_v98']=dict(databases=32,hashes_match_original=True,MC2executions=4472,recorded_C_executions=3200,C_missing_executions=1272,
      unknown_operands_or_field_executions=1352,new_native_executions=0,new_regime_promotions=0)
    write(O/'operator-analysis-coverage-ledger-v98.json',op)
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v97.json'));audit.update(previous_goal_turn='PROGRESS:v97capacity/clipping/workspaceandsharedlivebudget;182postchecks',input_checkpoint=97,followup_checkpoint=98)
    audit['receiver_budget_refinement_v97']['superseded_capacity_evidence']='v98 bindsprofile-recordedC/worldSizefor3200targetexecutions;1272remainunknown. Source-default-onlystatusisretainedashistory,notcurrentall-executionclaim.'
    audit['raw_operator_evidence_v98']=dict(same_hash_raw_databases=32,recorded_C_executions=3200,unbound_C_executions=1272,all9_operand_or_field_regimes_still_pending=True,
      no_new_framework_research=True,no_new_inference_capture_or_device_work=True,new_whole_requirement_closure=False)
    for r in audit['requirements']:
        if r['id']=='THEORY_AND_MEASURED_COMPARISON':r['evidence']+=names
        if r['id']=='NATIVE_OPERATOR_REGIMES':r['gap']='Exactrawattributeandoperandgaprecheck:all7MemSetand2BOOLfieldregimesremainpending;nativeownattributesareusedwherepresent,missingfieldsarenotzerowork.'
        if r['id']=='CRITICAL_EVIDENCE_AND_FINAL_CLOSURE':r['gap']=gap
    write(O/'whole-scope-requirement-audit-v98.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v98.json',dict(status='PROFILE_RAW_OWN_ATTRIBUTES_RECOVERED_THEN_ONLY_OPERATOR_COUNT_AND_CLIP_SOURCE',
      questions=['Does theoriginalsavedDBcontainownMC2attributesomittedfromthelocalprojection?',
        'Which exactexecutions carrycapacity/worldSizeandwhichremainunknown?',
        'DoesCdetermineGEMMrows,anddoesexpert_token_numsrepresentpreclipdemandorexpertprefix?',
        'DoownMemSet/BOOLattributesidentifydestinationlengthorallreaders?'],
      existing_cached_source_files=2,new_target_source_files=0,new_public_documents=0,no_framework_wide_research=True,
      server_operation='Readonlyhash/SQLitequeryofknownsavedcapturefilesonpreviouslyauthorized166;actualhost105verified.',
      new_model_Torch_NPU_capture_service_change_or_implementation=False,overall_complete=False))
    write(O/'raw-operator-validation-history-v98.json',dict(refinements=[
      'FreshSSHProxyCommand=false localfailureisnotaccountauthorizationfailure;correctexistingknownhostauthrestoredreadonly166connection. Nootherassetfallback.',
      'Initial128MiB/DBand2GiBtotalguardwasbelowknownpinnedsizes;noqueryranunderthatguard. Bounded256MiB/DB,4GiBtotal,streamedSHAverified32originalDBs3278102528Bunchanged.',
      'RepeatedcachehashinsideeachTASKwasremoved;oneverifiedcachehashperDBisreused. Onlyownlocalanalysisprocesswasstopped;notargetprocesswasqueriedorkilled.',
      'M19/BOOLv67records392andM12v63records960arecombinedanddeduplicated;the9pendingregimescover1352executions.',
      'record_op_args/_op_attr/profile_memory=false doesnotimplyallnativeattributesempty:3200MC2ownattributesaredecodedandjoinedtosamehashTASKidentities.',
      'GetCumsumForMMAIVprefixesacrosssourcesforeachlocalexpert;lastvectorispreclipperexpertdemand,summingvalidentriesgivesreceiverdemandinthatsourcecontract. Noactualcountvalueswerecaptured.',
      'C=65536isprofile-recordedcapacity,notprocessedrows/FLOPs/HBMtraffic/liveallocation/requestcriticalpath. Diagnosticplanisprepared,notexecuted.'],
      all_local_jobs_finished=True,overall_complete=False))
    updates={cid:dict(review_v98=dict(evidence=names,raw_MC2_C_recorded_executions=3200,raw_MC2_C_missing_executions=1272,
      source_count_and_clipping_contract_reviewed=True,source_reference_ops_per_processed_assignment=70778880,
      actual_receiver_count_or_MFU_or_peak_credit_known=False,new_relations=0,new_regime_promotions=0,implementation=False,native_test=False),
      conclusion_changing_gap=gap)for cid in ['F20','F21']}
    mods={
      'M12':dict(candidates=['F20','F21'],finding='32samehashrawDBsrecoverownC65536/worldSize16for3200targetM3MC2TASKs(2560root/640phaseunbound),1272otherMC2stillunknown. Countsumispreclipperexpertdemand;CisnotGEMMrows. 960M12MemSetTASKsremainwithoutoperands.',evidence=names,gap=gap),
      'M19':dict(candidates=[],finding='312SPpadding/shardMemSetTASKsrecheckedinoriginalDBs:ownattributesempty,shape/dtypeN/A,nooperandaddress/lengthtablefields. Parentgeometrydoesnotclosehelperwriteextentorfusionlegality.',evidence=[names[1],names[2],names[4]],gap=gap),
      'M05':dict(candidates=[],finding='BOOL[24]prefix80TASKshaveonlyIsStaticKernel:false;fieldidentity/alias/allreadersstillunknown. Nozero-elisionorworkdeletionpromoted.',evidence=[names[1],names[2],names[4]],gap=gap),
      'M21':dict(candidates=[],finding='Same80BOOLprefixTASKsremainambiguouslyM05/M21;staticnessmetadataisnotsemanticfieldidentity. Both16phase-unboundand64decode-compatibleexecutionsremainpending.',evidence=[names[1],names[2],names[4]],gap=gap)}
    remaining=list(read(O/'module-coverage-gaps-v97.json')['remaining'])
    remaining=[('HISTORICAL_GAP_SUPERSEDED_IN_PART_BY_V98: '+x)if x.startswith('v97:')else x for x in remaining]
    remaining.append('v98: Profile-recordedMC2C65536/worldSize16boundto3200exactTASKs;1272otherMC2Cremainunknown. Noactualsamecallreceivercounts/allrankbound/loadedABI/writetile/allocatorlifetimeproof. CapacitycannotstandinforprocessedrowsorHBMbytes. Raw32DBschemaandownattrsconfirm7MemSet1272nativeoperandgapsand2BOOL24/80fieldreaderidentitygaps;9/1352pendingunchanged. 359branchrecords24semantic,124pairs25groups471pending;2073/285598conditionalreview,849PMU/16phase/3136assignmentgapsremain. NoallmodulelowestMFUornumericallalgorithmceiling/criticalgain/goalclosure. Preparedfiniteoperator-onlyevidenceplanwasnotexecuted;continueindependentremainingoperatorrelationshipsandvalidresourcebounds.')
    report='''本轮只补影响算子融合、拆分重组和理论上限的证据。读取166原有采集文件，没有运行模型、设备测试或新profile，没有修改目标代码和环境。

32份原始数据库均与既有SHA256一致，原始字节合计3278102528。采集关闭record_op_args、_op_attr和profile_memory，但这不代表原生算子属性全部为空：本地投影遗漏了部分COMPUTE_TASK_INFO.attrInfo文本。

已将3200次目标M3/H5120/k6/localE24的DispatchFFNCombine，按相同数据库摘要、属性ID、输入输出shape/dtype及TASK身份，绑定到原生属性：capacity=65536、worldSize=16、is_trans_b=false、weight_nz=true、swiglu_limit=10。其中2560次有root绑定，640次阶段仍未知。其余1272次MC2的属性为空，不能沿用65536。group_name文本相同也不能替代跨卡同次调用的因果身份。

这个结果补齐了3200次执行的profile参数证据，尚未证明实际已加载机器码、工作区写范围或分配生命周期。以记录的C及匹配源码公式计算，单次工作区请求参考为1023993344B；这不是测得的实际占用、可回收峰值或HBM流量，不能乘以层数和迭代数作为节省。

定点源码只回答接收计数与GEMM行数的问题：GetCumsumForMMAIV沿EP source累加，每个local expert各自得到需求量；最后一组向量输出expert_token_nums，发生在后续容量裁剪之前。它不是沿expert维度的前缀和。对非负且无整数溢出的计数，receiver需求A是该向量之和，实际处理行数R=min(A,C)。当前没有采集这些计数值；BOOL[3]active-mask的值同样未知，M3也不能证明3行全部有效。

在常规稠密两次INT8 GEMM的参考算法中，算术量为F=2R(5120×4608+2304×5120)=70778880R OPS。C=65536不是R，不能据此计算实际FLOPs或MFU。3200次对应TASK中位耗时297033ns只能保留为同口径profile参照；在A和匹配物理容量上界未知时，F/(P_upper×T)仍为条件式。该参考算术量也不是允许任意算法、复用或容差时的无条件必要工作下界。全融合路径须重新审查必要工作与I/O，多个可重叠资源界取max，不能把独立算子界和容量申请量相加成时间下界。

9个未决regime没有被强行关掉：7个MemSet共1272次，原始own属性为空、shape/dtype为N/A，数据库没有其目的地址和长度；2个BOOL[24]初始化共80次只有IsStaticKernel:false，仍不知字段、alias及全部消费者。M12的960次和M19的312次MemSet保留父算子几何证据，但不足以判断原生helper是否可删除或融合。此次原始复查确认的是证据边界，没有新增算子执行或把未知工作记为0。

已准备最小补证计划，分别规定同次receiver计数、MemSet操作数和BOOL字段读者需要哪些字段、因果身份、覆盖分支与停止条件。计划没有执行；新的NPU小计数读取也属于诊断采集，不会伪装成读取旧profile。后续继续先用已有证据推进算子关系，只在确实影响结论时申请有界补证。

中央关系仍为124条条件合同、25个联合组、471对待汇入或复核；分支台账359条、24个语义模块。原生2073个regime/285598次执行条件分析，9个regime/1352次执行待关键字段；849条PMU缺失、16张阶段未知图、3136条模块归属歧义未改变。没有完成全模块最低MFU排名，没有完成全部融合和理论上限分析，Goal保持ACTIVE。
'''
    action=('继续原Goal,v98PROGRESS;v97PASS182checks。用户只要operatorfusion/split/reorg/theoreticalceilings,profilefirstthenpinpointcode,noframeworksurvey。'
      '先读raw-operator-summary-v98/PROFILE-FINDINGS/MC2-recordedattrs/capabilityaudit/remainingoperandstatus/ceilingcontract/minimalplan/wholeaudit。'
      '32rawDBhashes3278102528Bmatched;166readonlySSHactual105verified,nofreshmodel/Torch/NPU/service/sourcechange/capture。'
      'Positiveupdate:3200targetM3k6E24MC2exactTASKsprofileC65536,worldSize16,transB0,weightNZ1,swiglu10;2560root+640unbound。1272otherMC2Cunknown,doNOTreuse65536。'
      'v97allCunknown gap supersededonlyinthisscope;CnotA,recordedattrsnotloadedABI/writeextent/allocatorpeakproof。GetCumsumForMMAIVsourceprefixperexpert,lastvectorpreclip;sumvalidnonnegativeexpertcounts=A,processedR=min(A,C). NoactualAoractive-maskvalues. ClassicaltwoGEMMref70778880R_OPSnotuniversalnecessarywork;MFUunknown,Pupper/timevalidphysicalscopeonly。'
      '9/1352pendingunchanged:7MemSet1272emptyownattrs/NAdtypes/noaddresslength;2BOOL24/80onlyIsStaticKernel:false,nofield/allreaderidentity。Recheckknownnegativeevidenceisnotnewregimepromotion。'
      'No newpairs/groups/branches:124/25/471,359/24semantic;2073/285598conditional,849PMU16phase3136ambiguous,0wholeclosed/noallmodulelowestMFU。'
      'Preparedminimaloperator-onlyfiniteevidenceplanNOTexecuted;device-to-hostcounts/newcapturesrequireauthorizedboundedwindow. Do notrepeatparameter-gapsearchalreadyanswered. Continueotherremainingoperatorrelations/theoryorreadpinnedexistingIRonlyifactualindexedsourcecanchangeconclusion。'
      'TaskACTIVE,LoopEVALUATING,GoalACTIVE;verify_raw_operator_evidence_v98mustPASS;neverreruncompletedfinalizer。')
    fusion=deepcopy(read(R/'fusion-scan.json'));fusion['raw_operator_evidence_v98']=dict(evidence=[str(O/n)for n in names],new_pairs=0,new_graph_nodes=0,new_semantic_closures=0,recorded_C_bound_executions=3200,unknown_C_executions=1272,whole_model_exhaustion_proven=False)
    jobs=[dict(kind='profile',status='pass',description='Readonly original166existingcapture32DBs matchedSHA;recover3200MC2ownC/worldSizeattributesandexactTASKbindings,retain1272unknownC;noNPU/model/newcapture.',
      artifacts=[O/n for n in names[:5]]+[N/'original166-readonly-recheck-v98-fixed.json',N/'saved-metadata-config166-v53.json']),
      dict(kind='design-check',status='pass',description='Independentidentity/count/sourceclipchecks;capacitynotGEMMrowsorMFU/peakcredit,9operandfieldregimes1352remainpending;finiteevidenceplanpreparednotexecuted.',
        artifacts=[O/n for n in names[5:]]+[O/'raw-operator-analysis-preflight-v98.json',O/'raw-operator-validation-history-v98.json',O/'whole-scope-requirement-audit-v98.json'])]
    persist_refinement(version=98,previous=97,title='Raw operator attributes and capacity-to-work evidence boundary',
      summary=dict(filename='raw-operator-summary-v98.json',previous_turn_classification='PROGRESS:v97receivercapacity/workspace/sharedlivebudget;182postchecks',
        current_raw_databases=32,raw_hashes_all_match=True,current_MC2_TASKs=4472,profile_recorded_C65536_executions=3200,profile_C_missing_executions=1272,
        recorded_C_root_bound=2560,recorded_C_phase_unbound=640,actual_receiver_counts_known=False,new_target_source_files=0,new_public_documents=0,
        refined_candidates=2,refined_modules=4,new_semantic_families=0,branch_records=359,semantic_modules=24,new_pairs=0,new_groups=0,
        relation_clauses=124,higher_order_groups=25,pending_central_pair_reconciliation=471,scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,
        pending_regime_reconciliation=9,pending_execution_reconciliation=1352,new_native_promotions=0,actual_memory_credit_MFU_or_speedup_proved=False,
        implementation=False,native_test=False,remote_job_status='Readonly SSH saved-filehash/SQLitequeries completedon authorized166(actual105). No inference/model/Torch/NPU/capture/service/sourcechange;remainingworklocal.'),
      changes=['RecoverownrawMC2C/worldSizefor3200exactTASKsfrom32samehashDBs;1272Cremainunknown,supersedev97projectiongaponlyinprovedscope.',
        'Bindsourcepreclipperexpertcounts andprocessedR=min(A,C);separatecapacityfromclassicalGEMMwork,necessarybounds,physicalMFU/peakcredit.',
        'Recheck9/1352nativeoperand/fieldgapswithoutpromotingunknowns;preparefiniteoperator-onlyevidenceplanwithoutnewcapture.',
        'Authoritativestateandcoverageupdatedwithoutinventingnewpairs/branches/regimepromotionsorwholecompletion.'],
      candidate_updates=updates,module_updates=mods,remaining=remaining,action=action,jobs=jobs,report=report,read_first=names+['whole-scope-requirement-audit-v98.json'],
      publication_rows={},gate_updates=dict(overall_complete=False),scripts=[N/'query_saved_raw_profile_v98.py',N/'read_raw_operator_evidence_v98.py',N/'analyze_raw_operator_evidence_v98.py',N/'verify_raw_operator_evidence_v98.py',N/'persist_profile_refinement_v90.py'],
      executor_prefix='raw-operator',main_script=Path(__file__),source_read=False,additional_shared_writes={R/'fusion-scan.json':fusion})

if __name__=='__main__':main()
