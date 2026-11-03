"""Persist typed panel release/resource analysis and correct C1/C2 navigation identities."""
from copy import deepcopy
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import N,O,R,read,write
from persist_profile_refinement_v90 import persist_refinement

def main():
    assert not(O/'REVIEW-CHECKPOINT-v99.json').exists()
    assert read(O/'projection-panel-analysis-preflight-v99.json')['status']=='PASS'
    names=['projection-panel-inputs-v99.json','projection-panel-source-contract-v99.json','projection-panel-native-frontiers-v99.json',
      'projection-panel-work-and-resource-models-v99.json','projection-panel-reduction-and-precision-v99.json','projection-panel-release-DAG-models-v99.json',
      'projection-panel-mode-atom-audit-v99.json','projection-panel-composition-clauses-v99.json']
    packet=read(O/names[-1]);mode=read(O/names[-2]);ids=['F05','F06','F08','F18','F19','F33']
    gap=('RowJmap/queryhidden==globalhidden[J],actualweight/config/rounding/stride/captureandreader/generationbindingsremainconditional. '
      'ActualCANN9.2on-chip/GMhandoff,sharedVector/Cubecapacityandtilepublicationcostarenotproved. '
      'A3separateAIC/AIVGMvaluebridgesremaininthestateddomain;notHBMtraffic. L1fixedBcyclecutsassumemandatorydistinctuncompressedweights/24AIC/onewarmgrant. '
      'DAGserviceexamplesarenotmeasuredfusiongainsorcapacitylowerbounds. F18C2andF33C1normsarenotoneinvocationdespiteoldgenericatom;F19hasmodequalifiedvariants. '
      'Allnativeoperand/PMU/phase/assignmentandwholemodules/lowestMFUgapsremainopen.')
    rel=deepcopy(read(O/'candidate-relation-clauses-v98.json'));rel['clauses']+=deepcopy(packet['clauses'])
    group=next(g for g in rel['higher_order_groups']if g['id']==packet['extend_group'])
    for c in packet['new_members']:
        assert c not in group['members'];group['members'].append(c)
    group['panel_release_refinement_v99']=dict(evidence=names,
      contract='ChooseQA/SWA panel scheduling andC1optionalproducerunionwithindependentRMSgroups. Preservecurrentoverlaporpricewholeoutputbarrier;allowtile-readyinterfacesonlywithactualpublication/ownershipproof. F18isC2alternative,C1normisnotC2poolnorm. Notallgroupmembersoccurinthesameinvocation.',whole_group_closed=False)
    for c in rel['clauses']:
        if {c['a'],c['b']}=={'F18','F19'}:c['mode_refinement_v99']=dict(shared_norm_atom='compress.C2.norm512',no_C1_pool_identity=True,evidence=names)
        if {c['a'],c['b']}=={'F19','F33'}:c['mode_refinement_v99']=dict(shared_norm_atom='compress.C1.norm512',F19_C1_variant_not_original_C2_pool=True,evidence=names)
    rel.update(parent_relation_index='candidate-relation-clauses-v98.json',clause_count=128,selected_pair_count=128,new_pair_clauses=4,new_higher_order_groups=0,overall_complete=False)
    write(O/'candidate-relation-clauses-v99.json',rel)
    cov=deepcopy(read(O/'candidate-relation-coverage-v98.json'))
    for c in packet['clauses']:
        row=next(r for r in cov['pairs']if{r['a'],r['b']}=={c['a'],c['b']});assert not row['clauses']
        row.update(scope_state='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL',clauses=[c['id']],whole_pair_all_variants_closed=False,new_critical_path_gain_us=None)
    cov.update(parent='candidate-relation-coverage-v98.json',parent_relation_index='candidate-relation-clauses-v99.json',scoped_conditional_pairs=128,remaining_pair_relations=467,higher_order_groups=25,
      scope='128scopedconditionalpairs/25groups/35modelcandidates;467centralpairreconciliationsandallvariantgatesremain. Modequalifiednavigationatomsarenotnewnativeoperators.',
      projection_panel_refinement_v99=dict(new_pairs=4,new_groups=0,extended_group=packet['extend_group'],evidence=names),overall_complete=False)
    write(O/'candidate-relation-coverage-v99.json',cov)
    fps=deepcopy(read(O/'candidate-operator-footprints-v98.json'));ix={r['id']:r for r in read(O/'candidate-record-index-v98.json')['model_candidates']};modes={r['candidate']:r for r in mode['records']}
    for r in fps['candidates']:
        p=R/'candidates'/(r['id']+'.json');h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==ix[r['id']]['master_sha256'];r['captured_master_sha256']=h;r['captured_master_version']=98
        if r['id']in modes:
            row=modes[r['id']];r['operator_atom_union']=[a for a in r['operator_atom_union']if a!=row['previous_generic_atom']]+row['replacement_norm_atoms']
            r['projection_panel_mode_refinement_v99']=dict(evidence=[names[1],names[6]],replaces_generic_norm_only=True,no_new_native_operator_nodes=True)
        if r['id']in ids:r['projection_panel_refinement_v99']=dict(evidence=names,all_runtime_gates_proved=False)
    fps.update(parent='candidate-operator-footprints-v98.json',overall_complete=False);write(O/'candidate-operator-footprints-v99.json',fps);fpi={r['id']:r for r in fps['candidates']}
    nav=deepcopy(read(O/'candidate-pair-navigation-v98.json'))
    for row in nav['pairs']:
        row['shared_operator_atom_union']=sorted(set(fpi[row['a']]['operator_atom_union'])&set(fpi[row['b']]['operator_atom_union']))
        row['conditional_clause_ids']=[c['id']for c in rel['clauses']if{c['a'],c['b']}=={row['a'],row['b']}]
        row['semantic_pair_reconciliation_done_in_stated_scope']=bool(row['conditional_clause_ids'])
    nav.update(parent='candidate-pair-navigation-v98.json',parent_footprints='candidate-operator-footprints-v99.json',parent_relation_index='candidate-relation-clauses-v99.json',scoped_conditional_pairs=128,remaining_central_pair_reconciliation=467,
      projection_panel_refinement_v99=dict(mode_qualified_norm_atoms=3,recomputed_pair_intersections=595,new_pairs=4,evidence=names),overall_complete=False)
    write(O/'candidate-pair-navigation-v99.json',nav)
    op=deepcopy(read(O/'operator-analysis-coverage-ledger-v98.json'));op['projection_panel_refinement_v99']=dict(saved_frontiers=3200,reused_TASKs=19200,root_unique_frontiers=2496,
      mode_qualified_navigation_labels_not_observed_regimes=True,new_regimes_or_native_attributions=0,new_conditional_relations=4,new_groups=0)
    write(O/'operator-analysis-coverage-ledger-v99.json',op)
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v98.json'));audit.update(previous_goal_turn='PROGRESS:v98rawsamehashownMC2attributes3200andremainingoperandgaps;229postchecks',input_checkpoint=98,followup_checkpoint=99)
    for r in audit['requirements']:
        if r['id']=='CROSS_OPERATOR_LAYER_CARD_ITERATION':r.update(conditional_pair_relations=128,higher_order_groups=25,central_pairs_not_reconciled=467)
        if r['id']=='THEORY_AND_MEASURED_COMPARISON':r['evidence']+=names
    audit['projection_panel_refinement_v99']=dict(independentRMSgroups=True,profile_overlap_reconciled=True,runtime_gain_proved=False,
      generic_C1_C2_norm_identity_corrected=True,new_whole_requirement_closure=False)
    write(O/'whole-scope-requirement-audit-v99.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v99.json',dict(status='SAVED_TYPED_FRONTIERS_THEN_ONLY_PROJECTION_NORMALIZATION_MODE_AND_RELEASE_SOURCE',
      questions=['Doeswhole-outputjointprojectiondelayQnormwhichalreadyoverlapsKV?',
        'CanlocalQA3/globalKV24/C124shareoneproducerwithoutcomputingunusedQrows?',
        'WhichcompleteK/NrowstatisticsandBF16roundingarepreserved?',
        'Isgenericlatent.norm512reallyoneoperatoracrossF18C2andF33C1?'],
      cached_source_files=2,new_target_source_files=0,new_public_documents=0,no_framework_wide_research=True,no_server_model_Torch_NPU_capture_or_implementation=True,overall_complete=False))
    write(O/'projection-panel-validation-history-v99.json',dict(refinements=[
      'Priorv98PROGRESSverified229checks;32rawDBpositiveMC2evidenceisretained,notrequeriedorclaimednew.',
      'All3200savedQA/SWAfrontiersand19200TASKsreconciled,with2496rootuniquerolesand80source20ambiguousroleskept;v27source20overlapmedianscrosschecked.',
      'M3vsM24naive8/3workfactwasalreadyv61. NewC1triple/raggedunion/releaseboundskeepSWAonceanduseoneL0Bwarmgrant,notonepercandidate.',
      'OriginalF18masterisATTN_C2_SOURCEring_pool→compress_norm;genericnormatomdidnotproveC1samework. SplitF18C2,F33C1,F19bothmodevariantsandrecomputeall595navigationintersections.',
      'PartialKnormslosecross terms;completeKBF16Ntilemergesrequiretherealreduction/precisioncontract. NojointRMS1792/2304andnounqualifiedRMSscalarcancellation.',
      '32768DAGcasesincludezeroboundaries;explanatorybetter/equal/worsewitnessesuseonlypositiveservices. DAGfeasibilityisnotnativecapacityoractualgain.',
      'AIC→AIVGMvaluebridgeandINT8V→CGMbridgeareseparateinterfacecutsfromqr→quantcredit;GMvaluebytesarenotHBMtraffic. 24AICfixedBnumericcyclesarenotactualfrequency-basedtimelowerbounds.'],
      all_local_jobs_finished=True,overall_complete=False))
    updates={cid:dict(review_v99=dict(evidence=names,graph_frontiers=3200,strict_root_frontiers=2496,reused_native_TASKs=19200,
      new_relation_ids=[c['id']for c in packet['clauses']if cid in[c['a'],c['b']]],extended_group=packet['extend_group'],
      mode_qualified_norms=cid in modes,whole_output_barrier_not_universal=True,ragged_resource_models=2,DAG_cases=32768,implementation=False,native_test=False),conclusion_changing_gap=gap)for cid in ids}
    mods={
      'M07':dict(candidates=['F05','F06','F08','F33'],finding='All3200QA/SWAfrontiersshowQnormandquantoverlappingKVandQBoverlappingKVnorm. Jointproducerrequiresrowdomain/raggedpanelsandindependent1280/512RMS;whole-outputbarriercanloseoverlap. Oneunionwork/weights/GMbridgeaccounted.',evidence=names,gap=gap),
      'M08':dict(candidates=['F18','F19','F33'],finding='OriginalF18isC2pool→norm,F33C1directprojection→norm,F19forkhasbothmodevariants. Genericlatent.norm512falseidentitycorrected;C1triplehasthreedistinctprojection/normgroupsandretainedindex/longKVreaders.',evidence=names,gap=gap),
      'M23':dict(candidates=['F06','F33'],finding='QA3localandSWA/C124globalproducerunionsrequireactualrowJmapandrankweights/generations. TypedweightfixedBcuts5462/7168cycleswithonewarmgrantaredeclared-domainreferences;noactualwire/HBM/timegainorphysicalMFU.',evidence=names,gap=gap)}
    remaining=list(read(O/'module-coverage-gaps-v98.json')['remaining']);remaining.append('v99: QA/SWA/C1rowpanelreleaseand4centralpairsreviewed;genericF18C2/F33C1normfalseidentitycorrected,F19bothqualifiedmodes. 3200savedfrontiers19200TASKs,2496rootunique80source20ambiguouskept. Actualrowmap/gamma/eps/round/strides/weight/capture/GMpublication/sharedcapacityandallreadersremainconditional. FixedB24AIConeL0Bwarmcut5462/7168cyclesnotnativewallclock;classicalwork165150720/290979840OPS,wholeglobalconcat440401920/566231040OPS;GMbridges64512/113664valueBnotHBM. 32768idealDAGcasesnotactualgains. 128pairs25groups467pending,359branches24semantic,2073/285598conditional9/1352pending849/16/3136uncertaintyunchanged;noallmodulelowestMFUorwholeclosure.')
    report='''本轮围绕Q_a、SWA KV、C1投影与归一化的组合，补了分块就绪、必要工作和存储层边界。全部使用已有profile、冻结源码和离线数学，没有运行设备或修改目标实现。

复核3200组投影前沿、19200条既有TASK：每组都存在Q侧norm/quant与KV矩阵计算的重叠，以及Q_B与KV norm的重叠。严格root且SWA角色唯一的2496组中，Qnorm与KV计算的重叠中位数4500ns，Q_B与KVnorm为11400.5ns。其余640组阶段未知、layer20的80组SWA/C1角色歧义继续保留。重叠区间不是可回收请求延迟。

联合投影需要保留不同的行域和归一化域。当前例子是Q_a的3行×1280列、SWA的24行×512列；C1 source另有24行×512列。若用24行整块GEMM强行统一行域，QA/SWA算术量从165150720变为440401920 OPS，这个8/3问题此前v61已经指出。本轮将C1纳入同一union：原本需要290979840 OPS，整块2304列则变为566231040 OPS，比例72/37。可登记按行集合选择输出panel的候选，但必须证明query_hidden与global_hidden[J]、权重、行序和数值边界相同，不能把它当成现成的普通矩阵拼接。

SWA在F06与F33中是同一个生产者，联合工作和权重只计一次。三个独立输入读取的逻辑参考是522240B；已有SWA/C1联合后为276480B，再合入QA的边际输入读取差只有30720B；全部共享输入参考245760B。这些是明确基线下的逻辑读取差，不是实际HBM或时间收益。

在24AIC、稠密BF16、固定B经L1→L0、所有命名权重必读且不压缩/剪枝/去重、无导入计算结果的实现域，整个窗口只给一次初始L0B容量信用。QA/SWA和QA/SWA/C1的权重分别18350080B、23592960B；对应固定B服务下界5462、7168 cycles，常规有用算术界840、1480 cycles。这两个资源界取max。它们不是当前原生路径或物理频率下的时间下界，也没有证明所有合法算法的上限。

归一化依赖必须单独保留：Q的1280、SWA的512、C1的512各自有参数、epsilon和完整行统计；不能改成一个1792/2304宽RMS。部分K的结果平方和遗漏交叉项；完成K并按原语义舍入BF16后，才可在精度合同允许时合并N分块统计。Q的BF16 norm输出到DynamicQuant又有完整行amax依赖，v94的舍入反例继续约束融合。

A3分离AIC/AIV实现域内，投影结果仍需经GM交给Vector norm。两种投影union的BF16写+读值预算分别64512B、113664B，不能仅因放进一个kernel就全部删除；它们也不自动等于HBM流量。qr→quant的既有v94信用、INT8 Vector→Cube的保留GM接口另列，避免重复计算。

就绪合同决定融合是否有利。按相同服务时间的理想DAG，保留旧stage join并增加整个联合投影的输出屏障，会损失min(KV时间,Qnorm/quant时间)的重叠；若同时提前释放KV norm，则可能变好、相等或变差。32768个有限模型都保留这个区别。逐行/panel发布可以避免不必要的跨分支等待，但实际发布、同步、alias和有限Vector/Cube资源代价还未证明，因此没有把这个模型当成可达提速。

本轮还纠正了一个台账身份问题：F18原候选是C2 pool→norm，F33是C1投影→norm，原先共用compress.latent.norm512并不证明它们是同次算子。现分别使用C2.norm512和C1.norm512；F19保留两种模式的fork变体，并重算全部595对导航交集。C1/C2可出现在模型不同层，不能声称整个模型二选一，也不能将C2池化附加到C1。

中央新增4条条件关系，并扩展既有Q/KV join组，累计128条关系、25组联合合同，467对待汇入或复核。分支仍359条、24个语义模块；2073个regime/285598次执行条件分析，9个regime/1352次待关键字段，849 PMU/16阶段图/3136归属歧义未变。全模块最低MFU、所有融合变体和整体理论上限仍未完成。
'''
    action=('继续原Goal,v99PROGRESS;v98PASS229checks。用户operatorfusion/split/reorg/theoreticalceilings,profilefirstthenpinpointcode,noframeworksurvey。'
      '先读projection-panel-summary-v99/PROFILE-FINDINGS/source/native/work-resource/reduction/DAG/mode-atom/composition/wholeaudit。'
      '3200savedfrontiers19200TASKs,rootunique2496,allQN/quant-KVandQB-KVNormoverlappositive;source20ambiguous80andphase640unchanged。'
      'v61QA3vsKV24globalconcat8/3notnew;C1triplework290979840vs566231040ratio72/37,threeindependentnorm1280/512/512。RaggedJmapandactualrow/weight/precision/captureproofconditional。'
      'FixedB24AICmandatoryuncompressedweightsL1→L0oneinitialL0Bgrant=>5462/7168cycles;usefulBF16compute840/1480,maxnotadd,notnativephysicaltime. GMbridge64512/113664logicalvalueBretainedforAIC→AIV,notsavingHBM。'
      'Whole-outputjoincanloseexistingoverlap;ifremoveotherstagecontrolgateitmayhelporevenhurt32768DAGcases;notfeasiblespeedup/capacityLB. NeedcompleteKroundedoutputbeforeNstat;partialKnormcross-terms,independentnormgroups,v94quantprecisionremain。'
      'Correctgenericnormidentity:F18originalC2poolnormonly,F33C1directnormonly,F19bothmodeforkvariants. Candidatefootprints3andall595navintersectionsupdated;notnewobservedopsor3136physicalassignmentpromotion。'
      '4newREL99pairsF05F06/F06F08/F06F33/F18F33,HG75-Q-KV-JOINextendedF05F18F33,128/25/467;359/24semantic,2073/285598conditional9/1352pending849PMU16phase3136ambiguousunchanged,0wholeclosed/noallmodulelowestMFU。'
      'v98profileC65536world16bound3200MC2,other1272unknownnotrequeried;minimalnativeoperand/receiverplanpreparednotexecuted,noauthorizednewcapturethisturn。'
      'NexthighvalueknownremaininginterfacesF08F14/F12F14/F14F19:profileattentioninputreadinessandattention→inverseRoPE→WO_Aoutputprojection/headgroups;assessreordering/partialattentionmerge/weights/read-footprintwithoutassuminglinearprojectioncommuteswithBF16softmax/rope/headmixing. Onlypointsourceforactualoperatoralgebra. Continuebroaderfinitebranch/numericcapacitygapswithoutreplayingcompletedquestions。'
      'Alljobsfinishedlocal,noserver/model/Torch/NPU/capture/service/implementation. GoalACTIVE,TaskACTIVE,LoopEVALUATING;verify_projection_panel_release_v99mustPASS;neverreruncompletedfinalizer。')
    fusion=deepcopy(read(R/'fusion-scan.json'));fusion['projection_panel_refinement_v99']=dict(evidence=[str(O/n)for n in names],new_pairs=4,new_groups=0,new_native_operator_nodes=0,
      navigation_mode_label_correction=['F18:C2norm','F33:C1norm','F19:C1/C2variants'],whole_model_exhaustion_proven=False)
    jobinputs=[Path(x['path'])for x in read(O/names[0])['inputs']];sources=[Path(x['path'])for x in read(O/names[1])['parts']]
    jobs=[dict(kind='profile',status='pass',description='Reconcile3200savedQA/SWAfrontiers19200exactTASKs,existingoverlapsandsource20/phaseuncertainty;C1/C2mode-identityaudittargetedsourceonly.',
      artifacts=[O/n for n in names]+jobinputs+sources),dict(kind='design-check',status='pass',description='Raggedinput/productunionanddeclaredfixedB/GMbridgecuts;partialKnormcounterexampleand32768idealDAGcases;4conditionalrelations,nophysicalgainclaim.',
        artifacts=[O/'projection-panel-analysis-preflight-v99.json',O/'projection-panel-validation-history-v99.json',O/'whole-scope-requirement-audit-v99.json'])]
    persist_refinement(version=99,previous=98,title='Ragged projection panels, independent RMS groups and release constraints',
      summary=dict(filename='projection-panel-summary-v99.json',previous_turn_classification='PROGRESS:v98rawsamehashMC2ownattributesandoperandgaps;229postchecks',
        saved_frontiers=3200,reused_native_TASKs=19200,strict_root_unique_SWA=2496,ambiguous_source20_roles=80,phase_unbound=640,
        ragged_resource_models=2,ideal_DAG_models=32768,mode_qualified_candidate_footprints=3,recomputed_pair_navigation_intersections=595,
        new_target_source_files=0,new_public_documents=0,refined_candidates=6,refined_modules=3,new_semantic_families=0,branch_records=359,semantic_modules=24,
        new_pairs=4,new_groups=0,extended_group=packet['extend_group'],relation_clauses=128,higher_order_groups=25,pending_central_pair_reconciliation=467,
        scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,new_native_promotions=0,
        actual_MFU_physical_time_or_speedup_proved=False,implementation=False,native_test=False,remote_job_status='Alllocalexistingprofile/pinpointcachedoperatorsource/mathworkfinished;no server/model/Torch/NPU/capture/serviceorimplementation.'),
      changes=['Reconcile3200QA/SWAfrontiersandpreservedoverlap;whole-outputjoinandpanelreleasearenotthesamefusiongraph.',
        'AddC1raggedproducerunion/independentRMSreductions andsharedone-warmgrantresource/GMbridgecuts;notHBMtimeoractualphysicalceiling.',
        'CorrectF18C2/F33C1genericnormfalseidentityandF19modevariants;recompute595navigationintersectionswithoutnativeattributionpromotion.',
        '4conditionalpairclausesandoneexistinggroupextension=>128/25/467;fullgoal,moduleMFUandcriticalevidencegapsremainopen.'],
      candidate_updates=updates,module_updates=mods,remaining=remaining,action=action,jobs=jobs,report=report,read_first=names+['whole-scope-requirement-audit-v99.json'],
      publication_rows={},gate_updates=dict(overall_complete=False),scripts=[N/'analyze_projection_panel_release_v99.py',N/'verify_projection_panel_release_v99.py',N/'persist_profile_refinement_v90.py'],
      executor_prefix='projection-panel',main_script=Path(__file__),source_read=False,additional_shared_writes={R/'fusion-scan.json':fusion})

if __name__=='__main__':main()
