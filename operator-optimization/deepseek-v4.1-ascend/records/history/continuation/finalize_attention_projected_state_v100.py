"""Persist attention/output owner identities, headwise merge limitations and preprojected-value costs."""
from copy import deepcopy
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import N,O,R,read,write
from persist_profile_refinement_v90 import persist_refinement

def main():
    assert not(O/'REVIEW-CHECKPOINT-v100.json').exists()
    assert read(O/'attention-projected-state-analysis-preflight-v100.json')['status']=='PASS'
    names=['attention-projected-state-inputs-v100.json','attention-projected-state-source-contract-v100.json','attention-output-owner-frontiers-v100.json',
      'attention-headwise-merge-and-linear-state-v100.json','attention-projection-rounding-and-position-v100.json',
      'attention-preprojected-value-work-and-storage-v100.json','attention-projected-state-variants-v100.json','attention-projected-state-composition-clauses-v100.json']
    packet=read(O/names[-1]);ids=['F08','F12','F13','F14','F16','F19']
    gap=('CurrentBF16attention/inverseRoPE/OArounds,actualheadstatereachability/OAblockranks/structuredcommutationandfixedcheckpointvaluesremainunbound. '
      'PartialstateAPI/sink/empty/NaN/Inf/mergeprecision,exactselectedrow/cache/position/epoch/readersandphysicalquery/headownershipmustbind. '
      'PreprojectedVbankmodelsarechosenBF16/FLOATrepresentationswithbuild/liveness/weightplacementcosts,notuniversalmandatorymemoryoractualspeedup. '
      'ProjectedFLOATPVsupport/typedcapacityandA3GM/communicationhandoffsarenotfree;source20KVsharingdoesnotproveoutputweightsharing. '
      'Noall-moduleMFU,numericphysicalwholepathceilingorwholegoalclosure.')
    rel=deepcopy(read(O/'candidate-relation-clauses-v99.json'));rel['clauses']+=deepcopy(packet['clauses'])
    group=next(g for g in rel['higher_order_groups']if g['id']==packet['extend_group'])
    for c in packet['new_members']:assert c not in group['members'];group['members'].append(c)
    group['projected_head_state_refinement_v100']=dict(evidence=names,
      contract='SWA/compressedselectionthroughheadwiseattentionstatetoinverseRoPE/restore/OA. Retainm/l/headnumeratorsuntilhead-specificnormalizationorproveoutputspacecommutation. BF16round/queryposition/weights/cacheepochsandphysicalpartitionremain. PreprojectedVchangesrepresentation/work,notfreecacheCSE.',whole_group_closed=False)
    for c in rel['clauses']:
        if {c['a'],c['b']}=={'F13','F14'}:c['projected_head_state_refinement_v100']=dict(evidence=names,real_linearity_not_native_round_equivalence=True,head_collapse_not_generic_state_merge=True)
        if {c['a'],c['b']}=={'F14','F16'}:c['projected_head_state_refinement_v100']=dict(evidence=names,OA_counted_once=True,preV_not_same_as_OA_WOB_precomposition=True,placement_and_rounding_gates_open=True)
    rel.update(parent_relation_index='candidate-relation-clauses-v99.json',clause_count=131,selected_pair_count=131,new_pair_clauses=3,new_higher_order_groups=0,overall_complete=False)
    write(O/'candidate-relation-clauses-v100.json',rel)
    cov=deepcopy(read(O/'candidate-relation-coverage-v99.json'))
    for c in packet['clauses']:
        row=next(r for r in cov['pairs']if{r['a'],r['b']}=={c['a'],c['b']});assert not row['clauses']
        row.update(scope_state='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL',clauses=[c['id']],whole_pair_all_variants_closed=False,new_critical_path_gain_us=None)
    cov.update(parent='candidate-relation-coverage-v99.json',parent_relation_index='candidate-relation-clauses-v100.json',scoped_conditional_pairs=131,remaining_pair_relations=464,higher_order_groups=25,
      scope='131scopedconditionalpairs/25groups/35modelcandidates;464centralpairreconciliationsandfullvariant/fullmodelgatesremain. Historicaltemplateinterfaceclausesarenotallnativegraphbranches.',
      projected_attention_state_refinement_v100=dict(new_pairs=3,new_groups=0,extended_group=packet['extend_group'],evidence=names),overall_complete=False)
    write(O/'candidate-relation-coverage-v100.json',cov)
    fp=deepcopy(read(O/'candidate-operator-footprints-v99.json'));ix={r['id']:r for r in read(O/'candidate-record-index-v99.json')['model_candidates']}
    for r in fp['candidates']:
        p=R/'candidates'/(r['id']+'.json');h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==ix[r['id']]['master_sha256'];r['captured_master_sha256']=h;r['captured_master_version']=99
        if r['id']in ids:r['projected_attention_state_refinement_v100']=dict(evidence=names,hypothetical_value_banks_not_observed_operator_atoms=True,all_runtime_gates_proved=False)
    fp.update(parent='candidate-operator-footprints-v99.json',overall_complete=False);write(O/'candidate-operator-footprints-v100.json',fp)
    nav=deepcopy(read(O/'candidate-pair-navigation-v99.json'))
    for row in nav['pairs']:
        row['conditional_clause_ids']=[c['id']for c in rel['clauses']if{c['a'],c['b']}=={row['a'],row['b']}]
        row['semantic_pair_reconciliation_done_in_stated_scope']=bool(row['conditional_clause_ids'])
    nav.update(parent='candidate-pair-navigation-v99.json',parent_footprints='candidate-operator-footprints-v100.json',parent_relation_index='candidate-relation-clauses-v100.json',scoped_conditional_pairs=131,remaining_central_pair_reconciliation=464,
      projected_attention_state_refinement_v100=dict(new_pairs=3,evidence=names,all_current_model_interfaces_closed=False),overall_complete=False)
    write(O/'candidate-pair-navigation-v100.json',nav)
    op=deepcopy(read(O/'operator-analysis-coverage-ledger-v99.json'));op['projected_attention_state_refinement_v100']=dict(saved_frontiers=3200,reused_TASKs=23680,query_selection_frontiers=640,
      family_frontiers=dict(SWA_ONLY=160,CSA_C2=1440,CSA_C1=1600),new_regimes_or_native_attributions=0,new_conditional_relations=3,new_groups=0)
    write(O/'operator-analysis-coverage-ledger-v100.json',op)
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v99.json'));audit.update(previous_goal_turn='PROGRESS:v99projectionpanels/typedcuts/C1C2identitycorrection;200postchecks',input_checkpoint=99,followup_checkpoint=100)
    for r in audit['requirements']:
        if r['id']=='CROSS_OPERATOR_LAYER_CARD_ITERATION':r.update(conditional_pair_relations=131,higher_order_groups=25,central_pairs_not_reconciled=464)
        if r['id']=='THEORY_AND_MEASURED_COMPARISON':r['evidence']+=names
    audit['projected_attention_state_refinement_v100']=dict(headwise_merge_and_BF16_counterexamples=True,
      actual_fixed_model_rank_or_head_independence_proved=False,preprojection_legal_native_gain_proved=False,new_whole_requirement_closure=False)
    write(O/'whole-scope-requirement-audit-v100.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v100.json',dict(status='SAVED_SMLA_TO_OA_OWNER_FRONTIERS_THEN_ONLY_OPERATOR_STATE_ROUND_AND_RESTORE_CONTRACTS',
      questions=['CanSWAandcompressedpartialattentionbeprojectedandmixedbeforeheadwisemassesareknown?',
        'Doesreal-linearitypreserveoriginalBF16attention/inverseRoPEboundaries?',
        'Whatwork/storage/position/layercostdoesprojectedVrequire?',
        'DoobservedM3H64andM24H8operatorshavethesamephysicalrow/headowner?'],
      cached_source_files=6,new_target_source_files=0,new_public_documents=0,no_framework_wide_research=True,no_server_model_Torch_NPU_capture_or_implementation=True,overall_complete=False))
    write(O/'attention-projected-state-validation-history-v100.json',dict(refinements=[
      'v99PROGRESSrevalidated200checks;C1/C2qualifiednormidentitiesremainunchanged,nooldnegativegapre-query.',
      'Bind3200existingSMLA/inverseRoPE/pack/A2A/OA/OB/RSfrontiersplus640QLI/preparecontexts23680TASKs. M3H64vsM24H8equalcardinalitydoesnotproveowneridentityorallrankcausaljoin.',
      'Projectionacrossnormalizedpartialattentionlosesheadidentityevenwithperheadmasses;real-linearstate/rankconditionisconditionalonindependentstatefamily,notassertedactualcheckpointreachability.',
      'BF16RNEwitnesswith3zero-logitkeysandonezero-valuedzero-logitsinkyields3vs3.015625afterunrotatedscalarW3;independentFractionrounderchecksmodel,nonnative.',
      'CurrentemptyLSEoutputisnotavailablebranchstats. SplitattentionrequiresnewstateABI,stablefinite/empty/sink/roundingcontract;notcompositionofreturnedBF16outputs.',
      'U_newbuildworkandU_liveresidentbanksareseparate. IncludeBF16andFLOATprojectedbanks;FLOATmixedPVsupport/rateunknown,notpricedatBF16capacity.',
      'Sharedsource20KVwith20consumerlayersdoesnotproveprojectorweightsharing. ChosenheadexpandedbanksretainoriginalKforQK,canbedifferentfromcompressed/recomputed/on-demandrepresentations.',
      'Priorv83outputinterfacecreditandpriorv61OA-beforeA2A/WOprecompositionfactsarenotrecountedasnewoptimizations. NewPV-push/stateanalysisisanalgorithmconditionaldomain,notmeasuredgain.'],
      all_local_jobs_finished=True,overall_complete=False))
    updates={cid:dict(review_v100=dict(evidence=names,frontiers=3200,reused_native_TASKs=23680,selection_frontiers=640,
      headwise_state_and_precision_counterexamples=True,actual_checkpoint_rank_or_state_reachability_known=False,preprojected_value_work_models=105,preprojected_value_storage_models=24,
      new_relation_ids=[c['id']for c in packet['clauses']if cid in[c['a'],c['b']]],extended_group=packet['extend_group'],
      implementation=False,native_test=False,native_gain_proved=False),conclusion_changing_gap=gap)for cid in ids}
    mods={
      'M08':dict(candidates=['F19'],finding='CompressedKVlatentpublicationstillfeedsunchangedQKandfutureconsumers. AnOA-projectedVbankisdifferenthead/layer/query-positionrepresentation;source20sharedKVdoesnotprovesharedOAweights.',evidence=names,gap=gap),
      'M09':dict(candidates=['F12'],finding='640QLI/preparefrontierslinktotheexactSMLAconsumer. SelectedIDsarenotattentionheadmasses;earlySWA/compressedcompositionrequiresnewheadwise-stateABI,notoneprojectedbranchvector.',evidence=names,gap=gap),
      'M10':dict(candidates=['F08','F13','F14','F19'],finding='3200SMLAoutputfrontiersfixedBF16/inverseRoPE/pack/restore/OArowdomains. Evenknownperheadmassesdonotrepairhead-collapsedpartialprojection;preserveheadstateorprovestructuredcommutationandoriginalrounding.',evidence=names,gap=gap),
      'M11':dict(candidates=['F14','F16'],finding='OAgroupmixes8×512to1024. PushingWthroughPVhasexactfixed-queryrealidentitybutBF16counterexample3vs3.015625;queryinverseRoPEchangesW_eff. 105work/24storagecasesexposeheadexpandedVbuild/PVcosts,notactualspeedup.',evidence=names,gap=gap),
      'M23':dict(candidates=['F14','F16'],finding='RecordedattentionM3H64→send[8,3,8,512]→A2A→OAM24H8hasdifferentphysicalownership. HypotheticalM24H8attention/headownerandM3H64queryownerbanksneednewquery/weight/communicationlayout;equalvolumeisnotfreeonchipfusion.',evidence=names,gap=gap),
      'M24':dict(candidates=['F19'],finding='CacheU_newbuildcost,U_livelifetime,head/layerprojectoridentityandquerypositionareindependentreusekeys. 20source20consumersarenotoneprojected-valuegeneration;retainoriginalKandpublicationepochs.',evidence=names,gap=gap)}
    remaining=list(read(O/'module-coverage-gaps-v99.json')['remaining']);remaining.append('v100: Attention→OAstateboundary3pairsreviewed;131pairs25groups464centralpending,359branches24semanticunchanged. 3200savedfrontiers23680TASKs640QLIcontext,root2560/phase640. Headcollapsedpartialprojectionscannotgenericallymergeevenwithheadmasses;actualWblockranks/headstatereachabilityunknown. Real-linearitydoesnotpreserveBF16attentionround(witness3vs3.015625),queryinverseRoPEmakesW_effpositiondependent. 105logicalwork/24BF16-FLOATstoragecasesnotactualnativegainoruniversalIO;unchangedQKretainsK,source20sharedKV≠20consumerOAweights. PhysicalM3H64→M24H8ownership/capture/alias/rowepoch/link/typedrate/stateABI/empty/sink/nonfinite/precisiongatesremain. 2073/285598conditional9/1352pending849PMU16phase3136ambiguousunchanged;0wholeclosed/noallmodulelowestMFU.')
    report='''本轮分析attention能否与输出投影重组，范围只涉及算子状态、舍入、head/行域和必要工作。使用已有profile、冻结源码和离线模型，没有运行设备或修改目标实现。

已对齐3200组SMLA→逆RoPE→发送布局→all-to-all→WO_a→WO_b→reduce-scatter，共23680条既有TASK，并关联640组QLI/prepare。当前attention输出是[3,64,512] BF16；all-to-all后的WO_a输入是[24,1,4096]，权重[1,4096,1024]。因此一个投影group包含8个512维head、输出1024维。两侧元素总量相同不等于行/head所有权相同；前移投影或attention都需要重新核对权重、query、通信与就绪。2560组root和640组阶段未知的边界保持不变。

拆分SWA和压缩attention时，不能直接把两个已投影并混合head的分支向量按一个比例相加。各head的softmax质量不同，即使保留了每个head的LSE/质量，混合后的向量也可能丢失恢复信息。一个已分别归一化的两head反例中，两种情况的分支投影和都为[1,1]，head质量也相同，正确合并结果却分别为5/4和3/4。另一个未归一化分子反例保留了完全相同的逐head m/l，分支投影和仍相同，最终结果却为3/4和7/8。可保留每个head的m/l/分子，待质量确定后再混合head；也可另证输出空间与head缩放的特殊可交换结构。等待所有质量已知后再投影、相同head缩放等特殊情形没有被排除。

在实数线性状态编码、head状态可独立变化的域，若8个投影块各rank512，为支持任意head缩放需要保留4096个线性自由度，不能只保留一个1024维混合状态。实际checkpoint的块rank、head状态可达性和相关性尚未知，因此没有把这个条件命题写成当前固定模型的普适存储下界。当前API还设置return_softmax_lse=False，返回的FLOAT张量为空；新的split/merge必须显式增加状态合同，不能用已有BF16输出冒充未舍入的分支状态。

下面是保留原输出精度的一种候选依赖图；它尚未实现，分支合并也需原生精度复核：

```mermaid
flowchart LR
  S[SWA就绪] --> A[逐head分支状态]
  I[QLI与索引准备] --> C[压缩KV分支状态]
  A --> M[逐head合并与归一化]
  C --> M
  K[只计一次的sink] --> M
  M --> B[原attention BF16舍入]
  B --> P[逆RoPE与BF16舍入]
  P --> R[布局与head恢复]
  R --> O[WO_a投影]
```

把WO_a前移到PV之前，在固定query位置、精确实数下可以分配线性变换，但原生精度并不自动等价。BF16 RNE模型中，3个零logit key的value为1、1、2.015625，另加一个零logit、零value的sink。attention未舍入值为257/256，先舍入再乘BF16权重3得到3；先投影value再做加权，保留FP32或新增BF16中间舍入都会得到3.015625。该反例放在不旋转的448维范围即可成立，没有运行原生kernel，也未声称这些就是模型实际输入。

逆RoPE另有位置限制：前移后的矩阵为R_inv(query_position)·W_head。普通按key位置缓存的一个固定投影值不能自动服务未来不同query位置；key RoPE的位置也不能代替query逆RoPE的位置。只前移448个非旋转维度是另一个条件算法，仍保留64维tail、逆RoPE和投影，且不能绕过原attention BF16舍入。

工作量也要重算。以一个8-head投影group、D=512、R=1024和每query参考N个key为例，提前投影会把后续PV累积宽度从512增至1024，还增加U_new个key的投影构建工作。只比较常规稠密PV+OA，前移参考算术量更低需要U_new<M·(1−N/1024)；N≥1024时，这个模型即使缓存理想预热也无严格算术优势。只前移非旋转部分，PV加tail的宽度为1088，即原来的17/8。105组参考模型保留了可摊薄与不能摊薄的情况；它们不是实际有效key计数，也没有通过精度与所有权门槛。

一种按head保存投影后V的表示还会显著增加缓存。保持原K供未改写QK使用时，一个8-head owner、一个consumer的BF16投影bank新增量是原共享512维K平面的16倍，总量17倍；FLOAT投影bank则总量33倍。source20的KV被20层读取，但这些层的OA矩阵不因此相同；在20组不同投影的参考域，总量分别为321倍、641倍。若沿当前64-head query owner保存所有group，还需额外扩展。24组存储模型分别列出BF16/FLOAT、owner和consumer数量；这只是所选表示的逻辑值预算，压缩、低rank、按需保留或重算仍是其他域。FLOAT PV的混合类型支持与吞吐也未获得证明，不能沿用BF16峰值。

另一个拆分候选是先完成逐head QK与统计，再做PV。它可以使后续计算提前知道质量，但需要保存完整分数/概率及原tile状态序列，或付出QK重算。M个query、H个head、每query N个key时，完整FP32分数值预算为4MHN字节，BF16概率为2MHN字节，另一次QK参考算术为2MHDN OPS；这些是不同保留/重算方案，不能相加当额外HBM下界。当前tile队列不等于已有全N持久缓冲。全局归一化后直接舍入BF16 P也不自动等于当前在线exp/缩放/PV序列，仍需重放原序列或证明新精度；知道质量后，最终attention BF16和OA顺序门槛仍在。

因此当前可登记的结论是：保留原舍入的输出epilogue、增加逐head状态的split/merge、统计先行PV，以及提前投影V是不同候选。它们必须分别支付新状态、保留/重算、缓存/构建、位置/权重、通信和精度合同，不能合并记作免费消除attention输出或KV读写。v83已有输出布局信用和v61已有OA前移all-to-all分析继续复用，没有重复计算成新收益。

本轮新增F08↔F14、F12↔F14、F14↔F19三条条件合同，并扩展既有attention/output联合组。中央累计131条关系、25组联合合同、464对待汇入或复核。分支仍359条、24个语义模块；2073个regime/285598次执行条件分析，9个regime/1352次待关键字段，849条PMU/16阶段图/3136归属歧义未变。全模块最低MFU、完整融合覆盖和整体理论上限仍未完成，Goal保持ACTIVE。
'''
    action=('继续原Goal,v100PROGRESS;v99PASS200checks。用户operatorfusion/split/reorg/theoreticalceilings,profilefirstthenpinpointcode,noframeworksurvey。'
      '先读attention-projected-state-summary-v100/PROFILE-FINDINGS/source/frontiers/headmerge/roundposition/workstorage/variants/composition/wholeaudit。'
      '3200savedSMLA→inverseRoPE→pack→A2A→OA→OB→RSfrontiers23680TASKs,640QLIpreparecontexts;SWA160/CSA_C21440/CSA_C11600;root2560unbound640。'
      'RecordedSMLA3×64×512andOA24×1×4096→1024show8heads/groupwithdifferentphysicalrow/headowners;equalvolume≠owneravailability。'
      'Headmixedbranchprojectedvectorsnotgenericallymergeableevenwithperheadmasses(counter5/4vs3/4). Preserveheadm/l/uorproveDalphaW=WB. LinearDOFlowerΣrankrequiresindependentstates;actualranks/jointreachabilityUNKNOWN,notactualfixedmodelorbitsLB。'
      'CurrentAPIreturnLSEFalseemptyFLOAT;newsplitstateABI/sinkonce/empty/nonfinite/nativeexp/mergefidelityneeded。'
      'BF16RNEcounter3zero-logitkeysV[1,1,2.015625]+zero-valuesink,originalattentionroundthenW3=3vsVpreproject3.015625;notactualnativeinputs. W_eff=Rinv(querypos)Wquery-dependent,statickeycachecannotassumereuse。'
      '105workmodelsD512R1024H8;prePVwidth2×,Unew<M(1-N/1024)classicalPV+OAworkbreak-even,nonrot448+tailwidth1088=17/8. UnewbuildvUlivecacheseparate;24BF16/FLOATstoragecasesKretained. H8oneconsumer17×/33×total;20distinctOAconsumers321×/641×references,source20KVsharednotweights. H64queryownerworsebankexpansion;noactualmemoryfeasibility/nativegain。'
      '3newREL100pairsF08F14/F12F14/F14F19,HG83-ATTENTION-OUTPUTaddsF08F19;131/25/464. Noexistingv83packcredit/v61OAbeforeA2Afactrecount. 359/24semantic,2073/285598conditional9/1352pending849PMU16phase3136ambiguousunchanged0wholeclosed/noallmodulelowestMFU。'
      'Nextmovebeyondrepeatingthesealgebras: inspectremainingfinitegraph/runtimebranchdomainsandcentralnavigationwithscope-sensitivepruning. Empty464centralpairsarenotproofindependenceandnotnecessarily464unanalysedinterfaces;knownhistoricaldirectedgecoverageisnotallprimitive/branchcoverage. Prioritizeactualunexpandedoperatorbranches/backendalgorithmsorcriticalevidence,keepM00/M03onlyoperatorselection/readinessratherthanframeworksurvey。'
      'v98actualMC2profileCbound3200/1272unknownnotrequeried;nativeoperandplanpreparednotexecuted,noauthorizednewcapture. Alljobsfinishedlocal,no server/model/Torch/NPU/capture/service/implementation. GoalACTIVE,TaskACTIVE,LoopEVALUATING;verify_attention_projected_state_v100mustPASS;neverreruncompletedfinalizer。')
    fusion=deepcopy(read(R/'fusion-scan.json'));fusion['projected_attention_state_refinement_v100']=dict(evidence=[str(O/n)for n in names],new_pairs=3,new_groups=0,new_native_operator_nodes=0,
      actual_numeric_physical_ceiling_or_gain_proved=False,whole_model_exhaustion_proven=False)
    sources=[Path(x['path'])for x in read(O/names[1])['parts']];inputs=[Path(x['path'])for x in read(O/names[0])['inputs']];caches=[Path(x['path'])for x in read(O/names[0])['native_caches']]
    jobs=[dict(kind='profile',status='pass',description='Reconcile3200savedattention/outputownerfrontiers23680TASKsand640selectioncontexts;currentemptyLSE/BF16/restorecontractspinned,noNPUornewcapture.',
      artifacts=[O/n for n in names]+sources+inputs+caches+[Path(read(O/names[0])['graph_block_index']['path'])]),dict(kind='design-check',status='pass',description='Headwise-state/nullspaceandBF16roundcounterexamples,1280commonmassmodels,105work24storagecases;3conditionalrelationsnotactualfixed-modelrank/physicalceiling/gain.',
        artifacts=[O/'attention-projected-state-analysis-preflight-v100.json',O/'attention-projected-state-validation-history-v100.json',O/'whole-scope-requirement-audit-v100.json'])]
    persist_refinement(version=100,previous=99,title='Headwise attention states and output projection reorganization limits',
      summary=dict(filename='attention-projected-state-summary-v100.json',previous_turn_classification='PROGRESS:v99projectionpanelsandC1C2identitycorrection;200postchecks',
        saved_frontiers=3200,reused_native_TASKs=23680,selection_frontiers=640,root_bound=2560,phase_unbound=640,
        exact_real_work_models=105,projected_value_BF16_FLOAT_storage_models=24,actual_checkpoint_rank_or_head_reachability_known=False,
        new_target_source_files=0,new_public_documents=0,refined_candidates=6,refined_modules=6,new_semantic_families=0,branch_records=359,semantic_modules=24,
        new_pairs=3,new_groups=0,extended_group=packet['extend_group'],relation_clauses=131,higher_order_groups=25,pending_central_pair_reconciliation=464,
        scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,new_native_promotions=0,
        actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,implementation=False,native_test=False,remote_job_status='Alllocalexistingprofile/pinpointcachedoperatorsource/mathworkfinished;no server/model/Torch/NPU/capture/serviceorimplementation.'),
      changes=['Bind3200attention/outputownerfrontiersand640QLIpreparecontextswithoutinventingsamephysicalhead/rowownership.',
        'Establishheadwiseprojected-stateinformationlossandBF16/querypositiongates;actualcheckpointstructureandnativefidelityremainconditional.',
        '105work/24storagecasespricepreprojectedVheadexpansion/build/layer/positionandFLOATtypedgates;chosenrepresentationnotuniversalIOoractualgain.',
        '3conditionalpairclausesandoneexistinggroupextension=>131/25/464;wholecoverage/nativeoperand/moduleMFU/physicaltheorygapsremainopen.'],
      candidate_updates=updates,module_updates=mods,remaining=remaining,action=action,jobs=jobs,report=report,read_first=names+['whole-scope-requirement-audit-v100.json'],
      publication_rows={},gate_updates=dict(overall_complete=False),scripts=[N/'analyze_attention_projected_state_v100.py',N/'verify_attention_projected_state_v100.py',N/'analyze_norm_quant_v94.py',N/'analyze_HC_attention_fork_v93.py',N/'persist_profile_refinement_v90.py'],
      executor_prefix='attention-projected-state',main_script=Path(__file__),source_read=False,additional_shared_writes={R/'fusion-scan.json':fusion})

if __name__=='__main__':main()
