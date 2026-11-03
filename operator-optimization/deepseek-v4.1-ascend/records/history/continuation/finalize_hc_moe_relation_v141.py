"""Seal scoped HC-to-MoE composition; preserve every broader coverage gap."""
from pathlib import Path
from copy import deepcopy
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v90 import persist_refinement


def main():
    assert not(O/'REVIEW-CHECKPOINT-v141.json').exists()
    pre=read(O/'hc-moe-preflight-v141.json');assert pre['status']=='PASS'and pre['checks']>60000
    names=['hc-moe-inputs-v141.json','hc-moe-native-frontiers-v141.json','hc-moe-merge-correlation-audit-v141.json','hc-moe-source-contract-v141.json','hc-moe-observed-frontiers-v141.json','hc-moe-readiness-DAGs-v141.json','hc-moe-rounding-and-owner-witnesses-v141.json','hc-moe-work-and-lifetimes-v141.json','hc-moe-composition-v141.json']
    comp=read(O/'hc-moe-composition-v141.json')
    detail=dict(evidence=names,FFN_frontiers=960,source_CID_TASK_bindings=8638,source_API_only_roles=2,interlayer_bridges=936,Engram_guards=48,aux_guards=72,recorded_no_intervening_compute=816,reference_DAG_cases=3072,new_scoped_pair_relations=4,new_pair_clauses=4,new_groups=0,new_main_candidates=0,new_candidate_module_links=0,new_native_promotions=0,new_device_work=False,whole_requirement_closed=False)
    def fork(n):return deepcopy(read(O/(n+'-v140.json')))
    def save(n,d):write(O/(n+'-v141.json'),d)
    rel=fork('candidate-relation-clauses')
    for c in comp['clauses']:
        assert not any({x['a'],x['b']}=={c['a'],c['b']}for x in rel['clauses'])
        rel['clauses'].append(dict(id=c['id'],a=c['a'],b=c['b'],relation={'F20':'HC_DISTINCT_FFN_ROUTER_TYPED_EPOCH','F21':'HC_OPAQUE_MC2_RECEIVER_PROGRESS','F22':'HC_SHARED_TYPED_OVERLAP','F23':'HC_ROUNDED_MERGE_POST_NEXT_LAYER_GUARDS'}[c['b']],operator_claim=c['contract'],evidence=names,applicable_domain='960 saved target eager FFN frontiers,8638 source/CID TASKs and2 API-only roles;936 next-layer links. New fork/epilogues/overlap are conditional. Draft/other epochs/branches/allreader/nativeprecision remain open.',conditional_compatibility=True,all_runtime_gates_proved=False,speedup_additive=False,critical_path_gain_us=None,HC_moe_v141=deepcopy(detail)))
    for gid,member in comp['group_extensions']:
        g=next(g for g in rel['higher_order_groups']if g['id']==gid);assert member not in g['members'];g['members'].append(member);g.update(all_subsets_analyzed=False,legal_joint_fusion_proved=False);g['HC_moe_v141']=dict(**detail,whole_group_closed=False)
    rel.update(parent_relation_index='candidate-relation-clauses-v140.json',new_pair_clauses=4,new_higher_order_groups=0,clause_count=157,selected_pair_count=157,HC_moe_v141=detail,overall_complete=False);save('candidate-relation-clauses',rel)
    cv=fork('candidate-relation-coverage');nav=fork('candidate-pair-navigation')
    for c in comp['clauses']:
        for d in [cv,nav]:
            r=next(r for r in d['pairs']if {r['a'],r['b']}=={c['a'],c['b']});r.update(whole_pair_all_variants_closed=False,new_critical_path_gain_us=None,HC_moe_v141=deepcopy(detail))
            if d is cv:r.update(scope_state='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL',clauses=[c['id']])
            else:r.update(conditional_clause_ids=[c['id']],semantic_pair_reconciliation_done_in_stated_scope=True,evidence=names)
    cv.update(parent='candidate-relation-coverage-v140.json',parent_relation_index='candidate-relation-clauses-v141.json',scoped_conditional_pairs=157,remaining_pair_relations=438,HC_moe_v141=detail,scope='157 scoped conditional pairs/26 groups/438 pending;distinct FFN epochs,MC2 receiver contracts,typed shared overlap and rounded Merge/Post reviewed.Whole-pair/all-runtime/all-subsets open.',overall_complete=False);save('candidate-relation-coverage',cv)
    nav.update(parent='candidate-pair-navigation-v140.json',parent_footprints='candidate-operator-footprints-v141.json',parent_relation_index='candidate-relation-clauses-v141.json',scoped_conditional_pairs=157,remaining_central_pair_reconciliation=438,HC_moe_v141=detail);save('candidate-pair-navigation',nav)
    touched=['F01','F02','F03','F04','F20','F21','F22','F23'];fp=fork('candidate-operator-footprints')
    for r in fp['candidates']:
        if r['id']in touched:r['HC_moe_v141']=dict(evidence=names,no_new_native_operator_atoms=True,local_graph='hc-moe-composition-v141.json',all_runtime_or_new_native_fusion_gates_bound=False)
    fp['parent']='candidate-operator-footprints-v140.json';save('candidate-operator-footprints',fp)
    op=fork('operator-analysis-coverage-ledger');op['HC_moe_v141']=detail;save('operator-analysis-coverage-ledger',op)
    gap=('Need actual same-call owner/receiver masks,topk IDs,expert histogram/clipping,capacity/world attributes for these eager frontiers,MC2 flags/aliases/epochs,new split/prequant ABI and finite cross-rank progress. '
         'Need actual FFN coeff/pre4 publication and raw4H lifetime,loaded RmsNormCast BF16->FLOAT/router HF32/TopK tie/NaN/scale identities,shared full-I amax and quantizer equality before reuse. '
         'Down/Merge/Post fusion retains each BF16 round and source routed scaling exactly once;final reduction must be inactive or separately integrated and repriced. Shared overlap versus delayed Down/on-chip retention shares one resource/interface budget. '
         'Need next-layer full4H coeff/residual plus Engram48/aux72/tail reader/writer/alias/capture closure. API-only2 entries remain unpromoted. '
         'No native parity,physical capacity/MFU/lowest/actual speedup/allvariants closure;9/1352 native,849 PMU,16 phase,3136 assignments and broader candidate gaps remain.')
    obligations=fork('coverage-current-obligations')
    for r in obligations['obligations']:
        if r['id']in ['G00','G01','G02','G05','G06','G07','G08','G09','G10','G11','G13']:r['HC_moe_v141']=deepcopy(detail)
        if r['id']=='G07':r['remaining']='157 conditional pairs/26groups/438pending. F01-F20/F21/F22/F23 distinct attentionPost->FFNPre->roundedNormCast epoch;opaqueMC2 receiver work/progress;shared INT32/clamp/rowquant;Down/Merge/Post/nextHC rounds and reader guards scoped. '+gap
        if r['id']=='G10':r['remaining']='157 conditional pairs/26groups/438pending. HG93 addsF20-F23;HG73-MOE/HG84-HC-FFN-FORK addF01. 3072 arbitrary-service sufficient DAGs include both gains and losses;1536 delayed fused Down later than full overlap. Not physical bounds,actual gain or all-subset closure.'
        if r['id']=='G13':r['remaining']=('v141 960FFNfrontiers/8638nativeTASKbindings+2APIonly,936interlayerlinks/48Engram/72aux/816noCIDCompute;noabsence-to-readerclosure promotion. '
            'M800->P816,816->816,834->834 each320;notceil16rule. Shared afterMC2 sameStream960. Preserve RmsNormCast BF16 widenedFLOAT,source routed scaling once,fullIquant,R BF16 merge/Post rounds and final-reduction gate. '
            '3072DAGs,logical FLOAT/Down/Merge interface union and receiver work references are conditional,not measured savings/HBM/physical ceiling. '
            'Next pending F01-F28/F29/F30/F31/F35 cross target/draft metadata,epoch and terminal consumer interactions;reuse existing profile anchors and only inspect direct source for conclusion-changing gaps. '
            '358semantic/26modules/854nodes1346edges48hints/2073regimes285598conditional/9classes1352native/849PMU/16phase/3136assignment/0whole/physicalMFU-lowestunknown. No newNPU/model/capture/replayauthorization;Padv120consumed,nopendingapproval.')
    obligations.update(parent='coverage-current-obligations-v140.json',new_native_or_whole_closures=0,overall_complete=False);save('coverage-current-obligations',obligations)
    audit=fork('whole-scope-requirement-audit');audit.update(previous_goal_turn='PROGRESS:v140 HC QueryRoPE/key/oldlatent versioned fanout;postcheckPASS39891',input_checkpoint=140,followup_checkpoint=141,HC_moe_v141=detail)
    for r in audit['requirements']:
        if r['id']=='CROSS_OPERATOR_LAYER_CARD_ITERATION':r.update(conditional_pair_relations=157,central_pairs_not_reconciled=438,higher_order_groups=26)
        if r['id']in ['ALL_MODULE_CONDITIONAL_BRANCHES','CROSS_OPERATOR_LAYER_CARD_ITERATION','THEORY_AND_MEASURED_COMPARISON','CRITICAL_EVIDENCE_AND_FINAL_CLOSURE']:r.setdefault('evidence',[]).extend(names)
    save('whole-scope-requirement-audit',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v141.json',dict(status='PROFILE_FFN_MOE_HC_EPOCHS_THEN_TYPED_ROUNDING_AND_INTERLAYER_READER_GUARDS',questions=read(O/'hc-moe-inputs-v141.json')['questions'],source_scope='Only direct model844-920/MoE250-329,213-245,390-420/router194-227;reuse v56/v57/v63/v78/v98/v133 contracts. No framework/loader/SDK survey or target changes.',no_model_operator_implementation=True,no_remote_Torch_CANN_NPU_model_or_replay=True,Pad_authorization='GRANTED_AND_CONSUMED',overall_complete=False))
    write(O/'hc-moe-local-correction-v141.json',dict(status='CORRECTED_WITHOUT_NATIVE_PROMOTION',events=[
        'Initial strict selector failed db10/forward29531/block31 MERGE:0 exact native hits. Persisted source-positive audit proves aclnnForeachAddListV2 API44868 lacks CID/TASK;no nearest-time assignment.',
        'MERGE-only exception then exposed db24/forward77363/block39 SHARED_UP:0 native hits. Generalized only unique source-positive aclnn API with no exact linked TASK;preserve API-only grade and raw negative CID check.',
        'Final960x9 roles=8638native +2API-only;full raw verifier rechecks them and all936 bridge CID sets. Failed local selectors retained;no target runtime failure.'],new_device_work=False,overall_complete=False))
    write(O/'hc-moe-validation-history-v141.json',dict(events=[
        'Reused v140 authority and109-file manifest;previous turnPROGRESS,not completion.',
        'Profile-first 960FFN scopes and936adjacent-layer links;exact CPUsource/CID/Compute for8638TASKs,2API-only holes preserved.48Engram/72aux guards;816 no source-CID-Compute gaps not allreader closure.',
        'Distinct post-attention FFN input and attentionpre4/FFNcoeff/nextpre4 deadlines;router FLOAT is roundedBF16 widening. MC2 receiverA not ownerM*k/capacity;shared INT32 vsMC2FP16 contracts reused.',
        '3072 sufficient weighted DAGs:joint repriced2140faster/912slower/20ties;delayed fusedDown1536later/1536equal tofull overlap. Hypothetical costs,no native gain or optimum.',
        'Finite BF16 Merge/Post/NormCast counterexamples and full4H information-loss/receiver-owner witnesses. Logical interface union,actual P table and H+32 record preserved;final-reduction gates explicit.',
        'Preflight PASS'+str(pre['checks'])+';REL141-01..04 conditional,157pairs/26groups/438pending. All wider native/global and broad conclusion gaps preserved.'],overall_complete=False))
    remaining=[r['id']+': '+r['title']+'. '+r['remaining']for r in obligations['obligations']]
    remaining.append('v141 157conditionalpairs/26groups/438pending;960FFN/8638native+2API-only/936bridges/48Engram72aux/3072DAGs.4472SPjoins;358/26semantic,854/1346/48partialgraph,9/1352native849PMU16phase3136ambiguity0whole unchanged.')
    action=('Continue original Goal;v141PROGRESS F01-F20/F21/F22/F23. Read hc-moe-summary/native-frontiers/source/observed/DAGs/rounding-owner-witnesses/worklifetimes/composition/currentobligations-v141;verify_hc_moe_relation_v141 mustPASS. '
        '960FFN frontiers9roles=8638source-CID-TASK+2APIonly:db10f29531b31MERGE44868,db24f77363b39SHARED_UP95736;never nearest-time promote.936interlayer links48Engram72aux816noCompute;negativeCIDset notallreader closure. M800->P816,816->816,834->834 each320,notceil16rule;shared afterMC2 sameStream960. '
        'MoE value comes attentionPostZ1->FFNPre(current attentionpre4)->NormCast;not preattentionY. Keep BF16Norm wideningFLOAT/routerHF32,FFNcoeff/Post/rawZ1 andFFNpre4nextlayer. SourceR alreadyscaled/preprocessed;never doublefactor. Keep final _maybe_reduce_final_output gate afterMerge beforeHCPost;fullfusedcomposition requiresno finalreduce ornewprovedhandoff. '
        'OpaqueMC2 alreadydispatch/GMM1/act/GMM2/return/unpermute;actualreceiverprocessedA notMlocal*k/capacity65536. H+32dispatchrecord notH+4;EP16 vsTP8;receiverhist/masks/flags/aliases/epochs/newprequantABI andfiniteprogress remain. SharedUpINT32/clamp10/fullI2304amax distinctfromMC2FP16;quantreuse needsactualsamecontract. '
        'DownBF16->MergeBF16->PostBF16 rounds kept eveninsideepilogue;BF16witnesses rejectdistribution/roundelision. Fullsharedoverlap vsdelayedDown/onchipretention competes:3072DAGs2140faster912slower20ties,1536fusedlater1536equal,hypotheticalservicesnotgains/physicalbounds. FLOATplane8MHgross/BF16reread2MH/net6MH chosenalternative;Down+Mergeunion8MH logicalonly,full4Hdeletioncredit0,finalreduce/readersgates. '
        'REL141-01..04;HG93addsF20-F23,HG73-MOE/HG84-HC-FFN-FORKaddF01.157conditionalpairs26groups438pending;358/26semantic854/1346/48partialgraph2073/285598conditional9/1352pending849PMU16phase3136ambiguity0whole/noallmodulephysicalMFU-lowest unchanged;broadercandidategapsretained. '
        'Next pending F01-F28/F29/F30/F31/F35 target/draft metadata/epoch and terminal consumers;reuse profile and direct source only if needed. All other pending pairs/branches remain. No remote/device/Torch/CANN/model/capture/replay/implementation;Padv120 grantconsumed,nopendingapproval. Do not editsealedv141 orrerunfinalizer. Versionedpostcheck/CURRENT must matchRECOVERY. TaskACTIVE Loop004EVALUATING GoalACTIVE.')
    report='''本轮补齐 F01–F20/F21/F22/F23：HC 与路由、MC2、共享专家及汇合/Post 的条件组合。整体目标仍未完成。

已有 profile 的960组FFN前沿对应8638个明确的源码/API/CID/TASK关联，另2个仅有API证据：db10、forward29531、block31的ForeachAdd API44868，及db24、forward77363、block39的共享Up API95736。两者未按时间邻近补配，不增加原生证据关闭数。输入M800→路由/MC2 P816、M816→816、M834→834各320组；不能推广成ceil16规则。960组共享分支都在MC2结束之后于同一stream开始，现有多流代码能力不等于这份profile已启用重叠。

MoE实际输入是attention HCPost的4H结果Z1，经FFN HCPre collapse，再经RmsNormCast得到BF16 U及其FLOAT扩展。不能将attention前的HC输出直接给router或expert。FFN collapse仍需要当前attention pre4；FFN系数分支仍读取完整Z1，并在FFNPost汇合，FFN pre4还要供下一层。提前发布FFN collapse是新的native合同，不会自动删除系数、残差或pre4工作。

RmsNormCast的FLOAT输入是已经BF16舍入后再扩展的值，不是未舍入的FP32归一化。Norm/router融合需保留这个舍入点、实际HF32/累加与TopK的排名、无偏权重、scale、tie/NaN政策。逻辑FLOAT临时plane的store/read参考为8MH B；若改由router重新读取BF16 U，另有2MH B读取，所选版本净参考6MH B；片上交接是另一版本，不能同时领取两份收益。当前BF16 U仍被MC2和共享分支需要。

DispatchFFNCombine已经把dispatch、两次GMM、激活/量化、return与weighted unpermute放入一个算子。不能凭内部阶段名称再主张删除独立launch。其常规矩阵工作参考为70778880×实际接收且处理的assignment数；owner本地M×6和物理capacity均不能代替该数。即使本地M为0也可能处理远端assignment并承担collective进度。原有EP16、capacity65536、H+32记录步长、内部GM别名、flag与分组合同继续保留，不能把另一轮服务器观察中的计数接到旧profile时延上。

共享分支采用INT32 Up、clamp10激活量化、全2304维row amax及BF16 Down，和MC2内部FP16路径不同。共享输入量化是否能复用给MC2，仍须证明实际行、mask、量化政策和scale一致，并引入相应prequant ABI。相同INT8类型不足以定案。

Down、共享/路由Merge和HCPost可以登记融合，但必须保留Down→BF16、Merge→BF16、Post→BF16三个数值边界。有限BF16反例说明将add系数分配进去会把1.5变为1.5078125；跨Post取消内层舍入也能改变下一collapse。融合可以把这些cast留在寄存器中，不能直接代数消掉。R代表原源码预处理/缩放后送入ForeachAdd的routed值，不再次乘路由因子。Merge之后的可选最终reduce也必须保留；完整Down/Merge/Post融合需该reduce不活跃，或有另行证明且计价的新handoff。

共享分支若完全提前执行，Down可能在routed及FFN系数之前结束；要同时删除Down/Merge临时接口，需要保留结果、占用片上资源，或延迟Down。因此不能将“最大共享重叠”与“全部epilogue流量节省”直接相加。3072个任意假设服务的完整前沿DAG保留两次HC系数/pre4期限：竞争重新计价后2140组较快、912组较慢、20组相同；即使服务时间不变，延后Down的融合版本也有1536组晚于完整重叠、1536组相同。这些是反例与条件排程，不是实测胜率、最优时间或物理上限。

所选Down与Merge的私有BF16接口读写各4MH B，联合去重为8MH B。它们是声明版本的逻辑接口参考，须满足最终reduce/其它reader已合法内部化；不是HBM流量、分配量或节省时间。FFNPost完整4H内容为8MH B，因仍有下一系数、残差等读者，本轮删除信用为0。

936条精确相邻层关联中，48条有Engram正调用、72条有辅助读者；816条未发现该CPU范围内source/CID关联的Compute TASK。没有Compute关联不等于没有reader、alias或写者。下一HC collapse可研究与Post组合，但需保留Post的BF16舍入、下一pre4及完整4H观察者；Engram修改与辅助读取均是明确保护条件。

本轮新增4条条件关系：153→157，待核对442→438；26个联合组的全子集和native合法性仍开放。35主候选、7历史框架记录、358语义行/26模块、4472次固定SP关联以及854节点/1346边/48提示的部分图保留。9类/1352次原生证据、849个PMU缺口、16个phase未绑定图和3136项归属歧义仍待补齐；整模块关闭0，全部模块的物理MFU及最低时间尚未建立。没有执行新的服务器或NPU任务。
'''
    updates={cid:dict(review_v141=dict(evidence=names,pairs=comp['new_pairs'],native_TASK_bindings=8638,API_only_roles=2,interlayer_bridges=936,scoped_conclusion_changing_gap=gap,existing_broad_conclusion_gap_preserved=True,no_native_fusion_or_physical_gain_proved=True))for cid in touched}
    inputs=[Path(x['path'])for x in read(O/'hc-moe-inputs-v141.json')['inputs']];refs=read(O/'hc-moe-native-frontiers-v141.json')['databases'];inputs +=[Path(x['raw_database']['path'])for x in refs]+[Path(x['cached_native']['path'])for x in refs]
    persist_refinement(version=141,previous=140,title='HC-to-MoE distinct epochs,opaque receiver progress and rounded shared-merge composition',
        summary=dict(filename='hc-moe-summary-v141.json',previous_turn_classification='PROGRESS:v140 HC QueryRoPE/key/oldlatent versioned fanout;PASS39891',**detail,refined_candidates=8,refined_modules=6,branch_records=358,semantic_modules=26,relation_clauses=157,higher_order_groups=26,pending_central_pair_reconciliation=438,partial_graph_node_rows=854,partial_graph_edge_rows=1346,structural_pair_hints=48,scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,model_operator_implementation=False,native_test=False,actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,Pad_authorization='GRANTED_AND_CONSUMED',remote_job_status='All local saved-profile/source/rounded-MoE DAG work complete;no remote/device/Torch/CANN/model/replay/service work.'),
        changes=['Bind960FFN scopes8638nativeTASKs+2API-only,936bridges48Engram72aux;noCIDpromotion.', 'Reconcile distinct FFN epochs,MC2 receiver work,shared typed overlap and three BF16 rounds;price delayedDown vs full overlap in3072referenceDAGs.', 'AddREL141-01..04;157conditionalpairs/26groups/438pending;native/global/broadercandidate gaps unchanged.'],candidate_updates=updates,
        module_updates={mid:dict(candidates=[],finding=f,evidence=names,gap=gap)for mid,f in {
            'M06':'AttentionPost->FFNPre->NormCast epoch,pre4/coeff/Post/full4H lifetimes retained;roundedPost-nextcollapse composition conditional.',
            'M12':'OpaqueMC2 native targetFFN scopes/receiver assignment work/progress androuterFLOAT/TopK contracts;API-only gaps not promoted.',
            'M13':'SharedINT32/clamp/fullIquant differsMC2FP16;full overlap and delayedDown/Merge/Post interface cuts share resources and preserveBF16 rounds/scaling once/finalreduce.',
            'M14':'48positiveEngram guards among936interlayer bridges;full4Hmutation/readiness cannotbe bypassed bycollapse.',
            'M22':'New fork/prequant/epilogue/capture ABI andlastreader plusqueuedwriter lifetimes unproved;API absence notreader closure.',
            'M23':'EP16 receiver workload isnotTP8/ownerM*k/capacity;split internalMC2 needs actualhist/flag/alias/finiteprogress andresource repricing.'}.items()},
        remaining=remaining,action=action,jobs=[
            dict(kind='profile',status='error',description='Strictnative selector initially required all9roles;two local missing-CID cases corrected to API-only with no time matching;not target failures.',artifacts=[O/'hc-moe-local-correction-v141.json',O/'hc-moe-merge-correlation-audit-v141.json']),
            dict(kind='profile',status='pass',description='Rawsource/API/CID/Compute validation8638TASKs+2API-only;936interlayer CIDsets/48Engram72aux;actual M/P shapes andsharedafterMC2 order.',artifacts=[O/'hc-moe-native-frontiers-v141.json',O/'hc-moe-observed-frontiers-v141.json']+inputs),
            dict(kind='design-check',status='pass',description='3072weightedDAGs,rounding/full4H/receiver counterexamples,logicalinterfaceunion and'+str(pre['checks'])+' preflightchecksPASS;four conditionalHC-MoErelations.',artifacts=[O/'hc-moe-preflight-v141.json',O/'hc-moe-validation-history-v141.json']+[O/n for n in names]+[O/'coverage-current-obligations-v141.json',O/'whole-scope-requirement-audit-v141.json'])],
        report=report,read_first=names+['coverage-current-obligations-v141.json'],publication_rows={},gate_updates=dict(overall_complete=False),scripts=[N/'analyze_hc_moe_native_v141.py',N/'analyze_hc_moe_relation_v141.py',N/'verify_hc_moe_relation_v141.py',N/'persist_profile_refinement_v90.py'],executor_prefix='hc-moe',main_script=Path(__file__),source_read=True)


if __name__=='__main__':main()
