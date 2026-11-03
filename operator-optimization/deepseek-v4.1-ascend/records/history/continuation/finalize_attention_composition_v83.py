"""Publish bounded operator-composition progress; global task remains active."""
from copy import deepcopy
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v5 import persist_refinement


def main():
    assert not(O/'REVIEW-CHECKPOINT-v83.json').exists()
    names=['operator-composition-inputs-v83.json','source-template-applicability-overlay-v83.json',
        'candidate-operator-footprints-v83.json','candidate-pair-navigation-v83.json',
        'attention-composition-clauses-v83.json','attention-joint-interface-models-v83.json',
        'joint-operator-resource-cut-proof-v83.json','ring-persistence-input-supplement-v83.json',
        'index_attention-distinct-score-proof-v83.json','attention-transfer-credit-ledger-v83.json']
    new=read(O/names[4]);assert new['new_pair_clauses']==22
    prior=read(O/'candidate-relation-clauses-v82.json')
    combined=deepcopy(prior);combined['clauses']+=deepcopy(new['clauses']);combined['higher_order_groups']+=deepcopy(new['higher_order_groups'])
    for clause in combined['clauses']:
        if clause['id'].startswith('REL83-'):
            clause['evidence']+=names[6:]
            if clause['id']in['REL83-08','REL83-22']:
                clause['distinct_score_contract']='Index QK/head score does not replace SMLA QK/softmax score; their dimensions, types, weights and query/key transforms differ. Both current conventional calculations remain unless additional fixed-weight equivalence is proved.'
    combined.update(clause_count=69,selected_pair_count=69,no_new_maincandidate=True,overall_complete=False)
    write(O/'candidate-relation-clauses-v83.json',combined)
    coverage=deepcopy(read(O/'candidate-relation-coverage-v82.json'))
    for clause in new['clauses']:
        pair=next(p for p in coverage['pairs']if {p['a'],p['b']}=={clause['a'],clause['b']})
        assert not pair['clauses']
        pair.update(scope_state='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL',clauses=[clause['id']],
            whole_pair_all_variants_closed=False,new_critical_path_gain_us=None)
    coverage.update(scoped_conditional_pairs=69,remaining_pair_relations=526,higher_order_groups=14,
        parent_relation_index='candidate-relation-clauses-v82.json',parent_coverage='candidate-relation-coverage-v82.json',
        operator_composition_refinement_v83=dict(operator_footprints=35,pair_navigation=595,new_pair_clauses=22,
            structural_template_overlay=[12,287,344],navigation_is_not_semantic_proof=True,evidence=names))
    write(O/'candidate-relation-coverage-v83.json',coverage)
    operators=deepcopy(read(O/'operator-analysis-coverage-ledger-v82.json'))
    operators['operator_composition_refinement_v83']=dict(new_promoted_regimes=0,new_promoted_executions=0,
        conditional_pair_clauses=22,evidence=names,uncertainties_or_attribution_changed=False)
    write(O/'operator-analysis-coverage-ledger-v83.json',operators)
    selections={
      'M07':{'M07-OP-04','M07-OP-05','M07-OP-06','M07-OP-07'},
      'M08':{'M08-OP-03','M08-OP-06','M08-OP-07','M08-OP-08','M08-OP-10','M08-OP-11','M08-OP-12','M08-OP-13'},
      'M09':{f'M09-OP-{i:02d}'for i in range(1,11)},
      'M10':{f'M10-OP-{i:02d}'for i in range(1,10)},
      'M11':{'M11-OP-04','M11-OP-06','M11-OP-09','M11-OP-13','M11-OP-14'},
      'M22':{'M22-S12','M22-S13','M22-S14'}}
    matrix=read(O/'module-phase-branch-matrix-v82.json')
    branches={m['id']:deepcopy(m['branches'])for m in matrix['modules']if m['id']in selections}
    for mid,rows in branches.items():
        for row in rows:
            if row['id']in selections[mid]:
                row['evidence']+=names
                row['operator_composition_v83']=dict(state='SCOPED_JOINT_OPERATOR_REORGANIZATION_CONSTRAINTS_AND_SYMBOLIC_RESOURCE_CUT_REVIEWED',
                    interpretation='Three-scale/index/cache/attention/pool/output chains composed with all named fanout, BF16/FP16 boundaries, persistent demand and true tile readiness retained. No new source branch/runtime-guard proof or native fusion claim.',evidence=names)
    write(O/'attention-composition-branch-inventory-v83.json',dict(modules=branches,
        extended_family_IDs=sorted(x for xs in selections.values()for x in xs),global_records=342,
        semantic_modules=24,untouched_other_module_matrices=20,whole_modules_closed=0,overall_complete=False))
    gap=('The22newpairclausesareconditionalcompositionanalysis,notwhole-pair/allvariantclosure. '
        'Futurecache/bank/state readers, precision and generation equality are explicit gates. '
        'No physicaljointA3INT8/BF16/Vectorcapacitybeta,actualGM/HBM/network/fmaxorrequestcriticalpathisnewlyattested. '
        '35footprints/595navigationrowsand12oldtemplates/287nodes344edgesarestructuralretrieval,notwholeprimitive/runtimeproof.')
    members=sorted({c[k]for c in new['clauses']for k in['a','b']})
    updates={cid:dict(review_v83=dict(new_clause_ids=[c['id']for c in new['clauses']if cid in[c['a'],c['b']]],
        footprints=names[2],joint_costs=names[5],joint_capacity_proof=names[6],relations='candidate-relation-clauses-v83.json',
        conditional_operator_analysis_deliverable=True,implementation=False,native_test=False),
        conclusion_changing_gap=gap)for cid in members}
    updates['F12']['next_profile_or_contract_action']='QLI/prepare/SMLA,IndexQ/weight/key andcompressedKV jointcontracts nowcentral. Retain separate index-vs-attention scores, chronologicalprefix/fillers andstablebanks. Furtherreconcileothercrosslayer/metadata/statevariants without doublecreditingexistingnativefusion/selfcopy/candidatebanksharing.'
    updates['F13']['next_profile_or_contract_action']='ComposeSWA/CMPcache/index readinessandSMLA outputinverseRoPE/layout whilepreservingBF16rounds/internalCVqueues. Splittingpartialattentionrequirescompatibleonline(m,l,a),sinkonce,allmetadata/readers andfiniteprogress;normalizedBF16outputswithabsentLSEcannotuniquelymerge. JointmixedCubeproofsymbolic;noactualgain.'
    updates['F34']['next_profile_or_contract_action']='Even-statefuturemaintenance doesnotremovebothrowsofcurrentcompletedpairorIndexKey/longKVfutureconsumers. Useactual3ringobjects/SWAanchoredtuple/privateandpairedlifetimecontracts;do notcarryeven-write/half-ringclaimintohalfprojectionwork orTopKscore reuse.'
    mods={mid:dict(finding='22 named attention/index/compressor/output pair clauses and3higher-order contracts reconciled with35operator-footprintnavigation. Joint scale/private-output/persistent-publication references and shared mixed-mode resource proof added;source templates corrected for activeV41/numerical/row/nativefusion scope.',
        evidence=names,candidates=[c for c in members if any(c in m['candidate_links']for m in matrix['modules']if m['id']==mid)],
        gap=gap,profile_scope='Operator fusion/split/reorganization and theoretical resource limits; reused saved evidence only.')for mid in selections}
    write(O/'PROFILE-CODE-QUESTION-GATE-v83.json',dict(status='OPERATOR_INTERFACE_COMPOSITION_AND_JOINT_TYPED_CAPACITY',
        previous_turn_classification='PROGRESS:v82DownAddVectorownership/rounding/resourceanalysis',
        questions=['Which named operator interfaces are private versus persistent, shared versus distinct numerical values?',
            'Can pair/triple composition preserve query/key/indices/slot/state generations and avoid doublecounting output/layout/OA?',
            'Which valid joint capacity bound follows for INT8+FP16/BF16+Vector on shared hardware without guessing exclusive service?'],
        new_source_read=False,new_target_source_read=False,new_remote_source_read=False,new_public_document_read=False,
        no_model_Torch_NPU_newcapture_server_service_or_processwork=True,overall_complete=False))
    remaining=list(read(O/'module-coverage-gaps-v82.json')['remaining'])
    remaining.append('v83 operator-composition batch:35candidatefootprints/595navigationrowscapture12historicaltemplates287nodes344edges withactiveV41/hiddenIndexDQ/mixedquantowner/rows/nativefusion/persistentfork overlay. Navigationnotsemanticcoverage.22newattention/index/compressor/outputclauses+3groups yield69pairs14groups526centralunreconciled. ScaleM3privateunion768BwhilequeryINT8VC24576Bremains; illustrativeQLI+SMLAinternalGM1,892,352BnotHBM. Even-onlyfuturestatewritesnotcurrentpairprojectionhalving;normalizedattentionoutputs0/1neednormalizers tochoose. Query/keyquant95vs96witnessreused. Jointcapacitysupportproof T>=lambda.W/beta(lambda)conditional;individualpeaksdonotproveadditivedurationsnorfreeoverlap. 25transferatoms/4computeatomskeepjointcreditunique.342records24semantic/2073regimes285598review9/1352pending/global84916/3136/0wholeclosure unchanged.')
    action=('继续原Goal，v83为PROGRESS，前v82为实质PROGRESS。严格围绕算子融合/拆分/重组与理论上限：profile先，代码只补具名算子缺口；本轮无新源码/框架扫描/服务器/模型/Torch/NPU/新profile/进程/实施。'
        '先读attention-composition-summary-v83、PROFILE-FINDINGS-v83，再读candidate-operator-footprints/candidate-pair-navigation/source-template-applicability-overlay/attention-composition-clauses/attention-joint-interface-models/joint-operator-resource-cut-proof-v83。'
        '35footprints是各候选variant原子并集导航，非同一同时可行图；595pairs导航不等于595语义审完。12旧G5templates287nodes344edges只检索基底，已标activeV41无postQBhead512RMS、IndexQ输入DQ隐含独立、mixedquantVectorBF16owner、rowmap和opaque-native已融合baseline；未修改旧图或升级runtimebinding。'
        '22newclauses涉及F08/09/10/11/12/13/14/16/17/18/19/34；总69pairclauses14groups526中央未汇入。IndexQ来自QR1280而IndexW来自hidden5120不能拼同输入GEMM；queryFP16scale-beforequant与keyFLOATscale-beforeFP16cache不同(95/96)。QLI和SMLA不同score计算不能互代。'
        '保留SWA/indexK+scale/longKV和indices/candidate未来读者、preRoPElatentfork、current/rollbackslot/request/row/weights/configgeneration。新private scalarM3savingmax768B只限两个FP16临时面，queryI8VC24576B和QLIscalar/SMLAscore-exp内部GM仍需计价。示例3queryL17408,n640时QINT8=427819008,headFP16=3342336,SMLABF16=251658240,internalGM1,892,352B；不是actualvalidlength/HBM/time。'
        'SMLA可先SWA后CMP但现已整核融合，拆需兼容(m,l,a)/sink一次/数值状态，不可从BF16normalizedoutputs和空LSE无损合并。inverse尾64与unpackedfullplane只在明确distincttransferbaseline下union计价，OA重叠只算一次。C2even历史state写不等于currentcompletedpair只算even投影；旧ring读先于新写，3ringobjects/共享SWAallocator和private/pairedABI合同保留。'
        '新共同资源定理：任意lambda>=0若所有合法瞬时混合精度分配sum(lambda*r)<=beta(lambda)，则T>=lambda.W/beta；未知actualbeta不填数。个别峰值只给max(Q8/C8,Qh/Ch)，不能推出两时间和，也不能说同时满峰免费overlap。'
        '下一批利用完整navigation筛选MHC前后/FFN/Engram/context/target-sampling/metadata/state的真实共享值和边，先汇入旧合同再补关键联合成本；勿为填满空格宣称独立，不要扩大研究框架。可按F01–04/F20–30/F31/F35的已知连接分组，补三算子及跨层/迭代/卡组合。'
        '保35model+7历史framework、26modules342records24semantic、2073/285598review9/1352pending/global849PMU16phase3136ambiguity/0wholeclosures，actuallowestmoduleMFU与全typedphysicalcapacity仍未知。检索误名C2-physical-ring-storage-models-v79已由3真实ringstorage79文件补齐，见ring-persistence-input-supplement；无运行失败。所有localjobs结束，Goalactive，未全部覆盖不complete。')
    report='''本轮把已保存的算子分析按生产消费、共享结果和持久读者汇入同一张关系表，没有新增框架代码扫描或设备执行。

新增22条条件关系，覆盖IndexQ/IndexWeight/key缓存/QLI/稀疏注意力、C2投影/池化/状态、以及注意力输出布局/两级投影；增加3组多候选合同。当前累计69条两两关系、14组多候选关系，仍有526个中央两两关系待汇入或复核。35项候选均有算子范围导航，595行导航不等于595项语义证明。

本轮明确了以下会改变融合预算的约束：

- IndexQ来自归一化后的QR1280，IndexWeight来自hidden5120，不能按同输入GEMM合并。可分析的接口在后续FP16权重与query scale汇合处。两张私有FP16临时面的M=3写读预算合计768B；即使全部吸收，24,576B的INT8 query Cube/Vector接口仍在。
- QLI的INT8索引分数与SMLA的BF16注意力分数有不同维度、权重和数值流程，不能因内核组合而互相替代。QLI和SMLA原生已包含的内部融合也不能再次计作收益。
- SWA、IndexK/scale、压缩KV、selected indices及C1 candidate bank有后续层或迭代读者。当前消费者可以提前接入，并不自动允许删除这些发布。压缩latent的IndexKey读者仍需读取RoPE前的BF16值。
- C2只维护偶数历史状态，不代表当前池化投影可减半。每个已完成pair仍需要两行输入，另有旧ring残留和未来tail更新需求。
- 将SWA和压缩注意力拆开时，仅有各自归一化输出不足以合并：相同的输出0和1，在两部分归一化质量为1:1或3:1时，合并结果分别是0.5或0.25。还需兼容的在线状态、sink计入一次及原数值顺序。

共同资源上限也已补成明确条件：若所有合法混合执行都满足“各类瞬时吞吐的加权和≤β”，则时间下界为“各类必要工作的加权和÷β”。只知道独立INT8和BF16峰值时，不能直接将各自时间相加，也不能据此认定它们可同时跑满。当前A3混合模式、Vector及实际GM/L2/HBM/网络容量缺少足够证据的项仍保留符号。

对旧源码图增加了适用域覆盖说明：当前V4.1没有QB后head512 RMS；IndexQ输入量化、混合核最终输出归属、query/KV行域和原生已融合阶段均须按已保存合同解释。旧图的12个模板、287个节点、344条边是结构导航，未升级为实际执行图。

本轮细化6个模块的既有分支，342条分支记录、24个语义模块及原有profile缺口均保留。整体目标仍未完成，后续继续审阅剩余跨模块、跨层、跨卡及跨迭代组合。
'''
    write(O/'attention-composition-validation-history-v83.json',dict(
        previous_checkpoint_verifier=dict(version=82,result='PASS',artifacts=67),
        retrieval_name_resolution='One stale optional filename resolved to3existingringartifacts withhashes;not a native/code execution failure.',
        new_execution_errors=0,device_execution=False,all_local_jobs_finished=True,overall_complete=False))
    jobs=[dict(kind='review',status='pass',description='Reused frozen operator contracts/sourcegraph;35v82masterhashes captured into navigation footprints.12templates287nodes344edges scoped by6applicability corrections;no source/framework/devicework.',artifacts=[O/x for x in names[:4]]),
        dict(kind='design-check',status='pass',description='22conditionalpairclauses/3higherordercontracts withprivate/persistent/fork/rounding/readinessconstraints;scope69pairs14groups526pending. Existing nativefusionandsharedOA/interfacescountedonce.',artifacts=[O/x for x in names[4:6]]),
        dict(kind='design-check',status='pass',description='Analyticmixed-resourcecapacitysupportbound,typedjointwindow,normalized-outputinformationlossandpair-rowdemandreview. Actualcapacity/traffic/criticalpathgainunclaimed;staleringsourcename resolved.',artifacts=[O/x for x in names[6:]]+[O/'attention-composition-validation-history-v83.json'])]
    persist_refinement(version=83,previous=82,title='Operator interface composition and joint typed resource cuts',
        summary=dict(filename='attention-composition-summary-v83.json',previous_turn_classification='PROGRESS:v82quantizedVectoroutput/DownAddroundingandresourceanalysis',
            operator_footprints=35,pair_navigation_rows=595,historical_source_templates=12,historical_template_nodes=287,historical_template_edges=344,
            new_conditional_pair_clauses=22,relation_clauses=69,new_higher_order_groups=3,higher_order_groups=14,pending_central_pair_reconciliation=526,
            refined_candidates=len(members),refined_modules=6,retained_branch_records=342,semantic_modules=24,
            scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
            new_target_source_files=0,new_remote_source_read=False,new_public_document_read=False,implementation=False,native_test=False,
            actual_mixed_mode_capacity_proved=False,actual_lowest_MFU_module_established=False,
            remote_job_status='Alllocaloperator-contract/mathworkcomplete;no remoteserver/model/Torch/NPU/capture/service/processwork.'),
        changes=['35operatorfootprints/595pairnavigation withcorrectedhistoricalgraphscope,notsemantic/runtimecoveragepromotion.',
            '22newattention/index/compressor/outputconditionalclauses+3higherordercontracts;69pairs14groups526centralpending.',
            'Jointprecision/persistentfanfout/currentpair-demand/outputtransferunionandmixed-resourceanalyticcapacityproof;actualratesunknown;no gainclaim.',
            '12candidate/6module updates preserve342records24semantic andallregime/globaluncertaintycounts.'],
        candidate_updates=updates,module_updates=mods,semantic_branches=branches,remaining=remaining,action=action,jobs=jobs,report=report,
        read_first=names+['candidate-relation-clauses-v83.json','candidate-relation-coverage-v83.json'],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/'analyze_attention_composition_v83.py',N/'refine_joint_resource_cut_v83.py',N/'build_attention_transfer_ledger_v83.py',N/'verify_attention_composition_v83.py'],
        executor_prefix='attention-composition',main_script=Path(__file__),source_read=False)


if __name__=='__main__':main()
