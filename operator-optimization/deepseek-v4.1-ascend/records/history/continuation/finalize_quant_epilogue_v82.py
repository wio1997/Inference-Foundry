"""Persist the bounded operator-output ownership/fusion analysis; no device work."""
from pathlib import Path
from copy import deepcopy
from finalize_origin_coverage_v9 import N, O, R, read, write
from persist_profile_refinement_v5 import persist_refinement


def main():
    assert not (O/'REVIEW-CHECKPOINT-v82.json').exists()
    profile=read(O/'quant-vector-owner-profile-v82.json')
    assert profile['executions']==14143
    names=['quant-epilogue-public-source-index-v82.json','quant-epilogue-primary-details-v82.json',
        'quant-vector-owner-profile-v82.json','quant-epilogue-source-contract-v82.json',
        'DownAdd-epilogue-variants-v82.json','epilogue-rounding-witnesses-v82.json',
        'quant-epilogue-boundary-models-v82.json','Q_B-index-epilogue-extension-v82.json',
        'shared-quant-owner-CV-interfaces-v82.json']
    shared=[]
    for rows in [2,3,800,816,834]:
        h=5120;intermediate=2304
        shared.append(dict(M=rows,H=h,I=intermediate,
            DQ_INT8_to_GateUp_CV_value_bytes=2*rows*h,
            GateUp_INT32_to_DSQ_CV_value_bytes=16*rows*intermediate,
            DSQ_INT8_to_Down_VC_value_bytes=2*rows*intermediate,
            Down_INT32_to_VectorDequant_CV_value_bytes=8*rows*h,
            four_bridge_total_value_bytes=10*rows*h+18*rows*intermediate,
            separate_BF16_S_VV_interface_value_bytes=4*rows*h,
            old_Cube_owned_BF16_Down_domain_four_bridge_bytes=6*rows*h+18*rows*intermediate))
    write(O/names[-1],dict(models=shared,
        domain='A3 fixed ownership:Vector DQ, Cube INT32 GateUp, Vector DSQ, Cube INT32 Down, Vector per-token dequant and merge. Uncompressed full values each stored/read once at every private GM CV/VC interface. No pointer/load-path/native-branch attestation.',
        correction='The final BF16 Down->Add edge can be VV even though the internal INT32 Down->dequant edge is CV. v80 6MH+18MI remains only its Cube-owned BF16-output domain; this per-token Vector-output domain has 10MH+18MI in four internal bridges. It may remove a separate 4MH BF16 VV surface, not its 8MH INT32 CV bridge.',
        weight_and_scale_and_other_reader_costs_excluded=True,actual_HBM_bytes_or_gain=None,overall_complete=False))
    relation=deepcopy(read(O/'candidate-relation-clauses-v81.json'))
    clause=next(r for r in relation['clauses']if r['id']=='REL73-09')
    clause['refinement_v82']=dict(operator_claim='Post-BF16-cast Vector DownAdd can remove BF16 VV materialization while retaining raw INT32 CV bridge. Atomic fresh/in-place/copy-first are distinct domains, not free equivalent epilogues.',
        evidence=names,all_runtime_gates_proved=False,critical_path_gain_us=None)
    assert not any({c['a'],c['b']}=={'F21','F23'}for c in relation['clauses'])
    relation['clauses'].append(dict(id='REL82-01',a='F21',b='F23',relation='ROUTED_DESTINATION_BANK_AND_SHARED_EPILOGUE_READINESS',
        operator_claim='Avoiding R-copy in the fresh atomic variant requires routed producer to publish the exact transformed/rounded R into the final bank. Old-R readers/aliases and row/pad/generation, synchronization and finite-resource progress must permit the new ownership. Delaying shared epilogue cannot wait for work queued behind it on the same stream.',
        applicable_domain='Joint routed MC2/combine output ownership and shared Down BF16 join; new output ABI/bank generation conditional.',
        evidence=names,conditional_compatibility=True,all_runtime_gates_proved=False,speedup_additive=False,critical_path_gain_us=None))
    for group in relation['higher_order_groups']:
        if group['id']in['HG73-MOE','HG80-A3-LOCAL-PATH']:
            group['epilogue_owner_refinement_v82']=dict(contract='Determine BF16 output owner inside mixed quantized matmul. Preserve complete INT32 sum and BF16 S cast before merge; count raw INT32 CV and removable BF16 VV surfaces separately. Choose VectorFresh or atomic bank/alias domain before costs; keep readiness acyclic and count merge once.',evidence=names)
    relation.update(clause_count=47,selected_pair_count=47,no_new_maincandidate=True,overall_complete=False)
    write(O/'candidate-relation-clauses-v82.json',relation)
    coverage=deepcopy(read(O/'candidate-relation-coverage-v81.json'))
    pair=next(p for p in coverage['pairs']if {p['a'],p['b']}=={'F21','F23'})
    assert not pair['clauses']
    pair.update(scope_state='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL',clauses=['REL82-01'],whole_pair_all_variants_closed=False,new_critical_path_gain_us=None)
    coverage.update(scoped_conditional_pairs=47,remaining_pair_relations=548,parent_relation_index='candidate-relation-clauses-v81.json',parent_coverage='candidate-relation-coverage-v81.json',
        quant_epilogue_refinement_v82=dict(evidence=names,new_pair='REL82-01',all_pair_variants_closed=False))
    write(O/'candidate-relation-coverage-v82.json',coverage)
    operators=deepcopy(read(O/'operator-analysis-coverage-ledger-v81.json'))
    operators['quant_epilogue_owner_review_v82']=dict(refined_existing_regimes=31,refined_existing_executions=14143,
        BF16_output_regimes=22,BF16_output_executions=9671,promoted_regimes=0,promoted_executions=0,evidence=names)
    write(O/'operator-analysis-coverage-ledger-v82.json',operators)
    matrix=read(O/'module-phase-branch-matrix-v81.json')
    selected_ids={'M07':{'M07-OP-03'},'M09':{'M09-OP-04'},
        'M13':{'M13-OP-01','M13-OP-08','M13-OP-10','M13-OP-11','M13-OP-12','M13-OP-13'},'M22':{'M22-S14'}}
    branches={m['id']:deepcopy(m['branches'])for m in matrix['modules']if m['id']in selected_ids}
    for mid,rows in branches.items():
        for branch in rows:
            if branch['id']in selected_ids[mid]:
                branch['evidence']+=names
                branch['quant_epilogue_review_v82']='Pinned per-token operator source shows Vector BF16 output after raw INT32 GM. Saved mixed/AIV work supports ownership hypothesis; native branch unknown. Conditional VectorFresh and atomic/copy/bank variants, rounding counterexamples and directional/shared symbolic cuts reviewed; all reader/alias/generation/progress gates retained.'
    write(O/'quant-epilogue-branch-inventory-v82.json',dict(modules=branches,extended_families=sorted(x for xs in selected_ids.values()for x in xs),
        records=342,semantic_modules=24,untouched_module_matrices=22,whole_modules_closed=0,overall_complete=False))
    write(O/'PROFILE-CODE-QUESTION-GATE-v82.json',dict(status='OPERATOR_OUTPUT_OWNERSHIP_AND_POST_BF16_CAST_FUSION',
        questions=['Can the BF16 Down output already be Vector-owned, so merge is VV rather than another CV bridge?',
            'Which rounding/readiness/alias/mode conditions distinguish fresh Vector merge, in-place atomic and copy-first atomic?',
            'Which value transfers remain and what symbolic time/throughput ceilings follow in each stated domain?'],
        scope='Saved profile first; five pinned public QuantBatchMatmulV3 entry/per-token/epilogue files and two primary API pages only. Existing caller contracts reused; no new target source read.',
        new_target_source_read=False,new_target_source_files_obtained=0,new_public_operator_source_files=5,new_public_API_document_snapshots=2,
        repository_wide_audit=False,new_remote_server_or_device_work=False,overall_complete=False))
    write(O/'quant-epilogue-validation-history-v82.json',dict(errors=[dict(kind='BOUNDED_LOCAL_READ_RANGE',
        command='Read quant_batch_matmul_v3.cpp ranges (12,171),(503,577) by direct list indexing; file has576 lines.',
        error_type='IndexError',effects='Read-only display stopped after available source lines; no evidence/target/state mutation. Following bounded min(end,len(lines)) reads completed.',resolved=True)],
        capture_and_profile_and_math_jobs_complete=True,native_tests=0,overall_complete=False))
    gap='Loaded quantized kernel/tiling is not source-attested; public per-token epilogue is a conditional implementation domain. Full fanout/alias, precise native Add/atomic/saturation and acyclic tile-ready/output-bank generation remain conditions. Physical Vector/GM/L2/HBM/network upper capacities and all-model composition remain unbound.'
    updates={cid:dict(review_v82=dict(evidence=names,conditional_operator_analysis_deliverable=True,new_device_work=False,
        output_owner='BF16 per-token quant output can be Vector-owned after internal INT32 CV GM, per pinned public branch; current native instruction owner not attested.'),conclusion_changing_gap=gap)
        for cid in['F07','F09','F21','F22','F23']}
    updates['F23'].update(profile_review_state='POST_BF16_VECTOR_MERGE_AND_ATOMIC_OUTPUT_BANK_DOMAINS_REVIEWED_CONDITIONAL',
        next_profile_or_contract_action='Bounded DownAdd operator/API alternatives and symbolic value/capacity analysis complete; native branch identity/precision/fanout remain explicit conditions rather than reasons to demand implementation. Reconcile joint routed destination and all remaining operator-graph composition using saved evidence.')
    updates['F22']['next_profile_or_contract_action']='Use distinct internal INT32 CV and final BF16 VV ownership. Four-bridge per-token owner model10MH+18MI, old6MH+18MI only Cube-owned BF16 domain; neither is actual HBM. Existing TP/amax/raw-reduction alternatives remain separate.'
    updates['F07']['next_profile_or_contract_action']='Keep active V4.1 no-head512-RMS. Append tail64 RoPE after original Vector BF16 cast if partners/C-S/readiness co-owned; retain full512 output and raw INT32 CV cost, retile/buffer gates. Prior49,152B M3 tail budget is same surface, not additive.'
    updates['F09']['next_profile_or_contract_action']='IndexQ BF16-output Vector epilogue may extend to original RoPE/head quant only after BF16 cast with exact FP16 scale/div/RINT/row-amax contracts and complete readers. Do not delete raw INT32 bridge or claim full subgraph savings from one BF16 plane.'
    mods={mid:dict(finding='Operator-specific BF16 quant output ownership, post-cast fusion, atomic bank/readiness conditions, numerical counterexamples and symbolic directional/shared-cut ceilings analyzed from saved31regimes/14143exec and5publicfiles/2APIpages. No framework audit or device work.',
        evidence=names,candidates=[c for c in updates if any(c in m['candidate_links']for m in matrix['modules']if m['id']==mid)],gap=gap,
        profile_scope='Fusion legality and theoretical upper bounds only; no promotion of unbound source/alias/native precision.')for mid in selected_ids}
    remaining=list(read(O/'module-coverage-gaps-v81.json')['remaining'])
    remaining.append('v82 scoped DownAdd frontier refined:22 BF16 mixed regimes9671exec positiveAIV,9 INT32control4472exec noAIV.5pinnedpublic QuantBatchMatmulV3files prove Vector postcast branch,not installedCANN9.2binding. Preserve BF16S thenBF16Add;precastbias0vs1/Kpartial1vs0counterexamples. VectorFresh removes4MH BF16VV surface while8MH INT32CV remains;atomicinplace/copy/routedbank aredifferentconditionaldomains. Per-token whole-shared4bridges10MH+18MI not v80Cube-owned6MH+18MI.47pairclauses11groups548centralpending;342records24semantic/2073regimes285598review+9/1352/global84916/3136/0wholeclosures unchanged.')
    action=('继续原Goal，v82为PROGRESS：本轮只围绕算子融合与理论上限，未研究框架/实施/跑设备。先读quant-epilogue-summary-v82、PROFILE-FINDINGS-v82、quant-vector-owner-profile/quant-epilogue-source-contract/DownAdd-epilogue-variants/quant-epilogue-boundary-models/shared-quant-owner-CV-interfaces-v82。'
        '5public QuantBatchMatmulV3文件固定a61e016a71f422cb280af0c024975f13629ca100，并非installedCANN9.2.0内核证明。Down/QB/IndexQ22regimes9671exec全部positiveAIV，GateUpINT32control9/4472zeroAIV；profile只支持mixedwork不证明cast指令。'
        'Down原BF16round后在AIVUB接merge可条件省S写读4MH，M3=61440B/M800=16384000B；rawINT32Down→Vector8MH仍存。原0与precastbias1、fullK1与BF16partial0有限数学反例；不取消两次round、不做device/nativebitexactclaim。'
        'freshVector/atomicinplace/copy-first/routed-direct-final-bank/Fixpipe是5分域；atomicBF16API支持不证明原Add精度/特殊值/饱和/alias/完成顺序。copyR多4MH会吃掉参考saving，直接producer写finalbank需F21F23联合合同。等待routed不能同stream自依赖/堵塞producer，需tile事件/资源/UB/GMliveness计价。'
        'Down边界含rawI32时16n→12n逻辑valuebytes，reads/writes各8n→6n；T下界max(2Kn/Ci8,6n/Bread,6n/Bwrite,12n/Bshared_ifvalid,Lvector,Ldependency)，C/B须真capacityupper。752TOP/1.6TB只reference；GM不等于HBM，差值不是提速。共享per-token4桥10MH+18MI，旧6MH+18MI仅CubeBF16owner域。QB延伸只tail64且保full512，无head512RMS。'
        '下一批优先汇入原fusion-scan/sourcegraph/candidate-footprint已有关系，按共享输入/输出、生产消费、同资源、状态generation分组定点复核，不从595空格猜独立，也不泛扫框架。可先读现有candidate-relation-clauses/coverage-v82、原FUSION_SCAN/graph索引；补具体组合的必要工作、读者、精度和资源cut。'
        '本轮47pairclauses/11groups/548central未汇入，35model+7legacyframework/26modules342records24semantic；2073/285598review+9/1352pending、849PMU16phase3136ambiguity、0wholeclosure均保留。无需为条件分析补实现/新profile；只查会改结论的具名缺口。Goal active，未全覆盖不能complete。所有本地任务已结束，无remote/model/Torch/NPU/service/processwork。')
    report='''本轮按“算子能否融合、理论上限是什么”收敛取证，没有扩大为框架研究。

保存的profile中，Down、Q_B、IndexQ共22种BF16输出规约、9,671次执行均有Vector工作；GateUp的9种INT32规约、4,472次执行没有Vector工作。规约与PMU一一绑定，未改变原有模块归属和不确定性。

固定commit的[QuantBatchMatmulV3 per-token算子源码](https://gitcode.com/cann/ops-nn/blob/a61e016a71f422cb280af0c024975f13629ca100/matmul/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_pertoken_opt.h)显示：Cube写出INT32，Vector读取并反量化，最后在UB内转成BF16再写出。这提供了“BF16转换后接普通Vector加法”的条件路径。公开分支尚未绑定当前CANN9.2二进制，profile也不能单独确定具体转换指令。

融合应保留Down的BF16舍入，再执行原加法及其舍入。有限FP32/BF16-RNE模型中，d=257、R=-256时，原顺序结果为0，直接在首次BF16转换前加R则为1；把257和-256两个INT32部分和先分别转BF16再累加，也会把正确的1变成0。这些是数学反例，没有执行设备测试。

在原INT32中转保留、S没有未吸收读者、R已就绪且结果Z须写GM的域内，普通Vector融合可省掉S的一次BF16写出与读回，预算为4×M×5120字节：M=3为61,440B，M=800为16,384,000B。Down局部边界含内部INT32中转时，逻辑值搬运预算由16n变为12n，读写各由8n变为6n。它们不等于HBM流量或已实现的提速。

[SetAtomicAdd文档](https://asc.gitcode.com/api/SIMD-API/basic_api/atomic_operations/SetAtomicAdd.html)给出A3的BF16及UB→GM支持，但不会自动初始化目的值，且2201存在设备级BF16饱和模式。原子方案仍需匹配原加法的数值语义、旧R读者、目标bank及完成顺序。先复制R到新Z再原子加，额外复制成本可能抵消上述逻辑流量节省。

[Fixpipe的A3类型表](https://asc.gitcode.com/api/SIMD-API/basic_api/cube_compute_ISASI/cube_compute_store/Fixpipe_L0CToGM.html)未列出INT32→BF16，不能把其他代际的能力当作当前INT8 Down的直接替换。该结论限定于公开API，不是硬件不可能性证明。

理论模型按同一资源的必要工作推导，并允许资源重叠。令n=M×5120、K=2304，在上述固定实现域内，T≥max(2Kn/C_INT8,6n/B_GM读,6n/B_GM写,12n/B_共享〔若证明共享容量〕,L_Vector,L_真实依赖)。实际容量上限缺失的项保留符号；公开752TOPS、1.6TB/s只作参考，不能据此声称实际MFU或可达提速。

本轮更新5项既有候选，增加F21/F23的目标bank与就绪关系，台账为47条两两关系、11组多候选关系，中央尚有548个两两关系待汇入或复核。342条模块分支、24个语义模块及原有profile缺口均保留，整体任务尚未完成。
'''
    source_paths=[Path(x['local_path'])for x in read(O/names[0])['files']+read(O/names[1])['files']]
    jobs=[dict(kind='review',status='pass',description='Pinned5public operatorfiles+2primaryAPIpages;source/blob/document hashes recorded. Only quant-output ownership and fusion/API semantics inspected; no newtargetsource/framework/server/devicework.',artifacts=[O/names[0],O/names[1]]+source_paths),
        dict(kind='profile',status='pass',description='Saved31regimes14143TASK/PMUrecords matched exactly:9671BF16output calls positiveAIV;4472INT32GateUp controls zeroAIV. No missingPMUin subset or globaluncertainty promotion.',artifacts=[O/names[2]]),
        dict(kind='design-check',status='pass',description='FiveconditionalDownAdddomains,5shapes,2finite roundingcounterexamples,typedCV/VVboundary andsymbolicresourcecut analysis;no gain/nativelegality claim.',artifacts=[O/x for x in names[3:]]),
        dict(kind='review',status='error',description='Resolved read-only local source-display IndexError from range577on576linefile. No target/evidence/state mutation; corrected bounded reads completed.',artifacts=[O/'quant-epilogue-validation-history-v82.json'])]
    persist_refinement(version=82,previous=81,title='Quantized operator output ownership, post-BF16 fusion and theoretical boundary',
        summary=dict(filename='quant-epilogue-summary-v82.json',previous_turn_classification='PROGRESS:v81verifiedv80architectureanalysis;thisturn operator-specificepilogueanalysis',
            public_operator_source_files=5,public_API_documents=2,profile_refined_regimes=31,profile_refined_executions=14143,
            BF16_output_regimes=22,BF16_output_executions=9671,INT32_control_executions=4472,conditional_DownAdd_variants=5,shape_models=5,
            numerical_counterexamples=2,new_native_legality_or_speedup_claim=False,retained_records=342,retained_semantic_modules=24,
            relation_clauses=47,higher_order_groups=11,pending_central_pair_reconciliation=548,scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,
            pending_regime_reconciliation=9,pending_execution_reconciliation=1352,new_target_source_files=0,new_remote_source_read=False,
            implementation=False,native_test=False,actual_lowest_MFU_module_established=False,
            remote_job_status='All local saved-profile/math/public-source work complete;no remote server/model/Torch/NPU/capture/service/process work.'),
        changes=['9671BF16mixed/AIV+4472INT32control savedrecords bindoperatorownerhypothesis without source-to-nativepromotion.',
            'FiveDownAdddomains preserveBF16rounding andpriceINT32CVversusBF16VV,alias/mode/bank/readiness/resources;2counterexamples anddirectional/sharedsymbolicceilings.',
            '5existingcandidates/4modules refined;47pairclauses11groups548centralpending;globalcoverageanduncertainties preserved.'],
        candidate_updates=updates,module_updates=mods,semantic_branches=branches,remaining=remaining,action=action,jobs=jobs,report=report,
        read_first=names[:9]+['candidate-relation-clauses-v82.json','candidate-relation-coverage-v82.json'],publication_rows={},
        gate_updates=dict(overall_complete=False),scripts=[N/'capture_quant_epilogue_sources_v82.py',N/'capture_quant_epilogue_details_v82.py',
            N/'analyze_quant_vector_owner_v82.py',N/'model_quant_epilogue_fusion_v82.py',N/'verify_quant_epilogue_v82.py'],
        executor_prefix='quant-epilogue',main_script=Path(__file__),source_read=False)


if __name__=='__main__': main()
