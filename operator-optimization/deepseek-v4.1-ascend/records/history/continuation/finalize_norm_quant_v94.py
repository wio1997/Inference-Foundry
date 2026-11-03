"""Persist operator-targeted precision/GM-interface/resource refinements without target edits."""
from copy import deepcopy
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import N,O,R,read,write
from persist_profile_refinement_v90 import persist_refinement

def main():
    assert not(O/'REVIEW-CHECKPOINT-v94.json').exists()
    assert read(O/'norm-quant-analysis-preflight-v94.json')['status']=='PASS'
    names=['norm-quant-operator-source-contract-v94.json','norm-quant-native-frontiers-v94.json',
      'norm-quant-precision-and-row-models-v94.json','norm-quant-interface-and-resource-models-v94.json','norm-quant-variants-v94.json']
    nf=read(O/names[1]);members=['F05','F06','F09']
    gap=('Native RmsNorm/DynamicQuantV2 arithmetic, scale/zero/nonfinite modes, allowed numerical tolerance and all qr readers remain unbound. '
      'Public fused keys1/2 retain FP32 normalized values; matching API name/A3 dtype is not bitwise compatibility or installed CANN9.2 attestation. '
      'Fusion preserving BF16 qr in UB is conditional on real ownership/row/gamma/epsilon/generation/lifetime. GM interface values are not HBM, and A3 Vector→Cube INT8 GM bridge remains in the stated owner domain. '
      'Joint weight cuts require conventional full dense outputs/distinct uncompressed planes/L1 path/one initial grant/no precomputed partials; clocks/cache/Vector reductions/native schedule/critical path are not closed.')
    rel=deepcopy(read(O/'candidate-relation-clauses-v93.json'))
    clause=next(r for r in rel['clauses']if r['id']=='REL73-10')
    clause['norm_quant_precision_refinement_v94']=dict(evidence=names,
      contract='One identical qr quant producer can serve QB/IndexQ. Preserve BF16 qr before amax; do not substitute a FP32-normalized fused API or cancel a row scalar without precision/scale-observer proof. Reprice the selected whole interface graph once.',
      native_or_whole_pair_closed=False)
    for group in rel['higher_order_groups']:
        if group['id']in ['HG73-Q','HG74-Q-ATTENTION','HG80-A3-LOCAL-PATH']:
            group['norm_quant_precision_refinement_v94']=dict(evidence=names,contract=clause['norm_quant_precision_refinement_v94']['contract'],whole_group_closed=False)
    rel.update(parent_relation_index='candidate-relation-clauses-v93.json',new_pair_clauses=0,new_higher_order_groups=0,overall_complete=False);write(O/'candidate-relation-clauses-v94.json',rel)
    cov=deepcopy(read(O/'candidate-relation-coverage-v93.json'));cov.update(parent='candidate-relation-coverage-v93.json',parent_relation_index='candidate-relation-clauses-v94.json',
      norm_quant_refinement_v94=dict(existing_clause='REL73-10',new_pairs=0,new_groups=0,evidence=names),overall_complete=False);write(O/'candidate-relation-coverage-v94.json',cov)
    fps=deepcopy(read(O/'candidate-operator-footprints-v93.json'));index={r['id']:r for r in read(O/'candidate-record-index-v93.json')['model_candidates']}
    for row in fps['candidates']:
        path=R/'candidates'/(row['id']+'.json');h=hashlib.sha256(path.read_bytes()).hexdigest();assert h==index[row['id']]['master_sha256']
        row['captured_master_sha256']=h;row['captured_master_version']=93
        if row['id']in ['F05','F09']:
            row['operator_atom_union']+=['attn.qr.BF16_round_before_DQ'];row['norm_quant_refinement_v94']=dict(evidence=names,virtual_precision_boundary_not_new_executed_operator=True)
    fps.update(parent='candidate-operator-footprints-v93.json',overall_complete=False);write(O/'candidate-operator-footprints-v94.json',fps)
    nav=deepcopy(read(O/'candidate-pair-navigation-v93.json'));fpmap={r['id']:r for r in fps['candidates']}
    for pair in nav['pairs']:
        pair['shared_operator_atom_union']=sorted(set(fpmap[pair['a']]['operator_atom_union'])&set(fpmap[pair['b']]['operator_atom_union']))
    nav.update(parent='candidate-pair-navigation-v93.json',parent_footprints='candidate-operator-footprints-v94.json',overall_complete=False);write(O/'candidate-pair-navigation-v94.json',nav)
    op=deepcopy(read(O/'operator-analysis-coverage-ledger-v93.json'));ids={r['regime_id']for r in nf['observed_regimes']}
    for r in op['regimes']:
        if r['regime_id']in ids:
            assert r['state']=='SCOPED_OPERATOR_CONDITIONAL_FUSION_AND_WORK_IO_REVIEWED'
            r['scoped_conditional_review_evidence']+=names
    op['norm_quant_refinement_v94']=dict(existing_regimes_refined=22,existing_executions=9566,graph_frontiers=3200,index_query_forks=640,
      native_TASKs_reused=14080,new_native_promotions=0,new_physical_attribution_or_MFU=False)
    write(O/'operator-analysis-coverage-ledger-v94.json',op)
    matrix=read(O/'module-phase-branch-matrix-v93.json');branches=deepcopy(next(m['branches']for m in matrix['modules']if m['id']=='M07'))
    assert not any(r['id']=='M07-OP-14'for r in branches)
    branches.append(dict(id='M07-OP-14',branch='QR RMS→BF16 rounding→one dynamic quantizer→QB/IndexQ fork: exact-real row-scale identity, native API precision and shared interface/resource accounting.',
      finding='3200 graph frontiers/640 index-query forks bound to typed TASKs. Public fused keys1/2 quantize FP32 normalized values; a finite BF16 model gives INT8 code42 vs43. Preserving the BF16 round in UB remains a conditional fusion. Joint graph costs retain GM VC bridge and one weight warm grant.',
      state='SCOPED_OPERATOR_PRECISION_AND_SHARED_RESOURCE_CONTRACT_REVIEWED;WHOLE_MODULE_OPEN',evidence=names,gap=gap,
      scope='Current QR1280 typed profile and declared finite arithmetic/owner/path domains; not native exact-output attestation or all numerical modes/callers.'))
    mods={mid:dict(finding='QR norm/quant fork now has precision substitution gate, conditional shared-graph GM budgets and joint typed weight cuts. No native precision, HBM saving or speedup claim.',
      evidence=names,candidates=[c for c in members if c in next(m['candidate_links']for m in matrix['modules']if m['id']==mid)],gap=gap,
      profile_scope='Operator fusion legality and theory only; cached pinpoint callers plus public operator kernels.')for mid in ['M07','M09','M22','M23']}
    updates={c:dict(review_v94=dict(evidence=names,graph_frontiers=3200,index_query_forks=640,refined_native_regimes=22,refined_native_executions=9566,
      BF16_to_INT8_rounding_witness=[42,43],exact_real_identity_not_native_precision=True,new_pairs=0,new_groups=0,
      single_interface_graph_no_additive_credits=True,native_or_all_readers_closed=False,implementation=False,native_test=False),
      conclusion_changing_gap=gap)for c in members}
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v93.json'));audit.update(previous_goal_turn='PROGRESS:v93 persisted 7new central relations/shared resource/deadline contracts;201 post checks',input_checkpoint=93,followup_checkpoint=94)
    for r in audit['requirements']:
        if r['id']=='ALL_MODULE_CONDITIONAL_BRANCHES':r['branch_records']=356
        if r['id']=='THEORY_AND_MEASURED_COMPARISON':r['evidence']+=names
    audit['operator_target_instruction']=dict(scope='Code only answers fusion/split/reorganization legality, necessary work and capacity bounds. Implementation selection/readiness/consumers are included only when they change an operator conclusion.',
      new_framework_research=False,framework_records_are_provenance_not_new_target=True)
    audit['norm_quant_refinement_v94']=dict(precision_gate='BF16-rounded chain vs public FP32-normalized fused branches',finite_math_not_native_test=True,
      no_whole_candidate_module_or_goal_closed=True)
    write(O/'whole-scope-requirement-audit-v94.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v94.json',dict(status='PROFILE_OPERATOR_PRECISION_QUESTION_THEN_PINPOINT_NATIVE_OPERATOR_SOURCE',
      questions=['Can the positive RMS row scalar cancel inside per-token quantization without changing current BF16-rounded INT8 codes?',
        'Does the existing fused native API preserve this precision boundary?',
        'What GM interfaces remain after one shared quant producer and which weight work still limits the stated domain?'],
      cached_target_source_files_reread=1,new_target_source_files=0,new_public_operator_kernel_files=5,new_public_commit_docs=1,
      no_framework_wide_research=True,no_model_Torch_NPU_capture_or_server_work=True,overall_complete=False))
    write(O/'norm-quant-validation-history-v94.json',dict(primary_retrieval_limitations=['TwoHTTP200 documentation URLs returned client bootstrap HTML; not treated as usable document bodies. Web text and pinned public repo MD provide operator contract.',
      'Directory entry URL returned a list; explicit commit-and-filename API corrected it; failed lookup not used as source evidence.'],
      scope_corrections=['Deduplicated cost cases with irrelevant qr-retention flag and single-consumer joint-GEMM alias;48 stated interface cases, not48 independent opportunities.',
      'F32/BF16 arithmetic witness is a generic valid input in the stated model, not a captured tensor or native CANN test.'],
      no_remote_or_native_jobs=True,all_local_jobs_finished=True,overall_complete=False))
    remaining=list(read(O/'module-coverage-gaps-v93.json')['remaining'])
    remaining.append('v94: QR norm/quant precision and joint interfaces refined, not a new independent candidate. 3200 graph frontiers/640 index forks/14080 oldTASKs;22 regimes9566 calls no promotion. Public fused keys1/2 keep FP32 norm while current native chain materializes BF16. Declared N1280 gamma1 eps1e-6 witness code42 vs43 forbids unconditional bitwise substitution/cancellation. Preserve BF16 inUB; exact-real row scalar still observed by outputscale/dequant. 48 deduplicated logical graph cases over6 observed M;A3 INT8 VC GM bridge retained, no HBM/speedup. QB+IndexQ47.18592MB distinctweights conditional fixedB/L1 onegrant cut14848cycles;withQA19115, same INT8 compute720 and BF16200 independent max, no mixed-issue sum. ExistingREL73-10/HG73/HG74/HG80 refined;122/25/473 unchanged.356 branches24 semantic;9/1352 pending and849/16/3136 uncertainty;lowest all-module MFU not established;Goal ACTIVE.')
    report='''本轮只回答 Q Norm→动态量化→Q_B/Index Query 的融合合法性与理论成本问题，未进行设备运行或框架扩展研究。

已有 profile 的3200组图内边界已连到Q_A、RmsNorm、DynamicQuant和Q_B的原生TASK，其中640组还有第二次同形量化与Index Query。共复用14080条既有TASK；2560组有阶段归属，640组仍阶段未知。另按类型筛出22个1280维norm/quant组合、9566次既有执行，行数为2、3、800、816、830、834。匹配形状不单独证明指针、参数或内容相同，也未提升新的执行归属。

实数域中，正标量c满足Q(cz)=Q(z)，而量化scale仍是c·max|z|/127。这个恒等式不能直接用于当前链：RmsNorm先输出BF16，再交给DynamicQuant。声明的N=1280、gamma=1、epsilon=1e-6模型，输入前两项为1和43/128，其余为0；FP32归一化前两项约33.89505/11.38662，BF16舍入为34/11.375。直接FP32量化或忽略公共行缩放时第二个码为43，保留BF16舍入后为42。该输入和运算模型不是实测张量或设备复现；它足以否定该声明域中的无条件位级等价，尚不能判断模型容差下是否可接受。

核对官方算子源码的固定commit后，AddRmsNormDynamicQuant的普通和单行分支确实让FP32归一化结果直接进入量化，没有归一化BF16的中间舍入；xOut是Add结果。其余tiling分支、实际安装版本、空x2适配、全部读者和允许误差仍需独立核对。因此，现有融合API是精度与版本受限候选。保留当前BF16舍入于UB中的自定义融合仍可登记，不因需要舍入就必须把qr写回GM。正标量若延迟到行scale/反量化使用，也不能凭INT8码不变就删除其数据相关计算。

量化还保留行内依赖：同一已见前缀[1,0.25]，未见后缀分别为0.5或2时，第一个最终量化码为127或64。不可撤销的一遍量化需要该行amax；独立行仍可流水，临时代码加修正、重算等另建域，未被此反例排除。

联合接口模型以当前两次量化为一个基线，避免重复记账。M=3、K=1280、两路量化消费者时，所列GM接口值共38448B；仅共享一次量化为26916B，差11532B，与v61旧结果一致。再融合norm并且qr无其他GM读者时为11556B，联合差26892B；若仍须写出qr则为19236B，差19212B。在还满足两路GEMM共用输入tile及scale服务条件时，前一种模型为7704B，联合差30744B。它们是替代计算图的值访问模型，不能作为独立收益相加。A3当前Vector产INT8、Cube消费的域中，INT8 GM桥仍在；GM值字节不等于HBM流量。

两套INT8权重仍合计47185920B。在常规稠密、完整输出、未压缩互异权重、强制L1→L0B、24AIC且窗口仅一份初始L0B额度的域中，联合服务下界为14848周期；含Q_A权重则为19115周期。窗口覆盖全部计价服务，初态无已计算部分结果；没有强制整张输入等待，也没有从某个实测TASK起点套界。单独同模式INT8有效计算下界为720周期，Q_A的BF16参考为200周期；不同模式未直接相加。实际输入路径、压缩/重用、频率、Vector归约和关键路径未定，不能认定物理瓶颈或给出提速。

本轮细化已有REL73-10与三组联合合同，中央关系仍为122条/25组，473对待汇入或复核；未把这些细化算作新的独立融合收益。分支台账356条、24个语义模块，全部目标尚未完成：9个regime/1352次执行仍缺关键字段，849条PMU缺失、16张阶段未知图、3136条模块归属歧义未变，也尚未建立全模块同口径最低MFU。
'''
    action=('继续原Goal，v94PROGRESS且v93PROGRESS已201checks验证。先读norm-quant-summary-v94/PROFILE-FINDINGS-v94/source-contract/native-frontiers/precision-row/interface-resource/variants/fullscopeaudit。'
      '用户只要算子融合和理论上限；profile优先，只定点查影响依赖精度布局读者/资源的代码。不得做framework通读或把框架穷举当验收。'
      '本轮3200QRnorm/quant/QB图边界640IndexQ forks复用14080TASKs，22regimes9566calls,root2560/640unknown,indexroot512;no new executions/promotions。'
      'Exactrealc>0 Q(cz)=Q(z),但scale=c*amax/127仍可观察；N1280 gamma1 eps1e-6 x[1,43/128,0...] F32norm→BF16[34,11.375]使INT8第二码42vs43。generic arithmetic witness notcapturedvalue/nativeCANN，allowabletolerance/zero/nonfinite/loadedprecision仍未知。'
      'Publicops-nn commita61e016a71f422cb280af0c024975f13629ca100 AddRmsNormDynamicQuant key1/2直接FLOATnorm→quant，无BF16qrround；xOut是Add，notqr。Installed9.2/otherkeys/nullx2/allreaders未闭；保留BF16roundinUB可条件融合，不可直接claim API bitwise dropin。'
      '两量化基线38,448B，sharequant26,916差11,532；normfused+share noqrstore11,556差26,892，keepqr19,236差19,212；jointinput/scaleread noqr7,704差30,744，都是同基线替代GM值模型不相加，INT8VC GM桥仍在notHBM。48去重case不是48新机会。'
      'QB+IndexQ47,185,920B固定B/L1distinctuncompressed/24AIC/onegrant/nopartialsconditional14848cycles；含QA19115，同INT8compute720，BF16QA200 separate max,notmixedsum。Deadline bothrequiredoutputs notearlierQB-only；nofixedtasktimeorigin/fullinputbarrier。'
      '现有REL73-10/HG73Q/HG74Q/HG80refined0newpairs0groups，122/25/473；356records24semantic/26;2073/285598review9/1352pending849PMU16phase3136ambiguityunchanged0wholeclosed。No allmodulelowestphysicalMFUoractualgain。'
      '下一步优先其他影响融合/理论结论且尚未定案的直接算子边界；复用已有答案，不再复读QRscale或HCfork。现有profile/frozen/math/必要primary，no server/model/Torch/NPU/capture/service/upload。GoalACTIVE/TaskACTIVE/LoopEVALUATING。verify_norm_quant_v94要PASS，不重跑已成finalizer。')
    primary=N/'norm-quant-primary-v94';primary_files=[p for p in primary.iterdir()if p.is_file()]
    jobs=[dict(kind='profile',status='pass',description='Bind3200existingQRnorm/quantchains640IndexQforks/14080TASKs and22regimes9566calls;typedBF16precisionboundary,phase/identitygatesretained.',
      artifacts=[O/names[0],O/names[1],Path(read(O/names[0])['parts'][-1]['path'])]),
      dict(kind='design-check',status='pass',description='BF16/INT8roundingwitness,2300exactrationalidentitycases,48deduplicatedlogicalinterfacesand2conditionaljointweightcuts;noNativeprecisionorHBMspeedupclaim.',
        artifacts=[O/n for n in names[2:]]+[O/'norm-quant-analysis-preflight-v94.json',O/'norm-quant-validation-history-v94.json',O/'whole-scope-requirement-audit-v94.json',
          O/'typed-capacity-primary-contract-v89.json',O/'A3-datapath-primary-evidence-v80.json',R/'fusion-scan.json']+primary_files)]
    persist_refinement(version=94,previous=93,title='QR norm dynamic quant precision, shared interfaces and typed weight cuts',
      summary=dict(filename='norm-quant-summary-v94.json',previous_turn_classification='PROGRESS:v93nativeHCattentiondeadlines/sharedresourcecutsand7relationspersisted',
        native_frontiers=3200,index_query_forks=640,reused_native_TASKs=14080,observed_regimes=22,observed_executions=9566,new_native_promotions=0,
        exact_real_cases=2300,logical_interface_cases=48,conditional_weight_cuts=2,new_public_operator_source_files=5,new_public_commit_docs=1,new_target_source_files=0,
        public_fused_API_bitwise_drop_in_proved=False,native_precision_or_physical_ceiling_or_speedup_proved=False,
        refined_candidates=3,refined_modules=4,new_semantic_families=1,branch_records=356,semantic_modules=24,
        new_pairs=0,new_groups=0,relation_clauses=122,higher_order_groups=25,pending_central_pair_reconciliation=473,
        scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
        implementation=False,native_test=False,remote_job_status='Alllocalprofile/source/mathworkfinished;no server/model/Torch/NPU/capture/servicework.'),
      changes=['BindtypedQRnorm/quantforks andrecordactualBF16betweenoperators;newoperatorAPIprecisionquestionansweredonlyindeclaredsource/modeldomains.',
        'Distinguishrealrow-scalinginvariancefromBF16quantcodesandobservablerowscale;publicnativefusionnotassumedbitwisecompatible.',
        'RepriceonewholeGMinterfacegraph,retainA3INT8VCbridgeanddistinctweights;48cases/2resourcecutsnoindependentgainaddition.',
        'RefineexistingREL73-10/threejointgroupswithoutpairinflation;356records24semanticandallcriticalunknowns/GoalACTIVEretained.'],
      candidate_updates=updates,module_updates=mods,semantic_branches={'M07':branches},remaining=remaining,action=action,jobs=jobs,report=report,read_first=names+['whole-scope-requirement-audit-v94.json'],
      publication_rows={'F05':'| F05 | Q norm → 动态量化 → Q 投影 | 条件融合／精度受限 | 保留RMS输出BF16舍入与行amax，可在UB衔接量化。公开融合API已查分支直接量化FP32归一化值，不能据名称认定位级等价；与Index Query共用量化时按同一接口图计账。 |'},
      gate_updates=dict(overall_complete=False),scripts=[N/'analyze_norm_quant_v94.py',N/'verify_norm_quant_v94.py',N/'persist_profile_refinement_v90.py'],
      executor_prefix='norm-quant',main_script=Path(__file__),source_read=False)

if __name__=='__main__':main()
