"""Persist conditional Engram mask demand and composition review; no target implementation."""
from pathlib import Path
from copy import deepcopy
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v90 import persist_refinement

def main():
    assert not(O/'REVIEW-CHECKPOINT-v118.json').exists()
    assert read(O/'engram-mask-preflight-v118.json')['status']=='PASS'
    names=['engram-mask-inputs-v118.json','engram-mask-source-contract-v118.json','engram-mask-profile-scope-v118.json',
        'engram-mask-residual-reference-v118.json','engram-ma<REDACTED_TOKEN>.json','engram-ma<REDACTED_TOKEN>.json']
    detail=dict(evidence=names,existing_WKV_executions_reused=208,unrotated_fused_body_expanded=True,
        finite_host_scalar_references=522240,independent_small_demand_sets=3876,conditional_cost_models=26,
        actual_mask_counts_or_bypass_certificate_attested=False,new_pairs=0,new_groups=0,new_native_promotions=0,whole_requirement_closed=False)
    for base in ['candidate-relation-clauses','candidate-relation-coverage','candidate-pair-navigation','candidate-operator-footprints','operator-analysis-coverage-ledger']:
        d=deepcopy(read(O/(base+'-v117.json')));d['Engram_mask_demand_v118']=deepcopy(detail)
        if base=='candidate-relation-clauses':
            rel=next(r for r in d['clauses']if r['id']=='REL73-20')
            rel['mask_demand_v118']=dict(**detail,contract='False gate mask may remove key columns/dot under observer/numeric gates,while value residual/lookup and history remain. Select one demanded row/head partition;mask-driven deletion cannot double-count fullWKVhoist or fullKVmemo hits.')
            group=next(g for g in d['higher_order_groups']if g['id']=='HG84-HC-ENGRAM')
            group['mask_demand_v118']=dict(**detail,contract='Mask-ready demand vs pure-WKV early hoist and typed contentmemo are alternative work/readiness choices. Value-only entries lack key validity;all updatedBF16hidden/HC/aux readers remain.')
        write(O/(base+'-v118.json'),d)
    cov=read(O/'candidate-relation-coverage-v118.json')
    obligations=deepcopy(read(O/'coverage-current-obligations-v117.json'))
    for g in obligations['obligations']:
        if g['id']=='G00':g['remaining']=g['remaining'].replace('355 semantic','356 semantic')
        if g['id']=='G10':g['remaining']=f"Central{cov['scoped_conditional_pairs']} conditional relations/{cov['higher_order_groups']} groups leave{cov['remaining_pair_relations']} pairs to reconcile;"+g['remaining'].split(';',1)[1]
        if g['id']in ['G00','G01','G02','G05','G06','G07','G09','G10','G13']:g['Engram_mask_demand_v118']=deepcopy(detail)
        if g['id']=='G13':g['remaining']=('Engram inactive-mask key/value backward demand and local unrotated fusedgate body are now explicitly reviewed;do not repeat the priorv62 zero*Inf/-0 warning as a new finding. '
            'ActualA/Bmaskcounts,finite/signedzero certificate and itscost,loadedGEMM/nativegateprecision,allkey/value/error/readers,mask-readiness and chosen row/head/memo/hoist schedule remain named evidence questions. '
            'v92filter,v101recovery,v106projection,v108MinTokens,v110penaltyhistogram andv117descriptorcap are prior answers;their actualruntime/owner/provider fields remain unknown.')
    obligations.update(parent='coverage-current-obligations-v117.json',new_native_or_whole_closures=0,overall_complete=False)
    write(O/'coverage-current-obligations-v118.json',obligations)
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v117.json'))
    audit.update(previous_goal_turn='PROGRESS:v117five-entry profiler observability and noninvertible samplingflags;postcheckPASS275',input_checkpoint=117,followup_checkpoint=118)
    audit['Engram_mask_demand_v118']=deepcopy(detail)
    for r in audit['requirements']:
        if r['id']=='ALL_MODULE_CONDITIONAL_BRANCHES':r['branch_records']=356
        if r['id']in ['ALL_MODULE_CONDITIONAL_BRANCHES','CROSS_OPERATOR_LAYER_CARD_ITERATION','THEORY_AND_MEASURED_COMPARISON','CRITICAL_EVIDENCE_AND_FINAL_CLOSURE']:
            r.setdefault('evidence',[]).extend(names)
    write(O/'whole-scope-requirement-audit-v118.json',audit)
    gap=('Falsemask does not make the whole row dead: preserve exactFP32hidden+0*value thenBF16round unless independentlycertifiedbypass. '
        'Conditional key/dot/rotation removal needs completekey/KV/error/FPstatus/hook/reader closure;valueprojection/lookup and history/DEADpublication remain unless separatelyproveddead. '
        'ActualA/B,maskfieldidentity,certificatecost/epochs,loadedrow-shapeGEMM/FP32-HF32/quantizer bits,packing/capture/alias/owner and typedcapacity remainunknown. '
        'Useonehoist/memo/demand/partition schedule,no duplicate work or byte credit. Noactualphysicalfloor/MFU/speedup/native/wholeclosure.')
    branches=deepcopy(next(m['branches']for m in read(O/'module-phase-branch-matrix-v117.json')['modules']if m['id']=='M14'))
    r=next(r for r in branches if r['id']=='M14-OP-19')
    r['evidence']=list(dict.fromkeys(r['evidence']+names));r['unrotated_gate_body_review_v118']=dict(evidence=names,
        local_body_expanded=True,shape_stride_empty_guards_retained=True,source_no_FP_fusion_setting=True,
        actual_current_unrotated_TASK_hit=False,other_TP1_CPU_deviceINT8_dummy_families_not_promoted=True)
    assert all(r['id']!='M14-OP-21'for r in branches)
    branches.append(dict(id='M14-OP-21',branch='Inactive gate mask:exacthidden+0*value residual;conditional key-column/dot/rotation pruning,value-only demand and separatelycertified whole-row bypass;hoist/memo/split-schedule composition.',
        state='SCOPED_CONDITIONAL_FALSE_MASK_DATAFLOW_AND_DEMAND_REVIEW;WHOLE_MODULE_OPEN',evidence=names,gap=gap,
        scope='Per-token false branch in existing common/fused gate output semantics,not a recovered false-mask count or a native implemented optimization.',actual_mask_counts_bound=False))
    assert len(branches)==21
    write(O/'PROFILE-CODE-QUESTION-GATE-v118.json',dict(status='FIVE_WKV_GEOMETRIES_208EXEC_THEN_ONLY_GATE_MASK_AND_KEY_VALUE_DEMAND',
        questions=read(O/names[0])['operator_questions'],source_scope='Existing frozen local common/layer/model clauses plus cached unrotated Triton gate body;no remote or target imports.',
        prior_answers_reused='v62 maskzero special-value caution,rotated gate and lookup contracts;v84 pureWKVhoist/memo andHC;v114hashhistory contracts.',
        no_new_model_Torch_NPU_context_capture_replay_or_implementation=True,
        native_Pad_probe='PREPARED_NOT_EXECUTED;separate pending reply',overall_complete=False))
    write(O/'engram-mask-validation-history-v118.json',dict(events=[
        'Latest v117revalidatedPASS275. All central pending pairs have empty structural hints;no arbitrary blank-pair independence claims generated.',
        'Selected explicit operator question from M14alternative/false-mask scope. Cached unrotated engram_gate.py body exists;v84 pins common/layer/model but its pure-WKV review didnot expand this kernel body. No remote retrieval.',
        'v62 already statesgate0doesnotpermitreturnhiddenforNaN/Inf/signedzero. This warning is reused,not claimednew;current work separates dead key output from still-demanded value and composes itwithhoist/memo/lookup.',
        'Reuse fiveBF16WKVgeometries/208executions. Tlocaldescriptor doesnot identify global requests or actualmaskA.32phase-unbound preserved;no BOOL24field identity or TPsum promotion.',
        'Host scalar reference checks all65280finiteBF16hidden values against8value classes=522240cases. Four negativezero/positivevalue rawcopy mismatches,plusInf/NaN witnesses. No native precision/NaNpayload/FPstatus claim.',
        '26demand models use conventionaldenseMAC D*H*(4A+T-B),not universalallalgorithmnecessity. Certificates and shape-dependentGEMMprecision/readers remain conditional;certificateconstruction costcannotbe free.',
        'T3,A1,B0 hasMACratio7/15 butactive-full/inactive-value schedule requests6weightblocks vs5 baseline;value-first/key-active requests4inputrows vs3. Weightcallcontent/uniquecontents differ fromHBMtraffic.',
        'Independent output-coordinate demand enumeration3876cases andsource/numerical/coveragepreflightPASS265. Oneexplicit M14inactive-mask semanticrow added355→356,oldunrotatedgenericrow retained/refined.',
        'No new candidate/pair/group/native/wholeclosure;F24/F25existingrelation/HG84composition refined. Pad44B stillnotrunpendingitsseparate explicitreply. No server/NPU/model/target source or environment action.'],overall_complete=False))
    remaining=[g['id']+': '+g['title']+'. '+g['remaining']for g in obligations['obligations']]
    remaining.append('v118:356semanticrows26modulesincludesexplicitM14false-mask key/value demand.522240hostscalarrefs/3876small demandsets/26conditionalmodels,208WKVexec reused;A/B/certificate/precision/readersremainunknown.134pairs26groups461pendingunchanged;Pad44Bnotrun.')
    report='''本轮展开Engram掩码为假的算子需求，补读已缓存的非旋转Triton gate函数体，复用已有旋转路径及WKV profile，没有运行服务器或NPU。v62已经指出“gate为0也不能直接返回hidden”；本轮没有把这条旧警告当作新发现，而是进一步拆清key/value需求与组合代价。

普通旋转路径先做hidden旋转、key/hidden RMS、加权dot和gate，再把token_mask为假的gate填为+0；非旋转融合体同样在全部RMS/dot之后选择gate=0，随后仍加载共享value，计算hidden+gate*value并写回hidden dtype。后者仅在rotation=None、NPU和特定能力guard成立时调用，stride不符抛错，零token不发kernel。它使用(tokens,HC)程序网格，H5120补到BLOCK8192、num_warps4、enable_fp_fusion=False都是源码设置，不证明当前910C的执行分支、物理占用率、实际指令或8192元素必要工作。已有208次WKV记录继续归于原来的旋转普通路径范围，没有替换成这个融合体。

在有效地址/shape、没有其他key/KV/FPstatus/错误/hook读者、并保留数值与状态合同的域内，false-mask行的key投影、key/hidden norm、weighted dot和gate旋转不再影响输出；真正留下的是 roundBF16(addFP32(float(hidden),mulFP32(+0,float(value))))。因此可以研究key/value拆分，不能把整行、lookup或历史写入一起删除。0*Inf、0*NaN仍会使输出NaN；hidden=-0、value为正的有限值时，原残差变为+0，直接复制hidden会改变位。对全部65280种有限BF16 hidden及8种value类别，522240项主机标量参考通过。该结果验证此标量域，不是原生NPU精度或NaN payload测试。

对有限BF16 h/v及分离的IEEE RNE运算，直接复制h的精确标量条件是h不是-0，或v的signbit为1。一种更简单的充分整行合同是所有hidden有限且不是-0，所有投影后的value有限。当前没有这样的实际负载证书；输入有限也不能保证投影不溢出。证书建立、扫描、版本检查、存储和就绪时间都要计价；若先算完value才验证有限性，这份投影工作已经发生，不能再算作节省。保留NaN/Inf/有符号零等原行为的value摘要或分类接口是另一个可能的表示域，本轮没有把完整value GEMM称为所有算法的必要工作。

令T为本地输入描述行数、A为真掩码行数、B为另有充分证书可整行绕过的假掩码行数，0≤B≤T-A。模型D6144、H5120、HC=4下，选定常规稠密点积域的需求参考为 MAC=D*H*(4A+T-B)，原基线为D*H*5T；私有BF16 KV值数同样从5HT变为H*(4A+T-B)。B=0时所有行仍需value，只有key按A行计算。26个形状/条件模型和3876个独立小维度输出坐标集合核对通过。它们是明确算法域的计算/接口参考，不是全算法下界，不是HBM流量或数值物理上限。

拆分代价不能遗漏。若先算所有R=T-B行value，再算A行key，逻辑输入接口会引用R+A行；若先算A行完整KV，再算其余R-A行value，每行输入只引用一次，但两个调用都请求value权重块。当0<A<R时，后者每调用权重内容合计6块，而原来是5块。以profile中T3形状、假设A1/B0为例，MAC参考降到7/15；完整活跃行+非活跃value拆法的权重调用内容却是377487360B，对比基线314572800B，增加20%。value优先拆法则读4行逻辑feature，对比基线3行。唯一冷权重内容仍为314572800B；这些数量都不能直接当成实际HBM或提速。仅A=0时四块key权重整体无需求，value块仍为62914560B。新的融合分块可以减少接口重复，但还要支付tiling、mask、精度、调度和资源代价。

与已有方案的组合也明确收紧：v84若已经提前算了全WKV，之后发现false mask不能追溯节省已发生计算；应比较等mask就绪后按需算与提前全算的工作/重叠权衡。value-only缓存项不能算完整KV命中，后来true-mask读者仍缺key，必须保留内容、表/权重/量化/布局/精度generation及有效字段标签。value仍需求时，它依赖相同6144维lookup特征，不能同时领取F24整行UVA/lookup删除收益。即使gate lookup可删，v114的DEAD/pad/history cache写入、commit/rollback、其他层读者及集体通信参与仍须单独审查。

本轮复用5个WKV regime、208次执行，保留32次phase未绑定及TP副本口径。真实A/B、mask字段身份、负载证书、加载kernel、改变GEMM行数/列数后的舍入、量化输出耦合、所有读者和mask-ready时间都未知；80次待定BOOL[24]没有被指定为Engram mask。现有TASK时间不能当作这项改写的可删除延迟。必要工作与同精度/层级能力上界仍须分别证明，再组合时间下界；本轮没有获得新的数值物理时间下界、MFU或提速。

M14新增一条显式false-mask语义行，355→356条、26个暂定模块，原非旋转/其他分支条目保留并补入函数体证据。F24/F25既有REL73-20与HG84联合合同细化，134条条件关系、26个联合组、461对待核对不变。9类/1352次关键原生字段、849缺PMU、16个未绑定phase图、3136项含糊归属和0整模块关闭保持；整体目标继续。v111单独的44B Pad探针仍未执行。
'''
    action=('继续原Goal，v118PROGRESS。先读engram-mask-summary/source-contract/backward-demand/composition/current-obligations-v118；verify_engram_mask_v118须PASS。'
        'v117manifest72files/PASS275，v118preflightPASS265。新补本地缓存unrotatedTritongatebody+M14falsemask实际输出语义，非profilebranchpromotion。'
        'v62已知0*Inf/-0不可直接copy，别重复当新发现。新参考522240finiteBF16hostresidual、3876independentdemandsets、26models。'
        'Atrue_mask,Bcertifiedfalsebypass,R=T-B，常规denseMAC D6144*H5120*(4A+R)vs5T。B0 false仍需value/lookup；key/norm/dot/rotation仅在allreader/error/precisiongates下死。certificatecost/epoch不可free，先算value验finite不能retro-saveMAC。'
        'T3A1B0MAC7/15，但activefull+inactivevalue请求6weightblocks377487360Bvs5blocks314572800；value-first+keyactive输入4rowvs3。唯一冷内容≠percall内容≠HBM。只A0减整4keyweights，不能用A比例减weight。'
        'masklate不能和fullWKVhoist重复省；value-onlymemo不是fullKVhit，后true需keys；lookup删除不删除history/DEAD/rollback/otherlayers/collective。AllTupdatedhidden/HC/auxpublication仍保留。'
        'Reuse208WKVexec包括32phaseunbound，Tlocaldescriptor非globalunique;A/Bactualunknown，BOOL24待定80次未归Engram。nativeunrotatedguard/stride/empty/noFPfusion源码设置不等于actualA3执行/occupancy。'
        'M14OP21使356semanticrows26modules，134relations26groups461pending、813/1297/46graph、2073/285598conditional9/1352pending849PMU16phase3136ambiguity0whole/noallMFU不变。'
        '全部localCPU工作结束，无server/NPU/model/implementation；v111Pad44B仍待单独async明确reply，原v109一次probe授权已完成不可复用。'
        '继续其他真正未展开operator关系/分支或现有call-bound关键operand/typedcapacity证据；不要再泛查框架或枚举空pair。TaskACTIVE Loop004EVALUATING GoalACTIVE，不改封存v118或重跑finalizer。')
    inputs=[Path(r['path'])for r in read(O/names[0])['inputs']]
    persist_refinement(version=118,previous=117,title='Engram false-mask key/value demand and hoist-memo split tradeoffs',
        summary=dict(filename='engram-mask-summary-v118.json',previous_turn_classification='PROGRESS:v117five-entry profiler observability;PASS275',
            existing_WKV_executions_reused=208,newly_expanded_local_kernel_body=1,finite_host_scalar_references=522240,
            independent_small_demand_sets=3876,conditional_cost_models=26,explicit_false_mask_semantic_rows_added=1,
            actual_mask_counts_or_bypass_certificate_attested=False,refined_candidates=2,refined_modules=1,
            new_pairs=0,new_groups=0,branch_records=356,semantic_modules=26,relation_clauses=134,higher_order_groups=26,pending_central_pair_reconciliation=461,
            partial_graph_node_rows=813,partial_graph_edge_rows=1297,structural_pair_hints=46,
            scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
            new_native_promotions=0,new_device_work=False,implementation=False,native_test=False,
            actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,native_Pad_probe='PREPARED_NOT_EXECUTED;separate pending reply',
            remote_job_status='All localCPU/source/profile analysis finished;no server/Torch/NPU/model/context/capture/replay/implementation. Padprobe notrun.'),
        changes=['Expand cached unrotated gate operatorbody and add explicit inactive-mask semantic demand branch,without reassigning current208WKVexecutionsto it.',
            'Derive conditional key-only pruning,value residual/certificate semantics and26conventionaldense interface/work models;522240hostreferences/3876independentdemandsets.',
            'Reconcile demand with existingWKVhoist/memo/lookup/history;expose duplicate weight/input costs and preserve all loadedprecision/readers/epochs/runtimegates.'],
        candidate_updates={'F25':dict(review_v118=deepcopy(detail),conclusion_changing_gap=gap),
            'F24':dict(review_v118=dict(evidence=names,false_gate_mask_not_automatic_lookup_or_history_deletion=True,
                value_only_WKV_keeps_dense_lookup_row_demand=True,new_native_or_runtime_mask_promotion=False),conclusion_changing_gap=gap)},
        module_updates={'M14':dict(candidates=['F24','F25'],finding='Explicit false-mask demand separates key/dot/rotation from exact value residual and certified bypass;row/column split schedules trade input and weight references,not a proven runtimegain. Localunrotatedbody/guards nowexpanded.',evidence=names,gap=gap)},
        semantic_branches={'M14':branches},remaining=remaining,action=action,
        jobs=[dict(kind='profile',status='pass',description='Reuse5WKVregimes208exec andlocalgate/caller source;preservephase32/localTP/maskfieldunknowns whileexpandingfalsemaskdataflow.',artifacts=[O/n for n in names]+inputs),
            dict(kind='design-check',status='pass',description='522240scalarreferences3876demandsets26models;onefalse-masksemanticrow andexistingREL73-20/HG84compositionrefined,noactualnativegain.',artifacts=[O/'engram-mask-preflight-v118.json',O/'engram-mask-validation-history-v118.json',O/'whole-scope-requirement-audit-v118.json',O/'coverage-current-obligations-v118.json'])],
        report=report,read_first=names+['coverage-current-obligations-v118.json'],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/'analyze_engram_mask_v118.py',N/'verify_engram_mask_v118.py',N/'persist_profile_refinement_v90.py'],
        executor_prefix='engram-mask',main_script=Path(__file__),source_read=True)

if __name__=='__main__':main()
