"""Persist one specific processor/argmax fusion subgraph,not framework research."""
from pathlib import Path
from copy import deepcopy
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v90 import persist_refinement

def main():
    assert not(O/'REVIEW-CHECKPOINT-v108.json').exists()
    assert read(O/'stop-mask-preflight-v108.json')['status']=='PASS'
    names=['stop-mask-inputs-v108.json','stop-mask-source-contract-v108.json','stop-mask-fusion-algebra-v108.json',
        'stop-ma<REDACTED_TOKEN>.json','stop-mask-work-and-ceiling-v108.json','stop-mask-composition-gates-v108.json']
    for base in ['candidate-relation-clauses','candidate-relation-coverage','candidate-operator-footprints',
        'candidate-pair-navigation','operator-analysis-coverage-ledger']:
        d=deepcopy(read(O/(base+'-v107.json')));d['stop_mask_review_v108']=dict(evidence=names,
            candidate='F35',new_pairs=0,new_native_atoms_or_regimes=0,all_existing_counts_retained=True)
        write(O/(base+'-v108.json'),d)
    obligations=deepcopy(read(O/'coverage-current-obligations-v107.json'))
    for r in obligations['obligations']:
        if r['id']=='G13':
            r['remaining']='MinTokens structured mask/finite-stop restoration and later finite thinking-force overwrite now have a scoped mergeable argmax/lazy-restoration contract and symbolic direct-query bounds. This is not actual runtime applicability or all-processor closure. Remaining specific penalty-count/mask construction,other genuine readers and captured parameter-bank questions need prior-answer checks;v92 filter/v101 recovery are already reviewed.'
            r['stop_mask_review_v108']=dict(evidence=names,one_specific_operator_subgraph_reviewed=True,actual_flags_or_all_readers_closed=False)
        if r['id']in ['G06','G07','G08']:
            r['stop_mask_review_v108']=dict(evidence=names[2:],reference_domain='NoNaN FLOAT with exact global index tie contract;worst-case direct query/comparison domain for symbolic bounds.',physical_capacity_bound=False)
    obligations.update(parent='coverage-current-obligations-v107.json',new_native_or_whole_closures=0,overall_complete=False)
    write(O/'coverage-current-obligations-v108.json',obligations)
    branches=deepcopy(next(m['branches']for m in read(O/'module-phase-branch-matrix-v107.json')['modules']if m['id']=='M17'))
    gap=('MinTokens leading-mask/structured all--inf finite-stop restoration andlater FLOAT1e9 overwrite are now source/conditional algebra reviewed. '
        'Actualflags/row-owner/stop/force/storageepochs,earlierprocessorvalues,allraw/processed/logprob/entropy/random/mutationreaders,nativeNaN/ties/precision/capture/publication remain gates. '
        'KeepCPUforcingstateeffects;finiteforceisnotabsolutepriority. Logicalseparateplane14RV+RbudgetandlazyreadsarenotHBM/nativegain;symboliccapacityunknown. Otherprocessor/countbodiesnotclosed.')
    for r in branches:
        if r['id']=='M17-OP-09':
            r['evidence']=list(dict.fromkeys(r['evidence']+names))
            r['stop_mask_review_v108']=dict(source_body_reviewed=True,oracle_cases=120000,lazy_cases=20000,merge_checks=360000,
                finite_counterexamples=6,actual_flags_or_all_readers_closed=False,evidence=names)
            r['gap']=gap
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v107.json'))
    audit.update(previous_goal_turn='PROGRESS:v107 operand-evidence capability/static SDK and two stale gap corrections;postcheck PASS149',input_checkpoint=107,followup_checkpoint=108)
    audit['stop_mask_review_v108']=dict(evidence=names,operator_subgraph='MinTokens mask/conditional restoration -> finite forcing -> argmax',
        one_pass_summary_and_lazy_split_conditional=True,worst_case_comparison_query_ceiling='SYMBOLIC_CEILING',
        actual_runtime_or_physical_gain=False,whole_requirement_closed=False)
    write(O/'whole-scope-requirement-audit-v108.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v108.json',dict(status='EXISTING_F35_PROCESSOR_GAP_THEN_FIVE_CACHED_PRECISE_OPERATOR_SOURCES',
        questions=['Does structured MinTokens always leave masked stop logits at -inf?',
            'Can mask/restore and argmax share a mergeable state or lazily read stop logits?',
            'Does later finite forcing preserve the restoration decision and old max candidate?',
            'Which explicit materialized interfaces and direct-query lower bounds apply?'],
        existing_native_boundary_evidence='v28/v65/v102;no actual MinTokens/force branch attestation added',
        cached_source_files=5,new_target_source_files=0,new_device_work=False,no_framework_wide_research=True,overall_complete=False))
    write(O/'stop-mask-validation-history-v108.json',dict(events=[
        'CPU analyzer completed finite oracle/merge checks but initially selected V from a head-model row without that field;stopped before checkpoint/shared mutation.',
        'Read V129280 from the existing sampling_models schema;reran CPU analyzer and independent source/unit/witness verifier before persistence.'],
        target_unchanged=True,new_device_work=False,overall_complete=False))
    remaining=[r['id']+': '+r['title']+'. '+r['remaining']for r in obligations['obligations']]
    remaining.append('Current obligations-v108 remain open. Probe-v107 is PREPARED_NOT_EXECUTED and awaiting explicit reply;no device diagnostic done. This checkpoint is a specific operator conditional contract,not a current native savings/MFU/whole-completion claim.')
    report='''本轮围绕一个具体算子子图补证：MinTokens屏蔽/恢复 → thinking有限值写入 → argmax。原F35台账保留了processor约束，但没有展开这条局部融合合同。复用既有profile边界后，仅定点读取5份缓存源码，没有远程执行、新采集或目标实现。

MinTokens按请求的remaining=min_tokens-current_length，仅屏蔽其前min(max(remaining,0),num_draft_tokens)个draft位置。结构化输出有额外分支：屏蔽后整行全为-inf时，恢复屏蔽前有限的stop logits。恢复值来自已经经过更早penalty/allowed/bad-word处理的输入；不能从更早raw logits恢复而复活禁用token。

因此不能把所有mask简单求并集。例子x=[-inf,4]、stop={1}、结构化输出：原路径恢复4并返回ID1，永久屏蔽会变成全-inf并按声明的最低索引规则返回ID0。原生比较器/NaN和实际flags仍须绑定，不能拿这个参考例子声称当前native分支已执行。

在无NaN、固定合法索引、精确全局ID平局合同的声明域中，可把此子图表示为可合并的三个状态：A为排除后续强制写入位置的屏蔽后最大值/ID；B为同样排除强制位置的有限可恢复stop最大值/ID；H为所有位置在强制写入之前是否存在非-inf值。H必须包含随后会被强制的位置。结构化且H为假时合并A/B，否则用A；再合并强制位置的常数1e9/ID。块间只需合并候选和布尔状态，最终ID需等待全行谓词，但不强制物化整行中间张量。

后续thinking操作写的是有限FLOAT 1e9，不能直接替换成强制token ID：其他logit=1e10仍能胜出。它还会把旧值覆盖为1e9，不能仅把常数与旧最大值取max；例如[2e9,1.5e9]强制位置0后，原ID从0变为1。强制写入也不能移到MinTokens恢复判定之前。CPU force_index、bonus_token_forced及行布局状态影响仍须保留。

无后续强制写入时，还有一种拆分图：先只读未屏蔽的n=V-|S|项，若发现非-inf最大值就无需读取stop值；仅当结构化且未屏蔽项全-inf时，再读取|S|项恢复候选。全-inf且无可恢复有限值时须返回全局最低索引0，不能返回最低未屏蔽索引。该图可少读条件分支数据，但新控制/合流/元数据及延迟需要另计；这是候选图与公式，未实现设备kernel。

显式分别物化的参考基线中，对R条需检查恢复的FLOAT行，gather、isneginf、all和最终argmax的逻辑访问为4TV+14RV+R字节；一遍候选归约图的FLOAT输入参考为4TV，另加索引、标志、候选合并等成本。14RV+R是该基线的可消除中间接口预算，未证明原后端实际按这些plane物化，也不是HBM节省或可叠加到既有F35信用的独立收益。R=0或分支未启用时，不存在这部分收益。

理论上限保留符号：无forcing、无结构化恢复、至少n>=2个任意有限未屏蔽FLOAT值、无先验摘要、按直接查询/比较并从声明存储边界读取的最坏情况域，需要n个word和n-1次比较。若有效容量上界分别为U_storage和U_cmp，则T>=max(4n/U_storage,(n-1)/U_cmp)。这不是每个输入或所有算法的下界，存储初态/缓存/复用需要固定；实际容量尚未绑定，未给原生时间、MFU或数值提速。结构化且n>=2、|S|>=2的对抗输入可需要读V项，也不能当作每次都必须读V项。

CPU参考覆盖120000个四元素子域例子、20000个lazy分支例子和360000个分块合并核对，并给出6个错误改写反例。有限枚举不证明原生实现或全部输入。实际MinTokens/结构化/force开关、全部raw/processed/logprob/entropy/random和mutation读者仍是条件；如需完整处理后logits，仍须生成它们。

本轮细化F35和M17-OP-09，不新增候选、关系、原生执行或整模块关闭。35项模型候选、353条语义记录/26暂定模块、132条条件关系/25组/463待核对对不变。9类/1352次执行的关键字段、全模块最低MFU和整体物理理论上限仍未完成。v107自有小图探针仍待明确授权，未运行。
'''
    action=('继续原Goal,v108PROGRESS：先读stop-mask-summary/report/source-contract/algebra/validation/work-ceiling/composition及current-obligations-v108。'
        'v107已PASS149，本轮针对F35/M17OP9真实processor缺口：MinTokens structured全-inf恢复finite pre-MinTokens stop，后续thinking覆盖有限FLOAT1e9且保留CPUstate。'
        'NoNaN/globalIDtie域A/B/H tile-summary：A/B排除forced位置，H仍包含forced；restore后再merge常数。不能maskunion/rawrestore/force直返ID/max旧winner/前移force/删Hforced。'
        '无force lazy先n未mask项，仅structured且全部-inf时读s stop；all-inf最后返回全局0。120000oracle/20000lazy/360000merge和6counterexamples为CPU有限参考非nativeproof。'
        '显式materialized gather/predicate/all+argmax 4TV+14RV+R vs一遍4TV加新summarymetadata，非native/HBM/可叠加F35gain；R及actualflags未知。'
        'SYMBOLIC noforcing/noStructured n>=2 arbitraryfinite/directquery/noinitialresidency worstcase nwords,n-1comparisons；T>=max(4n/Ustorage,(n-1)/Ucmp)，实际容量未知非allalgorithms/perinstance。'
        '下步继续具名operator证据：如penaltycount/mask构造需先查prioranswers再定点body，或真正loadedbank/reader问题，别重读v92pivot/v101recovery/v106projection/v108MinTokens。'
        '探针v107仍PREPARED_NOT_EXECUTED，async明确同意前不得run；获准先跳过占用卡，不kill/startstoppedcontainer。'
        '35/353/26、132/25/463、813/1297/46、2073/285598conditional、9/1352pending、849PMU16phase3136ambiguity、0whole/noallMFU不变。'
        'GoalACTIVE TaskACTIVE Loop004EVALUATING；verify_stop_mask_fusion_v108 mustPASS；不重跑完成finalizers/改已完成artifacts。')
    inputs=[Path(r['path'])for r in read(O/names[0])['inputs']]
    persist_refinement(version=108,previous=107,title='Structured stop-mask restoration and forced-value argmax fusion',
        summary=dict(filename='stop-mask-summary-v108.json',previous_turn_classification='PROGRESS:v107 operand evidence gate and stale-gap corrections;postcheck PASS149',
            cached_source_files=5,new_target_source_files=0,new_operator_subgraph_contracts=1,oracle_cases=120000,lazy_cases=20000,merge_checks=360000,counterexamples=6,
            refined_candidates=1,refined_modules=1,branch_records=353,semantic_modules=26,new_pairs=0,new_groups=0,
            relation_clauses=132,higher_order_groups=25,pending_central_pair_reconciliation=463,
            partial_graph_node_rows=813,partial_graph_edge_rows=1297,structural_pair_hints=46,
            scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
            new_native_promotions=0,ceiling_status='SYMBOLIC_CEILING',actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,
            device_probe_status='PREPARED_NOT_EXECUTED;PENDING_REPLY',implementation=False,native_test=False,
            remote_job_status='CPU-only cached-source/operator algebra/finite references finished;no remote/model/Torch/NPU/capture. Probe v107 not executed.'),
        changes=['Review exact MinTokens finite-stop restoration and later finite forcing order;reject six unsound rewrites.',
            'Derive mergeable A/B/H argmax and no-force lazy restoration candidate graphs with full state/reader gates.',
            'Price one explicit materialized interface baseline and scoped symbolic direct-query/comparison lower bound;no native gain claim.'],
        candidate_updates={'F35':dict(review_v108=dict(evidence=names,scoped_MinTokens_force_argmax_contract=True,oracle_cases=120000,
            lazy_cases=20000,merge_checks=360000,counterexamples=6,ceiling_status='SYMBOLIC_CEILING',
            actual_flags_or_all_consumers_closed=False,new_native_promotions=0,implementation=False,native_test=False),conclusion_changing_gap=gap)},
        module_updates={'M17':dict(candidates=['F35'],finding='MinTokens structured stop restoration plus later finite thinking-force overwrite can conditionally use a mergeable A/B/H argmax summary or no-force lazy stop reads;preserve input rounding,all score/mutation observers and forcing state. Source/native applicability still conditional.',evidence=names,gap=gap)},
        semantic_branches={'M17':branches},remaining=remaining,action=action,
        jobs=[dict(kind='profile',status='pass',description='Existing F35 profile boundaries then five exact cached operator source files;MinTokens/force native applicability remains unbound.',artifacts=[O/n for n in names]+inputs),
            dict(kind='design-check',status='pass',description='CPU120000oracle/20000lazy/360000merge checks,six unsound-rewrite witnesses and scoped symbolic query/comparison resource bound;no target implementation.',artifacts=[O/'stop-mask-preflight-v108.json',O/'stop-mask-validation-history-v108.json',O/'whole-scope-requirement-audit-v108.json',O/'coverage-current-obligations-v108.json'])],
        report=report,read_first=names+['coverage-current-obligations-v108.json'],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/'analyze_stop_mask_fusion_v108.py',N/'verify_stop_mask_fusion_v108.py',N/'persist_profile_refinement_v90.py'],
        executor_prefix='stop-mask',main_script=Path(__file__),source_read=False)

if __name__=='__main__':main()
