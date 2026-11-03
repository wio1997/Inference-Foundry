"""Persist operator operand evidence limits and remove two stale unread-body gaps."""
from pathlib import Path
from copy import deepcopy
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v90 import persist_refinement

def main():
    assert not(O/'REVIEW-CHECKPOINT-v107.json').exists()
    assert read(O/'operand-evidence-preflight-v107.json')['status']=='PASS'
    names=['operand-evidence-inputs-v107.json','operand-evidence-source-contract-v107.json',
        'operand-evidence-method-gates-v107.json','operand-evidence-plan-v107.json','sampling-prior-answer-routing-v107.json']
    for base in ['candidate-relation-clauses','candidate-relation-coverage','candidate-operator-footprints',
        'candidate-pair-navigation','operator-analysis-coverage-ledger']:
        d=deepcopy(read(O/(base+'-v106.json')))
        d['operand_evidence_review_v107']=dict(evidence=names,new_relations=0,new_native_promotions=0,
            probe_status='PREPARED_NOT_EXECUTED',all_previous_operator_analysis_retained=True)
        write(O/(base+'-v107.json'),d)
    obligations=deepcopy(read(O/'coverage-current-obligations-v106.json'))
    for r in obligations['obligations']:
        if r['id']in ['G03','G04']:
            r['operand_evidence_review_v107']=dict(evidence=names[:4],
                generic_SDK_getter_not_typed_operand_decoder=True,probe_prepared_not_executed=True,
                new_saved_model_operand_identity_or_runtime_values=0)
        if r['id']=='G13':
            r['remaining']='Observed Indexer projection costs were scoped in v106. Upstream monolithic NPU filter body was already reviewed in v92 and recovery fallback in v101;these are not unread source gaps. Select only genuine operator-changing processor/reader or captured parameter-bank questions after checking prior answers. Actual flags/native modes/all readers remain open.'
            r['prior_answer_routing_v107']='sampling-prior-answer-routing-v107.json'
    obligations.update(parent='coverage-current-obligations-v106.json',new_native_or_whole_closures=0,overall_complete=False)
    write(O/'coverage-current-obligations-v107.json',obligations)
    matrix=read(O/'module-phase-branch-matrix-v106.json')
    branches=deepcopy(next(m['branches']for m in matrix['modules']if m['id']=='M17'))
    replacement=('Exact upstream monolithic NPU filter source/contract already reviewed in v92;recovery fallback in v101. '
        'Do not count these as unread bodies. Actual runtime flags/native precision/RNG,banks and remaining processor/readers still require a specific operator-changing evidence question.')
    changed=[]
    for r in branches:
        if r['id']in ['M17-OP-04','M17-OP-07']:
            assert 'UpstreamTritonfilterbodyandremainingprocessors/readers remainseparateopenquestions.'in r['gap']
            r['gap']=r['gap'].replace('UpstreamTritonfilterbodyandremainingprocessors/readers remainseparateopenquestions.',replacement)
            r['evidence']=list(dict.fromkeys(r['evidence']+['sampling-prior-answer-routing-v107.json','pivot-filter-source-contract-v92.json']))
            r['prior_answer_routing_v107']=dict(stale_unread_body_claim_corrected=True,new_body_analysis=False,all_readers_closed=False)
            changed.append(r['id'])
    assert changed==['M17-OP-04','M17-OP-07']
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v106.json'))
    audit.update(previous_goal_turn='PROGRESS:v106 typed Indexer projection/shared QA-Weight panel;postcheck PASS230',input_checkpoint=106,followup_checkpoint=107)
    audit['operand_evidence_review_v107']=dict(evidence=names,installed_header_matches_saved_v12=True,
        generic_getter_does_not_establish_typed_MemSet_range=True,two_stale_unread_body_rows_corrected=True,
        no_model_operand_closure=True,no_new_device_work=True,whole_requirement_closed=False)
    write(O/'whole-scope-requirement-audit-v107.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v107.json',dict(
        status='SAVED_MISSING_OPERANDS_THEN_EXACT_INSTALLED_SDK_INTERFACE_ONLY',
        questions=['Can available graph getters establish the destination/write extent needed to delete or merge MemSet?',
            'Which count/size fields must not be conflated?',
            'Is the older unread upstream filter body gap already answered?'],
        exact_source_files_copied=4,control_only_model_runner_files=1,SDK_headers=2,Torch_NPU_graph_wrapper_files=1,
        loaded_ABI_or_model_operand_identity_attested=False,new_device_work=False,
        device_probe='PREPARED_NOT_EXECUTED;PENDING_EXPLICIT_REPLY',no_framework_wide_research=True,overall_complete=False))
    remaining=[r['id']+': '+r['title']+'. '+r['remaining']for r in obligations['obligations']]
    remaining.append('Current authority:coverage-current-obligations-v107.json. Own tiny probe is prepared,not executed;approval pending. It would test metadata capability only,not close historical model operand identity. Nine pending native regimes,physical capacities and whole operator scope remain open.')
    gap='Seven MemSet/1272 and two BOOL24/80 remain unresolved. Installed generic graph-task getter has no dedicated typed fill range;opaque argument size is not write extent. New probe is only prepared and has no saved-model correlation. Same-call storage/role/readers/ABI/update epochs remain necessary.'
    mods={
        'M12':dict(candidates=[],finding='Installed SDK can enumerate generic captured task metadata in its declared contract;it does not by itself bind pending MemSet ranges or MC2 receive work. Byte and D32 count units are separated;no delete/merge/capacity shrink promoted.',evidence=names[:4],gap=gap),
        'M19':dict(candidates=[],finding='The pending pad-owned MemSet roles still need exact destination/write extent and all readers. Generic argsSize/opInfoSize cannot substitute for these fields;own tiny capture capability probe is prepared only.',evidence=names[:4],gap=gap),
        'M17':dict(candidates=[],finding='Correct two stale unread upstream-filter claims using the exact already-reviewed v92 source hash and NPU monolithic route. Preserve actual runtime flags/native modes/RNG/parameter-bank/all-reader conditions;no new sampler gain claimed.',evidence=[names[-1],'pivot-filter-source-contract-v92.json'],gap=replacement)}
    report='''本轮只补算子结论所需的证据通路，并修正两处旧台账；没有新的融合收益或数值理论上限。

原profile仍缺7类MemSet共1272次执行的目标地址、实际写入长度、重叠和workspace/payload角色，以及2类BOOL[24]共80次执行的字段和全部读者。已有ConstantPadNd父调用归属不能替代这些字段，因此不能批准删除、合并或缩小相关存储。

从原166容器只读复制的CANN头文件SHA与历史v12一致。安装头文件声明aclmdlRITaskGetParams，但任务类型/union没有专门的MemSet区间结构；kernel argsSize和opInfoSize表示通用参数信息大小，不能当作实际填充字节数。接口声明也不证明已加载库符号、运行时支持或历史模型参数身份。

API单位必须区分：aclrtMemsetAsync的count是字节数，maxCount是容量；aclrtMemsetD32Async的N是uint32元素数，实际写入4N字节，memSize是容量。同为64B写入的受控例子可以分别使用count=64和N=16。这里没有据此推断保存的MemSet使用了哪条API。

已准备一个只处理自有小图的接口能力探针，应用张量合计8240B；驱动上下文和分配器开销不受这个数限制。探针默认只描述计划，不导入Torch、不加载驱动库。启用运行模式需明确授权并先跳过占用卡；计划不加载模型、不replay图、不解引用opaque参数、不修改既有图任务。当前没有执行探针，没有任何新NPU诊断结果。即使探针成功，也只能证明受控调用的接口能力，不能自动补齐原模型9类缺口；仍需原调用/图/参数版本/存储生命周期的精确关联。

M17-OP-04和M17-OP-07仍写着上游Triton过滤主体未读，但v92已针对相同SHA的源文件审查NPU单体路径，v101也已审查recovery fallback。两条旧措辞已更正；实际开关、原生精度、RNG、参数bank和完整读者仍未绑定。纠正重复项不是新增算子分析。

35项模型候选、353条语义记录/26暂定模块、132条条件关系/25联合组/463待核对对保持不变。2073类/285598次执行仍仅为声明范围的条件审查；9类/1352次执行仍待关键字段。没有整模块关闭，没有实测提速、全模块最低MFU或整体物理理论上限结论。
'''
    action=('继续原Goal,v107PROGRESS：先读operand-evidence-summary/report/method-gates/plan/source-contract和current-obligations-v107。'
        'v106已PASS230；本轮原166只读dockerarchive复制4files，runner及acl_rt.h历史匹配；无容器进程/库load/Torch/NPU/model。'
        'Installed genericGetParams taskenum/union无MemSet typedrange，argsSize/opInfoSize非writeextent；bytecount vsD32N(4N)分开。'
        'Own8240Bapplicationpayload capturemetadata probe仅PREPARED_NOT_EXECUTED，用户async问题待回复；显式affirmative前不得--run，不以elapsed作批准。'
        '若获准先读占用跳过占用卡，无kill/startstoppedcontainer授权；probe也不能替代savedmodel同call/graph/update/storageepochs。'
        'v92已审NPUupstreammonolithicfilter/v101recoveryfallback，M17OP4/7旧unread措辞已纠正，不再重复body分析。'
        '下一步优先真实算子目标：从prioranswers排除重复后，审具体processor/readers或参数bank问题；继续fusion必要work/resource/precision/reader合同。'
        '35/353/26、132/25/463、813/1297/46、2073/285598conditional、9/1352pending、849PMU16phase3136ambiguity、0whole/noallMFU均不变。'
        'GoalACTIVE/TaskACTIVE/Loop004EVALUATING；never rerun completed finalizers or mutate completed checkpoint artifacts；verify_operand_evidence_v107 mustPASS。')
    inputs=[Path(r['path'])for r in read(O/names[0])['inputs']]
    persist_refinement(version=107,previous=106,title='Operator operand evidence capability and stale gap correction',
        summary=dict(filename='operand-evidence-summary-v107.json',previous_turn_classification='PROGRESS:v106 Indexer projection models and upstream QA-Weight relation;postcheck PASS230',
            copied_source_files=4,installed_header_historical_match=True,evidence_methods=5,unit_reference_cases=12,
            corrected_stale_unread_branch_rows=2,new_fusion_findings=0,refined_candidates=0,refined_modules=3,
            branch_records=353,semantic_modules=26,new_pairs=0,new_groups=0,relation_clauses=132,higher_order_groups=25,pending_central_pair_reconciliation=463,
            partial_graph_node_rows=813,partial_graph_edge_rows=1297,structural_pair_hints=46,
            scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
            new_native_promotions=0,probe_status='PREPARED_NOT_EXECUTED',probe_authorization='PENDING_ASYNC_REPLY',
            actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,implementation=False,native_test=False,
            remote_job_status='Read-only source archive copy finished;SSH closed. No model/container process/library load/Torch/NPU/capture. Probe prepared only;pending explicit reply.'),
        changes=['Pin the installed graph getter contract;do not misread opaque packet size as native fill length.',
            'Separate byte-fill count,D32 element count,capacity and epoch/call identity acceptance conditions.',
            'Prepare but do not execute bounded own-graph capability probe;retain all nine model operand gaps.',
            'Correct two stale unread filter-body claims from exact v92 evidence;no duplicate new analysis or native promotion.'],
        candidate_updates={},module_updates=mods,semantic_branches={'M17':branches},remaining=remaining,action=action,
        jobs=[dict(kind='profile',status='pass',description='Read-only original-container SDK source evidence,installed-header control and explicit operand capability limits;no new device work.',artifacts=[O/n for n in names]+inputs),
            dict(kind='design-check',status='pass',description='CPU-only unit/ABI-reference and evidence-integrity checks;route old filter question to v92;probe remains unexecuted.',artifacts=[O/'operand-evidence-preflight-v107.json',O/'whole-scope-requirement-audit-v107.json',O/'coverage-current-obligations-v107.json'])],
        report=report,read_first=names+['coverage-current-obligations-v107.json'],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/'analyze_operand_evidence_v107.py',N/'verify_operand_evidence_v107.py',N/'persist_profile_refinement_v90.py',N/'read_sdk_operand_contract_v107.py'],
        executor_prefix='operand-evidence',main_script=Path(__file__),source_read=True)

if __name__=='__main__':main()
