"""Persist current requirement dependencies and source-resolved PCP operator selectors."""
from pathlib import Path
from copy import deepcopy
import hashlib
from finalize_origin_coverage_v9 import N,O,R,read,write
from persist_profile_refinement_v90 import persist_refinement

def main():
    assert not(O/'REVIEW-CHECKPOINT-v105.json').exists()
    assert read(O/'coverage-dependency-analysis-preflight-v105.json')['status']=='PASS'
    names=['coverage-dependency-inputs-v105.json','operator-selector-source-index-v105.json','operator-selector-contract-v105.json',
        'operator-selector-case-models-v105.json','coverage-branch-reconciliations-v105.json',
        'coverage-semantic-evidence-audit-v105.json','coverage-gap-history-before-v105.json',
        'coverage-current-obligations-v105.json','coverage-gap-history-routing-v105.json']
    gap=('Exact PCP helper source bodies are now read. Factory uses context PCP plus Ascend singleton DSA-CP and rejects both enabled;builder decode sharding also requires DCP1 and speculative_config=None. '
        'Context/builder/draft-clone/loaded historical objects remain unbound. Raw C65536 binds3200 target records,not loaded ABI or receiver demand;1272 capacities remain unknown. '
        '353 current semantic rows/391 evidence-file integrity checks and14 obligation groups do not prove complete branches,all readers/epochs/precision/composed costs,physical ceilings or MFU. '
        'Keep9native regimes/1352executions and849PMU/16phase/3136ambiguity open; no whole closure.')
    rel=deepcopy(read(O/'candidate-relation-clauses-v104.json'))
    rel.update(parent_relation_index='candidate-relation-clauses-v104.json',new_pair_clauses=0,new_higher_order_groups=0,
        coverage_dependency_review_v105=dict(evidence=names,
            scope='Exact selector source refines applicability of existing F31/F32 PCP domains;source K5 puredecode is replicated in this predicate. No new pair or runtime config attestation.'),overall_complete=False)
    write(O/'candidate-relation-clauses-v105.json',rel)
    cov=deepcopy(read(O/'candidate-relation-coverage-v104.json'))
    cov.update(parent='candidate-relation-coverage-v104.json',parent_relation_index='candidate-relation-clauses-v105.json',
        coverage_dependency_review_v105=dict(evidence=names,new_pairs=0,new_groups=0,new_native_promotions=0),overall_complete=False)
    write(O/'candidate-relation-coverage-v105.json',cov)
    fps=deepcopy(read(O/'candidate-operator-footprints-v104.json'));ix={r['id']:r for r in read(O/'candidate-record-index-v104.json')['model_candidates']}
    for r in fps['candidates']:
        h=hashlib.sha256((R/'candidates'/(r['id']+'.json')).read_bytes()).hexdigest();assert h==ix[r['id']]['master_sha256']
        r['captured_master_sha256']=h;r['captured_master_version']=104
        if r['id']in ['F21','F31','F32']:r['coverage_dependency_review_v105']=dict(evidence=names,new_native_atoms=0,actual_configuration_or_ABI_promotion=False)
    fps.update(parent='candidate-operator-footprints-v104.json',overall_complete=False);write(O/'candidate-operator-footprints-v105.json',fps)
    nav=deepcopy(read(O/'candidate-pair-navigation-v104.json'))
    nav.update(parent='candidate-pair-navigation-v104.json',parent_footprints='candidate-operator-footprints-v105.json',parent_relation_index='candidate-relation-clauses-v105.json',
        coverage_dependency_review_v105=dict(evidence=names,existing_domains_refined_not_new_pair_coverage=True),overall_complete=False)
    write(O/'candidate-pair-navigation-v105.json',nav)
    op=deepcopy(read(O/'operator-analysis-coverage-ledger-v104.json'))
    op['coverage_dependency_review_v105']=dict(new_helper_source_files=2,source_case_models=96,new_native_regimes=0,new_native_executions=0,
        new_native_attribution_or_phase_promotions=0,current_C_gap_qualified_with_existing_v98_raw_evidence=True)
    write(O/'operator-analysis-coverage-ledger-v105.json',op)
    branches=read(O/'coverage-branch-reconciliations-v105.json')['semantic_branches']
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v104.json'))
    audit.update(previous_goal_turn='PROGRESS:v104 zero-scheduled evidence and operator-semantic boundary reconciliation;269postchecks',input_checkpoint=104,followup_checkpoint=105)
    for r in audit['requirements']:
        if r['id']in ['MODEL_VERSION_HARDWARE_LOAD','ALL_MODULE_CONDITIONAL_BRANCHES','CRITICAL_EVIDENCE_AND_FINAL_CLOSURE']:
            r.setdefault('evidence',[]).extend(names)
    audit['coverage_dependency_review_v105']=dict(semantic_rows=353,semantic_modules=26,evidence_files=391,
        reference_integrity_not_completeness=True,old_remaining_records_preserved=55,current_obligation_groups=14,original_requirements_retained=9,
        exact_PCP_selector_definitions_now_read=True,source_spec_present_decode_shard=False,
        context_config_or_historical_loaded_objects_attested=False,new_native_promotions=0,whole_requirement_closed=False)
    write(O/'whole-scope-requirement-audit-v105.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v105.json',dict(status='CURRENT353ROW_EVIDENCE_AUDIT_THEN_TWO_MISSING_OPERATOR_SELECTOR_DEFINITIONS',
        questions=['Which historical pending-source claims are still true with the full saved assets?',
            'What exact predicates choose legacy/PCP/base and decode-sharded metadata?',
            'Does configured speculative decoding satisfy the existing PCP decode-shard predicate?',
            'Which nine original requirements remain unproved after retaining all historical constraints?'],
        new_helper_source_files=2,new_public_documents=0,
        read_only_archive_copy_of_known_container=True,no_framework_wide_research=True,
        no_model_Torch_NPU_capture_service_or_container_process=True,overall_complete=False))
    write(O/'coverage-dependency-validation-history-v105.json',dict(refinements=[
        'v104 predecessor revalidated PASS269. All353 semantic rows,147 exact gap texts and55 prior remaining records inspected as evidence-dependency surfaces,not automatically complete/incomplete by label.',
        'Local frozen/source supplements and existing source receipts lacked attention/utils.py and common_cp.py. One xinyufeng SSH login reached166 actual105 root;the old v98 control handle was absent,not treated as live. Credential not stored in scripts/receipts.',
        'Read-only docker archive copied only known model_runner control and two exact helper files. Frozen capture-index and runner hashes matched. No container process/model/Torch/NPU/health/process/configuration query or new capture. Dedicated SSH control closed after read.',
        'Exact enable_pcp context predicate and decode-shard argument predicate now read. Legacy DSA singleton/context PCP mutex retained. Context,Ascend singleton,builder argument and cloned draft config are deliberately not merged into one unobserved runtime value.',
        'Six factory cases and96 builder guard/metadata cases checked. DCP1 is only the explicit modeled constructor precondition;other initialized-group/config/bank validity is assumed,not asserted by the case table. Graph flag does not override spec-present False decode-shard predicate.',
        '353 effective semantic rows retain original evidence plus3 new source-evidence artifacts:391 unique file pins. This checks reference/integrity and explicit four-row refinements,not all underlying content or whole-model branch completeness.',
        'M12-OP19 source-era capacity gap now states partial v98 raw evidence3200C65536 and1272othermissing,while loaded ABI/demand/writeextents remain open. No new native capacity/demand attribution.',
        'All55 old remaining texts preserved verbatim in archive/routing;current remaining becomes14 obligation groups plus archive pointer. All9 original requirements map to at least one open obligation. Historical snapshot counts are no longer current counters.',
        'Source K5 puredecode PCP-specific global restore branch is absent in the stated argument domain;deleting an absent branch earns0incremental savings. This is not total communication0 or actual PCP/path/config attestation.',
        'No branch/pair/group/regime/native phase/graph count promoted. All local jobs terminal;current physical/MFU/critical evidence gaps remain.'],overall_complete=False))
    updates={cid:dict(review_v105=dict(evidence=names,exact_PCP_selectors_source_reviewed=True,
        source_spec_present_decode_shard=False,current_runtime_config_or_historical_helper_binding_proved=False,
        source_case_models=96,semantic_reference_rows=353,evidence_file_pins=391,
        implementation=False,native_test=False,new_runtime_executions=0),conclusion_changing_gap=gap)for cid in ['F21','F31','F32']}
    mods={
        'M00':dict(candidates=['F31','F32'],finding='Exact PCP/context and DSA singleton selector definitions retrieved and linked;legacy+PCP mutex and separate constructor config domain retained. No actual loaded selector/current PCP degree promotion.',evidence=names,gap=gap),
        'M12':dict(candidates=['F21'],finding='Capacity gap qualified with existing v98 rawC65536 on3200 target records;1272other/loaded ABI/receiver counts/write extents/allocator epochs remain unknown. No new work or memory credit.',evidence=names,gap=gap),
        'M21':dict(candidates=['F31','F32'],finding='Previously unread exact PCP helper definitions now source-reviewed. Existing decode sharding requires no speculative config;main spec-present puredecode metadata is local in this domain,global prefill still global. Context/argument/draft epochs remain conditional.',evidence=names,gap=gap)}
    obligations=read(O/'coverage-current-obligations-v105.json')
    remaining=[r['id']+': '+r['title']+'. '+r['remaining']for r in obligations['obligations']]
    remaining.append('Historical status/constraint texts are preserved in coverage-gap-history-before-v105.json and coverage-gap-history-routing-v105.json;do not reuse historical counts or resolved source-access statements as current blockers. Current353row obligations and9 original requirements remain open;minimal-operator-evidence-plan-v98.json is prepared/not executed.')
    report='''本轮审计了当前353条算子语义记录的证据依赖，并针对会影响跨卡算子路径的两个缺失定义补证。输入台账原有388份唯一证据文件；加入新源码合同后核对391份文件。引用存在和哈希匹配只证明资料完整性，不等于这些资料已经证明全部分支、融合或理论上限。

原容器只读取回attention/utils.py与attention/context_parallel/common_cp.py，另读取已知model_runner控制文件校验。166实际主机105身份、原采集索引及控制源码SHA均匹配；未启动容器内进程、模型、Torch/NPU或新采集，没有健康/进程/配置扫描，完成后关闭本轮SSH控制连接。新文件的历史加载对象身份仍未被反向证明。

两个此前未读的定义现在明确：enable_pcp检查当前配置上下文的prefill_context_parallel_size>1；is_pcp_decode_sharding_enabled检查传入配置的PCP>1、DCP=1且speculative_config为None。enable_dsa_cp另读AscendConfig singleton。已有factory禁止legacy DSA-CP和PCP同时开启。上下文、singleton、builder参数与draft clone并非自动同一个配置，不能用单个默认值代替它们的实际身份。

这会改变候选适用范围。在声明的main K=5/speculative_config存在的源码域里，现有PCP helper的decode分片谓词为False；pure decode返回local metadata，不进入该PCP global hidden restore路径。启用PCP本身不会让这段代码自动分片 speculative decode。global batch若有prefill，仍需global metadata和对应缓存/恢复参与；空local query也不能删除全局义务。full graph标志没有覆盖这个谓词。6个factory与96个builder/metadata条件用例已保存；这些不是实际运行标志取证。

已有v42已经登记replicated decode的局部路径，本轮将其精确前提补齐，没有把原本不存在的PCP restore当成新增收益，也没有声称全部通信为0。若要对speculative decode采用另一种分片，需要新的多token因果行、KV发布/恢复、capture及全读者协议，不能只修改选择开关。实际main PCP度、上下文/参数关系、draft clone和已加载helper仍有未绑定条件。

另将M12-OP19的容量描述与v98对齐：3200条target记录已有raw C=65536；1272条其他容量仍缺。原生ABI、same-call receiver demand、跨rank输入上界、写范围和真实分配生命周期仍未知。raw容量不是实际GEMM行数，也没有产生新的流量、显存或算力收益。

旧remaining列表累计55条历史状态，含不同版本的计数和已补齐的源码缺口。本轮逐条原样保留在历史归档及路由表，当前待办改列14项证据义务，覆盖原有9项需求：完整算子范围、版本/加载身份、实际负载分支、9个native操作数、MC2需求与存储安全、读者/代际、数值合法性、组合必要工作、物理能力、图/源码域、候选组合、归属/因果、全模块利用率及具名待复核问题。未以重排而关闭任何原需求。

当前353条记录/26个暂定模块、35项模型候选；131条条件关系/25组联合合同/464对待汇入或复核，局部图813节点行/1297边行未变。9个regime/1352次执行关键字段、849 PMU/16阶段图/3136归属歧义及最低模块MFU均仍未补齐。整体目标保持进行中。
'''
    action=('继续原Goal,v105PROGRESS；v104PASS269，本轮preflight691。先读coverage-dependency-summary-v105/report/current-obligations/semantic-evidence-audit/selector-contract/case-models/wholeaudit。'
        '353effective semanticrows391uniqueevidencepins26modules；引用/哈希完整不等于全部内容/分支证明。55old remaining原样archive，current14obligationgroups+historypointer覆盖9originalrequirements，别再把旧计数/已补源码缺口当当前blocker。'
        '新只读166actual105 knowncontainer取2helper+runnercontrol，cap/runnerSHA匹配；无containerproc/model/Torch/NPU/capture/service/health/process/configscan，SSHcontrol已关闭。Newhelper历史loadedidentity未知。'
        'Exact enable_pcp=current-context PCP>1；enable_dsa_cp=Ascend singleton；factory两者true报错。Decode shard=builderarg PCP>1&DCP1&spec_configNone；spec-present K5 domain纯decode为False/localmetadata，globalprefill仍global，fullgraph不override。Config上下文/singleton/arg/draftclone别默认为同一个。6factory96buildercases，DCPguardpass不是所有ctor有效性。无actualPCP/flagpromotion。'
        'M12OP19 gap补充已有v98 rawC3200=65536、1272otherunknown；loadedABI/demand/allrankbound/writeextent/allocator仍unknown。无newnativephase/regime/branch/pair/group/graphcounts：353/26,131/25/464,813/1297/46,2073/285598conditional9/1352pending849PMU16phase3136ambiguity0whole/no allmoduleMFU。'
        'Next prioritize G13 named truly unexpanded operator/cost questions against prior evidence,e.g M09 query/weight projection shape-specific work/resource scope and optional filter/reader bodies;first reconcile existing v60/v64/v82/v99 answers rather than inventing similar variants. G03/G04 critical operands still need bounded evidence planv98,not more broad code. Current requirement proof dependencies are authoritative. '
        'All jobs terminal. GoalACTIVE TaskACTIVE Loop004EVALUATING；verify_coverage_dependencies_v105 mustPASS；never rerun completed finalizer or alter completed artifacts.')
    inputs=[Path(r['path'])for r in read(O/names[0])['inputs']]
    evidence=[Path(r['path'])for r in read(O/'coverage-semantic-evidence-audit-v105.json')['artifact_pins']]
    sources=[Path(r['local_path'])for r in read(O/'operator-selector-source-index-v105.json')['files']]
    jobs=[dict(kind='profile',status='pass',description='Read-only original-container archive of two missing operator-choice helper definitions with pinned runner control;353 effective-row/391 evidence-file dependency audit and preserved55 historical gaps.',artifacts=[O/n for n in names]+inputs+evidence+sources),
        dict(kind='design-check',status='pass',description='Six selector and96 conditional PCP-builder cases;separate context/singleton/argument domains,partial raw-C evidence and14 open obligation groups covering9 original requirements,no new runtime flag or physical result.',
            artifacts=[O/'coverage-dependency-analysis-preflight-v105.json',O/'coverage-dependency-validation-history-v105.json',O/'whole-scope-requirement-audit-v105.json'])]
    persist_refinement(version=105,previous=104,title='Current operator coverage proof dependencies and exact PCP source selectors',
        summary=dict(filename='coverage-dependency-summary-v105.json',previous_turn_classification='PROGRESS:v104 operator-boundary reconciliation and zero-scheduled evidence;269postchecks',
            semantic_rows=353,semantic_modules=26,evidence_file_pins=391,baseline_evidence_files=388,historical_remaining_records_preserved=55,current_obligation_groups=14,
            original_requirements_retained=9,new_helper_source_files=2,new_public_documents=0,selector_factory_cases=6,PCP_builder_source_cases=96,
            actual_config_or_historical_loaded_helper_objects_attested=False,refined_candidates=3,refined_modules=3,new_semantic_families=0,branch_records=353,
            new_pairs=0,new_groups=0,relation_clauses=131,higher_order_groups=25,pending_central_pair_reconciliation=464,
            partial_graph_node_rows=813,partial_graph_edge_rows=1297,structural_pair_hints=46,scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,
            pending_regime_reconciliation=9,pending_execution_reconciliation=1352,new_native_promotions=0,actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,
            implementation=False,native_test=False,remote_job_status='Read-only source archive task finished;dedicated SSH control closed;all local jobs terminal;no model/device/container-process/capture/service/environment or implementation work.'),
        changes=['Resolve two previously unread exact PCP source selectors using read-only original-container archive with matching runner/capture control.',
            'Derive source spec-present puredecode replicated metadata domain,retaining context/argument/loaded identity conditions and no absent-branch gain.',
            'Audit353 effective semantic rows/391 evidence files;preserve55 historical gaps and replace current status stack with14 open proof obligations covering9 original requirements.',
            'Qualify M12 capacity gap with existing partial raw-attribute proof;no runtime/native/physical or whole-scope promotion.'],
        candidate_updates=updates,module_updates=mods,semantic_branches=branches,remaining=remaining,action=action,jobs=jobs,report=report,
        read_first=names+['whole-scope-requirement-audit-v105.json'],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/'read_operator_selectors_v105.py',N/'analyze_coverage_dependencies_v105.py',N/'verify_coverage_dependencies_v105.py',N/'persist_profile_refinement_v90.py'],
        executor_prefix='coverage-dependency',main_script=Path(__file__),source_read=True)

if __name__=='__main__':main()
