"""Audit current evidence dependencies and resolve exact PCP operator selectors."""
from pathlib import Path
from collections import Counter
from copy import deepcopy
from itertools import product
import ast,hashlib,json
from finalize_origin_coverage_v9 import N,O,W,read,write

def pin(p):
    p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def function_part(p,name):
    text=Path(p).read_text();node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name==name)
    return dict(**pin(p),symbol=name,first_line=node.lineno,last_line=node.end_lineno,source=text.splitlines()[node.lineno-1:node.end_lineno])
def segment(p,a,b):return dict(**pin(p),first_line=a,last_line=b,source=Path(p).read_text().splitlines()[a-1:b])

def main():
    assert not(O/'REVIEW-CHECKPOINT-v105.json').exists()
    index=read(O/'operator-selector-source-index-v105.json');by={r['path']:Path(r['local_path'])for r in index['files']}
    root='/vllm-workspace/vllm-ascend/vllm_ascend/'
    pcp=by[root+'attention/utils.py'];decode=by[root+'attention/context_parallel/common_cp.py']
    oldroot=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend'
    utils=oldroot/'utils.py';caller=oldroot/'attention/context_parallel/dsa_v41_cp.py'
    parts=[function_part(pcp,'enable_pcp'),function_part(decode,'is_pcp_decode_sharding_enabled'),
        function_part(utils,'enable_dsa_cp'),function_part(caller,'get_v41_cp_classes'),
        segment(caller,261,278),segment(caller,380,443)]
    write(O/'operator-selector-contract-v105.json',dict(parts=parts,source_index='operator-selector-source-index-v105.json',
        facts=['enable_pcp reads current-context VllmConfig.parallel_config.prefill_context_parallel_size>1.',
            'enable_dsa_cp reads the validated AscendConfig singleton enable_dsa_cp. It is a different configuration source from the context and constructor argument.',
            'Factory rejects simultaneous DSA-CP and PCP;otherwise chooses legacy DSACP, PCP, or base V4.1 classes in that order.',
            'is_pcp_decode_sharding_enabled reads its argument:PCP>1 AND DCP==1 AND speculative_config is None. Full graph mode does not override this predicate.',
            'PCP builder independently rejects DCP!=1, stores that predicate, and uses global_prefill OR decode_sharded to select global metadata. Without either it returns local metadata.',
            'Thus in the declared main K5/speculative-config-present source domain, this existing PCP path does not shard pure decode. Prefill/mixed-global batches still need global metadata even with an empty local query.',
            'Context config,Ascend singleton,builder argument and draft clones require identity/epoch binding. Their values cannot be silently equated. Current source bytes do not attest historical loaded helper objects.'],
        newly_source_resolved=['attention/utils.py:enable_pcp','common_cp.py:is_pcp_decode_sharding_enabled'],
        current_main_PCP_degree_or_loaded_selector_attested=False,new_helper_source_files=2,new_public_documents=0,
        operator_scope_only=True,overall_complete=False))
    factory=[]
    for legacy,p in product([False,True],[1,2,4]):
        factory.append(dict(ascend_singleton_legacy_DSA_CP=legacy,context_prefill_CP=p,
            result='ERROR_MUTEX'if legacy and p>1 else'LEGACY_DSACP'if legacy else'PCP'if p>1 else'BASE_V41'))
    cases=[]
    for p,d,s,prefill,empty,graph in product([1,2,4],[1,2],[False,True],[False,True],[False,True],[False,True]):
        shard=p>1 and d==1 and not s;valid=d==1;needs=prefill or shard
        cases.append(dict(builder_prefill_CP=p,builder_decode_CP=d,speculative_config_present=s,
            global_has_prefill=prefill,local_query_empty=empty,full_graph=graph,
            PCP_explicit_DCP_guard_passes=valid,decode_shard_predicate=shard,
            needs_global_metadata=needs if valid else None,
            metadata_result=('GLOBAL_PLUS_EMPTY_LOCAL'if empty else'GLOBAL_PLUS_LOCAL')if valid and needs else'LOCAL_ONLY'if valid else'EXPLICIT_DCP_GUARD_ERROR',
            actual_case_execution_attested=False))
    assert len(factory)==6 and len(cases)==96
    write(O/'operator-selector-case-models-v105.json',dict(factory_cases=factory,PCP_builder_cases=cases,
        context_and_constructor_configs_not_implicitly_joined=True,
        case_domain='Other constructor/group/bank/config validity is assumed; only the explicit DCP guard and listed metadata predicates are modeled.',
        source_K5_pure_decode=dict(speculative_config_present=True,decode_shard_predicate=False,
            existing_PCP_global_hidden_restore_branch=False,incremental_savings_from_deleting_that_absent_branch=0,
            limitation='PCP-specific metadata/restore branch only. Not all TP/EP/DP communication or an attestation that PCP is current. Cloned draft config must be evaluated separately.'),
        spec_compatible_new_partition='Changing the predicate or metadata protocol to shard speculative decode is a new conditional operator graph requiring multi-token causal rows, KV/publication/restore/capture and all-reader precision contracts;not a flag-only use of the current path.',
        theory_status='INCOMPLETE',no_numeric_ceiling_MFU_or_speedup=True,overall_complete=False))

    matrix=read(O/'module-phase-branch-matrix-v104.json')
    effective=deepcopy(matrix);updated={}
    selector_evidence=['operator-selector-source-index-v105.json','operator-selector-contract-v105.json','operator-selector-case-models-v105.json']
    for m in effective['modules']:
        for b in m['branches']:
            if b['id']in ['M00-OP-03','M21-S01','M21-S14']:
                b['evidence']=list(dict.fromkeys(b['evidence']+selector_evidence))
                b['selector_source_review_v105']=dict(exact_helper_bodies_reviewed=True,evidence=selector_evidence,
                    source_spec_present_decode_shard=False,actual_context_config_or_loaded_helper_attested=False)
                if b['id']=='M21-S01':
                    b['state']='EXACT_PCP_SELECTOR_BODIES_REVIEWED;RUNTIME_CONFIG_AND_LOADED_IDENTITY_OPEN'
                    b['gap']='Exact enable_pcp and decode-shard definitions are now read; enable_dsa_cp source is linked. Context config, Ascend singleton and constructor argument/clone identities differ and remain unbound; actual build-info/binary/selector objects and graph bank epochs are not attested.'
                if b['id']=='M21-S14':
                    b['state']='PCP_SOURCE_SELECTOR_AND_SPEC_PRESENT_DECODE_CASE_REVIEWED;ACTUAL_CONFIG_OPEN'
                    b['gap']='In the declared constructor-argument domain PCP>1,DCP1,speculative_config present,decode sharding is False;pure decode returns local metadata while any global prefill still needs global ownership. Actual context/main/draft config identities,PCP degree,loaded helper and all row/cache/collective/capture contracts remain conditional.'
            if b['id']=='M12-OP-19':
                old='Current166 loaded C/max_output_size and same-call all-rank input bounds/receiver counts are not attested by source defaults or old169hashes.'
                assert b['gap'].startswith(old)
                b['gap']=b['gap'].replace(old,'Raw profile attributes bind C=65536 to3200 targetM3 executions;1272 other capacities remain unknown. Loaded max_output_size ABI and same-call all-rank input bounds/receiver demand are still not attested.',1)
                b['raw_attribute_review_v105']=dict(evidence=['MC2-attribute-to-ceiling-contract-v98.json'],raw_target_capacity_bound=True,loaded_ABI_or_demand_bound=False)
            if m['id']in ['M00','M12','M21']:updated[m['id']]=m['branches']
    write(O/'coverage-branch-reconciliations-v105.json',dict(input_matrix='module-phase-branch-matrix-v104.json',
        semantic_branches=updated,changed_branch_ids=['M00-OP-03','M12-OP-19','M21-S01','M21-S14'],
        new_or_deleted_rows=0,whole_branch_closure_claim=False,overall_complete=False))
    refs=Counter();rows=[];groups={}
    for m in effective['modules']:
        for b in m['branches']:
            refs.update(b['evidence']);gid=groups.setdefault(b['gap'],len(groups))
            rows.append(dict(module=m['id'],branch_id=b['id'],state=b['state'],scope=b['scope'],gap=b['gap'],gap_group=gid,
                evidence=b['evidence'],evidence_reference_valid=all((O/f).is_file()for f in b['evidence']),
                retained_followup_fields=sorted(k for k in b if k not in ['id','branch','state','scope','gap','evidence']),
                applicability='EXCLUDED_LEGACY_REFERENCE'if b['id']=='M07-OP-12'else'NOT_GLOBALLY_CERTIFIED_FROM_THIS_AUDIT',
                whole_branch_completion_proved_by_reference_checks=False))
    assert len(rows)==353 and len(refs)==391 and all(r['evidence_reference_valid']for r in rows)
    pins=[dict(**pin(O/f),referenced_rows=n)for f,n in sorted(refs.items())]
    write(O/'coverage-semantic-evidence-audit-v105.json',dict(rows=rows,artifact_pins=pins,
        input_matrix='module-phase-branch-matrix-v104.json',applied_reconciliations='coverage-branch-reconciliations-v105.json',
        modules=26,semantic_rows=353,baseline_unique_evidence_artifacts=388,unique_evidence_artifacts=391,exact_gap_text_groups=len(groups),
        legacy_exclusion_rows=['M07-OP-12'],scope_universe_still_provisional=True,
        reference_integrity_is_not_content_or_completeness_proof=True,
        all_operator_branches_or_whole_model_complete=False,overall_complete=False))
    write(O/'coverage-gap-history-before-v105.json',read(O/'module-coverage-gaps-v104.json'))
    obligations=[
        ('G00','Original full operator scope and completeness','26 provisional module inventories,353 semantic rows and35 model candidates are indexes;all applicable operator branches and graph variants are not proven exhausted.', ['ALL_MODULE_CONDITIONAL_BRANCHES','CURRENT_FINITE_GRAPH_LEDGER']),
        ('G01','Loaded implementation and operand identity','Bind model/CANN/loaded native ABI,typed arithmetic modes,parameter/layout/callback versions. Two PCP definitions are now read;historical loaded objects remain unknown.', ['MODEL_VERSION_HARDWARE_LOAD']),
        ('G02','Actual fixed-load branch and row domain','Bind runtime guards,valid/accepted/draft/query/visible-key counts,DP/TP/CP/EP ownership and phase/dummy domains. Captured request intent/scheduler scalars do not attest all effective guards.', ['MODEL_VERSION_HARDWARE_LOAD','ALL_MODULE_CONDITIONAL_BRANCHES']),
        ('G03','Nine unresolved native operand regimes','Seven MemSet/1272 executions need destination,length,overlap and workspace/payload role;two BOOL[24]/80 need field/all-reader identity. Missing own attrs/operands are not inferred from neighboring shapes.', ['NATIVE_OPERATOR_REGIMES','CRITICAL_EVIDENCE_AND_FINAL_CLOSURE']),
        ('G04','MC2 receive work and storage safety','Raw C65536 is bound to3200 target records;1272 other capacities remain unknown. Same-call receiver demand/all-rank input bounds,loaded ABI/write extents and actual allocator/capture lifetimes are still missing. C is not GEMM rows.', ['NATIVE_OPERATOR_REGIMES','THEORY_AND_MEASURED_COMPARISON','CRITICAL_EVIDENCE_AND_FINAL_CLOSURE']),
        ('G05','Readers,aliasing and cross-iteration state','Bind complete demanded output/read sets,mutation/error/RNG effects,slot/key/weight epochs and old-reader/queued-writer/retirement lifetimes. Current cache/state publication is baseline.', ['CROSS_OPERATOR_LAYER_CARD_ITERATION','ALL_MODULE_CONDITIONAL_BRANCHES']),
        ('G06','Numerical legality in a declared domain','Preserve or explicitly change BF16/FLOAT/HF32/quant/TopK/RNG/online reduction boundaries. Real-arithmetic identities and finite witnesses do not attest all native precision modes.', ['THEORY_AND_MEASURED_COMPARISON','CROSS_OPERATOR_LAYER_CARD_ITERATION']),
        ('G07','Necessary work and one selected composed graph','Reconcile all readers/new work/packing/recomputation/publication/resource competition for each chosen representation. No duplicate gain or independent resource sum;current conditional cases are not all variants.', ['THEORY_AND_MEASURED_COMPARISON','CROSS_OPERATOR_LAYER_CARD_ITERATION']),
        ('G08','Typed physical capacity and storage/network limits','Need valid capacity upper bounds at actual precision/clock/mixed issue/storage tier/UVA/Vector/communication scope. Logical bytes,requested allocation,measured peaks and PMU busy are not universal capacities.', ['THEORY_AND_MEASURED_COMPARISON','LOWEST_MODULE_UTILIZATION']),
        ('G09','Finite graph and source-domain reconciliation','Partial graph catalog813nodes/1297edges/46structural hints is not the whole native/semantic graph. Missing actual bank/parameter/owner maps and conditional source questions must be distinguished from unavailable fields.', ['CURRENT_FINITE_GRAPH_LEDGER','ALL_MODULE_CONDITIONAL_BRANCHES']),
        ('G10','Whole cross-candidate composition','Central131 conditional relations/25 groups leave464 pairs to reconcile;blank rows are not independence. Do not enumerate arbitrary empty pairs or relabel old local answers as new work.', ['CROSS_OPERATOR_LAYER_CARD_ITERATION']),
        ('G11','Attribution and causal request effects','849missingPMU,16phase-unboundgraphs,3136ambiguous assignments remain. Task/COMM/wrapper sums are not recoverable request latency;current166 CPU/producer-ready transport is unmeasured where needed.', ['CAUSES_AND_CRITICAL_PATH']),
        ('G12','Comparable module utilization and lowest module','All-module common physical MFU denominator/ranking remains NOT_ESTABLISHED. Conditional work/profiling and symbolic resource bounds do not identify the lowest module.', ['LOWEST_MODULE_UTILIZATION','CAUSES_AND_CRITICAL_PATH']),
        ('G13','Named unresolved analysis versus existing-answer review','Review genuine unexpanded operator bodies/readers or unpriced resources against existing evidence before more source work. Examples:M09 projection cost scope,M17 optional upstream filter/processors,M21/M22 captured-bank maps. These labels are not proof that all are still unread.', ['ALL_MODULE_CONDITIONAL_BRANCHES','CURRENT_FINITE_GRAPH_LEDGER','CRITICAL_EVIDENCE_AND_FINAL_CLOSURE'])]
    write(O/'coverage-current-obligations-v105.json',dict(obligations=[dict(id=i,title=t,remaining=g,requirements=req,closed=False)for i,t,g,req in obligations],
        historical_remaining='coverage-gap-history-before-v105.json',historical_records_retained=55,
        source_access_question_resolved=['enable_pcp definition','is_pcp_decode_sharding_enabled definition'],
        stale_claims_to_correct=[dict(branch='M12-OP-19',old=next(b['gap']for m in matrix['modules']if m['id']=='M12'for b in m['branches']if b['id']=='M12-OP-19'),
            current='Add post-v98 partial raw-attribute evidence:3200 target records have C65536;1272 others,loaded ABI and demand remain open. This is not a loaded max_output_size proof.'),
            dict(branch='M21-S01',old='Two PCP helper definitions require unavailable source access',current='Definitions retrieved read-only;context/config/loaded identity remains open'),
            dict(branch='M21-S14',old='Exact decode-shard selector unread',current='PCP>1 AND DCP1 AND no speculative config;argument/context epochs remain open')],
        outside_independent_operator_research_scope=['Exhaustive unrelated HTTP/admission/tokenizer/transport framework variants unless they change a specified operator input,reader,state,error or readiness conclusion. Existing provenance/constraints remain retained.'],
        no_whole_scope_completion_or_optimization_percentage=True,overall_complete=False))
    old_remaining=read(O/'module-coverage-gaps-v104.json')['remaining']
    write(O/'coverage-gap-history-routing-v105.json',dict(rows=[dict(old_index=i,full_text=s,
        disposition='HISTORICAL_STATUS_AND_CONSTRAINT_PRESERVED;NOT_A_CURRENT_COUNT_OR_NEW_OPEN_QUESTION_BY_ITSELF',
        current_obligation_registry='coverage-current-obligations-v105.json',closed_or_dropped=False)for i,s in enumerate(old_remaining)],
        historical_texts_retained=55,current_counts_source='whole-scope-requirement-audit-v105.json',overall_complete=False))
    inputs=[O/'module-phase-branch-matrix-v104.json',O/'module-coverage-gaps-v104.json',O/'whole-scope-requirement-audit-v104.json',
        O/'M21-captured-config-and-selected-dispatch-contract-v42.json',O/'M21-PCP-alternative-case-and-consumer-contract-v42.json',
        O/'MC2-attribute-to-ceiling-contract-v98.json',O/'minimal-operator-evidence-plan-v98.json',
        O/'operator-selector-source-index-v105.json',N/'operator-selector-source166-v105.json']
    write(O/'coverage-dependency-inputs-v105.json',dict(inputs=[pin(p)for p in inputs]+[pin(Path(p['path']))for p in parts],
        semantic_evidence_pins_in='coverage-semantic-evidence-audit-v105.json',
        new_helper_source_files=2,no_model_Torch_NPU_capture_service_or_container_process=True,overall_complete=False))
    print(json.dumps(dict(status='ANALYZED',semantic_rows=353,evidence_files=391,PCP_source_cases=96,factory_cases=6,
        historical_remaining_records_preserved=55,overall_complete=False)))

if __name__=='__main__':main()
