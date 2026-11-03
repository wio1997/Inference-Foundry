"""Independently check selector ASTs,cases,current coverage references and preserved obligations."""
from pathlib import Path
from collections import Counter
from copy import deepcopy
import argparse,ast,base64,hashlib,json
from finalize_origin_coverage_v9 import N,O,R,read,write

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');a=parser.parse_args();checks=[]
    def check(name,value):
        assert value,name
        checks.append(name)
    def pinned(r):
        p=Path(r['path']);return p.stat().st_size==r['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256']
    for r in read(O/'coverage-dependency-inputs-v105.json')['inputs']:check('input:'+r['path'],pinned(r))
    index=read(O/'operator-selector-source-index-v105.json');receipt=read(N/'operator-selector-source166-v105.json')['result']
    check('verified_original_archive_host',receipt['hostname']=='bm-jn-zs-zone1-910C-64G-10-105'and receipt['uid']==0 and receipt['readonly_archive_copy']and not receipt['container_process_started'])
    check('not_device_or_service_work',not receipt['model_Torch_NPU_or_capture']and not receipt['service_health_process_or_configuration_queries'])
    for r in index['files']:
        source=next(x for x in receipt['files']if x['path']==r['path']);b=base64.b64decode(source['source_b64']);p=Path(r['local_path'])
        check('source_archive_exact:'+r['path'],p.read_bytes()==b and hashlib.sha256(b).hexdigest()==r['sha256']==source['sha256'])
    check('control_match_not_historical_helper_binding',index['historical_runner_control_matched']and not index['historical_loaded_helper_objects_attested']and index['new_helper_files']==2)
    check('dedicated_socket_closed',not(N/'ssh166-v105-xinyufeng.sock').exists())
    contract=read(O/'operator-selector-contract-v105.json');symbols={}
    for r in contract['parts']:
        check('contractpin:'+r['path'],pinned(r));lines=Path(r['path']).read_text().splitlines()
        check('contractsegment:'+str(r['first_line'])+Path(r['path']).name,lines[r['first_line']-1:r['last_line']]==r['source'])
        if 'symbol'in r:
            n=next(n for n in ast.parse(Path(r['path']).read_text()).body if isinstance(n,ast.FunctionDef)and n.name==r['symbol']);symbols[r['symbol']]=n
    ret=lambda n:next(x.value for x in reversed(n.body)if isinstance(x,ast.Return))
    check('context_PCP_predicate_exact',ast.unparse(ret(symbols['enable_pcp']))=='parallel_config.prefill_context_parallel_size > 1')
    dec=ret(symbols['is_pcp_decode_sharding_enabled'])
    check('decode_three_guards_exact',isinstance(dec,ast.BoolOp)and isinstance(dec.op,ast.And)and len(dec.values)==3 and
        [ast.unparse(x)for x in dec.values]==['parallel_config.prefill_context_parallel_size > 1','parallel_config.decode_context_parallel_size == 1','vllm_config.speculative_config is None'])
    check('DSA_singleton_predicate',ast.unparse(ret(symbols['enable_dsa_cp']))=='get_ascend_config().enable_dsa_cp')
    factory=ast.unparse(symbols['get_v41_cp_classes'])
    check('factory_mutex_and_order','if use_dsa_cp and use_pcp:'in factory and 'raise ValueError'in factory and factory.index('if use_dsa_cp:')<factory.index('if use_pcp:'))
    cases=read(O/'operator-selector-case-models-v105.json')
    for r in cases['factory_cases']:
        legacy,p=r['ascend_singleton_legacy_DSA_CP'],r['context_prefill_CP']
        expect='ERROR_MUTEX'if legacy and p>1 else'LEGACY_DSACP'if legacy else'PCP'if p>1 else'BASE_V41'
        check('factory_case:'+str((legacy,p)),r['result']==expect)
    check('six_factory_cases',len(cases['factory_cases'])==6)
    seen={}
    for r in cases['PCP_builder_cases']:
        p,d,s,g,z,h=r['builder_prefill_CP'],r['builder_decode_CP'],r['speculative_config_present'],r['global_has_prefill'],r['local_query_empty'],r['full_graph']
        guard=d==1;shard=p>1 and d==1 and not s;need=(g or shard)if guard else None
        check('builder_case:'+str((p,d,s,g,z,h)),r['PCP_explicit_DCP_guard_passes']==guard and r['decode_shard_predicate']==shard and r['needs_global_metadata']==need and not r['actual_case_execution_attested'])
        key=p,d,s,g,z
        result=(r['PCP_explicit_DCP_guard_passes'],r['decode_shard_predicate'],r['needs_global_metadata'],r['metadata_result'])
        if key in seen:check('graph_not_predicate_override:'+str(key),seen[key]==result)
        else:seen[key]=result
        if s:assert not shard
    check('96_pcp_cases',len(cases['PCP_builder_cases'])==96 and len(seen)==48)
    check('source_spec_present_puredecode_no_extra_credit',cases['source_K5_pure_decode']['incremental_savings_from_deleting_that_absent_branch']==0 and not cases['source_K5_pure_decode']['decode_shard_predicate']and cases['context_and_constructor_configs_not_implicitly_joined'])
    patch=read(O/'coverage-branch-reconciliations-v105.json');matrix=deepcopy(read(O/'module-phase-branch-matrix-v104.json'))
    for m in matrix['modules']:
        if m['id']in patch['semantic_branches']:m['branches']=patch['semantic_branches'][m['id']]
    byid={b['id']:b for m in matrix['modules']for b in m['branches']};audit=read(O/'coverage-semantic-evidence-audit-v105.json');refs=Counter()
    for r in audit['rows']:
        b=byid[r['branch_id']]
        assert r['state']==b['state']and r['scope']==b['scope']and r['gap']==b['gap']and r['evidence']==b['evidence']
        assert not r['whole_branch_completion_proved_by_reference_checks'];refs.update(r['evidence'])
    check('353_current_reconciled_rows',len(audit['rows'])==len(byid)==353 and len(refs)==391 and audit['baseline_unique_evidence_artifacts']==388)
    for r in audit['artifact_pins']:check('semantic_evidence:'+r['path'],pinned(r)and r['referenced_rows']==refs[Path(r['path']).name])
    check('no_reference_integrity_as_completion',audit['reference_integrity_is_not_content_or_completeness_proof']and not audit['all_operator_branches_or_whole_model_complete'])
    check('legacy_exclusion_not_new_native_work',audit['legacy_exclusion_rows']==['M07-OP-12']and next(r for r in audit['rows']if r['branch_id']=='M07-OP-12')['applicability']=='EXCLUDED_LEGACY_REFERENCE')
    check('partial_capacity_not_loaded_proof','3200'in byid['M12-OP-19']['gap']and '1272'in byid['M12-OP-19']['gap']and not byid['M12-OP-19']['raw_attribute_review_v105']['loaded_ABI_or_demand_bound'])
    check('unread_selectors_resolved_only_to_source',byid['M21-S01']['selector_source_review_v105']['exact_helper_bodies_reviewed']and not byid['M21-S14']['selector_source_review_v105']['actual_context_config_or_loaded_helper_attested'])
    old=read(O/'module-coverage-gaps-v104.json');history=read(O/'coverage-gap-history-before-v105.json');routing=read(O/'coverage-gap-history-routing-v105.json')
    check('all55_history_preserved',history==old and len(routing['rows'])==55 and [r['full_text']for r in routing['rows']]==old['remaining']and all(not r['closed_or_dropped']for r in routing['rows']))
    obligations=read(O/'coverage-current-obligations-v105.json');req=read(O/'whole-scope-requirement-audit-v104.json')
    required={r['id']for r in req['requirements']};covered={i for r in obligations['obligations']for i in r['requirements']}
    check('all9_original_requirements_retained',covered==required and len(required)==9 and len(obligations['obligations'])==14 and all(not r['closed']for r in obligations['obligations']))
    if a.preflight:manifest=read(O/'checkpoint-manifest-v104.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v105.json');check('PROGRESSnotwhole',cp['status']=='PROGRESS'and not cp['overall_complete'])
        now=read(O/'module-phase-branch-matrix-v105.json');check('actual_patch_matches_audit',[(m['id'],m['branches'])for m in now['modules']]==[(m['id'],m['branches'])for m in matrix['modules']])
        check('353_26_counts',sum(len(m['branches'])for m in now['modules'])==353 and all(m.get('branch_schema')=='SEMANTIC_BRANCHES_V1'for m in now['modules']))
        current=read(O/'module-coverage-gaps-v105.json')['remaining'];check('current_gaps_not_history_status_stack',len(current)==15 and len(history['remaining'])==55)
        for name,keys in [('candidate-relation-coverage',['scoped_conditional_pairs','higher_order_groups','remaining_pair_relations']),('operator-analysis-coverage-ledger',['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty'])]:
            prev=read(O/(name+'-v104.json'));new=read(O/(name+'-v105.json'))
            for k in keys:check('unchanged:'+name+':'+k,prev[k]==new[k])
        for r in read(O/'candidate-record-index-v105.json')['model_candidates']:check('master:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        manifest=read(O/'checkpoint-manifest-v105.json')
    for r in manifest['artifacts']:check('manifest:'+r['path'],pinned(r))
    result=dict(status='PASS',checks=len(checks),preflight=a.preflight,semantic_rows=353,evidence_files=391,source_builder_cases=96,new_helper_files=2,
        historical_remaining_retained=55,no_new_native_or_runtime_guard_promotions=True,overall_complete=False)
    if a.preflight:write(O/'coverage-dependency-analysis-preflight-v105.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
