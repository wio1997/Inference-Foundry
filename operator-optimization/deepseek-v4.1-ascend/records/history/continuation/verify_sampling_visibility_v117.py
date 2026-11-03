"""Independent descriptor-budget witnesses and persisted coverage checks; CPU only."""
from pathlib import Path
import argparse,ast,hashlib,json
from finalize_origin_coverage_v9 import N,O,T,L,read,write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');a=ap.parse_args();checks=[]
    def ck(n,v):assert v,n;checks.append(n)
    def pinned(r):
        b=Path(r['path']).read_bytes();return len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256']
    for r in read(O/'sampling-visibility-inputs-v117.json')['inputs']:ck('input:'+r['path'],pinned(r))
    c=read(O/'sampling-visibility-source-contract-v117.json');texts={}
    for r in c['parts']:
        ck('source:'+r['path'],pinned(r));s=Path(r['path']).read_text();texts[r['path']]=s
        ck('source_segment:'+r['path']+str(r['first_line']),s.splitlines()[r['first_line']-1:r['last_line']]==r['source'])
    driver=next(t for p,t in texts.items()if p.endswith('/driver.py'))
    pub=next(t for p,t in texts.items()if p.endswith('/aprof_pub.h'))
    ck('installed_shared_budget5','#define MSPROF_GE_TENSOR_DATA_NUM 5'in pub)
    ck('installed_rank_budget8','#define MSPROF_GE_TENSOR_DATA_SHAPE_LEN 8'in pub)
    ck('shared_loop_and_output_increment','i < tensorShapes.size() && tensorCount < MSPROF_GE_TENSOR_DATA_NUM'in driver and 'fillTensorData(tensorCount, MSPROF_GE_TENSOR_TYPE_OUTPUT);\n          tensorCount++;'in driver)
    ck('firstfive_dtype_only','if idx < 5'in driver)
    ck('nonempty_shapes_only','if (!tmp.empty()) tensorShapes.push_back(tmp);'in driver)
    ck('no_loaded_ABI_promotion',c['installed_source_vs_historical_loaded_launcher'].startswith('UNKNOWN'))
    d=read(O/'sampling-visibility-profile-projection-v117.json')
    ck('3_72_12',len(d['greedy_regimes'])==3 and d['greedy_executions_reused']==72 and d['projection_witnesses']==12)
    for r in d['greedy_regimes']:
        prefixes=[v['descriptors']for v in r['variants']]
        ck('same_projection:'+r['regime_id'],len(prefixes)==4 and all(x==prefixes[0]for x in prefixes))
        ck('firstfive_no_optional:'+r['regime_id'],[x['name']for x in prefixes[0]]==['output_token_ids_ptr','cu_num_draft_tokens_ptr','draft_token_ids_ptr','target_argmax_ptr','bonus_token_ids_ptr'])
        ck('indistinguishable_flags:'+r['regime_id'],{(v['is_greedy_pointer_present'],v['SYNTHETIC_MODE'])for v in r['variants']}=={(False,False),(True,False),(False,True),(True,True)})
        ck('observed_prefix_types:'+r['regime_id'],';'.join(x['dtype']for x in prefixes[0]if x['role']=='input')==r['signature']['inputDataTypes_text'])
        ck('no_new_runtime_call:'+r['regime_id'],r['claim'].startswith('Source-compatible indistinguishable'))
    ck('hash88_missing_output_not_nowrite',d['hash_executions_reused']==88 and d['hash_output_pointer_ordinal_one_based']==14 and not d['runtime_all_greedy_or_synthetic_mode_recovered'])
    for r in d['hash_regimes']:
        ck('hash_prefix:'+r['regime_id'],r['signature']['inputDataTypes_text']=='INT32;INT32;DT_UINT1;INT64;INT32'and r['signature']['outputDataTypes_text']=='N/A')
    # Independent reference expands IO observations, then slices the one shared packet budget.
    refs=read(O/'sampling-visibility-proof-and-boundaries-v117.json')['boundaries'];ck('78_boundaries',len(refs)==78)
    for r in refs:
        expanded=[(i,role)for i in range(r['pointer_count'])for role in (('in','out')if i==r['inout_index']else('in',))]
        ck('budget:'+str((r['pointer_count'],r['inout_index'])),r['description_count']==len(expanded[:5]))
    consequence=read(O/'sampling-visibility-consequences-v117.json')
    ck('no_new_rows_pairs_device',all(consequence[k]==0 for k in ['new_pairs','new_groups','new_semantic_rows','new_native_regimes','new_native_executions'])and not consequence['new_device_work'])
    ck('Pad_still_not_run',consequence['native_Pad_probe'].startswith('PREPARED_NOT_EXECUTED'))
    ck('ssh_closed',not(N/'ssh166-sampling-v117.sock').exists())
    if a.preflight:manifest=read(O/'checkpoint-manifest-v116.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v117.json');ck('progress_not_complete',cp['status']=='PROGRESS'and not cp['overall_complete'])
        m=read(O/'module-phase-branch-matrix-v117.json');ck('355_rows_26_modules',len(m['modules'])==26 and sum(len(x['branches'])for x in m['modules'])==355)
        cov=read(O/'candidate-relation-coverage-v117.json');ck('134_26_461',(cov['scoped_conditional_pairs'],cov['higher_order_groups'],cov['remaining_pair_relations'])==(134,26,461))
        old=read(O/'operator-analysis-coverage-ledger-v116.json');new=read(O/'operator-analysis-coverage-ledger-v117.json')
        for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']:ck('unchanged_native:'+k,old[k]==new[k])
        o=read(O/'coverage-current-obligations-v117.json');gs={g['id']:g for g in o['obligations']}
        ck('14_open_obligations',len(gs)==14 and not o['overall_complete'])
        ck('current_text_consistent','355 semantic'in gs['G00']['remaining']and '/26 groups leave461'in gs['G10']['remaining'])
        for r in read(O/'candidate-record-index-v117.json')['model_candidates']:ck('candidate_digest:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        ck('Task_active_Loop_evaluating',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING')
        manifest=read(O/'checkpoint-manifest-v117.json')
    for r in manifest['artifacts']:ck('manifest:'+r['path'],pinned(r))
    result=dict(status='PASS',checks=len(checks),preflight=a.preflight,shared_tensor_descriptor_cap=5,
        equivalent_flag_projection_witnesses=12,boundary_cases=78,actual_runtime_flags_recovered=False,
        new_device_work=False,new_native_promotions=0,overall_complete=False)
    if a.preflight:write(O/'sampling-visibility-preflight-v117.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
