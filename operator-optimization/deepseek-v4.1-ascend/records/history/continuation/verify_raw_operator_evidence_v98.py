"""Independent evidence/count/identity and algebra checks;no target execution."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,itertools,json
from finalize_origin_coverage_v9 import N,O,R,read,write
from mirror_saved_profile_core_v53 import CACHE

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');a=parser.parse_args();checks=[]
    def check(name,value):
        assert value,name
        checks.append(name)
    raw=read(O/'raw-operator-evidence-extraction-v98.json');check('readonlycapture',raw['returncode']==0 and not raw['result']['new_capture'])
    refs=read(CACHE/'analysis/profile-capabilities-index-v12-v2.json')['databases'];dbs=raw['result']['databases'];check('32DBs',len(dbs)==len(refs)==32)
    for d,r in zip(dbs,refs):
        check('rawidentity:'+str(d['database_index']),(d['path'],d['bytes'],d['sha256'])==(r['path'],r['bytes'],r['sha256'])and d['stat_unchanged'])
    audit=read(O/'raw-profile-capability-audit-v98.json');check('3GiBidentitystream',audit['raw_database_bytes']==sum(r['bytes']for r in refs)==3278102528)
    check('same27tables',audit['schema_variants']==1 and audit['tables_per_database']==27)
    check('nooperandcolumns',all(not d['candidate_address_or_argument_columns']for d in dbs))
    check('notallattributesempty',audit['operator_attribute_census']['DispatchFFNCombine']==dict(executions=4472,with_nonempty_attr=3200))
    check('PMUtableretained','TASK_PMU_INFO'in audit['raw_tables_and_columns'])
    for c in audit['capture_configurations']:
        check('config:'+c['original_path'],c['torch_npu_version']=='2.10.0.post7'and c['cann_version']=='9.2.0'and not c['common_config']['profile_memory']and not c['experimental_config']['record_op_args']and not c['experimental_config']['_op_attr'])
    m=read(O/'MC2-recorded-attributes-and-executions-v98.json');rows=m['executions'];check('4472uniqueTASKs',len(rows)==len({(r['database_index'],r['task_record_rowid'])for r in rows})==4472)
    known=[r for r in rows if r['recorded_capacity']is not None];missing=[r for r in rows if r['recorded_capacity']is None]
    check('3200recorded1272unknown',len(known)==3200 and len(missing)==1272)
    check('recordedattrs',all(r['recorded_capacity']==65536 and r['recorded_world_size']==16 and not r['recorded_is_trans_b']and r['recorded_weight_nz']and r['recorded_swiglu_limit']==10 for r in known))
    check('unknownnotfilled',all(r['raw_attribute_text']==''and r['recorded_world_size']is None for r in missing))
    check('allrecordedM3target',all(r['input_shapes']=='"3,5120;24,5120,4608;24,2304,5120;3,6;110592;122880;3,6;3"'for r in known))
    check('phaseunboundnotclosed',(m['root_bound_executions'],m['phase_unbound_executions'])==(2560,640))
    expected=Counter()
    for db in dbs:
        for g in db['target_operator_regimes']['DispatchFFNCombine']:expected[(db['database_index'],g['attrInfo'],g['attrs'],g['input_shapes'])]+=g['executions']
    actual=Counter((r['database_index'],r['raw_attribute_reference_id'],r['raw_attribute_text'],r['input_shapes'])for r in rows)
    check('ownattributesameDBfivefieldjoin',expected==actual)
    for d in m['per_database']:
        p=CACHE/f"analysis/native-execution-identity-v14/db-{d['database_index']:02d}.json"
        check('cachehash:'+str(d['database_index']),hashlib.sha256(p.read_bytes()).hexdigest()==d['cache_sha256'])
    check('allocationnotmeasured',m['actual_allocator_peak_bytes']is None and m['actual_receiver_count']is None and m['source_formula_and_loaded_ABI_binding_still_conditional'])
    c=read(O/'MC2-attribute-to-ceiling-contract-v98.json');check('twoGEMMclassicalarithmetic',c['reference_OPS_per_processed_assignment']==2*5120*(4608+2304)==70778880)
    check('noinventedMFU',c['current_dense_work_or_MFU']is None and not c['numerical_physical_ceiling_or_critical_gain_proved'])
    # Independently reduce the source's processed-prefix clipping,includingempty/zero/exactcap/overflowdomains.
    cases=0
    for n in range(6):
        for counts in itertools.product(range(5),repeat=n):
            for C in [0,1,3,8,17,65536]:
                processed=0;out=[]
                for demand in counts:
                    take=min(demand,max(0,C-processed));out.append(take);processed+=take
                assert sum(out)==min(sum(counts),C);cases+=1
    check('perexpertclipidentity',cases==23436)
    p=read(O/'remaining-operand-evidence-status-v98.json');check('ninepending1352',len(p['pending_regimes'])==9 and sum(r['executions']for r in p['pending_regimes'])==1352)
    rec=p['execution_bindings'];check('1352uniqueoperandTASKs',len(rec)==len({(r['database_index'],r['task_record_rowid'])for r in rec})==1352)
    check('MemSet1272emptyownattrs',sum(r['operator']=='MemSet'for r in rec)==1272 and all(r['raw_attribute_text']==''and r['input_shapes']=='N/A'for r in rec if r['operator']=='MemSet'))
    check('BOOL80staticnessnotfield',sum(r['operator']=='ZerosLike'for r in rec)==80 and all(r['raw_attribute_text']=='IsStaticKernel:false'and r['input_shapes']=='"24"'for r in rec if r['operator']=='ZerosLike'))
    plan=read(O/'minimal-operator-evidence-plan-v98.json');check('preparednotnewcapture',plan['status']=='PREPARED_NOT_EXECUTED'and plan['quantitative_scope']==dict(new_native_executions=0,new_capture=0,device_to_host_copies=0))
    for src in [m['source'],c['source'],plan['optional_next_windows'][0]['source']]:
        q=Path(src['path']);check('pinnedsource:'+q.name,hashlib.sha256(q.read_bytes()).hexdigest()==src['sha256'])
        lines=q.read_text().splitlines()
        for seg in src['segments']:check('sourcesegment:'+str(seg['first_line']),lines[seg['first_line']-1:seg['last_line']]==seg['source'])
    if a.preflight:
        manifest=read(O/'checkpoint-manifest-v97.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v98.json');check('progressnotcomplete',cp['status']=='PROGRESS'and not cp['overall_complete'])
        cov=read(O/'candidate-relation-coverage-v98.json');check('relationsunchanged',cov['scoped_conditional_pairs']==124 and cov['higher_order_groups']==25 and cov['remaining_pair_relations']==471)
        bm=read(O/'module-phase-branch-matrix-v98.json');check('branchesunchanged',sum(len(m['branches'])for m in bm['modules'])==359 and sum(m.get('branch_schema')=='SEMANTIC_BRANCHES_V1'for m in bm['modules'])==24)
        op=read(O/'operator-analysis-coverage-ledger-v98.json');old=read(O/'operator-analysis-coverage-ledger-v97.json')
        for f in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']:check('nativeuniverse:'+f,op[f]==old[f])
        audit=read(O/'whole-scope-requirement-audit-v98.json');check('ninewholegoalgapsremain',len(audit['requirements'])==9 and not audit['overall_complete'])
        check('lowestMFUunknown',next(r for r in audit['requirements']if r['id']=='LOWEST_MODULE_UTILIZATION')['state']=='NOT_ESTABLISHED')
        for x in read(O/'candidate-record-index-v98.json')['model_candidates']:check('master:'+x['id'],hashlib.sha256(Path(x['master_record']).read_bytes()).hexdigest()==x['master_sha256'])
        manifest=read(O/'checkpoint-manifest-v98.json')
    for row in manifest['artifacts']:
        p=Path(row['path']);check('manifest:'+p.name,p.stat().st_size==row['bytes']and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'])
    result=dict(status='PASS',checks=len(checks),preflight=a.preflight,clip_identity_cases=cases,recorded_capacity_executions=3200,unbound_capacity_executions=1272,new_device_work=False,overall_complete=False)
    if a.preflight:write(O/'raw-operator-analysis-preflight-v98.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
