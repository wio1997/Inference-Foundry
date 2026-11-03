"""Independent CPU-only verification; does not import the opt-in probe or accelerator APIs."""
from pathlib import Path
import argparse,ast,ctypes,hashlib,json,subprocess,sys
from finalize_origin_coverage_v9 import N,O,read,write

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');a=parser.parse_args();checks=[]
    def ck(n,x):assert x,n;checks.append(n)
    def pinned(r):
        p=Path(r['path']);b=p.read_bytes();return len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256']
    for r in read(O/'operand-evidence-inputs-v107.json')['inputs']:ck('input:'+r['path'],pinned(r))
    source=read(O/'operand-evidence-source-contract-v107.json')
    for p in source['parts']:
        ck('segment:'+p['path']+str(p['first_line']),pinned(p)and Path(p['path']).read_text().splitlines()[p['first_line']-1:p['last_line']]==p['source'])
    header=Path(next(r['local_path']for r in read(O/'sdk-operand-source-index-v107.json')['files']if r['path'].endswith('/acl_rt.h'))).read_text()
    enum=header.split('typedef enum aclmdlRITaskType {',1)[1].split('} aclmdlRITaskType;',1)[0]
    ck('no_typed_memset_enum','MEMSET'not in enum.upper()and 'ACL_MODEL_RI_TASK_KERNEL'in enum)
    compact=''.join(header.split())
    ck('count_units_are_distinct','aclrtMemsetD32Async(void*ptr,size_tmemSize,uint32_tvalue,size_tN,aclrtStreamstream);'in compact and 'aclrtMemsetAsync(void*devPtr,size_tmaxCount,int32_tvalue,size_tcount,aclrtStreamstream);'in compact and 'Number of uint32_t elements to set'in header and 'the number of byte to set'in header)
    ck('generic_packet_fields','size_t argsSize;'in header and 'size_t opInfoSize;'in header and not source['kernel_argsSize_means_write_extent'])
    plan=read(O/'operand-evidence-plan-v107.json');desc=plan['description'];probe=Path(plan['probe']['path'])
    ck('prepared_not_run',plan['status']==desc['status']=='PREPARED_NOT_EXECUTED'and desc['own_application_payload_bytes']==8240)
    ck('not_total_memory',plan['application_payload_not_total_context_memory']and desc['driver_context_and_allocator_overhead_not_bounded_by_payload'])
    ck('bounded_metadata_only',desc['graph_replay_calls']==desc['model_loads']==desc['tensor_contents_copied_to_host']==desc['graph_task_mutator_calls']==0)
    tree=ast.parse(probe.read_text());imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    ck('default_no_accelerator_import',all(not any(x in ast.unparse(n)for x in ['torch','torch_npu'])for n in imports))
    cp=subprocess.run([sys.executable,str(probe)],text=True,capture_output=True,check=True)
    ck('default_matches_plan',json.loads(cp.stdout)==desc)
    ck('runtime_opt_in',"if not args.run:print(json.dumps(describe(),ensure_ascii=False));return"in probe.read_text()and 'graph.replay('not in probe.read_text())
    cases=plan['API_unit_reference_cases'];ck('12_unit_cases',len(cases)==12)
    for i,r in enumerate(cases):
        p,c,n,o=r['provider'],r['capacity_bytes'],r['count'],r['offset'];actual=r['reference']
        if p=='opaque-kernel':expected='UNKNOWN_PROVIDER';nb=None
        elif any(type(x)is not int or x<0 for x in [c,n,o])or n*(4 if p.endswith('D32Async')else 1)>c or o+c>2**64-1:
            expected='INVALID_REFERENCE_DOMAIN';nb=None
        else:expected='DECLARED_API_REFERENCE_ONLY';nb=n*(4 if p.endswith('D32Async')else 1)
        ck('unit_case:'+str(i),actual['state']==expected and actual['written_bytes']==nb)
    ck('same_write_different_count',cases[0]['reference']['written_bytes']==cases[1]['reference']['written_bytes']==64 and cases[0]['count']==64 and cases[1]['count']==16)
    route=read(O/'sampling-prior-answer-routing-v107.json');prior=read(O/'pivot-filter-source-contract-v92.json')
    ck('exact_prior_answer',route['source']['sha256']==prior['sha256']and pinned(route['source'])and route['no_new_fusion_relation_or_numeric_ceiling'])
    ck('NPU_not_CUDA_split',prior['NPU_route']['finding']==route['NPU_route']['finding']and not prior['actual_random_batch_invariant_or_native_compilation_attested'])
    gate=read(O/'operand-evidence-method-gates-v107.json')
    ck('five_evidence_routes',len(gate['methods'])==5 and gate['consequences']['task_id_without_graph_epoch_insufficient'])
    if a.preflight:manifest=read(O/'checkpoint-manifest-v106.json')
    else:
        checkpoint=read(O/'REVIEW-CHECKPOINT-v107.json');ck('PROGRESS_not_complete',checkpoint['status']=='PROGRESS'and not checkpoint['overall_complete'])
        matrix=read(O/'module-phase-branch-matrix-v107.json');ck('353semantic26',sum(len(m['branches'])for m in matrix['modules'])==353 and len(matrix['modules'])==26)
        old=read(O/'module-phase-branch-matrix-v106.json');oldm=next(m for m in old['modules']if m['id']=='M17');newm=next(m for m in matrix['modules']if m['id']=='M17')
        ck('same_branch_identity',[r['id']for r in oldm['branches']]==[r['id']for r in newm['branches']])
        for r in newm['branches']:
            if r['id']in route['corrected_rows']:ck('stale_corrected:'+r['id'],'UpstreamTritonfilterbodyandremainingprocessors/readers remainseparateopenquestions.'not in r['gap']and r['prior_answer_routing_v107']['stale_unread_body_claim_corrected'])
        old=read(O/'operator-analysis-coverage-ledger-v106.json');new=read(O/'operator-analysis-coverage-ledger-v107.json')
        for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']:ck('unchanged:'+k,old[k]==new[k])
        rel=read(O/'candidate-relation-coverage-v107.json');ck('132_25_463',(rel['scoped_conditional_pairs'],rel['higher_order_groups'],rel['remaining_pair_relations'])==(132,25,463))
        ob=read(O/'coverage-current-obligations-v107.json');ck('14_current_open',len(ob['obligations'])==14 and all(not r['closed']for r in ob['obligations']))
        for r in read(O/'candidate-record-index-v107.json')['model_candidates']:ck('master:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        manifest=read(O/'checkpoint-manifest-v107.json')
    for r in manifest['artifacts']:ck('manifest:'+r['path'],pinned(r))
    result=dict(status='PASS',checks=len(checks),preflight=a.preflight,unit_reference_cases=12,stale_branch_rows=2,new_native_promotions=0,device_probe_executed=False,overall_complete=False)
    if a.preflight:write(O/'operand-evidence-preflight-v107.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
