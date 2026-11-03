"""Verify actual nested host-API scopes and tested CPU-only diagnostic readiness."""
from collections import defaultdict
from pathlib import Path
import argparse,hashlib,json,sqlite3,subprocess
from finalize_origin_coverage_v9 import N,O,T,L,read,write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=[]
    def ck(name,value):
        assert value,name
        checks.append(name)
    def pinned(r):
        b=Path(r['path']).read_bytes()
        return len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256']
    for r in read(O/'host-launch-inputs-v126.json')['inputs']:ck('input:'+r['path'],pinned(r))
    coverage=read(O/'host-launch-original-API-coverage-v126.json');bydb=defaultdict(list)
    for r in coverage['records']:bydb[r['database_index']].append(r)
    ck('all1272',{(r['database_index'],r['task_record_rowid'])for r in coverage['records']}=={(r['database_index'],r['TASK']['task_record_rowid'])for r in read(O/'operand-correlation-native-records-v125.json')['records']if r['operator']=='MemSet'})
    for db in coverage['databases']:
        ck('DB_SHA:'+str(db['database_index']),pinned(db))
        con=sqlite3.connect(Path(db['path']).as_uri()+'?mode=ro&immutable=1',uri=True);con.row_factory=sqlite3.Row
        names={r['id']:r['value']for r in con.execute("SELECT * FROM STRING_IDS WHERE value='aclrtLaunchKernelWithHostArgs'")}
        hosts=[]
        for name_id in names:hosts.extend(dict(r)for r in con.execute('SELECT rowid AS cann_api_rowid,* FROM CANN_API WHERE name=?',(name_id,)))
        for r in bydb[db['database_index']]:
            launch=r['runtime_launch'];task=con.execute('SELECT *FROM TASK WHERE rowid=?',(r['task_record_rowid'],)).fetchone()
            ck('exact_launch_CID:'+str((db['database_index'],r['task_record_rowid'])),task['connectionId']==launch['connectionId']and task['modelId']==0xffffffff)
            candidates=[h for h in hosts if h['globalTid']==launch['globalTid']and h['startNs']>=launch['startNs']and h['endNs']<=launch['endNs']]
            recorded=r['same_thread_nested_host_argument_APIs']
            ck('unique_actual_host_API:'+str((db['database_index'],r['task_record_rowid'])),len(candidates)==len(recorded)==1 and all(candidates[0][k]==recorded[0][k]for k in candidates[0]))
            ck('packet_not_in_profile:'+str((db['database_index'],r['task_record_rowid'])),not r['host_packet_bytes_saved_in_profile']and not r['native_destination_or_length_recovered'])
        con.close()
    abi=read(O/'host-launch-installed-ABI-contract-v126.json');ck('installed_header',pinned(abi['header']))
    lines=Path(abi['header']['path']).read_text().splitlines()
    for s in abi['segments']:ck('header_segment:'+str(s['first_line']),s['source']==lines[s['first_line']-1:s['last_line']])
    cpu=read(O/'host-packet-CPU-core-result-v126.json')
    for r in cpu['build_artifacts']:ck('CPU_binary:'+r['path'],pinned(r))
    d=N/'eager-host-launch-source-v126/cpu-build'
    for exe in [d/'selftest',d/'selftest-sanitized']:
        p=subprocess.run([str(exe)],capture_output=True,text=True,check=True,timeout=10)
        ck('CPU_selftest:'+exe.name,json.loads(p.stdout)['status']=='PASS'and not p.stderr)
    ck('source_specific_constructed_fixture',not cpu['fixture']['actual_new_host_packet_observed']and cpu['fixture']['args_bytes']==152 and cpu['fixture']['source_span_union']==[[0,32]]and cpu['fixture']['input_unchanged']and cpu['fixture']['ambiguous_placeholder_rejected'])
    ck('CPU_only_no_native_hook',not cpu['CANN_library_loaded']and not cpu['new_NPU_context']and not cpu['device_or_tensor_pointer_dereferenced']and not cpu['installed_adapter_or_live_native_interception_tested'])
    ready=read(O/'host-launch-collector-readiness-v126.json')
    ck('adapter_not_ready_for_device_claim',ready['status']=='CPU_COPY_CORE_VALIDATED;NATIVE_ADAPTER_NOT_INSTALLED_OR_TESTED'and ready['original_pending_operands_recovered']==0 and not ready['additional_device_probe_or_model_capture_authorized'])
    ck('explicit_remaining_native_gates',len(ready['native_adapter_gates'])==8 and ready['proposed_unexecuted_adapter_limits']['bytes_per_packet']==4096)
    if args.preflight:manifest=read(O/'checkpoint-manifest-v125.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v126.json');ck('progress_open',cp['status']=='PROGRESS'and not cp['overall_complete'])
        cov=read(O/'candidate-relation-coverage-v126.json');ck('same135_26_460',(cov['scoped_conditional_pairs'],cov['higher_order_groups'],cov['remaining_pair_relations'])==(135,26,460))
        matrix=read(O/'module-phase-branch-matrix-v126.json');ck('same358_26',len(matrix['modules'])==26 and sum(len(m['branches'])for m in matrix['modules'])==358)
        old=read(O/'operator-analysis-coverage-ledger-v125.json');new=read(O/'operator-analysis-coverage-ledger-v126.json')
        ck('native_uncertainty_unchanged',all(old[k]==new[k]for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']))
        for r in read(O/'candidate-record-index-v126.json')['model_candidates']:ck('candidate:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        ck('task_loop_active',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING')
        manifest=read(O/'checkpoint-manifest-v126.json')
    for r in manifest['artifacts']:ck('manifest:'+r['path'],pinned(r))
    result=dict(status='PASS',checks=len(checks),preflight=args.preflight,original_eager_host_API_scopes=1272,
        CPU_core_and_sanitizers='PASS',live_CANN_interception_tested=False,
        original_pending_promotions=0,new_device_work=False,overall_complete=False)
    if args.preflight:write(O/'host-launch-preflight-v126.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
