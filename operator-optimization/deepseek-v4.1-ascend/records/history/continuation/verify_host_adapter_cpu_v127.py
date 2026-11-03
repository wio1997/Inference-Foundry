"""Verify saved CPU fixture behavior and evidence pins; no remote/runtime calls."""
from pathlib import Path
import argparse,hashlib,json
from finalize_origin_coverage_v9 import N,O,T,L,read,write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');args=ap.parse_args();checks=[]
    def ck(name,value):
        assert value,name
        checks.append(name)
    def pinned(r):
        p=Path(r['path']);b=p.read_bytes()
        return len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256']
    for r in read(O/'host-adapter-inputs-v127.json')['inputs']:ck('input:'+r['path'],pinned(r))
    for name in ['host_packet_snapshot.h','host_packet_snapshot.c']:
        ck('unchanged_core:'+name,(N/'eager-host-adapter-v127'/name).read_bytes()==(N/'eager-host-launch-source-v126'/name).read_bytes())
    for suffix in ['v127','v127b','v127c']:
        receipt=read(N/('host-adapter-CPU166-'+suffix+'.json'));result=read(O/('host-adapter-CPU-results-'+suffix+'.json'))
        ck('SSH_receipt:'+suffix,receipt['returncode']==0 and receipt['envelope']['container_pid']==2032500 and json.loads(receipt['envelope']['stdout'])==result and not receipt['new_device_work'])
    a=read(O/'host-adapter-CPU-results-v127.json');ck('retained_build_failure',a['status']=='CPU_BUILD_ERROR'and a['builds'][-1]['returncode']!=0 and 'unused-result'in a['builds'][-1]['stderr'])
    b=read(O/'host-adapter-CPU-results-v127b.json');c=read(O/'host-adapter-CPU-results-v127c.json')
    def hosts(r):return [x for x in r['records']if x['type']=='host_args']
    def footer(r):return next((x for x in r['records']if x['type']=='summary'),None)
    expected=['disabled','normal','no_ids','failure','non_target','name_failure','oversize','recursive','many','byte_budget','zero_quota','unarmed','expired','early_exit','bad_dir','fork','late_local','direct_lookup_bypass','missing_backend','asan_one','asan_many','asan_recursive','asan_oversize']
    cases={x['name']:x for x in c['cases']}
    ck('exact_cases',list(cases)==expected)
    ck('CPU_Linux_aarch64',c['platform']==dict(system='Linux',machine='aarch64'))
    ck('six_builds',len(c['builds'])==6 and all(x['returncode']==0 and not x['stderr']for x in c['builds']))
    ck('real_runtime_not_loaded',not c['CANN_runtime_loaded_in_tests']and not c['Torch_or_NPU_context_created']and not c['real_CANN_interception_or_original_operand_recovery']and not c['original_container_or_service_changed']and not c['new_device_work'])
    for source in c['sources']:
        path=N/'eager-host-adapter-v127'/source['name'];blob=path.read_bytes()
        ck('remote_source:'+source['name'],len(blob)==source['bytes']and hashlib.sha256(blob).hexdigest()==source['sha256'])
    old=next(x for x in b['sources']if x['name']=='fake_runtime.c')
    ck('b_fixture_preserved',hashlib.sha256((N/'eager-host-adapter-v127/fake_runtime-attempt-b.c').read_bytes()).hexdigest()==old['sha256'])
    ck('read_only_exports',c['installed_CANN_library_read_only']and len(c['installed_library_symbols'])==4 and all(len(v)==1 for v in c['installed_library_symbols'].values()))
    for name,table in c['binary_dynamic_tables'].items():ck('no_CANN_DT_NEEDED:'+name,'libascendcl'not in table and 'libtorch'not in table)
    for name,r in cases.items():
        ck(name+':exit',r['returncode']==0 and r['stderr']=='')
        if name!='early_exit':ck(name+':backend_status',r['stdout']['status']=='PASS'and not r['stdout'].get('bad_inputs',0))
        hs=hosts(r);f=footer(r)
        if r['files']:
            ck(name+':one_file',len(r['files'])==1)
            blob=('\n'.join(json.dumps(x,separators=(',',':'))for x in r['records'])+'\n').encode()
            ck(name+':raw_log_hash',len(blob)==r['files'][0]['bytes']and hashlib.sha256(blob).hexdigest()==r['files'][0]['sha256'])
        for h in hs:
            raw=bytes.fromhex(h['args_hex'])
            ck(name+':snapshot_size:'+str(h['reservation']),len(raw)==h['args_size']==152)
            ck(name+':before_relocation:'+str(h['reservation']),raw[0]==0x42 and raw[1]==h['tid']%256 and raw[8:16]==b'\0'*8 and h['placeholders']==[[8,24]])
            ck(name+':unbound_capture:'+str(h['reservation']),h['capture_scope']=='NOT_QUERIED')
            ck(name+':times:'+str(h['reservation']),h['api_end_realtime_ns']>=h['api_begin_realtime_ns']>0)
        if f:
            ck(name+':footer_last',r['records'][-1]==f and f['written']==len(hs)and len([x for x in r['records']if x['type']=='summary'])==1)
            ck(name+':budget',f['written']<=f['reserved']<=256 and r['files'][0]['bytes']<=f['byte_limit']and f['bytes_before_footer']+512<=f['byte_limit'])
    for name in ['disabled','direct_lookup_bypass','bad_dir']:ck(name+':no_log_is_not_no_work',not cases[name]['records']and cases[name]['stdout']['calls']==1)
    ck('disabled_no_metadata',cases['disabled']['stdout']['name_queries']==0)
    ck('normal_ids',hosts(cases['normal'])[0]['task_id_rc']==0 and hosts(cases['normal'])[0]['task_id']==1 and hosts(cases['normal'])[0]['stream_id']==13)
    no=hosts(cases['no_ids'])[0];ck('optional_getters_off',no['task_id_rc']==no['stream_id_rc']==-1 and cases['no_ids']['stdout']['last_task_queries']==cases['no_ids']['stdout']['stream_queries']==0)
    fail=hosts(cases['failure'])[0];ck('failed_submission_not_kernel_execution',fail['launch_rc']==77 and fail['task_id_rc']==-1 and cases['failure']['stdout']['last_task_queries']==0)
    for name,key in [('non_target','non_target'),('name_failure','name_failed'),('oversize','bad_packet'),('zero_quota','quota_skips'),('byte_budget','byte_skips')]:
        ck(name+':explicit_skip',not hosts(cases[name])and footer(cases[name])[key]==1)
    for name,n in [('many',7),('asan_many',56)]:
        r=cases[name];f=footer(r);hs=hosts(r)
        ck(name+':quota',r['stdout']['calls']==160 and f['seen']==160 and f['reserved']==f['written']==n and f['quota_skips']==160-n)
        ck(name+':unique_reservations',sorted(x['reservation']for x in hs)==list(range(1,n+1)))
        ck(name+':multiple_threads',len({x['tid']for x in hs})>=2)
    for name in ['recursive','asan_recursive']:
        r=cases[name];h=hosts(r)[0];f=footer(r)
        ck(name+':guard_exercised',r['stdout']['calls']==f['seen']==2 and f['reentrant_skips']==h['reentrant_skips_during_call']==1 and len(hosts(r))==1)
        ck(name+':nested_last_task',h['task_id']==2)
    for name in ['recursive','asan_recursive']:
        r=next(x for x in b['cases']if x['name']==name)
        ck('b:'+name+':bypass_witness',r['stdout']['calls']==2 and footer(r)['seen']==1 and footer(r)['reentrant_skips']==0)
    ck('arm_gate',not footer(cases['unarmed'])['armed']and cases['unarmed']['stdout']['name_queries']==0)
    ck('expiry_stops_getters',cases['expired']['stdout']['calls']==2 and cases['expired']['stdout']['name_queries']==1 and footer(cases['expired'])['time_skips']==1 and len(hosts(cases['expired']))==1)
    ck('abnormal_exit_incomplete',cases['early_exit']['stdout']is None and len(hosts(cases['early_exit']))==1 and footer(cases['early_exit'])is None)
    ck('fork_forward_only',len(hosts(cases['fork']))==1 and footer(cases['fork'])['seen']==1)
    ck('late_local_fallback',len(hosts(cases['late_local']))==1 and cases['late_local']['stdout']['calls']==1)
    ck('direct_lookup_bypass',cases['direct_lookup_bypass']['stdout']['direct_handle_lookup_bypasses_adapter'])
    ck('handshake_no_backend',not(cases['missing_backend']['stdout']['probe_bits']&1)and cases['missing_backend']['stdout']['no_runtime_call']and footer(cases['missing_backend'])['seen']==0)
    ready=read(O/'host-adapter-collector-readiness-v127.json')
    ck('real_gates_remain',len(ready['remaining_gates'])==8 and not ready['full_model_capture_ready']and not ready['actual_CANN_interception_tested']and not ready['installed_into_model']and ready['original_pending_operands_recovered']==0 and not ready['additional_device_probe_or_model_capture_authorized'])
    close=read(N/'host-adapter-CPU-session-close-v127.json');ck('session_closed',close['returncode']==0 and close['all_test_subprocesses_terminal']and not close['new_device_work'])
    if args.preflight:manifest=read(O/'checkpoint-manifest-v126.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v127.json');ck('open_progress',cp['status']=='PROGRESS'and not cp['overall_complete']and cp['whole_modules_closed']==0)
        cov=read(O/'candidate-relation-coverage-v127.json');ck('same135_26_460',(cov['scoped_conditional_pairs'],cov['higher_order_groups'],cov['remaining_pair_relations'])==(135,26,460))
        m=read(O/'module-phase-branch-matrix-v127.json');ck('same358_26',len(m['modules'])==26 and sum(len(x['branches'])for x in m['modules'])==358)
        old=read(O/'operator-analysis-coverage-ledger-v126.json');new=read(O/'operator-analysis-coverage-ledger-v127.json')
        ck('native_uncertainty_unchanged',all(old[k]==new[k]for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']))
        for r in read(O/'candidate-record-index-v127.json')['model_candidates']:ck('candidate:'+r['id'],hashlib.sha256(Path(r['master_record']).read_bytes()).hexdigest()==r['master_sha256'])
        ck('task_loop_active',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING')
        manifest=read(O/'checkpoint-manifest-v127.json')
    for r in manifest['artifacts']:ck('manifest:'+r['path'],pinned(r))
    result=dict(status='PASS',checks=len(checks),preflight=args.preflight,CPU_cases=23,sanitizer_cases=4,
        actual_CANN_interception_tested=False,original_pending_promotions=0,new_device_work=False,overall_complete=False)
    if args.preflight:write(O/'host-adapter-preflight-v127.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
