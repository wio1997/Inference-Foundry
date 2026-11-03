"""Audit the one authorized capture and its limited conclusions; no accelerator operations."""
from pathlib import Path
import argparse,ast,hashlib,json,re
from finalize_origin_coverage_v9 import N,O,read,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');a=p.parse_args();checks=[]
    def ck(n,v):assert v,n;checks.append(n)
    def pinned(r):
        b=Path(r['path']).read_bytes();return len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256']
    for r in read(O/'operand-probe-inputs-v109.json')['inputs']:ck('input:'+r['path'],pinned(r))
    original=(N/'diagnostic-operand-probe-v107/inspect_graph_params.py').read_text();fixed=(N/'diagnostic-operand-probe-v109/inspect_graph_params.py').read_text()
    old="if not output.is_relative_to(Path('/data/tiankuan/wio')) or output.exists():\n        raise RuntimeError('Output must be a new independent directory under /data/tiankuan/wio')"
    new="if output.parent != Path('/tmp') or not output.name.startswith('wio-operand-probe-v109-') or output.exists():\n        raise RuntimeError('Output must be a new independent /tmp/wio-operand-probe-v109-* directory')"
    ck('only_output_path_changed',fixed==original.replace(old,new))
    ck('original_prepared_hash',hashlib.sha256(original.encode()).hexdigest()==read(O/'operand-evidence-plan-v107.json')['probe']['sha256'])
    first=read(N/'operand-probe166-v109.json')['result'];r=read(N/'operand-probe166-v109b.json')['result']
    ck('first_stopped_before_library_load',first['attempt']['returncode']==1 and 'Read-only file system'in first['attempt']['stderr']and original.index('output.mkdir(')<original.index('lib=ctypes.CDLL(')<original.index('import torch'))
    ck('write_recovery_within_authorization','允许最小探针'in read(N/'operand-probe-submission-v109b.json')['authorized_by'])
    ck('selected_container_unchanged',first['container_guard']['stdout']==r['container_guard']['stdout']==read(O/'operand-probe-error-code-index-v109.json')['container_after']['stdout']=='running 2032500\n')
    for key in ['resource_guard','resource_after']:
        s=r[key]['stdout'];rows=[line for line in s.splitlines()if'0000:'in line]
        ck(key+':idle',r[key]['returncode']==0 and all('No running processes found in NPU '+str(i)in s for i in range(8))and len(rows)==16 and all(line.split('|')[3].split()[0]=='0'for line in rows))
    probe=read(N/'operand-probe-results-v109b/probe-result.json');dump=read(N/'operand-probe-results-v109b/own-tiny-graph.json')
    ck('successful_capture_metadata',r['attempt']['returncode']==0 and probe['status']=='CAPTURED_METADATA_QUERIED_NO_REPLAY'and len(probe['tasks'])==4)
    ck('not_saved_model_evidence',not probe['saved_model_correlation']and not probe['typed_MemSet_destination_or_length_decoded'])
    ck('no_replay_or_model_calls',probe['plan']['graph_replay_calls']==probe['plan']['model_loads']==probe['plan']['graph_task_mutator_calls']==0 and 'graph.replay('not in fixed)
    ck('one_capture_syntax',fixed.count('with torch_npu.npu.graph(graph):')==1)
    ck('fresh_debug_dump',hashlib.sha256((N/'operand-probe-results-v109b/own-tiny-graph.json').read_bytes()).hexdigest()==probe['debug_dump_sha256'])
    byid={d['args']['Task Id']:d for d in dump}
    ck('one_local_seq_join_only',set(byid)=={t['seq_id']for t in probe['tasks']}=={0,1,2,3}and len({t['stream_handle']for t in probe['tasks']})==1)
    ck('two_controlled_memcpy',[byid[i]['args']['Task Type']for i in [0,1]]==['MEMCPY_ASYNC','MEMCPY_ASYNC'])
    ck('notification_task',byid[3]['args']['Task Type']=='NOTIFY_RECORD')
    good=[t for t in probe['tasks']if t['get_params_returncode']==0];bad=[t for t in probe['tasks']if t['get_params_returncode']==107000]
    ck('one_success_three_invalid',len(good)==1 and len(bad)==3 and all(t['type']==0 for t in bad))
    k=good[0];ck('specific_kernel_contract',(k['type'],k['seq_id'],k['argsSize'],k['opInfoSize'],k['isHostArgs'],k['numBlocks'])==(1,2,64,136,0,1))
    words=[int(x,16)for x in re.findall(r'0x[0-9a-fA-F]+',byid[2]['args']['Kernel Args'])]
    ck('known_packet_not_semantic_field',len(words)==8 and words.count(probe['bool_output_address'])==2 and probe['bool_input_address']not in words and words.count(24)==3 and not k['args_bytes_dereferenced'])
    ec=read(O/'operand-probe-error-code-index-v109b.json');ck('exact_error_name',re.search(r'^#define ACL_ERROR_RT_PARAM_INVALID\s+107000\b',Path(ec['local_path']).read_text(),re.M)is not None)
    obs=read(O/'operand-probe-observations-v109.json');gate=read(O/'operand-probe-capability-gates-v109.json')
    ck('dump_not_execution_times',not obs['debug_dump_ts_dur_used_as_execution_times'])
    ck('control_socket_closed',not obs['cleanup']['SSH_control_socket_present']and not(N/'ssh166-probe-v109.sock').exists())
    ck('authorization_no_longer_pending',gate['authorization_status']=='GRANTED_AND_CONSUMED_FOR_THIS_MINIMAL_PROBE;NO_PENDING_REPLY')
    ck('nine_model_regimes_remain',gate['retained_model_gaps']['MemSet_regimes']==7 and gate['retained_model_gaps']['BOOL24_regimes']==2 and not gate['numeric_physical_ceiling_MFU_speedup_or_whole_closure'])
    if a.preflight:manifest=read(O/'checkpoint-manifest-v108.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v109.json');ck('PROGRESSnotcomplete',cp['status']=='PROGRESS'and not cp['overall_complete'])
        for base in ['candidate-profile-ledger','framework-candidate-ledger','module-profile-ledger']:ck('honest_device_flag:'+base,read(O/(base+'-v109.json'))['new_device_work'])
        matrix=read(O/'module-phase-branch-matrix-v109.json');ck('353rows26modules',sum(len(m['branches'])for m in matrix['modules'])==353 and len(matrix['modules'])==26)
        old=read(O/'operator-analysis-coverage-ledger-v108.json');new=read(O/'operator-analysis-coverage-ledger-v109.json')
        for key in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']:ck('unchanged_model:'+key,old[key]==new[key])
        ck('132_25_463',tuple(read(O/'candidate-relation-coverage-v109.json')[x]for x in ['scoped_conditional_pairs','higher_order_groups','remaining_pair_relations'])==(132,25,463))
        jobs=read(O/'task-run-registration-v109.json')['runs'];ck('failed_predevice_attempt_retained',len(jobs)==3 and sum(x['status']=='error'for x in jobs)==1 and sum(x['status']=='pass'for x in jobs)==2)
        for x in read(O/'candidate-record-index-v109.json')['model_candidates']:ck('master:'+x['id'],hashlib.sha256(Path(x['master_record']).read_bytes()).hexdigest()==x['master_sha256'])
        manifest=read(O/'checkpoint-manifest-v109.json')
    for row in manifest['artifacts']:ck('manifest:'+row['path'],pinned(row))
    result=dict(status='PASS',checks=len(checks),preflight=a.preflight,own_captured_tasks=4,successful_kernel_param_queries=1,invalid_param_queries=3,
        model_regimes_promoted=0,model_graph_replays=0,overall_complete=False)
    if a.preflight:write(O/'operand-probe-preflight-v109.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
