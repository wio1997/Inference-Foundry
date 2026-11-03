"""Independent own-capture guard, packet-offset and source-store-range audit."""
from pathlib import Path
import argparse,hashlib,json,re,struct
from finalize_origin_coverage_v9 import N,O,T,L,read,write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');a=ap.parse_args();checks=[]
    def ck(n,v):assert v,n;checks.append(n)
    def pinned(r):
        b=Path(r['path']).read_bytes();return len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256']
    for r in read(O/'native-pad-inputs-v120.json')['inputs']:ck('input:'+r['path'],pinned(r))
    original=read(N/'native-pad-probe166-v120.json')['result'];r=read(N/'native-pad-probe166-v120b.json')['result']
    ck('one_actual_capture_not_two',original['capture_attempts']==0 and 'attempt'not in original and r['capture_attempts']==1 and r['attempt']['returncode']==0)
    ck('same_approved_source',original['source_sha256']==r['source_sha256']=='f48726e647c8bfb67b9b99fbc4cbec159d855af473ec4436964b7ca3850174ca')
    ck('container_unchanged',r['container_guard']==r['container_after']and r['container_after']['stdout']=='running 2032500\n')
    for field in ['resource_guard','resource_after']:
        s=r[field]['stdout'];rows=[line for line in s.splitlines()if '0000:'in line]
        ck(field+'_idle16',r[field]['returncode']==0 and len(rows)==16 and all(line.split('|')[3].strip().split()[0]=='0'for line in rows))
        ck(field+'_8_noprocess',all('No running processes found in NPU '+str(i)in s for i in range(8)))
    ck('SSH_closed',not(N/'ssh166-pad-v120.sock').exists())
    folder=N/'native-pad-probe-results-v120b';probe=read(folder/'probe-result.json');dump=read(folder/'own-native-pad-graph.json')
    ck('capture_no_replay',probe['status']=='CAPTURED_METADATA_QUERIED_NO_REPLAY'and probe['plan']['graph_replay_calls']==0 and probe['plan']['model_loads']==0)
    ck('44B_only_application',probe['plan']['own_application_input_output_bytes']==20+24 and probe['plan']['driver_context_allocator_internal_casts_and_workspaces_not_bounded_by_44'])
    ck('six_tasks_one_stream',len(probe['tasks'])==len(dump)==6 and len({t['stream_handle']for t in probe['tasks']})==1)
    ck('five_kernel_params_default_invalid',[t['get_params_returncode']for t in probe['tasks']]==[0,0,0,0,0,107000])
    ck('no_GPU_deref_or_tensor_data_read',probe['plan']['tensor_payload_copied_to_host']==0 and not probe['plan']['opaque_device_args_dereferenced']and all(not t.get('args_bytes_dereferenced',False)for t in probe['tasks']))
    i=next(i for i,g in enumerate(dump)if g['name'].startswith('MemSet_'));task=probe['tasks'][i];g=dump[i]
    words=[int(v,16)for v in re.findall(r'0x[0-9a-fA-F]+',g['args']['Kernel Args'])];raw=b''.join(v.to_bytes(8,'little')for v in words)
    ck('packet152_not_fill',len(raw)==task['argsSize']==152 and len(words)==19)
    ck('observed_interior_pointer',words[1]==task['args_address']+24 and 24+96<=len(raw))
    ints=struct.unpack('<24i',raw[24:120])
    ck('source_mixed_layout_controls',ints[0:4]==(1,1,16320,2040)and ints[18:24]==(1,1,48,0,0,0))
    last_elements=struct.unpack_from('<q',raw,24+48)[0]
    ck('one_core_eight_FP32_elements',last_elements==8 and ints[1]==1 and ints[18]*32==8*4==32)
    ck('no_full_loops',struct.unpack_from('<q',raw,24+56)[0]==0 and struct.unpack_from('<q',raw,24+64)[0]==0)
    pad=next(g for g in dump if g['name'].startswith('PadV3_'));pw=[int(v,16)for v in re.findall(r'0x[0-9a-fA-F]+',pad['args']['Kernel Args'])]
    ck('workspace_pointer_matches_Pad',words[0]==pw[5]and words[0]not in[probe['bool_input_address'],probe['bool_output_address']])
    contract=read(O/'native-pad-packet-and-store-contract-v120.json')
    ck('old_packet_hypothesis_rejected',contract['old_whole_packet_prediction_bytes']==128 and contract['old_strict_helper_rejection']=='Argument-packet packing unproved'and contract['argsSize']==152)
    ck('source32_not_advertised1536',contract['source_compatible_requested_zero_store_bytes']==32 and contract['advertised_workspace_metadata_bytes']==1536 and not contract['actual_allocator_extent_or_HBM_bytes_measured'])
    ck('one_requested_span',contract['source_requested_store_spans']['spans']==[dict(core=0,begin=0,end=32,kind='tail')])
    ck('trailer_unexplained',raw[128:]==bytes(24)and contract['remaining24bytes_semantics'].startswith('UNKNOWN'))
    consequences=read(O/'native-pad-consequences-v120.json')
    ck('no_model_gaps_promoted',(consequences['pending_original_regimes'],consequences['pending_original_executions'],consequences['new_native_promotions'])==(9,1352,0))
    ck('authorization_consumed',consequences['authorized_diagnostic'].startswith('GRANTED_AND_CONSUMED'))
    if a.preflight:manifest=read(O/'checkpoint-manifest-v119.json')
    else:
        cp=read(O/'REVIEW-CHECKPOINT-v120.json');ck('progress_open',cp['status']=='PROGRESS'and not cp['overall_complete'])
        matrix=read(O/'module-phase-branch-matrix-v120.json');ck('356_26',len(matrix['modules'])==26 and sum(len(m['branches'])for m in matrix['modules'])==356)
        cov=read(O/'candidate-relation-coverage-v120.json');ck('134_26_461',(cov['scoped_conditional_pairs'],cov['higher_order_groups'],cov['remaining_pair_relations'])==(134,26,461))
        old=read(O/'operator-analysis-coverage-ledger-v119.json');new=read(O/'operator-analysis-coverage-ledger-v120.json')
        for k in ['scoped_conditional_review_regimes','scoped_conditional_review_executions','remaining_regime_reconciliation_count','remaining_execution_reconciliation_count','unchanged_global_uncertainty']:ck('native_unchanged:'+k,old[k]==new[k])
        ck('honest_device_work',read(O/'module-profile-ledger-v120.json')['new_device_work']and read(O/'native-pad-summary-v120.json')['new_device_work'])
        ck('Task_active_Loop_evaluating',read(T/'index.json')['status']=='ACTIVE'and read(L/'state.json')['status']=='EVALUATING')
        for row in read(O/'candidate-record-index-v120.json')['model_candidates']:ck('master:'+row['id'],hashlib.sha256(Path(row['master_record']).read_bytes()).hexdigest()==row['master_sha256'])
        manifest=read(O/'checkpoint-manifest-v120.json')
    for row in manifest['artifacts']:ck('manifest:'+row['path'],pinned(row))
    result=dict(status='PASS',checks=len(checks),preflight=a.preflight,actual_captures=1,
        native_MemSet_packet_bytes=152,source_requested_store_bytes=32,actual_model_operand_promotions=0,
        new_device_work=True,model_loads=0,graph_replays=0,overall_complete=False)
    if a.preflight:write(O/'native-pad-preflight-v120.json',result)
    print(json.dumps(result))

if __name__=='__main__':main()
