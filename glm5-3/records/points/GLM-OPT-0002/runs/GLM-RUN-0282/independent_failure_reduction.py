from pathlib import Path
import hashlib,json,re
repo=Path('/Users/wio/work/Inference-Foundry-glm5-3');r=repo/'glm5-3/records/points/GLM-OPT-0002/runs/GLM-RUN-0282';read=lambda n:json.loads((r/n).read_text())
assert read('state.json')['status']==read('diagnostic_status.json')['status']=='failed'
assert 'witness_mc2_native.py' in read('failure.json')['error'] and 'No such file or directory' in read('failure.json')['error']
assert read('diagnostic_status.json')['terminal_verified'] and read('diagnostic_status.json')['recovery_used'] and not read('diagnostic_status.json')['native_correctness']
reqs=[]
for label in ('correctness_mode0_short','H6_H5_recovery_warm'):
    raw=(r/(label+'_D.sse')).read_text();assert raw.rstrip().endswith('data: [DONE]')
    events=[json.loads(x[6:]) for x in raw.splitlines() if x.startswith('data: ') and x!='data: [DONE]'];assert all('error' not in e for e in events)
    ids=[];reasons=[];usage=[]
    for e in events:
        if e.get('usage'):usage.append(e['usage'])
        for c in e.get('choices',[]):
            ids+=c.get('token_ids') or c.get('delta',{}).get('token_ids') or []
            if c.get('finish_reason'):reasons.append(c['finish_reason'])
    assert ids==[785,1196,374,10156,264,3405,304,8452] and reasons[-1]=='length'
    assert usage[-1]==dict(prompt_tokens=2334,completion_tokens=8,total_tokens=2342)
    assert ids==read(label+'_result.json')['token_ids']
    assert [e['value'] for e in read(label+'_events.json') if e['value'] is not None]==events
    t=read(label+'_transfer.json');rid=read(label+'_P.raw')['kv_transfer_params']['remote_request_id'];assert t['remote_request_id']==rid and t['all16_native_transfer_success']
    assert len(t['rows'])==16 and {int(re.search(r'local_device_id (\d+)',x).group(1)) for x in t['rows']}==set(range(16))
    assert read(label+'_request.json')['cache_salt']=='GLM-RUN-0282-'+label
    reqs.append(dict(label=label,exact_tokens=8,all16_transfer=True))
assert len([p for p in r.glob('*_result.json') if not p.name.endswith('CPU_result.json')])==2
caps={rank:read('witnesses/h13_capture_rank%d.json'%rank) for rank in range(16)}
for rank,z in caps.items():
    assert z['rank']==rank and z['raw_model_shared'] and z['independent_pool'] and z['bank_pools'][0]!=z['bank_pools'][1]
    assert len(z['entries'])==2 and len(z['entries'][0])==len(z['entries'][1])==1
    a,b=z['entries'][0][0],z['entries'][1][0];assert a['descriptor']==b['descriptor'] and a['input_addresses']==b['input_addresses']
    assert a['graph_count']>=1 and b['graph_count']>=1 and {o['data_ptr'] for o in a['output']}.isdisjoint({o['data_ptr'] for o in b['output']})
    geometry=[v for v in z['layouts'] if v['fast_gate'] and v['input']['shape']==[2,6144] and v['router_input']['shape']==[2,256] and v['padded_num_tokens']==16 and not v['replace_allreduce'] and v['tp_size']==16]
    assert {v['mode'] for v in geometry}=={0,1}
    for v in geometry:
        assert v['tp_rank']==rank and v['bytes_equal'] and v['input_unchanged']
        for name,shape,dtype in [('hidden_states',[1,6144],'torch.bfloat16'),('router_logits',[1,256],'torch.float32'),('mc2_mask',[1],'torch.bool')]:
            a,b=v['original'][name],v['candidate'][name];assert a['shape']==b['shape']==shape and a['dtype']==b['dtype']==dtype and a['stride']==b['stride'] and a['format']==b['format']==2
    run=read('witnesses/h13_runtime_t1_rank%d.json'%rank)
    assert run['mode']==0 and run['rank']==rank and run['pid']==z['pid'] and run['graph_count']>=1 and run['eager_breaks']==0 and run['pool']==z['bank_pools'][0] and run['output']==z['entries'][0][0]['output']
assert not list((r/'witnesses').glob('h13_runtime_t2_rank*.json'))
w=read('retained_witness.json');g=read('guards_after.json');stack=read('retained_stack.json')
assert all(z['health']==200 and z['idle'] and len(z['device_owners'])==16 for z in g.values())
assert g['166']==read('guards_before.json')['166']
assert w['H6_mode']==w['H5_event_mode']==1 and w['H9_mode']==w['H11_mode']==0
assert len(w['H6']['rows'])==len(w['H5_rows'])==len(w['H9_rows'])==len(w['H11_rows'])==16
assert all(z['dispatch_cached_true'] and z['combine_cached_true'] and z['mode']==1 for z in w['H6']['rows'])
assert all(z['returned_none'] and not z['configured_overlap'] and z['mode']==1 for z in w['H5_rows'])
assert all('FULL' in z['runtime_mode'] and z['capture_num_graphs']>=1 and z['eager_breaks']==0 for z in w['H9_rows'])
assert stack['terminal_verified'] and stack['H6'] and stack['H5'] and stack['target_FULL'] and not stack['H13'] and not stack['H11'] and not stack['H12']
row=dict(status='FAILED_DRIVER_DEPENDENCY',independent_raw_reduction=True,cause='missing read-only witness_mc2_native.py in Run282 before first aggregate witness',two_banks_capture_all16=True,byte_layout_format_inputs_equal_all16=True,shared_capture_inputs_and_disjoint_outputs=True,only_bankA_runtime_verified=True,bankB_PD_correctness=False,naturalEOS_candidate_correctness=False,requests=reqs,one_recovery_completed=True,all16_H6_H5_FULL_health_idle=True,P249_unchanged=True,H13_admitted=False,H13_performance_gain=None,Current=None,amendment='RECOVERY_WITNESS_AMENDMENT.json; helper exact796b49d7 was added only for already-declared recovery; initial spec/FAILED unchanged')
p=r/'failure_reduced.json';assert not p.exists();p.write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row))
