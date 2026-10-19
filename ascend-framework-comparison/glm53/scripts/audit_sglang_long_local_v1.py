"""Independent local audit of captured native SDMA and completed logical cases."""
import argparse,collections,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
for name in ['before','after','client','output']:parser.add_argument('--'+name,required=True)
args=parser.parse_args();paths={name:(ROOT/getattr(args,name)).resolve() for name in ['before','after','client','output']}
assert all(path.is_relative_to(ROOT) for path in paths.values())
client=json.loads((paths['client']/'artifacts/semantic-result.json').read_text())
register={};writes=[];raw_sha={}
for node in [34,35,36,41]:
    old=json.loads(json.loads((paths['before']/f'node-{node}.json').read_text())['stdout'])
    raw=(paths['after']/f'node-{node}.json').read_bytes();outer=json.loads(raw);assert outer['rc']==0
    current=json.loads(outer['stdout']);assert current['model']['state']['Running'];raw_sha[str(node)]=hashlib.sha256(raw).hexdigest()
    for name,artifact in current['persistent_audit'].items():
        encoded=''.join(json.dumps(row,sort_keys=True,default=str)+'\n' for row in artifact['rows']).encode()
        assert len(encoded)==artifact['bytes'] and hashlib.sha256(encoded).hexdigest()==artifact['sha256']
        prior=old['persistent_audit'].get(name,{}).get('rows',[]);assert artifact['rows'][:len(prior)]==prior
        for row in artifact['rows']:
            if row['event']=='initialize_complete':assert row['protocol']=='SDMA' and row['protocol_value']==1 and row['rc']==0
            if row['event']=='model_state_registered':
                state=row['metadata'];assert state['hidden_kv_layers']==78 and state['draft_kv_layers']==1
                assert state['kv_cache_dtype_str']=='bfloat16' and state['kv_buf_groups']==3
                assert collections.Counter(state['kv_item_lens'])=={131072:79,16384:79,32768:79}
                assert sum(state['aux_item_lens'])==22016
                register[row['session']]={'node':node,'DP':state['system_dp_rank'],'engine_rank':state['engine_rank']}
        for row in artifact['rows'][len(prior):]:
            if row['event']=='write_complete':
                assert row['rc']==0 and row['payload_bytes_completed']==row['payload_bytes_requested']
                assert row['physical_link_bytes'] is None
                writes.append(row)
assert len(register)==64
expected=collections.defaultdict(lambda:{'pages':0,'requests':0})
for row in client['gateway']:
    assert row['status']=='D_STREAM_COMPLETE' and row['prefill_http_status']==row['decode_http_status']==200
    dp=row['route']['D']['global_dp'];expected[dp]['pages']+=(row['input_tokens']+127)//128;expected[dp]['requests']+=1
groups=collections.defaultdict(list)
for row in writes:
    src=register[row['local_session']];dst=register[row['remote_session']]
    assert src['node'] in [34,35] and dst['node'] in [36,41]
    groups[row['remote_session']].append(row)
assert len(groups)==2*len(expected)
observed=[]
for session,rows in groups.items():
    identity=register[session];want=expected[identity['DP']]
    main=[r for r in rows if r['context']['kind']=='send_kvcache']
    aux=[r for r in rows if r['context']['kind']=='send_aux']
    assert len(main)+len(aux)==len(rows)
    assert all(r['ranges']==237 and r['payload_bytes_completed']%14237696==0 for r in main)
    assert len(aux)==want['requests'] and all(r['ranges']==11 and r['payload_bytes_completed']==22016 for r in aux)
    assert sum(r['payload_bytes_completed']//14237696 for r in main)==want['pages']
    observed.append(identity|{'main_native_calls':len(main),'main_pages':want['pages'],'aux_native_calls':len(aux)})
report={'project_id':'glm53','status':'COMPLETED_LOGICAL_CASES_NATIVE_ALL_STATE_SDMA_BYTES_PASSED',
        'client_run':args.client,'audit_run':args.after,'completed_logical_cases':len(client['gateway']),
        'all_six_long_cases_completed':len(client['cases'])==6,
        'new_native_write_completions':len(writes),'all_native_rc':[0],
        'descriptor_bytes_by_state':{kind:sum(r['payload_bytes_completed'] for r in writes if r['context']['kind']==kind) for kind in ['send_kvcache','send_aux']},
        'D_workers':observed,'physical_link_bytes':None,'raw_node_sha256':raw_sha,
        'L3_qualification':'Separate exact-known-key readiness, actualstoragehit andfactualguards required; this isPDonly.',
        'formal_requests':0,'baseline_accepted':False}
paths['output'].write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
