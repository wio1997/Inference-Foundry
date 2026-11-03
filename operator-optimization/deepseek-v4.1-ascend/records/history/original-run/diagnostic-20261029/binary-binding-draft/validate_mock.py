import collections, hashlib, json, pathlib, sys
p=pathlib.Path(sys.argv[1]); raw=p.read_bytes(); rows=[json.loads(x) for x in raw.splitlines()]
counts=collections.Counter(r['event'] for r in rows)
assert all(0 < counts[k] <= n for k,n in {'load_data':403,'get_function':401,'get_function_entry':400,'launch_host_args':400,'launch_begin':400,'unload':401,'load_file':1}.items()), counts
assert len({r['pid'] for r in rows}) == 2
for pid in {r['pid'] for r in rows}:
    s=[r['seq'] for r in rows if r['pid']==pid]
    assert s == list(range(1,len(s)+1))
assert any(r.get('name_hex')=='6e6573746564' for r in rows)
sha=hashlib.sha256(b'abc').hexdigest()
assert sum(r.get('pre_call_sha256')==sha for r in rows)>0
assert sum(r['event']=='load_data' and r['rc']==0 and r['hash_status']=='unknown' for r in rows)==2
assert sum(r['event']=='load_data' and r['rc']==-7 and r['hash_status']=='unknown' for r in rows)==1
assert all(r['event']!='get_function' or r['name_hex'] in (b'kernel"test'.hex(),b'nested'.hex()) for r in rows)
assert [r['event'] for r in rows if r['event'] in ('graph_execute_begin','graph_execute_end')] == ['graph_execute_begin','graph_execute_end']
g=[r for r in rows if r['event'].startswith('graph_execute_')];assert g[0]['call']==g[1]['call'] and g[0]['model_ri']==g[1]['model_ri']
assert any(r['event']=='stream_get_id' and r['stream_id']==31 for r in rows)
assert any(r['event']=='thread_last_task_id' and r['task_id']==55 for r in rows)
assert any(r['event']=='capture_get_info' and r['model_ri']=='0x4444' for r in rows)
allowed={'v','pid','tid','seq','monotonic_ns','event','rc','size','handle','pre_call_sha256','hash_status','bin','func','name_hex','name_status','entry','stream','blocks','dropped_since_last','record_cap_reached','call','id_valid','stream_id','task_id','result_valid','capture_status','model_ri'}
assert all(set(r)<=allowed for r in rows)
print(json.dumps({'status':'PASS','total':len(rows),'counts':dict(counts),'pids':len({r['pid'] for r in rows}),'trace_sha256':hashlib.sha256(raw).hexdigest(),'sha256_test_vector':sha,'null_failed_and_bad_pointer':'PASS','nested_original_event_recorded':'PASS','per_pid_sequences':'PASS','scope':'CPU mock only'},indent=2))
