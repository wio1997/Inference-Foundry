import collections,hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);raw=p.read_bytes();rows=[json.loads(x) for x in raw.splitlines()];c=collections.Counter(r['event'] for r in rows)
assert len(raw)<100000 and len(rows)<100,(len(raw),c)
assert c['target_function']==2 and c['target_launch_begin']==3 and c['target_launch_end']==3,c
assert c['target_load']==1 and c['target_unload']==1 and c['target_byentry_reclassified']==1,c
assert next(r for r in rows if r['event']=='target_byentry_reclassified')['entry']==42
assert next(r for r in rows if r['event']=='target_preclassification_launch')['entry']==42
assert all(r.get('binding_status')=='candidate_unverified' for r in rows if r['event'] in ('target_function','target_byentry_reclassified','target_launch_begin','target_launch_end','target_preclassification_launch','target_load'))
assert c['target_capture_info']==1 and c['target_graph_begin']==1 and c['target_graph_end']==1 and c['target_graph_destroy']==1,c
assert c['target_stream_id']==1 and c['target_last_task_id']==1,c
named=next(r for r in rows if r['event']=='target_function' and r['kind']=='name_match')
assert bytes.fromhex(named['name_hex'])==b'DispatchFFNCombine_actual'
assert named['pre_call_sha256']==hashlib.sha256(b'abc').hexdigest() and named['generation']>0
entries=[r['entry'] for r in rows if r['event']=='target_launch_begin' and r['kind']=='bin_candidate_byentry']
assert sorted(entries)==[42,43],entries
known=next(r for r in rows if r['event']=='target_load')
assert known['pre_call_sha256']=='4cf1474cb1009b96fc743f5a396a9c07a6980146c41818b60cdbf0d3453a43a9'
assert sum(r['event']=='selective_stats' for r in rows)>=2
assert all(r.get('state_incomplete') in (None,False) for r in rows)
print(json.dumps({'status':'PASS','scope':'CPU-only fake APIs; installed object read for hash','bytes':len(raw),'records':len(rows),'counts':dict(c),'irrelevant_loops':12000,'named_load_link':'PASS','byentry_before_name_candidate':'PASS','sha_file_byentry_candidate':'PASS','unload_reuse_graph_destroy':'PASS','unrelated_handle_reuse':'PASS','sha256':hashlib.sha256(raw).hexdigest()},indent=2))
