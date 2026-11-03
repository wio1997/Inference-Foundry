import json,pathlib,sys
p=pathlib.Path(sys.argv[1]);mode=sys.argv[2];raw=p.read_bytes();rows=[json.loads(x) for x in raw.splitlines()]
if mode=='evict':
    stats=next(r for r in rows if r['event']=='selective_stats' and r['reason']=='process_exit')
    begins=[r for r in rows if r['event']=='target_launch_begin']
    assert stats['bin_evicted']>0 and stats['state_incomplete'] and len(begins)==1 and begins[0]['state_incomplete']
elif mode=='missing':
    assert sum(r['event']=='target_launch_begin' for r in rows)==0
    assert any(r['event']=='target_unload' for r in rows)
    assert next(r for r in rows if r['event']=='selective_stats')['missing_bin']>=1
    assert next(r for r in rows if r['event']=='target_function')['state_incomplete']
    assert next(r for r in rows if r['event']=='target_function')['binding_status']=='candidate_unverified'
elif mode=='pending':
    stats=next(r for r in rows if r['event']=='selective_stats' and r['reason']=='process_exit')
    assert stats['pending_seen']==6 and stats['pending_drops']>=4 and stats['state_incomplete']
    assert sum(r['event']=='target_preclassification_launch' for r in rows)==2
else:
    assert len(raw)<=4096 and rows[-1]['event']=='trace_cap' and sum(r['event']=='trace_cap' for r in rows)==1
print(json.dumps({'mode':mode,'status':'PASS','bytes':len(raw),'records':len(rows),'scope':'CPU mock'}))
