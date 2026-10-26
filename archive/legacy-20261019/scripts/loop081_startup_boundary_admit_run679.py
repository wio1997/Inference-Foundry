"""Strict diagnostic artifact admission; not a Product performance promotion."""
import json,sys,time,shutil
from pathlib import Path

def admit(root,phase,offset,version):
    p=root/phase;on=phase=='on';requests=[]
    for label in ('warmup','bench48_1'):
        b=json.loads((p/(label+'.json')).read_text());s=b['summary'];rr=b['requests']
        assert s['n']==s['success']==len(rr)==48 and s['fail']==0 and s['concurrency']==12 and s['max_tokens']==1024
        assert all(x['output_tokens']==1024 and x['error'] is None and x['response_id'] for x in rr)
        assert len({x['response_id'] for x in rr})==48
        if phase!='off_a':
            ref=json.loads((root/'off_a'/(label+'.json')).read_text())['requests']
            assert {v['i']:v['input_tokens'] for v in rr}=={v['i']:v['input_tokens'] for v in ref}
        requests.extend(rr)
    assert len({r['response_id'] for r in requests})==96
    core=[json.loads(x) for x in (root/'boundary/core.jsonl').read_text().splitlines()]
    decoded={x['external_id']:x['req_id'] for x in core if x['kind']=='socket_add_decoded'}
    reqids={decoded[x['response_id']] for x in requests};assert len(reqids)==96
    for rid in reqids:
        for kind in ('socket_add_decoded','preprocess_begin','preprocess_end','queue_put_begin','queue_put_end'):
            assert len([x for x in core if x['kind']==kind and x.get('req_id')==rid])==1,(kind,rid)
    submits={x['submit_id'] for x in core if x['kind']=='schedule_end_submit'}
    seen=set();runtime_reqids=set()
    expected_mode=dict(phase=phase,enabled=on,version=version)
    for c in range(offset+1,offset+9):
        rows=[]
        for rank in range(8):
            file=root/'runtime'/f'rank{rank}_cohort{c}.json';r=json.loads(file.read_text());rows.append(r)
            assert r['rank']==rank and r['cohort']==c and r['pass'] and r['host_mirror_exact'] and r['generated_output_counts']==[1024]*12
            assert r['oracle_target_calls_after_handoff']==0 and 'FULL' in r['target_graph_mode']
            assert r['metadata_graph_capture'] and r['metadata_graph_replays']==r['cycles']
            assert r['slot_certificate_mode']==expected_mode and r['slot_audit_mode']==expected_mode and r['slot_audit_pending']==0
            cert=r['slot_certificate']
            if on:assert cert['eligible'] and not cert['invalidated'] and cert['used_cycles']==8
            else:assert cert is None
            assert len(r['dspark_slot_refresh_audit'])==16
            packet=json.loads((root/'boundary'/f'rank{rank}_cohort{c}.json').read_text());ev=packet['events']
            assert packet['rank']==rank and packet['cohort']==c
            assert all(x.get('submit_id') in submits for x in ev)
            hand=[x for x in ev if x['kind']=='handoff_entry'];assert len(hand)==1 and set(hand[0]['req_ids'])==set(r['req_ids'])
            sid=hand[0]['submit_id']
            for kind in ('runtime_built','runtime_begin','runtime_end','output_ready'):
                assert len([x for x in ev if x['kind']==kind and x['submit_id']==sid])==1,(rank,c,kind)
            for begin,end in [('target_begin','target_end'),('seed_begin','seed_end'),('certificate_begin','certificate_end'),('audit_flush_begin','audit_flush_end'),('validation_begin','validation_end'),('capture_begin','capture_end')]:
                aa=[x for x in ev if x['kind']==begin];bb=[x for x in ev if x['kind']==end]
                assert len(aa)==len(bb),(rank,c,begin)
                assert all(a['submit_id']==b['submit_id'] and a['t_ns']<=b['t_ns'] for a,b in zip(aa,bb))
            for begin in ('certificate_begin','audit_flush_begin','validation_begin','capture_begin'):
                assert len([x for x in ev if x['kind']==begin and x['submit_id']==sid])==1,(rank,c,begin)
            # Each terminal submission must reach Core settlement and publication.
            for kind in ('result_ready','scheduler_update_begin','scheduler_update_end','output_enqueue_begin','output_enqueue_end'):
                assert any(x['kind']==kind and x.get('submit_id')==sid for x in core),(c,kind)
            flushes=[json.loads(x) for x in (root/'boundary'/f'rank{rank}_flush.jsonl').read_text().splitlines()]
            ff=[x for x in flushes if x['cohort']==c];assert len(ff)==1 and ff[0]['end_ns']>=ff[0]['begin_ns']
            seen.add((rank,c));shutil.copyfile(file,p/'runtime'/file.name)
        for field in ('cycles','staged_output_counts','overshoot_tokens','acceptance_window_means'):
            assert all(x[field]==rows[0][field] for x in rows)
        runtime_reqids.update(rows[0]['req_ids'])
    assert runtime_reqids==reqids
    route=[json.loads(x) for x in (root/'hash_routing.jsonl').read_text().splitlines()]
    assert len(route)>=2*version
    route=route[:2*version]
    assert route[-1]['initial_requests']==96*version
    assert all(x['enabled'] and x['phase']=='on' and x['version']==1 and x['memo_misses']==48 and x['memo_hits']==x['initial_requests']-48 for x in route)
    measured=json.loads((p/'bench48_1.json').read_text())
    result=dict(pass_gate=True,scope='diagnostic only; observer affects timings',phase=phase,offset=offset,requests=96,runtime_rows=len(seen),measured_summary=measured['summary'])
    (p/'summary.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    root=Path(sys.argv[1]);phase=sys.argv[2];offset=int(sys.argv[3]);version=int(sys.argv[4]);deadline=time.monotonic()+15
    while True:
        try:result=admit(root,phase,offset,version);break
        except (AssertionError,FileNotFoundError,json.JSONDecodeError,KeyError):
            if time.monotonic()>=deadline:raise
            time.sleep(.1)
    print(json.dumps(result))
