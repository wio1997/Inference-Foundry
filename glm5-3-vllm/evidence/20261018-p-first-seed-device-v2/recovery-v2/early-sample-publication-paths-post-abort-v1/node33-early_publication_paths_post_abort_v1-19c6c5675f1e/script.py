from pathlib import Path
import json,socket,subprocess,urllib.request
assert socket.gethostname()=='B5-5U-Atlas900RCKA3-33'
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/early-publication-live1-20261019/node33')
cid='8d81520a203ef0baaf71860185d06fcf1d0d60e907ba1e0029852c7cb8f62cfd'
plan=json.loads((root/'plan.json').read_text());d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
assert d['State']['Running'] and d['Id']==json.loads((root/'launch.json').read_text())['container_id'] and d['Image']==plan['image']['id']
probe=root/'early-sample-publication-probe';assert not(probe/'enable').exists()
rows=[json.loads(s) for p in sorted(probe.glob('rank-*.jsonl')) for s in p.read_text().splitlines()]
assert sorted({z['rank'] for z in rows})==list(range(16))
summary={}
for rank in range(16):
    data=[z for z in rows if z['rank']==rank]
    kept=[z for z in data if z['kind']=='native_late_kept']
    selected=[z for z in kept if z['early_copy']]
    retired=[z for z in data if z['kind']=='late_retired' and z['early_copy']]
    assert selected and all(z['same_copy_stream'] and z['native_event_for_InputBatch'] for z in kept)
    assert all(z['bitwise_equal'] and z['source_kept_until_native_late'] for z in retired)
    assert {z['seq'] for z in selected}=={z['seq'] for z in retired}, 'Selected sources not retired/compared'
    summary[str(rank)]=dict(native_late_records=len(kept),early_selected=len(selected),late_bitwise_compared=len(retired),
                            empty_drains=sum(z['kind']=='empty_drain' for z in data))
acks=[json.loads(s) for p in probe.glob('scheduler-*.jsonl') for s in p.read_text().splitlines()]
assert acks and any(z['early_sample'] for z in acks)
assert all(z['rank_count']==16 and z['previous']<=z['retired']<=z['step']<=z['scheduled_through'] for z in acks)
assert all(z['retired']<z['step'] if z['early_sample'] else z['retired']==z['step'] for z in acks)
assert acks[-1]['retired']==acks[-1]['scheduled_through'] and acks[-1]['deferred_entries']==0
idle={}
for n,port in [(32,29481),(33,29482)]:
    with urllib.request.urlopen('http://172.16.12.%d:%d/metrics'%(n,port),timeout=5) as f:text=f.read().decode()
    lines=[z for z in text.splitlines() if z.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{','vllm:kv_cache_usage_perc{'))]
    assert len(lines)>=3 and all(float(z.rsplit(' ',1)[1])==0 for z in lines);idle[str(n)]=lines
report=dict(completed=True,all16_paths=True,all_selected_late_copies_bitwise=True,all16_retirement_ACK=True,
            ranks=summary,scheduler_ACKs=len(acks),last_ACK=acks[-1],idle_metrics=idle,
            scope='Finite real PD path/ownership proof. Native event-for-InputBatch follows unchanged native caller and preserved late object, not a separate pointer snapshot. Retirement ACKs cover all16. No general quality, conflict-window timing or service performance claim.')
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(report)+'\nGLM53_RESULT_JSON_END')
