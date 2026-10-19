from pathlib import Path
import json
import socket

assert socket.gethostname()=='B5-5U-Atlas900RCKA3-33'
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/indexer-k-graph-live1-20261019/node33')
probe=root/'indexer-k-graph-overlap-probe'
rows=[]
for p in sorted(probe.glob('rank-*.jsonl')):
    frames=[json.loads(s) for s in p.read_text().splitlines()]
    ranks={z['rank'] for z in frames}
    assert len(ranks)==1
    calls=[c for f in frames for c in f['calls'] if c['selected']]
    assert calls, 'No actual selected eager MTP K graph at '+p.name
    assert all(c['current_native_bitwise'] and c['independent_inputs'] and c['consumed_once']
               and c['added_input_copies']==3 and c['native_cache_write_retained']
               and c['native_main_prolog_retained'] for c in calls)
    rows.append(dict(rank=next(iter(ranks)),frames=len(frames),selected=len(calls),
                     N=sorted({c['N'] for c in calls}),Indexer_C8=sorted({c['Indexer_C8'] for c in calls}),
                     cold=sum(c['cold_graph'] for c in calls),graph_entries_max=max(f['graph_entries'] for f in frames),
                     all_selected_current_native_bitwise=True,independent_owned_inputs=True,
                     consumed_once=True,native_main_and_cache_writes_retained=True))
assert sorted(z['rank'] for z in rows)==list(range(16))
assert not (probe/'enable').exists()
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(dict(completed=True,all16=True,rows=rows,
      scope='Actual selected MTP eager K producer vs current original forward_k, output lifetime through original consumer, functional observer phase only. Does not prove all16 physical copy transport or service performance.'))+'\nGLM53_RESULT_JSON_END')
