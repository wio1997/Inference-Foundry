from pathlib import Path
import json,socket
assert socket.gethostname()=='B5-5U-Atlas900RCKA3-33'
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/rope-step-reuse-live1-20261019/node33')
probe=root/'rope-step-reuse-probe';rows=[]
for p in sorted(probe.glob('rank-*.jsonl')):
 frames=[json.loads(s) for s in p.read_text().splitlines()]
 ranks={z['rank'] for z in frames};assert len(ranks)==1
 calls=[v for z in frames for v in z['calls'] if v['selected']]
 assert calls and all(v['reused_lookups']==v['bitwise_comparisons']==1 and v['owned_copy_bitwise'] and v['independent_indexer_storage'] for v in calls)
 steps=sorted({v['draft_index'] for v in calls});assert steps==[1,2,3,4]
 rows.append(dict(rank=next(iter(ranks)),frames=len(frames),selected=len(calls),steps=steps,N=sorted({v['N'] for v in calls}),original_indexer_copy_retained=True,all_selected_bitwise=True,independent_indexer_storage=True))
assert sorted(v['rank'] for v in rows)==list(range(16))
assert not(probe/'enable').exists()
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps(dict(completed=True,all16=True,rows=rows,scope='Actual same-step shared source matches original getter bitwise; Indexer still owns distinct copies for steps1-4. Functional observer phase only, no timing claim.'))+'\nGLM53_RESULT_JSON_END')
