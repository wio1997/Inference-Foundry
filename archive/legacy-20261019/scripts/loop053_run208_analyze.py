#!/usr/bin/env python3
"""Compare first eager prefill metadata/address fingerprints across two legal cohorts."""
import json,collections
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'evidence/20260925_loop053_native_prefill/run208'
bench={tag:json.loads((out/f'{tag}.json').read_text())['summary'] for tag in ('warmup48','A','B')}
assert all(x['success']==x['n'] and x['max_tokens']==1024 for x in bench.values())
def flatten(node,path='',tensors=None):
 if tensors is None:tensors={}
 if isinstance(node,dict):
  if node.get('kind')=='tensor':
   tensors[path]=node;return tensors
  if 'items' in node and node.get('kind') in ('dict','list','tuple'):
   vals=node['items'];pairs=vals.items() if isinstance(vals,dict) else enumerate(vals)
   for k,v in pairs:flatten(v,path+'.'+str(k),tensors)
  if 'attrs' in node:
   for k,v in node['attrs'].items():flatten(v,path+'.'+k,tensors)
 return tensors
pairs=[]
for rank in range(8):
 a=json.loads((out/'fingerprints'/f'A_rank{rank}.json').read_text());b=json.loads((out/'fingerprints'/f'B_rank{rank}.json').read_text())
 assert a['rank']==b['rank']==rank
 ta={};tb={}
 for key in ('args','kwargs','attn_metadata'):
  flatten(a[key],key,ta);flatten(b[key],key,tb)
 common=set(ta)&set(tb)
 same_layout=sum((ta[k]['shape'],ta[k]['dtype'],ta[k]['stride'])==(tb[k]['shape'],tb[k]['dtype'],tb[k]['stride']) for k in common)
 addr_change=[k for k in common if ta[k]['data_ptr']!=tb[k]['data_ptr']]
 hash_keys=[k for k in common if 'value_sha256' in ta[k] and 'value_sha256' in tb[k]]
 hash_change=[k for k in hash_keys if ta[k]['value_sha256']!=tb[k]['value_sha256']]
 layout_change=[k for k in common if (ta[k]['shape'],ta[k]['dtype'],ta[k]['stride'])!=(tb[k]['shape'],tb[k]['dtype'],tb[k]['stride'])]
 pairs.append({'rank':rank,'A_signature':{k:a[k] for k in ('num_tokens_padded','num_actual_tokens_raw','num_actual_tokens','num_reqs','cudagraph_runtime_mode')},'B_signature':{k:b[k] for k in ('num_tokens_padded','num_actual_tokens_raw','num_actual_tokens','num_reqs','cudagraph_runtime_mode')},'A_tensor_fields':len(ta),'B_tensor_fields':len(tb),'common_fields':len(common),'same_layout_fields':same_layout,'changed_address_fields':len(addr_change),'hashed_common_fields':len(hash_keys),'changed_small_integer_hash_fields':len(hash_change),'layout_change_paths':sorted(layout_change)[:30],'address_change_paths_sample':sorted(addr_change)[:30],'value_hash_change_paths_sample':sorted(hash_change)[:30],'A_nodes_remaining':a['nodes_remaining'],'B_nodes_remaining':b['nodes_remaining'],'A_hashed_tensors':a['small_integer_tensors_hashed'],'B_hashed_tensors':b['small_integer_tensors_hashed']})
rr=[]
for tag,cohort in (('A',5),('B',6)):
 x=[json.loads((out/'runtime'/f'rank{r}_cohort{cohort}.json').read_text()) for r in range(8)]
 assert all(z['pass'] and z['host_mirror_exact'] and z['target_graph_mode']=='FULL' for z in x)
 rr.append({'tag':tag,'cohort':cohort,'all_rank_pass':True,'cycles':[z['cycles'] for z in x]})
result={'run':'run208','contract':'Extreme warmup48+A12+B12 max_tokens1024, all8 ranks; read-only first eager prefill metadata/address probe','bench_summary':bench,'rank_pairs':pairs,'runtime':rr,'limits':['Fingerprints describe only first eager-token prefill in each cohort and selected small integer tensor values; dynamic tensors larger than2048 elements have layout/address only.','Changed pointers may be accommodated by explicit replay-input copies, but graph capture does not update them automatically.','Equal shape and pointers do not prove equal values or safe KV replay; exact write set still needed.','No graph capture or performance claim.']}
(out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'bench_duration_s':{k:v['duration_s'] for k,v in bench.items()},'rank_pairs':[{'rank':p['rank'],'common':p['common_fields'],'layout_change':len(p['layout_change_paths']),'address_change':p['changed_address_fields'],'hash_change':p['changed_small_integer_hash_fields']} for p in pairs],'runtime':rr},indent=2))
