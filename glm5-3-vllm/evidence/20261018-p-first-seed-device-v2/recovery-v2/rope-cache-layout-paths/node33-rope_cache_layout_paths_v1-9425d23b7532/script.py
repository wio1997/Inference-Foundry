from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,subprocess
root=Path('/data/wio/glm5-3-vllm-execution/20261018-state-tail/rope-cache-layout-live1-20261019/node33');cid='246f4f1090349ae9b8018af114460537df89c7073547190f5d0949326edc2046';d=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];plan=json.loads((root/'plan.json').read_text());assert d['State']['Running'] and d['Name']=='/'+plan['container']['name'] and d['Image']==plan['image']['id'];assert not (root/'rope-cache-layout-probe/enable').exists()
rows=[];inventories={};files=[]
for p in sorted((root/'rope-cache-layout-probe').glob('rank-*.jsonl')):
 b=p.read_bytes();files.append({'path':p.name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()});group={}
 for line in b.decode().splitlines():
  z=json.loads(line);rank=z['rank'];inv=z['inventory'];inventories[rank]=inv
  for rid in z['ids']:
   key=(rank,rid);v=group.setdefault(key,{'rank':rank,'request_id':rid,'frames':0,'calls':0,'selected':0,'native':0,'requested_values':set(),'Ns':set(),'use_cache_values':set(),'fallbacks':Counter(),'all_persistent_addresses_unchanged':True,'all_full_cache_bitwise_equal':True,'native_layouts':set(),'cached_layouts':set(),'cache_bytes':set()});v['frames']+=1;v['requested_values'].add(z['requested']);v['all_full_cache_bitwise_equal'] &= inv['full_cache_bitwise_equal'];v['cache_bytes'].add(inv['cache_bytes']);v['native_layouts'].update(tuple(t['stride']) for t in inv['native']);v['cached_layouts'].update(tuple(t['stride']) for t in (inv['cached'] or []))
   for c in z['calls']:
    v['calls']+=1;v['selected']+=bool(c['selected']);v['native']+=not c['selected'];v['Ns'].add(c['N']);v['use_cache_values'].add(c['use_cache']);v['all_persistent_addresses_unchanged'] &= c['persistent_output_addresses_unchanged']
    if c['fallback']:v['fallbacks'][c['fallback']]+=1
 for v in group.values():
  for k in ['requested_values','Ns','use_cache_values','native_layouts','cached_layouts','cache_bytes']:v[k]=sorted(v[k])
  v['fallbacks']=dict(v['fallbacks']);rows.append(v)
assert sorted(inventories)==list(range(16))
print('GLM53_RESULT_JSON_BEGIN\n'+json.dumps({'completed':True,'CID':cid,'ranks':list(range(16)),'inventories':inventories,'files':files,'rows':rows,'marker_present':False,'scope':'Actual native SFA get_cos_and_sin_mla calls and source-layout selector for each worker/request, cache bitwise/owner/bytes and persistent output address invariants. Shared batched calls may be attributed to several request IDs; counts are path evidence, not per-request service cost. Raw JSONL retained at model root, small reduction only.'})+'\nGLM53_RESULT_JSON_END')
