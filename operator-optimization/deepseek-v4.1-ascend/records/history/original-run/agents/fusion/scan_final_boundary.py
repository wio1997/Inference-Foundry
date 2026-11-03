#!/usr/bin/env python3
"""CPU-only source-level bounded enumeration; not a semantic fusion proof."""
import hashlib, itertools, json
from pathlib import Path
D=Path(__file__).resolve().parent
T=D.parent.parent
src=T/'sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py'
g1=T/'graphs/G1.json'; g5=T/'graphs/source-templates-G5.json'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
# Names match source stages, not measured kernels. Explicit views/data side effects.
base=[('last_hc_post','h'),('last_hc_pre','m'),('h','cast_fp32'),('m','unsqueeze'),('cast_fp32','multiply'),('unsqueeze','multiply'),('multiply','sum_hc_axis'),('sum_hc_axis','cast_original_dtype'),('cast_original_dtype','norm'),('norm','return_hidden'),('return_hidden','logits_processor')]
# Branches transcribed model.py 1595-1607; MTP/SP only when condition true.
def edges(sp,mtp):
 e=list(base)
 if sp and mtp:
  e=[( ('gather_h' if a=='h' and b=='cast_fp32' else 'gather_m' if a=='m' and b=='unsqueeze' else a),b) for a,b in e]
  e += [('h','gather_h'),('m','gather_m'),('gather_h','flatten'),('flatten','mtp_buffer_copy')]
 elif mtp: e += [('h','flatten'),('flatten','mtp_buffer_copy')]
 if sp and not mtp:
  e.remove(('cast_original_dtype','norm')); e += [('cast_original_dtype','gather_collapsed'),('gather_collapsed','norm')]
 return sorted(e)
def enumerate_connected(e):
 nodes=sorted({x for ab in e for x in ab})
 result=[]
 for k in (2,3,4):
  for c in itertools.combinations(nodes,k):
   cs=set(c); sub=[list(x) for x in e if set(x)<=cs]
   if len(sub)<k-1: continue
   seen={c[0]}
   while True:
    new=seen|{b for a,b in sub if a in seen}|{a for a,b in sub if b in seen}
    if new==seen: break
    seen=new
   if len(seen)==k: result.append({'nodes':list(c),'edges':sub})
 return result
variants=[]
for sp,mtp in itertools.product((False,True),repeat=2):
 e=edges(sp,mtp);c=enumerate_connected(e)
 variants.append({'sequence_parallel':sp,'mtp_buffer_present':mtp,'edges':e,'candidate_connected_sets':c,'counts':{str(k):sum(len(x['nodes'])==k for x in c) for k in (2,3,4)}})
out={'id':'FINAL_HEAD_SOURCE_BOUNDARY_V1','version':'1','source_snapshots':{str(x.relative_to(T)):sha(x) for x in (src,g1,g5)},'source_ranges':{'mtp_allocation':'model.py:975-986','target_boundary':'model.py:1592-1607','collapse':'model.py:854-856','logits':'model.py:1776-1783'},'scope':'Four source-level variants. Does not assert actual baseline flag values or lowered runtime kernels. enumerate 2-4 connected sets including boundary and side-effect nodes; connectivity only, no feasibility or byte savings inferred.','enumerator_sha256':sha(Path(__file__)),'variants':variants}
(D/'final-boundary-structural-scan.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
for v in variants: print('SP=%s MTP=%s counts=%s'%(v['sequence_parallel'],v['mtp_buffer_present'],v['counts']))
print('input_sha256',out['source_snapshots'])
