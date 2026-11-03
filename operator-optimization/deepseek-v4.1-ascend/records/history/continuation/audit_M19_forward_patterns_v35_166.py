from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,re,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
SRC=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-PMU-complete-v18/M19-allrank-DSpark-forward-embedding-HC-decoder-and-communication-scope-v35.json');b=SRC.read_bytes();assert hashlib.sha256(b).hexdigest()=='abdd564691080a7f537a4f65668d72768461ad637e29633641c77e05a5ba5fa6';d=json.loads(b);OUT=SRC.parent/'M19-source-ordered-forward-patterns-typed-rows-and-group-audit-v35.json';assert not OUT.exists()
headers={h['chain_key']:h for h in d['chain_headers']};tt=defaultdict(list)
for t in d['native_TASKs']:
 for aid in{a['api_rowid']for a in t['source_APIs']}:tt[(t['database_index'],aid)].append(t)
def types(tasks):return dict(Counter(t['definition']['opType_text']if t['definition']else'NO_COMPUTE'for t in tasks))
def rows(c):
 s=c.get('inputShapes_text','');m=re.search(r'-?\d+',s);return int(m.group())if m else None
streams=defaultdict(list)
for a in d['native_APIs']:streams[(a['database_index'],a['API']['globalTid'])].append(a)
cycles=[];bad=[];covered=set()
for (di,tid),aa in streams.items():
 cur=None
 for a in sorted(aa,key=lambda a:(a['API']['startNs'],a['API']['api_rowid'])):
  api=a['API'];tasks=tt[(di,api['api_rowid'])];start=any(t['definition']and t['definition']['opType_text']=='Tile'for t in tasks)and any('/dspark.py(249)'in f for f in api['project_frames']);end=any('/dspark.py(275)'in f for f in api['project_frames'])
  if start:
   if cur is not None:bad.append(dict(kind='UNFINISHED_BEFORE_NEXT_REPEAT',database_index=di,start=cur[0]['API']['api_rowid']))
   cur=[]
  if cur is not None:cur.append(a)
  if end and cur is not None:
   unique={(t['database_index'],t['TASK']['task_record_rowid']):t for x in cur for t in tt[(di,x['API']['api_rowid'])]};tasks=list(unique.values());c=types(tasks)
   tile=next(t['definition']for t in tasks if t['definition']and t['definition']['opType_text']=='Tile');hc=next(t['definition']for t in tasks if t['definition']and t['definition']['opType_text']=='HcPre');empty=[t for t in tasks if t['definition']and t['definition']['opType_text']=='MatMulV2'and any(': _project_kv'in f for x in t['source_APIs']for f in x['project_frames'])]
   check=c.get('Tile')==1 and c.get('HcPre')==6 and c.get('HcPost')==6 and c.get('DispatchFFNCombine')==3 and c.get('ReduceSum')==1
   z=dict(database_index=di,globalTid=tid,start_repeat_API=cur[0]['API']['api_rowid'],end_gather_API=api['api_rowid'],phases=sorted({x['phase']for x in cur}),global_repeat_rows=rows(tile),local_first_HC_rows=rows(hc),type_counts=c,TASKs=len(tasks),cache_only_KV=len(empty),sparse_attention=c.get('SparseFlashMla',0),TASK_keys=[list(k)for k in unique],classification='COMPLETE_SOURCE_ORDER_PATTERN'if check else'PATTERN_MISMATCH');cycles.append(z);covered.update(unique)
   if not check:bad.append(dict(kind='PATTERN_MISMATCH',cycle=z))
   cur=None
 if cur is not None:bad.append(dict(kind='UNFINISHED_END',database_index=di,start=cur[0]['API']['api_rowid']))
unassigned=[t for t in d['native_TASKs']if(t['database_index'],t['TASK']['task_record_rowid'])not in covered]
focus=[]
for h in headers.values():
 if any(c['signature']['op']in['Tile','HcPre','HcPost','RmsNormCast','MatMulV2','SparseFlashMla','ScatterNdUpdateSk','TransposeBatchMatMul']for c in h['typed_shapes']):
  focus.append(dict(chain_key=h['chain_key'],phases=h['phases'],TASKs=h['TASKs'],typed_shapes=h['typed_shapes'],leaf_project_frames=h['project_frames'][:3],specific_model_sites=[f for f in h['project_frames']if '/deepseek_v41/model.py('in f or '/attention/dsa_v41.py('in f]))
group_rows=[]
for r in d['collectives']:
 fs=[f for a in r['source_candidates']for f in a['API']['project_frames']];role='EMBEDDING'if any('vocab_parallel_embedding.py'in f for f in fs)else'FINAL_SP_GATHER'if any('/dspark.py(275)'in f for f in fs)else'DECODER_INPUT_SP_GATHER'if any('/model.py(898)'in f for f in fs)else'DECODER_OTHER'
 op=r['operation'];group_rows.append(dict(database_index=r['database_index'],COMM_OP_rowid=op['COMM_OP_rowid'],role=role,group_name=op.get('group_name'),family=op['family'],count=op['count'],dtype=op['dtype'],rankSize=op['rankSize'],phases=r['phases'],source_frames=fs))
groups=[]
for di in sorted({r['database_index']for r in group_rows}):
 z=[r for r in group_rows if r['database_index']==di];sets={role:sorted({str(r['group_name'])for r in z if r['role']==role})for role in{r['role']for r in z}};groups.append(dict(database_index=di,groups_by_role=sets,all_same_opaque_group_name=len({str(r['group_name'])for r in z})==1))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),complete_forward_patterns=len(cycles),pattern_issues=len(bad),pattern_regimes=[dict(phase=k[0],global_rows=k[1],local_rows=k[2],cache_only_KV_per_pattern=k[3],sparse_attention_per_pattern=k[4],patterns=v)for k,v in Counter(('|'.join(c['phases']),c['global_repeat_rows'],c['local_first_HC_rows'],c['cache_only_KV'],c['sparse_attention'])for c in cycles).items()],covered_after_repeat_TASKs=len(covered),outside_pattern_TASKs=len(unassigned),outside_pattern_types=types(unassigned),groups=groups,group_role_regimes=[dict(role=k[0],family=k[1],count=k[2],rankSize=k[3],operations=v)for k,v in Counter((r['role'],r['family'],r['count'],r['rankSize'])for r in group_rows).items()],whole_module_closed=False,overall_complete=False)
result=dict(summary=summary,cycles=cycles,issues=bad,unassigned_TASKs=unassigned,focus_chain_headers=focus,communication_roles=group_rows,limits=['Source-order pattern is not a formal Python invocation/request/pointer epoch; no nearest-time association to target roots.','Opaque group-name equality/rankSize supports same recorded communication group, not topology or new collective numerical equivalence.','Unassigned tasks before repeat retained, not dropped.','Cache-only KV is positive global work; no local Q does not imply no computation/collective/cache requirement.','CPU saved-profile only; no source/model/torch/NPU/service/health work.'])
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(summary=summary,focus_chain_headers=focus,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
