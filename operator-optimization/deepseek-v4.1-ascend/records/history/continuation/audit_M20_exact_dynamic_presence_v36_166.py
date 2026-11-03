from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,hashlib,re,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
E=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis/native-execution-identity-v14');M=E.parent/'native-PMU-complete-v18';OUT=M/'M20-exact-confidence-scheduler-and-verify-consumer-native-presence-v36.json';assert not OUT.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
fs=pin(E/'native-compute-fullchains-v14-v2.json','6c7bf4bd3a7dcf557327aeadc4a37ec4b9b23cb1ba675353bbd6e5d56a4e1ca0');ix=pin(E/'index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4');rows=[];selected={}
def roles(frames):
 result=set()
 specs=[('/vllm_ascend/spec_decode/utils.py',278,635,'DYNAMIC_SCHEDULER'),('/model_executor/models/qwen3_dspark.py',185,189,'CONFIDENCE_HEAD'),('/deepseek_v41/dspark.py',439,441,'CONFIDENCE_WRAPPER'),('/worker/model_runner_v1.py',2205,2226,'VERIFY_RESULT_CONSUMER'),('/spec_decode/llm_base_proposer.py',2211,2228,'BACKUP_CLONE_DISCARD_COMPUTE_SURFACE')]
 for f in frames:
  for path,lo,hi,role in specs:
   m=re.search(re.escape(path)+r'\((\d+)\)',f)
   if m and lo<=int(m.group(1))<=hi:result.add(role)
 return sorted(result)
for db in fs['databases']:
 di=db['database_index'];selected[di]={}
 for a in db['API_records']:
  rr=roles(a['project_frames'])
  if rr:rows.append(dict(database_index=di,API=a,roles=rr));selected[di][a['api_rowid']]=rr
tasks=[]
for ref in ix['databases']:
 di=ref['database_index']
 if not selected[di]:continue
 d=pin(Path(ref['path']),ref['sha256']);cc={c['globalTaskId']:c for c in d['COMPUTE_definitions']}
 for t in d['TASK_records']:
  hits=set(t.get('native_linked_API_rowids',[]))&selected[di].keys()
  if hits:tasks.append(dict(database_index=di,TASK=t,definition=cc.get(t['globalTaskId']),roles=sorted({r for a in hits for r in selected[di][a]})))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),all_native_API_records_checked=sum(len(db['API_records'])for db in fs['databases']),matched_APIs=len(rows),matched_TASKs=len(tasks),API_roles=dict(Counter(r for a in rows for r in a['roles'])),TASK_roles=dict(Counter(r for t in tasks for r in t['roles'])),TASK_types=dict(Counter(t['definition']['opType_text']if t['definition']else'NO_COMPUTE'for t in tasks)),overall_complete=False)
result=dict(summary=summary,APIs=rows,TASKs=tasks,limits=['Exact source-location query supplements zero confidence-name hits; no runtime flag inferred from absence.','This native ComputeInfo-linked surface does not exhaust memcpy/H2D/D2H/CPU int(Tensor)/synchronization records.','Source CPU backup/clone/discard and consumer scalar conversions remain documented even when0ComputeInfo hits.','No source/model/torch/NPU/service/health work.'])
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(summary=summary,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
