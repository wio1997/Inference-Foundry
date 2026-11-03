from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis');E=P/'native-execution-identity-v14';M=P/'native-PMU-complete-v18';OUT=M/'M17-exact-probability-recovery-kernel-name-surface-v33.json';assert not OUT.exists()
def pin(p,s):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==s;return json.loads(b)
ix=pin(E/'index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4');tokens=['sample_recovered_tokens','rejection_random_sample','softmax','exponential','uniform','multinomial'];rows=[];types=Counter();hits=Counter()
for ref in ix['databases']:
 d=pin(Path(ref['path']),ref['sha256']);cc={c['globalTaskId']:c for c in d['COMPUTE_definitions']}
 for t in d['TASK_records']:
  c=cc.get(t['globalTaskId'])
  if not c:continue
  types[c['opType_text']]+=1;text=(c['opType_text']+' '+str(c.get('name_text',''))).lower();matched=[name for name in tokens if name in text]
  if matched:
   rows.append(dict(database_index=ref['database_index'],TASK_rowid=t['task_record_rowid'],op=c['opType_text'],name=c.get('name_text'),matched=matched,native_linked_API_rowids=t.get('native_linked_API_rowids',[]),native_link_field_present='native_linked_API_rowids'in t));hits.update(matched)
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),exact_name_queries=tokens,matched_counts={name:hits[name]for name in tokens},matched_TASKs=len(rows),all_ComputeInfo_optype_execution_counts=dict(types),only_name_surface_not_runtime_guard=True,overall_complete=False)
result=dict(summary=summary,matched_rows=rows,limits=['QueriesmatchComputeInfotype/nameonly;absenceisnotactualguardfalseorproofofnocustom/fused/unexportedwork.','Knownrecovery/randomkernelnamequeriescomplementearliergenerictypebandwhichdidnotinclude sample_recovered_tokens.','Noassignmentofunlinkedworkbyname/time;noCPU/device/requestgainclaim.','Nonewsource/model/torch/NPU/service/healthwork.'])
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
print(json.dumps(dict(summary=summary,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest()),ensure_ascii=False))
