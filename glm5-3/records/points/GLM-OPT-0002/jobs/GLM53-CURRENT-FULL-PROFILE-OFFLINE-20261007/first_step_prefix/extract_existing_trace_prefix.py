from pathlib import Path
import json,gzip,base64,hashlib,collections
j=Path('/data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/jobs/GLM53-CURRENT-FULL-PROFILE-OFFLINE-20261007');result=[]
for rank in (0,13,15):
 identity=json.loads((j/'frontier'/('rank%d.json'%rank)).read_text());path=Path(identity['path']);raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==identity['sha256'];obj=json.loads(raw);del raw;es=obj['traceEvents'] if isinstance(obj,dict) else obj
 meta={e['pid']:e.get('args',{}).get('name') for e in es if e.get('ph')=='M' and e.get('name')=='process_name'}
 pids={p for p,n in meta.items() if n in ('Python','CANN','Ascend Hardware','Communication')};first=float(identity['graph_execute'][0]['event']['ts']);lo=first-1000000;hi=first
 selected=[dict(index=i,event=e) for i,e in enumerate(es) if e.get('pid') in pids and e.get('ph') in ('X','s','f') and lo<=float(e.get('ts',0))+float(e.get('dur',0)) and float(e.get('ts',0))<hi]
 data=dict(rank=rank,trace_path=str(path),trace_sha256=identity['sha256'],lo_us=lo,hi_us=hi,process_names=meta,events=selected,NPU_requests=0,new_profile=0,parser_rerun=0)
 blob=gzip.compress(json.dumps(data).encode());result.append(dict(rank=rank,data=base64.b64encode(blob).decode(),events=len(selected),bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest()))
 del obj,es,selected
print(json.dumps(result))
