from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,hashlib,gzip,socket
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72')
ROOT=P/'analysis/native-execution-identity-v14';b=(ROOT/'saved-profile-surfaces-v14.json').read_bytes();assert hashlib.sha256(b).hexdigest()=='5e9b2972791b9d02814abea7256d07b72fbe043d715efab5f74cdcf518649792';inv=json.loads(b)
OUT=ROOT/'frontend-CPU-trace-inventory-v14.json';assert not OUT.exists()
results=[]
for spec in inv['interesting_files']:
 if not spec['relative_path'].endswith('.pt.trace.json.gz'):continue
 path=P/'runtime'/spec['relative_path'];raw=path.read_bytes();assert len(raw)==spec['bytes'];plain=gzip.decompress(raw);assert len(plain)<100*1024**2
 d=json.loads(plain);events=d.get('traceEvents',[]);cats=Counter(e.get('cat','NO_CATEGORY')for e in events);argkeys=Counter(k for e in events for k in e.get('args',{}));names=defaultdict(list);records=[]
 for i,e in enumerate(events):
  if e.get('ph')!='X':continue
  row={k:e[k]for k in ['cat','name','ph','pid','tid','ts','dur','id']if k in e};row['trace_event_index']=i
  # Do not copy arbitrary args: request/token text can be present in traces.
  row['relation_args']={k:v for k,v in e.get('args',{}).items()if k in ['Python id','Python parent id','Python module id','External id','Sequence number','Call stack']}
  records.append(row);names[(row.get('cat'),row['name'])].append(row)
 aggregates=[dict(category=k[0],name=k[1],records=len(v),sum_inclusive_us=sum(e.get('dur',0)for e in v),maximum_inclusive_us=max(e.get('dur',0)for e in v),pid_tid_pairs=sorted({(e.get('pid'),e.get('tid'))for e in v}))for k,v in names.items()]
 aggregates.sort(key=lambda r:r['sum_inclusive_us'],reverse=True)
 relevant=[a for a in aggregates if any(s in a['name'].lower()for s in ['tokeniz','detoken','output','stream','serialize','json','hash','process_request','generate','inputprocessor'])]
 result=dict(path=str(path),raw_sha256=hashlib.sha256(raw).hexdigest(),decompressed_sha256=hashlib.sha256(plain).hexdigest(),raw_bytes=len(raw),decompressed_bytes=len(plain),top_level_keys=list(d),top_level_scalars={k:v for k,v in d.items()if k!='traceEvents'and isinstance(v,(str,int,float,bool,type(None)))},event_categories=dict(cats),phase_counts=dict(Counter(e.get('ph')for e in events)),argument_key_counts=dict(argkeys),complete_events=records,function_name_aggregates=aggregates,relevant_name_aggregates=relevant)
 results.append(result)
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),frontend_files=len(results),complete_events=sum(len(r['complete_events'])for r in results),category_counts=dict(sum((Counter(r['event_categories'])for r in results),Counter())),relevant_name_groups=sum(len(r['relevant_name_aggregates'])for r in results),no_target_source_DB_or_device_work=True,whole_modules_closed=0,overall_complete=False,
 limits=['Inclusive function wall durations overlap and may include waiting; they are not exclusive CPU time, request gain or an on-CPU profile.', 'Only whitelisted relation arguments copied; raw trace remains immutable on server. Function names/absence alone do not settle every frontend branch or actual prefix-hash cost.'])
payload=(json.dumps(dict(summary=summary,files=results),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
public=dict(summary=summary,files=[{k:v for k,v in r.items()if k not in ['complete_events','function_name_aggregates','relevant_name_aggregates']}for r in results],top_inclusive_functions=[r['function_name_aggregates'][:15]for r in results],relevant_functions=[r['relevant_name_aggregates'][:35]for r in results],full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest())
if len(json.dumps(public).encode())>30000:public.pop('relevant_functions');public['display_truncated']=True
print(json.dumps(public,ensure_ascii=False))
