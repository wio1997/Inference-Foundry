from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import json,hashlib,sqlite3,socket,time
assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis')
R=P/'native-PMU-complete-v18';R.mkdir(exist_ok=True);OUT=R/'index.json';assert not OUT.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
idx=pin(P/'native-execution-identity-v14/index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4');caps=pin(P/'profile-capabilities-index-v12-v2.json','79ba556ef3855b9066392e438371a0a9d3b38ea7511269ce7ff2a6814529a095');t0=time.monotonic();refs=[]
for ref in idx['databases']:
 di=ref['database_index'];dest=R/('db-%02d.json'%di)
 if dest.exists():
  b=dest.read_bytes();x=json.loads(b);assert x['native_source_part_sha256']==ref['sha256'];refs.append(dict(database_index=di,path=str(dest),sha256=hashlib.sha256(b).hexdigest(),summary=x['summary']));continue
 d=pin(Path(ref['path']),ref['sha256']);path=Path(d['database_path']);before=path.stat();assert hashlib.sha256(path.read_bytes()).hexdigest()==d['database_sha256']
 db=sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True);db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON');db.set_progress_handler(lambda:int(time.monotonic()-t0>145),100000);strings=dict(db.execute('SELECT id,value FROM STRING_IDS'))
 groups=defaultdict(list);names=Counter();values=Counter()
 for raw in db.execute('SELECT rowid AS pmu_rowid,* FROM TASK_PMU_INFO'):
  r=dict(raw);r['metric_name']=strings.get(r['name'],str(r['name']));names[r['metric_name']]+=1;values[type(r['value']).__name__]+=1;groups[(r['globalTaskId'],r['timestampNs'])].append(r)
 old={(p['globalTaskId'],p['timestampNs']):p for p in d['PMU_execution_identity_candidates']};bygid=defaultdict(list);defs={c['globalTaskId']:c for c in d['COMPUTE_definitions']}
 for t in d['TASK_records']:bygid[t['globalTaskId']].append(t)
 out=[]
 for (gid,stamp),rs in groups.items():
  if (gid,stamp)in old:cs=old[(gid,stamp)]['containing_TASK_rows']
  else:cs=[t['task_record_rowid']for t in bygid[gid]if t['startNs']<=stamp<=t['endNs']]
  duplicate=Counter(r['metric_name']for r in rs);out.append(dict(globalTaskId=gid,timestampNs=stamp,native_TASK_candidates=cs,op_type=defs.get(gid,{}).get('opType_text'),metric_rows=rs,duplicate_metric_names={k:v for k,v in duplicate.items()if v>1}))
 oldmetrics={(p['globalTaskId'],p['timestampNs'],r['metric_name']):r['value']for p in old.values()for r in p['metric_rows']}
 newmetrics={(p['globalTaskId'],p['timestampNs'],r['metric_name']):r['value']for p in out for r in p['metric_rows']}
 assert all(newmetrics[k]==v for k,v in oldmetrics.items());db.close();after=path.stat();assert(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
 summary=dict(database_index=di,PMU_rows=sum(names.values()),metric_name_counts=dict(names),metric_groups=len(out),previous_four_metric_groups=len(old),extra_groups_vs_previous_four=len(set(groups)-set(old)),native_TASK_candidate_hist=dict(Counter(len(p['native_TASK_candidates'])for p in out)),duplicate_metric_group_count=sum(bool(p['duplicate_metric_names'])for p in out),value_types=dict(values),previous_four_metric_values_unchanged=True,source_DB_stat_unchanged=True)
 payload=(json.dumps(dict(native_source_part_sha256=ref['sha256'],database_path=str(path),database_sha256=d['database_sha256'],summary=summary,groups=out),ensure_ascii=False,separators=(',',':'))+'\n').encode();dest.write_bytes(payload);refs.append(dict(database_index=di,path=str(dest),sha256=hashlib.sha256(payload).hexdigest(),summary=summary))
def config_fields(v,prefix=''):
 out={}
 if isinstance(v,dict):
  for k,x in v.items():
   q=prefix+'.'+k if prefix else k
   if any(key in k.lower()for key in ['aic_metrics','aic_metric','profiler_level','export_type','record_shapes','with_stack']):out[q]=x
   elif isinstance(x,(list,dict)):out.update(config_fields(x,q))
 elif isinstance(v,list):
  for i,x in enumerate(v):out.update(config_fields(x,prefix+'['+str(i)+']'))
 return out
configs=[]
for ref in caps['profiler_infos']:
 obj=pin(Path(ref['path']),ref['sha256']);configs.append(dict(path=ref['path'],sha256=ref['sha256'],metric_configuration=config_fields(obj)))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),databases=len(refs),all_PMU_rows=sum(r['summary']['PMU_rows']for r in refs),all_metric_groups=sum(r['summary']['metric_groups']for r in refs),metric_name_counts=dict(sum((Counter(r['summary']['metric_name_counts'])for r in refs),Counter())),previous_four_metric_groups=sum(r['summary']['previous_four_metric_groups']for r in refs),extra_groups_vs_previous_four=sum(r['summary']['extra_groups_vs_previous_four']for r in refs),all_previous_four_values_unchanged=all(r['summary']['previous_four_metric_values_unchanged']for r in refs),metric_configurations=dict(Counter(json.dumps(c['metric_configuration'],sort_keys=True)for c in configs)),no_new_source_model_or_device_work=True,overall_complete=False,limits=['AllTASK_PMU_INFO rows retained,not a four-field-only selection. Exact UID/timestamp candidate cardinality retained; no nearest-time assignment.','Metric names andraw values are not automatically physical bytes,MAC work or normalized utilization.','Missing capture metric groups cannot be manufactured by source inspection or missing-as-zero.'])
payload=(json.dumps(dict(summary=summary,databases=refs,capture_configurations=configs),ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(payload)
first=pin(Path(refs[0]['path']),refs[0]['sha256']);samples=[]
for p in first['groups']:
 if p['op_type']not in {x['op_type']for x in samples}:samples.append(p)
 if len(samples)>=3:break
print(json.dumps(dict(summary=summary,first_group_samples=samples,full_result=str(OUT),result_sha256=hashlib.sha256(payload).hexdigest()),ensure_ascii=False))
