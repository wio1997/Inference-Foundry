"""Rank recorded native names without inventing missing Level0 opType/CPU data."""
import collections,concurrent.futures,hashlib,json,sqlite3
from pathlib import Path
H=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/npu166-official-db')
def union_ns(intervals):
 total=0;end=None
 for a,b in sorted(intervals):
  if end is None or a>end:total+=max(0,b-a);end=b
  elif b>end:total+=b-end;end=b
 return total
def one(db):
 c=sqlite3.connect('file:'+str(db)+'?mode=ro&immutable=1',uri=True);c.row_factory=sqlite3.Row
 tables={r[0] for r in c.execute("select name from sqlite_master where type='table'")}
 strings=dict(c.execute('select id,value from STRING_IDS'));s=lambda i:strings.get(i,str(i))
 info={r['globalTaskId']:dict(r) for r in c.execute('select * from COMPUTE_TASK_INFO')}
 groups={};regimes={};models={};counts=collections.Counter();controls=[]
 for t in c.execute('select rowid,* from TASK'):
  counts[s(t['taskType'])]+=1
  if s(t['taskType']) in ['MODEL_EXECUTE','MODEL_WAIT_COMPLETE']:controls.append(dict(t))
  x=info.get(t['globalTaskId'])
  if not x:continue
  name=s(x['name']);a=t['startNs'];b=t['endNs'];g=groups.setdefault(name,{'count':0,'sum_ns':0,'min_ns':None,'max_ns':0});g['count']+=1;g['sum_ns']+=b-a;g['min_ns']=min(g['min_ns'],b-a) if g['min_ns'] is not None else b-a;g['max_ns']=max(g['max_ns'],b-a)
  key=json.dumps([name,s(x['opType']),s(x['inputDataTypes']),s(x['inputShapes']),s(x['outputShapes']),x['blockNum'],t['modelId']],separators=(',',':'))
  g=regimes.setdefault(key,{'count':0,'sum_ns':0,'min_ns':None,'max_ns':0});g['count']+=1;g['sum_ns']+=b-a;g['min_ns']=min(g['min_ns'],b-a) if g['min_ns'] is not None else b-a;g['max_ns']=max(g['max_ns'],b-a)
  model=models.setdefault(str(t['modelId']),{'intervals':[],'compute_count':0,'sum_ns':0});model['intervals'].append((a,b));model['compute_count']+=1;model['sum_ns']+=b-a
 for g in models.values():
  intervals=g.pop('intervals');g['active_union_ns']=union_ns(intervals);g['span_ns']=max(b for a,b in intervals)-min(a for a,b in intervals)
 extras={name:c.execute('select count(*) from '+name).fetchone()[0] if name in tables else 0 for name in ['PYTORCH_API','TASK_PMU_INFO']}
 c.close();h=hashlib.sha256()
 with db.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return {'profile':db.parent.parent.name,'database':str(db),'db_bytes':db.stat().st_size,'db_sha256':h.hexdigest(),
  'task_types':dict(counts),'native_names':groups,'regimes':regimes,'native_model_scopes':models,'native_model_controls':controls,
  'CPU_and_PMU_rows':extras,'opType_field_unavailable_does_not_identify_operator_from_shape':True}
if __name__=='__main__':
 dbs=sorted((H/'profiles').glob('*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler*.db'));assert len(dbs)==32
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,dbs))
 names={}
 for r in rows:
  for name,g in r['native_names'].items():
   x=names.setdefault(name,{'count':0,'sum_ns_across_ranks_windows':0,'profiles':0});x['count']+=g['count'];x['sum_ns_across_ranks_windows']+=g['sum_ns'];x['profiles']+=1
 ranking=sorted([{'native_name':k,**v} for k,v in names.items()],key=lambda x:-x['sum_ns_across_ranks_windows'])
 out={'status':'ALL32_NPU_ONLY_NATIVE_NAME_RANKING','profiles':rows,'native_name_ranking':ranking,
  'limits':['Level0 opType is N/A;rank by the recorded native name. Do not split AivKernel into collective families by name/shape guesses.',
    'Model IDs are local to current process/device/lifetime;do not join oldmodel49 to newmodel49.',
    'Different window lengths and varying workload states preclude comparing total task shares as production speedup.',
    'No CPU source/operand/all-reader relation is supplied by this NPU-only cohort.'],
  'historical_native_or_phase_promotions':0,'whole_modules_closed':0,'overall_complete':False}
 (H/'ALL32-NPU-ONLY-RANKING.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'status':out['status'],'profiles':len(rows),'compute_executions':sum(sum(g['count'] for g in r['native_names'].values()) for r in rows),'top':ranking[:10],'CPU_and_PMU_rows':{k:sum(r['CPU_and_PMU_rows'][k] for r in rows) for k in ['PYTORCH_API','TASK_PMU_INFO']}}))
