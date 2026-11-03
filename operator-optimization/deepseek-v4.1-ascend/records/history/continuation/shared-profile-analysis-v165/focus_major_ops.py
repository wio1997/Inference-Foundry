"""Resolve only the ranking's major operator/collective questions from vendor DBs."""
import collections,concurrent.futures,json,sqlite3,statistics
from pathlib import Path
H=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/runtime/analysis/builtin-v164-official-db')
OPS={'DispatchFFNCombine','SparseFlashMla','VllmAscendQuantLightningIndexerV2','HcPre','QuantBatchMatmulV3','MatMulV2'}
METRICS={'aic_mac_ratio','aiv_vec_ratio','aiv_scalar_ratio','aiv_mte2_ratio','aiv_mte3_ratio'}
def summarize(db):
 c=sqlite3.connect('file:'+str(db)+'?mode=ro&immutable=1',uri=True);c.row_factory=sqlite3.Row
 ids={r['id']:r['value'] for r in c.execute('select id,value from STRING_IDS')};s=lambda x:ids.get(x,str(x))
 native={r['globalTaskId']:dict(r) for r in c.execute('select * from COMPUTE_TASK_INFO') if s(r['opType']) in OPS}
 metric_ids={i for i,v in ids.items() if v in METRICS};pmu=collections.defaultdict(dict);duplicates=0
 for r in c.execute('select globalTaskId,name,value from TASK_PMU_INFO where name in ('+','.join('?' for _ in metric_ids)+')',tuple(metric_ids)):
  if r['globalTaskId'] not in native:continue
  d=pmu[r['globalTaskId']];name=s(r['name'])
  if name in d:duplicates+=1;d[name]=None
  else:d[name]=r['value']
 groups={};examples=[];graph_domains=collections.Counter()
 for t in c.execute('select * from TASK'):
  x=native.get(t['globalTaskId'])
  if x is None:continue
  graph_domains[(s(x['opType']),t['modelId'],t['contextId'])]+=1
  key=json.dumps([s(x['opType']),s(x['inputShapes']),t['modelId'],t['contextId']],separators=(',',':'))
  g=groups.setdefault(key,{'durations_ns':[],'metrics':collections.defaultdict(list),'tasks_with_no_selected_pmu':0})
  g['durations_ns'].append(max(0,t['endNs']-t['startNs']))
  values=pmu.get(t['globalTaskId'],{})
  if not values:g['tasks_with_no_selected_pmu']+=1
  for name,v in values.items():
   if v is not None:g['metrics'][name].append(v)
  if s(x['opType'])=='DispatchFFNCombine' and len(examples)<2:
   links=list(c.execute('select p.rowid,p.name,p.callchainId from PYTORCH_API p join CONNECTION_IDS i on p.connectionId=i.id where p.type=50001 and i.connectionId=?',(t['connectionId'],)))
   if links:
    examples.append({'task_global_id':t['globalTaskId'],'connection_id':t['connectionId'],'model_id':t['modelId'],'context_id':t['contextId'],'input_shapes':s(x['inputShapes']),
      'CPU_connections':[{'rowid':r['rowid'],'op':s(r['name']),'callchain_id':r['callchainId'],
        'stack':[s(z[0]) for z in c.execute('select stack from PYTORCH_CALLCHAINS where id=? order by stackDepth',(r['callchainId'],))]} for r in links]})
 result=[]
 for key,g in groups.items():
  a=sorted(g['durations_ns']);result.append({'regime':json.loads(key),'count':len(a),'sum_ns':sum(a),'min_ns':a[0],'median_ns':statistics.median(a),'max_ns':a[-1],
      'reported_PMU':{name:{'count':len(v),'mean':statistics.mean(v),'min':min(v),'max':max(v)} for name,v in g['metrics'].items()},'tasks_with_no_selected_pmu':g['tasks_with_no_selected_pmu']})
 comm=[]
 for r in c.execute('select * from COMMUNICATION_OP'):
  comm.append({'op':s(r['opName']),'group':s(r['groupName']),'op_type':s(r['opType']),'count':r['count'],'data_type':s(r['dataType']),'rank_size':r['rankSize'],'device':r['deviceId'],'start_ns':r['startNs'],'end_ns':r['endNs'],'connection_id':r['connectionId']})
 c.close()
 return {'profile':db.parent.parent.name,'regimes':result,'explicit_graph_domains':[{'op':k[0],'model_id':k[1],'context_id':k[2],'count':v} for k,v in graph_domains.items()],
  'duplicate_selected_task_PMU_fields':duplicates,'DispatchFFNCombine_CPU_stack_examples':examples,'collective_records':comm}
if __name__=='__main__':
 dbs=sorted((H/'profiles').glob('*/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler*.db'));assert len(dbs)==32
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(summarize,dbs))
 out={'status':'MAJOR_OP_PROFILE_FOCUS_COMPLETE','profiles':rows,'scope':'Actual vendor task/shape/PMU/CPU-correlation records;no timing-nearest or cross-process pointer joins.',
      'limits':['PMU ratios are recorded pipeline metrics,not HBM bytes or a physical MFU ceiling.','Model/context sentinel values retained literally;not interpreted as independent graph identities.','Collective records do not prove a cross-rank dependency until group/operation incarnation and uniqueness are checked.'],'overall_complete':False}
 (H/'MAJOR-OP-FOCUS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'status':out['status'],'profiles':len(rows),'regimes':sum(len(r['regimes']) for r in rows)}))
