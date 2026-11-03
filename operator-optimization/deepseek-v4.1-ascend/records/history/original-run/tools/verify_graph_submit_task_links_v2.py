"""Check recorded CANN connection IDs against device-model task windows."""
from pathlib import Path
from collections import defaultdict,Counter
import csv,json,sqlite3
T=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
ROOT=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces')
O=T/'graph-submit-task-link-validation-v2';O.mkdir(exist_ok=False)
allcalls=list(csv.DictReader((T/'graph-submit-candidate-analysis-v2/rank_layer0_submit_candidates.csv').open()))
mapping=json.loads((T/'captured-layer-attribution/module-call-metrics.json').read_text());byrank=defaultdict(list)
for r in allcalls:
 if r['candidate_count_in_10ms']=='1':byrank[r['rank']].append(r)
out=[];schemas={}
for rank,calls in sorted(byrank.items()):
 db=next((ROOT/rank/'ASCEND_PROFILER_OUTPUT').glob('ascend_pytorch_profiler_*.db'))
 c=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True);c.execute('PRAGMA query_only=ON');c.execute('PRAGMA temp_store=MEMORY')
 if not schemas:
  names=[r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'") if r[0].startswith('ENUM_')]
  for name in names:
   cols=list(c.execute('PRAGMA table_info("'+name.replace('"','""')+'")'))
   rows=list(c.execute('SELECT * FROM "'+name.replace('"','""')+'" LIMIT 3'))
   schemas[name]={'columns':cols,'first_three_rows':rows}
   if cols:
    col=cols[0][1];schemas[name]['possible_task_type_rows']=list(c.execute('SELECT * FROM "'+name.replace('"','""')+'" WHERE "'+col.replace('"','""')+'" IN (2957,2958) LIMIT 10'))
 for r in calls:
  s=int(r['layer_start_ns']);cid=int(r['candidate_connection_id'])
  m=[x for x in mapping if x['rank']==rank and x['layer']==0 and x['window_start_ns']==s];assert len(m)==1;m=m[0]
  # Find same-model device-task intervals covering the full source-validated
  # layer bracket, then ask if the selected API connection matches uniquely.
  sql='''SELECT startNs,endNs,modelId,streamId,taskId,taskType,connectionId,globalTaskId
         FROM TASK WHERE modelId=? AND startNs<=? AND endNs>=?'''
  enclosing=list(c.execute(sql,(int(m['model_id']),s,m['window_end_ns'])))
  linked=[x for x in enclosing if x[6]==cid]
  cann=list(c.execute('''SELECT a.startNs,a.endNs,s.value FROM CANN_API a JOIN STRING_IDS s ON a.name=s.id WHERE a.connectionId=?''',(cid,)))
  out.append({'rank':rank,'dp':r['dp'],'ordinal':int(r['target_forward_ordinal']),
    'interior':r['interior_runs_1_2_3']=='True','model_id':int(m['model_id']),'api_connection_id':cid,
    'cann_records':cann,'enclosing_same_model_task_count':len(enclosing),
    'enclosing_same_model_task_linked_to_api_count':len(linked),
    'linked_tasks':linked,'unique_same_model_parent_and_connection':len(enclosing)==1 and len(linked)==1,
    'claims_limit':'Exporter task interval + same modelId + exact connection join; formal runtime model-handle identity and clock-error bounds not independently established.'})
 c.close()
with (O/'call_results.json').open('w') as f:json.dump(out,f,indent=2)
groups=defaultdict(list)
for r in out:groups[r['ordinal']].append(r)
fully=[{'ordinal':o,'ranks':len(rs),'all_interior':all(x['interior'] for x in rs)} for o,rs in sorted(groups.items()) if len(rs)==16 and all(x['unique_same_model_parent_and_connection'] for x in rs)]
counts=Counter((r['enclosing_same_model_task_count'],r['enclosing_same_model_task_linked_to_api_count']) for r in out)
types=Counter(x[5] for r in out for x in r['linked_tasks'])
doc={'status':'EXACT_API_CONNECTION_AND_MODEL_TASK_INTERVAL_AUDIT','candidate_calls_checked':len(out),
 'unique_linked_count':sum(r['unique_same_model_parent_and_connection'] for r in out),
 'enclosing_vs_linked_counts':{str(k):v for k,v in counts.items()},'linked_task_types':dict(types),
 'paired_forward_groups_with_16_unique_links':fully,'enum_tables_sample':schemas,
 'limitations':['CANN connection IDs provide exact export joins. The source-validated layer is assigned to a containing same-model task by time interval and uniqueness.',
                'Even a unique exporter association is not independent validation of runtime clock synchronization or a causal estimate.',
                'The preceding 10ms candidate window excluded the other captured calls; no claim is made for those.'],
 'no_device_work':True}
assert len(json.dumps(doc).encode())<32768
(O/'summary.json').write_text(json.dumps(doc,indent=2)+'\n');print(json.dumps(doc,indent=2))
