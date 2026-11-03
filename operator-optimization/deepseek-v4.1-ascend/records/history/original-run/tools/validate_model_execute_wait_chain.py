"""Read exact submit-connection TASK names; do not infer causal overhead."""
from pathlib import Path
from collections import defaultdict,Counter
import json,sqlite3
T=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
ROOT=Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-20261028/diagnostic/runtime/diagnostic-traces')
O=T/'model-execute-wait-chain';O.mkdir(exist_ok=False)
calls=json.loads((T/'graph-submit-task-link-validation/call_results.json').read_text());byrank=defaultdict(list)
for x in calls:byrank[x['rank']].append(x)
detail=[]
for rank,rs in sorted(byrank.items()):
 db=next((ROOT/rank/'ASCEND_PROFILER_OUTPUT').glob('ascend_pytorch_profiler_*.db'))
 c=sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True);c.execute('PRAGMA query_only=ON');c.execute('PRAGMA temp_store=MEMORY')
 for r in rs:
  rows=list(c.execute('''SELECT t.startNs,t.endNs,t.modelId,t.streamId,t.taskId,s.value,t.connectionId
              FROM TASK t LEFT JOIN STRING_IDS s ON t.taskType=s.id
              WHERE t.connectionId=? AND t.modelId=? ORDER BY t.startNs''',(r['api_connection_id'],r['model_id'])))
  counts=Counter(x[5] for x in rows);execs=[x for x in rows if x[5]=='MODEL_EXECUTE'];waits=[x for x in rows if x[5]=='MODEL_WAIT_COMPLETE']
  linked_wait_starts={x[0] for x in r['linked_tasks']};identified=[x for x in waits if x[0] in linked_wait_starts]
  detail.append({'rank':rank,'dp':r['dp'],'ordinal':r['ordinal'],'interior':r['interior'],
    'connection_id':r['api_connection_id'],'model_id':r['model_id'],'same_connection_model_task_name_counts':dict(counts),
    'one_execute_one_wait':len(execs)==1 and len(identified)==1,'source_validated_layer_within_linked_wait':r['unique_same_model_parent_and_connection'],
    'tasks':rows,'cann_execute_api':r['cann_records'],
    'interpretation':'Recorded API connection joins MODEL_EXECUTE and MODEL_WAIT_COMPLETE with same modelId; wait interval contains the layer. Naming does not partition pure compute from waiting.'})
 c.close()
groups=defaultdict(list)
for d in detail:groups[d['ordinal']].append(d)
paired=[{'ordinal':o,'ranks':len(v),'all_interior':all(x['interior'] for x in v)} for o,v in sorted(groups.items()) if len(v)==16 and all(x['one_execute_one_wait'] and x['source_validated_layer_within_linked_wait'] for x in v)]
doc={'status':'EXPORTED_API_MODEL_EXECUTE_WAIT_CONNECTION_CHAIN','calls_checked':len(detail),
     'one_execute_one_wait_count':sum(x['one_execute_one_wait'] for x in detail),
     'also_unique_wait_contains_source_validated_layer_count':sum(x['one_execute_one_wait'] and x['source_validated_layer_within_linked_wait'] for x in detail),
     'name_counts':dict(sum((Counter(x['same_connection_model_task_name_counts']) for x in detail),Counter())),
     'fully_linked_16_rank_forwards':paired,
     'limitations':['Exact exported connection ID and modelId, with source layer located by a unique containing wait interval.',
       'Not a formal model-handle/request identity certificate, clock-calibration certificate or measurement of causal delay.',
       'Only 2ms-candidate subset included; 30 captured layer0 calls not covered.',
       'MODEL_WAIT_COMPLETE duration is not automatically pure NPU compute, pure host wait, or removable time.'],
     'no_device_work':True}
assert len(json.dumps(doc).encode())<32768
(O/'summary.json').write_text(json.dumps(doc,indent=2)+'\n');(O/'calls.json').write_text(json.dumps(detail,indent=2)+'\n');print(json.dumps(doc,indent=2))
