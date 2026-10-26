"""Controller-only independent audit of already captured 1220/1280 raw artifacts."""
import collections,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];run=ROOT/'runs/20261017T1280-vllm-v4-long-state-v1'
receipt=json.loads((run/'receipt.json').read_text());assert all(v==0 for v in receipt['nodes'].values())
objects={};mc={};pd=[];native=[]
for node in [30,31,43,45]:
 raw=(run/f'node-{node}.json').read_bytes();outer=json.loads(raw);assert outer['rc']==0
 o=json.loads(outer['stdout']);objects[node]=o;assert o['model']['state']['Running'] and o['/health']['status']==200
 rows=[]
 for name,a in o['persistent_audit'].items():
  # Validate canonical capture content against the original JSONL SHA. Separators/ordering match helper's sorted dump.
  data=''.join(json.dumps(r,sort_keys=True)+'\n' for r in a['rows']).encode()
  assert len(data)==a['bytes'] and hashlib.sha256(data).hexdigest()==a['sha256'],name
  rows.extend(a['rows'])
 native.extend(r for r in rows if r['event']=='initialize_complete')
 batches=[r for r in rows if r['event']=='native_get_batch_complete']
 if node in [30,31]:
  assert len(batches)==152
  bypid=collections.defaultdict(list)
  for r in batches:
   assert r['total_key_count']==581 and r['batch_limit_blocks']==32 and r['native_results']==[0]*r['key_count'] and not r['fallback'] and r['retry_attempts']==0
   bypid[r['pid']].append(r)
  assert len(bypid)==8
  for pid,rs in bypid.items():
   assert len(rs)==19 and sorted(r['first_key_index'] for r in rs)==list(range(0,581,32))
   assert sum(r['key_count'] for r in rs)==581
   assert sum(r['payload_bytes_requested'] for r in rs)==57495388160//8
  mc[str(node)]={'workers':8,'complete_batches':152,'keys_per_worker':581,'payload_descriptor_bytes':sum(r['payload_bytes_requested'] for r in batches),'native_result_codes':[0],'retry_attempts':0,'fallback':False}
 for r in rows:
  if r['event']=='model_state_read_complete' and any('sem-vllm-v4' in s for s in r.get('request_ids',[])):
   assert r['rc']==0;pd.append({'node':node,**r})
assert len(native)==64 and all(r['protocol']=='SDMA' and r['protocol_value']==1 and r['rc']==0 for r in native)
assert len(pd)==12 and sum(r['payload_bytes_completed'] for r in pd)==90646773760
metrics=objects[43]['semantic_metrics_artifacts'];assert len(metrics)==36
original_index=json.loads((ROOT/'runs/20261017T1220-long-semantic-v2/artifact-index.json').read_text())
# Every captured metric body has its own SHA and must exactly match the independently extracted original artifact.
for fn,a in metrics.items():
 assert hashlib.sha256(a['text'].encode()).hexdigest()==a['sha256']
 assert original_index[fn]['sha256']==a['sha256'] and original_index[fn]['bytes']==len(a['text'].encode())
 local=run/'artifacts'/fn;local.parent.mkdir(exist_ok=True);local.write_text(a['text'])
def val(fn,name,engine=None,source=None):
 values=[]
 for line in metrics[fn]['text'].splitlines():
  if not line.startswith(name+'{'):continue
  if engine is not None and f'engine="{engine}"' not in line:continue
  if source is not None and f'source="{source}"' not in line:continue
  values.append(float(line.rsplit(' ',1)[1]))
 assert values,(fn,name,engine,source)
 return sum(values)
cache=[]
for case,node,engine in [(0,30,1),(1,31,3)]:
 before=f'metrics-case{case}-prefix-warmup-PD-node{node}.txt';after=f'metrics-case{case}-cross-DP-storage-PD-node{node}.txt'
 delta={s:val(after,'vllm:prompt_tokens_by_source_total',engine,s)-val(before,'vllm:prompt_tokens_by_source_total',engine,s) for s in ['local_compute','local_cache_hit','external_kv_transfer']}
 assert delta=={'local_compute':5632,'local_cache_hit':0,'external_kv_transfer':74368},delta
 cache.append({'semantic_case':case,'P_global_DP':engine,'source_token_delta':delta})
mtp=[];prev='before'
for case in [0,1]:
 for label in ['direct-D','cold-PD','prefix-warmup-PD','cross-DP-storage-PD']:
  tag=f'case{case}-{label}'
  delta={k:sum(val(f'metrics-{tag}-node{n}.txt',f'vllm:spec_decode_num_{k}_total')-val(f'metrics-{prev}-node{n}.txt',f'vllm:spec_decode_num_{k}_total') for n in [43,45]) for k in ['drafts','draft_tokens','accepted_tokens']}
  if label!='prefix-warmup-PD':assert delta['drafts']>0 and delta['draft_tokens']==5*delta['drafts'] and delta['accepted_tokens']>0
  delta['accepted_fraction']=delta['accepted_tokens']/delta['draft_tokens'] if delta['draft_tokens'] else None
  mtp.append({'case':tag,**delta});prev=tag
report={'project_id':'glm53','status':'TWO_FACTUAL_GUARDS_AND_COMPLETE_NATIVE_CACHE_PD_AUDIT_PASSED','run':'runs/20261017T1220-long-semantic-v2','audit_run':str(run.relative_to(ROOT)),'formal_requests':0,'baseline_accepted':False,'MC_readback':mc,'MC_total_payload_descriptor_bytes':sum(x['payload_descriptor_bytes'] for x in mc.values()),'PD_completed_worker_reads':len(pd),'PD_payload_descriptor_bytes':sum(r['payload_bytes_completed'] for r in pd),'physical_link_bytes':None,'SDMA_initialized_workers':len(native),'cross_P_DP_cache_origins':cache,'native_MTP_metric_increments':mtp,'metrics_count':len(metrics),'limitations':['Two factual retrieval guards only; broad numerical equivalence unknown, whole output hashes differ.','Functional timing cannot identify graph benefit: target graphs and SDK log level changed together vs v3.','Per-request graph runtime metrics not enabled yet; startup capture/replay-once logs can include warmup.','Descriptor payload bytes include logical copies/replication; physical link counters not available.']}
(ROOT/'results/vllm-v4-long-native-audit.json').write_text(json.dumps(report,indent=2)+'\n')
p=ROOT/'results/model-long-semantic-v2.json';o=json.loads(p.read_text());o['status']='V4_TARGET_GRAPH_TWO_80K_FACTUAL_GUARDS_AND_COMPLETE_NATIVE_CACHE_PD_AUDIT_PASSED';o['server_audit']='results/vllm-v4-long-native-audit.json';o['timing_confound']='Graph activation and SDK log level changed together versus v3; functional elapsed differences are not causal graph performance.';p.write_text(json.dumps(o,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['native_MTP_metric_increments','MC_readback','limitations']},indent=2));print('MTP',mtp)
