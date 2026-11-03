from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,hashlib,socket,statistics

assert socket.gethostname()=='bm-jn-zs-zone1-910C-64G-10-105'
P=Path('/data/tiankuan/wio/main/vllm-ascend-reproduce-172-on-166-20261030/analysis/whole-scope-continuation-20261030/hostprof-20261030T212812Z-360e0c72/analysis')
E=P/'native-execution-identity-v14';M=P/'native-PMU-complete-v18'
OUT=M/'current-compressor-state-and-pool-norm-boundaries-v24.json';assert not OUT.exists()
def pin(p,sha):
 b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==sha;return json.loads(b)
def stats(v):
 return dict(n=len(v),minimum=min(v),median=statistics.median(v),maximum=max(v))if v else None
res=pin(M/'resource-functional-attribution-and-uncertainty-v20-v3.json','c8e2c128d208597aeeb07a2b434506c930397d773b0ab5767a0fadba7e4840c4')
gr=pin(E/'graph-blocks-PMU-and-MC2-v14.json','a035c4de08191656141ad0727e8305a6444cc77e6184d4951f86136021af0991')
ix=pin(E/'index.json','a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4')
prep=pin(P/'prepare-field-origins-v12.json','c23b75f3fbd3ae3d3d7152f6882524e899bade53001ea2f1befa09ce943f091f')
shape_rows=[r for r in res['shape_role_context_regimes']if 'M08'in r['module_candidates']]
native_leaves=[r for r in res['attribution_rules']['native_leaf_assignments']if 'M08'in r['module_candidates']]
bydb=defaultdict(list)
for g in gr['graph_executions']:bydb[g['database_index']].append(g)
rows=[];errors=[];counts=Counter();gaps=defaultdict(list);intervening=Counter();typed=Counter();streamorders=Counter();phasecounts=Counter()
for ref in ix['databases']:
 di=ref['database_index'];d=pin(Path(ref['path']),ref['sha256']);tt={t['task_record_rowid']:t for t in d['TASK_records']};cc={c['globalTaskId']:c for c in d['COMPUTE_definitions']}
 def wrap(t):return dict(TASK=t,definition=cc[t['globalTaskId']])
 for g in bydb[di]:
  phase=g['phase_records'][0]['phase']if len(g['phase_records'])==1 else'PHASE_UNBOUND'
  for block in g['blocks']:
   tasks=sorted([tt[k]for k in block['task_record_rowids']],key=lambda t:(t['startNs'],t['endNs'],t['task_record_rowid']))
   pools=[t for t in tasks if cc[t['globalTaskId']]['opType_text']=='_pool_kernel']
   updates=[t for t in tasks if cc[t['globalTaskId']]['opType_text']=='_cache_update_kernel']
   counts['graph_pool']+=len(pools);counts['graph_ring_update']+=len(updates)
   if not pools and not updates:continue
   assert len(pools)==len(updates)==1,(di,g['graph_ordinal'],block['block_ordinal'])
   pool,update=pools[0],updates[0]
   key=[t for t in tasks if cc[t['globalTaskId']]['opType_text']in ['MatMulV2','MatMulV3']and cc[t['globalTaskId']]['inputShapes_text'].strip('"')=='24,512;128,512']
   norms=[t for t in tasks if cc[t['globalTaskId']]['opType_text']=='RmsNorm'and cc[t['globalTaskId']]['inputShapes_text'].strip('"')=='24,512;512'and t['startNs']>=pool['endNs']and(len(key)!=1 or t['endNs']<=key[0]['startNs'])]
   projections=[t for t in tasks if cc[t['globalTaskId']]['opType_text']in ['MatMulV2','MatMulV3']and cc[t['globalTaskId']]['outputDataTypes_text']=='FLOAT'and cc[t['globalTaskId']]['outputShapes_text'].strip('"')=='24,512']
   zeros=[t for t in tasks if cc[t['globalTaskId']]['opType_text']=='ZerosLike'and cc[t['globalTaskId']]['outputShapes_text'].strip('"')=='24,512']
   if not(len(key)==len(norms)==len(zeros)==1 and len(projections)==2):errors.append(dict(database_index=di,graph_ordinal=g['graph_ordinal'],block_ordinal=block['block_ordinal'],key=len(key),norms=len(norms),zeros=len(zeros),projections=len(projections)))
   rec=dict(database_index=di,graph_ordinal=g['graph_ordinal'],block_ordinal=block['block_ordinal'],phase=phase,pool=wrap(pool),ring_update=wrap(update),key_projections=[wrap(t)for t in key],pooled_norm_candidates=[wrap(t)for t in norms],input_projection_candidates=[wrap(t)for t in projections],pooled_zero_candidates=[wrap(t)for t in zeros])
   counts['selected_C2_blocks']+=1;phasecounts[phase]+=1
   if len(norms)==1:
    norm=norms[0];gaps['pool_to_update_ns'].append(update['startNs']-pool['endNs']);gaps['update_to_norm_ns'].append(norm['startNs']-update['endNs']);gaps['pool_to_norm_ns'].append(norm['startNs']-pool['endNs'])
    streamorders[str(pool['streamId']==update['streamId']==norm['streamId'])]+=1
    between=[t for t in tasks if t['task_record_rowid']not in[pool['task_record_rowid'],norm['task_record_rowid']]and pool['endNs']<=t['startNs']and t['endNs']<=norm['startNs']]
    intervening.update(cc[t['globalTaskId']]['opType_text']for t in between);rec['intervening_compute_between_pool_and_norm']=[wrap(t)for t in between]
   sig={k:cc[pool['globalTaskId']].get(k)for k in ['opType_text','inputShapes_text','inputDataTypes_text','outputShapes_text','outputDataTypes_text','blockNum','mixBlockNum']};typed[json.dumps(sig,sort_keys=True)]+=1
   rec['selected_publication_tasks']=[wrap(t)for t in tasks if cc[t['globalTaskId']]['opType_text']in ['ScatterNdUpdateSk','InplacePartialRotaryMul','DynamicQuant','VllmAscendQuantLightningIndexerV2','SparseFlashMla']]
   rows.append(rec)
assert counts['graph_pool']==counts['graph_ring_update']==counts['selected_C2_blocks']==240
metadata_keys=Counter();metadata_examples=[]
for root in prep['prepare_roots']:
 meta=root.get('metadata',{});metadata_keys.update(meta.keys())
 if len(metadata_examples)<2:metadata_examples.append(dict(root_key=root['root_key'],metadata=meta))
summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),counts=dict(counts),C2_block_ordinals=dict(Counter(r['block_ordinal']for r in rows)),phase_counts=dict(phasecounts),pair_identification_errors=errors,pool_update_norm_same_TASK_stream=dict(streamorders),boundary_gaps_ns={k:stats(v)for k,v in gaps.items()},intervening_pool_to_norm_compute=dict(intervening),pool_typed_regimes=[dict(signature=json.loads(k),executions=v)for k,v in typed.items()],M08_shape_role_context_regimes=len(shape_rows),native_M08_leaf_assignments=len(native_leaves),prepare_roots=len(prep['prepare_roots']),saved_metadata_key_presence=dict(metadata_keys),no_new_source_DB_model_device_or_service_work=True,overall_complete=False,limits=['Graph block/order/shape identifies structural candidates,not tensor pointer identity or allconsumers.','640phase-unbound graph layer blocks stayunbound;selectedC2phasecounts separate.','Metadata keys/host schedules do not supply unrecorded device payload,ring addresses or page generations.','Measured order and gaps are not necessary dependencies or removable latency.','No raw attrInfo is interpreted as own-op parameters;v23 restrictions persist.'])
result=dict(summary=summary,M08_shape_role_context_regimes=shape_rows,native_M08_leaf_assignments=native_leaves,graph_C2_boundaries=rows,prepare_metadata_examples=metadata_examples)
b=(json.dumps(result,ensure_ascii=False,separators=(',',':'))+'\n').encode();OUT.write_bytes(b)
public=dict(summary=summary,M08_role_counts=[dict(origin=r['origin'],role=r['role'],context=r['context'],signature=r['signature'],executions=r['summary']['executions'])for r in shape_rows],native_M08_leaf_assignments=native_leaves,first_boundary=rows[:1],prepare_metadata_examples=metadata_examples,full_result=str(OUT),result_sha256=hashlib.sha256(b).hexdigest())
print(json.dumps(public,ensure_ascii=False))
