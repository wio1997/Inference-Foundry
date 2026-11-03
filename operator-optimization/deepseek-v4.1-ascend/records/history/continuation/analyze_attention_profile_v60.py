"""Reconcile native SMLA/QLI operators and exact graph chains, without workload execution."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter,defaultdict
import hashlib,json,statistics
from finalize_origin_coverage_v9 import N,O,read,write
from mirror_saved_profile_core_v53 import CACHE

def stat(xs):
 a=sorted(xs)
 return dict(n=len(a),minimum=a[0],median=statistics.median(a),p90=a[int(.9*(len(a)-1))],maximum=a[-1]) if a else None
def shapes(s):return [[int(x) for x in v.split(',')] if v else None for v in s.strip('"').split(';')]
def family(d):
 a=shapes(d['inputShapes_text']);types=d['inputDataTypes_text'].split(';');op=d.get('opType_text',d.get('operator'))
 if op=='SparseFlashMla':
  if types[2]!='DT_UNDEFINED':return 'CSA_C2' if a[2][1]==64 else 'CSA_C1'
  return 'ORI_SPARSE_DRAFT' if types[3]!='DT_UNDEFINED' else 'SWA_ONLY'
 if op=='VllmAscendQuantLightningIndexerV2':
  z=shapes(d['outputShapes_text'])
  return 'MODE2_CONSUMER' if a[-1] is not None else 'MODE1_CANDIDATE_SOURCE' if z[-1] is not None else 'MODE3_UNFILTERED'
 raise ValueError(op)

def main():
 assert not (O/'attention-indexer-profile-and-edges-v60.json').exists()
 stamp=datetime.now(timezone.utc).isoformat();led=read(O/'operator-shape-ceiling-ledger-v55.json')
 primary=['SparseFlashMla','VllmAscendQuantLightningIndexerV2'];helpers=['_quantize_indexer_query_kernel','_prepare_indexer_indices_kernel']
 regs=[r for r in led['observed_regimes'] if r['operator'] in primary+helpers]
 counts=Counter();rows=[]
 for r in regs:
  op=r['operator'];counts[op]+=r['executions'];row=dict(r)
  if op in primary:
   row['family']=family(r['signature']);row['M']=shapes(r['signature']['inputShapes_text'])[0][0]
   row['semantic_live_output_bytes']=row['M']*(64*512*2 if op=='SparseFlashMla' else (512*4+(2048*4 if row['family']=='MODE1_CANDIDATE_SOURCE' else 0)))
   row['empty_output_contract']='SMLAreturnLSEFalse→empty0elements;QLIreturnValue0→no sparsevalues;candidateoutputonlymode1. Emptyprofile-shape string isnot universallya scalar.'
  else:row['M']='Not recorded inComputeInfo;exactgraphchainsbindM3. No runtime shapeinvented forotherN/Aregimes.'
  rows.append(row)
 assert len(regs)==51 and counts=={'SparseFlashMla':4367,'VllmAscendQuantLightningIndexerV2':832,'_quantize_indexer_query_kernel':832,'_prepare_indexer_indices_kernel':832}
 graph=read(CACHE/'analysis/native-execution-identity-v14/graph-blocks-PMU-and-MC2-v14.json');pending=defaultdict(list)
 for g in graph['graph_executions']:pending[g['database_index']].append(g)
 native_counts=Counter();smla=[];qli=[];helper_counts=Counter();frames=[]
 for i in range(32):
  native=read(CACHE/f'analysis/native-execution-identity-v14/db-{i:02d}.json');ts={r['task_record_rowid']:r for r in native['TASK_records']};defs={r['globalTaskId']:r for r in native['COMPUTE_definitions']}
  def op(t):return defs[t['globalTaskId']]['opType_text']
  def compact(t):return dict(task_record_rowid=t['task_record_rowid'],op=op(t),streamId=t['streamId'],startNs=t['startNs'],endNs=t['endNs'],duration_ns=t['endNs']-t['startNs'])
  for t in ts.values():
   if t['globalTaskId'] in defs and op(t) in primary+helpers:native_counts[op(t)]+=1
  for g in pending[i]:
   frameS=frameQ=frameP=frameR=0;qli_blocks=[]
   for b in g['blocks']:
    tasks=sorted([ts[rid] for rid in b['task_record_rowids']],key=lambda t:(t['startNs'],t['task_record_rowid']))
    ss=[t for t in tasks if op(t)=='SparseFlashMla'];assert len(ss)==1;s=ss[0];d=defs[s['globalTaskId']];f=family(d);idx=tasks.index(s)
    rope=tasks[idx+1];trans=tasks[idx+2]
    assert op(rope)=='InplacePartialRotaryMul' and op(trans)=='Transpose'
    assert shapes(defs[rope['globalTaskId']]['inputShapes_text'])==[[3,1,64,512],[3,1,1,64],[3,1,1,64]]
    assert shapes(defs[trans['globalTaskId']]['inputShapes_text'])==[[3,8,8,512],[4]]
    assert s['streamId']==rope['streamId']==trans['streamId']
    common=dict(database_index=i,graph_ordinal=g['graph_ordinal'],block_ordinal=b['block_ordinal'],root_bound=bool(g['prepare_root_keys']))
    smla.append(dict(common,family=f,SMLA=compact(s),inverse_RoPE=compact(rope),send_layout=compact(trans),
      SMLA_to_RoPE_gap_ns=rope['startNs']-s['endNs'],RoPE_to_layout_gap_ns=trans['startNs']-rope['endNs'],
      causal_scope='Graphblock+exactoperatorIDs andpinnedcaller;notactualtensor alias/generation/readiness proof.'))
    frameS+=s['endNs']-s['startNs'];frameR+=rope['endNs']-rope['startNs']
    qq=[t for t in tasks if op(t)=='VllmAscendQuantLightningIndexerV2']
    if not qq:continue
    assert len(qq)==1;q=qq[0];qidx=tasks.index(q);prepare=tasks[qidx+1]
    quant=[t for t in tasks if op(t)=='_quantize_indexer_query_kernel'];assert len(quant)==1
    quant=quant[0];assert op(prepare)=='_prepare_indexer_indices_kernel'
    assert q['streamId']==prepare['streamId'];assert quant['endNs']<=q['startNs']
    f=family(defs[q['globalTaskId']]);qli_blocks.append(b['block_ordinal'])
    qli.append(dict(common,mode=f,QLI=compact(q),query_quant=compact(quant),prepare=compact(prepare),SMLA_task_rowid=s['task_record_rowid'],
      QLI_to_prepare_gap_ns=prepare['startNs']-q['endNs'],prepare_to_SMLA_span_ns=s['startNs']-prepare['endNs'],
      quant_to_QLI_gap_ns=q['startNs']-quant['endNs'],
      intervening_after_prepare=dict(Counter(op(t) for t in tasks[qidx+2:idx]))))
    frameQ+=q['endNs']-q['startNs'];frameP+=prepare['endNs']-prepare['startNs']
   assert qli_blocks==[2,8,14,20,24,28,32,36]
   frames.append(dict(database_index=i,graph_ordinal=g['graph_ordinal'],root_bound=bool(g['prepare_root_keys']),SMLA_task_sum_ns=frameS,QLI_task_sum_ns=frameQ,
    prepare_task_sum_ns=frameP,inverse_RoPE_task_sum_ns=frameR,graph_device_control_span_ns=g['graph_control_pair_ns'],
    SMLA_exposure=frameS/g['graph_control_pair_ns'],QLI_exposure=frameQ/g['graph_control_pair_ns']))
 assert native_counts==counts and len(smla)==3200 and len(qli)==640
 groupS=defaultdict(list);groupQ=defaultdict(list);groupP=defaultdict(list);groupQuant=defaultdict(list)
 for r in smla:
  if r['root_bound']:groupS[r['family']].append(r['SMLA']['duration_ns'])
 for r in qli:
  if r['root_bound']:
   for d,key in [(groupQ,'QLI'),(groupP,'prepare'),(groupQuant,'query_quant')]:d[r['mode']].append(r[key]['duration_ns'])
 bound=[r for r in frames if r['root_bound']]
 write(O/'attention-indexer-profile-and-edges-v60.json',dict(generated_utc=stamp,regime_count=51,primary_regimes=39,helper_regimes=12,regimes=rows,
  native_execution_counts=dict(native_counts),graph_SMLA_chains=smla,graph_QLI_chains=qli,graph_frames=frames,
  root_SMLA_ns={k:stat(v) for k,v in groupS.items()},root_QLI_ns={k:stat(v) for k,v in groupQ.items()},
  root_prepare_ns={k:stat(v) for k,v in groupP.items()},root_quant_ns={k:stat(v) for k,v in groupQuant.items()},
  root_frame_SMLA_exposure=stat([r['SMLA_exposure'] for r in bound]),root_frame_QLI_exposure=stat([r['QLI_exposure'] for r in bound]),
  root_frame_QLI_task_sum_ns=stat([r['QLI_task_sum_ns'] for r in bound]),
  limits=['32DBnativeexecutions conserved,notonerequestcriticalpath. 64root/16phase-unboundgraphs retained.',
   'HelperN/AComputeInfo cannotbeassignedallruntimeMbynearesttime;M3exactgraphrolechains useoperatorIDs andpinnedsource.',
   'Profile hasnoactualtensorvalues/opargs,so perqueryvalidlength,TopKvalidcount/cachepageunion/candidatecontents remain symbolic.',
   'Exposure shares are not MFU/achievablespeedup/strictendtoendbound. Unbound/context regimes withsameshape arenotpairedbenchmarks.'],overall_complete=False))
 print(json.dumps(dict(regimes=51,native_counts=dict(native_counts),SMLA_graph_chains=len(smla),QLI_graph_chains=len(qli),
   root_SMLA_ns={k:stat(v) for k,v in groupS.items()},root_QLI_ns={k:stat(v) for k,v in groupQ.items()},
   root_prepare_ns={k:stat(v) for k,v in groupP.items()},root_quant_ns={k:stat(v) for k,v in groupQuant.items()},
   root_QLI_exposure=stat([r['QLI_exposure'] for r in bound]),overall_complete=False)))

if __name__=='__main__':main()
