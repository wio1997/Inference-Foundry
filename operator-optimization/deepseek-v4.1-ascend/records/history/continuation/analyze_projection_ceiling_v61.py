"""Captured projection geometry, source-conditioned BMM interpretation and exact graph chains."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,math,statistics,hashlib
from finalize_origin_coverage_v9 import N,O,read,write
from mirror_saved_profile_core_v53 import CACHE

def shape(s):return [[int(x) for x in a.split(',')] if a else None for a in s.strip('"').split(';')]
def stat(xs):
 a=sorted(xs)
 return dict(n=len(a),minimum=a[0],median=statistics.median(a),p90=a[int(.9*(len(a)-1))],maximum=a[-1]) if a else None
KEYS=['blockNum','mixBlockNum','opType_text','inputShapes_text','inputDataTypes_text','outputShapes_text','outputDataTypes_text']
def key(d):return tuple(d[k] for k in KEYS)

def main():
 assert not (O/'projection-work-and-reference-bounds-v61.json').exists()
 stamp=datetime.now(timezone.utc).isoformat();led=read(O/'operator-shape-ceiling-ledger-v55.json')
 ops=['MatMulV2','MatMulV3','QuantBatchMatmulV3','TransposeBatchMatMul'];mods={'M07','M08','M09','M11'}
 regs=[r for r in led['observed_regimes'] if r['operator'] in ops and mods.intersection(r['module_candidates'])]
 assert len(regs)==69 and sum(r['executions'] for r in regs)==24814
 signatures={key(r['signature']) for r in regs}
 lookalikes=[r for r in led['observed_regimes'] if key(r['signature']) in signatures and r not in regs]
 assert len(lookalikes)==5 and sum(r['executions'] for r in lookalikes)==264 and all(r['module_candidates']==['M18'] for r in lookalikes)
 regs+=lookalikes
 rows=[];roles=Counter();ms=set();native=Counter();B=1.6e12;P=376e12;Pi=752e12
 for r in regs:
  d=r['signature'];a=shape(d['inputShapes_text']);z=shape(d['outputShapes_text']);op=r['operator'];dt=d['inputDataTypes_text']
  if op=='TransposeBatchMatMul':
   M=a[0][0];K=4096;Nn=1024
   assert a==[[M,1,K],[1,K,Nn]] and z==[[M,1,Nn]] and dt=='DT_BF16;DT_BF16'
   role='WO_A_GROUP1';typ='BF16';q=2*(M*K+K*Nn+M*Nn);weight=2*K*Nn;cap=P
   geometry='perm_x1(1,0,2): [T,1,K]→[1,T,K], weight[1,K,R], perm_y(1,0,2);oneGEMM M=T,notT independentM1GEMMs. Singletongroup permutationscanbeviews subjecttostride support.'
  else:
   M,K=a[0];assert z[0][0]==M;Nn=z[0][1]
   if op=='QuantBatchMatmulV3':
    assert math.prod(a[1])==K*Nn and a[2]==[Nn] and a[3]==[M] and d['outputDataTypes_text']=='DT_BF16'
    typ='INT8_WITH_BF16_SCALE_FLOAT_ROWSCALE_BF16_OUT';q=M*K+K*Nn+2*Nn+4*M+2*M*Nn;weight=K*Nn;cap=Pi
    assert (K,Nn) in [(1280,32768),(1280,4096)]
    role='Q_B' if Nn==32768 else 'INDEX_QUERY_B'
    geometry='LogicalK,Nfrominput/outputandweightnumel;FRACTAL_NZaxesnotguessed. Biasabsentinobserveddescriptor;row/weightscales included.'
   else:
    assert a[1]==[Nn,K]
    if dt=='FLOAT;FLOAT':typ='FLOAT_MODE_UNBOUND';cap=None;factor=4
    else:assert dt=='DT_BF16;DT_BF16';typ='BF16';cap=P;factor=2
    q=factor*(M*K+K*Nn+M*Nn);weight=factor*K*Nn
    if (K,Nn)==(5120,1280):role='Q_A'
    elif (K,Nn)==(5120,512):role='DRAFT_SHARED_CONTEXT_KV' if r['module_candidates']==['M18'] else 'C2_KV_OR_GATE' if cap is None else 'C1_PROJECT' if r['module_candidates']==['M08'] else 'SWA_OR_C1_ROLE_AMBIGUOUS' if len(r['module_candidates'])>1 else 'SWA_KV'
    elif (K,Nn)==(5120,32):role='INDEX_WEIGHTS'
    elif (K,Nn)==(512,128):role='INDEX_KEY'
    elif (K,Nn)==(1024,5120):role='WO_B'
    else:raise AssertionError((M,K,Nn,r['regime_id']))
    geometry='Ordinaryrow-linear projection withrecordedweight[N,K];FLOATdescriptor/hf32eligibility alone doesnotprovefullFP32/HF32executionmode orcapacity.'
  f=2*M*K*Nn;roles[role]+=r['executions'];ms.add(M)
  rows.append(dict(regime_id=r['regime_id'],module_candidates=r['module_candidates'],role=role,operator=op,M=M,K=K,N=Nn,precision=typ,
   executions=r['executions'],origin=r['origin'],context=r['context'],signature=d,
   conventional_dense_ops_2perMAC=f,one_read_write_logical_bytes=q,weight_bytes=weight,
   logical_arithmetic_intensity_ops_per_byte=f/q,reference_compute_us=f/cap*1e6 if cap else None,
   reference_cold_io_us=q/B*1e6,conditional_reference_floor_us=max(f/cap,q/B)*1e6 if cap else None,
   reference_dominant_term=('COMPUTE' if f/cap>=q/B else 'COLD_VALUE_IO') if cap else 'FLOAT_COMPUTE_MODE_UNBOUND',
   measured_native_ns=r['measured_native_task_duration'],geometry=geometry,
   qualification='Densealgorithm/valueIO,publishedper-die376halfTF/752INT8TOPS/1.6TB/s only;notactualcachetraffic,typedphysicalpeak/MFU orendtoendgain. Tiling/partialbuffers/Vector/dequant/sync/biasnotpricedbeyondrecordedoperands.'))
 graph=read(CACHE/'analysis/native-execution-identity-v14/graph-blocks-PMU-and-MC2-v14.json');pending=defaultdict(list)
 for g in graph['graph_executions']:pending[g['database_index']].append(g)
 chains=[];times=defaultdict(list);c2pairs=[];shared=[];frames=[]
 for i in range(32):
  d=read(CACHE/f'analysis/native-execution-identity-v14/db-{i:02d}.json');ts={r['task_record_rowid']:r for r in d['TASK_records']};defs={r['globalTaskId']:r for r in d['COMPUTE_definitions']}
  for t in ts.values():
   if t['globalTaskId'] in defs and key(defs[t['globalTaskId']]) in signatures:native[defs[t['globalTaskId']]['opType_text']]+=1
  def op(t):return defs[t['globalTaskId']]['opType_text']
  def ins(t):return shape(defs[t['globalTaskId']]['inputShapes_text'])
  def out(t):return shape(defs[t['globalTaskId']]['outputShapes_text'])
  def dtype(t):return defs[t['globalTaskId']]['inputDataTypes_text']
  def comp(t):return dict(task_record_rowid=t['task_record_rowid'],op=op(t),startNs=t['startNs'],endNs=t['endNs'],duration_ns=t['endNs']-t['startNs'],streamId=t['streamId'])
  def one(a):assert len(a)==1,len(a);return a[0]
  for g in pending[i]:
   total=0
   for b in g['blocks']:
    tasks=sorted([ts[rid] for rid in b['task_record_rowids']],key=lambda t:(t['startNs'],t['task_record_rowid']))
    qa=one([t for t in tasks if op(t) in ['MatMulV2','MatMulV3'] and ins(t)==[[3,5120],[1280,5120]]])
    qb=one([t for t in tasks if op(t)=='QuantBatchMatmulV3' and out(t)==[[3,32768]]])
    qnorm=one([t for t in tasks if op(t)=='RmsNorm' and ins(t)==[[3,1280],[1280]]])
    dq=[t for t in tasks if op(t)=='DynamicQuant' and ins(t)==[[3,1280]]]
    kv_cands=[t for t in tasks if op(t) in ['MatMulV2','MatMulV3'] and ins(t)==[[24,5120],[512,5120]] and dtype(t)=='DT_BF16;DT_BF16']
    kv=one([t for t in kv_cands if t['endNs']<=qb['startNs']])
    knorm=one([t for t in tasks if op(t)=='RmsNorm' and ins(t)==[[24,512],[512]] and t['streamId']==kv['streamId']])
    oa=one([t for t in tasks if op(t)=='TransposeBatchMatMul']);ob=one([t for t in tasks if op(t) in ['MatMulV2','MatMulV3'] and ins(t)==[[24,1024],[5120,1024]]])
    rs=one([t for t in tasks if op(t)=='hcom_reduceScatter_'])
    assert qa['endNs']<=qnorm['startNs']<=dq[0]['startNs'] and dq[0]['endNs']<=qb['startNs']
    assert oa['endNs']<=ob['startNs'] and ob['endNs']<=rs['startNs']
    x=dict(database_index=i,graph_ordinal=g['graph_ordinal'],block_ordinal=b['block_ordinal'],root_bound=bool(g['prepare_root_keys']))
    stages={'Q_A':qa,'Q_NORM':qnorm,'Q_QUANT':dq[0],'Q_B':qb,'SWA_KV':kv,'KV_NORM':knorm,'WO_A':oa,'WO_B':ob,'REDUCE_SCATTER':rs}
    chains.append(dict(x,stages={k:comp(t) for k,t in stages.items()},KV_BF16_candidates=len(kv_cands),
     output_chain_span_ns=rs['endNs']-oa['startNs'],output_projection_only_span_ns=ob['endNs']-oa['startNs'],
     role_scope='ExacttypedTASKs andsourceorder/streamcontract;atC1source20SWAKVderivedbyend-beforeQB. Originalfunctional160ambiguousassignmentrecords retained,notrewrittenasphysicaltensoridentity.'))
    indexq=[t for t in tasks if op(t)=='QuantBatchMatmulV3' and out(t)==[[3,4096]]]
    assert len(dq)==1+len(indexq)
    if indexq:
     iq=one(indexq);assert dq[1]['endNs']<=iq['startNs']
     iw=one([t for t in tasks if op(t)=='MatMulV2' and ins(t)==[[3,5120],[32,5120]]])
     shared.append(dict(x,Q_quant=comp(dq[0]),index_quant=comp(dq[1]),Q_B=comp(qb),index_Q_B=comp(iq),index_weights=comp(iw),
      source_input_identity='Pinnedpreprocess passes thesameqr tensor toQ_B andindexer.project_query;DQmatchingNativeTypedRecords corroboratesbutdoesnotitselfproveactualcontents/equalityparameters.'))
     stages.update(INDEX_Q_B=iq,INDEX_WEIGHTS=iw)
    fp=[t for t in tasks if op(t) in ['MatMulV2','MatMulV3'] and ins(t)==[[24,5120],[512,5120]] and dtype(t)=='FLOAT;FLOAT']
    if fp:
     assert len(fp)==2 and fp[0]['endNs']<=fp[1]['startNs']
     c2pairs.append(dict(x,first_source_WKV=comp(fp[0]),second_source_WGATE=comp(fp[1]),
      scope='Sourceexecuteswkvthenwgateononereusedhidden_states_fp32. Rolelabels source-derived;noactualweight/binarygenerationsinvented.'))
    for t in tasks:
     if key(defs[t['globalTaskId']]) in signatures:total+=t['endNs']-t['startNs']
    if x['root_bound']:
     for name,t in stages.items():times[name].append(t['endNs']-t['startNs'])
   frames.append(dict(database_index=i,graph_ordinal=g['graph_ordinal'],root_bound=bool(g['prepare_root_keys']),selected_projection_task_sum_ns=total,
    graph_device_control_span_ns=g['graph_control_pair_ns'],exposure=total/g['graph_control_pair_ns']))
 expected=Counter()
 for r in regs:expected[r['operator']]+=r['executions']
 assert native==expected,(native,expected)
 assert len(chains)==3200 and len(shared)==640 and len(c2pairs)==240
 write(O/'projection-work-and-reference-bounds-v61.json',dict(generated_utc=stamp,regimes=rows,regime_count=74,captured_execution_count=25078,
  core_projection_regimes=69,core_projection_executions=24814,other_role_lookalike_regimes=5,other_role_lookalike_executions=264,
  native_execution_counts=dict(native),source_roles_executions=dict(roles),logical_M_domains=sorted(ms),
  scope='69capturedregimes/24814exec attributedtoM07/M08/M09/M11,including160M07|M08ambiguousrecords,plus5same-geometryM18shared-contextregimes/264exec keptasdistinctroles. Densecostmodeledforlookalikes,notnewM18consumerfusionproof. Othermodelmatmulrolesretainprior/separateanalysis.',
  cold_reference_not_actual_runtime_lower_bound_without_traffic_capacity_assumptions=True,full_numeric_FLOAT_capacity_unbound=True,overall_complete=False))
 write(O/'projection-graph-chains-v61.json',dict(generated_utc=stamp,graph_projection_chains=chains,QR_shared_quant_candidates=shared,C2_FP32_projection_pairs=c2pairs,graph_frames=frames,
  root_bound_stage_ns={k:stat(v) for k,v in times.items()},root_selected_projection_exposure=stat([r['exposure'] for r in frames if r['root_bound']]),
  all_graph_bmm_group_count=1,graph_Q_rows=3,graph_KV_and_output_rows=24,
  limits=['64root/16unboundgraphs retained;timingsumsandgapsnotactualcriticalpath/gain.',
   'Source-derivedroleandmathematicalworkdo notattestactualtensoralias/generation/dtypeprecisionpolicy/weightcopies.',
   'QAlocalrows3versusKV/outputglobal24excludesnaivesamerowconcatenation. DirecttargetM830/834versusglobal6668andpadded6672domainsmustremainseparate.'],overall_complete=False))
 print(json.dumps(dict(regimes=74,executions=25078,core_regimes=69,lookalike_shared_context_regimes=5,native=dict(native),chains=len(chains),shared_QR_quant_candidates=len(shared),C2pairs=len(c2pairs),root_stage_ns={k:stat(v) for k,v in times.items()},overall_complete=False)))

if __name__=='__main__':main()
