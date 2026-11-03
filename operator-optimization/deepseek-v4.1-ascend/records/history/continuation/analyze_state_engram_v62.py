"""Exact functional populations and operator-level work/IO; no device execution."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import math
from finalize_origin_coverage_v9 import N,O,read,write
from mirror_saved_profile_core_v53 import CACHE
from analyze_projection_ceiling_v61 import shape,stat,KEYS,key

def fullkey(mod,op,role,origin,context,d):return (tuple(mod),op,role,origin,context,key(d))
DT={'DT_BF16':2,'FLOAT':4,'INT32':4,'INT64':8,'BOOL':1,'DT_UINT1':1}
def dimensions(s):
 if s=='N/A':return None
 return shape(s)
def bytesparts(shapes,dtypes):
 if shapes is None:return None
 return [math.prod(s)*DT[t] if s is not None and t in DT else None for s,t in zip(shapes,dtypes.split(';'))]
def main():
 assert not (O/'state-engram-regime-models-v62.json').exists()
 regs=[r for r in read(O/'operator-shape-ceiling-ledger-v55.json')['observed_regimes'] if 'M08' in r['module_candidates'] or r['module_candidates']==['M14']]
 assert len(regs)==276 and sum(r['executions'] for r in regs)==13704
 lk={fullkey(r['module_candidates'],r['operator'],r['role'],r['origin'],r['context'],r['signature']):r for r in regs};assert len(lk)==len(regs)
 assignments=read(CACHE/'analysis/native-PMU-complete-v18/resource-functional-attribution-and-uncertainty-v20-v3.json')['execution_assignments']
 selected=defaultdict(dict)
 for a in assignments:
  if 'M08' in a[3] or a[3]==['M14']:selected[a[0]][a[1]]=a
 assert sum(map(len,selected.values()))==13704
 graphs=defaultdict(list)
 for g in read(CACHE/'analysis/native-execution-identity-v14/graph-blocks-PMU-and-MC2-v14.json')['graph_executions']:graphs[g['database_index']].append(g)
 times=defaultdict(list);rids=defaultdict(list);chains=[];bodies=[];stage_times=defaultdict(list);native=Counter()
 for db in range(32):
  d=read(CACHE/f'analysis/native-execution-identity-v14/db-{db:02d}.json');tasks={t['task_record_rowid']:t for t in d['TASK_records']};defs={t['globalTaskId']:t for t in d['COMPUTE_definitions']}
  for tid,a in selected[db].items():
   t=tasks[tid];de=defs[t['globalTaskId']];r=lk[fullkey(a[3],de['opType_text'],a[5],a[2],a[6],de)]
   rid=r['regime_id'];times[rid].append(t['endNs']-t['startNs']);rids[rid].append([db,tid]);native[(tuple(a[3]),de['opType_text'])]+=1
  def op(t):return defs[t['globalTaskId']]['opType_text']
  def ins(t):return dimensions(defs[t['globalTaskId']]['inputShapes_text'])
  def comp(t):return dict(TASK_rowid=t['task_record_rowid'],operator=op(t),streamId=t['streamId'],startNs=t['startNs'],endNs=t['endNs'],duration_ns=t['endNs']-t['startNs'])
  def one(xs):assert len(xs)==1,len(xs);return xs[0]
  for g in graphs[db]:
   root=bool(g['prepare_root_keys']);base=dict(database_index=db,graph_ordinal=g['graph_ordinal'],root_bound=root)
   for b in g['blocks']:
    ts=sorted([tasks[i] for i in b['task_record_rowids']],key=lambda t:(t['startNs'],t['task_record_rowid']))
    pools=[t for t in ts if op(t)=='_pool_kernel']
    if pools:
     p=one(pools);u=one([t for t in ts if op(t)=='_cache_update_kernel']);norm=one([t for t in ts if op(t)=='RmsNorm' and ins(t)==[[24,512],[512]] and t['startNs']>=u['endNs']])
     z=one([t for t in ts if op(t)=='ZerosLike' and ins(t)==[[24,512]]])
     assert z['endNs']<=p['startNs']<=u['startNs'] and u['endNs']<=norm['startNs']
     assert p['streamId']==u['streamId']==norm['streamId']
     x=dict(base,block_ordinal=b['block_ordinal'],M_descriptor=24,zero=comp(z),pool=comp(p),update=comp(u),norm=comp(norm),pool_to_norm_end_span_ns=norm['endNs']-p['startNs'],update_is_not_norm_data_dependency='Norm reads pooled BF16, not updated ring; update must follow pool old-ring reads, other resources/lifetimes retained.')
     chains.append(x)
     if root:
      for k,t in [('C2_ZERO',z),('C2_POOL',p),('C2_UPDATE',u),('C2_NORM',norm)]:stage_times[k].append(t['endNs']-t['startNs'])
   outside=[x['task'] for x in g['outside_marker_tasks']]
   for layer in [1,14]:
    ts=sorted([t for t in outside if g['blocks'][layer-1]['endNs']<=t['startNs'] and t['endNs']<=g['blocks'][layer]['startNs'] and t['task_record_rowid'] in selected[db] and selected[db][t['task_record_rowid']][3]==['M14']],key=lambda t:t['startNs'])
    assert len(ts)==38,(db,g['graph_ordinal'],layer,len(ts))
    w=one([t for t in ts if op(t)=='MatMulV2' and ins(t)==[[3,6144],[25600,6144]]]);rot=one([t for t in ts if op(t)=='MatMulV2' and ins(t)==[[1920,32],[32,32]]])
    assert w['endNs']<=rot['startNs']
    param=[t for t in ts if (op(t)=='Cast' and ins(t)==[[4,5120]]) or (op(t)=='Mul' and ins(t)==[[4,5120],[4,5120]])];assert len(param)==3
    rotationcast=one([t for t in ts if op(t)=='Cast' and ins(t)==[[32,32]]])
    x=dict(base,layer=layer,stages=[comp(t) for t in ts],WKV=comp(w),rotation=comp(rot),parameter_prep_TASKs=[comp(t) for t in param],rotation_cast=comp(rotationcast),operator_count=38,selected_task_sum_ns=sum(t['endNs']-t['startNs'] for t in ts),body_span_ns=ts[-1]['endNs']-ts[0]['startNs'])
    bodies.append(x)
    if root:
     for k,t in [('ENGRAM_WKV',w),('ENGRAM_ROTATION',rot)]:stage_times[k].append(t['endNs']-t['startNs'])
     stage_times['ENGRAM_PARAMETER_PREP_SUM'].append(sum(t['endNs']-t['startNs'] for t in param))
     stage_times['ENGRAM_OTHER_GATING_SUM'].append(x['selected_task_sum_ns']-(w['endNs']-w['startNs'])-(rot['endNs']-rot['startNs'])-sum(t['endNs']-t['startNs'] for t in param))
 assert len(chains)==240 and Counter(x['block_ordinal'] for x in chains)=={2:80,8:80,14:80}
 assert len(bodies)==160 and sum(x['root_bound'] for x in bodies)==128
 projection={r['regime_id']:r for r in read(O/'projection-work-and-reference-bounds-v61.json')['regimes']}
 rows=[]
 for r in regs:
  rid=r['regime_id'];assert len(times[rid])==r['executions'],(rid,len(times[rid]),r['executions'])
  de=r['signature'];a=dimensions(de['inputShapes_text']);z=dimensions(de['outputShapes_text']);ib=bytesparts(a,de['inputDataTypes_text']);ob=bytesparts(z,de['outputDataTypes_text']);outn=math.prod(z[0]) if z and z[0] is not None else None
  op=r['operator'];model={};io=None
  if op in ['MatMulV2','MatMulV3']:
   if rid in projection:model=dict(reused_projection_model='projection-work-and-reference-bounds-v61.json#'+rid,**{k:projection[rid][k] for k in ['M','K','N','conventional_dense_ops_2perMAC','weight_bytes','one_read_write_logical_bytes','precision','reference_compute_us','reference_cold_io_us','conditional_reference_floor_us']});io=model['one_read_write_logical_bytes']
   else:
    m,k=a[0];nn=z[0][1];assert a[1]==[nn,k];assert (k,nn) in [(6144,25600),(32,32)]
    half=de['inputDataTypes_text']=='DT_BF16;DT_BF16';factor=2 if half else 4;f=2*m*k*nn;io=factor*(m*k+k*nn+m*nn)
    model=dict(M=m,K=k,N=nn,source_role='WKV_COMBINED_KEY_AND_VALUE' if half else 'GATE_BLOCK32_ROTATION',conventional_dense_ops_2perMAC=f,weight_bytes=factor*k*nn,one_read_write_logical_bytes=io,precision='BF16' if half else 'FLOAT_MODE_UNBOUND',reference_compute_us=f/376e12*1e6 if half else None,reference_cold_io_us=io/1.6e12*1e6,conditional_reference_floor_us=max(f/376e12,io/1.6e12)*1e6 if half else None)
  elif op=='ZerosLike':io=ob[0];model=dict(value_stores=outn,value_loads=0,source_gate='Lookuplocalheadzeros176 havefullwriteelision. C2zerosdefineincomplete/unwrittenrows. Dummyglobalbank/masktailzeros remainlive; do not blanketremove.')
  elif op in ['Transpose','TensorMove','Slice']:
   io=2*ob[0];model=dict(selected_payload_bytes=ob[0],copy_value_IO_assumption='One selected payload read+write ifmaterialized; Slice doesnotreadfullparentextent, metadata/pointer ignored. Alias/viewelisionrequiresallreaders,lifetime,stride andmutationproof.',fusion='LookupTPtranspose→bankwrite→SPclone can be conditionaldirectrowlayout, withfullglobalbankstillpublishedforallreaders. Gatefinalcopy can bedirectstoreunderalias/externalconsumercontract.')
  elif op=='Cast':io=ib[0]+ob[0];model=dict(cast_elements=outn,unknown_scalar_GM_loads=False,contract='Staticparametercasts version-cacheable; residualhidden/key/valuecasts canbeabsorbedindependently while preservingsource BF16value andFP32arithmetic.')
  elif op in ['ReduceMean','ReduceSum']:
   nin=math.prod(a[0]);model=dict(reduction_adds=nin-outn,mean_scalar_divisions=outn if op=='ReduceMean' else 0,reduction_order_not_freely_reassociated=True);io=ib[0]+ob[0]
  elif op=='RmsNorm':
   m,h=a[0];io=2*m*h+2*h+2*m*h+4*m;model=dict(rows=m,H=h,squares=m*h,reduction_adds=m*(h-1),ordinary_scalar_arithmetic=2*m,rsqrt=m,output_multiplies=2*m*h,BF16_input_and_output_rounding_preserved=True,auxiliary_rstd_bytes=4*m,conditional_fusion='C2pool→norm→64tailRoPE→cache can be sourceconditioned; rstdreaders/rowmask/pad/slotchecksandKVfork remainexplicit.')
  elif op=='InplacePartialRotaryMul':
   m=a[0][0];io=m*(2*64*2+2*64*4);model=dict(rows=m,rotated_tail=64,unchanged_prefix=448,ordinary_multiplies=2*m*64,ordinary_adds=m*64,contract='BF16RMSoutputroundretained beforeFP32cos/sinrotate andBF16output; tail-onlylogicalvalueIO,not512wideinputloadmandate.')
  elif op=='ScatterNdUpdateSk':
   m,h=a[2];io=None;model=dict(descriptor_update_rows=m,H=h,all_valid_reference_payload_bytes=m*(4*h+8),all_valid_update_write_bytes=2*m*h,actual_valid_rows_unbound=True,cache_extent_bytes=math.prod(a[0])*2,contract='ReadselectedBF16updatesandINT32rowpairs,writeselectedcache. Descriptorwholecacheextent isallocationnotpercalltraffic. Fusionwithnorm/RoPE conditionalonactualslot/validprefixandfuturecacheconsumers.')
  elif op in ['_pool_kernel','_cache_update_kernel']:
   model=dict(source_parameterized_model='C2-state-and-pool-costs-v62.json',recorded_T=(a[0][0] if a else 24),actual_request_segments_and_valid_pages_unbound=True)
  elif op.startswith('_engram_') or op in ['_write_hash_cache_kernel','_hash_ids_kernel']:
   model=dict(source_parameterized_model='Engram-hash-UVA-and-gate-costs-v62.json',specific_kernel=op,actual_hash_IDs_ownership_formats_and_hit_ratio_unbound=True)
  else:
   assert op in ['Mul','Pows','Add','Rsqrt','Abs','ClipByValueV2','Sqrt','Less','Neg','SelectV2','Sigmoid','LogicalNot','MaskedFill','Equal','LogicalOr'],op
   model=dict(output_elements=outn,pointwise_semantic_operations=outn,unit='sourcepointwiseoperation;notallordinaryFLOPs',special_or_compare=op in ['Rsqrt','Sqrt','Sigmoid','Less','SelectV2','LogicalNot','MaskedFill','Equal','LogicalOr','Abs','ClipByValueV2'],qualification='Pows exponent2 identified bysource; source signbit mustnotbereplacedbyLesslabelalone. Broadcast inputs countedonce logically,notperoutput-load. Scalar/emptydescriptor operandsnotassumedGMloads.')
   if op=='Pows':model['ordinary_square_multiplies']=outn
   io=sum(x for x in ib if x is not None)+sum(x for x in ob if x is not None)
  rows.append(dict(regime_id=rid,module_candidates=r['module_candidates'],operator=op,role=r['role'],origin=r['origin'],context=r['context'],executions=r['executions'],signature=de,native_TASK_keys=rids[rid],measured_native_task_ns=stat(times[rid]),work_model=model,one_read_write_logical_value_bytes=io,known_shaped_input_extents_bytes=ib,known_shaped_output_extents_bytes=ob,empty_or_scalar_descriptors_not_interpreted=bool(a and any(s is None for s in a)),source_role_ambiguity_retained=len(r['module_candidates'])>1,physical_numeric_ceiling_proved=False,consumer_fusion_review_complete_in_this_turn=True))
 assert len(rows)==276 and sum(r['executions'] for r in rows)==13704
 write(O/'state-engram-regime-models-v62.json',dict(generated_utc=datetime.now(timezone.utc).isoformat(),regimes=rows,regime_count=len(rows),captured_execution_count=13704,M08_member_regimes=56,M08_member_executions=4024,M14_strict_regimes=220,M14_strict_executions=9680,ambiguous_M07_M08_regimes=8,ambiguous_M07_M08_executions=1600,exact_population_join='functionalmodule+op+role+origin+context+sevenComputeInfofields; nativeTASKkeycountconserved. Shapeonlyjoinforbidden.',reference_scope='Workcounts/logicalvalueIO;published376BF16TF/1.6TBonlyconditionalcoldreference. FLOAT/Vector/INT64remainder/UVAtypedcapacitiesunbound;noactualHBMtrafficorcriticalpathgain.',BOOL_storage_assumption='Source torch.bool bank1byte; DT_UINT1 treatedasonebyteonlyunderthissource-storageassumption,notpackedeightbits.',whole_modules_closed=0,overall_complete=False))
 write(O/'state-engram-graph-operator-chains-v62.json',dict(C2_chains=chains,Engram_bodies=bodies,C2_root_bound_chains=sum(x['root_bound'] for x in chains),Engram_root_bound_bodies=128,root_stage_ns={k:stat(v) for k,v in stage_times.items()},scope='240C2chainsand160Engram38TASKbodiesstructurallyboundto80graphblocks,64root/16unbound. Directoperatorrowsmodeledwithoutinventingunavailablebodyrootjoins. Ancestryv30total9712versusfunctional9680keptdistinct;noforcedsamepopulation.',overall_complete=False))
 print(dict(regimes=len(rows),executions=13704,C2_chains=len(chains),Engram_bodies=len(bodies),root_stage_ns={k:stat(v) for k,v in stage_times.items()}))

if __name__=='__main__':main()
