"""Functional native populations, IndexKey chains and draft context work/selected traffic."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import math
from finalize_origin_coverage_v9 import N,O,read,write
from mirror_saved_profile_core_v53 import CACHE
from analyze_state_engram_v62 import fullkey,dimensions
from analyze_projection_ceiling_v61 import stat
DT={'DT_BF16':2,'FLOAT':4,'FLOAT16':2,'INT32':4,'INT64':8,'INT8':1,'BOOL':1}
def valuebytes(sh,dt):return [None if x is None or t not in DT else math.prod(x)*DT[t] for x,t in zip(sh,dt.split(';'))] if sh else None
def main():
 assert not (O/'index-context-regime-models-v64.json').exists()
 old=read(O/'operator-analysis-coverage-ledger-v63.json');cv={r['regime_id']:r for r in old['regimes']};universe=read(O/'operator-shape-ceiling-ledger-v55.json')['observed_regimes']
 regs=[r for r in universe if r['module_candidates']==['M18'] or (r['module_candidates']==['M09'] and cv[r['regime_id']]['state'].startswith('REGIME_'))]
 assert len(regs)==242 and sum(r['executions'] for r in regs)==15976
 lk={fullkey(r['module_candidates'],r['operator'],r['role'],r['origin'],r['context'],r['signature']):r for r in regs};selected=defaultdict(dict)
 for a in read(CACHE/'analysis/native-PMU-complete-v18/resource-functional-attribution-and-uncertainty-v20-v3.json')['execution_assignments']:
  if a[3] in [['M09'],['M18']]:selected[a[0]][a[1]]=a
 graphs=defaultdict(list)
 for g in read(CACHE/'analysis/native-execution-identity-v14/graph-blocks-PMU-and-MC2-v14.json')['graph_executions']:graphs[g['database_index']].append(g)
 times=defaultdict(list);ids=defaultdict(list);miss=Counter();keychains=[];aux=[];KVcycles=[];mainchains=[];stages=defaultdict(list)
 expected_cycle=read(O/'M18-current-aux-context-and-publication-profile-v34.json')['root_and_source_order_audit']['cycle_type_hist'][0]['types']
 for db in range(32):
  d=read(CACHE/f'analysis/native-execution-identity-v14/db-{db:02d}.json');ts={t['task_record_rowid']:t for t in d['TASK_records']};defs={r['globalTaskId']:r for r in d['COMPUTE_definitions']}
  for tid,a in selected[db].items():
   t=ts[tid];de=defs[t['globalTaskId']];k=fullkey(a[3],de['opType_text'],a[5],a[2],a[6],de)
   if k in lk:
    rid=lk[k]['regime_id'];times[rid].append(t['endNs']-t['startNs']);ids[rid].append([db,tid]);miss[rid]+=not a[11]
  def de(t):return defs[t['globalTaskId']]
  def op(t):return de(t)['opType_text']
  def ins(t):return dimensions(de(t)['inputShapes_text'])
  def outs(t):return dimensions(de(t)['outputShapes_text'])
  def comp(t):return dict(TASK_rowid=t['task_record_rowid'],operator=op(t),streamId=t['streamId'],startNs=t['startNs'],endNs=t['endNs'],duration_ns=t['endNs']-t['startNs'])
  def one(a):assert len(a)==1,len(a);return a[0]
  for g in graphs[db]:
   root=bool(g['prepare_root_keys']);base=dict(database_index=db,graph_ordinal=g['graph_ordinal'],root_bound=root)
   for b in g['blocks']:
    tasks=sorted([ts[i] for i in b['task_record_rowids']],key=lambda t:t['startNs']);wk=[t for t in tasks if op(t)=='MatMulV2' and ins(t)==[[24,512],[128,512]]]
    if not wk:continue
    w=one(wk);n=one([t for t in tasks if op(t)=='RmsNorm' and ins(t)==[[24,128],[128]]]);r=one([t for t in tasks if op(t)=='InplacePartialRotaryMul' and ins(t)==[[24,1,1,128],[24,1,1,64],[24,1,1,64]]]);q=one([t for t in tasks if op(t)=='DynamicQuant' and ins(t)==[[24,128]]]);c=one([t for t in tasks if op(t)=='Cast' and ins(t)==[[24,1]] and de(t)['outputDataTypes_text']=='FLOAT16'])
    ks=one([t for t in tasks if op(t)=='ScatterNdUpdateSk' and de(t)['inputDataTypes_text']=='INT8;INT32;INT8']);ss=one([t for t in tasks if op(t)=='ScatterNdUpdateSk' and de(t)['inputDataTypes_text']=='FLOAT16;INT32;FLOAT16'])
    assert w['endNs']<=n['startNs'] and n['endNs']<=r['startNs'] and r['endNs']<=q['startNs'] and q['endNs']<=ks['startNs'] and q['endNs']<=c['startNs']<=ss['startNs']
    keychains.append(dict(base,block_ordinal=b['block_ordinal'],stages={k:comp(t) for k,t in dict(WK=w,RMS=n,RoPE=r,DynamicQuant=q,KeyScatter=ks,ScaleFP16=c,ScaleScatter=ss).items()},scope='Sourcetypedwork/order;normalKquantFLOATscaleBEFOREFP16cachecast,allmetadata/alias/validrowconditionsretained.'))
    if root:
     for k,t in [('WK',w),('KEY_RMS',n),('KEY_ROPE',r),('KEY_DQ_FLOAT_SCALE',q),('KEY_SCATTER',ks),('KEY_SCALE_CAST',c),('KEY_SCALE_SCATTER',ss)]:stages[k].append(t['endNs']-t['startNs'])
   outside=[t['task'] for t in g['outside_marker_tasks']]
   for block in [37,38,39]:
    win=[t for t in outside if g['blocks'][block-1]['endNs']<=t['startNs'] and t['endNs']<=g['blocks'][block]['startNs']]
    mean=one([t for t in win if op(t)=='ReduceMean' and ins(t)==[[3,4,5120],[1]]]);collective=one([t for t in win if op(t)=='hcom_allGather_']);assert mean['endNs']<=collective['startNs']
    aux.append(dict(base,checkpoint_before_block=block,mean=comp(mean),allgather=comp(collective),scope='FunctionalM18meanpluslinkedM23collective;do notpromoteM23regimeshere.'))
    if root:stages['AUX_MEAN'].append(mean['endNs']-mean['startNs']);stages['AUX_ALLGATHER'].append(collective['endNs']-collective['startNs'])
  mt=sorted([ts[i] for i,a in selected[db].items() if a[3]==['M18']],key=lambda t:t['startNs'])
  for anchor in [t for t in mt if op(t)=='MatMulV2' and ins(t)[0][1]==5120]:
   m=ins(anchor)[0][0];end=one([t for t in mt if op(t)=='ScatterNdUpdateSk' and t['startNs']>=anchor['endNs']][:1]);group=[t for t in mt if anchor['startNs']<=t['startNs'] and t['endNs']<=end['endNs']]
   assert len(group)==30 and Counter(op(t) for t in group)==expected_cycle,(db,m,len(group),Counter(op(t) for t in group))
   assert ins(end)[2]==[m,512]
   KVcycles.append(dict(database_index=db,rows=m,source_context=selected[db][anchor['task_record_rowid']][6],stages=[comp(t) for t in group],scope='Exactsource-ordered30TASKcycle,nottensor/requestepochidentity.'))
  for anchor in [t for t in mt if op(t)=='MatMulV2' and ins(t)[0][1]==15360]:
   m=ins(anchor)[0][0];tr=one([t for t in mt if op(t)=='Transpose' and ins(t)==[[8,m,640],[3]] and t['startNs']>=anchor['endNs']][:1]);norm=one([t for t in mt if op(t)=='RmsNorm' and ins(t)==[[m,5120],[5120]] and t['startNs']>=tr['endNs']][:1])
   mainchains.append(dict(database_index=db,M=m,GEMM=comp(anchor),gather_transpose=comp(tr),full5120_norm=comp(norm),scope='88source-orderedmainprojectionpipelines;fullfeaturegatherlayout,not640-widenorm.'))
 assert len(keychains)==320 and Counter(x['block_ordinal'] for x in keychains)=={2:80,8:80,14:80,20:80}
 assert len(aux)==240 and len(KVcycles)==312 and len(mainchains)==88
 rows=[]
 for r in regs:
  rid=r['regime_id'];assert len(times[rid])==r['executions'] and miss[rid]==r['missing_PMU'],rid
  de=r['signature'];op=r['operator'];a=dimensions(de['inputShapes_text']);z=dimensions(de['outputShapes_text']);ib=valuebytes(a,de['inputDataTypes_text']);ob=valuebytes(z,de['outputDataTypes_text']);outn=math.prod(z[0]) if z and z[0] is not None else None;io=None;model={};mid=r['module_candidates'][0]
  if op=='MatMulV2':
   m,k=a[0];n=z[0][1];assert a[1]==[n,k];assert (k,n) in [(15360,640),(5120,512)];f=2*m*k*n;io=2*(m*k+k*n+m*n)
   model=dict(M=m,K=k,N=n,dense_ops_2perMAC=f,BF16_weight_bytes=2*k*n,reference_compute_us=f/376e12*1e6,cold_value_IO_reference_us=io/1.6e12*1e6,conditional_reference_floor_us=max(f/376e12,io/1.6e12)*1e6,role='MAIN_COLUMN640_CONTEXT_COMBINE' if k==15360 else 'DRAFT_SHARED_CONTEXT_KV_NOT_SWA',qualification='PublishedBF16referenceonly;actualHBM/cache/nativecapacity/accumulationnotattested. F27/F29specificconsumercontractsnowattachedtothe5oldgeometry-onlyregimes.')
  elif op=='RmsNorm':
   m,h=a[0];io=4*m*h+2*h+4*m;model=dict(M=m,H=h,ordinary_ops=4*m*h+m,rsqrt=m,aux_rstd_bytes=4*m,contract='BF16input/FP32norm-policy/BF16outputboundary;mainfull5120after8shards,KV512andIndexKey128distinctdomains. Preservegather/layout/otherreaders/auxrstdconditions.')
  elif op=='DynamicQuant':
   m,h=a[0];io=3*m*h+4*m;model=dict(M=m,H=h,INT8_values=m*h,FLOAT_scales=m,reference_abs=m*h,reference_max_compares=m*(h-1),quantize_elements=m*h,role='INDEX_KEY_FLOAT_SCALE_QUANT_BEFORE_FP16_CACHE_SCALE' if h==128 else 'INDEX_QUERY_INPUT_1280_FLOAT_SCALE_QUANT',contract='ScaleFLOATusedforquant,thenFP16cachecastforkeys;notcustomQuery128-headFP16-beforequant. Preserveactualnativeamax/zero/nonfinite/round/saturatepolicy.')
  elif op=='InplacePartialRotaryMul':
   m=a[0][0];heads=math.prod(a[0][1:-1]);n=m*heads*64;io=4*n+8*m*64;model=dict(M=m,heads=heads,D=128,tail=64,ordinary_muls=2*n,ordinary_adds=n,sign_flips=n//2,BF16_RINT=n,contract='IndexQuery32headsorIndexKey1head,FLOATcos/sinbroadcast,currentv63ABAcontract. Norm/projectedBF16barriersretained.')
  elif op=='ScatterNdUpdateSk':
   m,h=a[2];dt=de['inputDataTypes_text'].split(';')[0];size=DT[dt];model=dict(M_descriptor=m,H=h,element_bytes=size,all_valid_reference_read_indices_write_bytes=m*(2*h*size+8),actual_valid_rows_unbound=True,cache_descriptor_bytes=math.prod(a[0])*size,contract='Selectedrowsandpairs,notwholecache. IndexINT8key+FP16scaleplanespreservepublication/duplicate-slotsemantics;M18BF16KVfollowsper-layergroup/slotgeneration.')
  elif op in ['Cast','Mul']:
   if outn is not None:
    io=sum(x for x in ib if x is not None)+sum(x for x in ob if x is not None)
   model=dict(elements=outn,contract='M09weightsBF16→FP32times2^-6→FP16;keyscaleFLOAT→FP16AFTERINT8quant. M18C/SFLOAT→BF16nativecastsBEFORERotary,and3INTslotcastsretainintegerwidth. Empty/scalaroperandsnotassumedextraGMloads.')
  elif op=='Index':
   assert a[0]==[1048576,1,1,64];m=z[0][0];io=8*m*64+4*m;model=dict(rows=m,row_width=64,table_allocation_bytes=4*1048576*64,selected_table_read_bytes=4*m*64,FP32_output_bytes=4*m*64,INT32_index_read_bytes=4*m,contract='Onlyselectedtableentries,negative-indexwrap/originalbounds/errorpolicyretained. 4Index/cycle=2C/SconfigpairsunderTensor/defaultposmap,tablepointers/configidentityunrecorded;filter/unionhoistconditional.')
  elif op=='IndexCheck':
   m=a[1][0];model=dict(index_elements=m,index_dtype='INT32',table_rows=1048576,reference_range_compare_count=2*m,negative_wrap_elements_at_most=m,reference_index_read_bytes=4*m,output_bytes=None,contract='Bounds/checksideeffectoforiginaladvancedindex;noN/Aoutputpayloadguess. Nativecheck/reduction/errorflag/syncunpriced;fusionmustretainnegativeindexwrappinganderrorbehaviorordomainproof,notdeletebydtype.')
  elif op in ['Slice','Transpose','TensorMove','ConcatD']:
   size=DT[de['outputDataTypes_text'].split(';')[0]];io=2*size*outn;model=dict(selected_output_elements=outn,materialized_read_write_bytes=io,contract='Selectedpayloadonly.Sliceavoidfullparentextent;mainTranspose[8,M,640]requiresstrideawarefull5120norm. M09candidateTensorMovefreshmode1output→persistentbankneedsoutABI,lifetimes/allreaders;notprepare-indicesalreadyaliasedbank.')
  elif op=='ViewCopy':
   n=math.prod(a[4]);io=4*n;model=dict(actual_source_tail_BF16_elements=n,base_storage_descriptor_elements=a[0][0],selected_read_write_bytes=io,contract='Rotaryresultcopybackintooriginal64tailview;descriptorwholeM512storageNOTcopyextent. Tailprivacy/originalmutationobserversmatter;avoidclaimwhole512copy.')
  elif op=='RotaryPositionEmbedding':
   n=math.prod(z[0]);io=8*n;model=dict(BF16_tail_elements=n,ordinary_muls=2*n,ordinary_adds=n,sign_elements=n//2,cos_sin_already_BF16=True,internal_native_arithmetic_policy_unbound=True,contract='M18npu_rotary_mulloweredBF16cos/sin;do notsubstituteM09FLOATcosABAwithoutkeepingC/Sroundandnativearithmetic. OutputBF16andoriginaltailViewCopymutationremain.')
  elif op=='ReduceMean':
   m,hc,h=a[0];assert hc==4;io=10*m*h;model=dict(M=m,HC=hc,H=h,reduce_adds=3*m*h,divide_or_scale_elements=m*h,BF16_output_elements=m*h,contract='TargetHCmeanBEFORE3checkpointgathers/mainconcatGEMM;keepBF16meanround. Movingrowselect/gatherrequiresselectoravailabilityandallauxreaders;crosslayerFP32partialnotBF16partialsum.')
  elif op=='BroadcastTo':io=DT[de['outputDataTypes_text']]*outn;model=dict(stores=outn,scalar_GM_load_unpriced=True,contract='Materializedblocksizebroadcastforslotmodcanfoldintoscalaraddressmath;blocksizevalue/generationandintwidthretained.')
  else:
   assert op in ['GreaterEqual','ClipByValueV2','FloorDiv','FloorMod','Pack','LogicalNot','MaskedFill'],op
   io=sum(x for x in ib if x is not None)+sum(x for x in ob if x is not None);model=dict(output_elements=outn,reference_pointwise_or_pack_elements=outn,contract='Slotsallnegative→invalidpair(-1,-1),positiveclamp/floorDiv/modwithwidenedintermediatesandINT32pairs. NottokenID-1-onlysanitizer. Preserveper-layergroup/B/slotepoch/duplicatewriteorder;metadataCSEnewkernelconditions.')
  rows.append(dict(regime_id=rid,module_candidates=r['module_candidates'],operator=op,role=r['role'],origin=r['origin'],context=r['context'],executions=r['executions'],signature=de,native_TASK_keys=ids[rid],missing_PMU=miss[rid],measured_native_task_ns=stat(times[rid]),work_model=model,one_read_write_logical_value_bytes=io,previous_coverage_state=cv[rid]['state'],consumer_fusion_review_complete_in_this_turn=True,physical_numeric_ceiling_proved=False))
 write(O/'index-context-regime-models-v64.json',dict(generated_utc=datetime.now(timezone.utc).isoformat(),regimes=rows,regime_count=242,captured_execution_count=15976,M09_regimes=72,M09_executions=6040,M18_regimes=170,M18_executions=9936,M18_previous_geometry_only_regimes=5,M18_previous_geometry_only_executions=264,scope='72M09pending+all170M18strictfunctionalregimes;v34ancestry10760+separatefirstpass480isadifferentpopulation,notforcedinto9936. Dense/layer-specific/numeric/logicalIOmodelsonly;allactualphysicalceilingsremainconditional.',overall_complete=False))
 write(O/'index-context-native-chains-v64.json',dict(IndexKey_graph_chains=keychains,IndexKey_root_bound_chains=sum(x['root_bound'] for x in keychains),aux_checkpoint_graph_chains=aux,aux_root_bound_chains=sum(x['root_bound'] for x in aux),context_KV_source_order_cycles=KVcycles,main_projection_source_order_chains=mainchains,root_stage_ns={k:stat(v) for k,v in stages.items()},limits='Nativekeys/ordernotactualtensor/requestepochproof. M23auxCOMMrecordslinkedbutnotpromotedhere;all16unboundgraphsretained.',overall_complete=False))
 print(dict(regimes=242,executions=15976,IndexKey_chains=len(keychains),aux_chains=len(aux),KV_cycles=len(KVcycles),main_chains=len(mainchains),root_stage_ns={k:stat(v) for k,v in stages.items()}))

if __name__=='__main__':main()
