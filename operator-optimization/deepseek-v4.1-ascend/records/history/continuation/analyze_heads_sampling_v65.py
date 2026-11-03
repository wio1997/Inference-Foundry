"""Native functional head/sampling/state populations and source-ordered operator chains."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import math
from finalize_origin_coverage_v9 import N,O,read,write
from mirror_saved_profile_core_v53 import CACHE
from analyze_state_engram_v62 import fullkey,dimensions
from analyze_projection_ceiling_v61 import stat
DT={'DT_BF16':2,'FLOAT':4,'INT32':4,'INT64':8,'BOOL':1}
def vb(sh,dt):return [None if s is None or t not in DT else math.prod(s)*DT[t] for s,t in zip(sh,dt.split(';'))] if sh else None
def main():
 assert not (O/'heads-sampling-regime-models-v65.json').exists()
 cv={r['regime_id']:r for r in read(O/'operator-analysis-coverage-ledger-v64.json')['regimes']};mods=[['M15'],['M16'],['M17'],['M20'],['M24'],['M04','M16']]
 regs=[r for r in read(O/'operator-shape-ceiling-ledger-v55.json')['observed_regimes'] if r['module_candidates'] in mods and cv[r['regime_id']]['state'].startswith('REGIME_')]
 assert len(regs)==166 and sum(r['executions'] for r in regs)==9792
 lk={fullkey(r['module_candidates'],r['operator'],r['role'],r['origin'],r['context'],r['signature']):r for r in regs};selected=defaultdict(dict)
 for a in read(CACHE/'analysis/native-PMU-complete-v18/resource-functional-attribution-and-uncertainty-v20-v3.json')['execution_assignments']:
  if a[3] in mods:selected[a[0]][a[1]]=a
 times=defaultdict(list);ids=defaultdict(list);miss=Counter();tail=[];markov=[];rejections=[];states=[];head=[];stage_times=defaultdict(list)
 graph_root={(g['database_index'],g['graph_ordinal']):bool(g['prepare_root_keys']) for g in read(CACHE/'analysis/native-execution-identity-v14/graph-blocks-PMU-and-MC2-v14.json')['graph_executions']}
 for db in range(32):
  e=read(CACHE/f'analysis/native-execution-identity-v14/db-{db:02d}.json');tasks={t['task_record_rowid']:t for t in e['TASK_records']};defs={r['globalTaskId']:r for r in e['COMPUTE_definitions']};a=selected[db]
  for tid,ass in a.items():
   t=tasks[tid];d=defs[t['globalTaskId']];r=lk[fullkey(ass[3],d['opType_text'],ass[5],ass[2],ass[6],d)];rid=r['regime_id'];times[rid].append(t['endNs']-t['startNs']);ids[rid].append([db,tid]);miss[rid]+=not ass[11]
  def de(t):return defs[t['globalTaskId']]
  def op(t):return de(t)['opType_text']
  def ins(t):return dimensions(de(t)['inputShapes_text'])
  def outs(t):return dimensions(de(t)['outputShapes_text'])
  def role(t):return a[t['task_record_rowid']][5]
  def comp(t):return dict(TASK_rowid=t['task_record_rowid'],operator=op(t),streamId=t['streamId'],startNs=t['startNs'],endNs=t['endNs'],duration_ns=t['endNs']-t['startNs'])
  def one(xs):assert len(xs)==1,len(xs);return xs[0]
  def latest(xs,before):return one(sorted([t for t in xs if t['endNs']<=before],key=lambda t:t['endNs'])[-1:])
  ts=sorted([tasks[i] for i in a],key=lambda t:t['startNs']);m15=[t for t in ts if a[t['task_record_rowid']][3]==['M15']];m20=[t for t in ts if a[t['task_record_rowid']][3]==['M20']];m17=[t for t in ts if a[t['task_record_rowid']][3]==['M17']];m24=[t for t in ts if a[t['task_record_rowid']][3]==['M24']]
  for n in [t for t in m15 if op(t)=='RmsNorm']:
   M=ins(n)[0][0];s=latest([t for t in m15 if op(t)=='TensorMove' and ins(t)==[[M,20480]]],n['startNs']);group=[t for t in m15 if s['startNs']<=t['startNs'] and t['endNs']<=n['endNs']]
   assert Counter(op(t) for t in group)=={'TensorMove':1,'Cast':2,'Mul':1,'ReduceSum':1,'RmsNorm':1}
   ass=a[n['task_record_rowid']];root=graph_root.get((db,ass[8]));tail.append(dict(database_index=db,N=M,graph_ordinal=ass[8],root_bound=root,stages=[comp(t) for t in group],scope='104targettails;separate104draft/helpercollapse4TASKstagesremainper-regimemodels,notpairedtoRMSbyshape.'))
   if root:
    for t in group:
     name='HC_INPUT_CAST' if op(t)=='Cast' and ins(t)[0][-1]==5120 and len(ins(t)[0])==3 else 'HC_COLLAPSED_BF16' if op(t)=='Cast' else op(t)
     stage_times[name].append(t['endNs']-t['startNs'])
  for mm in [t for t in ts if op(t)=='MatMulV2' and a[t['task_record_rowid']][3]==['M16']]:
   M,K=ins(mm)[0];Nn=outs(mm)[0][1]
   if K==5120:
    tr=one([t for t in ts if op(t)=='Transpose' and ins(t)==[[8,M,16160],[3]] and t['startNs']>=mm['endNs']][:1]);head.append(dict(database_index=db,M=M,GEMM=comp(mm),gather_transpose=comp(tr),context=a[mm['task_record_rowid']][6],scope='192baseheadsource-orderedGEMM/Transposepairs;linkedcollectiveM23notpromotedhere.'))
    continue
   assert K==256 and Nn==129280
   emb=latest([t for t in m20 if op(t)=='GatherV2' and outs(t)==[[M,256]]],mm['startNs']);ad=one([t for t in m20 if op(t)=='Add' and outs(t)==[[M,129280]] and t['startNs']>=mm['endNs']][:1]);ar=one([t for t in m20 if op(t)=='ArgMaxV2' and ins(t)[0]==[M,129280] and t['startNs']>=ad['endNs']][:1]);cast=one([t for t in m20 if op(t)=='Cast' and ins(t)==[[M]] and t['startNs']>=ar['endNs']][:1]);copy=one([t for t in m20 if op(t)=='ViewCopy' and ins(t)[4]==[M] and t['startNs']>=cast['endNs']][:1]);group=[t for t in m20 if emb['startNs']<=t['startNs'] and t['endNs']<=copy['endNs']]
   assert Counter(op(t) for t in group)=={'GatherV2':1,'Slice':3,'Add':1,'ViewCopy':2,'ArgMaxV2':1,'Cast':1},(db,M,Counter(op(t) for t in group))
   markov.append(dict(database_index=db,B=M,context=a[mm['task_record_rowid']][6],MarkovGEMM=comp(mm),embedding=comp(emb),BF16_add=comp(ad),BF16_argmax=comp(ar),INT64_cast=comp(cast),token_column_copy=comp(copy),other_stages=[comp(t) for t in group],scope='520source-orderedsteps,notinventedrequest/rootepochs;IDafterstepi feedsnextembedding.'))
  for kernel in [t for t in m17 if op(t)=='rejection_greedy_sample_triton']:
   B=ins(kernel)[0][0];T=ins(kernel)[1][0];ar=latest([t for t in m17 if op(t)=='ArgMaxV2' and 'rejection_sample' in role(t) and ins(t)[0]==[T,129280]],kernel['startNs']);clone=latest([t for t in m17 if op(t)=='TensorMove' and ins(t)==[[T,129280]]],ar['startNs']);fill=latest([t for t in m17 if op(t)=='Fill' and outs(t)==[[B,6]]],kernel['startNs'])
   rejections.append(dict(database_index=db,B=B,T_capacity=T,Kmax=5,working_clone=comp(clone),output_initialization=comp(fill),target_argmax=comp(ar),rejection=comp(kernel),scope='72source-orderedgreedyTritonchains. Actualcu_lengths/flags/indices/RNGunrecorded;partialoutputwritesneed-1initialization.'))
  for start in [t for t in m24 if op(t)=='ClipByValueV2']:
   end=one([t for t in m24 if op(t)=='Cast' and de(t)['inputDataTypes_text']=='INT64' and de(t)['outputDataTypes_text']=='INT32' and '(86)' in role(t) and t['startNs']>=start['endNs']][:1]);group=[t for t in m24 if start['startNs']<=t['startNs'] and t['endNs']<=end['endNs']]
   assert len(group)==17 and Counter(op(t) for t in group)=={'ClipByValueV2':1,'IndexCheck':3,'Index':3,'GreaterEqual':1,'Cast':4,'Greater':1,'LogicalAnd':1,'Add':1,'SelectV2':2}
   states.append(dict(database_index=db,N=ins(start)[0][0],stages=[comp(t) for t in group],scope='72statecorrectioncalls:oldgatherssnapshotbeforetwowrites;singleprogrampreloadordoublebuffergates.'))
 assert len(tail)==104 and len(head)==192 and len(markov)==520 and len(rejections)==72 and len(states)==72
 assert len({(r['database_index'],r['MarkovGEMM']['TASK_rowid']) for r in markov})==520
 rows=[]
 for r in regs:
  rid=r['regime_id'];assert len(times[rid])==r['executions'] and miss[rid]==r['missing_PMU'],rid
  d=r['signature'];op=r['operator'];a=dimensions(d['inputShapes_text']);z=dimensions(d['outputShapes_text']);ib=vb(a,d['inputDataTypes_text']);ob=vb(z,d['outputDataTypes_text']);on=math.prod(z[0]) if z and z[0] is not None else None;io=None;model={}
  if op=='MatMulV2':
   M,K=a[0];V=z[0][1];assert a[1]==[V,K] and (K,V) in [(5120,16160),(256,129280)];F=2*M*K*V;io=2*(M*K+K*V+M*V);model=dict(M=M,K=K,N=V,role='TP8_BASE_HEAD' if K==5120 else 'REPLICATED_MARKOV_HEAD',dense_ops_2perMAC=F,BF16_weights_bytes=2*K*V,reference_compute_us=F/376e12*1e6,reference_cold_value_IO_us=io/1.6e12*1e6,conditional_reference_floor_us=max(F/376e12,io/1.6e12)*1e6,contract='Ordinarydensealgorithm/publishedBF16reference,notcompulsoryHBM/physicalpeak. BF16head/MarkovoutputandBF16Addbarriersretained. Coldweightperstepvsloopcachelambda1..5separate.')
  elif op=='RmsNorm':
   M,H=a[0];io=4*M*H+2*H+4*M;model=dict(M=M,H=H,ordinary_ops=4*M*H+M,rsqrt=M,aux_rstd_if_materialized_bytes=4*M,contract='Targetfinalnormanddraftheadnormdistinctroles/rowdomains. KeepHCcollapsedBF16beforenorm,BF16normbeforeGEMM,optionalbias/allnormalizedcontentreaders;statsdescriptornotphysicalwriteproof.')
  elif op=='ArgMaxV2':
   M,V=a[0];size=DT[d['inputDataTypes_text'].split(';')[0]];io=size*M*V+4*M;model=dict(M=M,V=V,comparisons=M*(V-1),source_score_dtype=d['inputDataTypes_text'].split(';')[0],native_INT32_output_bytes=4*M,contract='BF16MarkovvsFLOATtargetargmax. Preserveprocessed/roundedvalues,baselinecolumnties/NaN/padding/remaporder;globalV<2^31doesnotaloneauthorizeABI/dtypechange.')
  elif op=='Cast':
   io=None if on is None else sum(x for x in ib if x is not None)+sum(x for x in ob if x is not None);model=dict(elements=on,contract='HC BF16→FLOAT/weightedcollapse→BF16,full-vocabBF16→FLOATandargmax32→64→final32differ. Integercount/statewidthsandsnapshotvaluesretained;scalar/emptyextentnotguessed.')
  elif op=='ReduceSum':
   nin=math.prod(a[0]);io=ib[0]+ob[0];model=dict(reduction_adds=nin-on,HC_terms=4,contract='FP32premixweightedHCcollapse,sum4thenBF16cast;notnorm4featuresorunweightedmean.')
  elif op in ['Mul','Add']:
   io=sum(x for x in ib if x is not None)+sum(x for x in ob if x is not None);model=dict(pointwise_arithmetic_elements=on,contract='HCweightedFLOATMulvsMarkovBF16base+biasAddvsINT32stateAdd. BF16separateGEMMoutputs+BF16AddcannotbeoneunroundedcombinedGEMM;INT32correctioncast/addwidthpreserved.')
  elif op=='GatherV2':
   B,R=z[0];io=4*B*R+8*B;model=dict(rows=B,width=R,selected_BF16_table_read_bytes=2*B*R,table_descriptor_bytes=math.prod(a[0])*2,contract='Markovw1selectedrows,notwhole129280×256tablepercall. RecurrencecurrentT[:,i],seedincluded;retainedembeddingforconfidenceT[:,:D],notoutputT[:,1:D+1].')
  elif op=='ViewCopy':
   n=math.prod(a[4]);size=DT[d['inputDataTypes_text'].split(';')[0]];io=2*size*n;model=dict(actual_selected_source_elements=n,base_storage_descriptor_elements=a[0][0],selected_read_write_bytes=io,contract='OneMarkovscorestepB×VortokenBcolumnwrite,notwholeB×5V/Bmax×6base. Originalstridedinplacemutation/readersretained;ifdeadscoreprivatecannewfusedargmaxavoidstore.')
  elif op in ['Slice','Transpose','TensorMove','Index']:
   size=DT[d['outputDataTypes_text'].split(';')[0]];io=2*size*on
   if op=='Index':io+=DT[d['inputDataTypes_text'].split(';')[-1]]*math.prod(a[-1])
   model=dict(selected_output_elements=on,selected_payload_read_write_bytes=io,contract='Indexselectedrows/Sliceselectedstep/TransposeM×8×Vp/nativeTensorMoveselectedplane. Noentireparentextenttraffic. RawHCMTPsnapshotandworkingraw-preservationclonehaveindependentreaders;score/headrank-majorlayoutnotordinary2Dview.')
  elif op=='IndexCheck':
   M=a[1][0];model=dict(index_elements=M,reference_range_comparisons=2*M,output_bytes=None,bounds='Originalsourcetensorextentperexecution;mergedChecksignaturecanhavemultipleN. Preservebounds/negativewrap/errororgivenvalid-domainproof.',native_errorflag_IO_and_sync_unbound=True)
  elif op=='Fill':
   size=DT[d['outputDataTypes_text']];io=size*on;model=dict(stores=on,contract='RejectionB×6INT32-1placeholderlive:currentgreedykernelwritesprefix/bonusonlyanddoesnotfullwriteunaccepted/non-greedyrows. Newfullwritekernelcanfoldinit;cannotdropcurrentFillbyname.')
  elif op=='rejection_greedy_sample_triton':
   B=a[0][0];T=a[1][0];model=dict(B=B,draft_token_capacity=T,max_speculative_tokens=z[0][1]-1,comparison_upper_bound=T,actual_prefix_comparisons_and_writes_unbound=True,input_read_upper_reference_bytes=12*T+12*B,output_write_upper_reference_bytes=4*math.prod(z[0]),contract='INT32draft/INT64targetargmax/INT32bonus;ordinaryfirstmismatchstop,syntheticusesuniform/rateswithsameRNG. Output-1suffixretained;is_greedy/actualcu_values/flagsnotattestedbyshapes.')
  else:
   assert op in ['ClipByValueV2','GreaterEqual','Greater','LogicalAnd','SelectV2'],op
   io=sum(x for x in ib if x is not None)+sum(x for x in ob if x is not None);model=dict(pointwise_or_select_elements=on,contract='Batchcorrectiong=max(prevpos,0),participating=prevpos>=0&&prevdraft>0,corrected=oldcomputed[g]+INT32validcount[g],otherwiseCPU;acceptedelseoldaccepted. Readsnapshot/alias/intwidth/streamobserversrequired,notanordinaryinplaceparallelgather-map.')
  rows.append(dict(regime_id=rid,module_candidates=r['module_candidates'],operator=op,role=r['role'],origin=r['origin'],context=r['context'],executions=r['executions'],signature=d,native_TASK_keys=ids[rid],missing_PMU=miss[rid],measured_native_task_ns=stat(times[rid]),work_model=model,one_read_write_logical_value_bytes=io,consumer_fusion_review_complete_in_this_turn=True,physical_numeric_ceiling_proved=False))
 write(O/'heads-sampling-regime-models-v65.json',dict(generated_utc=datetime.now(timezone.utc).isoformat(),regimes=rows,regime_count=166,captured_execution_count=9792,scope='M15/16/17/20/24strictplusM04|M16selection176ambiguityretained. 192basehead+520MarkovGEMMs;allwork/IOconditional,notnativephysicalcapacity/causalgain.',overall_complete=False))
 write(O/'heads-sampling-native-chains-v65.json',dict(target_tails=tail,base_heads=head,Markov_steps=markov,greedy_rejections=rejections,state_corrections=states,root_graph_target_tail_stage_ns={k:stat(v) for k,v in stage_times.items()},limits='SourceorderedTASKidentitiesnotactualrequest/tensor/RNG/flagorIvalues. 104targettailvs104draftcollapsehelperkeptseparate;M23linkedheadcollectivesnotpromotedhere.',overall_complete=False))
 print(dict(regimes=166,executions=9792,target_tails=len(tail),base_heads=len(head),Markov_steps=len(markov),rejections=len(rejections),state_calls=len(states),root_tail_stage_ns={k:stat(v) for k,v in stage_times.items()}))
if __name__=='__main__':main()
