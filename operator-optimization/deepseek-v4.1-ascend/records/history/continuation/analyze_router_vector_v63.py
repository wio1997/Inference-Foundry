"""Native per-regime operator work and graph links; preserve unpriced MemSet semantics."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import math,sqlite3
from finalize_origin_coverage_v9 import N,O,read,write
from mirror_saved_profile_core_v53 import CACHE
from analyze_state_engram_v62 import fullkey,dimensions,bytesparts
from analyze_projection_ceiling_v61 import stat

def main():
 assert not (O/'router-vector-regime-models-v63.json').exists()
 cov=read(O/'operator-analysis-coverage-ledger-v62.json');universe={r['regime_id']:r for r in read(O/'operator-shape-ceiling-ledger-v55.json')['observed_regimes']}
 regs=[universe[r['regime_id']] for r in cov['regimes'] if r['state'].startswith('REGIME_') and r['module_candidates'] in [['M12'],['M07'],['M11'],['M07','M09']]]
 assert len(regs)==145 and sum(r['executions'] for r in regs)==75802
 lk={fullkey(r['module_candidates'],r['operator'],r['role'],r['origin'],r['context'],r['signature']):r for r in regs};selected=defaultdict(dict)
 for a in read(CACHE/'analysis/native-PMU-complete-v18/resource-functional-attribution-and-uncertainty-v20-v3.json')['execution_assignments']:
  if a[3] in [['M12'],['M07'],['M11'],['M07','M09']]:selected[a[0]][a[1]]=a
 graphs=defaultdict(list)
 for g in read(CACHE/'analysis/native-execution-identity-v14/graph-blocks-PMU-and-MC2-v14.json')['graph_executions']:graphs[g['database_index']].append(g)
 projection_chains={(r['database_index'],r['graph_ordinal'],r['block_ordinal']):r for r in read(O/'projection-graph-chains-v61.json')['graph_projection_chains']}
 times=defaultdict(list);ids=defaultdict(list);PMU_missing=Counter();chains=[];stages=defaultdict(list);memset=[];schema_evidence=[]
 for db in range(32):
  d=read(CACHE/f'analysis/native-execution-identity-v14/db-{db:02d}.json');ts={t['task_record_rowid']:t for t in d['TASK_records']};defs={r['globalTaskId']:r for r in d['COMPUTE_definitions']}
  for tid,a in selected[db].items():
   t=ts[tid];de=defs[t['globalTaskId']];k=fullkey(a[3],de['opType_text'],a[5],a[2],a[6],de)
   if k not in lk:continue
   r=lk[k];rid=r['regime_id'];times[rid].append(t['endNs']-t['startNs']);ids[rid].append([db,tid]);PMU_missing[rid]+=not a[11]
   if de['opType_text']=='MemSet':memset.append(dict(database_index=db,TASK=t,definition=de,functional_assignment=a))
  def op(t):return defs[t['globalTaskId']]['opType_text']
  def ins(t):return dimensions(defs[t['globalTaskId']]['inputShapes_text'])
  def comp(t):return dict(TASK_rowid=t['task_record_rowid'],operator=op(t),streamId=t['streamId'],startNs=t['startNs'],endNs=t['endNs'],duration_ns=t['endNs']-t['startNs'])
  def one(a):assert len(a)==1,len(a);return a[0]
  for g in graphs[db]:
   root=bool(g['prepare_root_keys'])
   for b in g['blocks']:
    tasks=sorted([ts[i] for i in b['task_record_rowids']],key=lambda t:(t['startNs'],t['task_record_rowid']))
    norm=one([t for t in tasks if op(t)=='RmsNormCast']);router=one([t for t in tasks if op(t)=='MatMulV3' and ins(t)==[[3,5120],[384,5120]]]);top=one([t for t in tasks if op(t)=='MoeGatingTopKHash'])
    cast=one([t for t in tasks if op(t)=='Cast' and ins(t)==[[3]]]);eq=one([t for t in tasks if op(t)=='Equal' and ins(t)==[[3],None]]);fill=one([t for t in tasks if op(t)=='Fill' and ins(t)==[[1],None]]);sel=one([t for t in tasks if op(t)=='SelectV2' and ins(t)==[[3],None,[3]]])
    assert norm['endNs']<=router['startNs'] and router['endNs']<=top['startNs']
    assert cast['endNs']<=eq['startNs'] and eq['endNs']<=sel['startNs'] and fill['endNs']<=sel['startNs']<=top['startNs']
    pc=projection_chains[(db,g['graph_ordinal'],b['block_ordinal'])]
    qn=ts[pc['stages']['Q_NORM']['task_record_rowid']];kn=ts[pc['stages']['KV_NORM']['task_record_rowid']]
    qrope=one([t for t in tasks if op(t)=='InplacePartialRotaryMul' and ins(t)==[[3,1,64,512],[3,1,1,64],[3,1,1,64]] and t['startNs']<next(t['startNs'] for t in tasks if op(t)=='SparseFlashMla')])
    oro=one([t for t in tasks if op(t)=='InplacePartialRotaryMul' and ins(t)==[[3,1,64,512],[3,1,1,64],[3,1,1,64]] and t['startNs']>qrope['endNs']])
    tr=one([t for t in tasks if op(t)=='Transpose' and ins(t)==[[3,8,8,512],[4]]]);dq=[t for t in tasks if op(t)=='DynamicQuant' and ins(t)==[[3,1280]]]
    assert len(dq) in [1,2] and qn['endNs']<=dq[0]['startNs'] and oro['endNs']<=tr['startNs']
    row=dict(database_index=db,graph_ordinal=g['graph_ordinal'],block_ordinal=b['block_ordinal'],root_bound=root,router_stages={k:comp(t) for k,t in dict(norm=norm,router=router,ID_cast=cast,ID_equal=eq,ID_fill=fill,ID_select=sel,topk=top).items()},Q_norm=comp(qn),KV_norm=comp(kn),Q_input_quants=[comp(t) for t in dq],query_RoPE=comp(qrope),output_inverse_RoPE=comp(oro),send_layout_transpose=comp(tr),scope='ExacttypedTASK/structuralblock+sourceorder;functionalambiguitiesnotrewrittenastensoridentity. Norm800→routerTopK816notpairedbytime.')
    chains.append(row)
    if root:
     for k,t in [('FFN_NORM_CAST',norm),('ROUTER_FLOAT',router),('TOPK',top),('ID_CAST',cast),('ID_EQUAL',eq),('ID_FILL',fill),('ID_SELECT',sel),('Q_RMS1280',qn),('KV_RMS512',kn),('Q_ROPE64',qrope),('OUTPUT_INVERSE_ROPE64',oro),('SEND_LAYOUT',tr),('Q_DYNAMIC_QUANT',dq[0])]:stages[k].append(t['endNs']-t['startNs'])
  # All 32 raw DBs: determine whether the saved schema can price MemSet.
  remote=Path(d['database_path']);idx=max(i for i,x in enumerate(remote.parts) if x=='analysis');rel=Path(*remote.parts[idx:]);local=CACHE/rel;assert local.is_file(),local
  conn=sqlite3.connect(local.as_uri()+'?mode=ro&immutable=1',uri=True);conn.execute('pragma query_only=1')
  tables=[r[0] for r in conn.execute("select name from sqlite_master where type='table'")]
  api=[r[1] for r in conn.execute('pragma table_info(CANN_API)')];task=[r[1] for r in conn.execute('pragma table_info(TASK)')]
  schema_evidence.append(dict(database_index=db,tables=tables,CANN_API_columns=api,TASK_columns=task,operand_or_memory_trace_tables=[t for t in tables if t in ['MEMORY_RECORD','NPU_MEMORY','MEMCPY_INFO','MEMSET_INFO','OP_ARGS']]))
  conn.close()
 assert len(chains)==3200 and sum(c['root_bound'] for c in chains)==2560
 rows=[]
 for r in regs:
  rid=r['regime_id'];assert len(times[rid])==r['executions'] and PMU_missing[rid]==r['missing_PMU'],rid
  de=r['signature'];op=r['operator'];a=dimensions(de['inputShapes_text']);z=dimensions(de['outputShapes_text']);outn=math.prod(z[0]) if z and z[0] is not None else None;model={};io=None;complete=True
  if op in ['RmsNorm','RmsNormCast']:
   m,h=a[0];dual=op=='RmsNormCast';io=(8*m*h+2*h) if dual else 4*m*h+2*h+4*m
   model=dict(M=m,H=h,ordinary_scalar_arithmetic=4*m*h+m,rsqrt_elements=m,input_BF16_and_output_BF16_rounding=True,dual_output=dual,FP32_output_is_widened_rounded_BF16=dual,auxiliary_rstd_bytes=0 if dual else 4*m,qualified_cost='ConventionalfullrowRMS,square/reduction/mean+eps/rsqrt/2multiply,notactualVectorinstructions. Liveotherconsumersandnativeorderremain.',fusion='Dualoutputalreadyfused. WidenroundedBF16inroutercanremoveFP32interface;Qnorm→DQcankeepBF16roundedrowonchip. SWA norm→RoPE→cachemustpreservefull512reduceandIndexKeyforkwhereapplicable.')
  elif op in ['MatMulV2','MatMulV3']:
   m,h=a[0];e=z[0][1];assert a[1]==[e,h] and e in [128,384] and de['inputDataTypes_text']=='FLOAT;FLOAT';io=4*(m*h+h*e+m*e)
   model=dict(M=m,H=h,E=e,conventional_dense_ops_2perMAC=2*m*h*e,FP32_weights_bytes=4*h*e,FLOAT_active_mode_and_capacity_unbound=True,cold_value_IO_reference_us=io/1.6e12*1e6,actual_physical_floor_us=None,fusion='RouterGEMM→nonlinear/bias/TopK requiresfullEselectionorlocalcandidate+merge,sameFP32accumulationandBF16normround. No E384→k6GEMMpruningwithoutknownexactexpertset.')
  elif op=='MoeGatingTopKHash':
   m,e=a[0];k=z[0][1];assert a[3] is None and de['inputDataTypes_text'].split(';')[3]=='DT_UNDEFINED'
   io=4*m*e+8*e+8*m+8*m*k
   model=dict(M=m,E=e,k=k,typed_optional_tid2eid_absent=True,normType2_source_work=dict(exp_elements=m*e,add_one_elements=m*e,ln_elements=m*e,sqrt_elements=m*e,bias_add_elements=m*e,selected_weight_reduce_adds=m*(k-1),selected_eps_adds=m,selected_weight_divisions=m*k,selected_weight_scale_multiplies=m*k),sort='Sourcewithout-groupSortallEwithinUB. PartialTopKalternativeisnotprovennativecomparator/tie/NaN-equivalent.',unique_two_bias_vector_bytes=8*e,source_per_program_two_bias_load_bytes=8*e*de['blockNum'],full_descriptor_third_output_bytes=4*m*e,third_output_written_under_source_out_flag_false=False,third_output_GM_traffic_counted=0,source_UB_fixed_E_buffers_bytes=60*e,source_UB_additional_k_aligned_buffers_unpriced=True,contract='ActualnormType2sourceExp→+1→Ln→SqrtinFP32,addvision/textbiasforselection,gatherUNBIASEDnormscores,selectedsum+eps→Div→scalingevenrenorm0. Stable-softplusrewriteisnotassumedequivalent. SourceconsistentG1/normType2,notexecutedtiling/binaryattestation.')
  elif op=='DynamicQuant':
   m,h=a[0];io=3*m*h+4*m;model=dict(M=m,H=h,input_BF16_bytes=2*m*h,output_INT8_bytes=m*h,FLOAT_row_scale_bytes=4*m,reference_abs_elements=m*h,reference_row_max_comparisons=m*(h-1),reference_quantize_elements=m*h,contract='Ordinarynpu_dynamic_quantrow1280/FLOATscale,NOTIndexerhead128FP16scale. Exactamax/scale/round/saturate/zero/nonfinitepolicyremainsnativecontract;RMSBF16roundcannotcancelalgebraically.',fusion='RMS→quantandduplicateQRquantsreuse/jointGEMMreadbudgetscombinedinrouter-vector-fusion-costs-v63.json. Actualtypedrowownershippreserved.')
  elif op=='InplacePartialRotaryMul':
   m=a[0][0];heads=math.prod(a[0][1:-1]);n=m*heads*64;io=4*n+8*m*64;model=dict(M=m,heads=heads,head_width=512,rotated_tail=64,BF16_tail_values=n,ordinary_multiplies=2*n,ordinary_adds=n,sign_flip_elements=n//2,BF16_to_FLOAT_cast_elements=n,FLOAT_to_BF16_RINT_elements=n,unique_cos_sin_BYTES=8*m*64,contract='Currentkernelstridedtail-onlyGMreads/writes,cos/sinbroadcastacrossheads,interleavepartnergatherandparitysignthenadd;CAST_RINT. InverseusesnegateSinflagornegatedsinargumentpercapability. PreserveproducerBF16boundary;full512planeisnottailtraffic.')
  elif op=='ScatterNdUpdateSk':
   m,h=a[2];model=dict(M_descriptor=m,H=h,all_valid_reference_update_read_indices_write_bytes=m*(4*h+8),actual_valid_rows_unbound=True,cache_descriptor_bytes=math.prod(a[0])*2,contract='OnlyselectedBF16rows+INT32pairsandcachewrites;wholecacheextentnottraffic. FuseKVnorm/tailRoPEstorewithactualslots/pad/duplicateorder/futurecacheobservers.')
  elif op=='Transpose':
   io=4*outn;model=dict(output_BF16_elements=outn,materialized_read_write_bytes=io,contract='RestoreTPheads[localT,8,8,512]→[8,localT,8,512]. Directsendlayoutpossiblewithalltoallrow/headorder,BF16attentionboundary/padding/banklifetime. v60combinedoutputRoPElayoutbudgetretained,notnewextraTransposegain.')
  elif op=='TensorMove':
   io=4*outn;model=dict(output_BF16_elements=outn,materialized_read_write_bytes=io,source_role='PROJECTED_OUTPUT_BANK_COPY' if '_forward_o_proj' in r['role'] else 'CP_ATTENTION_PREFIX_COPY',contract='WO_Bcanconditionalwriteexistingoutputbankdirectly;CPprefixcopycancomposeproducer+suffixzero+sendlayout. Allalias/capturedbank/readiness/padsemanticsneeded,notblanketremoval.')
  elif op=='ZerosLike':
   io=2*outn;model=dict(BF16_stores=outn,loads=0,source_role='REDUCE_SCATTER_PADDING_SUFFIX' if '_ascend_sp_reduce_scatter_impl' in r['role'] else 'CP_OUTPUT_OR_EXCHANGE_PADDING',contract='RSsuffixzerosarelivecollectiveinput;preservezeroorvirtualzeroequivalent. CPfullzerobufferprefixmaythenbeoverwrittenbyactiveoutput,onlythatcoveredprefixzeroisredundantunderbounds;localemptyqueriescanstillneedglobalprojection.')
  elif op=='ConcatD':
   io=4*outn;model=dict(M_real=a[0][0],padding_rows=a[1][0],M_padded=z[0][0],H=z[0][1],materialized_read_write_bytes=io,contract='OutputprojectionproducerwritesPpaddedbankdirectlywithsamezeros/prefixviewconditional;collectivecountPandsameBF16partialreductionstillrequired. Do notdropzerosorchangeRSshapeunilaterally.')
  elif op=='PadV3':
   dtype=de['inputDataTypes_text'].split(';')[0];size={'DT_BF16':2,'FLOAT':4,'INT8':1,'INT64':8}[dtype];inn=math.prod(a[0]);io=size*(inn+outn);model=dict(input_elements=inn,output_elements=outn,element_bytes=size,zero_suffix_store_bytes=size*(outn-inn),value_read_write_bytes=io,contract='MC2SPpaddingalignshidden/logits/mask/IDs,maskpaddedfalseandIDs0. Producerdirectpaddedoutputretainssamevalidprefix,mask/otherconsumers. MemSetN/Asize/destinationNOTinferredfrompairedPadextent.')
  elif op=='MemSet':
   complete=False;model=dict(work_count=None,traffic_bytes=None,destination_kind=None,reason='N/AComputeInfoandCANN_API/TASKschemasdonotrecordoperandaddress/length. CannotdecideworkspaceflaginitializationversuspayloadzeroorsafelyclaimitremovedbyPadfusion.',condition='Ifseparatepayloadzerooverwrittenandallreadersknown,itcanfoldintoproducer. Ifworkspace/controlclear,protocolmustpreserveit. Existingcapturememory/op_argsdisabled;nohardwaretimeorHMBbytefill.')
  elif op=='Cast':
   dt=de['inputDataTypes_text'];ot=de['outputDataTypes_text'];size={'INT32':4,'INT64':8,'BOOL':1,'INT8':1}
   scalar=a[0] is None;io=None if scalar else outn*(size[dt]+size[ot]);model=dict(elements=None if scalar else outn,source_scalar_fill_parameter_possible=scalar,contract='INT32→INT64IDsisexact;BOOL→INT8pad→BOOLcanfuseonlywithall0/1maskpreserved. EmptyshapeBOOL→INT8maybescalarFalsefilling,notautomaticzero/oneelementGMtraffic. Noinputvalue/generationidentityfromsameaddress.')
  else:
   assert op in ['Equal','Fill','SelectV2'],op
   if op=='Equal':io=9*outn;model=dict(INT64_compare_elements=outn,operation='id==-1',scalar_GM_load_not_assumed=True)
   elif op=='Fill':io=8*outn;model=dict(INT64_zero_stores=outn,source_dimension_operand_not_value_input=True)
   else:io=17*outn;model=dict(INT64_select_elements=outn,operation='where(id==-1,0,id)',scalar_GM_load_not_assumed=True)
   model['fusion']='FusewithIDcast/topKinputorpergenerationcrosslayerreuse;f(x)=0onlyifx==-1elseINT64x commuteswithzero-pad/rowselect;keepothernegativevisionIDs,rawIDsandlayerparameters/collectives.'
  rows.append(dict(regime_id=rid,module_candidates=r['module_candidates'],operator=op,role=r['role'],origin=r['origin'],context=r['context'],signature=de,executions=r['executions'],native_TASK_keys=ids[rid],missing_PMU=PMU_missing[rid],measured_native_task_ns=stat(times[rid]),work_model=model,one_read_write_logical_value_bytes=io,consumer_fusion_review_complete_in_this_turn=complete,physical_numeric_ceiling_proved=False))
 assert sum(len(r['native_TASK_keys']) for r in rows)==75802 and len(memset)==960
 write(O/'router-vector-regime-models-v63.json',dict(generated_utc=datetime.now(timezone.utc).isoformat(),regimes=rows,regime_count=145,captured_execution_count=75802,completed_scoped_conditional_regimes=142,completed_scoped_conditional_executions=74842,profile_insufficient_MemSet_regimes=3,profile_insufficient_MemSet_executions=960,population='Exactfunctionalmodule/op/role/origin/context/7ComputeInfofields;145previouslypendingregimes. 1280M07|M09quantambiguityretained.',logical_value_IO_not_actual_HBM_traffic=True,FLOAT_Vector_sort_special_function_capacity_unbound=True,overall_complete=False))
 write(O/'router-vector-graph-chains-v63.json',dict(chains=chains,root_bound_chains=2560,phase_unbound_chains=640,root_stage_ns={k:stat(v) for k,v in stages.items()},qualification='ObservedTASKtimingsandordersnottensoralias/metadata/nativebinaryortheendtoendgain. ExactIDandnorm/quant/RoPEstageidentitiesaddperregimecoveragewithoutchangingoldproofcontracts.',overall_complete=False))
 write(O/'router-MemSet-profile-insufficiency-v63.json',dict(executions=960,native_records=memset,regimes=[r['regime_id'] for r in rows if r['operator']=='MemSet'],raw_schema_audit=schema_evidence,all_32_DBs_have_no_operand_memory_tables=all(not x['operand_or_memory_trace_tables'] for x in schema_evidence),retained_scope='All3regimesstaypendingexplicitwork/traffic/fusiondecision. No sourceguess/duration*bandwidthorwholePadextentfill. Savedprofilecannotresolveaddress/length;onlyfutureauthorizedexistingarg/IR/runtimeartifactcouldproveit,noneassumed.',overall_complete=False))
 print(dict(regimes=145,executions=75802,scoped_reviewed_regimes=142,scoped_reviewed_executions=74842,MemSet_pending_regimes=3,MemSet_pending_executions=960,graph_chains=len(chains),root_stage_ns={k:stat(v) for k,v in stages.items()}))

if __name__=='__main__':main()
