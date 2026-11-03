"""Draft Post/Pre, context/terminal demand and conditional faithful-replay representation."""
from pathlib import Path
from collections import Counter
from copy import deepcopy
from itertools import product
from fractions import Fraction
import json,struct
from finalize_origin_coverage_v9 import N,O,read,write
from analyze_post_projection_v151 import pin


def bf(v):
 n=struct.unpack('>I',struct.pack('>f',float(v)))[0];return struct.unpack('>f',struct.pack('>I',(n+32767+((n>>16)&1))&0xffff0000))[0]


def det(m):
 a=[[Fraction(v)for v in row]for row in m];d=Fraction(1)
 for i in range(len(a)):
  j=next((j for j in range(i,len(a))if a[j][i]),None)
  if j is None:return Fraction(0)
  if i!=j:a[i],a[j]=a[j],a[i];d=-d
  p=a[i][i];d*=p
  for k in range(i+1,len(a)):
   f=a[k][i]/p
   for l in range(i+1,len(a)):a[k][l]-=f*a[i][l]
 return d


def main():
 assert not(O/'post-draft-terminal-native-joins-v154.json').exists()
 cur=read(N/'CURRENT-POSTCHECK.json');assert cur['checkpoint']==153 and cur['status']=='PASS'
 a=read(O/'hc-draft-native-frontiers-v142.json');b=read(O/'norm-terminal-native-frontiers-v146.json');p=read(O/'post-pre-native-frontiers-v147.json');bb=b['bindings'];pb=p['bindings'];d146={(z['database_index'],z['source_order_ordinal']):z for z in b['draft_frontiers']};di147={(z['database_index'],z['source_order_ordinal'],z['block']):z for z in p['draft_intra']};dc147={(z['database_index'],z['source_order_ordinal'],z['from_block']):z for z in p['draft_cross']};binding={};draft=[];tails=[];cnt=Counter();geo=Counter();payloads={}
 def keep(di,r):
  z=deepcopy(r);assert z.get('database_index',di)==di;z['database_index']=di;key=(di,z['TASK']['task_record_rowid'])
  if key in binding:assert binding[key]['TASK']==z['TASK']and binding[key]['API']==z['API']
  else:binding[key]=z
  return list(key)
 def eq(x,y):assert x['TASK']==y['TASK']and x['API']==y['API']
 for i,z in enumerate(a['records']):
  di=z['database_index'];q=d146[di,z['source_order_ordinal']];assert z['source_ancestry']==q['ancestry'];roles={'REPEAT':keep(di,z['repeat']),'SEED_COPY':keep(di,z['seed'])};edges=[];cross=[];local=[]
  for j,l in enumerate(z['layers']):
   for nm,r in l['stages'].items():roles['L%d_%s'%(j,nm)]=keep(di,r)
   eq(l['stages']['NORM'],bb[q['layers'][j]['INPUT_NORM']]);eq(l['stages']['FFN_NORM'],bb[q['layers'][j]['FFN_NORM']]);e=di147[di,z['source_order_ordinal'],j];eq(l['stages']['POST'],pb[e['left']]);eq(l['stages']['FFN_PRE'],pb[e['right']]);eq(l['stages']['HC'],pb[e['earlier_premix_and_Post_coeff_provider']]);eq(l['stages']['FFN_POST'],pb[e['future_Post']]);edges.append(dict(block=j,left_role='L%d_POST'%j,right_role='L%d_FFN_PRE'%j,incoming_premix_role='L%d_HC'%j,future_Post_role='L%d_FFN_POST'%j))
   if j<2:
    e=dc147[di,z['source_order_ordinal'],j];eq(l['stages']['FFN_POST'],pb[e['left']]);eq(z['layers'][j+1]['stages']['HC'],pb[e['right']]);eq(l['stages']['FFN_PRE'],pb[e['incoming_premix_provider']]);cross.append(dict(from_block=j,to_block=j+1,left_role='L%d_FFN_POST'%j,right_role='L%d_HC'%(j+1),incoming_premix_role='L%d_FFN_PRE'%j))
   local.append(int(l['stages']['HC']['definition']['inputShapes_text'].strip('"').split(',')[0]))
  for nm,r in z['collapse'].items():eq(r,bb[q['collapse'][nm]]);roles['FINAL_COLLAPSE_'+nm]=keep(di,r)
  roles['HEAD_NORM']=keep(di,bb[q['head_norm']])
  for nm,ix in q['main'].items():roles[nm]=keep(di,bb[ix])
  hr=int(bb[q['head_norm']]['definition']['inputShapes_text'].strip('"').split(',')[0]);rr=int(z['repeat']['definition']['outputShapes_text'].strip('"').split(',')[0]);cnt[z['source_ancestry']]+=1;cnt['draft_intra_edges']+=len(edges);cnt['draft_cross_edges']+=len(cross);geo[q['ancestry'],q['context_rows'],q['query_rows'],tuple(local),rr,hr]+=1;draft.append(dict(database_index=di,source_order_ordinal=z['source_order_ordinal'],old_v142_record_index=i,ancestry=q['ancestry'],roles=roles,intra=edges,cross=cross,context_rows=q['context_rows'],query_rows=q['query_rows'],repeat_rows=rr,owner_HC_rows_by_layer=local,head_norm_rows=hr,main_context_present=bool(q['main']),raw_collapse_to_global_head_requires_transport_and_row_mapping=True,actual_seed_query_context_weight_bank_or_client_phase_bound=False,optional_confidence_execution_inferred=False))
 intra=p['target_intra']
 for i,z in enumerate(b['target_tails']):
  di=z['database_index'];s={nm:bb[ix]for nm,ix in z['stages'].items()};candidates=[e for e in intra if e['database_index']==di and pb[e['earlier_premix_and_Post_coeff_provider']]['TASK']['task_record_rowid']==s['LAST_HC']['TASK']['task_record_rowid']];assert len(candidates)==1;e=candidates[0];eq(s['LAST_HC'],pb[e['earlier_premix_and_Post_coeff_provider']]);eq(s['LAST_FFN_POST'],pb[e['future_Post']]);roles={nm:keep(di,r)for nm,r in s.items()};roles['LAST_ATTENTION_POST']=keep(di,pb[e['left']]);roles['LAST_FFN_PRE']=keep(di,pb[e['right']]);g=int(s['TERMINAL_NORM']['definition']['inputShapes_text'].strip('"').split(',')[0]);tails.append(dict(database_index=di,forward_API_rowid=z['forward_API_rowid'],old_v146_tail_index=i,last_block=e['block'],roles=roles,collective_indices=z['collectives'],M=z['owner_rows'],G=g,T=z['cache_rows'],snapshot_readers_or_output_row_order_values_bound=False,terminal_raw_snapshot_is_independent_publication=True,no_next_HCPre_in_reviewed_terminal_source_branch=True))
 payloads['post-draft-terminal-native-joins-v154.json']=dict(draft_invocations=draft,target_tails=tails,bindings=list(binding.values()),unique_TASK_bindings=len(binding),role_bindings=sum(len(z['roles'])for z in draft)+sum(len(z['roles'])for z in tails),collectives=deepcopy(b['collectives']),collective_count=len(b['collectives']),databases=deepcopy(b['databases']),counts=dict(cnt),draft_geometry=[dict(ancestry=k[0],context_rows=k[1],query_rows=k[2],owner_HC_rows=list(k[3]),repeat_rows=k[4],head_norm_rows=k[5],frontiers=v)for k,v in geo.items()],scope='104draftcalls312attnPost->FFNPre208FFNPost->nextattnPreedgesand24targetterminaltails. JoinusesidenticalHC/Norm/FFNNorm/Collapse/PostAPI+TASKIDs,notordinalalone. DUMMY/PROPOSEsourceancestrynotformalclientphase;localowner/repeat/globalheadrowsdiffer.48MTPcommunicationsretainrawfamilycontrols/nonComputeambiguity.',new_native_operand_or_phase_promotions=0,new_device_work=False,overall_complete=False)
 # Conditional storage of original basis values and all public per-row Post parameters.
 rep=[]
 for H,rows,depth in product([5120],[1,2,3],range(7)):
  basis=(depth+1)*2*H;params=depth*(4*4+4)*4;content=basis+params;raw=4*2*H
  rep.append(dict(H=H,owner_rows=rows,Post_depth=depth,original_embedding_and_update_BF16_value_content_per_row=basis,all_original_resmix16_postmix4_FLOAT_content_per_row=params,declared_replay_input_content_per_row=content,raw_4H_BF16_content_per_row=raw,raw_minus_declared_replay_input_content_per_row=raw-content,declared_total_input_content=rows*content,raw_total_content=rows*raw,additional_retained_value_work_mathstate_scratch_or_abi_cost_not_included=True,actual_native_encoder_or_consumer_proved=False))
 # Lossless replay reference must execute each channel BF16 rounding; it is not a linear rank-2 shortcut.
 rankcase=None;alpha=[Fraction(1,16),Fraction(1,4),Fraction(1,2),Fraction(3,4)]
 for ev,av in product([[1,2,3,4],[0.05,1,1.7,3.5],[1,1.01,2.03,7.1]],[[0.1,0.11,0.13,0.19],[1,0.3,-0.7,0.13],[0.05,2.5,-1.7,0.03]]):
  E=[bf(v)for v in ev];A=[bf(v)for v in av];linear=[[Fraction(E[j])+alpha[c]*Fraction(A[j])for j in range(4)]for c in range(4)];rounded=[[bf(v)for v in row]for row in linear];dr=det(rounded)
  if dr:
   rankcase=dict(embedding_BF16=E,update_BF16=A,postmix_FLOAT=[float(v)for v in alpha],resmix_identity_reference=True,unrounded_matrix=[[str(v)for v in row]for row in linear],unrounded_rank_upper=2,unrounded_determinant=str(det(linear)),rounded_Post_BF16=rounded,rounded_determinant=str(dr),rounded_rank=4,scope='Exactfiniteconstructedidentity-resmix/FP32dyadicproductsumthenBF16RNEreference,notloadednativeparameters/operationorder. Rulesoutusingunroundedrank2linearalgebrafordemandedBF16raw;doesnotruleoutstoringfactorsplusfaithfulper-channelroundingreplay.',actual_native_parity_proved=False);break
 assert rankcase
 payloads['post-draft-terminal-replay-representation-v154.json']=dict(cases=rep,count=len(rep),model='RetainoriginalembeddingE_BF16andoneBF16attention/FFNupdatevectorperPostplusFULL16resmix+4postmixFLOATscalarsperrowperPost. DecodeeveryoriginalchannelandPostBF16roundinoriginaloperationorder;noRprefold. Firstrepeatvirtualrepresentationonlyunderactualrow/shard/pad/firstpremixcontract.',rank_round_counterexample=rankcase,reference_input_content_formula='2H(d+1)+80d bytes/row versus8H raw4HBF16,plusS_other foractualreplaystate/scratch/newABI/allreader/capture/progress. Conditionalinputcontentnotallocationoruniversalstoragebound.',required_gates=['OriginalPostinputs/params/executable/mathmode/determinismandroundorderfrozen;allerrors/sideeffects/observerspreserved.', 'LocalHCownerrowsnotglobalrepeat/headrows;correctSP/padding/request/slot/generationmappingrequired.', 'FutureHCfull4Hstats/currentcoeff/collapse/futurePost/rawheadreadersmustreceiveexactoriginalBF16values,notunroundedlowrankfactors.', 'OriginalCANNkernelsstillrequirematerializedGMrawunlessnewtile/encodedconsumerABI;replaymayextendinputlifetimesandrepeatcompute. Scratch/weight/transport/arena/finiteprogresspricedseparately.', 'Encodingdepth>=3canexceedrawcontentalreadyatH5120. Materialize/reset/stream/recomputearealternativegraphs;notclaimall-depthcompression.', 'InputE/updatevectorsmayalreadyexistbutretainingthemextendsliveness;countaliasesandNEWcopiesonce,nofreecapacityoractualspeedupproved.'],whole_native_or_physical_gain_proved=False,overall_complete=False)
 # Target full raw MTP snapshot stays. Collapse/norm row pruning and fusion share ports.
 cases=[];H=5120;G=4
 for hm,cm,fuse in product(range(16),range(16),[False,True]):
  head={r for r in range(G)if hm>>r&1};conf={r for r in range(G)if cm>>r&1};needed=head|conf;ordered=list(reversed(sorted(head)))+([min(head)]if head else[]);ports={'FULL_RAW_SNAPSHOT_READ':8*G*H,'FULL_RAW_SNAPSHOT_WRITE':8*G*H};pruned=set();fused=set()
  for r in range(G):
   rowports={f'R{r}_CAST_F_WRITE':16*H,f'R{r}_CAST_F_READ':16*H,f'R{r}_MUL_F_WRITE':16*H,f'R{r}_MUL_F_READ':16*H,f'R{r}_SUM_F_WRITE':4*H,f'R{r}_SUM_F_READ':4*H,f'R{r}_COLLAPSE_BF16_WRITE':2*H,f'R{r}_HEAD_NORM_BF16_READ':2*H};ports.update(rowports)
   if r in conf:ports[f'R{r}_CONF_RAW_BF16_READ']=2*H
   privateF={k for k in rowports if '_F_'in k}
   if r not in needed:pruned.update(privateF|{f'R{r}_COLLAPSE_BF16_WRITE'})
   if r not in head:pruned.add(f'R{r}_HEAD_NORM_BF16_READ')
   if fuse and r in needed:fused.update(privateF)
   if fuse and r in head:
    fused.add(f'R{r}_HEAD_NORM_BF16_READ')
    if r not in conf:fused.add(f'R{r}_COLLAPSE_BF16_WRITE')
  removed=pruned|fused;left={k:v for k,v in ports.items()if k not in removed};cases.append(dict(G=G,H=H,head_unique_rows=sorted(head),ordered_head_multiset=ordered,K=len(ordered),confidence_raw_rows=sorted(conf),collapse_unique_rows=sorted(needed),fuse=fuse,baseline_ports=ports,pruned_ports=sorted(pruned),fused_remaining_ports=sorted(fused),remaining_ports=left,logical_union_credit=sum(ports[k]for k in removed),full_snapshot_retained=True,all_native_reader_order_RNG_or_epoch_gates_closed=False))
 payloads['post-draft-terminal-demand-and-interface-cases-v154.json']=dict(cases=cases,count=len(cases),domain='N4selectedtargetterminalrowwisecollapse/normgraphwithfullrawMTPsnapshotkept,optionaldeclaredrawconfidenceinputdemand. EveryBF16privatepoint/rawconfidence/orderedduplicateheadoccurrencepreserved unless explicitlyinternal. ActualU/K/confidencebranch/reader/error/hook/RNG/rowmap/gamma/epsilon/modeunbound.',raw_observer_scope='The confidence_raw_rows parameter is an abstract raw-collapsed observer demand. Draft confidence is one instance; no target confidence branch or configured target+draft combined graph is asserted.',old_full_collapse_norm_private_reference_per_row=76*H,head_logits_K_work_pruned_from_unique_U=False,physical_HBM_or_allocation_time_gain=False,overall_complete=False)
 payloads['post-draft-terminal-source-and-epoch-contract-v154.json']=dict(facts=['Draftinitialglobalrepeat,HCownerrows,collapseownerrowsandhead_normglobalrowsaredifferenttransportdomains. Source104callsare16DUMMY88PROPOSE,notactualepisode/phasebinding;None-metadatazero-outputbranchseparate.', '312draftattentionPost->FFNPreand208cross-layerFFNPost->nextHCPreusedistinctincomingpremix/currentcoeff/pre4/rawfuturePost. Firstrepeatedembedding/onehotpremixpropertydoesnotpersistaslinearchannelequalitythroughPostBF16updates.', 'DraftcontextcombinesTARGETentering-layerauxmeans,usesmainprojection/normandlayer-specificcontextWkv/norm/C/S/slots. Queryusesdraftembedding/currentHC/inputnorm. Earlycontextprecomputeanddelayedjointcontext-queryprojectionhaveoppositeretirement/readycosts;sameHdoesnotCSE.', 'FinaldraftlastFFNPosthasterminalcollapseinreviewedflowratherthaninventednextHCPre. RawcollapsedBF16confidenceinputdistinctfromroundedheadRMSinput. NativeheadRMSdoesnotattestoptionalconfidenceexecution/absence.', 'CurrentseedcomesfromearliersamplerderivedIDs. Currentproposal/Markov/confidence/keepandnexttargetacceptedstate/nextseedaredifferentgenerations;latekeepcannotretroactivelypruneitsownnoncausalproducer/metadata.', '24targetterminaltailsretainfullG4HMTPpublicationandsource-bound48rawCOMMrecords. CachedAPI-TASKabsenceforhiddenAGisnotzero-work;rawconnectionmembersincludecontroltasksnotuniqueComputeproof. HeadU/KselectionmayreduceROW-WISElatercollapse/normunderallreadersbutdoesnotdeletefullrawsnapshotorupstreamcache/state/auxrows.', 'Referencebackwardtwo-layerN4demand12288casesretained;neededinputrowsaretransitivequery/KV/state/auxclosure,notheadKalone. Realrouting/error/RNG/EPLB/collective/capture/statebranchesremainopen.'],generation_nodes=['SAMPLE0','SEED0','META0','DRAFT_BODY0','RAW_COLLAPSE0','MARKOV0','CONF0','KEEP0','VERIFY1','ACCEPTED1','SEED1'],generation_edges=[['SAMPLE0','SEED0'],['SEED0','META0'],['META0','DRAFT_BODY0'],['DRAFT_BODY0','RAW_COLLAPSE0'],['SEED0','MARKOV0'],['RAW_COLLAPSE0','MARKOV0'],['RAW_COLLAPSE0','CONF0'],['MARKOV0','CONF0'],['CONF0','KEEP0'],['KEEP0','VERIFY1'],['VERIFY1','ACCEPTED1'],['ACCEPTED1','SEED1']],illegal_retrocausal_edge=['KEEP0','META0'],source_contracts=['hc-draft-source-contract-v142.json','norm-terminal-source-contract-v146.json','post-pre-source-contract-v147.json'],new_source_read=False,new_device_work=False,overall_complete=False)
 contracts={'F28':'Draftrepeat/firstpremixinitialdomainandlaterPostPreversionedrawrepresentationscomposeonlywithactualglobal-to-ownerrows,pad/params/nativeBF16rounds/determinism. Faithfulinput+fullPostparamreplayisnewconsumer/scratch/work/lifetimegraph,notlinear-rankshortcutorunchangedoriginalGMABI.', 'F29':'DraftPostPre/queryinputnormusesnewdraftresidual;contextmain/KVnormusescombinedoldTARGETauxinputwithlayer-specificweights/positions/slots. Overlap/jointprojectionmustmeetbothready/retirementcontracts andallrawfuturePost/oldcontextreaders;noH5120CSE orseed/accepted-epochreuse.', 'F30':'IntermediatePostPreandterminaldraftPost/collapse/headNorm/Markov/confidencearedifferentbranches. KeeprawcollapsedBF16confidenceandoriginalrounds/seed/RNG/rows;latekeepdoesnotprunecurrentdecoder/metadata. Faithfulterminaldecodedrawmayfeedcollapseonlywithallrawreaders/newABIproof.', 'F35':'TargetlastFFNPost/fullrawMTPsnapshot/headcollapse/normandorderedlogitsdemandcomposewithoutinventingterminalnextPre. Fullrawsnapshotstaysunlessallobserverclosure;selectedUcollapse/norm+remaininginterfacefusionuniononce,keepconfidence/raw/Kduplicate/RNG/stateandbackwardcache/auxrowdemands.'}
 payloads['post-draft-terminal-composition-v154.json']=dict(clauses=[dict(id='REL154-%02d'%i,a='F03',b=k,contract=v)for i,(k,v)in enumerate(contracts.items(),1)],new_pairs=[['F03',k]for k in contracts],new_pair_clauses=4,new_groups=0,group_extensions=[['HG73-TAIL','F03']],group_reviews=['HG85-VERIFY-CONTEXT-DRAFT','HG93-HC-ATTENTION-FORK-DEADLINE','HG86-TERMINAL-QUERY-STATE-CUT'],existing_pair_refinement=['F03','F26'],all_runtime_gates_proved=False,whole_F03_or_pair_or_all_subsets_closed=False,speedup_additive=False,overall_complete=False)
 files=['post-attention-postcheck-result-v153p.json','post-attention-postcheck-correction-v153p.json','hc-draft-native-frontiers-v142.json','hc-draft-source-contract-v142.json','hc-draft-value-epoch-and-demand-witnesses-v142.json','norm-terminal-native-frontiers-v146.json','norm-terminal-source-contract-v146.json','norm-terminal-backward-demand-cases-v146.json','norm-terminal-value-row-epoch-witnesses-v146.json','post-pre-native-frontiers-v147.json','post-pre-source-contract-v147.json','precision-field-source-and-primary-contract-v148.json'];payloads['post-draft-terminal-inputs-v154.json']=dict(inputs=[pin(O/f)for f in files]+[pin(N/'verify_post_attention_v153p.py')]+[z['raw_database']for z in b['databases']]+[z['cached_native']for z in b['databases']],questions=['HowdodraftintermediatePostPreandtarget/draftterminalbranchescomposewithinit/context/cache/confidence/sampling?', 'Canfaithfulreplayretaininputvectors+fullPostparameterswithoutinvalidlinearBF16shortcut,andwhatmemory/work/ABIcostsremain?', 'Whatrow/private-interfaceworkcanbeprunedwhilefullMTP/rawconfidence/orderedhead/RNG/accepted-stateandbackwardcache/auxdemandremain?'],new_source_read=False,new_remote_or_device_activity=False,overall_complete=False)
 for name,payload in payloads.items():assert not(O/name).exists();write(O/name,payload)
 print(json.dumps(dict(draft_invocations=len(draft),draft_intra=cnt['draft_intra_edges'],draft_cross=cnt['draft_cross_edges'],source_ancestry=dict(cnt),target_tails=len(tails),unique_bindings=len(binding),role_bindings=sum(len(z['roles'])for z in draft)+sum(len(z['roles'])for z in tails),collectives=len(b['collectives']),representation_cases=len(rep),terminal_demand_cases=len(cases),rank_round_determinant=rankcase['rounded_determinant'],new_pairs=4,new_device_work=False)))


if __name__=='__main__':main()
