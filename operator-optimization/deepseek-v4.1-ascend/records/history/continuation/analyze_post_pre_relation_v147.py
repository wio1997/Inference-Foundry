"""Reprice Post/Pre raw versions, typed MoE pads and one interface/resource union."""
from pathlib import Path
from collections import Counter,defaultdict
from itertools import product
from fractions import Fraction
import hashlib,json,math,statistics,struct
from finalize_origin_coverage_v9 import N,O,W,read,write


def pin(p):
 p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def bf16(x):
 u=struct.unpack('>I',struct.pack('>f',x))[0];return struct.unpack('>f',struct.pack('>I',(u+32767+((u>>16)&1))&0xffff0000))[0]


def ports(m,p,h,e,aux):
 z={'X_STORE':8*m*h,'X_PRE_LOAD':8*m*h,'X_FUTURE_POST_LOAD':8*m*h,'Y_STORE':2*m*h,'Y_NORM_LOAD':2*m*h,'ZB_STORE':2*m*h,'ZB_SHARED_LOAD':2*m*h,'ZF_STORE':4*m*h,'ZF_ROUTER_LOAD':4*m*h,'LF_STORE':4*m*e}
 if p>m:z.update(ZB_PAD_READ=2*m*h,ZP_STORE=2*p*h,ZP_MC2_LOAD=2*p*h,LF_PAD_READ=4*m*e,LP_STORE=4*p*e,LP_TOPK_LOAD=4*p*e)
 else:z.update(ZB_MC2_LOAD=2*m*h,LF_TOPK_LOAD=4*m*e)
 if aux:z['AUX_X_LOAD']=8*m*h
 return z


def removed(m,p,h,e,aux,actions):
 out=set();mp={'POST_PRE_FORWARD':['X_PRE_LOAD'],'COLLAPSE_NORM':['Y_STORE','Y_NORM_LOAD'],'NORM_SHARED_FORWARD':['ZB_SHARED_LOAD'],'NORM_ROUTER':['ZF_STORE','ZF_ROUTER_LOAD'],'NORM_PAD_DIRECT':['ZB_STORE','ZB_PAD_READ']if p>m else[],'ROUTER_PAD_DIRECT':['LF_STORE','LF_PAD_READ']if p>m else[],'POST_AUX_FORWARD':['AUX_X_LOAD']if aux else[]}
 for a in actions:out.update(mp[a])
 return sorted(out)


def main():
 assert not(O/'REVIEW-CHECKPOINT-v147.json').exists()
 prof=read(O/'post-pre-native-frontiers-v147.json');b=prof['bindings'];root=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend';parts=[]
 for rel,a,z in [('models/deepseek_v41/model.py',269,329),('models/deepseek_v41/model.py',830,920),('models/deepseek_v41/model.py',1556,1592),('ops/fused_moe/prepare_finalize.py',255,296),('ops/fused_moe/fused_moe.py',228,242),('ops/fused_moe/fused_moe.py',390,419),('ops/fused_moe/router/fused_topk_router.py',155,210),('models/deepseek_v41/dspark.py',248,275)]:
  p=root/rel;parts.append(dict(**pin(p),first_line=a,last_line=z,source=p.read_text().splitlines()[a-1:z]))
 facts=[
  'Current target attention Post->FFNPre uses the earlier attentionPre attn_pre as incomingpremix;FFNPre currentcoefficients/ffn_pre remain separate. FFNPost->nextAttentionPre uses previousFFNPre ffn_pre,not currentnextcoefficients. Draft uses the same source decoder under distinct weights/banks. All 2416joinededges are source-key/native-ID joins,not actualpointer/parameter/clientepisode ornewtile-publication proof.',
  'F03 is Post->nextPre structural reorganization,not F02 inputRMS alone. Updated full4H residual is still a futurePost residual andcurrentcoefficient input. In a selected graph retaining raw4H GM publication,forwarding Post->Pre removes a chosen8MH inputport,not also8MH Xstore. Removing Xstore instead requires a separate legal retention/recompute/representation plus allreader/capture/progress proof.',
  '936targetcrossedges include48positiveEngram and72positiveaux sites,disjoint inthese savedroots;816lackthese positiveguards butdo not establish allreaderabsence. Auxmean observes old entering residual;Engram may update it before nextHCcoeff/Y/futurePost uses newversion. Lookup readiness may hoist,but hidden-dependent update cannotbe crossed asidentity. General configs canhavebothreaders anddifferent lifetimes.',
  'RmsNormCast produces rounded BF16 normalizedZB and FLOAT widening perpriorv63 contract;router consumes FLOAT,shared DynamicQuant consumes BF16,MC2 consumes prepared BF16. Bypassing BF16round or reusing an old raw/normalized input quantpacket acrossPostmix is not universally equivalent. CurrentY/stat/coeff/pre4/rawstate remain.',
  'NativeM/P casesare800/816x320,816/816x320,834/834x320. Only800/816 selecteddomain has3prepare padding calls forhidden/logits/mask:960exactsharedAPI PadV3/MemSetgroups,2880TASKsiblingsincluding960maskCast. RawPadV3 signatures identify logicalM->P operators,not MemSet destination/length/workspace/allocation. AllthreeM12 pendingMemSetregimes320each remainpending;original9/1352unchanged.',
  'Direct normproducer ofpaddedBF16 plane may expose its validM viewtosharedDynamicQuant whileMC2takesP;directrouterproducer may writefullP logitsincludingzero-tail. These are newbank/view/stride/guard/capture/error/ready protocols. Neither remaps sharedDQ fromMtoP nor provesquantizeridentityorreceiveremptytails. MC2 receiver assignmentA differs M*k;original mask/ID remap/TopK P/fullerror/RNG/EPLB/counter/peerprotocol remain.',
  '768interfacecases use7chosenactions×3geometries×auxflag in a declared materialized-raw-residual reference. RemoveuniqueportIDs once;Xstore/futurePostread remain. Actualalias/reader/newproducer/nativeprecision/capacitygates are not closed. Logicalinterfacecredit isnotphysicalHBM/time orindependentfusiongains.',
  'Retainingfull4Hrawrowpayload needs40960B perrow atH5120 before otherworkspace/weights/coeff/communication reserves,in that explicitfullrow-retentiondomain. This isnot a universalonchip bound:smaller Htiles,recompute,encoded/reduced demand are alternatives. FullHstatistics/amax needcross-tile information butnot an invented universal wholeoperatorbarrier.',
  'Newfusion holdingrawbanks acrossopaqueMC2 cancompete withcommunicationprogressbuffers;abstractwaitcycle is a counterexample toassumingfreeonchipretention,not an actualcurrentdeadlock. Sharedoverlap/delayedDown alternatives remain fromv141;no imagined MC2internal launchdeletions.',
  'Symbolicceilings restrictto declaredclassicaldenseprogram,matchingrequiredwork andvalidtypedcapacityuppers. NextHCcoeff2M(4H)24 androuter2MHE shareonecapacityonlyifmatchingnativeFLOAT/HF32/commonCube pool isproved. Shared+receiverINT8workcancombineononepool onlywithactualreceiverA andprecision/kernelownership. These gatesareunproved;physicalnumericbound/MFU/lowest remainsNOT_ESTABLISHED.'
 ]
 write(O/'post-pre-source-contract-v147.json',dict(parts=parts,facts=facts,prior_not_new_primitives=['v58Post/collapse/norm/Gram/referenceprogram','v63roundedBF16RmsNormCast/router','v141shared-overlap-vs-delayedDown','v146distinctnorms/aux/terminalstatecuts'],all_native_new_ABI_or_runtime_gates_proved=False,new_device_work=False,overall_complete=False))
 ge=Counter();cross=Counter();stats=Counter();observed=[];dur=[];scope_envelopes=defaultdict(list)
 for name in ['target_intra','target_cross','draft_intra','draft_cross']:
  for r in prof[name]:
   l,z=b[r['left']],b[r['right']];m=int(l['definition']['outputShapes_text'].strip('"').split(';')[0].split(',')[-3]);lo=l['TASK'];hi=z['TASK'];dur.append((name,lo['endNs']-lo['startNs'],hi['endNs']-hi['startNs'],hi['startNs']-lo['endNs']));observed.append(dict(kind=name,key=[r['database_index'],r.get('forward_API_rowid',r.get('source_order_ordinal')),r.get('block',r.get('to_block'))],M=m,left_TASK=lo['task_record_rowid'],right_TASK=hi['task_record_rowid'],same_native_stream=lo['streamId']==hi['streamId'],observed_device_gap_ns=hi['startNs']-lo['endNs'],actual_new_tile_gain_ns=None))
   if name=='target_intra':
    p0=int(b[r['consumers']['MC2']]['definition']['inputShapes_text'].strip('"').split(',')[0]);ge[m,p0]+=1;tt=[b[i]['TASK']for i in [r['right'],r['norm'],r['consumers']['ROUTER']]];scope_envelopes[m,p0].append(max(t['endNs']for t in tt)-min(t['startNs']for t in tt))
   if name=='target_cross':cross[bool(r['Engram_positive_APIs']),r['aux_mean']is not None]+=1
 def st(v):return dict(count=len(v),min=min(v),median=statistics.median(v),max=max(v))
 write(O/'post-pre-observed-frontiers-v147.json',dict(records=observed,geometry=[dict(M=m,P=p,frontiers=n)for(m,p),n in sorted(ge.items())],targetcross_guard_counts=[dict(Engram=e,aux=a,edges=n)for(e,a),n in sorted(cross.items())],task_only_by_edgekind={k:dict(left_duration_ns=st([x[1]for x in dur if x[0]==k]),right_duration_ns=st([x[2]for x in dur if x[0]==k]),recorded_gap_ns=st([x[3]for x in dur if x[0]==k]))for k in ['target_intra','target_cross','draft_intra','draft_cross']},TASK_durations_or_gaps_not_causal_request_gain_or_allreaderproof=True,new_native_operand_or_phase_promotions=0,new_device_work=False,overall_complete=False))
 acts=['POST_PRE_FORWARD','COLLAPSE_NORM','NORM_SHARED_FORWARD','NORM_ROUTER','NORM_PAD_DIRECT','ROUTER_PAD_DIRECT','POST_AUX_FORWARD'];cases=[];h=5120;e=384
 for(m,p),n in sorted(ge.items()):
  for aux,mask in product([False,True],range(128)):
   actions=[a for i,a in enumerate(acts)if mask>>i&1];base=ports(m,p,h,e,aux);rm=removed(m,p,h,e,aux,actions);credit=sum(base[k]for k in rm);remain={k:v for k,v in base.items()if k not in rm};headline=sum({'POST_PRE_FORWARD':16*m*h,'COLLAPSE_NORM':4*m*h,'NORM_SHARED_FORWARD':4*m*h,'NORM_ROUTER':8*m*h,'NORM_PAD_DIRECT':4*m*h if p>m else 0,'ROUTER_PAD_DIRECT':8*m*e if p>m else 0,'POST_AUX_FORWARD':8*m*h if aux else 0}[a]for a in actions)
   cases.append(dict(M=m,P=p,H=h,E=e,aux=aux,actions=actions,baseline_port_bytes=base,removed_ports=rm,remaining_port_bytes=remain,logical_union_credit=credit,baseline_selected_port_sum=sum(base.values()),remaining_selected_port_sum=sum(remain.values()),incorrect_standalone_headline_sum=headline,headline_overcredit=headline-credit,all_native_runtime_or_reader_gates_proved=False))
 write(O/'post-pre-interface-union-cases-v147.json',dict(cases=cases,count=len(cases),action_universe=acts,domain='Chosenlogicalmaterializationgraph retains raw4H GMstoreandfuturePostread;noEngrammutationinsidecoregraph. Auxflagis sameversionreader;Engramrequiresseparateversionhandoff. A differentretention/representationmayremovestoreonlybehindnewgates. No actual HBM/resource/time/',physical_gain_or_lowerbound=False,new_device_work=False,overall_complete=False))
 bounds=[]
 for(m,p),n in sorted(ge.items()):
  coeff=2*m*4*h*24;router=2*m*h*e;shared=6*m*h*2304
  bounds.append(dict(M=m,P=p,H=h,E=e,shared_I=2304,frontiers=n,classical_next_HC_coefficient_ops=coeff,classical_router_ops=router,classical_shared_INT8_GEMM_ops=shared,classical_same_mode_coeff_router_union=coeff+router,classical_receiver_two_GEMM_ops='6*A_receiver*H*I_expert',capacity_conditional_expressions=['L>=max(W_coeff/U_coeff_upper,W_router/U_router_upper,W_shared/U_shared_upper,W_receiver/U_receiver_upper)','If actualsameFLOAT_HF32mode/pool: L>=(W_coeff+W_router)/U_common_upper','If actualsameINT8pool/receiverprecision: L>=(W_shared+W_receiver)/U_INT8_common_upper'],measured_native_Pre_Norm_Router_TASK_envelope_ns=st(scope_envelopes[m,p]),selected_current_tasks_not_coefficient_microtime_or_request_latency=True,necessary_physical_IO_or_wire_lower_bytes=None,valid_actual_physical_capacity_upper=None,time_lower_bound_ns=None,measured_comparable_time_ns=None,measured_over_bound_ratio=None,theory_state='SYMBOLIC_CEILING_IN_DECLARED_CLASSICAL_PROGRAM;ACTUAL_PHYSICAL_NOT_ESTABLISHED',not_all_equivalent_algorithms_bound=True))
 retain=[dict(tile_rows=k,H=h,raw4H_BF16_payload=8*k*h,actual_available_payload_budget=None,otherworkspace_weights_coeff_communication_reserves_unpriced=True,fullrow_retention_only_not_universal=True)for k in [1,2,4,16,64,800,816,834]]
 write(O/'post-pre-symbolic-resource-and-retention-v147.json',dict(geometry=bounds,retained_payload_cases=retain,work_domain='Fixedclassicaldensecoefficient/routerandshared/twoexpertGEMMs,neededrow/assignmentswithnoalgebraicweightfold/prune/recompute/statreuse;notanallalgorithmslowerbound. Exactnativearithmetic/precision/mapping/resourcepool andreceiverAunknown. Initial/finalphysicalstoragecutsnotattested.',validity='Necessarywork/validcapacityupperandnonduplicatecommonpoolworkcanboundtime;logicalremovedportsarenotcompulsoryphysicalbytes. Allowresourceoverlap via max,not sumstandalonekernelbounds. No optimal/feasiblescheduletimeisusedaslowerbound.',new_device_work=False,overall_complete=False))
 a=[Fraction(1),Fraction(1,512)];a1=[Fraction(1),Fraction(3,1024)];bias=[-Fraction(255,256),Fraction(0)];quant=lambda v:[round(Fraction(127)*x/max(abs(y)for y in v))for x in v];v=[x+y for x,y in zip(a,bias)];v1=[x+y for x,y in zip(a1,bias)];assert quant(a)==quant(a1)and quant(v)!=quant(v1)
 post=[1+1/256,-1.,0.,0.];weights=[.5]*4;rounded=list(map(bf16,post));old=bf16(sum(x*y for x,y in zip(rounded,weights)));skip=bf16(sum(x*y for x,y in zip(post,weights)));assert old!=skip
 write(O/'post-pre-round-quant-version-progress-witnesses-v147.json',dict(Post_BF16_boundary=dict(Post_unrounded=post,Post_rounded=rounded,incoming_premix=weights,original_collapse_BF16=old,wrong_skip_Post_round_collapse_BF16=skip,native_kernel_or_frozen_coefficients_attested=False),quant_not_affine_homomorphism=dict(old0=list(map(float,a)),old1=list(map(float,a1)),same_old_INT8=quant(a),same_old_scale=float(Fraction(1,127)),same_Post_branch=list(map(float,bias)),Post0=list(map(float,v)),Post1=list(map(float,v1)),new_INT8_0=quant(v),new_INT8_1=quant(v1),same_new_scale=float(Fraction(1,256*127)),reference_policy='Constructedfiniterationalamax127/RNE;notnativeDynamicQuant/TRUNC/zero/nonfinite/FP16policyattestation',old_packet_sufficient=False),aux_Engram_versions=dict(rawX0=[1.,1.],Engram_update=[0.,2.],rawX1=[1.,3.],aux_of_old=1.,wrong_aux_of_new=2.,next_pre_input='X1',early_pre_on_X0_legal=False,actual_hook_values_attested=False),finite_progress=dict(wait_edges=[['MC2_COMPLETE','COMM_BUFFER_AVAILABLE'],['COMM_BUFFER_AVAILABLE','RAW_BANK_RELEASE'],['RAW_BANK_RELEASE','MC2_COMPLETE']],required_progress_reservation_or_alternative=True,actual_native_or_current_deadlock=False),scope='Finite/referenceBF16/rational/abstractversionandwait-graphcounterexamples;noactualtensor/nativeparity/physicalgain.',new_device_work=False,overall_complete=False))
 pairs=[['F03',x]for x in ['F20','F21','F22','F24','F27','F31']];contracts=[
  'Post->FFNPre/norm/router fusion retains roundedPost/Y/ZB/FLOATwidening,earlierpremix/currentcoeff outputs andactualM/P/IDs/masks/fullTopK/error/EPLBrowdomains;onevalueinterface/resourceunion,notoldquantorpre-attnZreuse.',
  'Post/nextHCearlyhandoff andopaqueMC2pipeline need receiverA/capacity/peerquant/capture/currentpre4/futurePost/rawbankdeadlines. Retainingrawbanksmustreservefiniteprogress;notM*kreceiverwork orimaginedinternal launchdeletion.',
  'Norm/sharedDQ/Down/Merge/Post/nextPre compose onlywithoriginalBF16rounds/full-Hamax/INT32-CVownership;sharedM versusMC2P needsnewvalid-view/paddedproducerABI,notunchangedquantpacketreuseoradditiveprivatecaps.',
  'AcrossFFNPost/nextPre,EngramlookupmayhoistbutupdatedhiddenX1mustfeedcurrentcoeff/collapse/futurePost;oldX0aux andbank readersstay.48positivehooksnotallparams/cache/lifetimeclosure;fullrawstorecannotbeerasedbysingleforwardedread.',
  'Post/Auxmeanproducer mayavoidchosenrawold-versionreadwhileemittingdistinctaux andupdatedresidual/Prevalues;72positiveauxedgesretainrawfuturePost/coeff andtransport/allreader gates. Noaux=Y/Zornew-version substitution.',
  'Norm/routerpaddedproducer andmask/IDpublication mayremoveonechosenmaterializationunderactualM/P/epoch/false-tail/capture/progresscontracts.960exactPadV3/MemSetshared-APIgroupsdo notbindMemSetaddresses/length/workspace;all9/1352staypending.'
 ];ext=[['HG74-AUX-CONTEXT','F03'],['HG76-TOKEN-PREDICATES','F03'],['HG85-VERIFY-CONTEXT-DRAFT','F03']]
 write(O/'post-pre-composition-v147.json',dict(clauses=[dict(id='REL147-'+str(i+1).zfill(2),a=a,b=b,contract=c)for i,((a,b),c)in enumerate(zip(pairs,contracts))],new_pairs=pairs,group_extensions=ext,new_groups=0,all_runtime_gates_proved=False,whole_pair_or_all_subsets_closed=False,speedup_additive=False,overall_complete=False))
 names=['post-pre-native-frontiers-v147.json','hc-sp-eager-native-frontiers-v134.json','hc-moe-native-frontiers-v141.json','hc-moe-source-contract-v141.json','hc-draft-native-frontiers-v142.json','HC-fusion-split-reorganization-variants-v58.json','HC-work-traffic-and-reference-bounds-v58.json','router-vector-numerical-contract-v63.json','norm-terminal-postcheck-result-v146p.json','operator-analysis-coverage-ledger-v146.json','candidate-relation-clauses-v146.json','norm-terminal-source-contract-v146.json']
 write(O/'post-pre-inputs-v147.json',dict(inputs=[pin(O/n)for n in names],questions=['WhichPost/Pre rawversionandpremixprovidersfeedtheMoE/shared/draftsuccessors?','Whichlogicalportscanbe removed oncewhilelate rawPost/aux/Engramreaders remain?','Can typedM800/P816 Pad/MemSetpartnersorquantpacketidentityactuallyjustifyfusionandphysicalceilings?'],profile_first=True,target_source_scope='DirectHC/Post/Pre/norm/MoEprepare/quant/router/aux/Engram/draftcallers,notframeworksurvey',new_device_work=False,overall_complete=False))
 print(json.dumps(dict(status='SCOPED_POST_PRE_MOE_REVIEW',unique_TASKs=prof['unique_TASK_bindings'],Post_Pre_edges=2416,pad_groups=960,pad_sibling_TASKs=2880,interface_cases=len(cases),new_pairs=6,new_native_operand_or_phase_promotions=0,new_device_work=False,overall_complete=False)))


if __name__=='__main__':main()
