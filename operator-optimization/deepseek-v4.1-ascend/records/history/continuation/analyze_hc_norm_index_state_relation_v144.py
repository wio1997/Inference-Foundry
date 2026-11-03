"""Input-RMS to typed Index/C2 consumers, pair algebra and value retirement."""
from pathlib import Path
from itertools import product
from collections import Counter
import json,hashlib,math,struct
from finalize_origin_coverage_v9 import N,O,W,read,write


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]


def bf16(x):
    u=struct.unpack('<I',struct.pack('<f',x))[0];return struct.unpack('<f',struct.pack('<I',(u+32767+((u>>16)&1))&0xffff0000))[0]


def pair(a,b,x,y,one=False,complement=False):
    a,b,x,y=map(f32,[a,b,x,y]);mx=max(a,b)
    if one:
        e=f32(math.exp(f32(min(a,b)-mx)));d=f32(1+e);p=f32(1/d);q=f32(e/d)
        if complement:q=f32(1-p)
        p0,p1=(p,q)if a>=b else(q,p)
    else:
        e0=f32(math.exp(f32(a-mx)));e1=f32(math.exp(f32(b-mx)));d=f32(e0+e1);p0,p1=f32(e0/d),f32(e1/d)
    return dict(p=[p0,p1],out=f32(f32(p0*x)+f32(p1*y)))


def schedule(p,early=False,norm_fork=False,reprice=False):
    z=max(p['hc_y'],p['hc_coeff'])+p['input_norm'];q=z+p['query']+(p['extra']if reprice else 0);cast=max(z if early else q,p['metadata'])+1;proj=cast+p['projection']+(p['extra']if reprice else 0)
    old_read=max(proj,p['old_generation'])+1;pool=old_read+p['pool'];ring=(old_read if norm_fork else pool)+p['ring'];norm=(pool if norm_fork else max(pool,ring))+p['latent_norm'];endpoints=norm+p['key_long'];qli=max(q+2,endpoints)+2;post=max(p['hc_coeff'],q+3,endpoints+3,qli+3)
    return dict(Z=z,Q=q,CAST=cast,PROJ=proj,OLD_READ=old_read,POOL=pool,RING=ring,NORM=norm,ENDPOINTS=endpoints,QLI=qli,Post_input=post,Z_reader_sufficient_release=max(q,cast),FLOAT_input_sufficient_release=proj,projected_KVScore_sufficient_release=max(pool,ring),next_ring_read_deadline=post+p['future_gap'],future_ring_deadline_satisfied=ring<=post+p['future_gap'])


def main():
    assert not(O/'REVIEW-CHECKPOINT-v144.json').exists()
    native=read(O/'hc-norm-index-state-native-joins-v144.json');root=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend';parts=[]
    for rel,a,b in [('models/deepseek_v41/model.py',887,920),('attention/dsa_v41.py',350,385),('attention/dsa_v41.py',395,437),('attention/dsa_v41.py',503,565),('models/deepseek_v41/indexer.py',106,125),('models/deepseek_v41/indexer.py',167,179),('models/deepseek_v41/indexer.py',253,260),('models/deepseek_v41/compressor.py',16,33),('models/deepseek_v41/compressor.py',55,71)]:
        p=root/rel;parts.append(dict(**pin(p),first_line=a,last_line=b,source=p.read_text().splitlines()[a-1:b]))
    p=N/'state-engram-operator-source-v62/vllm_ascend/ops/triton/compressor/compressor_triton.py'
    for a,b in [(85,111),(266,320),(324,364),(647,721)]:parts.append(dict(**pin(p),first_line=a,last_line=b,source=p.read_text().splitlines()[a-1:b]))
    facts=[
      'InputRMS Z is roundedBF16 H5120. QA/IndexWeight use thisZ inqueryrowdomain;IndexQ uses separatelynormalizedQR1280 withnativeDQ beforeINT8projection. C1 usesZBF16,whileC2 widensZ toFLOAT thenuses distinctFLOATWkv/Wgate. These arethreeinput/value/precisiondomains,not a single same-H wideGEMM.',
      'A QA1280+Weight32 panel may composewithinputRMS/newcarrier onlyaftertheactualsamequeryrow andweight/round/stridecontract isproved. CombiningitwithC2 requiresmixedBF16/FLOAT matrixmodes androwmaps;castingallweightstoBF16orusingHCraw4H asinputchangescontract.',
      'C2 pairpool isfeaturewise softmax overtwooriginalrequestpartners usingFP32projectedscoreandK. Multiplyingnormalizedinputbyacommonfactor changesbothscoresandvalues;softmaxtemperaturechangesdirection,sofollowingRMScannotgenerallycancelinputRMS. Different per-rowfactors orold-ring/currentmixstronger.',
      'ForfiniteRATIO2 reference scores,oneexp(min-max),twoseparateprobabilities1/(1+e) ande/(1+e),restoredoriginalroworder suffice inoneFLOATmodel. Replacingthesmallerprobabilitywith1-pmax losesnonzeroswhenpmaxroundsto1. Nativeexp0/FTZ/ties/NaN/Inf/reduction andactualcompilerinstructions mustbeverified;noactualSFUorlaunchsavinginferredfromsource.',
      'NativeTOKEN_ALIGNEDpoolstoresBF16completions atoriginalcompletion-tokenrows,zeroesotherrows thenfullT512RMS/bias follows. Fusingor specializingpair/norm mustretainpoolBF16round,512stat,allnoncompletionoutputs/rstd/bias andoriginalrequestseg/page/historybranches. Currentinputprojectionvalidrowunionremainsfull byv137.',
      'C2 oldFP32ring history is alreadyprojectedfromitsownroundedZ generation. Missingcurrentpairpartnerscome fromcurrentownerprojection,notstalehistory;SPshardsare notnewrequestsegments. PooloutputBF16orpostnormlatentcannotreconstructfullFP32tailstate.',
      'Z canlogicallyretire afteritsdirectreaders QA/SWA/Weight/C1/C2cast finish,notautomaticallyafterIndex/attention/ringoutputscomplete. ActualreadfrontiersmaybefinerthanTASKcompletion. C2FLOATinput lasts throughtwoGEMMreads;projectedKV/scorelastthroughpool AND anydeferredringupdate. Allotherreaders/physicalaliases/queuedwriters/capturelivenessarestillgates.',
      'Thus allowingpoolednorm/index/longcacheworkbeforedeferredringcommitdoesnotbyitselfretireprojectedKV/score. PreservebuffersorwriteexactFP32tailinline/recomputeunderanewcontract;statecommitstillprecedesactualnextgenerationold-ringreads. Fullringcapacityandoriginalcachebudgetunchanged.',
      'IndexKey128nativeRMS/RoPE/DynamicQuant FLOATscale->FP16cache differsfromIndexQueryper-head128RoPE/customquantFP16-beforeINT8. IndexWeight BF16->FLOAT/64->FP16andQLIproductremain. Inputnorm cannotfoldthroughthelatent512norm/IndexWk/keynorm/statepoolasoneprojection.',
      'LongKVreadsnormalizedlatent512tail64 whileIndexWkneedstheoldunrotatedvalue. Preserveper-rangeWAR/versionedprefix+tail/CSlastreaders anddistinctIndexK/Longslotepochs. NewinputnormcarriercompletiondoesnotretireHCrawX/currentcoeff/Postresidual/pre4 orC2futurestate.',
      'EarlyC2productionfromZ canstartbeforecurrentQuery/Indexercontroljoin onlyifcorrectcompressormetadata,oldringgenerationandcube/vector/queueprogressareprovided. Currenttaskintervalsareobservations,notrecoverabletime. Pair-specialization,cast/jointGEMM,inputnormfusion andstatecuts shareoneinterface/resourcegraph.'
    ]
    write(O/'hc-norm-index-state-source-contract-v144.json',dict(parts=parts,facts=facts,prior_answers=['v106/v139QAWeight/IndexQtypedinputs andscales','v135/v136C1C2key/ring/WAR','v137allvalidcurrentprojectionrowunion andfullTnorm','v140latentversions/CSreaders','v143inputRMSstat/BF16/carrieralternatives'],new_device_work=False,all_runtime_or_native_gates_closed=False,overall_complete=False))
    observed=[];geom=Counter();c2stats=Counter()
    for r in native['records']:
        s=r['stages'];m=int(s['INPUT_NORM']['definition']['inputShapes_text'].strip('"').split(',')[0]);q=int(s['QA']['definition']['inputShapes_text'].strip('"').split(',')[0]);t=int(s['SWA_KV']['definition']['inputShapes_text'].strip('"').split(',')[0]);geom[r['mode'],m,q,t]+=1
        row=dict(key=[r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']],mode=r['mode'],M=m,Q=q,T=t,inputRMS_before_Weight=s['INPUT_NORM']['TASK']['endNs']<=s['WEIGHT']['TASK']['startNs'],QRnorm_before_IndexQ=s['QA_NORM']['TASK']['endNs']<=s['INDEX_Q']['TASK']['startNs'],actual_Z_or_QR_quantizer_equal_or_state_epoch_bound=False)
        if r['mode']=='C2':
            row.update(QueryRoPE_before_C2cast=s['QUERY_ROPE']['TASK']['endNs']<=s['C2_INPUT_CAST']['TASK']['startNs'],IndexRoPE_before_C2cast=s['INDEX_ROPE']['TASK']['endNs']<=s['C2_INPUT_CAST']['TASK']['startNs'],ringupdate_before_poolnorm=s['CACHE_RING_UPDATE']['TASK']['endNs']<=s['CACHE_POOL_NORM']['TASK']['startNs'],IndexQuant_intersects_C2KV=max(s['QUERY_QUANT']['TASK']['startNs'],s['C2_KV']['TASK']['startNs'])<min(s['QUERY_QUANT']['TASK']['endNs'],s['C2_KV']['TASK']['endNs']))
            for k in ['QueryRoPE_before_C2cast','IndexRoPE_before_C2cast','ringupdate_before_poolnorm','IndexQuant_intersects_C2KV']:c2stats[k]+=row[k]
        observed.append(row)
    write(O/'hc-norm-index-state-observed-v144.json',dict(records=observed,C2_recorded_orders=dict(c2stats),scope='Source/CID keys/currentTASKorder only;noactualreadfrontier/alias/criticalpathornewphysicalgain.',new_device_work=False,overall_complete=False))
    def pool(ks,sc):
        return [sum(math.exp(sc[i][h]-max(x[h]for x in sc))*ks[i][h]for i in range(2))/sum(math.exp(sc[i][h]-max(x[h]for x in sc))for i in range(2))for h in range(2)]
    k=[[1.,0.],[0.,1.]];sc=[[1.,1.],[0.,0.]];base=pool(k,sc);scaled=pool([[2*a for a in row]for row in k],[[2*a for a in row]for row in sc]);mixed=pool([k[0],[2*a for a in k[1]]],[sc[0],[2*a for a in sc[1]]])
    ref=pair(0,-20,0,1e8);bad=pair(0,-20,0,1e8,True,True);assert ref['out']>0 and bad['out']==0
    write(O/'hc-norm-index-state-pool-and-precision-witnesses-v144.json',dict(
      pool_not_homogeneous=dict(K=k,scores=sc,pooled=base,common_input_factor=2,scaled_pooled=scaled,original_direction_ratio=base[0]/base[1],scaled_direction_ratio=scaled[0]/scaled[1],following_scalar_RMS_cannot_restore_direction=True),
      old_ring_current_factor=dict(old_K=k[0],old_score=sc[0],current_K=k[1],current_score=sc[1],old_state_kept=True,original=base,current_factor2=mixed,one_global_rescale_suffices=False),
      small_probability=dict(scores=[0,-20],values=[0,1e8],reference=ref,one_minus_pmax=bad,BF16_reference=bf16(ref['out']),BF16_complement=bf16(bad['out']),frozen_actual_values_or_native_policy=False),
      mixed_weight_precision=dict(Z=1.,FLOAT_weight=1+1/512,BF16_weight=bf16(1+1/512),FLOAT_reference_output=1+1/512,BF16_downgraded_output=1.,actual_HF32_accumulation_policy_attested=False),
      lifetime_fence=dict(Z_reader_end=5,FLOAT_projection_read_end=9,pool_done=12,ring_done=20,projected_plane_discard_after_pool_wrong=True,Z_need_not_wait_ring=True,next_actual_old_ring_read=16,ring_commit_meets_fixed_future_deadline=False),
      witness_scope='Constructed typed/reference algebra and abstractpage-generation deadlines;not actualloadedweights/tensors/nativeprecision/capacity/latency.',new_device_work=False,overall_complete=False))
    paircases=[];complement_bad=0
    for a,b,x,y in product([-80,-20,-2,0,2,20,80],[-80,-20,-2,0,2,20,80],[-1,0,1,1e8],[-1,0,1,1e8]):
        ref=pair(a,b,x,y);one=pair(a,b,x,y,True);bad=pair(a,b,x,y,True,True);assert ref==one;complement_bad+=ref!=bad;paircases.append(dict(scores=[a,b],values=[x,y],reference=ref,one_exp=one,complement=bad))
    write(O/'hc-norm-index-state-pair-algebra-v144.json',dict(cases=paircases,count=len(paircases),reference_one_exp_matches=len(paircases),complement_changes=complement_bad,scope='FiniteFLOAT reference withmath.exp roundedtoFLOAT,exp0=1,originaltwoindependentdivisionprobabilities andoriginalrowreductionorder. Nativeapproximation/FTZ/ties/NaN/Inf/opcodecount/compiledexp0foldingnotattested;notanactualSFUlowerboundorspeedup.',new_device_work=False,overall_complete=False))
    cases=[];classes=Counter();admissible=Counter()
    for vals in product([3,17],[2,7],[8,30],[4,25],[3,8],[2,27],[3,8],[3,13],[0,40],[0,25],[0,20],[0,9,35]):
        p=dict(zip(['hc_coeff','input_norm','query','projection','pool','ring','latent_norm','key_long','metadata','old_generation','future_gap','extra'],vals),hc_y=2)
        ans={k:schedule(p,e,n,pr)for k,e,n,pr in [('baseline',False,False,False),('early_C2_only',True,False,False),('norm_state_fork_only',False,True,False),('joint_ideal',True,True,False),('joint_repriced',True,True,True)]};a,b=ans['joint_repriced']['Post_input'],ans['baseline']['Post_input'];state='FASTER'if a<b else'SLOWER'if a>b else'TIED';classes[state]+=1;admissible['FUTURE_READY'if ans['joint_repriced']['future_ring_deadline_satisfied']else'FUTURE_VIOLATED']+=1;admissible['READY_'+state if ans['joint_repriced']['future_ring_deadline_satisfied']else'VIOLATED_'+state]+=1;cases.append(dict(parameters=p,**ans))
    write(O/'hc-norm-index-state-readiness-DAGs-v144.json',dict(cases=cases,count=len(cases),joint_repriced_classes=dict(classes),future_deadline_classes=dict(admissible),scope='ChosenC2positive-query full-frontier reference;currentHCcomplete beforeinputNorm inallvariants,noF01forkcredit. EarlyC2readsroundedZonlywithmetadata andold-generationavailability;poolold-read frontier ischosen sufficientreadcompletion,nouniversalwholepoolbarrier. Norm/key-long composite preservesoldlatentWAR. Timesarbitrary,extra reprices query/C2projection. Retirements arelogicalvalue sufficientfrontiers withallotherdirectZreaders/aliases gated.',endpoint='AttentionPost input-ready plusseparateactualnext-ring-read deadline;violatedfuturecasesnotadmissiblegains',physical_bound_or_actual_gain=False,new_device_work=False,overall_complete=False))
    work=[]
    for(mode,m,q,t),count in sorted(geom.items()):
        h=5120;d=512;is2=mode=='C2';work.append(dict(mode=mode,M=m,Q=q,T=t,frontiers=count,H=h,input_BF16_Z_row_content=2*h,IndexWeight_QA_shared_row_input_reference=2*q*h,
            C2_two_FLOAT_matrix_classical_ops=4*t*h*d if is2 else None,C2_separate_FLOAT_input_interface=12*t*h if is2 else None,C2_joint_existing_FLOAT_plane_read_credit=4*t*h if is2 else None,C2_after_joint_FLOAT_plane_interface=8*t*h if is2 else None,C2_cast_joint_full_union_reference=12*t*h if is2 else None,
            C2_projected_FP32_KVScore_plane_content=8*t*d if is2 else None,C2_pool_completions_BF16_content='1024*E'if is2 else None,C2_required_future_projected_tail_content='4096*U plusoriginalbranch/ownerhalos'if is2 else None,
            C2_valid_current_projection_row_elision_from_pool_norm_or_even_state=0 if is2 else None,tagged_all_QA_Weight_SWA_C2_panel_row_value_content=2*1312+2*512+4*1024 if is2 else None,
            panel_precision_rule='QA+Weight1312BF16 separate1280norm/32weightpostscale;SWA512BF16;C2KVScore1024FLOAT. Differentinputtypes andweightmodes neednewmixedkernel,notoneordinaryBF16GEMM.',
            all_Z_reader_or_alias_state_observer_gates_closed=False,actual_E_B_U_or_physical_HBM_time_or_allocation=None))
    write(O/'hc-norm-index-state-work-and-lifetimes-v144.json',dict(geometry=work,interface_rule='Declaredvalueinterfacesonly. FLOATplane oldwrite4TH+twoinputreads8TH=12TH. JointGEMMonexistingplane saves4TH,remaining8TH. Cast+jointelisioncanremoveoneunion12TH iflegal;do not add4THagain. IndexWeightQAsharedinputcreditoverlapsv106/v143 andrequiresactualsharedrows. Tagged7744Brow excludesmetadata/headers/statehalos anddoesnotproveallrowsneedallpanels/newcarrierfit.',retirement_rule='LastdirectZread/lastFLOATinputGEMMread/lastprojectedKVScorepoolANDringread aredistinctlogicalfrontiers;TASKcompletion isonesufficientchosenreadbound,notactualminimum. HCrawXcoeff/Post/pre4/aux/C2oldstate andqueuedwritersremain.',bound_contract='Classicaldense/memorycontent referenceonly. PhysicalLneedsprovablynecessaryworkandvalidjointcapacityupper,actualreadsets/traffic/epochs;maxnonduplicatedresourcebounds,notstandalonesums. NoallmoduleMFU/lowestestablished.',new_device_work=False,overall_complete=False))
    pairs=[['F02',x]for x in ['F09','F10','F11','F17','F18','F19']];contracts=[
      'InputRMSZ/QA/QRnorm/nativeDQ/IndexQ aretypedserialvalues. Inputnormfusion orcarrierreuse mustretainQR1280BF16/fullrowquant/scales,IndexQRoPE/customquant andallQRreaders,notoneH5120projection.',
      'InputRMS+QA/Weight1312shared-rowpanelconditional;WeightusesZwhileIndexQusesQR. KeepseparateQA1280norm,Weight32BF16/FLOAT/FP16/product,packedstrides andcurrentquery/metadata epochs;creditsalreadyoverlapv106/v143.',
      'InputRMS-to-latent-to-IndexKeyincludesC1linear/norm orC2pool/norm,thenIndexWk/key128norm/RoPE/nativeDQscale/store. Preserveoldlatentread,distinctquery/keyquantpolicies/noncompletion/slotgenerations;cannotfoldallnorms.',
      'InputRMSBF16Z widenedFLOAT feedsC2KV/scoretwoFLOATmatrices. Joint1024projection/castinterfacecutsrequireoriginalprecision/round/rowmaps,newmixedpanelhandoff andoneunioncredit;otherZreadersandfuturetailstate remain.',
      'C2scoretemperature meansdownstreampoolnormdoesnotcancelinputRMS. Ratio2one-exp specialization possibleonlyunderoriginalfinite/nativeprobability/round/nonfinitecontracts;1-pmaxcounterexample. PoolBF16/fullTnorm/bias/readers andallcurrentvalidprojectionrowsretained.',
      'InputZ,FLOATGEMMinput,projectedKVScore,normalizedlatentIndex/LongforkandfutureFP32ring havedifferentretirements. EarlyC2/normforkmustretainprojectedtailuntilcommit/nextreader,oldRingWAR/CS/slotepochs andHCrawXcoeff/Post/pre4;notwhole-indexorupdate->normbarrier.'
    ]
    ex=[['HG73-C2',x]for x in ['F02','F09','F10','F11']]+[['HG87-KV-CURRENT-READ-AND-COMMIT',x]for x in ['F09','F10','F17']]
    write(O/'hc-norm-index-state-composition-v144.json',dict(clauses=[dict(id='REL144-0'+str(i+1),a=a,b=b,contract=c)for i,((a,b),c)in enumerate(zip(pairs,contracts))],new_pairs=pairs,new_pair_clauses=6,group_extensions=ex,new_groups=0,all_runtime_gates_proved=False,whole_pair_or_all_subsets_closed=False,speedup_additive=False,overall_complete=False))
    names=['hc-norm-index-state-native-joins-v144.json','hc-norm-projection-native-frontiers-v143.json','hc-norm-projection-source-contract-v143.json','hc-norm-projection-stat-and-round-witnesses-v143.json','hc-index-query-weight-native-producers-v139.json','hc-index-typed-input-and-round-contract-v139.json','hc-index-scalar-round-and-cycle-witnesses-v139.json','hc-sparse-eager-native-frontiers-v138.json','hc-c1-native-consumers-v135.json','hc-c2-native-consumers-v136.json','hc-c2-source-and-state-contract-v136.json','hc-c2-pair-owner-and-state-witnesses-v136.json','hc-c2-current-row-demand-proof-v137.json','hc-c2-pool-norm-delivery-and-work-v137.json','hc-c2-norm-state-source-contract-v137.json','hc-rope-latent-version-work-and-lifetimes-v140.json','index-projection-composition-v106.json','candidate-relation-clauses-v143.json','hc-norm-projection-postcheck-result-v143p.json']
    write(O/'hc-norm-index-state-inputs-v144.json',dict(inputs=[pin(O/n)for n in names],questions=['WhichinputRMSZ/QR/latent valuesandprecisionsmayjointlyfeedIndex/C2withoutfoldingnonlinearrounds?','CanC2inputscalardropthroughsoftmaxpool+norm,andwhatRATIO2specializationpreservessmallprobabilities?','WhenmayZ,FLOATinput,projectedKVScoreandringstateeachretireifnorm/cacheworkadvances?','Howshouldcast/jointprojection/panel/ownerstatecutsbeunifiedwithoutdoublecountingorviolatingfutureepochs?'],profile_first=True,new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='MODELED',frontiers=192,native_TASK_union=5064,pair_cases=len(paircases),complement_changes=complement_bad,DAGs=len(cases),classes=classes,future=admissible,new_pairs=6,overall_complete=False)))


if __name__=='__main__':main()
