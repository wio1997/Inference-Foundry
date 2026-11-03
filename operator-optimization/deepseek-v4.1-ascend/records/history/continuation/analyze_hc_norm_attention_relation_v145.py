"""Explicit owner/carrier/head/reduction readiness and corrected padded row domains."""
from pathlib import Path
from collections import Counter
from itertools import product
import json,hashlib,struct
from finalize_origin_coverage_v9 import N,O,W,read,write
from analyze_hc_norm_index_state_relation_v144 import schedule as old_C2


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]


def carrier_model(p,owner_fused=False,projected=False,direct_copy=False):
    P=8;y=p['Y'];c=p['C'];extra=p['extra'];own=[max(y[r],c[r])+p['norm']+(extra if owner_fused else 2)for r in range(P)];z=[max(own)+p['Gz']+r%2 for r in range(P)];q=[];kv=[]
    for r in range(P):
        if projected:
            need=[r]if p['query_owners']=='ALIGNED'else list(range(P));qa=max(own[k]+1+k%3+extra for k in need)+p['Gp']+r%2;k=max(own[k]+2+k%2+extra for k in range(P))+p['Gp']+r%2
        else:qa=k=z[r]
        q.append(qa+p['query']);kv.append(k+p['kv'])
    attn=[max(q[r],kv[r])+4 for r in range(P)];send=[attn[r]+4+(p['late_head']if r==p['slow_owner']else 0)for r in range(P)];head=[max(send)+2+r%3 for r in range(P)];wa=[head[r]+3+r%2 for r in range(P)];wb=[wa[r]+4 for r in range(P)];out=[wb[r]+(extra if direct_copy else p['copy'])for r in range(P)];rs=[max(out)+2+r%2 for r in range(P)];post=[max(rs[r],c[r])for r in range(P)]
    return dict(owner_Z=own,normalized_carrier_Z=z,Query=q,KV_published=kv,attention=attn,head_send=send,head_receive=head,WO_A=wa,WO_B=wb,stable_output=out,RS=rs,Post_input=post,end=max(post))


def C2_released(p,z):
    q=z+p['query']+p['extra'];cast=max(z,p['metadata'])+1;proj=cast+p['projection']+p['extra'];old=max(proj,p['old_generation'])+1;pool=old+p['pool'];ring=old+p['ring'];norm=pool+p['latent_norm'];ep=norm+p['key_long'];qli=max(q+2,ep)+2;post=max(p['hc_coeff'],q+3,ep+3,qli+3)
    return dict(Z=z,Q=q,CAST=cast,PROJ=proj,OLD_READ=old,POOL=pool,RING=ring,NORM=norm,ENDPOINTS=ep,QLI=qli,Post_input=post,Z_reader_sufficient_release=max(q,cast),FLOAT_input_sufficient_release=proj,projected_KVScore_sufficient_release=max(pool,ring),next_ring_read_deadline=post+p['future_gap'],future_ring_deadline_satisfied=ring<=post+p['future_gap'])


def main():
    assert not(O/'REVIEW-CHECKPOINT-v145.json').exists()
    native=read(O/'hc-norm-attention-native-frontiers-v145.json');root=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend';parts=[]
    for rel,a,b in [('models/deepseek_v41/model.py',887,920),('models/common/ops/sequence_parallel.py',22,27),('models/common/ops/sequence_parallel.py',67,78),('attention/context_parallel/dsa_cp.py',77,91),('attention/context_parallel/dsa_v41_cp.py',194,225),('attention/dsa_v41.py',567,611),('attention/dsa_v41.py',613,683),('attention/dsa_v41.py',700,716)]:
        p=root/rel;parts.append(dict(**pin(p),first_line=a,last_line=b,source=p.read_text().splitlines()[a-1:b]))
    p=N/'projection-operator-source-v61/vllm_ascend/attention/context_parallel/dsa_cp.py';parts.append(dict(**pin(p),first_line=1754,last_line=1798,source=p.read_text().splitlines()[1753:1798]))
    write(O/'hc-norm-attention-source-contract-v145.json',dict(parts=parts,facts=[
      'Zready meansallrequiredrows atconsumer,notowner-localinputRMSend. Currentnormalizedcarriercollective needsitsactualmembers/rowgeneration;newownerQA/KVcarrierneeds typedproducer/transportandallotherZreaders. Explicitall-owner/receiver/head/RSnodesare chosenfullfrontiers,notuniversallyrequiredwholeoperatorbarriers.',
      'RawHCCL opName/group labels matchacrossobservedeight-rankcohortswithouttimestamps. Thatdoesnotattestexporterglobaltagsemantics,client/request/tensoridentity,clockalignmentorformalphasebinding. OneA2A reusesexistingv16decodedqueueflowproof,notdirectTorchCIDornearest-time.',
      'ProfileWO_A/WOB/OUTPUT_COPYrowsG=P*M,includingG6672whilecacheclippedT6668andlastQueryQ830. Theseare differentlogical domains. InputcarrierbuffercontentusesG,QAreadsQ,cacheprojectionsreadT. Earlier v143N=Tformula describedaT-clippedreference,notfullcurrentGmaterialization.',
      'CurrentlargeeagerWO_B isBF16MatMulV3followedbyTensorMoveinto providedstableoutputbufferline1798. Newout-producerneedstheactualoutput/stride/alias/capture/nativeprecisioncontract;sourcecommentalonewasnotproofcopyabsent.',
      'AtlastQ830 frontiers,sourceheadrestore createszeroM834full64-headbuffer,copiesQvalidrows,thenpermutes/packstoAlltoAll. DirectinverseRoPE/pack/zero-tailproducercanbereviewedonlywithoriginalvalidpositions/BF16round/paircoords/newABI/lastreaders;count/payloaddoesnotvanish.',
      'CurrentGalignedprojectionoutput alreadyfitsRSgeometry;noRSzero/catobservedinselected960windows. SourceunalignedRSbranchstillretained. Do notinvent4-rowRSconcatbyconfusingcacheTwithoutputG,orpruneG-T rowswithoutallpadding/state/error/NaN/observerproof.',
      'Eachoutputrecipientheadgroup needsactualsenderquery/headchunks;WO_Acomplete4096K/BF16thenWO_B/BF16partialandnativeSUM/RSoutputroundremain. MovingWO_AbeforeA2Aneedsallgroupweights/newlayout;currentP8one-groupweightsarenotunchangedconsumerABI.',
      'Inputnorm affectscurrentQA/QR/IndexQ/Weightandattentionvalues. SharedTopK/candidatemembershipcanbepreparedonlyunderrow/universe/generationconditions;it isnotreuseofcurrentqueryscores orattentionheadsoftmaxmasses. BF16roundingcanchangeaTopKorderinaconstructeddotmodel.',
      'AttentionSWA/CMPpartialmerges mustkeephead-normalizationstate/sink/occurrences andnativeBF16/onlineordering. CurrentLSEFalseemptydoesnotprovideastatistic. Mergeafterheadmixisnotgenerallyrecoverable;normalizerorprojectedcarrierfusiondoesnoterasecache/state/metadatareaders.',
      'Copy-out/pad/pack/inputcarrier/outputRSpipelinecuts shareoneinterface andresourcegraph. AllC/HCPostrawresidual/pre4/nextFFNandnewqueue/capture/progressgates remain. ExplicitreferenceDAGtimesarenotphysicalbounds oractualrecoverablelatency.'],old_v144_scope_audit_reconciled=True,native_or_physical_gain_proved=False,new_device_work=False,overall_complete=False))
    geom=Counter();observed=[];wo=Counter();queue=0
    for r in native['records']:
        s=r['stages'];m,q,t=r['carrier_sender_rows'],r['query_rows'],r['cache_rows'];g=int(s['WO_A']['definition']['inputShapes_text'].strip('"').split(',')[0]);geom[m,q,t,g]+=1;wo[s['WO_B']['definition']['opType_text']]+=1;queue+='queue_flow_proof'in r['collectives']['HEAD_A2A'];observed.append(dict(key=[r['database_index'],r['forward_API_rowid'],r['source_block_ordinal']],M=m,Q=q,T=t,G=g,head_padding_present='HEAD_ZERO'in s,RS_padding_positive_TASKs=[k for k in ['RS_ZERO','RS_CAT']if k in s],WO_B_copy_shape=s['OUTPUT_COPY']['definition']['outputShapes_text'],all_owner_values_or_clocks_bound=False))
    work=[]
    for(m,q,t,g),n in sorted(geom.items()):
        h=5120;headrow=64*512*2;perpeer=m*8*512
        work.append(dict(M=m,Q=q,T=t,G=g,frontiers=n,P=8,H=h,SP_sender_value_content=2*m*h,global_Z_materialized_content=2*g*h,current_global_Z_QA_KV_interface=2*h*(g+q+t),legacy_v143_T_clipped_reference=4*t*h+2*q*h,global_materialization_correction=2*h*(g-t),rawY_union_receiver_interface=2*h*(g+t),relative_QA_shared_read_credit_unchanged=2*q*h,
          HEAD_A2A_perpeer_elements=perpeer,HEAD_A2A_remote_value_content=2*7*perpeer,head_send_buffer_content=m*headrow,valid_attention_value_content=q*headrow,received_head_buffer_content=g*4096*2,RS_perrank_output_elements=m*h,RS_input_content=2*g*h,
          WO_B_copy_private_interface_reference=4*g*h,remaining_stable_output_RS_interface=4*g*h,combined_output_interface_union_reference=8*g*h,WO_A_WO_B_classical_ops=2*g*(4096*1024+1024*5120),dead_padding_projection_row_elision_proved=False,
          head_pad_pack_old_selected_IO=(3*m+2*q)*headrow if q<m else 2*m*headrow,head_pad_pack_direct_selected_IO=(m+q)*headrow if q<m else 2*m*headrow,
          observed_RS_zero_cat_credit=0,actual_HBM_wire_allocation_or_gain=None))
    write(O/'hc-norm-attention-observed-and-work-v145.json',dict(records=observed,geometry=work,WO_B_operator_counts=dict(wo),direct_CID_A2A=960-queue,reused_queue_flow_A2A=queue,new_clock_or_phase_promotions=0,legacy_v143_absolute_interface_scope='T=clipped cache domain. CurrentfullcarrierwriteusesG. Correction2H(G-T),40960BforG6672/T6668. Relative4MH+2QHrawYalternativecreditunchangedwhencommonGbufferiscountedinbothgraphs;allreadergatesremain.',byte_scope='Declaredmaterializedvalueinterfaces/perpeercontent/classicaldensework,notHBM/wireorphysicalbounds. Directheadpad-packandcopy-outproducerneednewABI;combinedcutsoneunion. No inferredRSconcat/deadpadtail/newmemoryfit.',new_device_work=False,overall_complete=False))
    cases=[];classes=Counter();domain=Counter()
    for slow,bump,norm,gz,gp,q,kv,copy,late,owners,extra in product([0,7],[3,23],[1,5],[2,11],[1,16],[3,18],[2,14],[1,9],[0,17],['ALIGNED','ALL'],[0,10,30]):
        p=dict(Y=[1+r%2+(bump if r==slow else 0)for r in range(8)],C=[2+r%3+(bump//2 if r==7-slow else 0)for r in range(8)],slow_owner=slow,norm=norm,Gz=gz,Gp=gp,query=q,kv=kv,copy=copy,late_head=late,query_owners=owners,extra=extra);a={k:carrier_model(p,f,pc,dc)for k,f,pc,dc in [('baseline',False,False,False),('owner_norm_fused',True,False,False),('projected_only',False,True,False),('copy_direct_only',False,False,True),('joint',True,True,True)]};b,v=a['baseline']['end'],a['joint']['end'];classes['FASTER'if v<b else'SLOWER'if v>b else'TIED']+=1;domain[owners]+=1;cases.append(dict(parameters=p,**a))
    write(O/'hc-norm-attention-owner-DAGs-v145.json',dict(cases=cases,count=len(cases),classes=dict(classes),domain_counts=dict(domain),scope='P8hypotheticalchosenfullfrontierSWA-onlytwo-projectiondomain;allotherZreadersmustbeclosed/providedbeforeprojectedcarrierchoice. OwnerHCatomiccompletionretained,noF01partialYcredit. Per-recipientcarriertransport/headA2A/allTPpartialRSandcurrentcoeffexplicit. ALIGNEDqueryownerisconditionalmetadata,newABI. Fullrow/tilepipelinecanbefiner,newresourceserviceunbound.',endpoint='AttentionHCPostinput-ready,Post/FFN/nextlayernotpriced',physical_bound_or_actual_speedup=False,new_device_work=False,overall_complete=False))
    old=read(O/'hc-norm-index-state-readiness-DAGs-v144.json');release=[];oldcheck=0;newclasses=Counter();future=Counter()
    for i,x in enumerate(old['cases']):
        p=x['parameters'];z0=max(p['hc_y'],p['hc_coeff'])+p['input_norm'];own=[z0+(r%3)*(p['query']%5)+(p['old_generation']%7 if r==7 else 0)for r in range(8)];link=2+p['metadata']//20;z=max(own)+link;ans=C2_released(p,z);zero=C2_released(p,z0);assert zero==x['joint_repriced'];oldcheck+=1
        delta=ans['Post_input']-x['joint_repriced']['Post_input'];newclasses['LATER'if delta>0 else'UNCHANGED'if delta==0 else'EARLIER']+=1;future['READY'if ans['future_ring_deadline_satisfied']else'VIOLATED']+=1;release.append(dict(old_case_index=i,owner_Z_ready=own,carrier_service=link,consumer_Z_ready=z,legacy_assumed_Z_ready=z0,joint_with_explicit_Z_release=ans))
    write(O/'hc-norm-attention-C2-release-refinement-v145.json',dict(cases=release,count=len(release),zero_added_release_reproduces_old_cases=oldcheck,consumer_release_classes=dict(newclasses),chosen_future_classes=dict(future),scope='Refineall6144v144referencecaseswitharbitraryP8ownerreleaseandcarrierparameters;noactualcrossrankclock/tensoridentityorbandwidthinference. ThismakespreviousassumedZreadyexplicit;nooriginalglobalTprojectionfromonlyownerM.',new_device_work=False,overall_complete=False))
    raw=read(O/'hc-norm-projection-stat-and-round-witnesses-v143.json')['BF16_Z_GEMM'];a,b=raw['Z_FLOAT_prefix'],raw['Z_BF16_prefix'];keys=[[1.3359375,-1.],[1.,-3/1024]]
    def scores(q):return [f32(f32(q[0]*k[0])+f32(q[1]*k[1]))for k in keys]
    ra,rb=scores(a),scores(b);assert ra[0]>ra[1]and rb[0]<rb[1]
    P,M,L,D=2,2,2,2;send=[[[[1000*r+100*t+10*h+d for d in range(D)]for h in range(P*L)]for t in range(M)]for r in range(P)];recv=[[[send[r][t][g*L+h][d]for d in range(D)]for h in range(L)]for g in range(P)for r in range(P)for t in range(M)]
    tile=dict(partial_ready_by_headgroup=[[1,100],[2,100]],RS_service=1,FFN_next_ready_service=[100,1]);full=max(max(x)for x in tile['partial_ready_by_headgroup'])+1;tile_r=[max(row[j]for row in tile['partial_ready_by_headgroup'])+1 for j in range(2)];full_next=[full+x for x in tile['FFN_next_ready_service']];tile_next=[x+y for x,y in zip(tile_r,tile['FFN_next_ready_service'])]
    write(O/'hc-norm-attention-map-release-and-score-witnesses-v145.json',dict(
      same_counts_wrong_head_map=dict(P=P,M=M,local_heads=L,D=D,send=send,recv_rank_major=recv,wrong_receiver_rows=list(reversed(recv)),equal_numel_not_mapping=True,actual_profile_payload_attested=False),
      input_round_changes_positive_dot_TopK=dict(unrounded_query=a,rounded_query=b,keys=keys,unrounded_scores=ra,rounded_scores=rb,unrounded_top1=0,rounded_top1=1,current_native_QLI_quant_scores_or_frozen_values=False),
      localZ_not_consumerZ=dict(owner_Z=[1,1,1,1,1,1,1,30],carrier_service=5,wrong_localZ_projection_start=1,correct_fullcarrier_consumerZ=35,actual_times=False),
      crosslayer_row_pipeline=dict(**tile,current_full_RS_allowners=full,chosen_tile_RS=tile_r,current_next_owner_ready=full_next,tile_next_owner_ready=tile_next,current_next_fullcarrier_gate=max(full_next),tile_next_fullcarrier_gate=max(tile_next),native_partial_publish_or_gain=False,scope='Abstract2headgroupsx2rowowners,originalBF16partialandnativeSUMcontractassumedretained;newtilehandoff/progress/servicesunproved.'),
      witness_scope='Constructeddata/time/scorecounterexamplesonly;notactualweights,queryscores,requestmaps,clockalignment,newnativeprecisionorcapacity.',new_device_work=False,overall_complete=False))
    pairs=[['F02',x]for x in ['F12','F13','F14','F15','F16']];contracts=[
      'InputnormcurrentZ/QR/IndexWeight drivescurrentQLIselection;staticcandidate/sharedmembership preparationonlyunderproperrows/universe/generation. Do notreusequeryscores/headmasses orcancelBF16normonTopKwithoutmargin/nativeproof.',
      'Inputnormalizer/ownercarrier mustsatisfyquery/cache/metadataandallownerreleasebeforeSWA/CSAreads. Headwisepartialstate/sink/occurrences/nativeonlineBF16rounds retained;fusiondoesnotmakefullcache/stateorotherZreadersdead.',
      'Normalize->attention->inverseRoPE->headexchange->WO_Aisahigher-orderpathwithtypedrounds/positionandrow-headmaps. Headrestoreallrequiredsenderchunks,complete4096K/BF16outputandweights/newABIpriced;noscalarlinearjump.',
      'CurrentM/Q/T/Gdifferent:Q830headzero/copytoM834;G6672WOA/WOB/stablecopydespiteT6668cacheclip. Newdirectpad-pack/outputcopyproducerchoicesconditional;RSisalignedcurrentdomain,notinvented4-rowconcat/deadpadding.',
      'Allownerinputcarrier,allqueryownerheadA2A,allTPpartialRSandcurrentHCcoeff/Post/pre4joinspricedonce. Copy-onlyversusfulloutputRSstreaminginterfacesareunionchoices;partialrowcrosslayerpipelinecanbefinerbutneedsnativeSUM/round/finiteprogress/capturecontract.'
    ]
    ex=[['HG73-OUTPUT','F02'],['HG75-Q-KV-JOIN','F02'],['HG86-TERMINAL-QUERY-STATE-CUT','F02']]
    write(O/'hc-norm-attention-composition-v145.json',dict(clauses=[dict(id='REL145-0'+str(i+1),a=a,b=b,contract=c)for i,((a,b),c)in enumerate(zip(pairs,contracts))],new_pairs=pairs,new_pair_clauses=5,group_extensions=ex,all_runtime_gates_proved=False,whole_pair_or_all_subsets_closed=False,speedup_additive=False,overall_complete=False))
    names=['hc-norm-attention-native-frontiers-v145.json','hc-norm-attention-A2A-correlation-audit-v145.json','hc-sp-eager-native-frontiers-v134.json','hc-sparse-eager-native-frontiers-v138.json','hc-sparse-source-and-join-contract-v138.json','hc-sparse-shared-topk-views-v138.json','attention-projected-state-source-contract-v100.json','attention-projection-rounding-and-position-v100.json','communication-native-operator-population-v66.json','communication-role-contract-v16.json','communication-operator-contract-v66.json','hc-norm-projection-observed-and-work-v143.json','hc-norm-projection-stat-and-round-witnesses-v143.json','hc-norm-index-state-readiness-DAGs-v144.json','hc-norm-index-state-scope-audit-v144p.json','hc-norm-index-state-postcheck-result-v144p.json','candidate-relation-clauses-v144.json']
    write(O/'hc-norm-attention-inputs-v145.json',dict(inputs=[pin(O/n)for n in names]+[pin(N/'unassigned-communication166-v16.json')],questions=['HowdoSPownerready/carrier/headexchange/TPpartialRSandcurrentcoeffjointlygateinputnorm-to-attention/output?','AreTcache,Qquery,MuniformandGpaddedprojectionthe same,andwhatcurrentcopy/paddingworkisactuallypresent?','CanrecordedHCCLlabelsbindrequests/clocks,oronlyestablisheightranklabelconsistency?','Whichdirectout/padpack/tiledRSinterfacescancomposewithoutdeletingrounds/maps/otherreadersorcountingoverlappingcredits?'],profile_first=True,new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='MODELED',frontiers=960,TASKs=native['source_CID_TASK_bindings'],cohorts=360,geometry=[dict(M=k[0],Q=k[1],T=k[2],G=k[3],n=v)for k,v in geom.items()],owner_DAGs=len(cases),owner_classes=classes,C2_refined=len(release),C2_release_classes=newclasses,new_pairs=5,overall_complete=False)))


if __name__=='__main__':main()
