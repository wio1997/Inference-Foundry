"""Typed collective count models and operator repartition costs, without network guesses."""
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import hashlib,json,math
from finalize_origin_coverage_v9 import N,O,W,read,write
from analyze_projection_ceiling_v61 import stat
from model_router_vector_v63 import bf16,f32
S=W/'frozen-source-snapshot'
H,K,R,P=5120,4096,1024,8

def value_model(op,count,dtype,p):
    width={'BFP16':2,'FP32':4}[dtype];q=count*width
    if op=='hcom_allReduce_':
        return dict(count_semantics='entireperrankreductionarray',input_bytes=q,output_bytes=q,
            declared_ring_send_bytes=2*(p-1)*q/p,declared_ring_recv_bytes=2*(p-1)*q/p,
            declared_ring_reduction_adds=(p-1)*count/p,declared_ring_rounds=2*(p-1))
    if op=='hcom_allGather_':
        return dict(count_semantics='perranksendCount',input_bytes=q,output_bytes=p*q,
            declared_ring_send_bytes=(p-1)*q,declared_ring_recv_bytes=(p-1)*q,
            declared_ring_reduction_adds=0,declared_ring_rounds=p-1)
    if op=='hcom_reduceScatter_':
        return dict(count_semantics='perrankrecvCount;inputisPtimescount',input_bytes=p*q,output_bytes=q,
            declared_ring_send_bytes=(p-1)*q,declared_ring_recv_bytes=(p-1)*q,
            declared_ring_reduction_adds=(p-1)*count,declared_ring_rounds=p-1)
    assert op=='hcom_alltoall_'
    return dict(count_semantics='uniformperpeersendCount/recvCount;perrankinput/outputPtimescount',
        input_bytes=p*q,output_bytes=p*q,declared_pairwise_send_bytes=(p-1)*q,
        declared_pairwise_recv_bytes=(p-1)*q,declared_pairwise_reduction_adds=0,
        declared_pairwise_rounds=p-1,remote_payload_to_count_times_width_ratio=p-1)

def main():
    assert not (O/'communication-work-payload-models-v66.json').exists()
    pop=read(O/'communication-native-operator-population-v66.json');ops=pop['operations']
    pins=[]
    for x in read(O/'heads-sampling-operator-contract-v65.json')['source_files']:
        if x['path'].endswith(('models/deepseek_v41/model.py','models/deepseek_v41/dspark.py','ops/vocab_parallel_embedding.py')):pins.append(x)
    old=read(O/'M19-embedding-HC-CP-row-and-consumer-contract-v35.json')['source_files']
    for rel in ['attention/context_parallel/dsa_cp.py','attention/context_parallel/dsa_v41_cp.py']:
        p=S/'vllm-ascend/vllm_ascend'/rel
        pins.append(dict(path=str(p),sha256=old['/vllm-workspace/vllm-ascend/vllm_ascend/'+rel],original166_hash_binding=True))
    for x in pins:assert hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256']
    config=N/'sampling-guard-source-v55/data/tiankuan/wio/DeepSeek-V4.1-Flash-1017-w8a8/config.json'
    assert hashlib.sha256(config.read_bytes()).hexdigest()=='43d3b2bc55626893d59275b32e04068c0fc916f5c382eb91de73bdbade9c88f7'
    cfg=read(config).get('text_config',read(config));assert (cfg['o_groups'],cfg['num_attention_heads'],cfg['head_dim'],cfg['o_lora_rank'])==(8,64,512,1024)
    groups=defaultdict(list);semantic=defaultdict(list)
    for op in ops:
        assert op['rankSize']==8
        key=(op['operator'],op['dtype'],op['rankSize'],op['count'],op['group_name'])
        groups[key].append(op);semantic[(op['semantic_role'],)+key].append(op)
    assert len(groups)==61
    assert len({key[:4]for key in groups})==42
    rows=[]
    for key,rr in sorted(groups.items()):
        op,dt,p,c,group=key;vm=value_model(op,c,dt,p)
        rows.append(dict(operator=op,dtype=dt,P=p,count=c,group_name=group,calls=len(rr),
            COMM_OP_keys=[[r['database_index'],r['COMM_OP_rowid']]for r in rr],semantic_role_counts=dict(Counter(r['semantic_role']for r in rr)),
            node_namespace_counts=dict(Counter(r['node_namespace']for r in rr)),native_interval_ns=stat([r['duration_ns']for r in rr]),
            model=vm,input_output_value_bytes=vm['input_bytes']+vm['output_bytes'],
            physical_numeric_network_bound=False,
            contract='Declaredring/pairwisereferencealgorithm,purecopyorreductionworknotCubeMACwork. RING-RING-NHRlabeldoesnotprovesingleflat-ringtraffic. Needactualinjection/link/cut/topology/latency/HBM-L2andnativeprecisionbeforephysicalceiling;arbitraryinput-copyreceivenecessitydiffersfromreductionalgorithmtraffic.'))
    assert sum(r['calls']for r in rows)==14720
    # Identical native graph block domains: N=24 physical rows, local queries3.
    # Broader models also price Nexchange/Mprojection padding mismatches explicitly.
    repart=[]
    for m,a in [(24,24),(20,24),(6400,6400),(6528,6528),(6668,6672)]:
        wa=2*K*R;wb=2*R*H
        old_send_a=2*a*K*(P-1)/P;new_send_a=2*a*R*(P-1)/P
        rs=2*math.ceil(m/P)*H*(P-1);ag=2*math.ceil(m/P)*H*(P-1)
        base_f=2*m*(K*R+R*H)
        allpad_wa_f=2*a*K*R+2*m*R*H
        localall_f=2*a*(K*R+R*H)
        repart.append(dict(M_projection=m,A_CP_uniform_exchange_rows=a,
            baseline_WA_bytes_per_rank=wa,baseline_WB_bytes_per_rank=wb,
            pre_exchange_WA_all8groups_bytes_per_rank=P*wa,pre_exchange_WA_extra_bytes_per_rank=(P-1)*wa,
            query_owner_WA_WB_all8groups_bytes_per_rank=P*(wa+wb),query_owner_extra_weight_bytes_per_rank=(P-1)*(wa+wb),
            baseline_factorized_ops_per_rank=base_f,
            pre_exchange_all_padded_WA_ops_plus_baseline_WB_per_rank=allpad_wa_f,
            pre_exchange_all_padded_WA_compute_ratio=allpad_wa_f/base_f,
            owner_all_padded_WA_WB_ops_per_rank=localall_f,owner_all_padded_compute_ratio=localall_f/base_f,
            baseline_restore_AlltoAll_send_bytes_per_rank=old_send_a,
            pre_exchange_WA_rank1024_AlltoAll_send_bytes_per_rank=new_send_a,
            output_reduceScatter_send_bytes_per_rank=rs,input_AllGather_send_bytes_per_rank=ag,
            baseline_output_A2A_plus_RS_send_bytes=old_send_a+rs,
            pre_exchange_WA_output_A2A_plus_RS_send_bytes=new_send_a+rs,
            pre_exchange_output_payload_ratio=(new_send_a+rs)/(old_send_a+rs),
            baseline_input_AG_plus_output_send_bytes=ag+old_send_a+rs,
            pre_exchange_input_AG_plus_output_send_bytes=ag+new_send_a+rs,
            owner_all_groups_output_network_send_bytes=0,
            owner_all_groups_group_BF16_partial_plane_bytes=2*a*H,
            extra_WA_for40layers_bytes=40*(P-1)*wa,extra_WA_WB_for40layers_bytes=40*(P-1)*(wa+wb),
            baseline_compute_reference_us=base_f/376e12*1e6,
            all_weights_cold_value_reference_baseline_us=(wa+wb)/1.6e12*1e6,
            all_weights_cold_value_reference_pre_WA_us=(P*wa+wb)/1.6e12*1e6,
            all_weights_cold_value_reference_owner_us=P*(wa+wb)/1.6e12*1e6,
            contract='Conditionalnewpartition,notcurrentbranchalreadyoptimized. WA-beforeexchangeappliesall8groupweightsontokenowner,retainsBF16WAresultthenexchanges1024ratherthan4096;WBsameheadowner. Ownerbothstageskeepsall8BF16WBpartialsbeforebaseline-equivalentreduction. M=AhasunchangedpercarddenseFbut8xweightreplication;A>MuniformpaddedrowsaddFunlessvalid-rowprune+zero-fill/rowmapsproved. Trim/DSparkhidden-suffix/CPtokenranges/SPowners/nonfinitezero-policy/allreaders/nativeG8MlocalvsG1Mglobalprecisiongates. NetworkeliminationdoesnotproveHCCLsumcanbefreelyreassociated.'))
    first=repart[0]
    assert first['baseline_restore_AlltoAll_send_bytes_per_rank']==172032
    assert first['pre_exchange_WA_rank1024_AlltoAll_send_bytes_per_rank']==43008
    assert first['pre_exchange_output_payload_ratio']==2/3
    assert first['pre_exchange_WA_extra_bytes_per_rank']==58720256
    assert first['query_owner_extra_weight_bytes_per_rank']==132120576
    assert first['pre_exchange_all_padded_WA_compute_ratio']==1
    assert repart[1]['pre_exchange_all_padded_WA_compute_ratio']>1
    packed=[]
    for n in [24,6400,6528,6672]:
        local=math.ceil(n/P);hc=2*local*4*H;mix=4*local*4;q=hc+mix
        packed.append(dict(N_physical=n,HC_local_bytes=hc,premix_local_bytes=mix,
            separate_two_AllGather_send_bytes=(P-1)*q,packed_one_AllGather_send_bytes=(P-1)*q,
            ordinary_local_pack_value_bytes=2*q,ordinary_global_unpack_value_bytes=2*P*q,
            pack_plus_unpack_additional_value_bytes=2*(P+1)*q,
            pack_unpack_1p6TB_cold_value_reference_us=2*(P+1)*q/1.6e12*1e6,
            contract='BF16HC+FP32premixsamegenerationreadyforfinaltail;bytepackpurecopiesonly,newmultiinput/multioutputABIorpack/unpacklayout. Payloadunchanged,2→1collectivelaunchpossiblebutdtypealignment/allbanks/readiness/snapshotaliases/capturelastreadandpackoverheadpriced;notcastpremixBF16.'))
    embedding=[]
    for m in sorted({int(op['count']/H)for op in ops if 'EMBEDDING_ALLREDUCE'in op['semantic_role']}):
        a=math.ceil(m/P)*P;q=2*m*H
        ar=2*(P-1)*q/P;rs=(P-1)*2*(a/P)*H
        embedding.append(dict(M=m,padded_rows=a,baseline_AllReduce_send_bytes=ar,new_ReduceScatter_send_bytes=rs,
            reduction_only_payload_ratio=rs/ar,if_full_reader_requires_AG_total_send_bytes=2*rs,
            full_reader_RS_then_AG_to_original_AR_ratio=2*rs/ar,
            source_replicated_output_bytes=q,new_sharded_output_bytes=2*(a/P)*H,
            contract='Onlyifallfirstdecoder/preprocess/raw/confidence/CP/HCreaderscanuseSPownersornewprotocol. CurrentstandardvocabembeddingdeliberatelyARandexpectscompletefirst-layersequence;notdropin. AnyfullreaderAGrestorestotalARpayload(orworsewithpadding). Reduceorder/BF16zero-sign/nativeprecisionandtokenvalidity/mapmustmatch.'))
    # A finite BF16 reduction order witness, separate from actual HCCL accumulation policy.
    x,y,z=256.,-256.,1.
    left=bf16(f32(bf16(f32(x+y))+z));right=bf16(f32(x+bf16(f32(y+z))))
    # Pick a tie-sensitive example at larger exponent if this exactly cancels.
    if left==right:
        x,y,z=512.,-512.,1.;left=bf16(f32(bf16(f32(x+y))+z));right=bf16(f32(x+bf16(f32(y+z))))
    assert left!=right
    write(O/'communication-work-payload-models-v66.json',dict(count_group_regimes=rows,count_group_regime_count=61,
        type_dtype_rank_count_regime_count=42,
        semantic_role_count_group_regimes=len(semantic),semantic_role_count=sum(r['calls']for r in rows),
        operator_repartition_models=repart,HC_premix_mixed_dtype_gather_batch_models=packed,embedding_AR_to_RS_models=embedding,
        network_bound='NoactualGbit/s/GB/sorcommunicationmicrosecondfloor:requireddevice/host/injection/cut/topology/latencyandloadedalgorithmunbound. DeclaredalgorithmTc>=max(Ssend/Binj,Rrecv/Brecv,percutbytes/Bcut,roundlatency,QHBM/BHBM,QL2/BL2,dependency);ringvaluesarenotuniversalphysicallowerbounds.',
        overlap_bound='ForcompatiblepanelpipelineT>=max(TCube,TVector,Tcommunication)onlywithindependentresources/readydata;sumserializedphasesandsharedresourcework. AIVusesVector;RMS/quant/Ropecompetewithit. Currentintervaloverlapcategoriesnotremovablecriticaltime.',
        reduction_order_witness=dict(values=[x,y,z],left_BF16_pairwise=left,right_BF16_pairwise=right,actual_HCCL_accumulation_policy_not_proved=True),
        overall_complete=False))
    write(O/'communication-operator-contract-v66.json',dict(source_files=pins,config_path=str(config),config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        local_operator_clause_scope='restore_tp_heads77-90;DSAv41_project_output208-229;Woa/Wob1754-1803;modelSPboundary895-902;vocab_standard408-435. Existingv35/v61/v65pinsreused,no remote/frameworkaudit.',
        facts=[
            'restore_tp_headsreorders[localtokens,all64heads,512]intosend[TP,localtokens,8heads,512],thenuniformAlltoAll→[alltokens,8heads,512]. COMMcountisperpeer;M24count12288represents196608Bfullrankplaneand172032Bremotesend,not24576Btotal.',
            'CurrentWoaG1[alltokens,4096]→BF16[alltokens,1024],Wob[1024,5120]→BF16partial,thenSPReduceScatter. Woa-beforeexchangeisnewall8group-weightreplication,notfreecommutationwithcurrentonegroupweight.',
            'BeforeA2Acurrentoutputpadsqueryrowstoper_rank;afterA2Aitmaypadtohiddenrowsandtrimsbeforeprojection. Sourcefullrow/padding/CP-SPmappingmustbind;A6668vs6672andM20vs24cannotbeignored.',
            'WobpartialBF16thenHCCLreduceisnativeprecisionboundary. Localall-groupssumrequiresmatchingnativeaccumulation/orderorapprovederrorcriterion;BF16witnessrefutesunconditionalassociativity.',
            'HC/premixpurecopyAllGathershavedifferentdtypes. Bytepackcancombineonlywithnewlayout/ABIandsamegeneration/readiness;doNOTdowncastpremixorclaimpayloadreduction.',
            'Threeauxgathersoccuratdifferentcheckpoints;coalescingrequiresdeferringandretainingallsources/allreaders. FutureattentionlayersdependoninterveningHcPost/FFNstate,notbatchablebyshape.',
            'VocabularyembeddingARoutputfeedscompletefirst-layersequenceandmultiplepreprocess/raw/HC/CPreaders;AR→RSisconditionalnewownershipprotocol,notexistingclone-removal.',
            '3192COMMopswithoutComputeInfoseparatelyretained. COMMUNICATION_SCHEDULEhas3192definitionsbutop134examplejoinsNOTIFY_RECORD/WAITbyconnection,notoneexactScheduleTASK. Countcoincidencedoesnotprovebijection.',
            'All14720HCCLopsrankSize8;MC2EP16internalcommunication/CPU-Gloo-DPmetadataarenotmadevisiblebythiscensus. Existingv56EPmodelsremainseparate.' ],
        primary_API_references=[
            dict(url='https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/80RC3alpha003/apiref/hcclapiref/CANN%208.0.RC3.alpha003%20%E9%9B%86%E5%90%88%E9%80%9A%E4%BF%A1%E6%8E%A5%E5%8F%A3%E5%8F%82%E8%80%83%2001.pdf',supports='HcclAlltoAllsendCount/recvCountperpeer;APIaritycorroboratedbyfixedsourceP8restorelayout/nativecounts.'),
            dict(url='https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/82RC1alpha001/API/hcclapiref/hcclcpp_07_0024.html',supports='ReduceScatterrecvCount;inputPtimesrecvCount. OlderAPIarityreference,notcurrent9.2algorithm/capacityproof.')],
        new_remote_source_read=False,new_model_NPU_capture_service_process_or_implementation=False,current_physical_network_ceiling_proved=False,overall_complete=False))
    vs=[]
    def v(i,c,s,b,g,d='CONDITIONAL'):
        vs.append(dict(id=i,candidate=c,choice=s,work_IO_effect=b,conditions=g,decision=d,conditional_operator_analysis_deliverable=True,implementation=False,native_test=False,critical_path_gain_us=None))
    v('COMM-SP-PROJECT-BEFORE-AG','F32','ProjectownedrowsbeforefullhiddenAG','Existingroleplanesfeaturecontent10x/5x/2xratiosreconciledwithactualcountmodels','AllSWA/C1/C2/index/quant/norm/cacheconsumersandparameterreplicas/rowgeneration/CPglobalpoolstate;newpacketplanes/scalesandcollectiveorder. Noquery8xduplicateworkclaim.')
    v('COMM-GATHER-PROJECTION-PIPELINE','F32','StreamcompleteincomingrowsintoKVprojection','Keepnetworkpayload;avoidfullbatchwait/privatebufferlifetimeonly','KVnorm/quant/fullrowreadiness/oldlatentWAR/cachepublicationandallreaders,newmultioutputbankAPI,VectorAIVvsCubeandquantcontention;Tmaxonlyindependentresources.')
    v('COMM-ATTENTION-DIRECT-PACK','F14','AttentionkernelwritesTPsendlayoutdirectly','FoldfullattentionoutputTranspose/contiguouspackinterface','PreserveattentionBF16/exp/rescale/Ropeboundary/layout/validrowzero-paddingandotheroutputreaders;newoutputABI/capture. NotdropA2Apayload.')
    v('COMM-A2A-TO-WOA-PANEL','F15','ConsumecompletepeerheadpanelswhileA2Aarrives','Potentialcommunication/Cubepipelineandfullbatchliveness','Full4096KdotbeforeBF16WAround,perpeerarrival/event/tile/GMqueue/capacity;COMMkernelVECTORandpostopscontention,newABI. Nozero-HBMhandoffassumption.')
    v('COMM-WOA-BEFORE-A2A','F15','Tokenownercomputesall8groupWAthenexchanges1024','Restorepayload/4;M=AoutputA2A+RSsend2/3;percardWAweights+58.72MB/layer','Replicatecorrect8WAshards/version,changedG8localMtilingnativeprecision,allBF16WAoutputs/rowpadding/CP-SPownerplan/readers;40layerextra2.3488GB/coldloadcost. WorknotunchangedifA>Mwithoutprivatepadprune.')
    v('COMM-OWNER-WOA-WOB','F16','Tokenownercomputesbothstages/all8groupsandnative-equivalentreduce','CanremoveoutputA2A+RS;percardweights+132.12MB/layer,sameFonlyM=A','All8BF16WA/WBpartialsandnativeHCCLreducepolicy/rowowners/generation/allreaders,smalllocalMtiling/cache/capacity;40layerextra5.2848GB. NotoneunroundedcombinedGEMM.')
    v('COMM-WOB-RS-PANEL','F16','PublishroundedWobtilesintoReduceScatter','Currentreducepayloadremains;pipelineandpartialinterfacebudget','KeepBF16partialbeforecurrentnativeSUM,identicalTPgroups/order/rowpad andnotify/capture/readerprotocol;Cube/Vector/linkresourcecost.')
    v('COMM-RS-HCPOST','F03','ReduceScatterproducerfeedslocalHcPosttile','AvoidprivatelocalHresultwrite/readonlyifnewout/consumerAPI','NativeBF16reduceoutputbeforeHcPost,fullHCresidual/currentcoefficients/notfuturelayergeneration,allreaders/captureandtilepublication.')
    v('COMM-EMBED-AR-RS','F02','ReplaceembeddingARwithownerRS','Idealsend/2withalignedM;fullreaderAGrestoresARcostorworsepadding','Currentstandardembeddingfullsequenceexpectationandallpreprocess/raw/HC/confidence/CPreadersreworked;rowmask/IDerrors/gamma/zero-sign/nativereduceprecisionandownerplan.')
    v('COMM-HC-PREMIX-PACK','F26','Onebyte-packedcopygatherforHC+FP32premix','2→1launch,payloadunchanged;pack+unpack18qadditionalvaluebytesP8','Samegeneration/collectivegroup/readytime,mixedtypealignment,newlayoutormultiinput/outputABI,snapshotbankaliases/lastreader;keepFP32mix.')
    v('COMM-AUX-THREE-BATCH','F27','Defer3checkpointauxgathersandbatchafterthird','3→1launch,payloadunchanged,3localsourcesretained','Allauxreadersallowdeferral,currentoverlap/readinesslost,globalrank/checkpointlayoutunpackorindirectGEMM,newcapturebanks. DistinctcheckpointvaluesnotCSE.')
    v('COMM-DIRECT-DESTINATION','F31','Gatherdirectlyintopersistentconsumerbanks','Removeonlyprovenpostgathercopies','NewoutABI/stride/capacity/alignment/capturelastread/prefix-tailvisibility;v64candidateindexaliasalreadybaseline,notinventanothercopy.')
    v('COMM-GREEDY-SMALL-SUMMARY','F35','ReplacefullvocabAGwithprocessedglobalbestsummary','v65small-keyAG56Mvsfull226240M,payloadonly','Allraw/prob/entropy/recoveryreadersandNaN/ties/softcap/validcolumns/remap;INT64MAXsupportunbound,INT32wordAGfallbackpriced.')
    v('COMM-CONTEXT-FEATURE-RS-AG','F29','Retaincontextfeatureshard→KVFP32RS→ownerfullnorm→BF16AG','v64matrixF11/18;Pceil8M12paddingcommunicationworse','Keepglobal5120normstats/3separate512norms/nativepartialsum/fullcontextbankqueryreaders/metadata;notnewautomaticnetworksaving.')
    v('COMM-ENGRAM-ROW-HEAD-OWNERS','F24','ExchangeonlyrequestedEngramrows/heads','v62lookup/UVAandWKVheadpartialpricewithnewcommmodel','Request/hash/accepted-historygeneration,TP/destinationorder/zero-padding/allbankreaders;partialWKVplanehuge,notfreelocalheadshard.')
    v('COMM-RAGGED-PAD-TRIM','F15','Replaceuniformpaddedcollectiveswithvalid-countprotocol','Actualvalidn_pvsPmax(n_p)content;payloadcanfall','CurrentuniformHCCLcountcannotjustshrinklocally,needAlltoAllV/gathervmetadata/errors/allrankparticipation/order/capture;newzero-fill/trimandvariable-latencycost.')
    v('COMM-LOSSLESS-8BIT','F31','ReplacearbitraryBF16copyplaneswithINT8exactly','16bit→8bitcannotrepresentallarbitraryBF16valueswithoutsideinformation','Losslessdictionary/sparsityneedsactualvalues/metadata;approxquantrequiresnewerrorcontract/scales/roundingandcanchangeTopK. No profile-valueevidence.', 'EXCLUDED_AS_UNCONDITIONAL_EXACT_OPTIMIZATION')
    v('COMM-CROSS-LAYER-SAME-SHAPE-CSE','F03','Eliminatelayergathersbyidenticalshape/address','HC/attention/FFNupdatesproducefreshvalues','Onlysemanticgeneration-equality/allreaderscouldenablecache;futurelayersnotreadyandcheckpointvaluesdistinct.', 'EXCLUDED_FROM_CURRENT_EXACT_FUSION')
    v('COMM-CROSS-ITERATION-REUSE','F31','Reuseunchangedcommunicationplanes','Conditionaldelta/unchangedU,notfullshapecachehit','Request/token/acceptedstate/weights/Rope/slot/owner/rankgroupgenerationandallfutureobservers;existingKV/M21metadatareusealreadybaseline,comparekeys/protocolcost.')
    v('COMM-AIV-CORE-ALLOCATION','F16','TradeVectorcoresbetweencomm/purepostopsandCubeproducer','Resourceconstrainedpipelinecouldchangeexposure','Actualloadedalgorithm/corecontrol/support/competingnormquantRopeandlatency/capturegates;blockNum32doesnotmeasurephysicalutilizationorfree32cores.')
    write(O/'communication-fusion-variants-v66.json',dict(variants=vs,variant_count=len(vs),
        interaction_rules=['WApreexchangeandowner-allgroupsarealternativeweight/communicationpartitions. Combineoncewithdirectpack/panel/RS-HcPostinterfaces;neveraddtheirsavedcommtimes.',
            'HC/premixbatchandauxbatchreducecallsbutnotpayload;pack/unpack/readinessmaycostmorethansavedlatency.',
            'M=A24graphmodeldoesnotestablishN/Mcostforeager20→24or6668→6672padding;allreadernewprotocolsremainconditions.',
            '61grouplabel/countmodels(42type/dtype/P/countcombinations)coverall14720HCCLcalls,notMC2EP16/Gloo/hostcollectives. Schedule3192countnotbijectionproof.',
            'HCCLComputeInfo11528nativeTASKdurationsareexactlyCOMMintervalsforlinkedops;donotaddthemasseparatecompute+commtimeorMFUwork.',
            'Conditionaltheorydeliverablesdonotrequireimplementation;actualphysicalnetworkceilingandunobservedoperatorbranchesremainopen.'],whole_modules_closed=0,overall_complete=False))
    print(dict(group_count_models=61,type_dtype_P_count_models=42,all_calls=14720,semantic_role_count_regimes=len(semantic),source_pins=len(pins),variants=len(vs),
        M24_restore_send=[first['baseline_restore_AlltoAll_send_bytes_per_rank'],first['pre_exchange_WA_rank1024_AlltoAll_send_bytes_per_rank']],
        WA_extra_weights=first['pre_exchange_WA_extra_bytes_per_rank'],owner_extra_weights=first['query_owner_extra_weight_bytes_per_rank'],reduction_witness=[left,right]))

if __name__=='__main__':main()
