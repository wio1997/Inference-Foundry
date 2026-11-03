"""F29-F32 conditional joint row-owner/packet/publish analysis; CPU only."""
from pathlib import Path
from collections import Counter
import hashlib,json
from finalize_origin_coverage_v9 import N,O,W,read,write
from context_sp_models_v130 import carrier_case,pareto_cases,recipient_cases,writer_witness


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def part(p,a,b):return dict(**pin(p),first_line=a,last_line=b,source=p.read_text().splitlines()[a-1:b])


def main():
    assert not (O/'REVIEW-CHECKPOINT-v130.json').exists()
    root=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend';ds=root/'models/deepseek_v41/dspark.py';mo=root/'models/deepseek_v41/model.py';at=root/'attention/dsa_v41.py'
    cp=N/'projection-operator-source-v61/vllm_ascend/attention/context_parallel/dsa_cp.py'
    for p,h in [(ds,'fcee256905928917154d8dac94d6611bb83d02da441e8aa154a525ef51e6a601'),(at,'3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd')]:assert pin(p)['sha256']==h
    prof=read(O/'context-sp-profile-frontiers-v130.json');positive=[r for r in prof['records']if r['source_context']!='CALLER_DUMMY']
    assert len(positive)==88 and all(all(l['query_KV']for l in r['layers'])for r in positive)
    write(O/'context-sp-source-contract-v130.json',dict(parts=[part(ds,55,69),part(ds,105,148),part(ds,250,309),
        part(mo,887,903),part(at,319,337),part(at,350,382),part(at,690,717),part(cp,1867,1886)],
        facts=[
            'F32 is not target-only:312direct draft/dummy SPgathers sit in104source-orderfrontiers. 264proposalwindows eachcontain aBF16[M,5120]x[512,5120] KV task;48dummywindows havepositivezero-output atdsa_v41.py698instead.',
            'Context source is main_norm(main_proj(targetaux)). Current query input is layer mHC pre plusinputnorm. Equalwidth5120 doesnotmake their values identical.',
            'Within a selected draftlayer object,F29precompute andordinary queryKV source expressions use thatlayer self_attn.wkv/kv_norm. This supports sharingweights acrossdifferentrows,notCSE ofprojection outputs oractualloadedreplica proof.',
            'Draft role explicitlyhascompress_ratio0,is_kv_sourceFalse,is_index_sourceFalse. TheBF16SWA-only512carrier model doesnotcover targetC1/C2 extraFLOAT/quant/indexplanes.',
            'Context helper uses itsrotary module on64tailthenCat;ordinary/legacyqueryKV usesinplacepartialrotary anditsmetadata. Acommon512normedcarrier mustretain distinctrole-specific RoPE/position/numeric paths.',
            'Context global5120norm feature gather precedes allcontextKV precompute;88 exactsource-order windows retain it. Movingonly downstreamKV projection toowners doesnotremove thatgather orrequire fullmainWreplication.',
            'Per-layer context/query rowpacking andcontextlazy schedules areexistingv35answers. Newjoint choice is howmuch globallyavailablecontext toreplicate versusownercompute whenF32alreadyexchanges projectedqueryKV.',
            'Currentcontext writes precede querywrites to thesource cacheexpression;actualslots/bankepochs/readers remainunbound. A rank-majorpacked exchange cannoterase per-slot C→QWAW ornoncausal readerreadiness.'],
        communicator_gate='Bindactualcommunicatormembers/rankorder andfullcontextavailability atallselectedcomputeowners. EqualrankSize/group label isnotidentityproof;extra rawcontext movementmustbepriced ifthepremisefails.',
        dummy_gate='48positiveNone-metadatazero sites do notqualifyforthequeryKVcarrier model. Bypassingtheirinputgather additionallyrequiresgroup-uniformbranch,warmup/memory-profile/capture/collective-order/error contracts.',
        loaded_instances_weight_replicas_cache_recipient_masks_or_native_rounding_proved=False,new_device_work=False,overall_complete=False))
    regimes=Counter((r['context_rows'],r['query_KV_rows'],r['layers'][0]['per_sender_hidden_rows_from_count'],r['retained_main_feature_allGather']['count']*2)for r in positive)
    models=[];examined=0
    for (C,Q,mq,main),population in sorted(regimes.items()):
        assert main==1280*C
        front=pareto_cases(C,Q,mq,main_bytes=main);threshold=min(C,9*8*mq)
        cases=[carrier_case(C,Q,mq,s,main_bytes=main)for s in dict.fromkeys([0,threshold,C])]
        models.append(dict(context_rows=C,query_KV_rows=Q,query_sender_rows=mq,P=8,profile_frontiers=population,
            full_context_is_distributed_only_in_declared_replicated_receiver_domain=True,
            boundary_cases=cases,model_Pareto_cases=front,all_s_options_checked=C+1,
            max_distributed_context_rows_without_exceeding_old_query_carrier_content=threshold,
            thresholds_are_not_hardware_optima=True))
        examined+=C+1
    write(O/'context-sp-hybrid-owner-models-v130.json',dict(regimes=models,all_s_options_checked=examined,
        sender_content_formula='new=1024*(m_q+ceil(s/8));old queryhidden=10240*m_q;retainedmain=1280*C once,three distinctlayer carriers.',
        dense_operator_work_formula='oldpercard=3*2*(C+Q)*5120*512;new=3*2*(C-s+ceil(s/8)+m_q)*5120*512. Counts arecurrent/hypotheticaldensework,notnecessaryworklowerbounds.',
        no_extra_carrier_content_condition='ceil(s/8)<=9*m_q, equivalent s<=72*m_q in thisuniformsend512BF16ABI.',
        padding='Newqueryprojection maycompute8*m_q rows foronlyQvalidrows;ownedcontext8*ceil(s/8) mayalsopad. No free8xfactor ordropofinvalidwrites.',
        scope='Conditional replicated-receiver/per-layer BF16-512 fixed-countAGcarrier aftercontext globalnormalreadyavailable. No packing/weightcache/latency/linkprotocol/numeric equivalence established.',
        current_HCCL_wire_bytes_or_speedup_proved=False,new_device_work=False,overall_complete=False))
    recipients=recipient_cases()
    write(O/'context-sp-recipient-owner-models-v130.json',dict(cases=recipients,case_count=len(recipients),
        equation='Foronecontextrowwith|R|receiversandkcomputeownerswithinR:kdenseprojectioncopies and|R|-kuncompressed512BF16endpointrowdeliveries,assumingfullrawcontextalreadyavailableatallcomputeowners.',
        alternatives=['Replicatedreceivers:R=all8;owner-onlycontextintroducesadditionalKVcarriercontent.',
            'Recipient-localcontext:k=|R|;noendpointKVdeliveries,computeonlywherecache/readersrequireit.',
            'Intermediatecomputeclusters tradeprojectioncopies againstendpointdeliveries;queryinputinitiallySP-ownedcannotborrowglobalcontextavailability.'],
        actual_cache_recipient_or_full_reader_sets_attested=False,endpoint_content_not_network_wire_lower_bound=True,new_device_work=False,overall_complete=False))
    witness=writer_witness()
    write(O/'context-sp-order-and-sharing-witnesses-v130.json',dict(writer_order=witness,
        value_namespace=dict(context_input=[1,0],query_input=[0,1],same_identity_weight=True,linear_outputs_distinct=True,
            sharedweight_isnotsharedvalue=True),
        schedule_choice=['Earlyjoint3contextmaps→one1536outputwith3separate512norms isexistingF29choice;queryKVstillperlayer.',
            'DelayedperlayerC/Qrowpacking canuseone512outputmapperlayerwithrole-specificnorm/RoPE/publish. It mayloseearliercontextready/overlap.',
            'Hybridcontextownerwork isonechosenjointgraph;cannotaddfull3→1contextlaunchcredit andfullperlayerC/Qlaunchcredit independently.'],
        preserved_boundaries=['Full5120mainnorm andretainedfeaturegather','Per-layerweight/gamma/eps andBF16GEMM/normrounds','Context/queryRoPEdtype/config/position/readers',
            'Localqueryrawhidden/QA/quant-readers andCP/SProwmaps','Invalid/padding/negative slots andallcachewriter/readerepochs'],
        no_actual_slot_or_native_race_or_fusion_parity_proved=True,new_device_work=False,overall_complete=False))
    nodes=[('MAIN','Existingglobalmainnormcontext'),('CR','Replicatedcontextsubset'),('CS','Ownercontextsubset'),('QH','Currentlayerlocalqueryhidden'),('QRAW','LocalqueryQA/otherrawreaders'),
        ('PROJ','ConditionalperlayerrowpackedWKV512'),('NORM','Separateperrow512norm/BF16'),('KVAG','Newtagged512carrier'),('CROPE','OriginalcontextRoPE'),('QROPE','OriginalqueryRoPE'),
        ('CW','Contextcachewrite'),('QW','Querycachewrite'),('READ','SWAreadrange'),('NEXT','Nextlayerqueryhidden')]
    edges=[('MAIN','CR','localreplicatedsubset'),('MAIN','CS','ownerselect'),('CR','PROJ','contextRrows'),('CS','PROJ','contextSrows'),('QH','PROJ','ownedqueryrows'),('QH','QRAW','retainedrawconsumer'),
        ('PROJ','NORM','full512row'),('NORM','KVAG','ownedtaggedrows'),('NORM','CROPE','localreplicatedCrows'),('KVAG','CROPE','receivedCrows'),('KVAG','QROPE','receivedQrows'),
        ('CROPE','CW','correctcontextslotvalues'),('QROPE','QW','correctqueryslotvalues'),('CW','QW','WAWonlyonactualoverlap'),('CW','READ','requiredcontextrows'),('QW','READ','requiredqueryrows'),('QRAW','READ','queryQready'),('READ','NEXT','serialdecoderlayer')]
    write(O/'context-sp-joint-graph-and-conditions-v130.json',dict(nodes=[dict(id=i,label=l)for i,l in nodes],edges=[dict(producer=a,consumer=b,dependency=c)for a,b,c in edges],
        pair=['F29','F32'],domain='SelecteddraftSWAonly sourcewithsameper-layerparameters andnewexplicitlocalQ/projectedKVABI;cache-demand andnumeric applicability unknown.',
        alternatives='Replicated/globalrecipient vsrecipient-local/storageowner ands-hybrid aredifferentgraphs,notindependentgains.',
        capacity='Currentmainweightsunchanged;includeexistingglobalcontextbank,oldcache/readers,newC/Qpacks/carrier/queues andKVweightworking-set underoneactualmemorybudget. No fitclaim.',
        readiness='Use actualper-range RAW/WAW/reader-retirement edges. No wholeoperator/globalbarrier inferred from thislocaldependencydiagram.',
        context_bank_lifetime='Delayingcontextprojectionuntileachquerylayerisreadyextendsglobalcontext-bankreaders fromearlyprecompute throughthelastselectedlayer. Preserveinteriterationbankreuse andnewpack/queuecapacity;no freeearlierretirement.',
        native_graph_nodes_added=0,new_pair_clauses=1,new_higher_order_groups=0,new_device_work=False,overall_complete=False))
    inputs=[O/'verify-metadata-postcheck-result-v129p.json',O/'candidate-relation-clauses-v129.json',O/'candidate-pair-navigation-v129.json',
        O/'context-sp-profile-frontiers-v130.json',O/'host-launch-original-API-coverage-v126.json',O/'draft-cache-join-native-frontiers-v113.json',O/'draft-cache-join-inputs-v113.json',O/'draft-cache-join-source-contract-v113.json',O/'draft-cache-slot-conflict-algebra-v113.json',
        O/'verify-metadata-profile-scope-v129.json',N/'context-sp-local-run-v130.json',
        O/'communication-native-operator-population-v66.json',O/'communication-role-contract-v16.json',O/'F27-F29-row-feature-and-KV-reorganization-v34.json',O/'projection-source-contract.json',
        O/'context-sp-query-assumption-error-v130a.json',N/'analyze_context_sp_profile_v130a.py',ds,mo,at,cp]
    write(O/'context-sp-inputs-v130.json',dict(inputs=[pin(p)for p in inputs],questions=[
        'Whichdraft/dummy SPcalls actuallyhavequeryKV,andwhichpositiveNone-metadatazero branchmustnotborrowthecarrier model?',
        'Arecontextandquery values identical,oronlytheirper-layerprojection parameters potentiallyshareable?',
        'When doesownercomputingcontext helpafterprojectingquerybeforeSPgather,andwhichcommonfeaturegather/padding/content remains?',
        'Whichcache-recipient,RoPE,writerorder,numeric,bank/readiness andschedule choices controlthejointrepresentation?'],
        profile_first=True,no_framework_or_newSDKresearch=True,new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='F29_F32_HYBRID_OWNER_RELATION_MODELED',positive_frontiers=88,dummy_frontiers=16,all_s_options_checked=examined,
        recipient_cases=len(recipients),new_pairs=1,new_device_work=False)))


if __name__=='__main__':main()
