"""Planner-specific cache/fusion/layout alternatives and finite boundary witnesses."""
from collections import Counter
from itertools import product
from finalize_origin_coverage_v9 import N,O,read,write

def main():
    assert not (O/'planner-fusion-costs-v71.json').exists()
    p=read(O/'planner-regime-models-v71.json');assert(p['regime_count'],p['captured_execution_count'])==(21,608)
    contract=read(O/'planner-source-algorithm-contract-v71.json')
    s_schema=contract['defined_schema']['SMLA'];q_schema=contract['defined_schema']['QLI']
    assert (s_schema['FAmax']*s_schema['FAwords']+s_schema['FDmax']*s_schema['FDwords'])*4==3600
    assert (q_schema['AICmax']*q_schema['AICwords']+q_schema['AIVmax']*q_schema['AIVwords'])*4==3456
    publication=[]
    for r in p['regimes']:
        w=r['work_model']
        if r['operator']=='ZerosLike':
            publication.append(dict(regime_id=r['regime_id'],executions=r['executions'],kind='caller-zero',full_bank_write_bytes=4096,total_profile_value_write_bytes=4096*r['executions'],fullzero_not_nativeempty=True))
        else:
            publication.append(dict(regime_id=r['regime_id'],executions=r['executions'],kind=w['family'],known_input_metadata_bytes=w['known_nonempty_input_descriptor_bytes'],defined_prefix_bytes=w['defined_prefix_bytes'],unwritten_suffix_bytes=w['unwritten_suffix_bytes'],source_bank_copy_reference_bytes=8192,total_source_bank_copy_reference_bytes=8192*r['executions'],copy_TRACE_TASK_identity_proved=False,cache_hit_work_model='Literalkeyscan/compare(ifneeded)+lookup+samebankpublication;AICPUlaunchnotremovedbyin-kernelcache.',input_key_generation_ready_before_skip_required=True))
    assert sum(x.get('total_source_bank_copy_reference_bytes',0)for x in publication)==4333568
    assert sum(x.get('total_profile_value_write_bytes',0)for x in publication)==323584
    # The declared source store model counts the whole zero-initialized struct and per-core overwrites.
    stores=[]
    for u in range(1,25):
        for v in range(49):
            s_words=900+(24-u)+8*u+(48-v)+7*v
            assert 4*s_words==3888+28*u+24*v
            if v==0:
                q_words=864+(24-u)+8*u+48
                assert 4*q_words==3744+28*u
            stores.append(dict(U=u,V=v,SMLA_source_value_store_bytes=4*s_words,SMLA_final_defined_prefix_bytes=3600,QLI_source_value_store_bytes=3744+28*u if v==0 else None,QLI_final_defined_prefix_bytes=3456 if v==0 else None,compact_A3_nonSplitG_endpoints_reference_bytes=4*(2+4*u+6*v),scope='Implicitprefixenable/start-from-previous-end;2counts+4end/workspacewordsperAIC+6FDwordsperAIV. NewconsumerABI;unchangednative1024ABIcannotacceptthisformat.'))
    # Source-level saturation law: sparse-original only schedules min(rawL, boundedtopklength).
    saturation_cases=0
    for top in [0,1,128,133,256]:
        for raw in range(0,34817):
            effective=min(raw,top)
            if raw>=top:assert effective==top
            blocks=(effective+511)//512
            assert blocks<=1
            saturation_cases+=1
    cost=lambda m,n:6*((m+15)//16)+10*((n+63)//64)
    assert cost(128,64)==58 and cost(128,65)==68 and (64+2047)//2048==(65+2047)//2048
    assert cost(128,65)==cost(128,128)==68
    # Only use this enumeration under uniform contiguous SP8 padding/query-layout conditions.
    sp_cases=[]
    for B,padded in [(2,16),(4,24)]:
        Q=5;active=B*Q;local=padded//8;plans=[]
        for rank in range(8):
            lo=rank*local;hi=min(lo+local,active)
            counts=[max(0,min((b+1)*Q,hi)-max(b*Q,lo))for b in range(B)]
            assert sum(counts)==max(0,hi-lo)
            qsl=[0]
            for n in counts:qsl.append(qsl[-1]+n)
            plans.append(dict(rank=rank,query_counts=counts,local_QSL=qsl,T=sum(counts),logical_zero=sum(counts)==0))
        nonempty=[x for x in plans if x['T']]
        assert len(nonempty)==(5 if B==2 else 7)
        sp_cases.append(dict(B=B,global_query_rows=active,padded_rows=padded,rank_plans=plans,positive_plan_families=len({tuple(x['local_QSL'])for x in nonempty}),profile_population_consistency_only='B2global32calls=4*8→20positive12zero;B4global56calls=7*8→49positive7zero;total69/19matches. Thisisnotactualtensorvalue/hitrate/phaseproof.',normalized_SWA_reuse_domain='SameQSL/heads/SoC/flags;origSparse,noCmp,effectiveTopK133andvalidrawKV>=133. Otherattrs/validationandepochconditionsretained.'))
    assert [x['positive_plan_families']for x in sp_cases]==[3,7]
    # Descriptor suffixes not written by this implementation cannot be used as deterministic plan IDs.
    padding_cases=[]
    for defined in [3600,3456]:
        a=bytes(defined)+bytes([17])*(4096-defined);b=bytes(defined)+bytes([33])*(4096-defined)
        assert a[:defined]==b[:defined]and a!=b
        padding_cases.append(dict(defined_prefix_bytes=defined,suffix_bytes=4096-defined,same_defined_plan=True,same_full4096_bytes=False))
    # Native empty-KV plan enables AIC0, whereas the caller's no-query branch writes all zero.
    zero=bytearray(4096);native_empty=bytearray(4096);native_empty[:4]=(1).to_bytes(4,'little');assert zero!=native_empty
    # Hard row work bounds for valid scheduledT<=8192,B<=4 in these A3 branches.
    qrow_cases=0
    for B in [2,4]:
        for lengths in product(range(7),repeat=B):
            R=sum((n+3)//4 for n in lengths);assert R<=(sum(lengths)+3)//4+B-1;qrow_cases+=1
    assert sum((n+3)//4 for n in [1,1,1,8189])==2051
    variants=[]
    def add(i,choice,work,conditions,state='CONDITIONAL_OPERATOR_OPTION'):
        variants.append(dict(id=i,choice=choice,work_or_value_IO_model=work,conditions=conditions,decision=state,legal_for_all_actual_inputs_proved=False,measured_speedup=None))
    add('PLAN-DIRECT-OUT','Publishplanstraighttostableexistingbank','529sourcecopies8192B/call;4,333,568Breferenceinterfacebudget. Planneralgorithm/definedprefixwritesremain.','NativeAPIoutputisprovidedmetadataTensor;newTorchoutABIvaliddevice/type/contiguity/shape;noinput/oldreaderconflict,capturebankaddress/frontierandunwrittenrawsuffixobservercontract.')
    add('PLAN-AICPU-CACHE','CachepuredefinedprefixwithintheAICPUoperator','Avoidcostscan/greedysplitonvalidhit;keyscan+lookup+outputpublicationandAICPUlaunchremain.','Exactsource/effectiveattrs/presence/length/QSL/residual/topklengthkey;fresh/resetstate;definedprefixonly,nofull4096memcmp;noinventedhitrate.')
    add('PLAN-PUBLISHER-CACHE','Reusealreadycurrentplanbeforelaunch','CanavoidAICPUinvocationonlywhencurrentkeytrustedwithoutblockingnewD2H;fixedbankpublicationstillrequiredunlesssamebankepoch.','KeymustcomefromauthoritativeGPUproducerorprovedconsistentmirror;capture/defer/reset/reuse/lastreaderfences;cachemissfallbackandallvalidation.')
    add('PLAN-SHAPE-POINTER','UseB/shapesorunchangedinputpointersasplankey','Excluded:kernelreadslength/QSL/residual/topklengthcontents;changingcontentschangesrow/cost/domainwithsamedescriptors.','Noactualvalueequalityfrompointeridentityorprofilenames.','EXCLUDED_BY_VALUE_DEPENDENCE')
    add('PLAN-SWA-SATURATION','Usecoarsersparse-original-onlySWAkeyacrossiterations','ForvalidL>=topLen,actOri=min(L,topLen)=topLen;Lvariationdoesnotchangeorderedrowcacheifotherfieldsfixed. W128,Q5fullvisible133isaconditionalconstant.','hasOri/oriTopK>0,noCmp,nonemptyrows,validtopLen/presence/sameQSL/heads/SoC/flagsandvalidationdomain. DoNOTapplytargetcausal/compressedorrawL<topLen.')
    add('PLAN-SP-FAMILY','PrecomputeSP8localSWAplanfamilies','ConditionalB2has3positiveQSLfamilies,B4has7;zero separate. Mappingpopulation69positive19zeroagreeswith88draftcallbacks.','UniformcontiguousSP8padded16/24,Q5,orderedrequests/CPmetadata/proven133saturation. Countsagreementisnotactualkeyvalues/hitrate;eachrankbankgenerationstillrequired.')
    add('PLAN-QLI-NORMALFORM','Cacheequalorderedrowbounds/costnormalforms','SourceA3mBase128,S2base2048,6ceil(M16)+10ceil(S264),FDdisabled. SamecoarseS2blockcountdoesnotproveequalcost.','AllusedSplitInfo/rowCache/cursor/FDfieldsandbatchquerylayoutsamebyinduction;APIvalidity preserved. Derivingnormalformcost/keymaycostasmuchassmallplanner;launchsavingsneedsupstreamtrustedkey.')
    add('PLAN-QLI-2048-ONLY','Cachebyceil(K/2048)alone','Excluded:K64and65bothoneblockbutnormalM128cost58vs68.','Retainrowbounds/tail-cost/emptyflagsatminimum;notproofeachsuchpairalwayschangesfinalplan.','EXCLUDED_BY_COST_COUNTEREXAMPLE')
    add('PLAN-DEVICE-PRODUCER','Fusematchingplanfieldswithlocalmetadata/SWA-lengthproducer','AvoidseparateAICPUlaunchforselectedsmallrowdomain;stillneedglobalcostreduction/orderedassignmentandallfinalplanfields.','MustsupplyLOCALQSL/effectivecachelengths/CPwindow/attrsreadyatproducer. GlobalSWAQSLnotdrop-inforlocalSMLA. Preserveexactpartition/reductiontreeorproveallowedprecision;prefill/otherflagsfallback.')
    add('PLAN-BATCH-SHARE','Batchmultipleplansorshareequalkeysacrosslayers','Commoninputmetadataread/launchcanbeamortized;distinctmode/head/ratio/topk/rowlayoutkeysstillseparate;materializedbanksremain.','Inputsreadytogetherandallobservers/frontiers;doNOTdelayearlierneededplanorignoreSMLA-vsQLIschemadifference;addressesunboundvaluesnotanyequalityproof.')
    add('PLAN-CROSS-RANK','Shareplanvaluesacrosscardswithidenticaleffectivekey','Plancontainsrelativeworkspaceindices/ranges,notinput/cacheaddresses. Full4096broadcastandbarriercostmustbepricedagainstlocalplanwindow.','SameSoC/corecaps/source/flagsandLOCALrequestrows/lengths;CPmapsnormallydiffer. Noassumedrank0planuniversal;allconsumerbank/publicationconditions.')
    add('PLAN-COMPACT-ENDS','Reorganizefixedmetadataintoprefix-enable/endpoints','A3nonSplitGnewABI4*(2+4U+6V)Bversusdefined3600/3456;startsreconstructedfrompreviousend,enabledcoresimplicitprefix.','Allnativeconsumers/tiling/1024shapechecksneednewABI;reserved/maxS2fieldsimplicitonlydeclaredA3nonSplitGdomain. Partition/workspace/reductiontreeunchanged;cannotpasscompactbuffertounmodifiedop.')
    add('PLAN-FINAL-WRITE','Constructfinaldefinedprefixinsteadofzero-then-overwrite','SourceSMLA3888+28U+24Vvaluestorebytes→3600final;QLI3744+28U→3456(A3V0).','Compiler/CPUcachemayalreadycoalescestores;notHBMsavingclaim. Everyreserveddefinedwordzero;sourceprefixnotentire4096mustbeobservablyequivalent.')
    add('PLAN-ZERO-BANK','Reuseorconditionallyelide79callerzero-clears','Nativezeroswrite323584Btotal;establishedimmutablezeroentryorper-bankzero-invariantmayavoidrepeats.','Noquery/allreaderandlatestwriter/resetproof;retainATTENTIONcompletionfrontier. Copyingotherzerobankstill8192Bmaycostmorethan4096zero;doNOTreplacenativeall-emptyplan.')
    add('PLAN-EMPTY-STATIC','Cacheactualnativeall-emptyKVplanseparately','All-emptyBalanceScheduleenablescore0withBN2endB*Nkv,thenzerosdefinedstructandfillsranges.','Correctbatch/cores/schema/attrsandvalidatedall-zeroKVorQcondition;callerlogical-zeroall-disabledbankisdifferent. Keys/padding/frontierskept.')
    add('PLAN-SCHEMA-SWAP','ReuseSMLAblobasQLIblob','Excluded:FAstride9/FDoffset324wordsversusQLIstride8/FDoffset288words;partitionalgorithmsalsoS2base512vs2048.','Same1024shape/dtypedoesnotestablishcompatibleplan.','EXCLUDED_BY_SCHEMA_COUNTEREXAMPLE')
    write(O/'planner-fusion-costs-v71.json',dict(publication_models=publication,store_models=stores,conditional_SP8_SWA_families=sp_cases,source_bank_copy_reference_bytes_total=4333568,caller_zero_value_write_bytes_total=323584,copy_counts_scope='529sourcesemanticcopiesvalued,nottheexistenceof529separatelyboundComputeInfoMemcpyTASKs.',SMLA_row_work='R<=scheduledT<=8192;A3window128ororigTopK256givesori<=1blockandcmpTopK512givescmp<=1blockperrow. UnderthisdomainK<=2R. DistinctG8/64/otherflagskept.',QLI_row_work='R=sumceil(q_i/4)<=2051underT<=8192,B<=4. A3FDfalsebypinnedalgorithm,soAssignByBlocknotexecuted. NormalblockM128/S22048cost368heuristicunitsnotcycles. UnknownactualQSLpreventsperexecutionR.',theoretical_time_bound='max(actualAICPUinstruction/branchdependencycost,actualCPU/device-cachememorycost,launchandrequiredpublicationfrontiercriticalpath)inscope. 1.6TB/sHBM/Cube/VectorpeaksarenotAICPUcapacity. SavedTASKdurationisnotrequestgain.',all_physical_numeric_ceilings_proved=False,whole_modules_closed=0,overall_complete=False))
    write(O/'planner-fusion-variants-v71.json',dict(variants=variants,variant_count=len(variants),whole_modules_closed=0,overall_complete=False))
    write(O/'planner-math-validation-v71.json',dict(sparse_original_saturation_cases=saturation_cases,A3_store_UV_cases=len(stores),QLI_small_query_histogram_cases=qrow_cases,QLI_same2048bucket_cost_counterexample=dict(K=[64,65],blocks=[1,1],M=128,cost=[58,68]),QLI_finer_samecost_example=dict(K=[65,128],M=128,cost=[68,68]),undefined_suffix_memcmp_counterexamples=padding_cases,callerzero_vs_nativeempty_core_enable=[0,1],SP8_B2_positive_zero=[5,3],SP8_B4_positive_zero=[7,1],conditional_positive_QSL_family_counts=[3,7],actual_69_19_population_is_not_hitrate_proof=True,no_model_Torch_NPU_native_test=True,overall_complete=False))
    print(dict(variants=len(variants),publication_models=len(publication),store_cases=len(stores),saturation_cases=saturation_cases,qrow_cases=qrow_cases,source_copy_reference=4333568,caller_zero_writes=323584,SPfamilies=[3,7]))

if __name__=='__main__':main()
