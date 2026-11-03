"""F28/F32 first draft owner graph, copy initialization and cross-iteration row reuse."""
from pathlib import Path
import hashlib,json,math
from finalize_origin_coverage_v9 import N,O,W,R,read,write
from init_sp_models_v132 import copy_cases,route_cases,content,cache_epoch_cases


def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())


def main():
    assert not (O/'REVIEW-CHECKPOINT-v132.json').exists()
    root=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend';parts=[]
    for rel,a,b in [('models/deepseek_v41/dspark.py',248,274),('models/deepseek_v41/model.py',887,920),('models/deepseek_v41/model.py',397,405),
        ('models/common/ops/sequence_parallel.py',25,43),('models/common/ops/sequence_parallel.py',143,152),
        ('attention/context_parallel/dsa_v41_cp.py',188,196),('attention/context_parallel/dsa_v41_cp.py',209,225),
        ('attention/dsa_v41.py',350,382),('attention/dsa_v41.py',690,717)]:
        p=root/rel;parts.append(dict(**pin(p),first_line=a,last_line=b,source=p.read_text().splitlines()[a-1:b]))
    prof=read(O/'init-sp-native-frontiers-v132.json');assert prof['source_order_frontiers']==104
    write(O/'init-sp-source-contract-v132.json',dict(parts=parts,facts=[
        'CurrentdraftglobalembeddingARprecedesHC4repeatthenSPpad/shardandonehotFP32premix. FirstHcPrecoeff/pre/post/comb remainsliveevenifcollapsedYpipelineis split orcached.',
        '104exactkeyinitfrontiershavefirstHcPre/RmsNorm/SP;69positiveQA+globalKV,19positivepreprocess354localqueryempty/globalKV,16positiveNonezero. NoQisnotdummy andnotnoglobalKV/outputcollectives.',
        'SPownerownsceil(N/P)contiguouspaddedrows;activeCPqueryslicesstartandactualtokensindependently. ActualSP/CPincidenceisnotrecorded,despitematchingmanysmallMshapes.',
        'WQAisReplicatedLinear1280andselectednativeQAhasBF16output;WKV512/full512normisSWA-only. MovingQAtoSPownerthenroutingBF16QAcanalignjoint1792projectiononlywithmatchingweights/nativeM/rounds/allrawreaders. PreservefullQA1280normandlaterqueryprojection/headownership.',
        'StandardmaskedembeddingARfollowedSPmayusematchingorderedSUMRS,thendirectlocalorvirtualHCinit,thenY_BF16/full5120norm_BF16/ownerKV;otherembeddingplans/fullEreaders/alias/customopclone/numericSUM remainseparategates.',
        'HCcopylayoutreorderdoesnotfoldfull4Hcoefficientprojection/norm orremovefirstPostresidual. CurrentexternalonehotYstillhasliteral0*Inf/NaN/productsandreduction;notunconditionalEidentity.',
        'MetadataNonemaykillonlydeadattentioninputbranchunderuniformgroupandallreaders/effects. Localqueryemptykeepsglobalcacheandhead/outputcollectives;maskTruevalidprefixdoesnotproveattentionKVrowdead.',
        'Firstlayerpre-RoPEKVandrawQArowfunctioncanbeimmutable-tokenmemoizedonlywithcompleteembedding/parameter/math/nativeM/branch epochs andexactnumericcachefill. AllglobalIDs+replicatedcachecouldreplacefirstKVgather,whilelocalembedding/HCcoeff/residualremain.',
        'CachebeforeRoPEcannotreusefinalpositionedKV;SPpaddingzerorowisinitspecificandnottoken0embedding. CurrentNoSpec/querydummy/readers/tailsmustretainstate/shape/errors/capture. No cachehit/reuse distribution ispresent.'],
        actual_ownership_parameter_epochs_native_parity_or_allreaders_bound=False,new_device_work=False,overall_complete=False))
    copies=copy_cases();routes=route_cases()
    write(O/'init-sp-copy-and-mask-models-v132.json',dict(case_count=len(copies),cases=copies,scope='RawBF16bitcopyonlyincludingzero/signedzero/Inf/NaNpayloadpatterns;N0..24,P1..8,allranks,None/true/alternatingandsmallprefixmasks. Notnativearithmeticoraliassafety.',new_device_work=False,overall_complete=False))
    write(O/'init-sp-QA-owner-routing-models-v132.json',dict(case_count=len(routes),cases=routes,
        scope='N1..24/P1..8/allcyclicquery-ownerpermutations;independentintegerQArowfunctionverifiesrowtags/order. NotactualCPmaporCANNprecision.',
        alternatives=['KeepactualQAlocalrawrowsifalreadycolocated.','Routeonlymissingraw5120rows.','ProjectQA1280atSPowneranddeliverrow-taggedQA.','UniformKV512+QA1280gather1792withnewfull-outputABI.'],
        all_current_raw_consumers_closed=False,new_device_work=False,overall_complete=False))
    write(O/'init-sp-joint-work-content-v132.json',dict(geometries=[content(10),content(20)],
        domain='BF16STANDARDembeddingSUM/TP8andnewfixed-ownerfirstlayer512KV+1280QAABI. Countsaredeclaredtensor/valuecontentsandcurrent/candidatedensealgorithmrows;notunconditionalworkorwire/HBM/latencybounds.',
        nonadditive='CountglobalTile/padremovalonlyoncewithlocal/virtualHCinit;donotaddtheirfullcredits. F32firstnorm/SPinterfaceandvirtualHCYshareonegraph. AR→RSinputcanincreasebypadding;outputcontentdecreasesbutwiredependsalgorithm.',
        QA_padding='UniformownerjointprojectioncomputesAQArowsversusNcoveringvalidrows;demand-prunedalternativehasitsownroutedlayoutandcost. Originalquery-emptyrankcannotbesilentlyremoved.',
        existing_theory='Physicalcapacitiesandallreaderclosureunbound:SYMBOLIC_CEILINGonly. No actualMfU/lowest/criticalpathgain.',new_device_work=False,overall_complete=False))
    cfg=read(R/'evidence/model-config.json')['text_config'];assert (cfg['vocab_size'],cfg['hidden_size'],cfg['q_lora_rank'],cfg['hc_mult'])==(129280,5120,1280,4)
    V,H,D,Q=cfg['vocab_size'],5120,512,1280
    write(O/'init-sp-first-layer-token-reuse-v132.json',dict(config=pin(R/'evidence/model-config.json'),V=V,
        value_content=dict(full_replicated_pre_RoPE_KV_bytes=2*V*D,full_replicated_KV_and_raw_QA_bytes=2*V*(D+Q),
            KV_per_cached_token_bytes=2*D,KV_QA_per_cached_token_bytes=2*(D+Q),KV_full_vocabulary_dense_MAC_reference=V*H*D,
            KV_QA_full_vocabulary_dense_MAC_reference=V*H*(D+Q)),
        exact_function='Token→originalembeddingSUM_BF16→virtualinitialHC/externalonehotcollapsedY_BF16→inputRMS5120_BF16→WKV_BF16→KVnorm512_BF16;QAplaneisrawWQA_BF16beforeQA1280norm. Keepcurrentproduct/reduction/rounds,nativeMpolicy,errors/nonfinitebehavior.',
        cache_key='Firstdraftlayeridentity,token/domain,embedding/quant/reduceplan+parameter/gamma/eps+mathmode/nativeM/layout/adapterepochs;fixedoutcontractsandloadedobjects. Position/RoPE/slot/cacheepochappliedafterlookup. Actualstableepochsunknown.',
        table_storage_scope='Eachfull-vocabularycontentcountisforonespecifiedfrozenkeyversion. SimultaneousnativeM/math/parametervariantsmultiplyresidentcontent;otherwiseinvalidate/refill. No one-table-all-Mparityassumed.',
        protocols=['DemandcacheatSPownersreplacesonlyhitprojectionwork;KVgatherstillneededforreplicatedcacheconsumers.',
            'Replicatedall-tokenplaneswithallneededglobalIDsallowrecipientlookupinsteadfirstKVgather;priceNlookup/card,IDavailability,tablecontent,fill/invalidation/capture/readiness andalloldreaders.',
            'LazycacheKuses2K*512or2K*1792valuecontentpluskeys/metadata/queues;actualhits,evictionandmaintenanceareunknown. Coldfillandepochchangesarework,notfreeoverlap.'],
        finite_epoch_witnesses=cache_epoch_cases(),
        restrictions='OnlyfirstunchangedinitialHCgeneration;laterdecoderstate/context/positionedcachevaluesareotherfunctions. Stillneedrawembedding/localHCcoeff/pre/post/comb/firstPostresidualunlessseparateall-consumerproof. Precomputeisoptionaladditionalwork;tablesarevaluecontentsnotallocatedVRAMorHBMnecessarybytes.',
        actual_token_duplicates_hits_epochs_native_equivalence_or_benefit_established=False,new_device_work=False,overall_complete=False))
    write(O/'init-sp-selected-graph-and-guards-v132.json',dict(pair=['F28','F32'],nodes=['EMBED_PARTIAL','EMBED_RS','INIT_VIRTUAL_HC','HC_COEFF','Y_BF16','NORM_BF16','OWNER_KV','QA_ROUTE','KV_CARRIER','CURRENT_ATTN','HC_POST','NEXT_FFN'],
        edges=[['EMBED_PARTIAL','EMBED_RS'],['EMBED_RS','INIT_VIRTUAL_HC'],['INIT_VIRTUAL_HC','HC_COEFF'],['INIT_VIRTUAL_HC','Y_BF16'],['Y_BF16','NORM_BF16'],['NORM_BF16','OWNER_KV'],['NORM_BF16','QA_ROUTE'],['OWNER_KV','KV_CARRIER'],['KV_CARRIER','CURRENT_ATTN'],['QA_ROUTE','CURRENT_ATTN'],['CURRENT_ATTN','HC_POST'],['HC_COEFF','HC_POST'],['INIT_VIRTUAL_HC','HC_POST'],['HC_POST','NEXT_FFN']],
        no_unproved_edge_from_current_HCcoeff_to_Y=True,
        numeric_witness=dict(onehot_times_repeated_Inf='1*Inf+0*Inf+0*Inf+0*Inf yieldsNaN;directembeddingInfnotNaN',checked=math.isnan(1*math.inf+0*math.inf+0*math.inf+0*math.inf)),
        alternatives=['StandardARretainedwithlocal/virtualinitplusownerprojection.','MatchingSUMRSembeddingandnewinit/norm/projectedcarrier.','CurrentCPQAwithrawmissingrowsornewownerQAtransport/joint1792plane.','UniformmetadataNonedeadattentioninputbranch.','Firstgenerationtokenrowmemoizationwithdemandorreplicatedcache.'],
        all_alias_parameters_precision_events_masks_and_loaded_variants_bound=False,new_native_nodes=0,new_pair_clauses=1,new_groups=0,new_device_work=False,overall_complete=False))
    names=['aux-sp-postcheck-result-v131p.json','candidate-relation-clauses-v131.json','candidate-pair-navigation-v131.json','init-sp-native-frontiers-v132.json',
        'context-sp-profile-frontiers-v130.json','embedding-initialization-native-chains-v67.json','embedding-initialization-operator-contract-v67.json','embedding-initialization-fusion-variants-v67.json',
        'M19-embedding-HC-CP-row-and-consumer-contract-v35.json','HC-operator-numerical-and-dependency-contract-v59.json','cache-only-KV-branch-reconciliation-v18.json',
        'communication-native-operator-population-v66.json','coverage-current-obligations-v131.json']
    inputs=[O/n for n in names]+[Path(p['path'])for p in parts]+[R/'evidence/model-config.json']
    write(O/'init-sp-inputs-v132.json',dict(inputs=[pin(p)for p in sorted(set(inputs))],questions=['Whichfirstinit/HC/norm/SP/QAvsKV/dummybranchesarepositivelyrecorded?',
        'DoSPandCPownersalign,andwhatraworQAplanesremainwithprojectedKV?',
        'HowdoembeddingSUM/localorvirtualHC/firstnormcarrierandfirst-tokenreusecomposewithoutdoublecreditorlosingcoeff/residual/rounds?'],profile_first=True,new_device_work=False,overall_complete=False))
    print(json.dumps(dict(status='F28_F32_FIRST_INIT_OWNER_AND_TOKEN_REUSE_REVIEWED',copy_cases=len(copies),route_cases=len(routes),new_scoped_pair=1,new_device_work=False)))


if __name__=='__main__':main()
