from copy import deepcopy
from datetime import datetime,timezone,timedelta
import hashlib,json
from finalize_origin_coverage_v9 import N,O,R,W,T,L,read,write,ctl
from finalize_F33_consumers_v27 import graph_check

def packet(label,embedded=False):
    r=read(N/(label+'.json'));assert r['returncode']==0,label
    x=json.loads(r['stdout'])
    if embedded:
        b=(json.dumps(x['result'],ensure_ascii=False,separators=(',',':'))+'\n').encode()
        assert hashlib.sha256(b).hexdigest()==x['result_sha256'],label
    return x

def graph(nodes,edges):
    d=dict(nodes=[dict(id=x)for x in nodes],edges=[dict(a=a,b=b,kind=k)for a,b,k in edges]);graph_check(d);return d

def main():
    assert not(O/'REVIEW-CHECKPOINT-v30.json').exists(),'refuse duplicate finalization'
    now=datetime.now(timezone.utc);stamp=now.isoformat()
    p=packet('Engram-profile166-v30');s=packet('Engram-source166-v30',True);k=packet('Engram-kernel166-v30-v2',True);a=packet('Engram-root-audit166-v30-v2')
    ps=p['summary'];ss=a['summary'];sp=read(O/'F24-SP-row-shard-contract-v30.json');old=read(O/'hc-engram-tail-reconciliation-v6.json')
    assert read(N/'Engram-kernel166-v30.json')['returncode']==1
    assert read(N/'Engram-root-audit166-v30.json')['returncode']==1
    assert (ps['native_selected_APIs'],ps['native_selected_TASKs'],ps['native_fullchains'],ps['graph_Engram_segments'],ps['source_bound_collectives'])==(3536,3632,26,160,176)
    assert (ss['prepare_roots'],ss['roots_with_hash'],ss['collectives_unbound_or_ambiguous'],ss['native_unbound_or_ambiguous'],ss['graph_segments_unbound_or_ambiguous'])==(96,88,0,48,32)
    assert ss['graph_roles']==dict(WKV=160,qk_parameter_preparation=480,ordinary_gate_including_block_rotation=5440)
    assert ss['native_roles']['qk_parameter_preparation']==144
    assert sp['sha256']=='73bcf07050bce8b049dd0bbdb0a7504f6eba157a1e97a5033849ef31ef05fcb0'
    refs=[dict(label=label,receipt=str(N/(label+'.json')),full_result=x['full_result'],sha256=x['result_sha256'])for label,x in[('Engram-profile166-v30',p),('Engram-source166-v30',s),('Engram-kernel166-v30-v2',k),('Engram-root-audit166-v30-v2',a)]]
    files={f['path']:f['sha256']for x in[s,k]for f in x['result']['files']}
    files[sp['path']]=sp['sha256']
    common=next(f for f in old['source_provenance']if f['path'].endswith('/engram/common.py'));files[common['path']]=common['sha256']
    write(O/'Engram-current-profile-and-root-roles-v30.json',dict(generated_utc=stamp,profile=ps,root_and_typed_role_audit=ss,native_chain_headers=p['native_chain_headers'],evidence_refs=refs,
        scope='All32savedDBparts;26nativefullchains;80graphswithtwoEngramoutersegments;completeCOMMfullcallers. Thisancestryselectionisnotv20functionalPMUbucket.',
        counts=dict(native_compute=3632,graph_compute=6080,ancestry_selected_compute=9712,source_consistent_body_instances=208,WKV=208,qk_parameter_preparation=624,ordinary_gate=7072,native_other_preparation=1808),
        not_closed=['48nativeTASKsunboundtouniqueprepareroot;notassignedbycount/nearesttime.','32Engramsegmentsbelongto16phase-unboundgraphs;allretained.','176COMMrecordsand128ComputeInfohcomtasksaredifferenthierarchicalpopulations;not48lostcollectives.','Countsarenotlatency,actualcachehits,bytegenerationorresourcecapacity.'],whole_module_closed=False,overall_complete=False))
    contract=dict(generated_utc=stamp,evidence_grades=['MEASURED','DOCUMENTED','DERIVED','UNKNOWN'],source_files=files,evidence_refs=refs,
        reviewed_facts=[
            '88hash/req-index/cache-writetriplesand176UVAlookupsarepositivelypresent;onehashcallcoversallconfiguredlayers,andoneDPgatheralreadyfeedsbothlayers. Alreadyimplementedsharingisnotnewgain.',
            'Thetwoengram_dead_maskcallerchainsmodel1167/currentIDsandmodel1188/lookbackIDsoperateondifferenttensors. Eachhas88logicalOR/176equalTASKs;cannotdedupbyfunctionnameorcount.',
            'Optionalvalid_token_countextendscurrentdeadmask;token_indicesmayselecthash/maskrows. Maskispublishedbeforehash/tableexchange;all-deadranksmaystillneedDPparticipation. Noactualoptionalflag/maskvaluesarefabricated.',
            'Hashusesreq-indexoncepertoken,thenalllayers. Selectedkernelreads in_batch currentchunkfirst,knownnonnegativewindow_tokensecond,elseoptionalcache. Unknownlookback-1isnotavocabularytoken;negativepositionsandDEAD_IDblocktherollingngram.',
            'CachewriterreadsrawIDs/token_map/currentdead/slot_mappingandwritesmappedorDEAD_IDonlyforvalidtokenswithslot>=0. HashStateorderswritebeforehash. Historicalcachecontents,hitcounts,slotsandnext-generationreadersarenotrecorded.',
            'CurrentUVAlookupwrapperlaunchesrows=tokens*local_heads. Kernelprogram(row)mapsbijectivelyto(token,head_local),storesallWIDTHcolumns;ownedfalseexplicitlystoreszero. Itneverreadstheoutput.',
            'embed_gathered343allocateszeros[tokens,part_n_hash_cols,dim];lookupsetsLOCAL_HEADS=PAD_HEADS=part_n_hash_colsandcontiguousout.view. HenceinthecurrentnonemptyUVApathallallocatedelementsareoverwrittenbeforelookupreturns. Thisisalocalsource-domainproof,nottargetimplementationchangeorbinaryparitytest.',
            'Actualtypedlookupshapesare[M,3,256],notguessed6x128. IDsare[M,24],UVAoutput[M*3,256]. TP8gatherreconstructs24heads*256=6144features. Mregimes24,6400,6528,6672matchCOMMcountsM*3*256.',
            '176TP8BF16AllGathersuniquelybindto88roots,twoeach:64rootsM24with128root-boundgraphsegments;24longrootsM6400/6528/6672withnativeWKVrows800/816/834. Separately32segmentsfrom16graphshavenouniqueroot. 8rootsareunassociatedwithselectedhash/lookupwork;thisisnotaproofofnothingexecuted.',
            'Currentmodelwaitsmaskandper-layerlookupready,thenSPshardslookupbeforeWKV. Exactimportresolvedsp_shardimplpadsdim0totheTPmultiplewithzeros,selectscontiguousrankchunk,andcloneswhenalreadydivisible. Theconditionalrow/headexchangeidentitypreservesthisrowmap.',
            'DProwexchangeoccurspreTPheadgather. Currentnon-directoutputcopiesflattenedvaluestofixedglobalbanksandzerosbanktail;directoutputshortcutrequiresDP=TP=1. AnewSP-localbankisadifferentinterfaceandmustretainallotherconsumers,capturedaddresses,tailsemanticsandper-layerreadygenerations.',
            'Eachsource-consistentbodyhasWKV1,q/kcast+product3,ordinarygate34. Graph160plusnative48=208bodies;parameterpreparation480+144=624TASKs. q*khas4x5120FLOAT=81920logicalbytesperlayer;theactualrotationblockis32x32FLOAT=4096logicalbytes,alreadyblockdiagonal.',
            'Ordinarygateunrotateshidden.floatin32-featureblocks,computesseparatefull5120Dhidden/keyRMSandweightedFP32dot,signedsqrt/sigmoid/mask;addsvaluegatedtotheoriginalhidden.floatresidual,notunrotatedhidden. Value5120issharedbyfourHCstreams. Reusev6formulacontractratherthanreopeningentiregate.'],
        source_questions_locally_answered=['CurrentUVAoutputfullwritevslookup-prezero','Current/lookbackdeadmaskinputdistinction','Selectedhashhistorypriorityandcachewritefootprint','CurrentSPpad/chunkmapandper-layerlookuppublication','Actualparameterandblockrotationshapes'],
        retained_unknowns=['Actualslotcachehit/miss/acceptedhistory/request-remap/rollbackgenerationsandallfuturestateobservers.','Loadedbinarylink,numericalparity,newtileinterfaces,pointers/aliases/captured-bankgenerationandready/lastreaderorder.','AllnonSP/DPshared/TP1/CPU/GPUweight/slotless/force_dummy/empty/image/validcount/indicesbranchesandfutureworkloadcoverage.','CPUhash/UVAhostmemorypage/PCIe/cachecostandcriticalpathnotgivenbyComputeInfo;noCPUtimezero-fill.','Allmodules/conditionsnotclosed;v20resourceattributionand26INCOMPLETEtheoryrowsunchanged.'],
        target_implementation_changed=False,whole_module_closed=False,overall_complete=False)
    write(O/'Engram-hash-lookup-publication-and-gate-contract-v30.json',contract)
    # Exact store-domain argument, without importing/executing the model or a device kernel.
    coverage=[]
    for M,count in[(24,128),(6400,16),(6528,16),(6672,16)]:
        H,D=3,256
        assert M*H*D in[18432,4915200,5013504,5124096]
        coverage.append(dict(tokens=M,local_heads=H,pad_heads=H,width=D,TASKs=count,grid_rows=M*H,output_elements=M*H*D,source_role='embedding343ZerosLike->lookup397->UVAkernel229',old_output_read=False,invalid_id_value='BF16zero'))
    fullwrite=dict(generated_utc=stamp,evidence_grade='DERIVED_FROM_PINNED_SOURCE_PLUS_MEASURED_TYPED_DOMAINS',candidate='F24',variant='F24-v30-PREZERO',
        proof='ForT>0,H>0,D=WIDTHandPAD_HEADS=LOCAL_HEADS=H,gridr=0..T*H-1mapsr=tH+h. Storescoveroffset(tH+h)D+cforallc=0..D-1exactlyonce,sooffsetsareexactly0..THD-1. ownedfalsewriteszero;oldoutputneverloaded. Thusinitialoutputvaluescannotaffectnormalcompletionresults.',
        actual_domains=coverage,sufficient_conditions=['CurrentselectedUVAkernelandmatchinglaunch,normalcompletion,positiveT/H,validinputpointers.','ContiguousBF16outputwithsameextentandPAD_HEADS=LOCAL_HEADS;noalias/observerbeforefullwrite.','Preservesamefixedcaptureaddresses/lifetime/readinessandallrequiredoutside-extenttailinitialization.'],
        exclusions=['PAD_HEADS>LOCAL_HEADSleavespaddedheadrowsunwrittenunlessseparatelyinitialized.','Do notdeleteforce_dummy/output-banktailzerosormaskzeros.','CPU/GPUlookupbackendsanddifferentkernelsrequiretheirownwritecoverage.','Fullwriteproofdoesnotproveallocator/capturelegality,performancegainorloadedbinaryidentity.'],
        local_question_closed=True,implemented=False,native_validation=False,critical_path_gain_us=None,overall_complete=False)
    write(O/'F24-UVA-fullwrite-and-prezero-review-v30.json',fullwrite)
    # Pure data-permutation identity: gather heads then shard rows vs send destination rows then concatenate heads.
    examples=[]
    for M in[23,24]:
        P,H,D=8,3,2;Q=(M+P-1)//P
        local=[[[[(rank+1)*100000+row*1000+head*10+c for c in range(D)]for head in range(H)]for row in range(M)]for rank in range(P)]
        gathered=[[head for rank in range(P)for head in local[rank][row]]for row in range(M)]
        gathered += [[[0]*D for _ in range(P*H)]for _ in range(P*Q-M)]
        equal=[]
        for dest in range(P):
            baseline=gathered[dest*Q:(dest+1)*Q]
            exchanged=[]
            for j in range(Q):
                row=dest*Q+j
                exchanged.append([head for src in range(P)for head in(local[src][row]if row<M else[[0]*D for _ in range(H)])])
            assert baseline==exchanged
            equal.append(True)
        examples.append(dict(tokens=M,TP=P,local_heads=H,width=D,rows_per_rank=Q,zero_padding=P*Q-M,all_ranks_equal=all(equal)))
    exchange=dict(generated_utc=stamp,evidence_grade='DERIVED_CONDITIONAL_GRAPH_REORGANIZATION',candidate='F24',variant='F24-v30-ROW-HEAD-EXCHANGE',
        baseline='X_r=concat_heads_s B_s[M,h,d]; pad_rows(X_r,P); take_rows(rank_r,Q); trim_heads(n_hash_cols);flattenandfeedlocalWKV.',
        candidate_graph_identity='Eachsources sendsto destinationr onlyB_s[rows_r,h,d],includingidenticalzero-rowpadding. DestinationrconcatenatesreceivedheadblocksinsameTP-rankorderandappliessameheadtrim/flatten. Entry(r,j,s,h,c)equalsB_s[rQ+j,h,c]orzeroinbothgraphs;noarithmeticreductionorreassociation.',
        actual_signature=dict(TP=8,local_heads=3,width=256,global_heads=24,global_features=6144,rows=[24,6400,6528,6672],rows_per_rank=[3,800,816,834]),
        finite_illustrations=examples,
        prerequisites=['SameTPgroup/rankorderandphysicalrowgeneration/rowplanonallparticipants;DProwexchangealreadycomplete.','UseexactSPzero-pad/contiguous-rankchunkmapandheadtrim;retainscheduled/padded/validrows,sentinelsandmetadata.','Allconsumerscovered:ifanyreaderneedsfullglobalrows,retainitorgiveaseparateview;do notdropfullbankunconditionally.','Newlocalbank/captureABIhasfixedaddresses,correctupdate/lifetime/tails,andreadyisrecordedonlyafterallitsrows/headsarevalid.','Keepmaskreadyandper-layerpublication;groupingbothlayersmustnotdelaylayer1withoutanexplicittradeoff.','TP1/nonSP/unequalmetadata/capture/dummy/empty/DP-sharedvariantsretaintheirrequiredprotocol.'],
        logical_payload_only='ForMdivisiblebyP,remoteelementsreceivedperrankare(P-1)*M*h*dforAllGatherand(P-1)*(M/P)*h*dforrow/headexchange;thelogicalpayloadratioisP. Thisisnotphysicalwirebytes,HBMtraffic,collectiveschedulingorlatencyspeedup.',
        native_or_interface_validation=False,performance_gain=None,whole_module_closed=False,overall_complete=False)
    write(O/'F24-row-head-exchange-algebra-v30.json',exchange)
    history=dict(generated_utc=stamp,candidate='F24',variant='F24-v30-HISTORY',evidence_grade='DOCUMENTED_PRIORITY_WITH_CONDITIONAL_LIFETIME_REWRITE',
        exact_priority=['in_batch:input_ids/token_map/currentdead','elseknownlookback(window_token>=0):lookbackIDs/token_map/lookbackdead','elseoptionalcachefallback;slotlessusespad_id','negativepositionorDEAD_IDsetstickyblocked;blockedemitsPAD-derivedhash'],
        already_implemented=['Req-indexsearchhoistedoutofper-layerhash;currentbackendcommentreportscombinedsearch/bodycompilerabort,notsemanticimpossibility.','Hashalllayersinonecall;DPgatheralreadysharedbyalllayers.','Currentchunkandlookbackoverridecache;existinguse_slot_cacheFalsebranchisnotnewimplementation.'],
        sufficient_elimination_domain='Forallrelevanttokens/shiftsinwholedeclaredlifetime,everyrequirednonnegativehistorypositionoutsidecurrentchunkhasaknowncorrectrequest-generationlookbackentry,orindependentlyproveddead/padresult;noothercurrentorfuturecacheobserverrequireswrites. Thencachefallbackcannotchangetheemittedhashandcachewritescanbeomittedinthisdeclaredslotlessvariant.',
        counterexamples=['Lookbackmatrixwideenoughbutentry=-1:knownisfalse;nonnegativehistorymaycomefromcacheandaffecthash.','Thisstepallhitsdoesnotexcludeafuturestepfallback;droppingtodaywritecanchangefuturecachecontents.','Request/slotreuse,acceptedtokens/rollback,multimodalDEAD_IDandasyncwindowgenerationrequireconsistentmapping.','Twoengram_dead_maskcallsaredifferentdomains;lookupfullwriteproofdoesnotmakehashcachewriteadeadstore.'],
        actual_history_coverage_or_hit_count='UNKNOWN_NOT_RECOVERED_FROM_PROFILE',current_cache_writes_observed=88,current_cache_writes_removed=0,overall_complete=False)
    write(O/'F24-history-priority-and-state-reuse-v30.json',history)
    # Publication/data graphs are source-consistent projections, not pointer/event-complete proofs.
    nodes=['raw_IDs','lookback_IDs','row_metadata','slot_map','dead_current','dead_lookback','mask_ready','request_index','cache_before','cache_write','cache_after','hash_all_layers','DP_hash_gather','next_generation_cache_reader']
    edges=[('raw_IDs','dead_current','CURRENT_IMAGE_SENTINELS'),('lookback_IDs','dead_lookback','LOOKBACK_IMAGE_SENTINELS'),('dead_current','mask_ready','EARLY_MASK_PUBLICATION'),('row_metadata','request_index','QUERY_ROWS'),('raw_IDs','cache_write','MAPPED_IDS'),('dead_current','cache_write','DEAD_SENTINEL'),('slot_map','cache_write','VALID_SLOT'),('cache_before','cache_write','STATE_VERSION'),('cache_write','cache_after','WRITE_BEFORE_HASH'),('cache_after','hash_all_layers','OPTIONAL_FALLBACK'),('request_index','hash_all_layers','REQUEST_MAP'),('raw_IDs','hash_all_layers','CURRENT_CHUNK'),('lookback_IDs','hash_all_layers','WINDOW'),('dead_lookback','hash_all_layers','WINDOW_SENTINELS'),('row_metadata','hash_all_layers','POSITIONS_TABLES'),('hash_all_layers','DP_hash_gather','ALL_LAYER_HASHES'),('cache_after','next_generation_cache_reader','PERSISTENT_STATE_OBLIGATION')]
    for layer in[1,14]:
        q=lambda x:f'L{layer}_{x}'
        nodes += [q(x)for x in['prezero','UVA_lookup','DP_row_exchange','TP_head_AllGather','global_bank','layer_ready','SP_rows','WKV','gate']]
        edges += [('DP_hash_gather',q('UVA_lookup'),'LAYER_HASH_VIEW'),(q('prezero'),q('UVA_lookup'),'INITIAL_CONTENT_OVERWRITTEN_CURRENT_UVA'),(q('UVA_lookup'),q('DP_row_exchange'),'LOCAL_HEADS'),(q('DP_row_exchange'),q('TP_head_AllGather'),'OWN_DP_ROWS'),(q('TP_head_AllGather'),q('global_bank'),'FULL_HEADS_GLOBAL_ROWS'),(q('global_bank'),q('layer_ready'),'COPY_TAIL_AND_PUBLICATION'),(q('layer_ready'),q('SP_rows'),'WAIT_THEN_PAD_SHARD'),(q('SP_rows'),q('WKV'),'LOCAL_ROWS'),(q('WKV'),q('gate'),'BF16_KEY_SHARED_VALUE'),('mask_ready',q('gate'),'TOKEN_MASK')]
    baseline=graph(nodes,edges)
    nz=graph([x for x in nodes if not x.endswith('_prezero')],[e for e in edges if not e[0].endswith('_prezero')])
    nn=deepcopy(nodes);ee=deepcopy(edges)
    for layer in[1,14]:
        q=lambda x:f'L{layer}_{x}'
        gone={q(x)for x in['TP_head_AllGather','global_bank','layer_ready','SP_rows']};nn=[x for x in nn if x not in gone];ee=[e for e in ee if e[0]not in gone and e[1]not in gone]
        nn += [q(x)for x in['row_head_AlltoAll','local_bank','local_ready']]
        ee += [(q('DP_row_exchange'),q('row_head_AlltoAll'),'DESTINATION_ROW_BLOCKS'),(q('row_head_AlltoAll'),q('local_bank'),'TP_HEAD_ORDER_PLUS_TRIM'),(q('local_bank'),q('local_ready'),'CAPTURE_ABI_TAIL_PUBLICATION'),(q('local_ready'),q('WKV'),'WAIT_LOCAL_ROWS')]
    aa=graph(nn,ee)
    write(O/'F24-source-and-conditional-publication-graphs-v30.json',dict(generated_utc=stamp,baseline=baseline,candidate_without_current_UVA_prezero=nz,candidate_row_head_exchange=aa,scope='Separateconditionalgraphprojections;notautomaticcompositionorcompletepointer/eventDAG.',overall_complete=False))
    f24_variants=[
        dict(id='F24-v30-EXISTING',state='ALREADY_IMPLEMENTED',change='UVAhostgather+dequant,request-indexhoist,all-layerhash/sharedDPgather.',requires='Do notcountthesecurrentmechanismsasnewgain.'),
        dict(id='F24-v30-PREZERO',state='LOCAL_FULLWRITE_QUESTION_CLOSED;CONDITIONAL_CANDIDATE',change='Omit176lookup-prezerooperationsinthecurrentfullwriteUVApath.',requires='Currentlaunch/layout,all-elements-store,noearlyobserverandvalidcapture/allocatorABI;retainotherzeros/backends.'),
        dict(id='F24-v30-ROW-HEAD-EXCHANGE',state='DERIVED_CONDITIONAL_REORGANIZATION',change='TPheadAllGather+SProwshard->destination-row/headAlltoAll+localpublication.',requires='DP/TProwgroupmapping,allconsumers,capturebankABI/per-layerreadyandidenticalpadding/headtrim.'),
        dict(id='F24-v30-DIRECT-PUBLISH',state='CONDITIONAL_FUSION_OR_LAYOUT',change='Lookup/exchangewritesconsumer-readybankoracceptsstride-awareIDs;avoidintermediatecontiguous/transpose/copywhenownershipallows.',requires='Sourcegeneration/strides,registeredhostlifetime,heads/rowslayout,unchangedallliveviews/tails/lastread;currentTP1DP1directshortcutalreadyexists.'),
        dict(id='F24-v30-HISTORY',state='CONDITIONAL_CROSS_ITERATION_STATE',change='Useexisting/changedsufficientlookbackpublicationtoavoidunneededslotcachewrites/readbacks.',requires='Wholelifetimehistorycoverage,accepted/rollback/remap/deadsemanticsandnoothercacheobserver;88currentwritesremainnecessaryunlessprovedotherwise.'),
        dict(id='F24-v30-PIPELINE',state='CONDITIONAL_SPLIT_SCHEDULING',change='Tile/pipelinehash,registeredhostlookup,DP/TPexchangeandper-layerready.',requires='Dependency/tilevisibility/hostregistration/bandwidthcontracts;preservegroupcollectiveparticipationandearlylayerpublication.'),
        dict(id='F24-v30-EXCLUDED',state='EXCLUDED_AS_BLANKET_REWRITE',change='Dedupcurrent/lookbackmasks,dropall-deadrankcollectives,skipdummy/tailpublicationorcachewritebywindowwidthalone.',reason='Differentinputs,stateobserversandgroup/protocoleffectsaredistinguishable.')]
    f25_variants=[
        dict(id='F25-v30-PARAM',state='CONDITIONAL_PARAMETER_GENERATION_CSE',change='CacheFP32q*kperlayerandFP32blockrotationperimmutable/versionedparameterstorage.',requires='Sameoperationrounding/device/dtypeandload/weight/adaptor/version;invalidateonreload/move/cast/update,keepcapturedaddresses;samematrixshapeisnotparameteridentity.'),
        dict(id='F25-v30-GATE',state='CONDITIONAL_FUSION',change='Joinblock32rotation,casts,twofull-featureRMS,weighteddot,signedsqrt/sigmoid/maskandresidual.',requires='Originalresidualbasis,BF16boundaries,FP32order,eps,signbit/signedzero/NaN/Infand4HCsharedvalueconsumers;orthogonalitydoesnotcancelnumericalnorm/rounding.'),
        dict(id='F25-v30-WKV-TILE',state='CONDITIONAL_SPLIT_OR_LARGER_FUSION',change='WKVoutputtilesfeedkeynorm/dotandsharedvaluegatewithoutunnecessaryglobalintermediates.',requires='All5120featurescontributetonorm/dot;defergateuntilreductionsready,retainBF16WKVroundingandvalueforall4HCstreams.'),
        dict(id='F25-v30-PROJECTION-SHARD',state='CONDITIONAL_ALTERNATIVE_PARTITION;NOT_BITWISE_PROVED',change='Reviewlocal-headpartialWKVwithrowReduceScatterasanothercommunication/weightpartition.',requires='NewsplitreductionchangesFPaccumulation;exactbaselineBF16roundingnotproved. Comparelarger25600output/FP32partialcommunicationvs6144BF16inputexchange,weightsandgroup/captureconstraints;notafreespeedup.'),
        dict(id='F25-v30-EXCLUDED',state='EXCLUDED_AS_UNCONDITIONAL_REWRITE',change='Treatrotationasdense5120square,deleteRMSusingidealorthogonality,reusegateactivationsacrosslayersoroverwritevalueafteroneHCstream.',reason='Profileblock32,actualrounding/fullfeaturedependenceandfoursharedconsumerscontradictthoseclaims.')]
    f24_conditions=[
        ('actual_population','MEASURED','3536nativeAPI/3632TASK/26chains;160graphsegments;176source-boundCOMM. Hierarchicalnotadditive.'),
        ('root_phase','MEASURED_PARTIAL','176TPcollectivesuniquelybind88roots;48nativeTASKand32segmentsunboundremainunassigned.'),
        ('existing_hash_sharing','ALREADY_IMPLEMENTED','Onehashalllayers/oneDPgatherandhoistedrequestindex;notnewfusion.'),
        ('dead_mask_domains','DOCUMENTED_EXCLUSION','Currentandlookbackinputdomainsdifferent;optionalvalidcountandselectedrowsretained.'),
        ('UVA_fullwrite','LOCAL_SOURCE_QUESTION_CLOSED','GridT*H,bijection(t,h),WIDTHstores,invalidIDswritezero,nooutputloads;currentPAD=LOCAL.'),
        ('prezero_variant','CONDITIONAL','OnlycurrentnonemptyfullwriteUVAoutput;otherbackend/padhead/dummy/banktailzerosnotcovered.'),
        ('current_dimensions','MEASURED','M*3*256lookup,TP8->24*256=6144features;do notguessheadfactorization.'),
        ('SP_rows','DOCUMENTED','Exactimportimplzero-padsdim0toTPmultipleandselectscontiguousrankchunk;divisiblepathclones.'),
        ('row_head_identity','DERIVED_IN_STATED_DOMAIN','Gatherheads->pad/shardrowsequalsdestinationrowexchange->sameheadconcat/trim;purepermutation/zero-fill.'),
        ('all_consumers','UNKNOWN_RUNTIME_AND_PARTIAL_SOURCE','ObservedforwardusesSP-localrows;fullbankotherreaders/nonSPentrymustbeclosedbeforeblanketreplacement.'),
        ('DP_TP_participation','RETAINED_CONDITIONAL','DPexchangefirst,matchingTPgroup/rank/metadata,empty/all-deadparticipantrequirementsremain.'),
        ('publication','DOCUMENTED_CONDITIONAL','Maskpublishedearly;lookupreadyperlayerafterexchange/tails. Jointlayersmustnotdelaylayer1withoutpricing.'),
        ('capture_alias_strides','UNKNOWN_RUNTIME_GENERATION','Fixedglobalbank->localbankABI/capturedaddresses,readonlysnapshots/strides/lastreadstillneedconditions.'),
        ('history_priority','DOCUMENTED_LOCAL_QUESTION_CLOSED','Currentchunkthenknownwindowthenoptionalcache;DEAD/negativepositionsblock;writesprecedehash.'),
        ('history_elimination','CONDITIONAL_WHOLE_LIFETIME','Widthaloneinsufficient;allrequiredhistoryknown/correctandnoothercurrent/futurecacheobserver.'),
        ('backend_matrix','OPEN_CONDITIONS','CPU/GPUweights,slotless,TP1/DP1,DPshared,force_dummy,empty,image/indices/validcountnotallruntimeattested.'),
        ('host_memory_pipeline','CONDITIONAL_UNMEASURED','RegisteredUVApage/hostcache/PCIe/copy/overlapandCPUproducerreadinessnotzero-filled.'),
        ('physical_gain','UNKNOWN_NOT_LOGICAL_RATIO','Logicalpayloadratio8intheequalpartitionexampleisnotphysicalbandwidthorrequestspeedup.')]
    f25_conditions=[
        ('actual_body','MEASURED_SOURCE_CONSISTENT','160graph+48nativebodies;208WKV,624parameterprep,7072ordinarygateTASKs.'),
        ('rotation','MEASURED_DOCUMENTED','32x32blockalreadyused;notdense5120square.'),
        ('parameter_values','DOCUMENTED_CONDITIONAL','q*k81920logicalbytesperlayerandrotation4096logicalbytes;notmeasuredHBMtraffic/gain.'),
        ('parameter_generation','UNKNOWN_RUNTIME_LIFETIME','Cachekeymustincludeactualstorage/version/load/device/dtype,notshape;per-layerweightsdistinct.'),
        ('precision_boundaries','RETAINED','FP32cast/productoperationcontractandBF16WKV/residualoutputrounding.'),
        ('full_feature_reductions','RETAINED','Separatehidden/key5120DRMSplusweighteddot;epsilonandorderrequired.'),
        ('rotation_residual_basis','RETAINED','Unrotateforgateoriginal-basiscalculationbutaddtooriginalhidden.floatresidual.'),
        ('nonfinite_mask','RETAINED','Signedsqrt/signbit/sigmoid,masked_filland0*Inf/NaN/signedzerosmustretainsemantics.'),
        ('value_fanout','RETAINED','5120Dvaluefeedsall4HCstreams;notdeadwhenfirstconsumerfinishes.'),
        ('tile_visibility','CONDITIONAL_NEW_INTERFACE','WKV/key/value/reductionsrequireexactpartialready/lastreadercontract;wholekernelcompletiondoesnotprovetilevisibility.'),
        ('projection_partition','CONDITIONAL_NOT_BITWISE_PROVED','PartialWKV+ReduceScatterchangesarithmetic/communication;mustretainexactnessorthedeclaredtoleranceandallconsumers.'),
        ('relations','NONADDITIVE','F24rowheadexchange,F25projectionpartition/WKVgateandF03/F26residualboundariesoverlap;choosecompatiblegraphnotaddallremovals.'),
        ('phase_and_gain','UNKNOWN','16graphphasesunbound;noCPUscope/HBMpeak/requestgainornativenumericalcertificatefromcounts.')]
    for cid,variants,conditions in[('F24',f24_variants,f24_conditions),('F25',f25_variants,f25_conditions)]:
        write(O/(cid+'-variants-and-conditions-v30.json'),dict(generated_utc=stamp,candidate=cid,variants=variants,conditions=[dict(id=f'{cid}-v30-{j+1:02d}',branch=b,state=st,evidence=e)for j,(b,st,e)in enumerate(conditions)],all_conditions_closed=False,implementation=False,new_performance_result=False,overall_complete=False))
    write(O/'Engram-candidate-deduplication-v30.json',dict(generated_utc=stamp,existing_parents=['F24','F25'],new_main_candidate_created=False,model_candidate_count=35,reason='Prezero/communication/history/publicationandparameter/gate/projectionvariantsrefineexistingEngramcandidates;noinflatedF36.',relations=[dict(a='F24-rowhead',b='F25-projectionshard',relation='alternative_partition_not_additive'),dict(a='F24-directpublish',b='F24-rowhead',relation='overlap_bank_copy_and_rowmaterialization'),dict(a='F25-gate',b='F25-WKVtile',relation='composable_only_with_shared_precision_and_reduction_contract'),dict(a='F24-history',b='H07',relation='share_rawID_generation_obligations_but_different_state_and_sentinel_mapping')],overall_complete=False))
    model=deepcopy(read(O/'candidate-profile-ledger-v29.json'));fw=deepcopy(read(O/'framework-candidate-ledger-v29.json'));mods=deepcopy(read(O/'module-profile-ledger-v29.json'));matrix=deepcopy(read(O/'module-phase-branch-matrix-v29.json'))
    for cid,variants,conditions in[('F24',f24_variants,f24_conditions),('F25',f25_variants,f25_conditions)]:
        c=next(c for c in model['candidates']if c['id']==cid);c.update(profile_review_state='ALLRANK_ENGRAM_NATIVE_GRAPH_COMM_AND_TARGETED_STATE_PUBLICATION_CONTRACTS_REVIEWED;CONDITIONAL_UNPRICED',variants_v30=variants,condition_ledger=cid+'-variants-and-conditions-v30.json',target_source_gate=str(O/'PROFILE-CODE-QUESTION-GATE-v30.json'),legal_fusion_proved=False,candidate_closed=False,critical_path_gain_us=None)
        c.setdefault('scoped_detail',{})['current166_v30']=dict(native_TASKs=3632,graph_Engram_TASKs=6080,TP_collectives=176,root_bound_TP_collectives=176,body_instances=208,role='Ancestrygraph/typedpopulationnotautomaticPMUbucket',native_unbound_TASKs=48,phase_unbound_graph_segments=32)
        c['conclusion_changing_gap']='Selectedlocalfullwrite/historypriority/SProwmap/parameterdimensionsanswered;all-consumer/capture/state-generation/nativeprecisionandphysicalgainstillconditional.'
        c['next_profile_or_contract_action']='Do notrepeatalreadyansweredlocalcontractsorchaseallcompilerinternals;advanceunreviewedwholemodulefrontier. Specificruntimeevidenceonlyifneededforastrongerclaim.'
        c.setdefault('evidence_files',[]).extend(['Engram-current-profile-and-root-roles-v30.json','Engram-hash-lookup-publication-and-gate-contract-v30.json',cid+'-variants-and-conditions-v30.json'])
        q=R/'candidates'/(cid+'.json');master=read(q);write(O/(cid+'-master-before-v30.json'),master);master['review_v30']=dict(model_ledger=str(O/'candidate-profile-ledger-v30.json')+'#'+cid,contract=str(O/'Engram-hash-lookup-publication-and-gate-contract-v30.json'),variants=variants,conditions=len(conditions),implementation=False,new_performance_result=False);write(q,master)
    mm={m['id']:m for m in mods['modules']};bm={m['id']:m for m in matrix['modules']}
    notes={'M00':'PositivecurrentUVAhash/cacheandordinaryrotatedgatepathsbound;otherbackend/slotlessbranchesnotallattested.', 'M04':'Current/lookbackdeadmaskdomains,optionalvalidcount/indicesandrawIDs/slot-mapgenerationsretained.', 'M06':'Engrammaskandper-layerreadythenSProws;residualbasisand4HCvaluefanoutkept.', 'M14':'WholecaptureEngramscope:3536API/3632nativeTASK+6080graphcompute,176TP8COMM;fullwrite/historypriority/SPmapreviewed,conditionalF24/F25refined.', 'M21':'Hashreq-index/currentchunk/window/cachefallbackneedsrequest/row/slot/generationidentity;windowwidthaloneinsufficient.', 'M22':'Localbank/directpublication/parameterCSEneedsfixedcaptureABIandready/lastreader/weightversions;16graphphasesunbound.', 'M23':'All176TP8gathersbind88roots;row/headAlltoAllidentityderived,preserveDPfirst/group/rankorder/zero-paddingandallconsumers.', 'M24':'Cachewritescurrent/futurereadersandimmutableparametergenerationdistinguished;address/shapeorone-stepallhitnotlifetimeproof.'}
    for mid,text in notes.items():
        for cid in(['F24','F25']if mid in['M14','M22','M23','M24']else['F24']):
            if cid not in mm[mid]['candidate_links']:mm[mid]['candidate_links'].append(cid)
        mm[mid]['review_v30']=dict(finding=text,evidence=['Engram-current-profile-and-root-roles-v30.json','Engram-hash-lookup-publication-and-gate-contract-v30.json','F24-variants-and-conditions-v30.json','F25-variants-and-conditions-v30.json'],whole_module_closed=False)
        bm[mid]['branches'].append(dict(branch='v30Engramnative/graph/COMMhash-history-fullwrite-rowhead-publication-andparametercontracts',state='SCOPED_MEASURED_DOCUMENTED_DERIVED;WHOLE_MODULE_OPEN',evidence=text,gap='Allentry/consumer/alias/generation/nativeprecision/physicalgainclosurestillopen.'))
    for m in mods['modules']:m['phase_branch_record']='module-phase-branch-matrix-v30.json#'+m['id'];bm[m['id']]['candidate_links']=list(m['candidate_links'])
    for obj,parent,dest in[(model,'candidate-profile-ledger-v29.json','candidate-profile-ledger-v30.json'),(fw,'framework-candidate-ledger-v29.json','framework-candidate-ledger-v30.json'),(mods,'module-profile-ledger-v29.json','module-profile-ledger-v30.json')]:
        obj.update(generated_utc=stamp,parent_ledger=parent,whole_modules_closed=0,overall_complete=False,new_source_read=True,new_target_source_read=True,repository_wide_audit=False,new_device_work=False)
        if obj is model:obj['framework_candidate_ledger']='framework-candidate-ledger-v30.json'
        if obj is mods:obj.update(existing_model_candidates=35,framework_candidates=7)
        write(O/dest,obj)
    matrix.update(generated_utc=stamp,parent_matrix='module-phase-branch-matrix-v29.json');write(O/'module-phase-branch-matrix-v30.json',matrix)
    registry=[]
    for c in model['candidates']:
        q=R/'candidates'/(c['id']+'.json');assert q.exists();registry.append(dict(id=c['id'],master_record=str(q),master_sha256=hashlib.sha256(q.read_bytes()).hexdigest(),current_authority=str(O/'candidate-profile-ledger-v30.json')+'#'+c['id'],modules=[m['id']for m in mods['modules']if c['id']in m['candidate_links']]))
    assert len(registry)==35 and len(fw['candidates'])==7 and all(c['modules']for c in registry)
    write(O/'candidate-record-index-v30.json',dict(generated_utc=stamp,model_candidates=registry,framework_authority=str(O/'framework-candidate-ledger-v30.json'),unlinked_candidate_ids=[],whole_modules_closed=0,overall_complete=False))
    remaining=[x for x in deepcopy(read(O/'module-coverage-gaps-v29.json')['remaining'])if not x.startswith('NextwholecoveragefrontierM14/')]
    remaining.insert(3,'M14v30localcontractsanswered:176UVAprezerosfull-overwriteinselectedPAD=LOCALpath;TProw/headexchangeidentityandSPpad/chunkknown;cacheprioritycurrent/window/cacheandwritefootprintknown. Do notrepeatthemunknown. Allconsumer/fullbank/captureABI/historyrequest-slot/weightgenerationandnativeprecision/causalgainremainconditional. 48nativeTASKand32segmentsunboundretained;do notreplacev20functionalbucketbyancestry9712.')
    remaining.insert(4,'NextwholecoveragefrontierM15targetfinalHCcollapse/norm/MTPauxandtheirconsumers,orothergenuineunreviewedfinitebranch. Startsavednativefullcaller/graphsuffixandphase-ledger;reuseF26v11andF35v28consumerproofs. Pointreadonlyspecificmissingselectedproducer/consumer;nocurrentNPU/service/whole-repositoryaudit.')
    write(O/'module-coverage-gaps-v30.json',dict(generated_utc=stamp,remaining=remaining,whole_modules_closed=0,overall_complete=False))
    summary=dict(generated_utc=stamp,status='PROGRESS',previous_turn_classification='PROGRESS:v29F04precision/router408branch/F20IDcommutationdedup',native_APIs=3536,native_TASKs=3632,native_chains=26,graph_segments=160,graph_compute=6080,TP_collectives=176,TP_collectives_uniquely_root_bound=176,hash_roots=88,graph_unbound_segments=32,native_unbound_TASKs=48,body_instances=208,qk_parameter_preparation_TASKs=624,lookup_prezero_fullwrite_source_question_closed=True,row_head_exchange_identity_derived=True,actual_history_hits='UNKNOWN',F24_conditions=len(f24_conditions),F25_conditions=len(f25_conditions),new_model_candidates=[],refined_model_candidates=['F24','F25'],model_candidates=35,framework_candidates=7,whole_modules_closed=0,overall_complete=False,evidence_refs=refs,remote_jobs=6,remote_pass=4,remote_retained_errors=2,no_new_model_NPU_service_or_health_work=True,remaining=remaining)
    write(O/'Engram-whole-frontier-summary-v30.json',summary)
    gate=read(O/'PROFILE-CODE-QUESTION-GATE-v30.json');gate.update(status='PROFILE_AND_NECESSARY_POINT_CONTRACTS_REVIEWED',answered_by=['Engram-current-profile-and-root-roles-v30.json','Engram-hash-lookup-publication-and-gate-contract-v30.json','F24-UVA-fullwrite-and-prezero-review-v30.json','F24-row-head-exchange-algebra-v30.json','F24-history-priority-and-state-reuse-v30.json','F24-SP-row-shard-contract-v30.json'],closed_scope='SelectedUVAfullwrite/current-lookbackdistinction/hashpriority-cachefootprint/SPpad-chunk/currenttypedparameterandrotationdimensions.',not_closed='WholeM14/allmodules,actualstate/aliases/generations/binary/newprecision/physicalgain.',stop_rule='Localquestionsanswered;advancewholecoveragefrontier,notrepeatSDK/compiler/sourceauditforthealreadyconditionalcandidates.');write(O/'PROFILE-CODE-QUESTION-GATE-v30.json',gate)
    report='''按用户约定先充分使用profile，本轮从M14 Engram的全捕获范围补覆盖。全部32份保存分片中，26条原生完整调用链关联3536个API、3632条TASK；80张图的两个Engram外段共160段，每段38条compute，共6080。这里是Engram调用祖先/图段范围，合计9712不能自动替换v20的9680条功能资源归属，更不能用作延迟之和。完整COMM表另有176次源码绑定的TP8 BF16 AllGather，ComputeInfo只出现128次hcom是统计层级差异，不是少48个collective。

176次TP通信全部经原生来源、同线程execute marker和prepare root包含关系唯一绑定88个上下文，每个两层各一次：64个短输入上下文对应M24与128个已绑定图段，24个长输入上下文对应M6400/6528/6672及eager投影行800/816/834。还保留32个图段（16张图）阶段未绑定，以及48条没有唯一prepare root的原生TASK；未按次数或最近时间强配。8个root没有关联到所选hash/lookup，不等于该root没有任何活动。

先由profile锁定176个embedding343预清零及176次UVA lookup，再定点读实际导入kernel。当前lookup输出实际是[M,3,256]，IDs[M,24]，TP8重建24×256=6144特征；不能根据总元素数猜成6×128。wrapper发射tokens×local_heads个program，kernel按(token,local_head)逐行写全部WIDTH，越界ID也显式写零，完全不读旧output；当前PAD_HEADS=LOCAL_HEADS，输出连续。因此当前非空UVA路径的预清零在指定源码合同内属于完整覆写前的初始化，已登记F24条件变体。还需保持分配/capture地址、无提前观察和正常完成；没有实现删除，也不删除dummy、mask、固定bank尾部零或推广其他lookup后端。

第二项F24重组来自完整跨卡消费域：当前完整头AllGather后，model1581–1584才按SP切行，再给WKV。profile不能说明分片行序，因此仅解析已验SHA model.py的sp_shard导入，读取确切wrapper及12行实现：行轴先补零到TP倍数，再按rank取连续块，整除路径clone。由逐元素索引可证，在相同行/头rank顺序、padding和trim下，“拼完整heads再按目的rank取行”与“各源rank先发送目的行块、目的rank再拼heads”逐字节相同。登记按目标行/头AlltoAll及本地bank发布的条件变体。M整除P时每rank远端接收逻辑元素量的比值为P=8，但这不是物理链路流量或8倍提速；当前新bank ABI、其他完整bank消费者、图更新/别名/ready代际和所有参与rank仍需保持。DP行交换仍先完成，mask及每层ready不能被一个全层barrier延后第一层消费。

哈希链也由profile的88组req-index/cache-write/hash锁定，再补指定kernel。优先级确认为当前chunk、已知非负lookback token、可选slot cache回退；负位置和DEAD_ID形成blocked ngram。cache写入按有效slot写映射ID或DEAD_ID，并先于hash；代码不能恢复历史hit数和slot内容。只有整个声明生命周期的所需历史均由正确request-generation的chunk/window覆盖且没有其他当前/未来cache观察者，才允许用slotless路径省写。窗口宽度够大但元素为-1是反例，本轮全命中却下一轮回退也是反例。因此没有删除88次cache写。两次dead-mask分别接收当前IDs和lookback IDs，不是同输入重复，不能直接去重。当前一次hash覆盖全部层、一次DP gather供两层、UVA gather+dequant已组合，均不报成新优化。

F25从少量选段扩展到全部160图内及48原生body：共208个WKV、624个q/k参数cast和乘积TASK、7072个ordinary gate TASK。typed-shape核对证明每个body是1+3+34；参数q*k为4×5120 FP32，rotation实际是32×32 FP32分块旋转，不能虚构5120平方dense rotation。参数代际缓存归入既有F25，必须按真正storage/version/load/device/dtype失效并保留相同FP32算术/capture地址；shape相同不代表权重相同。gate复用v6精度合同：两路5120维RMS、加权dot、signed sqrt/sigmoid/mask、原始hidden.float残差以及4路HC共享value均保留。WKV tile到gate、分块旋转融合和本地head部分投影再ReduceScatter保留为条件替代；后者改变累加与通信量，不声称bitwise等价或免费收益。

F24补18项条件，F25补13项条件。所有机会均细化既有F24/F25，模型主候选仍35项、框架7项，不新增F36。第一次kernel读取只查top-level import，未解析wrapper内lazy import，在保护处停止；第一次role audit把导出FP32标签误写DT_FLOAT，也在保护处停止。两份失败保留，修正版分别限定lazy import和明确FLOAT/DT_FLOAT标签，未扩大仓库搜索。六次远端CPU作业4 pass/2保留error，本地只读确切SP helper并做索引证明，没有导入torch/模型或执行NPU/服务/新健康检查。最后健康证据仍v12post。

完整覆盖目标仍未完成，26个整模块全部保持open，26项理论比较仍INCOMPLETE。局部full-write/优先级/SP行序问题已回答，不再泛称“源码未知”；真实状态/消费者/代际/native精度/因果收益缺口独立保留。下一批转M15最终HC collapse、norm及MTP辅助状态的真实消费者，从保存fullcaller和图尾开始，复用F26/v11与F35/v28已有证据，再对新出现的确切缺口补代码。
'''
    (O/'PROFILE-FINDINGS-v30.md').write_text(report)
    pub=R/'publication-draft/FUSION-INVENTORY.md';(O/'FUSION-INVENTORY-before-v30.md').write_bytes(pub.read_bytes());lines=pub.read_text().splitlines()
    lines=[x.replace('当前持续分析台账（v29）','当前持续分析台账（v30）').replace(str(O/'candidate-profile-ledger-v29.json'),str(O/'candidate-profile-ledger-v30.json')).replace(str(O/'PROFILE-FINDINGS-v29.md'),str(O/'PROFILE-FINDINGS-v30.md'))for x in lines]
    lines=[('| F24 | Engram hash/lookup/ready；预清零、行头交换与状态重组 | 条件 | v30全范围3632原生TASK+160×38图段、176 TP8通信；当前UVA输出全覆写支持预清零条件省去，TP头gather→SP行切可改目的行/头交换但需全消费者/新bank/ready合同；cache写只能在全生命周期历史覆盖且无其他观察者时省去。 |')if x.startswith('| F24 |')else('| F25 | Engram WKV、分块旋转、gate/残差与参数代际复用 | 条件 | v30补齐208 body/624参数准备TASK；实际32×32分块旋转，保留两路5120维RMS、FP32算术、原始hidden残差和4HC共享value。参数缓存/WKV tile/投影分区为重叠条件替代，无实现或提速结论。 |')if x.startswith('| F25 |')else x for x in lines]
    lines += ['','v30持续分析台账：35项模型候选、7项框架候选，整模块仍open。当前记录 '+str(O/'Engram-whole-frontier-summary-v30.json')+'；旧枚举/报告成稿不代表全范围闭合。'];pub.write_text('\n'.join(lines)+'\n')
    action=('继续Loop004，v30实质进展；先读Engram-whole-frontier-summary-v30.json、Engram-hash-lookup-publication-and-gate-contract-v30.json、F24-UVA-fullwrite-and-prezero-review-v30.json、F24-row-head-exchange-algebra-v30.json、module-coverage-gaps-v30.json。'
        'M14全部32part/26nativechains=3536API3632TASK；80图160Engram段*38=6080，ancestry总9712不是v20M14功能归属9680，别覆盖PMU账。COMM176不是ComputeInfohcom128同层统计。'
        '176TP8 BF16gather全唯一bind88roots每root2：64短M24与128bound段；24长M6400/6528/6672->nativeWKV800/816/834。48nativeTASK无uniqueroot、32段来自16phase-unbound图仍不强配；8root未关联本选择≠没活动。'
        '当前lookup输出[M,3,256]、IDs[M,24]，TP8重建24*256=6144，别猜6x128。UVAgridT*H逐row全WIDTHstore，invalidID写零，无oldoutputload，PAD=LOCAL；176prezeros在该source域full-overwrite，F24条件省去；dummy/mask/tail/其他backend不可一并删。'
        'F24sp_shard导入确切文件sequence_parallel.py SHA73bcf07050bce8b049dd0bbdb0a7504f6eba157a1e97a5033849ef31ef05fcb0，impl32–43零pad到TP倍数后按rank连续块，整除clone。TPheadgather+SP切行与目的行/headAlltoAll逐索引同值；P8逻辑remote payload比8≠物理流量或提速。新SP-localbank/allconsumer/capture/rowmap/rankorder/ready代际仍条件，DP先行和每层ready别消除。'
        '88hash/cachewrite/reqindex已实现；currentchunk->knownwindow>=0->cachefallback。未知-1需要fallback，cachewritevalidslot映射ID/DEAD先于hash。全生命周期history完整且无当前/未来observer才可用slotless省写；宽度/本步allhit不是证明。两deadmask是current vslookback别CSE；一次hashalllayer/一次DPgather已实现别新报。'
        'F25共208body=160graph+48native，WKV208、qkparam624、ordinary7072；qk4*5120FP32=81920logicalB/层，rotation32²FP32=4096logicalB已block非dense5120²。参数storage/version/load/device/dtype失效；gate两5120RMS/FP32dot/signedsqrt/sigmoid/mask/原hidden.float残差/4HCvalue保留。WKVtile与部分head投影ReduceScatter是条件替代，不加算benefit/bitwise。'
        '远端M结果SHA profilef4daeb471f3bda1cec7e89a89b1ffe3791150f5e06b976fbcef4388f13d01f6f；source6a16b2c82916fa92d2662d41751c167c2c20993cdb641fd012f630dc736128ca；kernelv25b5d4058e1ea3c9ac1b0a8a7318dac841f3a4b1dcd8240e27845e631c934e7f6；rootauditv2a203880eccb27d353e12a74a7cd1f1f0ba55460c8d12372c7cba4ea7c851250c。'
        '6remoteCPUjobs4pass2error保留(lazyimport/FP32labelguards),本地主审SP精确helper+纯索引证明，无newmodel/NPU/service/health,lastv12post；Goalactive。35model7framework0wholemodule；F24条件18/F25条件13，noF36；3136candidate资源/849missingPMU/16phaseunbound/26theoryINCOMPLETE不变。'
        '下一批M15最终HCcollapse/norm/MTPaux consumer：先savednativefullcaller/graphsuffix和phaseledger，复用F26v11/F35v28，只有新候选关键缺口再定点source；别在已答M14fullwrite/historypriority/SPmap上循环或扫SDK/compiler。有限新live诊断仍freshhealth/ownership/frozenworkload/clone/restore，不用旧PID猜idle。')
    jobs=[('profile','pass','AllsavedEngramnativefullcallers/graphoutersegments/COMMscopepreservedwithnativeexecutionmultiplicityandhierarchy.',[N/'query_Engram_whole_profile_v30_166.py',N/'Engram-profile166-v30.json']),
        ('review','pass','Onlynamedhash/deadmask/lookup/publication/layermethodsread;sourceSHApinned;notrepositoryaudit.',[N/'read_Engram_cache_and_publication_contract_v30_166.py',N/'Engram-source166-v30.json']),
        ('review','error','Originaltop-level-importguardstoppedonfunction-localUVAimport;noresultemitted;retainedwithoutreclassifyingaspass.',[N/'read_Engram_kernel_write_and_history_contract_v30_166.py',N/'Engram-kernel166-v30.json']),
        ('review','pass','Correctedexactlazy-importresolutionandnamedkernelonly;fulloutputstore/historypriority/cachewritefootprintreviewed.',[N/'read_Engram_kernel_write_and_history_contract_v30_166_v2.py',N/'Engram-kernel166-v30-v2.json']),
        ('profile','error','TypedroleauditstoppedbecauseFP32rawtagisFLOATnotDT_FLOAT;guarderrorretained.',[N/'audit_Engram_roots_and_roles_v30_166.py',N/'Engram-root-audit166-v30.json']),
        ('profile','pass','ExplicitFLOAT/DT_FLOATcompatibilitypreservesrawtags;176exactrootlinks,typedparameter/gaterolesandunboundsretained.',[N/'audit_Engram_roots_and_roles_v30_166_v2.py',N/'Engram-root-audit166-v30-v2.json']),
        ('review','pass','Mainreadsimport-resolvedSPhelper,provesfullwrite/rowheadidentityandretainsstate/precisionconditions;F24/F25dedupand31conditionrows,noimplementation/gain.',[O/'Engram-hash-lookup-publication-and-gate-contract-v30.json',O/'F24-UVA-fullwrite-and-prezero-review-v30.json',O/'F24-row-head-exchange-algebra-v30.json',O/'F24-SP-row-shard-contract-v30.json',O/'F24-history-priority-and-state-reuse-v30.json',O/'F24-variants-and-conditions-v30.json',O/'F25-variants-and-conditions-v30.json',O/'PROFILE-FINDINGS-v30.md'])]
    runs=[]
    for j,(kind,status,desc,arts)in enumerate(jobs):
        rid='run-'+(now-timedelta(seconds=len(jobs)-j)).strftime('%Y%m%dT%H%M%SZ')
        run=ctl('new-run','--loop-dir',str(L),'--run-id',rid,'--case-key','c8-input32k-output1k-prefix90-166','--kind',kind,'--command','Preparedv30retrospectiveregistration: '+desc,'--code-ref','Savedprofile/exactEngramsource;targetunchanged;noNPU/model/servicework','--working-directory',str(N))
        args=['finish-run','--run-dir',run,'--status',status,'--correctness','not-applicable','--summary',desc]
        for q in arts:assert q.exists(),q;args+=['--artifact',str(q)]
        ctl(*args);runs.append(dict(run_dir=run,status=status,description=desc))
    cp=dict(status='PROGRESS',previous_turn_classification=summary['previous_turn_classification'],current_verified_changes=['WholecaptureEngramnative/graph/COMMscopeand176exactTP-rootassociationsauditedwithunboundsretained.','CurrentUVAfullwritecloseslocalprezeroquestion;conditionalrow/headexchangedataperutationderivedfromexactSPmap.','Hashhistory/write/publicationand624parameterpreparationrolesrefineexistingF24/F25with31conditions;noimplementation/gain.'],summary=str(O/'Engram-whole-frontier-summary-v30.json'),candidate_ledger=str(O/'candidate-profile-ledger-v30.json'),framework_candidate_ledger=str(O/'framework-candidate-ledger-v30.json'),module_ledger=str(O/'module-profile-ledger-v30.json'),branch_matrix=str(O/'module-phase-branch-matrix-v30.json'),report=str(O/'PROFILE-FINDINGS-v30.md'),source_gate=str(O/'PROFILE-CODE-QUESTION-GATE-v30.json'),coverage_gaps=str(O/'module-coverage-gaps-v30.json'),pending_full_scope_domains=remaining,remote_job_status='6CPU-onlyremotejobs terminal:4pass,2guarderrorsretained;localexactSPhelper/mainreviewpass;noNPU/model/service/healthwork.',model_candidates=35,framework_candidates=7,next_action=action,whole_modules_closed=0,overall_complete=False,updated_utc=stamp)
    write(O/'REVIEW-CHECKPOINT-v30.json',cp)
    for q in[L/'state.json',T/'index.json']:
        d=read(q);assert d['status']in['EVALUATING','ACTIVE'];d.update(next_action=action,updated_at=stamp);write(q,d)
    for q in[L/'status.md',T/'status.md']:
        q.write_text('\n'.join(('- 下一步：'+action)if x.startswith('- 下一步：')else('- 更新时间：`'+stamp+'`')if x.startswith('- 更新时间：')else x for x in q.read_text().splitlines())+'\n')
    (O/'profile-summary-before-v30.json').write_bytes((R/'profile-summary.json').read_bytes());prof=read(R/'profile-summary.json');prof['supplementary_Engram_v30']=dict(summary=cp['summary'],model_candidates=35,framework_candidates=7,whole_modules_closed=0,overall_complete=False);write(R/'profile-summary.json',prof)
    for q in[N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
        d=read(q);d.update(latest_profile_first_review=cp,overall_complete=False,whole_modules_closed=0,updated_utc=stamp);d['next_read_only_action'if q==R/'STATE.json'else'next_action']=action
        if q==W/'STATE.json':d.update(profile_candidate_ledger=cp['candidate_ledger'],profile_module_ledger=cp['module_ledger'])
        if q in[N/'RECOVERY.json',R/'PARALLEL-RESUME-PACK.json']:d['read_first']=[cp['summary'],str(O/'Engram-hash-lookup-publication-and-gate-contract-v30.json'),str(O/'F24-row-head-exchange-algebra-v30.json'),cp['coverage_gaps']]
        write(q,d)
    compact=ctl('compact','--task-dir',str(T));validation=ctl('validate','--task-dir',str(T));write(O/'task-run-registration-v30.json',dict(runs=runs,compact_output=compact,validation_output=validation,whole_modules_closed=0,overall_complete=False))
    arts=[q for q in O.glob('*v30*')if q.is_file()and not q.name.startswith('checkpoint-manifest')]
    arts += [N/(label+'.json')for label in['Engram-profile166-v30','Engram-source166-v30','Engram-kernel166-v30','Engram-kernel166-v30-v2','Engram-root-audit166-v30','Engram-root-audit166-v30-v2']]
    arts += [N/x for x in['query_Engram_whole_profile_v30_166.py','read_Engram_cache_and_publication_contract_v30_166.py','read_Engram_kernel_write_and_history_contract_v30_166.py','read_Engram_kernel_write_and_history_contract_v30_166_v2.py','audit_Engram_roots_and_roles_v30_166.py','audit_Engram_roots_and_roles_v30_166_v2.py','finalize_Engram_whole_frontier_v30.py']]
    arts += [q for q in N.glob('zcode-Engram-*v30*20261031*')if q.is_file()]
    arts += [R/'candidates/F24.json',R/'candidates/F25.json',pub,N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json',R/'profile-summary.json',L/'state.json',T/'index.json']
    arts=sorted(set(arts));write(O/'checkpoint-manifest-v30.json',dict(generated_utc=stamp,artifacts=[dict(path=str(q),bytes=q.stat().st_size,sha256=hashlib.sha256(q.read_bytes()).hexdigest())for q in arts],whole_modules_closed=0,overall_complete=False))
    print(json.dumps(dict(checkpoint=str(O/'REVIEW-CHECKPOINT-v30.json'),manifest_files=len(arts),runs=len(runs),pass_runs=5,retained_error_runs=2,model_candidates=35,framework_candidates=7,validation=validation,whole_modules_closed=0,overall_complete=False),ensure_ascii=False))

if __name__=='__main__':main()
