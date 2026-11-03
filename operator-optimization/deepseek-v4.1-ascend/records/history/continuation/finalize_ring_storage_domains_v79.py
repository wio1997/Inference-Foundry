"""Persist physical state accounting and operator-boundary scope reconciliation."""
from copy import deepcopy
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v5 import persist_refinement


def main():
    assert not(O/'REVIEW-CHECKPOINT-v79.json').exists()
    names=['ring-storage-source-contract-v79.json','ring-storage-capacity-models-v79.json','ring-state-layout-and-lifetime-proof-v79.json',
        'M22-adjacent-operator-boundary-crosswalk-v79.json','candidate-relation-clauses-v79.json','candidate-relation-coverage-v79.json',
        'operator-analysis-coverage-ledger-v79.json','ring-storage-branch-inventory-v79.json','next-operator-proof-questions-v79.json']
    for name in names[:4]:assert(O/name).exists()
    source=read(O/names[0]);cost=read(O/names[1]);assert cost['pool_divisor_bytes']==540928
    rel=deepcopy(read(O/'candidate-relation-clauses-v78.json'))
    assert not any({r['a'],r['b']}=={'F31','F34'}for r in rel['clauses'])
    rel['clauses'].append(dict(id='REL79-01',a='F31',b='F34',relation='STATE_OWNER_HALF_AND_GENERATION_ABI',
        operator_claim='Private/paired C2ring representation requires explicit request-owner slot/half/generation metadata and new native/capture ABI. Current globalblockp-onlymetadata cannot be reinterpreted by pointer equality;all oldreads/queuedwrites must retire before slot/wholeID reuse.',
        evidence=names[:4],applicable_domain='NormalV41sharedglobalIDpool and statedC2even-observation/private32domains;actualresidentS/capacityU/physicalaliases/nativebindings unknown.',
        conditional_compatibility=True,all_runtime_gates_proved=False,speedup_additive=False,critical_path_gain_us=None))
    rel.update(clause_count=46,selected_pair_count=46)
    h=next(g for g in rel['higher_order_groups']if g['id']=='HG73-C2')
    assert 'F31'not in h['members'];h['members'].append('F31')
    h['state_representation_refinement_v79']=dict(contract='Pick unchangedtuple/private32/private16/paired16state representation andmetadata ownerprotocol beforeF18poolnorm/F19cachefork/F34maintenancesavings. Compactphysicalstorage/read-writequotient and projection/quantization work are not independent gains.',evidence=names[:4],all_combinations_analyzed=False)
    write(O/names[4],rel)
    cg=deepcopy(read(O/'candidate-relation-coverage-v78.json'));p=next(p for p in cg['pairs']if(p['a'],p['b'])==('F31','F34'))
    assert p['clauses']==[];p.update(scope_state='SCOPED_OPERATOR_RELATION_REVIEWED_CONDITIONAL',clauses=['REL79-01'])
    cg.update(parent_coverage='candidate-relation-coverage-v78.json',scoped_conditional_pairs=46,remaining_pair_relations=549,
        parent_relation_index='candidate-relation-clauses-v79.json',ring_storage_relation_v79='REL79-01/HG73-C2',overall_complete=False);write(O/names[5],cg)
    cov=deepcopy(read(O/'operator-analysis-coverage-ledger-v78.json'));cov.update(parent_coverage_ledger='operator-analysis-coverage-ledger-v78.json',
        new_pending_regimes_reconciled=0,new_pending_executions_reconciled=0,
        ring_storage_source_refinement_v79=dict(direct_pool_records=72,direct_update_records=72,direct_state_N=17306,actual_graph_state_N_attested=False,source_ring_objects=3,
            no_native_assignments_or_regime_promotions=True),overall_complete=False);write(O/names[6],cov)
    old={m['id']:m for m in read(O/'module-phase-branch-matrix-v78.json')['modules']}
    branches={mid:deepcopy(old[mid]['branches'])for mid in ['M08','M21','M22','M24']};branches['M02']=[]
    defs=[
        ('Frozen prefixseed/request eligibility and hitcounters','2perDPseeds outsideformalstats;235904/262144formalhitcounters. SavedpromptlogprobNone/saltNone/sessionNone refineseligibility,not physicalpage/hitID/hashnamespace identity.','ActualperrequestH/blockIDs/namespace/CPUhashcost still unknown;counterratio notKVgeneration proof.'),
        ('Incremental prefix hashing and crossrequest reuse namespace','Existinghasher812–869 appends uncomputedfullblocks;exactfrozeninputprefix evidence retained. Modelstate/cachevalue reuse needsallrelevant keys.','Hashimplementation timing/extra keys/reuse namespace remainunmeasured;do not research unrelatedfrontend CPUinternals.'),
        ('V41 tuple descriptor and normalstorage domain','Pinnedplannerusesfourindependenttuplebackings andoneglobalIDcount;KVPP/offload/otherlayouts remainconditional. Logicalaliasviews arenotindependentallocations.','Actualnormalallocator/override/descriptorvalues notsnapshot;newdomainmustretain allstate/SWA/long/index consumers.'),
        ('Shared global block-ID ring reservations and residentowners','AllthreeCircularBufferSpecs shareonegroup;sourceCircularBufferManager claimsonecommonID perresidentrequest.18ratio2consumerlayers mapto3stateownerobjects at2/8/14.','Sresident includingpending/resumableowners andrequiredUbankcapacity unknown;max_num_seqs4onlyscheduledbatch,nocapS/Uproof.'),
        ('UnpackedA3 p128 physicalpage arithmetic','ThreeC2tuples max(73856payload,131072SWA,131072ring)=131072each;C1tuplemax147712payload. C=540928B/commonID;typed72pool+72updatestateviewsN17306.','DeclaredunpackedBF16/INT8FP16scale/specdomain,notcurrentpointer/budget/packedA5/allrankruntimeproof;graphNnotdescribed.'),
        ('Standalone retained-even compaction and SWAanchor exclusion','Existingtuplemax/divisor unchanged whenring32→16 orremovedunderSWAanchor. Priorstandalonecapacityexclusion retained;2.268332032GBlogicalview isnotreleasableallocation.','Actualpartial/fullbyteobservers andnewkernel/packingstates separate;no extraGiBclaim fromrawshapes.'),
        ('Detach full32row state into privateownerbank','Newnormal-domainvariantpreservesall32logicalrows butremovesonecommonIDreservationperresidentowner. Bankmust bebudgetedbeforepoolallocation.','Δfree=S−(N0−N1)<=S;Umax+sentinel+metadata costcanerasegain,especiallyS<U. Newowner/capture/nativeABIrequired.'),
        ('Detach compact16row retained-even state','Newvariantuses3*16*1024*4=196608B/owner;additionalF34even-observation/rollback/oldreadbeforewrite conditions required.','Currentkernel32rows/globalpnotdropin;S/U/invalidslot/generation andallreaders/transferstates remainconditional.'),
        ('Packtwo compact owners percommon-ID statepages','EachofthreeC2pages131072Bcancontaintwo65536Bcompactstates. IdealIDs>=ceil(S/2),maximumreleasedfloor(S/2),beforemetadata/fragmentation.','Newhalf-owner/refcount/free/migration ABI;cannotfreewholeIDuntilbothownersretire. Fixedcounterfactualcapacityboundnotrealizedscheme.'),
        ('Statepublication/recycle/crossiteration and alias lifetimes','Namespaceincludeslayer/rank/tuple/globalID orprivateowner/half/epoch/position/acceptedstate. ABAexample showsstableaddress+shape doesnotprotectqueuedoldwrites.','RetireallreadersANDwriters;epochcheckalonecannotdropoldvalues neededbyoldreaders. Bankallocationcountsarenotpercalltraffic.'),
        ('Eviction/free/preemption/connector/error sourceconditions','Recordedpreemptioncounterdelta0isretained;doesnotexcludeeviction/recycle/transfer/errororunrecordedphases. SlotCSE doesnotremoveperlayerKVwrites.','Currentlifetimes/activebranchesandallbyteobserversremainunboundwhereconclusionchanging;keepbaselineoutsideprovedlocaldomain.'),
        ('Crossmodel allocation/capacity and modulecost boundaries','CurrentM02hasnoprimaryComputeInfoassignment;notzerocostorlowestNPUutilization. Keepwarm/coldstate/ownedbanks/cachegroups/CPUpublication distinct.','Typedphysicalmemory/CPUcache/hash capacities,causalrequesteffect,andfullapplicablemodelbranch analysisremainopen.')]
    m02e=names[:6]+['C2-state-opportunity-contract-v24.json','C2-cache-object-and-allocation-contract-v25.json','C2-cache-allocation-condition-ledger-v25.json','state-engram-operator-contract-v62.json','request-guard-saved-profile-v54.json','all-module-fixed-case-domain-audit-v78.json']
    for i,(branch,finding,gap)in enumerate(defs,1):branches['M02'].append(dict(id=f'M02-OP-{i:02d}',branch=branch,state='SCOPED_OPERATOR_STATE_STORAGE_AND_LIFETIME_ANALYZED_CONDITIONAL;WHOLE_MODULE_OPEN',finding=finding,evidence=m02e,gap=gap,scope='Frozenprefix/state/physicalstorage/metadata constraints relevanttooperators;notwholeframeworkresearch/implementation.'))
    details={
        'M08-OP-07':'Normalsharedpoolgroups3ringobjects useoneglobalID/request;newpackedhalf orprivateownerrepresentation needsnewlogicalringpage/metadata/epochmapping. Preservealloldreads beforeoverwritingevens;not allsharedtuplebytes exclusivelyringowned.',
        'M08-OP-08':'OldSWAanchorcapacityexclusion remains forunchangedtuple;newdetached32/16orpaired16whole-state-storage domains nowpricedparametrically. Threeviews2.268GB eachnot3independentallocations. Sresident isnot scheduledB4;layoutcapacity isnotpercallIO orspeedup.',
        'M21-S09':'CurrentC2headerscarryglobalblockp32rowABI;privateowner/half/epoch requiresnewproducer/consumer/capture schema. Existingfusedmetadata builderdoesnotautomaticallyprovidehalfslotidentity. Keepzero/invalidpage0/rowcounts andregisteredmetadata planes.',
        'M21-S15':'Ringstatekey adds tuple/owner/half/generation whilemain/draft/rank/phase/capturedbank identities remain. Scalar/shape/address equalitynotpublication;allreaderANDwriterretirement before reuse,perconcreteABAexample.',
        'M22-S06':'168oldworkerstaticnavigation sites nowexplicitly route toexistingM04/M05/M21operatorinput/metadata contracts. OriginalUNREVIEWEDguardstatuses remain;no168guardproofclaim orsilentframeworkscopeexpansion.',
        'M22-S07':'149oldworkerstaticnavigation sites nowroute toM15–M20/M24scores/RNG/aux/acceptedstatecontracts. Treatpublishing/COW/rejection/lastuse obligationsasoperatorconditions;do notstudywholebookkeepingframework.',
        'M22-S08':'323oldworkerstaticnavigation sites nowroute toM00/M02/M08/M09/M10/M18/M24storage/weights/cachecontracts. Newringjointstorage mathdistinguishesdescriptorcapacity/liveownerreservation/privatestorage;actualnativealiases andepochsstillunknown.',
        'M22-S09':'260oldworkerstaticnavigation sites route toinput/output/selection/lifecycleboundarymodules. All900associations retainedbyexactsourcehash/siteID;not900newoperatorbranches. 91otherunreviewedlifecycle sites remaininS05.',
        'M22-S12':'Newringstate32/private16/paired16variantsdeclarephysicalstateowner/tuple/half/stablecapturebank outputs. Pointerdebug/assertionsorweakrefregistrycannotprove samevaluegeneration/allobservers;newbanksmustfitmemorybudgetbeforecapture.',
        'M22-S13':'ConcreteABAtrace oldAupdate→reassignslotB→oldAwrite showsrequiredretirement allreadersANDwriters beforeepochpublish. Genchecks alonecannotdiscardoldreads neededforcorrectAoutputs. PackedIDfreewaitsallhalfowners;crossrank/stream fencesretained.',
        'M24-OP-06':'Ringrepresentationsnamespace tuple/block orprivateownerslot/half/epoch;reuseaddress doesnotpermit statecrossrequest/acceptedprefix drift. NewprivatefixedUbudget mayreducefreeIDs atlowS,pairedlayoutfragmentation/migration anddrain required.'}
    for mid,bs in branches.items():
        if mid=='M02':continue
        for b in bs:
            if b['id']in details:
                b['evidence']+=names[:6];b['ring_storage_lifetime_review_v79']=details[b['id']]
                if b['id']in ['M22-S06','M22-S07','M22-S08','M22-S09']:b['state']='OPERATOR_BOUNDARY_REFERENCE_CROSSWALK_RECONCILED;STATIC_GUARD_AND_RUNTIME_PROOF_NOT_PROMOTED'
    branches['M08'].append(dict(id='M08-OP-13',branch='C2 full32 privatebank,retained-even16 privatebank orpairedcompactowners per sharedID;jointphysicalstorage/metadata reorganization.',
        state='CONDITIONAL_NEW_STORAGE_REPARTITION_AND_CAPACITY_ACCOUNTING_REVIEWED',evidence=names[:6],
        gap='NormalC540928domain/Sresident/Ucapacity/costofmetadata/sentinel/fragmentation andallreader-writerepochs unknown;implementation/nativeperformanceoutsideanalysis.',
        scope='DeclaredC2observationalquotient/full32state domain only;doesnotremoveprojectedKV/scores orSWA/long/index cachewriters.'))
    mapping={1:[1],2:[11],3:[2],4:[3,10,11],5:[1,2],6:[1,2,12],7:[12],8:[4,6,7,8,9],9:[3,4,5,6],10:[10,11],11:[1,2,10]}
    write(O/names[7],dict(modules=branches,previous_M02_chronological_records=11,new_M02_semantic_families=12,
        old_M02_record_to_new_families={str(k):[f'M02-OP-{n:02d}'for n in ns]for k,ns in mapping.items()},
        new_M08_family='M08-OP-13',extended_existing_families=sorted(details),protected_other_modules=21,
        previous_global_records=340,current_global_records=342,previous_semantic_modules=23,current_semantic_modules=24,remaining_chronological_modules=['M00','M03'],whole_modules_closed=0,overall_complete=False))
    write(O/names[8],dict(completed_bounded_questions=[dict(id='C2-PHYSICAL-STATE-STORAGE-REPARTITION',state='CONDITIONAL_OPERATOR_ANALYSIS_REVIEWED',evidence=names[:3]),
        dict(id='M22-ADJACENT-OPERATOR-BOUNDARY-REFERENCE',state='900NAVIGATIONREFERENCESROUTED;NOTGUARDPROOFS',evidence=[names[3]])],
        questions=[dict(id='TYPED-CEILING-AND-MODULE-RANKING',state='NEXT_ANALYSIS_NOT_COMPLETED',question='Bind or explicitlybound actualtypedCube/Vector/nonlinear/Sort/integer/memory/network capacities usingprimaryarchitecture/SDK evidence. Reconcileexistingmodulework/denominators andlowestutilizationrequirements;publishedpeakreferencefraction isnot MFU.'),
            dict(id='BATCH-EXISTING-CANDIDATE-COMPOSITION',state='OPEN',question='Reconcileoriginalfusion-scan/sourcegraph/candidate-footprints andexistingolderrelationsinbatches. 549centralpairs areunreconciledhere,notproofnoearlierreview;include resource/padding/precision/statecouplings andlargergraphs.'),
            dict(id='M22-LIFECYCLE-OPERATOR-RELEVANCE',state='OPEN',question='91oldS05staticlifecycle sites remainnavigation. Inspectonlythosecontrollingactualoperatorinput/output/schema or16unboundgraphphases,notallframeworkinternals.')],
        actual_ring_owner_bound_or_implementation_not_required_to_deliver_conditional_models=True,
        unchanged_pending='9regimes1352executions=7MemSet1272+2BOOL80;849PMU/16phaseunbound/3136ambiguous',overall_complete=False))
    write(O/'PROFILE-CODE-QUESTION-GATE-v79.json',dict(status='C2_RING_STORAGE_RESERVATION_AND_OPERATOR_LIFETIME_SCOPE',
        previous_turn_classification='PROGRESS:v78sharedMLPconstructor/precisionrepartitionandallmodulecrosswalk',
        questions=['Howmanyrealringobjects/groupreservations versusratio2consumerlayers?','Doesprivate/pairedstate actuallychangepoolcapacity onceSWAanchors/budgetrounding andresidentownersareincluded?','Whichowner/half/epoch ABIandreader/writerretirement isrequired?','Are900runnerstaticassociationsindependentoperators orreferences toexistingoperatorcontracts?'],
        allowed_source='Sevenalreadyfrozenoperator/model/storage/configclauses byteverifiedagainstoriginalarchive orsavedconfig;6savedtypedstateviewregimes/graphcounts andexisting1101M22sitecatalogue. No broadframeworkcodeaudit/newserver/model/devicework.',
        new_remote_source_read=False,new_source_files_obtained=0,local_operator_clause_read=True,overall_complete=False))
    gap='Newringrepresentation domains areconditionalonactualresidentS/requiredcapacityU/sharedpoolnormalABI/overrides andallfield/readers/writerstateepochs. SourceC540928unpackedlayout andoneID/owner aredeclared-domainfacts,notliveallocator proof. Logical2.268GB/stateview isnotpercalltraffic orindependentallocation. Actualtypedcapacity/moduleminimumranking/wholemodelbranchcomposition and9/1352operands remainopen;implementation aloneisnotrequiredforconditionalanalysis.'
    updates={cid:dict(review_v79=dict(storage=str(O/names[0]),capacity=str(O/names[1]),layout_lifetime=str(O/names[2]),scope_crosswalk=str(O/names[3]),relation=str(O/names[4]),
        conditional_operator_analysis_deliverable=True,implementation=False,native_test=False,new_performance_result=False))for cid in ['F31','F34']}
    updates['F34'].update(profile_review_state='C2_PHYSICAL_STORAGE_PRIVATE_PAIRED_DOMAINS_AND_OWNER_LIFETIME_REVIEWED_CONDITIONAL',conclusion_changing_gap=gap,
        next_profile_or_contract_action='RetainpriorinplaceSWAanchorcapacityexclusion;newprivate32/private16/paired16domainspricedparametrically withonecommonID/residentowner. NeedU/S/metadata/epoch/nativebindings foractualchoice,notGiBclaim. Advance typedcapacity/modulebounds andwholecandidatecomposition;do not re-open oldplanner529proof.')
    updates['F31'].update(next_profile_or_contract_action='Ringrepresentationneedsnewowner/half/epoch metadataABI andallreadersANDwriters retirement. M22adjacent900sites nowreferenceexistingoperatorcontracts withoutguardpromotion;only decisivefield/capture/nativepublication sourcequestions. Continue typedceilings/composition,notframeworkresearch.')
    mods={mid:dict(finding='C2source3ringobjects/oneglobalIDperresidentowner,C540928normalunpackedtuplebudget. Private32/16andpaired16 haveparametricreservation/budget/rounding/generationmodels;newownerABAproof. M22adjacent900navigationreferencesroutedwithoutguardstatuspromotion. M02legacy11→12semanticfamilies,M08addsOP13.',
        evidence=names,candidates=['F31','F34'],gap=gap,profile_scope='Onlyoperatorstatefootprint/storageinitial-state/padding/metadataowner/lifetime anddefinedlocalfusioncomposition constraints.')for mid in ['M02','M08','M21','M22','M24']}
    remaining=list(read(O/'module-coverage-gaps-v78.json')['remaining'])
    remaining += ['v79C2stateownership:18ratio2backbonelayersbutonly3KVsourcecompressor ringobjects2/8/14. NormalV41allstate groupclaimsoneglobalID/residentrequest,not3IDs;fourtupledivisor540928inunpackedp128specdomain. ActualresidentS/bankcapacityU/descriptor/override/budgetremainder/graphN unknown;maxseq4notSbound. Inplacehalf-ring oldSWAanchorcapacityexclusionunchanged.',
        'v79newphysicalvariantsprivate32/private16/paired16conditionals:Δfree=S−Nlost<=S;bankprebudgetD=(U+sentinel)*3*rowbytes+metadata,atS<Ucannegative. PairedcapacityupperfloorS/2beforemetadata/fragmentation,half-owner/ABA/oldreaderANDwriterretirement/protocol required. 2.268332032GBlogicalstateviewN17306isnotreleasableallocationorpercalltraffic;noGEMMFLOP ormeasuredgain.',
        'v79M22S06–09all900oldstaticnavigationassociationsroutedtoexistingoperatorcontracts;originalUNREVIEWEDstatusesstay,no900guardproof. Prior110guarddataflowsretained;91otherlifecycleS05sitesstillnavigation. M02legacy11→12semantic/M08+OP13=current342records24semanticmodules;M00M03chronological,21othermodulematricesunchanged.46pairclauses10groups/549centralpending;2073/285598review9/1352pending/global84916/3136/0wholeclosuresunchanged.']
    action=('继续原Goal,v79 PROGRESS(前v78有效)。用户profile优先代码只围绕算子融合/理论上限。先读ring-storage-source-contract/capacity-models/state-layout-and-lifetime-proof-v79、M22-adjacent-operator-boundary-crosswalk-v79、next-operator-proof-questions-v79。'
        '七份已有frozenoperator/model/storage/config文件验hash，六份验证originalzipmember，未新取source/未remote。3实际ringobjects2/8/14，不是18个ratio2consumerlayers；source共用globalIDpool，各3Circular specs合一group，normalmanager每residentrequest占1ID，不是3ID。'
        'NormalunpackedA3block128:3C2tuplecapacity131072，1C1tuple147712，ΣC540928。Ring32→16或移除state仍SWAanchor不改C，旧standalonecapacityexcluded成立。72pool+72updateDIRECT typedN17306/32/1024；graphN/A不能推指针或N，原80graphs240pairs相位192bound48unbound保持。'
        '新whole-state-storage三域(不实现)：private32保全部逻辑值，private16额外even-observation，paired16每三C2pages各放两owner的16rows。Private预算D=(U+z)*L*R+metadata必须先从cachebudget扣，Nnew=floor((B−D)/C)，Δfree=S−(Nold−Nnew)<=S。U是residentcapacity上界，max_num_seqs4是scheduled不是S/U证据；固定U在S小会减少freeIDs。z0需新validityABI，沿用0invalid要多一ownerpage。'
        '示例U4/S4/z0/noextrametadata：private32net+1..2IDs，private16+2..3IDs；S2同U4为private32−1..0、private16 0..1。只是budgetremainder范围公式不是实际提升/容量快照。Pair16理想每ID2owner故最多floorS/2释放，实际fragmentation/migration及metadata预算扣除另计。2.268GB单state逻辑view不独立allocation更不是每callIO；projection/pool/norm工作不因bank搬家自动省。'
        '32slotcompacteven映射half*16+((pos//2)%16)保留旧even残基，32条slot映射无冲突/50000整数检查非native/formaltests，仍守F34v24evenhistory/rollback/fullbyteobserver条件。private32也需新owner ABI但不需删odd。ABA旧A异步write→slotBepoch重用→旧write破坏B证明必须lastreader AND lastwriter drain；只比gen丢旧操作不能保旧reader。PairwholeID等两half都retire才释放。'
        'M22S06/S07/S08/S09=168/149/323/260共900旧static navigation一一按sourcehash+siteID路由现有operators契约，UNREVIEWED状态原样保留，不是900新guard proofs；已有110exactguard保留，其余91S05lifecycle仍navigation。M02旧11archive归一12semantic，M08旧12保留+OP13，M21/M22/M24旧IDs保留富化11条；现342record24semantic，21其他module完全不变。'
        '新REL79-01F31/F34 stateowner/half/epoch ABI，HG73C2加入F31。46clauses10groups549central未汇入，不等于549没研究或不能融合。下步优先typed hardware/SDK容量和模块同口径实际/上界/最低利用率证据，再批量从originalfusion-scan/sourcegraphs/candidatefootprints汇入旧关系/组合；控制框架代码阅读仅限算子schema/读写/alias/RNG/error/state-publication合同，不沿900导航展开框架研究。'
        '全2073/285598conditionalreview9/1352pending(7MemSet1272+2BOOL80)/849PMU16phase3136ambiguous/35model7legacyframework0wholemoduleclosure保护。3PASS，无本轮ERROR；未remote/model/Torch/NPU/新采集/服务/进程/实现。verify_ring_storage_domains_v79应PASS。Goalactive不因成稿complete。')
    report='''本轮补齐了C2 ring状态重组的物理容量和生命周期条件，并将图捕获台账中的相邻模块引用明确路由到算子契约。没有执行模型、Torch、设备测试、远程任务或优化实现。

固定配置有18个ratio2消费者，但只有层2、8、14三个KV源创建独立ring。普通V4.1池将这三个state spec合为一组，每个驻留请求占用一个共享全局block ID；不能按三层分别释放三个ID计价。当前已保存的72次pool和72次update描述了17306×32×1024的逻辑state视图，单视图约2.268GB，但它与SWA/长缓存共享backing，不是独立可释放分配，也不是每次调用的访存量。

在明确的非packed A3、block128、BF16长缓存/SWA和INT8 index＋FP16 scale域内，四个tuple页容量为131072、131072、131072、147712B，合计540928B/全局ID。原有SWA锚定结论保持：只把ring从32行缩成16行，仍不缩小原tuple分配。

新分析的是三个不同适用域：把完整32行状态拆到独立owner bank；再加保留偶数状态约束的16行bank；或者在共享页中放入两个请求的16行状态。前两种须先从可用预算扣除新bank，再计算pool页数。净空闲ID变化为S−(旧页数−新页数)，其中S为驻留state owner数，U为新bank容量。max_num_seqs=4只是调度批量，不能证明S或U最多4。

| 示例：固定U=4、新有效slot协议、未计额外metadata | 当前S=2 | 当前S=4 |
|---|---:|---:|
| private32净空闲ID变化 | −1至0 | ＋1至2 |
| private16净空闲ID变化 | 0至1 | ＋2至3 |

范围来自未知预算余数，表示条件容量核算，不是实际运行结果。沿用slot0无效规则还需额外预留页；metadata、对齐和生命周期可继续增加成本。双owner共享页在理想打包下最多释放floor(S/2)个ID，实际还受碎片、迁移和两个owner共同释放协议限制。

compact映射把两个owner分别放在0–15、16–31行，并保留旧32行中的偶数残基。它依赖已声明的历史/回滚可观测域，不能用于存在奇数行或全字节观察者的场景。另一个明确反例是ABA：旧请求A的排队写入未结束就把同地址交给B，旧写入随后会破坏B状态。因此回收必须等待全部旧读者和排队写者完成；地址不变或只比epoch都不足以证明安全。

M22原有900个相邻worker静态位置已逐一按原source hash和site ID路由到已有输入、采样、缓存及生命周期算子契约。原UNREVIEWED guard状态保留，没有把导航索引升级为900项语义证明；已有110项局部guard分析保持，另外91个生命周期位置仍待按算子相关性核对。M02的11条旧记录已归一为12个状态/存储语义分支，M08增加1个重组分支，完整历史均保留。

当前为342条模块分支记录、24个语义模块，46条候选对关系和10组组合约束。2,073类、285,598次执行保留条件分析；9类、1,352次关键操作数证据、实际typed容量及完整组合分析仍未补齐。下一步优先推进同口径理论上限和模块利用率证据，并批量汇入既有图与候选关系。Goal继续进行中。
'''
    jobs=[dict(kind='profile',status='pass',description='6DIRECTstateviewregimes72pool+72updateN17306/32/1024;480graphpool/updateTASKrecordsretaingraphNunknown. 900M22adjacentstaticIDs routedexactworkerhash withoutguardpromotion;110priorproofs/91lifecycle sitesretained.',artifacts=[O/names[0],O/names[3],N/'reconcile_operator_lifetime_scope_v79.py']),
        dict(kind='design-check',status='pass',description='7frozenfiles/6zipmembersverified;source3ringobjects/onecommonIDperresidentowner,normalC540928. 68privatebankbudgetmodels,32injectiveslots/50000integerchecks,ABAretirementcounterexample and5alternativestatevariants. Noactualownerbounds/nativecapacity/implementationclaimed.',artifacts=[O/names[i]for i in[0,1,2]]+[Path(r['path'])for r in source['source_files']]),
        dict(kind='review',status='pass',description='M02legacy11→12semantic/M08+OP13;retainalloldotherIDsand21othermodulematrices. NewF31F34relation46/549andHG73C2F31;2073/285598review9/1352pending/global84916/3136/0closuresprotected.',artifacts=[O/names[i]for i in[4,5,6,7,8]])]
    persist_refinement(version=79,previous=78,title='C2 physical state-storage domains, shared-ID capacity and operator lifetime contracts',
        summary=dict(filename='ring-storage-lifetime-summary-v79.json',previous_turn_classification='PROGRESS:v78sharedMLPconstructor/precisionrepartitionandallmodulecrosswalk',
            source_files_reused=7,archive_members_verified=6,new_source_files_obtained=0,new_remote_source_read=False,source_ring_objects=3,ratio2_consumers=18,
            shared_pool_divisor_bytes=540928,reservations_per_resident_request=1,private_bank_models=68,layout_slots=32,offline_integer_checks=50000,
            direct_state_views=[72,72],direct_state_N=17306,graph_state_N_attested=False,routed_M22_navigation_sites=900,new_M22_guard_proofs=0,prior_M22_guard_proofs=110,unrouted_M22_lifecycle_sites=91,
            current_semantic_modules=24,current_records=342,new_M02_families=12,new_M08_family='M08-OP-13',protected_other_modules=21,relation_clauses=46,higher_order_groups=10,pending_central_pair_reconciliation=549,
            scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
            implementation=False,native_test=False,new_performance_result=False,all_actual_physical_numeric_ceilings_proved=False,
            remote_job_status='No remote/server/model/Torch/NPU/capture/service/process work;local frozenoperator/storageclauses andsavedprofile/arithmetic only.'),
        changes=['NormalC2sharedpoolhas3objects/onecommonIDperresidentowner;private32/private16/paired16storagegraphsandbudget/rounding/ownerlifetimemodels,notGiBsavings.','900existingM22navigationassociations routedwithoutnewguardproofs;M02legacy11→12semantic/M08+OP13,current342records24semanticmodules.','NewF31F34owner/half/epochrelation46/549andHG73C2F31,allglobalprofileuncertainty/0wholeclosurespreserved.'],
        candidate_updates=updates,framework_updates={},module_updates=mods,semantic_branches=branches,remaining=remaining,action=action,jobs=jobs,report=report,
        read_first=names,publication_rows={},gate_updates=dict(overall_complete=False),scripts=[N/'analyze_ring_storage_domains_v79.py',N/'reconcile_operator_lifetime_scope_v79.py',N/'verify_ring_storage_domains_v79.py'],
        executor_prefix='ring-storage-domains',main_script=Path(__file__),source_read=False)


if __name__=='__main__':main()
