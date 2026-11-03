from copy import deepcopy
from collections import Counter
from datetime import datetime,timezone
from finalize_origin_coverage_v9 import N,O,read,write
from finalize_Engram_whole_frontier_v30 import packet,graph
from persist_profile_refinement_v2 import persist_refinement


def main():
    assert not (O/'REVIEW-CHECKPOINT-v39.json').exists()
    stamp=datetime.now(timezone.utc).isoformat()
    specs=[('M21-C2166-v39','read_M21_C2_and_fill_v39_166.py'),('M21-C2-fields166-v39','read_M21_C2_fields_and_fill_geometry_v39_166.py'),('M21-C2-fused166-v39','read_M21_C2_fused_builder_and_publication_v39_166.py')]
    pp={k:packet(k,True)for k,_ in specs}
    native=packet('M21-native166-v37',False)
    old_consumer=packet('C2-source166-v24',True)
    pins={}
    for p in pp.values():
        for f in p['result']['files']:
            old=pins.setdefault(f['path'],f['sha256']);assert old==f['sha256']
    for f in old_consumer['result']['files']:
        if f['path'].endswith('/attention/dsa_v41.py'):assert pins[f['path']]==f['sha256']
    refs=[dict(receipt=str(N/(k+'.json')),remote=p['full_result'],sha256=p['result_sha256'])for k,p in list(pp.items())+[('M21-native166-v37',native),('C2-source166-v24',old_consumer)]]
    old_coverage=read(O/'M21-profile-chain-coverage-frontier-v38.json')
    selected=[c for c in old_coverage['selected_chains']if c['state']=='UNANALYZED_FIELD_OR_CONSUMER']
    assert len(selected)==39 and sum(c['TASKs']for c in selected)==4280
    c2=[c for c in selected if ': build_c2_metadata'in c['leaf_project_frames'][0]]
    fills=[c for c in selected if '_get_block_table_and_slot_mapping'in c['leaf_project_frames'][0]]
    qsl=[c for c in selected if '/model_runner_v1.py(1462)'in c['leaf_project_frames'][0]]
    assert (len(c2),sum(c['TASKs']for c in c2),len(fills),sum(c['TASKs']for c in fills),len(qsl),sum(c['TASKs']for c in qsl))==(30,3256,8,936,1,88)
    keys={c['chain_key']for c in selected};heads=[h for h in native['native_chain_headers']if h['chain_key']in keys]
    op_counts=Counter();byline={}
    for h in heads:
        leaf=h['project_frames'][0];ctx=tuple(h['contexts']);z=byline.setdefault((leaf,ctx),Counter())
        for s in h['typed_shapes']:z[s['signature']['op']]+=s['TASKs'];op_counts[s['signature']['op']]+=s['TASKs']
    gather=sum(s['TASKs']for h in heads if ': build_c2_metadata'in h['project_frames'][0]for s in h['typed_shapes']if s['signature']['op']=='GatherElementsV2')
    assert gather==208
    files=dict(profile='M21-C2-and-initialization-profile-v39.json',contract='M21-C2-metadata-fill-and-existing-fusion-contract-v39.json',
        proof='F31-C2-existing-fused-builder-equivalence-v39.json',fill='H06-initialization-write-ownership-review-v39.json',
        variants='M21-C2-and-initialization-variants-v39.json',coverage='M21-profile-chain-coverage-frontier-v39.json',
        branches='M21-C2-initialization-branch-ledger-v39.json',graphs='M21-C2-and-initialization-graphs-v39.json')
    write(O/files['profile'],dict(generated_utc=stamp,reused_native_scope=dict(APIs=60404,TASKs=76033,chains=186),selected_chains=heads,
        selected_TASKs=4280,C2_TASKs=3256,block_slot_fill_TASKs=936,QSL_fill_TASKs=88,C2_FP32_gathers=208,
        line_context_ops=[dict(leaf=k[0],contexts=list(k[1]),ops=dict(v))for k,v in byline.items()],op_counts=dict(op_counts),evidence_refs=refs,
        notes=['104executions of each principal C2 source statement (88OTHER_RETAINED+16DUMMY);two skip zero statements16each. Aggregate31*104+32=3256 is arithmetic reconciliation,not reconstructed request/callback pairing.',
            'FP32gather tables[1048576,1,1,64] select6400/6528/6672/24rows inOTHER and24rows inDUMMY. No pairing of these metadata rows to6668pool rows bytime/count/ordinal.',
            'Sourcefive ring-row copy_ andcomplete/sourceposition copy_ are retained effects even without a separate positive ComputeInfo at each store. Typed event absence doesnot prove missingcopywork.',
            'QSLfill88 andslot/tablefill936 are different banks/effects. These are selectedTASK counts,not all hostfillAPI counts,physicalbytes,criticalpath ornewfunctionalPMU assignments.'],whole_module_closed=False,overall_complete=False))
    facts=[
        ('DOCUMENTED','C2 outputs have persistent banks:INT32[5,R]ring metadata,BOOL[N]complete,INT64[N]sourcepositions,FP32[N,1,1,D]cos/sin. Capacity depends onbuild-compressor setting;returnedviews andmetadata_group_id retain fixedbank addresses.'),
        ('MEASURED_PLUS_DOCUMENTED','CurrentpositiveC2elementwise/gather fullcallers are thelegacy callback branch1304-1373. Samecallback already calls build_c2_ring_metadata forpackedcache withfulltables;thehelper isexistingcode,not anew missingproducer.'),
        ('DOCUMENTED','Legacycallback usesGPU QSL/seq_lens/positions/firstblock,actualrequest/token extents andskip flag. used=max(min(end,A)-start,0)maskedbylive/skip;ringrows are[max(seq_len-query_len,0),used,start,start,firstblock ifused>0else0].'),
        ('DOCUMENTED','complete=(token_index<min(QSL[actual_reqs],actual_tokens))and(position.remainder(2)==1),maskedfalse onskip. sourceposition=position-1 ifcompleteelse0. BothfullFP32tables aregathered attheseindices,includingrow0 forincomplete/skip tokens.'),
        ('DERIVED','Preexisting fusedkernel implements thesamefive rows/complete/sourceposition/two tableloads in one launch. Itswrapper takesplainQSL/seq/positions/firstblock/fulltables/outputviews;no packed-cache object or packed ringdataformat isread. Reusing it outsidepackedguard isaconditional source-levelmechanism;no nativecompilation/numerical/performancevalidation claimed.'),
        ('DOCUMENTED','Fusedhelper sharescos-table strides forsin loads andcos-outputstrides forsin stores;wrapper checks shapes/width butnot these strideequalities. Preserve matching strides/currentI32/I64/FP32domains oradapt explicitseparatestrides. Negative-position remainder andinteger-narrowing domains mustnot beassumed equivalent outsidevalidatedvalues.'),
        ('DOCUMENTED','_publish_task already deduplicates shared[key] andreturns existingbuffer;otherwise schedules aDeviceMetadataTask orrunssynchronously fordisableddevice-metadata/capturedCOMPRESSOR mode. Caller publisheskeyc2:compressor onstablemaskbank. This isexistingdedup/asyncprotocol,not a new sharinggain.'),
        ('DOCUMENTED','Exactv24sameSHAconsumer waitsCOMPRESSOR frontier,projectsFP32KV/score,pool_projected consumes ring metadata andprivateoldring,then norm. Cachedsourcecos/sin orfallback get_cos/sourcepositions feedindexer.update_keys andinplace latentRoPE beforelongcachewrite. Keepboth consumers andoldring-before-update WAR.'),
        ('DOCUMENTED','skip_ring_update clears used andcomplete;ring0,startrows remain dynamic. sourceposition0still publishescos/sin table0values. It doesnotmeanallmetadatazero,identityRoPEorentirecallbackdead.'),
        ('DOCUMENTED','Nonencoder _get_block_table_and_slot_mapping viewsactualgroup slots/table,then fills slot[num_tokens:num_tokens_padded]with-1 andtable[num_reqs:num_reqs_padded]with0. Encoderbranchallocatesdifferentzero tensors. numpad defaultsnumactual viaor;worker passesactualunpadded/padded extents separately.'),
        ('DOCUMENTED_PLUS_DERIVED','ImportedPAD_SLOT_ID isliteral-1;v38fusedslotkernel alreadywrites-1 on[num_source_tokens,fused_max). A later slotfill isidempotentonly ifsamegroupbank/physicalinterval/generation,value,earliercoverage andnointerveningwrite. It doesnotcoverblocktable zeroorQSLsentinel writes anddoesnotprove all936fillTASK removable.'),
        ('DOCUMENTED','Worker1462 fillsquery_start_loc[num_reqs+1:]with-1. Prefix/invalidtail canbe readbycapturedfixedshapes;dirty-range maintenance needs knowninitialsentinel andall writer footprints,not merelycurrentactiveB.'),
        ('UNKNOWN','Livefield values/strides/pointers/owner generations,complete external/fullbank readers,physicaltraffic/cost andhistoricalexecutedbinaryidentity areunobserved. Keeptheseascandidateconditions;genuineunreviewedcoordinate/CP/proposer/QLI branchesremain separate.')]
    write(O/files['contract'],dict(generated_utc=stamp,source_pins=pins,evidence_refs=refs,facts=[dict(grade=g,statement=s)for g,s in facts],
        reused_contracts=['C2-state-opportunity-contract-v24.json','C2-cache-object-and-allocation-contract-v25.json','M18-partition-bank-RoPE-and-slot-contract-v34.json','M21-count-metadata-and-output-publication-contract-v37.json','M21-slot-rejection-state-and-packing-contract-v38.json'],
        local_questions_closed=['C2legacycoordinate/mask/source-tableoutputs','Existingfusedbuilder mechanismanddomainconditions','DirectC2pool/indexkey/longKVreaders','Existingpublicationdedup/frontier','QSL/slot/tablefillownership andconditionalidempotence'],implementation=False,whole_module_closed=False,overall_complete=False))
    write(O/files['proof'],dict(generated_utc=stamp,evidence_grade='SOURCE_DERIVATION_WITH_EXPLICIT_DOMAIN',existing_fused_builder=True,
        shared_formulas=dict(q='end-start',used='max(min(end,actual_tokens)-start,0) if req<actual_reqs and notskip else0',
            ring0='max(seq_len-q,0)',ring1='used',ring2='start',ring3='start',ring4='firstblock ifused>0else0',
            complete='i<min(QSL[actual_reqs],actual_tokens) and pos%2==1 and notskip',sourcepos='pos-1 ifcompleteelse0',cos='fullcos[sourcepos]',sin='fullsin[sourcepos]'),
        equivalence_domain=['ActualQSL/seq_lens andringoutputs fitbaselineINT32,positions/sourceindicesINT64 andvalid0<=actual_reqs<=R;allindex/addressbounds valid.',
            'Preservetorch.remainder versusTriton% semantics;nonnegativevalidpositions establishthesameparity. Negative unsupportedpositions mustnotbe silentlynormalized.',
            'Sourcecos/sin FP32tables havecompatiblewidths andthecurrenthelper sharedrow/dimstrides;outputcos/sin havecompatible sharedstrides. Otherwisepassseparatestrides ratherthanreuseunchangedhelper.',
            'Allpersistentoutputs aredisjointfromstill-readinputs oruseasnapshot-safeprotocol. Input/outputviewextent,capacity,padding andwriteeffects match.',
            'Bothfulltablesready andsameconfig/layer generation. Fulltableabsent fallback retained;no claimcachedrow0is1/0without valueproof.',
            'Keep COMPRESSOR publicationkey/group/frontier,capturedmodeandinput/reuse/last-readerfences includingR0/N0/skip. Originalwrappervalidation/errors retainedonvalidcall domain.'],
        why_not_packed_only='Selectedkernel readsno packedcacheobject;itsring5rowandRoPEoutput ABI matcheslegacycallback. Guard/platformbroadening isstillnotanexecuted910C/CANN9.2 correctnessorcompilationproof.',
        skip_effects=dict(used=0,complete=False,source_positions=0,cos_sin='Actualfulltablerow0 replicated;notassumedidentity.',retained_dynamic=['ring0','ring2','ring3','bankfrontier']),
        plan_reuse_warning='Identicalskip-inputkey doesnotproveoutputbankstillcontainscachedvaluesafteraninterveningnon-skipwriter. Needbank-generationattestation,invalidation,oraseparateimmutableplan/snapshot withsafeconsumerABI.',
        implementation=False,native_validation=False,critical_path_gain_us=None,whole_module_closed=False,overall_complete=False))
    write(O/files['fill'],dict(generated_utc=stamp,framework_candidate='H06',related=['F31','H02'],evidence_grade='DOCUMENTED_EFFECTS_PLUS_CONDITIONAL_STORE_IDENTITY',
        effects=[dict(bank='QSL',extent='[num_reqs+1,capacity)',value=-1),dict(bank='group_slot',extent='[num_tokens,num_tokens_padded)',value=-1),dict(bank='group_table',extent='[num_reqs,num_reqs_padded)',value=0)],
        duplicate_slot_proof='IfearlierfusedwriteE=[sourceN,fusedMax)andlaterL=[metaN,metaPadded) useidenticalphysicalbank/generation/value-1,sourceN<=metaN<=metaPadded<=fusedMax,andnointerveningwrites toL,thenremovinglaterunused-returnfillleavesallvaluesandrequiredconsumerreadiness unchanged. Thisisconditional,notcurrentallgroupbinding.',
        dirty_tail_proof='LetbankcapacityCbeinitializedtosentinelv,allnewnon-sentinelwritesbeprefix[0,a),andpreviouspublishedtail[a_prev,C)remainunchanged. Ifa<a_prevonly[a,a_prev)needsreset;ifa>=a_prevtail[a,C)alreadyhasv. Preserveinitialfullclear andinvalidatewhenanyotherwriter/alias/reallocation/reset changesinactivebytes.',
        shrink_example=dict(previous_active_end=4,current_active_end=2,must_reset='[2,4)',retained_tail='[4,C)',warning='Skippingalltailwritesonshrinkwouldleaveoldactivevalues;shapeorgraphbucketaloneisnotdirtytracking.'),
        obligations=['ActualCP/SP/globalunpadded extents andgroupidentity,notordinalcounts.','Allcaptured/fullbankreaders andallwriterfootprints,includingrequestshrink/grow/slotreassignment.','Previouslastreader completebeforeinplaceoverwrite;retainproducerready/reusefences.','Do notremoveencoderzeroallocation orblocktable/QSLwritesbasedonslotPADproof.','Noall936TASK/hostfillAPI/physicalbyte/time savinginferred.'],
        implemented=False,critical_path_gain_us=None,whole_module_closed=False,overall_complete=False))
    variants=[
        ('F31-v39-EXTEND-EXISTING-C2-BUILDER','F31','Reuse or adapt existing fused C2 metadata builder in the observed legacy output domain.','Samefive ringrows/mask/sourcepos/FP32cos/sin,valid arithmetic/index/stride domains,outputbank ABI andpublicationfrontier;existingpackedguardisnotproofofsemanticimpossibility.'),
        ('F31-v39-C2-LAZY-ROPE','F31','Fold source-position selection/table loads into compatible index-key/RoPE consumers,or share an exact readonly lookup plan.','Keepbothindexer.update_keys andlonglatentRoPE,fullsource table/config generation,allfixedbankreaders/captureABI;currentpublishdedupisalreadyimplemented.'),
        ('F31-v39-C2-COMPACT-COORDS','F31','Use a producer-consumer ABI that represents equal ring start rows once or derives fields in the consumer.','Currentrows2/3bothstartbut5-rowformatispublic toprivatehelper;newABIretainsbothlogicalfields,allpool/update offsets andoldringWAR;cannot simplydroponecurrentrow.'),
        ('F31-v39-C2-POSITION-PIPELINE','F31','Compose C2 coordinate/position production with compatible corrected-position metadata pipeline.','SameGPUauthoritativeQSL/seq/position epoch andrequestmap,oldstate snapshot,alllivepublishedbanks/CPalternative paths;newcrossprogramdependencies explicit.'),
        ('F31-v39-C2-SKIP-TOKEN-PLAN','F31','Reuse exact zero-complete/sourcepos and fulltable-row0 token plans for skip-update calls where bank generation permits.','Dynamicring0/startrows stillproduced;fulltable row0 notassumedidentity. Invalidateafteroverwritingwriteroruseimmutableplan;keepfrontier andallslot/cacheconsumers.'),
        ('H06-v39-REUSE-FUSED-SLOT-TAIL','H06','Avoid a later idempotent slot PAD fill when an earlier fused slot tail fully covers it.','Prove exactbank/value/CPextent/generation/nointerveningwriter. Doesnotcoverblocktablezero/QSLsentinel/encoderallocation;refinesv38tailcandidate.'),
        ('H06-v39-PAD-DIRTY-RANGE','H06','Maintain only newly dirtied inactive ranges of QSL,slot or block-table banks.','Initialsentinelclear/allwriterfootprints/shrink-grow/alias/capturefullreaders andlastreader fences explicit;perbankvaluesandunits retained.')]
    vv=[dict(id=i,parent=p,transformation=t,conditions=q,implemented=False,native_validation=False,critical_path_gain_us=None)for i,p,t,q in variants]
    reqs=[
        ('SCOPE','39chain4280TASK localcontractcoverage isselectednativeC2/fill scope,notwholeM21orfunctionalresource reclassification.'),
        ('EXISTING','PackedC2fusion,publish-keydedup,asyncfrontiers andv38fusedslotPADalreadyexist;none countednew.'),
        ('RING','Retainring[5,R] values,INT32layout/casts,actualreq/token clipping androw2/3logicaloffsets.'),
        ('PARITY','Keepremainderparity/sourcepos arithmetic;nonnegativevalidpositiondomain orvalidatednegative semantics required.'),
        ('BOUNDS','ExactQSLactualrequestindex,source-tablebounds andnormalinteger ranges;numreq/token0/skip paths preserved.'),
        ('STRIDES','Currenthelper sharescos/sin source/outputstrides;proveequal orprovideindependentstrides. Shapeequalityaloneinsufficient.'),
        ('DTYPES','CurrentI32QSL/seq/ring,I64positions,Boolmask,F32tables/out preserved;nonnormaldtypebranchseparate.'),
        ('TABLEGEN','Fullcos/sin ready/config/layer/version same;fallbackwithoutfulltablesretained.'),
        ('BANKALIAS','Noinputoverwrittenbeforeallreads;fixedoutputbanks/capacity/viewstrides/tailsandexternalreaders preserved.'),
        ('FUSIONBIND','Sourcemechanismnotexecutedkernelproof:910C/CANN9.2nativecompilation/numericcostremaincandidateconditions,notrequirementtoimplementfordiscovery.'),
        ('PUBLISH','Keepc2:compressor key,masksidentity/stage/group/frontier,dedup,andcapturedCOMPRESSORsynchronouspath;alreadysharedworknotadditive.'),
        ('READY','Input-ready/reusefence/allconsumerwaits/lastreaderrelease remainoncachehit,skip,emptyanddirectout.'),
        ('CONSUMERS','Poolringmetadata,FP32oldring,WAR,indexkeyupdateandlonglatentRoPE/cachewrites allretained.'),
        ('SKIP','skip clearsused/completebutnotring0/startrows;sourcecos/sinarestillactualtablerow0,notnecessarily1/0.'),
        ('CACHEOUTPUT','Inputplanmatchnotoutputbankgenerationmatch;invalidateafterinterveningwriteroruseimmutablealias-safeplan.'),
        ('LAZYABI','Removingmaterializedlookup/coordbanksrequiresallreaders/captureABI,newconsumerprotocolandfirstready proof.'),
        ('POSITION','CorrectedGPUQSL/seq/positions andCP/SProwmapstay authoritative;CPUupperbound/shapeinsufficient.'),
        ('SNAPSHOT','v38previousstate read-before-write preservedwhencomposingposition/C2 producers acrossrequests/iterations.'),
        ('PADSCOPE','QSL-1/group-slot-1/group-table0/encoderzerosdistinctbanks/effects;noallfillremovalclaim.'),
        ('SUBSET','Idempotentslotfill requires sourceN<=metaN<=metaPadded<=fusedMax inexactphysicalsamegroupbank.'),
        ('PADGEN','EarlierPADvalue-1andnointerveningwriter pluslastreader/readyfences required,notmerelysameTensorname.'),
        ('INITIAL','Dirtytracking startsfromknownsentinelinitializedbank;zeroallocatedQSLdoesnotinitiallyequal-1.'),
        ('DIRTY','Trackallnon-sentinelwriterfootprints,aliases/reallocation/reset andshrink/grow;activeB/capturebucketaloneinsufficient.'),
        ('CAPTURE','Allfixedshape/fullbankreaders/capturedaddresses andgroupownerlifecycle honored;requestshrink needsclearoldprefix.'),
        ('EVIDENCE','Typed208gathers/3256TASK/936fill/88QSL countsandlogicalshapes notcausalcriticalpath/physicalbytes/speedup.'),
        ('OVERLAP','C2fusion/lazylookup/compactcoords/positionpipeline/skipplan sharework;PADdirtyrange/duplicatetail overlapv38/H02 andcannotbesummed.')]
    write(O/files['variants'],dict(generated_utc=stamp,variants=vv,conditions=[dict(id='M21-v39-'+k,requirement=s,actual_current_legality_proved=False)for k,s in reqs],
        implemented=False,whole_module_closed=False,overall_complete=False))
    cs=deepcopy(old_coverage['selected_chains']);changed=[]
    for c in cs:
        if c['state']!='UNANALYZED_FIELD_OR_CONSUMER':continue
        c['previous_state']=c['state'];leaf=c['leaf_project_frames'][0]
        c['state']='ANALYZED_CONDITIONAL_C2_INTERFACE'if ': build_c2_metadata'in leaf else'ANALYZED_CONDITIONAL_INITIALIZATION_EFFECT'
        c['evidence']=[files['contract'],files['proof'],files['fill'],files['variants']];changed.append(c)
    assert len(changed)==39 and sum(c['TASKs']for c in changed)==4280
    assert not any(c['state']=='UNANALYZED_FIELD_OR_CONSUMER'for c in cs)
    partial=[c for c in cs if c['state'].startswith('REUSED_PARTIAL')];assert len(partial)==108
    remainder=[
        'Finiteper-field reconciliation of108partialchains (101CP ancestry+7proposer):compressed_slot_mapping201/202;DSAbuild1008-1042 page/slot/cmp/residual logic;reusev13knownbatch905/915 andv34RoPE/firstpass ratherthanblindreread.',
        'Extendtargetv13later-CP deadproducerproof to16dummy/88draft contexts onlyafterexactshared-batch/miss generation proof. CP suffix2792=2288prepare+416dummy+88draft notallredundant.',
        'QLI emptyquery/captured eager-reader behavior remainsunreviewed;SMLA/C2skip rulescannotbe borrowed. Selectednativechainnavigationdoesnotenumerateallinactivebranches/captureguards.',
        'ThenM22graph/queue/transport,M23-M25 andglobal finite module/branch/cross-layer/card/iteration completionaudit.35candidatesand0unassignedselectedchainsnotwholegoalcompletion.']
    write(O/files['coverage'],dict(generated_utc=stamp,selected_chains=cs,state_counts=dict(Counter(c['state']for c in cs)),newly_analyzed_scoped_chains=39,newly_analyzed_scoped_TASKs=4280,
        remaining_unassigned_chains=0,remaining_partial_chains=108,genuine_unanalyzed_frontiers=remainder,
        completion_policy='Zero unassigned native chains only completes navigation/localassignment in this186-chain selector. Partialfield/branch/reader andwholeprocess domains above remain realwork. Conditional implementation/performancevalidation isnot itselfanunanalysedrange.',whole_module_closed=False,overall_complete=False))
    branch_rows=[('LegacyC2callback','ANALYZED_CONDITIONAL','104principalstatementexecutions/208FP32gathers;31*104+32aggregate,notrequestpairing.'),
        ('Existingpackedfusedbuilder','ANALYZED_SOURCE_MECHANISM','Exactfive-row/mask/position/tableload ABI;legacyreuseconditionalonarithmetic/stride/bankdomain.'),
        ('skipRingUpdate/empty','ANALYZED_LOCAL_EFFECT','used/complete0,sourcepos0,row0tablevalues;dynamicringstart/seqfields andfrontierretained.'),
        ('Publication/capturedmode','ANALYZED_LOCAL_PROTOCOL','Alreadydedupc2key/deviceTask orsyncCOMPRESSOR;v37eventprotocol reused.'),
        ('Slot/table/QSLinitialization','ANALYZED_CONDITIONAL','Differentbanks/values;idempotentsubsetanddirtyrange proofconditionalonallwriters/generations.'),
        ('PartialCP/proposer/QLIreader','UNREVIEWED_REMAINDER','108partialchainfieldcontracts andQLIcapturedreaders needfiniteaudit,notreportcompletion.')]
    write(O/files['branches'],dict(generated_utc=stamp,branches=[dict(branch=x,state=y,evidence=z)for x,y,z in branch_rows],whole_module_closed=False,overall_complete=False))
    gg=graph(['GPU_QSL_seq_positions','same_epoch_full_rope_tables','existing_or_legacy_C2_builder','fixed_ring_and_source_banks','COMPRESSOR_frontier','pool_oldring_then_update','indexkey_and_longlatent_RoPE','all_last_readers_release'],[
        ('GPU_QSL_seq_positions','existing_or_legacy_C2_builder','EXACT_ACTUAL_EXTENTS_SKIP'),('same_epoch_full_rope_tables','existing_or_legacy_C2_builder','ROW_LOOKUPS_INCLUDING_ZERO'),('existing_or_legacy_C2_builder','fixed_ring_and_source_banks','PRESERVE_OUTPUT_ABI'),('fixed_ring_and_source_banks','COMPRESSOR_frontier','EXISTING_PUBLISH_KEY'),('COMPRESSOR_frontier','pool_oldring_then_update','READY_AND_RING_WAR'),('COMPRESSOR_frontier','indexkey_and_longlatent_RoPE','BOTH_SOURCE_READERS'),('pool_oldring_then_update','indexkey_and_longlatent_RoPE','LATENT_NORM_SOURCE'),('indexkey_and_longlatent_RoPE','all_last_readers_release','BANK_LIFETIME')])
    write(O/files['graphs'],dict(generated_utc=stamp,C2_source_and_conditional_builder=gg,graphs_acyclic=True,note='Data/protocol graph,notexecutionbinarybinding ormeasuredcriticalpath. F18/F19/F34 scheduling/pruningconditions retained.',implementation=False,whole_module_closed=False,overall_complete=False))
    evidence=list(files.values());common=dict(contract=str(O/files['contract']),proof=str(O/files['proof']),fill=str(O/files['fill']),conditions=str(O/files['variants']),coverage=str(O/files['coverage']),implementation=False,new_performance_result=False)
    candidate_updates={'F31':dict(review_v39=dict(common,variants=[v for v in vv if v['parent']=='F31']),variants_v39=[v for v in vv if v['parent']=='F31'],
        profile_review_state='C2_OUTPUTS_EXISTING_FUSED_BUILDER_AND_FILL_EFFECTS_REVIEWED;CONDITIONAL_UNPRICED',
        conclusion_changing_gap='108partialchainfieldreconciliation/QLIeager-capturedreaders/M22-M25 remain. Zero unassigned selectorchainsnotwholemodulecompletion.',
        next_profile_or_contract_action='Profilepartialfieldcensusfirst;pointreadcompressedslot201/202 andDSAbuild1008-1042 onlywherepriorcontractsinsufficient;reuse905/915/RoPE/proposerproofs;binddummyCPsharedmissscope.',all_consumers_closed=False),
        'F18':dict(review_v39=dict(common,relation='Metadatafusionretainsexactring5rows andfrontier;pool+norm/oldringWAR/nativeprecision remainv24conditional,noteliminatedbymetadata skip.')),
        'F19':dict(review_v39=dict(common,relation='C2sourcecos/sinfeedbothindexkeyupdate andinplacelonglatentRoPE;lazylookup/newABIkeepsbothreadersandpublicationorder.')),
        'F34':dict(review_v39=dict(common,relation='used/live/complete/skip exactmetadataformulasnowbound;actualdevicevalues/pageowner generationremainunobserved. No automaticringcapacitysavingor7/16measuredendpointclaim.'))}
    framework_updates={'H06':dict(review_v39=dict(common,variants=[v for v in vv if v['parent']=='H06'],relation='Perbankinactiveextent/dirtyrange/duplicatePAD rulesconditional;fixedgraphbankreadersretained.')),
        'H02':dict(review_v39=dict(common,relation='Init/metadataalreadysharedkey/devicepublication;C2fusiondoesnotprovehostqueue/EngineCorecost. Statefuloutputbanksrequireownlast-writergeneration.'))}
    notes={
        'M03':'QSL-1tail/group-slot-1/group-table0 sourceeffectsbound;initializationdirtytrackingneedseachbankwriterandcaptureextent,notactiveBalone.',
        'M08':'C2used/live/skip/ring0/start/pagefieldsnowexact;v24pool/ring/normandv25aliasownershipretained. Skipnotentirecallbackdead.',
        'M10':'Latermetadata slotPADmayduplicatev38fusedtailonlysamebank/exactCPextent/value/generation/nointerveningwrite;blocktablezero separate.',
        'M21':'Remaining39unassignedchains4280TASK localcontractclassified;C2existingfusedbuilder canbereusedconditionally.108partialfieldchains/QLIreadersremainfinitefrontier.',
        'M22':'COMPRESSOR publishalreadydedupskey andchoosescapturedsync/deviceTask. Fixedgroupbank/frontier/reuseprotocol retainedforfused/lazy/skipplans.',
        'M24':'Planinputkeynotoutputbankgeneration:interveningnon-skipwriter invalidatescachedskipbank contents;request/descriptor/page epochs retained.'}
    module_updates={k:dict(finding=v,evidence=evidence,candidates=['F31']+(['F18','F19','F34']if k=='M08'else[]),gap='PartialCP/coordinate/proposerfields/QLIreaders andM22-M25 stillgenuinelyunreviewed;newcandidateimplementationnotrequiredforconditionalanalysis.')for k,v in notes.items()}
    remaining=[s for s in deepcopy(read(O/'module-coverage-gaps-v38.json')['remaining'])if not s.startswith('NextM21truefrontier:')]
    remaining.insert(19,'M21v39answersall39previousunassignedC2/fillchains4280TASK. C2formula/ring5rows/skip effects/twoFP32tables/existingfusedbuilder/directv24sameSHApool/index/longreaders andpublishdedup bound;7variants/'+str(len(reqs))+'conditions. InitQSL-1/slot-1/table0 distinct,duplicatePADsubset/dirtyrangeproofconditional. Noall3256/936TASKgain orallskipdeadclaim. Native186chainselector0unassignedbut108partialfieldchains remain;notwholemoduleclosed.')
    remaining.insert(20,'NextM21finitefieldaudit:108partialchains (101CP ancestry+7proposer). Needcompressed_slot_mapping201/202 andDSA1008-1042slot/page/cmp-residual clauses;reconcile905/915,Rope173-186 andproposer242-318 withexistingv13/v34 beforeanynewread. CP140/141/1422792calls contexts:2288targetprepare+416dummy+88draft;extenddeadproducer onlyafterownsharedbatch/cachemiss proof. QLIemptyqueryeager/capturedreaders stillunreviewed. ThenM22-M25/globalfinitecompletionaudit. Conditionalgainunvalidatednotananalysisgap.')
    summary=dict(filename='M21-C2-metadata-and-initialization-summary-v39.json',previous_turn_classification='PROGRESS:v38slot/rejection/state',reused_native_chains=186,
        newly_analyzed_scoped_chains=39,newly_analyzed_scoped_TASKs=4280,C2_TASKs=3256,C2_FP32_gathers=208,block_slot_fill_TASKs=936,QSL_fill_TASKs=88,
        remaining_unassigned_chains=0,remaining_partial_chains=108,conditional_variants=len(vv),condition_rows=len(reqs),refined_model_candidates=['F31'],related_model_candidates=['F18','F19','F34'],refined_framework_candidates=['H06'],related_framework_candidates=['H02'],new_model_candidates=[],
        remote_jobs=3,remote_pass=3,remote_retained_errors=0,remote_job_status='3remoteCPUpointreads terminal/pass;existingv24sameSHAconsumer andv37nativeprofile reused. Mainsemantic/mathreviewpass. No model/torch/NPU/service/new-health work.',evidence_refs=refs,no_new_model_NPU_service_or_health_work=True)
    report=f'''本轮按 profile 补齐 M21 的 C2 metadata 和初始化写入：39 条原未归属链、4280 TASK，其中 C2 3256、block/slot fill936、QSL fill88。208 次 FP32 gather 来自两处 C2 RoPE 表查询。各主语句执行数104（88OTHER+16dummy）与32个skip零化TASK按31×104+32对齐；这是计数核对，不是请求/回调配对。source copy_ 的效果不因缺独立 ComputeInfo 而被忽略。

C2 metadata 的五行公式已确认：ring0=max(seq_len-query_len,0)，ring1=按实际请求/实际 token/skip 截断的used，ring2/3均start，ring4为used>0时首个block。完成行由有效 token 边界与 position 奇数决定，源位置为position−1，否则0。两个 FP32 表按源位置查询。现有 packed 分支已经有融合 builder；其直接 kernel 读普通QSL/seq/position/firstblock/FP32表、写同一五行和mask/source/rope ABI，没有读取 packed cache 对象。因此可条件扩展该既有机制到当前逐算子路径，无需重新发明metadata producer。保留 INT32/INT64/FP32、索引/整数值域、负位置 remainder、cos/sin 共享 stride、bank别名与frontier；没有编译/运行新kernel或声称native验证。

skip_ring_update 不等于整个回调可删：used/complete为0，但ring0和start仍动态，source位置0仍发布实际表第0行的cos/sin，不能默认1/0。复用skip plan还必须证明输出bank没有被中间非skip写入覆盖，单有相同输入key不足。_publish_task本就按c2:compressor去重，并在device metadata与captured COMPRESSOR分支选择异步Task或同步执行；不把已有去重算新增收益。

直接消费者复用v24同SHA源码：pool读ring metadata/旧FP32ring，indexer.update_keys和long latent RoPE均读sourcecos/sin，之后写各自cache。任何lazy lookup、坐标压缩或producer pipeline都需保留这些读者、旧读先于ring更新的WAR、固定bank和首读ready/末读reuse。

初始化效果也已分清：QSL容量尾填−1、slot padded区间填−1、block table inactive行填0；encoder分支另分配零tensor。PAD_SLOT_ID已核为−1。v38 fused slot kernel会先写−1 tail，后续slot fill可在同bank、同代际、完整区间覆盖且无中间写入时视为重复发布；这不证明所有936 fill TASK可删。Dirty-range可研究只重置新变脏区间，但必须先有正确sentinel初始化，跟踪全部writer/alias、batch缩小/增长和固定capture完整读者。比如active end从4缩为2，至少仍需清[2,4)，不能只因bucket不变省全部写。

登记{len(vv)}项条件细化、{len(reqs)}项约束，更新F31/H06并关联F18/F19/F34/H02。选中的186条native链现无未归属项，仍有108条部分字段链需要有限审计，且QLI eager/captured读者、M22-M25及全模块/分支/跨层跨卡跨迭代审计尚未完成。0未归属只表示本selector导航完整，整体不完成。未实现/未测速仅是候选证据级别，不单独算未分析缺口。

3次远端CPU点读均pass，复用旧consumer/profile，加主审4runs全部pass。35模型候选、7框架候选、0整模块关闭。26理论比较INCOMPLETE、3136candidate资源行、849缺PMU、16phase-unbound图保持。无model/torch导入、新NPU/服务/健康工作。Goal active。
'''
    jobs=[dict(kind='review',status='pass',description=desc,artifacts=[N/script,N/(label+'.json')])for(label,script),desc in zip(specs,[
        'ProfilepositiveC2callback andexactQSL/slot/tablefill producer/directcallguards.',
        'NamedC2banks/readers/geometry/import andfillactivegroup/PADsource localization.',
        'Exactexistingfusedwrapper/kernel/publicationhelper/aliases/PADliteral;noSDKscan.'])]
    jobs.append(dict(kind='review',status='pass',description='Mainmathreview:legacy/fusedformulacomparison,skip/outputgeneration andconditionalPADsubset/dirtyrangeproof;39chain4280TASK coverage,108partialremain.',artifacts=[O/s for s in evidence]+[N/'C2-source166-v24.json',N/'M21-native166-v37.json']))
    action=('继续Loop004，v39实质进展。先读M21-C2-metadata-and-initialization-summary-v39.json、M21-C2-metadata-fill-and-existing-fusion-contract-v39.json、F31-C2-existing-fused-builder-equivalence-v39.json、H06-initialization-write-ownership-review-v39.json、M21-profile-chain-coverage-frontier-v39.json、module-coverage-gaps-v39.json。'
        '39原unassigned4280TASK已局部合同分析：C2metadata3256/208FP32gather,blockslotfill936,QSLfill88；104主语句域88OTHER+16dummy,31*104+32仅aggregate不是pairing。C2existingpackedfusedbuilder代码已读，5ringrows/mask/sourcepos/F32gathers与legacy公式同域，不读packedcacheobject；可条件扩展，不是新missingproducer/已跑kernel。保I32QSLseqring/I64pos/F32tableout，cos/sin共享source/outputstrides条件/validbounds/remainder非负域，银行alias/capture/frontier。'
        'Ring0=max(seq-q,0),used=max(min(end,A)-start,0)live&&!skip，rows2/3start，row4used?firstblock:0；complete=(i<min(QSL[actualR],A))&&pos.remainder2==1&&!skip,sourcepos=complete?pos-1:0。skip不是allzero/deadcallback；动态ring0/start保留，实际table0cos/sin非默认1/0。输入key命中不等于输出bank未被中间writer覆盖，skipcache要bankgen/invalidate/immutableplan。'
        '_publish_task已有sharedkeydedup/deviceTask或capturedCOMPRESSOR同步；C2v24同DSASHAconsumer已复用pool/oldringWAR/indexkey+longlatentRoPE，不重复读。Lazylookup/compactcoords/positionpipeline需allreaders/newABI。'
        'PAD literal-1；QSL[R+1:]-1/groupSlot[N:Npad]-1/groupTable[R:Rpad]0不同银行，encoder另zeros。早fusedslotTail仅samephysicalgroupbank/sourceN<=metaN<=metaPad<=fusedMax/无中间写/同gen才覆盖laterfill；不称936全删。Dirtytail有初始sentinel+allprefixwriters才shrink时clear[a,a_prev),growth可reuse；捕获fullreader/alias/reusefence保持。'
        '7variants/'+str(len(reqs))+'conditions细化F31/H06关联F18/F19/F34/H02；3remote全pass+main=4runs4pass。35model7framework0module，26theoryINCOMPLETE/3136资源/849missingPMU/16unboundgraphs，无model/torch/NPU/service/health,lastv12post。'
        '下一步M21有限字段审计108partialchains(101CP+7proposer)：compressed_slot_mapping201/202、DSAbuild1008-1042slot/page/cmp/residual；先复用v13 batch905/915,v34RoPE173-186/DSparkfirstpass242-318再补真缺源码。CP140/141/142共2792=2288prepare+416dummy+88draft，targetv13later25deadproof不能未经sharedmiss范围证明套dummy/draft。QLIemptyquery eager/captured读者待查；随后M22-M25/globalfiniteaudit。186selector0unassigned不等wholeM21；未实现测速不算单独未分析，Goalactive因真缺口。')
    persist_refinement(version=39,previous=38,title='v39 M21 C2 existing fusion and initialization ownership',summary=summary,
        changes=['C2legacycoordinate/skip/RoPEoutputs andpreexistingfusedbuilder source-domain equivalence bound withoutnewNPUwork.',
            'PerbankQSL/slot/tableinitialization ownership andconditionalidempotent/dirtyrange proofs recorded;skip/outputbankgeneration hazards retained.',
            '39previousunassignedchains4280TASK analyzed;7conditionalrefinements/26conditions,108partialfieldchains/QLI/M22-M25 remain.'],
        candidate_updates=candidate_updates,framework_updates=framework_updates,module_updates=module_updates,remaining=remaining,action=action,jobs=jobs,report=report,
        read_first=[files['contract'],files['proof'],files['fill'],files['coverage']],
        publication_rows={'F31':'| F31 | Metadata、slot 与跨迭代输入输出发布重组 | 条件 | v39补齐C2五行ring/mask/source/RoPE合同，现有packed融合builder可在dtype/stride/bank域下条件扩展legacy路径；skip仍有动态字段与table0发布。QSL/slot/table初始化写入分账，PAD重复/dirty-range需完整owner和代际。186链selector零未归属，108部分字段链及QLI读者等仍待审计。 |'},
        gate_updates=dict(status='PROFILE_C2_AND_INITIALIZATION_LOCAL_CONTRACTS_REVIEWED',answered_by=evidence,
            closed_scope='C2coordinate/skip/fulltableoutputs/existingfusedmechanism/directconsumers/publication andperbankfilllocalcontracts.',not_closed='108partialfieldchainreconciliation/QLIeager-capturedreaders/fullM22-M25/globalmoduleaudit.',
            stop_rule='Do notrereadknownC2formulas orimplementkernel merelyforconditionalanalysis;advance108partialfieldandQLIreader questions.'),
        scripts=[N/s for _,s in specs],executor_prefix='zcode-M21-',main_script=N/'finalize_M21_C2_metadata_v39.py')


if __name__=='__main__':
    main()
