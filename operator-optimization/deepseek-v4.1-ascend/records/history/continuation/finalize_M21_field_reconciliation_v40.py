from copy import deepcopy
from collections import Counter
from datetime import datetime,timezone
import hashlib
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,W,read,write
from finalize_Engram_whole_frontier_v30 import packet,graph
from persist_profile_refinement_v2 import persist_refinement


def main():
    assert not (O/'REVIEW-CHECKPOINT-v40.json').exists()
    stamp=datetime.now(timezone.utc).isoformat()
    bound=packet('M21-bind166-v40-v2',True)
    native=packet('M21-native166-v37',False)
    rope=packet('M18-source166-v34',True)
    old_cp=read(O/'H02-CP-reuse-contract-v13.json')
    input_files=['M21-selected-coordinate-and-publisher-source-v40.json','M21-slot-builder-and-shared-length-clauses-v40.json','M21-QLI-plan-direct-consumer-source-v40.json','M21-existing-packed-slot-backend-localization-v40.json','M21-coordinate-cache-and-QLI-branch-source-v40.json','M21-worker-shared-batch-scope-source-v40.json','M21-CP-target-dummy-and-draft-source-scope-v40.json','M21-CP-global-and-local-batch-cache-contract-source-v40.json']
    frozen=[];source_pins={}
    for name in input_files:
        data=read(O/name)
        for f in data.get('files',[data]):
            if 'path' not in f:continue
            p=Path(f['path']);sha=hashlib.sha256(p.read_bytes()).hexdigest();assert sha==f['sha256'],p
            frozen.append(p);source_pins[str(p)]=sha
    # Consumer functions and DeviceOperator are exactly the166staged source,
    # not merely a similar local checkout.
    for f in bound['result']['files']:
        matching=[p for p in frozen if str(p).endswith(f['path'].split('/vllm_ascend/')[-1])]
        for p in matching:assert source_pins[str(p)]==f['sha256']
    assert read(O/'M21-local-helper-preparation-error-v40.json')['executor_returncode']==1
    old=read(O/'M21-profile-chain-coverage-frontier-v39.json')
    partial=[c for c in old['selected_chains']if c['state'].startswith('REUSED_PARTIAL')]
    assert len(partial)==108 and sum(c['TASKs']for c in partial)==66685
    def category(c):
        f=c['leaf_project_frames'][0]
        if ': compressed_slot_mapping'in f:return'compressed_physical_slot'
        if '/dsa_v41.py('in f and f.endswith(': build'):return'slot_page_and_c2length'
        if '_build_batch_metadata'in f:return'batch_authoritative_snapshot'
        if '/dsa_v41_cp.py('in f:return'CP_suffix_transform'
        if 'get_cos_and_sin_dsa'in f:return'RoPE_lookup'
        if 'set_inputs_first_pass'in f:return'DSpark_firstpass_publisher'
        raise AssertionError(f)
    cats={}
    for c in partial:
        z=cats.setdefault(category(c),dict(chains=0,TASKs=0));z['chains']+=1;z['TASKs']+=c['TASKs']
    assert cats=={'compressed_physical_slot':{'chains':12,'TASKs':2276},'RoPE_lookup':{'chains':10,'TASKs':2112},'slot_page_and_c2length':{'chains':64,'TASKs':47636},'DSpark_firstpass_publisher':{'chains':7,'TASKs':480},'CP_suffix_transform':{'chains':9,'TASKs':13960},'batch_authoritative_snapshot':{'chains':6,'TASKs':221}}
    refs=[dict(receipt=str(N/(k+'.json')),remote=p['full_result'],sha256=p['result_sha256'])for k,p in [('M21-bind166-v40-v2',bound),('M21-native166-v37',native),('M18-source166-v34',rope)]]
    names=dict(profile='M21-partial-field-profile-reconciliation-v40.json',contract='M21-slot-coordinate-draft-cache-and-QLI-consumer-contract-v40.json',
        proof='F31-slot-fusion-domain-and-draft-cache-gap-proof-v40.json',qli='F31-QLI-empty-query-and-capture-reader-review-v40.json',
        cp='H02-CP-target-dummy-draft-publication-review-v40.json',variants='M21-field-reconciliation-variants-v40.json',
        coverage='M21-profile-chain-coverage-frontier-v40.json',graphs='M21-field-reconciliation-graphs-v40.json')
    write(O/names['profile'],dict(generated_utc=stamp,reused_native_scope=dict(APIs=60404,TASKs=76033,chains=186),reconciled_categories=cats,reconciled_chains=108,reconciled_TASKs=66685,evidence_refs=refs,
        notes=['TheseTASKs werealreadyinthe186-chainselector,notnewwork. Sourcecalls/typedcounts arefieldnavigation,notphase/request/pointer/binaryidentity.',
            'CPsuffix9chains13960TASK comprise2792eachSub/Clip/Greater/Fill/Select. Scope2792=2288prepare+416dummy+88draft fromv37hostreconciliation;notallredundant.',
            'RoPE10chains2112TASK reusev34sameSHApurevalue-selection/readinesscontract. use_cache=True stillgathersinto runtime/specscratchbanks;itdoesnotmeancontentcachehit.',
            'DSparkfirstpass480TASKincludes24positiveTensorMoves,notabsenceofsmallcopywork. Existingfusedcopy-expand interface/rowbank obligationsfromv34retained.'],whole_module_closed=False,overall_complete=False))
    facts=[
        ('DOCUMENTED','compressed_slot_mapping(s,r)=where(s>=0 and(s+1)%r==0,s//r,-1). It checks physical-slot completion,not onlylogicalposition parity. Legacycoordinate validity startsatconvertedslot>=0;C2 additionallymasksactualtokenend,skipandpositionparitywhenpositionspresent.'),
        ('DOCUMENTED','Legacycoordinates are(page=floor(max(slot,0)/storageB),offset=max(slot,0)%storageB)orminus1,plusflatvalidslot. Existingpackedhelperwritesallthreeoutputsbutalsoappliesvalid_endtouncompressedratio1rows;itrequirespositionstensor/ratio1or2/contiguous[N,2]andflatoutputs.'),
        ('DERIVED','Thuspackedhelperisnotanunconditionallegacydrop-in. Reuseonlywhere itsstrongermaskisredundantoradaptmask/positions/ratioABItoexactlegacybranches. Preservephysicalcompletionandlogicalparityseparately;state-cachePAD/Nonebranchdoesnotusetheslotbuilder.'),
        ('DOCUMENTED','PerframeworkKVgroup sharedkeys slot:coordinates:c{ratio}:b{storageB} andslot:flat... alreadysharelong/index outputbanks;SWAgroupbanksremainseparate. C2lengthsquotient/remainderalreadycachedonceunderbatch_shared[lengths:c2]. Donotclaimtheseexistingcachesasnewoptimization.'),
        ('DOCUMENTED','_build_batch_metadatacopiesauthoritativeGPUlengths,zerosinactiveactual-requestsuffix,derivesCPUmaxfromavailablecanonicalmirror andcastspositionslong. Batchcachehitleaveslatercommon.seq_lens temporarycontentunread;shape/accessorcontractreusev13.'),
        ('DOCUMENTED','Workercreatesonecommon_v41_batch_metadata per_build_attention_metadata andonecommon_v41_metadata perKVgroup;cm_baseisshallowcopied. PositiveCPtarget/dummyfullcallersusethissameworkerpath. CPglobaluseschildcp_global dictionaries;localusesrootbatch.'),
        ('DERIVED','v13cache-hitdeadlocal-lengthproducerargumentextendstoworker/dummycachehitsunderthissamesourceflow. Firstlocal/globalcachemisses,QSLfixed-addresspublication,pergroupfieldsandneededdeviceauthoritativevaluesstaylive. Noexact400dummy-removalcountorcriticalpathgainisclaimed.'),
        ('DOCUMENTED','Draftcallerbuild_draft_attn_metadata constructs shared_dsa_draft_cache withcommon_ratio_to_sas_metadata andpasseskwargs tobuild_for_drafting. DSAwrapper839-842dropsallkwargsandcallsbuild(0,common). Thereisnocaller-providedcommon_v41_batch_metadatadictionaryeither;forwardingcurrentkwargsaloneisnotasharedbatchsolution.'),
        ('DOCUMENTED','Draftgroupsreplacephysicaltable/slotandmayapplyslidingwindowtomodifylengths. Anewconsumedsharedbatch/fieldinterfacecanreuseonlycompleteinput-equivalentfields;pergroupphysicalcoordinates,windowedlengths/operatorbanksmustnotbemergedbyB orcommonrequestIDs.'),
        ('DOCUMENTED','DSparkfirstpasspublishesnextseedandzeroestail,copiescontextprefix,runsalreadyfusedcopy-expandpergroup,thenupdatescadGPUQSL=arange*Q,seq=old_seq-rejected+Qwhenrejectcountpresent. CanonicalCPUmirroriscloned/extendedseparately;do notsubstituteoptimisticCPUlengthsforGPUvaluesoroverwriteseqbeforekernelread.'),
        ('DOCUMENTED','RoPEvalueplanisdeterministicselectionbyposition/config/layergroup,butuse_cacheTruewritesreusedruntime/specbuffers. CPglobal/localRoPEslicereusealreadyexists;latergathercanoverwriteviewsstillreadbyearliermetadata. Preserveepoch/lastreaders,notpointer-onlyCSE.'),
        ('DOCUMENTED','Indexer.select_projectedreturnsbeforemetadatareadforquery.shape[0]==0 ormax_cache_seq_len==0. OtherwiselegacyreadswaitsINDEXER id(qli_metadata)thenpassesblobtoQLIv3. Packed_commonalsoreads/waitsblobandhasformat-specificNoneerror;packedformatnotinterchangeablewithlegacy.'),
        ('DOCUMENTED','PackedQLImetadataproduceronlyzerosnum_actual_tokens0whennotmetadata_in_graph. LegacyproduceralwayscallsV2metadata. Logicalactual0withcapturednonemptyphysicalquerydoesnottriggerconsumerphysicalemptyguard;capturedgraphcontainingQLIstillneedsvalidplan.'),
        ('UNKNOWN','Actualpointers/completevalues/windowequivalence,allobserver/nativeformat/graph-replaybankbindingsandcriticalpathgainremainconditions. Thelocalcoordinate/CP/draft/QLIreaderquestionsarenowanswered;wholeM21phase/branchauditandM22-M25arestillunreviewed.')]
    write(O/names['contract'],dict(generated_utc=stamp,source_pins=source_pins,server_binding=refs[0],local_point_source_artifacts=input_files,facts=[dict(grade=g,statement=s)for g,s in facts],
        reuse=['H02-CP-reuse-contract-v13.json','M18-partition-bank-RoPE-and-slot-contract-v34.json','M21-slot-rejection-state-and-packing-contract-v38.json','M21-C2-metadata-fill-and-existing-fusion-contract-v39.json'],
        local_questions_closed=['Allsixpartialfieldcategories','Existingpackedslotmechanismandmaskmismatch','Actualdraftkwargs/cachehandoffgap','WorkerCPdummyversusdraftcacheownership','QLIphysicalempty/zero-cacheandcapturedlogicalemptyreaderboundaries'],implementation=False,whole_module_closed=False,overall_complete=False))
    write(O/names['proof'],dict(generated_utc=stamp,evidence_grade='DERIVED_WITH_EXPLICIT_COUNTEREXAMPLES',
        ratio1_tail_counterexample=dict(slots=[3,5],valid_end=1,num_tokens=2,page_size=4,legacy_flat=[3,5],packed_helper_flat=[3,-1],
            note='Source-leveldistinctiononthebroaderlegacyvalid-domain;notclaimthesevaluesoccurredorcurrentlegacyhasabug. Strongermaskmustbeprovedredundantoradapted.'),
        physical_parity_counterexample=dict(original_slot=2,logical_position=1,ratio=2,legacy_compressed_slot=-1,
            note='Logicaloddpositionalonecannotreplacephysical(s+1)%2completionwithoutanindependentalignmentproof.'),
        exact_reuse_conditions=['Uncompressedlegacyvalidityonlyslot>=0;existinghelperendmaskrequiresinactivepositive-slotexclusionorMASK_PADDED_TAILinterfaceadaptation.',
            'Ratio2preservesbothphysicalcompletionandlogicalparity/liveend/skip. LegacypositionsNonebranchneedsHAS_POSITIONorfallback;existingwrappercallspositions.stride().',
            'Legacygenericratiosbeyond1/2notacceptedbyexistingwrapper;retainfallbackornewexplicitgenericdomain.',
            'Correctstoragesize/INT32outputcasts/integeroverflow/floordivision/remainder,contiguouscoordinateandflatoutstrides. Alloriginalslotbanks/firstready/lastreadersretained.'],
        draft_handoff_gap=['Calleronlysharescommon_ratio_to_sas_metadata;notcommon_v41_batch_metadata.',
            'build_for_draftingdiscardskwargs,soforwardingexistingdictaloneisnotenough.',
            'Needsnewexplicitconsumedbatch/fieldinterfaceandgroup-localphysicalcachekeys. SplitcommonQSL/positionfieldsfromgroup-windowedlengthswherecompletebatchsignaturesdiffer.',
            'CanonicalCPUmirror/GPUauthoritativeseqepoch,CPglobal/localpartition,slidingwindow,operatorbanklifetimeandmetadataexecutortaskskept.'],
        implementation=False,native_validation=False,critical_path_gain_us=None,whole_module_closed=False,overall_complete=False))
    write(O/names['qli'],dict(generated_utc=stamp,reader='Indexer.select_projected181-298',
        early_return_domains=['Physicalquery.shape[0]==0','Host/source_metadata.max_cache_seq_len==0 withitsoriginaloutputinitializationeffects'],
        positive_reader='Otherwiselegacyselectreadsqli_metadata,waitsINDEXERid(blob),executesQLIv3;packedrunpathusesformat-specific_commonread/wait.',
        metadata_producer_branches=dict(packed='num_actual_tokens0&&!metadata_in_graph zero-bankguard;otherwisebackendmetadata',legacy='V2metadataunconditional;fixedbankcopy'),
        conditional_lazy_plan='Onlyifallactualreadersreturnbeforeblobread/waitandcapturedgraphbodyhasnometadata-consumingnode,canskipvalueproductionwhilepreservingstablefrontier/event/bank/observerprotocol. Do notborrowopaquezeroformatfromSMLA/packedQLI.',
        capture_limit='Logicalactual0isnotphysicalquery0. StaticcapturecanexecutenonemptyQLIwithupdateddevicecounts;bindgraphbody/bucket/epochinM22beforeanyper-replayskip. Sourceconsumerpredicateisclosed;actualgraphbindingisnot.',
        local_reader_question_closed=True,actual_graph_body_audited=False,implementation=False,whole_module_closed=False,overall_complete=False))
    write(O/names['cp'],dict(generated_utc=stamp,framework_candidate='H02',source_scope=dict(worker_shared_batch='Newrootdictper_build_attention_metadata;pergroupdictseparate;targetanddummypositivefullcallersmatch.',global_child='cp_globalchilddictswhenpassed.',draft='kwargsdroppedandnocommon_v41_batch_metadatacallerfield;freshbatchmissescurrentpath.'),
        current_profile=dict(CP_suffix_operations_per_type=2792,prepare=2288,dummy=416,draft=88,target_v13_later_suffix_measured=2200),
        proof='Afterlocalbatchcachedcoordinatesexist,latercomputedlocal_seq_lencontentisnotreadexceptshapeintheinspectedbasepath. Sameworkerflowmakesthislifetimeargumentapplicabletodummycachehitswithoutassuminginputequality. Retainfirstmiss/QSL/pergroupfields. Currentdraftfreshmissresultsarestilllive.',
        no_new_dummy_dead_count_attested=True,critical_path_gain_us=None,implementation=False,whole_module_closed=False,overall_complete=False))
    variants=[
        ('F31-v40-ADAPT-SLOT-COORDINATE-FUSION','F31','Reuse/adaptexistingthree-outputslotbuilderfortheobservedlegacyslot-coordinatechain.','Matchratio1tailmask,ratio2physical/logicalcompletion,positionsNone,genericratio/outputstride/dtype domains;existinghelperisnotunconditionaldrop-in.'),
        ('F31-v40-SLOT-PRODUCER-OUTPUTS','F31','Emitcompatibleflatandpage/rowcoordinatesalongsidegroupphysical-slotmapping.','Keeporiginalslotarrayandgroup-specificstorage/ratio/mask/positiongenerations,allphysicalcacheowners andcapturebank ABI;no crossgroup table reusebyshape.'),
        ('F31-v40-C2-QUOTIENT-REMAINDER','F31','ComputesharedC2quotient/remainderinonecompatibleper-requestproducer.','Existingbatchlengths:c2cachealreadyavoidsrepeatedgroupwork;preservefloor/remainder,INT32layouts/correctedseqgeneration andallmetadatareaders.'),
        ('F31-v40-DRAFT-CACHE-HANDOFF','F31','Addexplicitconsumedgroup-safecommonfield/batchcachehandofffordraftmetadata.','Callerneedsnewbatch/fieldcacheandwrapperkwargsforwarding;currentforwardingaloneinsufficient. Per-groupwindowedseq/physicalslots/operatorbanksremainseparateunlessfullsignatureequivalent.'),
        ('F31-v40-DRAFT-UNIFORM-GEOMETRY','F31','ReuseimmutableQSL/hostgeometrytemplatesorcomposeexistingseq-reject-plusQpublication.','KeyexactB/Q/dtype/order/CPpolicy;oldinputseqmustremainreadablebyalreadylaunchedcopy-expandkernel. PreserveCPUcanonicalmirrorandfullGPUlengtharithmetic/readers.'),
        ('F31-v40-QLI-LAZY-PLAN','F31','Avoidplanvalueproductiononlyforcompleteconsumer-bypassdomains.','Physicalempty/zero-cacheguardsandactualgraphbody/format/allreaderproofneeded;logicalactual0aloneinsufficient. Preservefixedfrontier/events/capturebankgeneration.'),
        ('H02-v40-CP-DUMMY-CACHE-HIT','H02','ExtendexistingCPcache-hitdeadproducerplacementtothesamesharedworker/dummyflow.','Retainfirstlocal/globalmiss,QSLfixedaddress/groupfields. Do notapplytocurrentdraftfreshmissesorclaim400measured removalswithoutbindings.'),
        ('H06-v40-SEED-TAIL-DIRTY-RANGE','H06','Specializeexistinginactive-rangeanalysisforDSparkseedzero-tailpublication.','Keepinitialseed/tailvalues,activeprefix/invalidrows,batchshrink/grow,allMarkov/capturelastreaders andbankgeneration;refinesv39dirtyrange,notadditivegain.')]
    vv=[dict(id=i,parent=p,transformation=t,conditions=c,implemented=False,native_validation=False,critical_path_gain_us=None)for i,p,t,c in variants]
    conditions=[
        ('SCOPE','108chains66685TASK isreconciliationofexistingselectorwork,notnewload/functionalPMUbucket/wholemodulecompletion.'),
        ('LEGACYMASK','Forratio1legacyonlyslotvalidity;existingpackedhelperextravalid_endmaskmustberedundantoradapted.'),
        ('PHYSICAL','Preserveoriginalphysicalslot>=0and(s+1)%ratio completion independentlyoflogicalpositionparity.'),
        ('POSITIONS','Legacyratio2positionsNonehasnoparitycheck;existinghelperrequirespositionsTensor. ExplicitHAS_POSITIONorfallbackneeded.'),
        ('GENERICRATIO','Existinghelperonlyratio1/2;genericlegacynon1ratiosretainseparatedomain.'),
        ('STATE','Compressor-statePADslot/Noneflatbranchdoesnotusecoordinatebuilder;ringownershipnotmerged.'),
        ('ARITH','INT32/I64narrowing,floor/remainder/overflow/storageblocksizeanderror/boundsdomainsretained.'),
        ('OUTSTRIDE','ExistinghelperstorescontiguousN2andflatstrides;otherlayoutsneedexplicitstrides/newABI.'),
        ('SLOTOWNER','Alloriginalgroupmapping/tableowners,active/tailoutputs andfixedcapturebanksretained;shapeequalitynotphysicalidentity.'),
        ('CACHED','Perslotkeylong/indexsharingandbatchlengths:c2cachealreadyexist;notnewCSEsaving.'),
        ('SEQGEN','C2div/remainderuseauthoritativeGPUlengthssameepoch;CPUmax/upperboundisnotreplacement.'),
        ('BATCHMISS','Preservefirstbatchcopy/zeroactualsuffix/longpositioncast/canonicalCPUmax/requestcountsemantics.'),
        ('WORKERSCOPE','Target/dummyshareworkerrootbatchandpergroupdict;cp_globalchildretained. Deadproducerargumentrequiresactualcachehit,no blanketCPremoval.'),
        ('DRAFTGAP','Callerdoesnotcurrentlyprovidesharedbatchcacheandwrapperdropskwargs;bothinterfacesneeded,notforwardunuseddictalone.'),
        ('GROUPWINDOW','Draftgroupsmaywindowseq/tableindependently;reuseonlyfullsignature-equivalentfieldsorsplitcommonfromgroupdependentvalues.'),
        ('CALLERABI','Forwardonlycompatiblekwargsandretainspec/blocksize/candidate/frontiersemantics;no group-localphysicalcachemixing.'),
        ('SEED','Initialnextseed,zerotail,B/Q/invalidrows andMarkovinputconsumerstaysame;dirtyrangeallwriter/readergenerationproofrequired.'),
        ('SEQPUBLISH','DraftGPUlength=oldseq-rejected+Q;copy-expandmustreaditsoldinputsfirst. Hostcanonicalmirrorclone/extenddistinct.'),
        ('TEMPLATE','QSL/hosttemplatekeyfullB/Q/dtype/order/CPpolicy,withimmutableasync/capturelifetime;notassumeQconstantacrossbudget/configchanges.'),
        ('ROPE','use_cacheTruerefreshesmutablebank;keepfullconfig/layer/positiongenerationandlastreaders. ExistingCPsliceandv34filteralreadyimplemented.'),
        ('QLIGUARD','Onlyphysicalemptyqueryorzeromaxcacheconsumerbranchbeforeblobread. Logicalactual0doesnotprovephysicalquery0.'),
        ('QLIGRAPH','ActualcapturebodycancontainanonemptyQLInodeevenwhenlogicalactual0;needsM22body/bucket/epochproofbeforeplan-skipping.'),
        ('QLIFORMAT','Packed/legacy/SMLAopaqueformatsnotinterchangeable;keepformat-specificNone/error/zeroeffects.'),
        ('QLIREADERS','Allplanobservers/readersmustbypass;keepINDEXERfrontierstable,event/reset/release andbanklastwriter-generation.'),
        ('PUBLICATION','DataCSE/fusiondoesnotremoveinput-ready/reusefences/dedupkeys/capturedparameterupdates.'),
        ('EVIDENCE','Noaliasequality/shape-basedepoch/physicaltraffic/causalgainfromcounts;3136resources/849missingPMU/16unboundgraphsremain.'),
        ('OVERLAP','Slotcoord/produceroutput/C2coord/draftcache/template/QLIplan andv13-v39variants overlap;pricejointly,notadditive.')]
    write(O/names['variants'],dict(generated_utc=stamp,variants=vv,conditions=[dict(id='M21-v40-'+i,requirement=s,actual_current_legality_proved=False)for i,s in conditions],implementation=False,whole_module_closed=False,overall_complete=False))
    cs=deepcopy(old['selected_chains'])
    for c in cs:
        if c['state'].startswith('REUSED_PARTIAL'):
            cat=category(c);c['previous_state']=c['state'];c['state']='ANALYZED_CONDITIONAL_'+cat.upper();c['evidence']=[names['contract'],names['proof'],names['qli'],names['cp']]
    assert len(cs)==186 and not any(c['state'].startswith('REUSED_PARTIAL')or c['state']=='UNANALYZED_FIELD_OR_CONSUMER'for c in cs)
    remaining_scope=[
        'M21phase/branch completionaudit mustcompare this186-chainnative/typedselector tooriginalmodulematrix,hostfieldinventory/capture/async/emptyguards andallmetadataobservers. Zero partialselectorchainsisnotwholeM21proof.',
        'M22fullgraph/replay/queue/transportsource ownership:actualcapturedQLIreaderbody,parameter/bankbinding,inputready/lastreaderbufferreuse andunbound16graphs notyetfullscopeaudited.',
        'M23-M25cross-card/cross-iteration/cache/scheduler/EngineCore/frontend scopes plusfiniteglobal26modulecompletionauditremain. Implementationandunvalidatedgainsarenottreatedasautomaticanalysisgaps.']
    write(O/names['coverage'],dict(generated_utc=stamp,selected_chains=cs,state_counts=dict(Counter(c['state']for c in cs)),reconciled_categories=cats,newly_reconciled_partial_chains=108,newly_reconciled_partial_TASKs=66685,remaining_unassigned_chains=0,remaining_partial_chains=0,
        genuine_unanalyzed_frontiers=remaining_scope,completion_policy='All186selectednativechainsnowhavealocalfield/interfacecontractorlinkedpreviousanalysis. Thisdoesnotprovetheoriginalmodule/branch/processscopeiscomplete. Unexecutedconditionalnativevalidationnotrequiredmerelyforregisteringanalysis.',whole_module_closed=False,overall_complete=False))
    g=graph(['old_corrected_group_state','group_specific_window_and_slots','common_field_signature','conditional_shared_field_cache','group_operator_banks','INDEXER_COMPRESSOR_ATTENTION_frontiers','actual_eager_or_captured_consumers','last_reader_release'],[
        ('old_corrected_group_state','group_specific_window_and_slots','PRESERVE_OLD_INPUT_EPOCH'),('group_specific_window_and_slots','common_field_signature','NO_ASSUMED_GROUP_EQUIVALENCE'),('common_field_signature','conditional_shared_field_cache','NEW_CONSUMED_DRAFT_ABI'),('group_specific_window_and_slots','group_operator_banks','KEEP_PHYSICAL_OWNERS'),('conditional_shared_field_cache','group_operator_banks','EXACT_SHARED_FIELDS_ONLY'),('group_operator_banks','INDEXER_COMPRESSOR_ATTENTION_frontiers','PUBLISH_FIXED_BANKS'),('INDEXER_COMPRESSOR_ATTENTION_frontiers','actual_eager_or_captured_consumers','ACTUAL_READER_BODY'),('actual_eager_or_captured_consumers','last_reader_release','NO_EARLY_REUSE')])
    write(O/names['graphs'],dict(generated_utc=stamp,conditional_draft_field_cache=g,graphs_acyclic=True,note='Source/interfacegraph,notproofactualgroupvaluesmatchorcapturedreadersbypass;no newnativeimplementation.',implementation=False,whole_module_closed=False,overall_complete=False))
    evidence=list(names.values())+input_files
    common=dict(contract=str(O/names['contract']),proof=str(O/names['proof']),QLI=str(O/names['qli']),CP=str(O/names['cp']),conditions=str(O/names['variants']),coverage=str(O/names['coverage']),implementation=False,new_performance_result=False)
    model_updates={'F31':dict(review_v40=dict(common,variants=[v for v in vv if v['parent']=='F31']),variants_v40=[v for v in vv if v['parent']=='F31'],profile_review_state='ALL186SELECTED_NATIVE_CHAIN_FIELD_INTERFACES_RECONCILED;MODULE_PHASE_BRANCH_AND_CAPTURE_SCOPE_AUDIT_PENDING',
        conclusion_changing_gap='M21originalphase/branchversusselectorauditandactualM22captured/transport/reusebodydomainsremain;localQLIguards/draftkwargs/slotformulaquestionsanswered.',next_profile_or_contract_action='AuditM21originalmatrixthenadvanceM22savedgraphs/queue/replaybankownership;do notrepeatalreadyclosed108fieldsourcequestions.',all_consumers_closed=False),
        'F29':dict(review_v40=dict(common,relation='Groupphysicalslot/parity/cacheownerandRoPEscratchlastreadersretained;commonfieldcachemustnotoverridewindowedlengths/CPviews.')),
        'F30':dict(review_v40=dict(common,relation='DSparkseed/zero-tail/uniformQSLandGPUlengthpublicationfeedMarkov/decodercontracts;keepcorrectQ/seedepochandoldinputreadbeforemetadatamutation.')),
        'F34':dict(review_v40=dict(common,relation='C2logicalcompleteandphysicalcompressedslotcompletionaredifferentpredicates;keepring/pageownerdomainandv25nocapacitysavingexclusion.'))}
    fw_updates={'H02':dict(review_v40=dict(common,variants=[v for v in vv if v['parent']=='H02'],relation='Worker/dummysharedcache-hitdomainbound;draftwrapperdropskwargsandcallerdoesnotprovidesharedbatch,sofreshmisssuffixlive.')),
        'H06':dict(review_v40=dict(common,variants=[v for v in vv if v['parent']=='H06'],relation='Ratio1packedtailmaskstrongerthanlegacy;seed/QSL/paddingdirtyproofneedscaptureextentandbankepochs,notshapeonly.'))}
    notes={'M03':'Batch/slot/CP/drafthostfieldscopebound;keyedtemplatesnewABIkeepauthoritativeGPUstateandallasyncowners.',
        'M09':'QLIselectphysicalempty/zerocacheearlyreturnandlegacy/packedread-waitguardslocated;logicalemptycapturednonemptyplanstilllive.',
        'M10':'Existingpackedslothelperhasstrongerratio1endmask;adapt/restrictdomainbeforelegacyreuse. Physicalcompletionandlogicalparitynotinterchangeable.',
        'M18':'Draftsharedcachecommentnotactualhandoff:callerhasnosharedbatchdictandwrapperdropskwargs. Newgroupsafecommonfieldprotocolconditional;pergroupwindowedlengths/slotskept.',
        'M20':'Seedzero-tail/copy-expand-beforecadseqmutation/uniformQSLandcanonicalCPUmirrorsemanticsbound;Markovseedanddecoderreadersretainedevents.',
        'M21':'108partialchains66685TASKreconciledinto6categories;all186selectedchainslocalfield/interfaceanalyzed. Originalphase/branch/modulecoverageauditstillneeded.',
        'M22':'QLIcapturebodyratherthanlogicalnum_actualtokensdecidesplanreader;actualgraph/parameterbank/ready/reuseownershipfullauditnext.',
        'M24':'Outputscratch/cachedcoordsandgroup-specificstatekeyneedsfullvalues/generation/lastreader;kwargsforwardingorpointerreusealoneinsufficient.'}
    module_updates={k:dict(finding=v,evidence=evidence,candidates=['F31']+(['F29']if k in['M10','M18','M24']else[])+(['F30']if k=='M20'else[]),gap='Localfieldquestionsanswered;originalM21phase/branchandM22-M25fullscopeauditsremainrealwork. Candidateimplementationnotrequiredforconditionalanalysis.')for k,v in notes.items()}
    remaining=[s for s in deepcopy(read(O/'module-coverage-gaps-v39.json')['remaining'])if not s.startswith('NextM21finitefieldaudit:')]
    remaining.insert(21,'M21v40reconciles108partialchains66685TASK into12compressedslot2276/64slotpage-C2length47636/6batch221/9CP13960/10RoPE2112/7firstpass480. Existingpackedslotbuilderratio1hasextravalid_endmask(counterexample[3,5],end1),notblinddrop-in;positionsNone/genericratio/outstridekept. DraftcallerhasonlySASsharedkwargsandDSA842dropskwargs;needsnewexplicitsharedbatch/fieldABIandgroupwindowedlengths/slotsownership. QLIphysicalempty/zerocacheguardsandcapturedlogicalemptydifferenceanswered.8variants/'+str(len(conditions))+'conditions;noallgainclaim.')
    remaining.insert(22,'NextM21scopeauditcompareoriginalmodulephase/branchmatrix/hostfieldinventoryandcapture/async/emptyobserverdomainsagainst186selectorwith0partial;do notmarkwholeM21fromselectoralone. ThenM22actualgraph/QLIbody/parameterbank/ready-reuse/queues-transport,thenM23-M25/globalfinite26moduleaudit. KnownCPdummycachehitargumentextendsworkerflow;currentdraftfreshmissstayslive. Do notrepeatalreadyboundslot/kwargs/QLIconsumerfunctionsorrequireimplementationtoqualifyconditionalanalysis.')
    summary=dict(filename='M21-field-reconciliation-summary-v40.json',previous_turn_classification='PROGRESS:v39C2initialization',reconciled_chains=108,reconciled_TASKs=66685,categories=cats,selected_native_chains=186,remaining_unassigned_chains=0,remaining_partial_chains=0,
        conditional_variants=len(vv),condition_rows=len(conditions),refined_model_candidates=['F31'],related_model_candidates=['F29','F30','F34'],refined_framework_candidates=['H02','H06'],new_model_candidates=[],
        remote_jobs=1,remote_pass=1,remote_retained_errors=0,local_retained_errors=2,
        remote_job_status='Oneexplicitv2remoteCPUread/hashjobterminalpass;localfrozenpointreadsandmainreviewpass. Initialmissinglocalhelperpreparationanddependentexecutorlocalrc1retainedseparately;firstattemptneverconnectedserver. No model/torch/NPU/service/new-health work.',evidence_refs=refs,no_new_model_NPU_service_or_health_work=True)
    report=f'''本轮先复核 v39 manifest，再按 profile 清理108条“部分分析”链，而非扩大源码扫描。108链/66685TASK分为6类：compressed physical slot12链2276TASK，slot/page/C2 length64链47636TASK，batch snapshot6链221TASK，CP suffix9链13960TASK，RoPE10链2112TASK，DSpark first-pass7链480TASK。大部分使用与166阶段源码SHA一致的本地冻结快照，RoPE复用v34证据。远端只读取本地缺失的精确slot helper，并核对三个消费者/DeviceOperator文件SHA。

最重要的排除：已有 packed slot融合helper并非legacy直接替换。ratio1时它还施加valid_end掩码，legacy只看slot非负。例如slots=[3,5]、valid_end=1，legacy flat=[3,5]，helper=[3,-1]。这是两条源码域的区别，不证明现场出现错误值。复用需证明额外掩码冗余，或显式调整mask ABI。ratio2还要同时保留物理slot完成条件和逻辑位置奇数条件；positionsNone、generic ratio、连续输出stride和state-cache PAD分支也不能丢。坐标/flat可与slot producer重组，但保留原始slot和所有组的物理所有权。

已有共享也核实了：long/index同组slot key共享持久[N,2]/flat输出，C2 quotient/remainder本已按lengths:c2每batch缓存一次，CPglobal/local已有RoPE view复用。它们不再计为新增优化。可以分析一次producer同时出quotient/remainder、共享不可变查询几何、以及合适的producer-consumer输出ABI。

发现draft共享接口缺口：调用方构造shared_dsa_draft_cache，但只有SAS共享字典，没有common_v41_batch_metadata；DSA build_for_drafting839-842又丢弃kwargs直接调用build。只转发当前kwargs仍不足，需要明确的新batch/字段共享接口。每组sliding_window可能改seq/table，必须把真正相同的QSL/positions等字段与组相关长度、physical slots、operator banks分开。不能仅因B或request IDs相同就整批共享。

CP target与dummy的完整调用栈走相同worker共享字典结构，v13“cache hit后长度临时量无内容读者”的结构证明可覆盖这一路后续cache hits；首个local/global miss、QSL固定地址发布和组字段保留。未给dummy虚报400次实测删除数量。draft当前没有共享batch传入，fresh miss结果仍有消费者，不能把target的2200次死生产者结论直接套过去。

QLI消费者已定点定位：select_projected在物理query.shape[0]==0或max_cache_seq_len==0时先返回；否则legacy读取/wait metadata后调用QLIv3。packed格式有自己的None检查和metadata路径；packed producer仅在num_actual_tokens0且非metadata_in_graph时零化。逻辑actual0不等于capture物理shape0，已有图中的QLI节点可能仍读plan。plan省略只登记为“全部实际消费者绕过且图体无读者”的条件方案，保持frontier/events/bank和格式；实际图体/参数银行绑定移交M22。

DSpark seed/tail、已有copy-expand、seq-rejected+Q、uniform QSL和canonical CPU镜像的发布顺序也已绑清，不能覆盖仍被kernel读取的旧seq或把CPU上界替换权威GPU值。RoPEuse_cache=True仍刷新mutable runtime/spec bank；省重复查询需要输出代际/末读者，不能用缓存flag或指针直接判断命中。

本轮登记{len(vv)}项条件细化、{len(conditions)}项约束。186条选中native链现均有局部字段/接口分析，但这只证明该selector的归属完整。还需将其与原M21 phase/branch/host/capture范围逐项对照，再做M22-M25及全模块有限完成审计；整体不完成，0整模块关闭。未实现/未测速不单独算未分析缺口。

保留两次本地错误：快照缺helper使准备中断；我随后误启动依赖该脚本的executor，它在本地读脚本阶段失败，未连接服务器。显式v2完成唯一一次远端CPU点读/核对。加本地点读与主审共5runs：3pass/2error。35模型候选、7框架候选；26理论比较INCOMPLETE、3136candidate资源行、849缺PMU、16phase-unbound图保持，无新模型/torch/NPU/服务/健康工作。Goal active。
'''
    local_artifacts=[O/s for s in input_files]+sorted(set(frozen))
    jobs=[dict(kind='review',status='pass',description='SameSHAfrozenpointreads/reusedRoPE/directCP-draft-QLIconsumerfieldreconciliation;no broadrepo/SDKaudit.',artifacts=local_artifacts),
        dict(kind='review',status='error',description='Localpreparationfailedbecausepackedslothelperwasabsentfromfreeze;noverify-scriptemitted,noremoteconnection.',artifacts=[O/'M21-local-helper-preparation-error-v40.json']),
        dict(kind='review',status='error',description='Dependentexecutorwasmistakenlystartedafterfailedpreparation;realrc1atlocalPath.read_text,noreceipt/SSH. Explicitv2usedlater.',artifacts=[N/'zcode-M21-bind-v40-output-20261031.json',N/'zcode-M21-bind-v40-task-20261031.txt']),
        dict(kind='review',status='pass',description='Explicitv2serverreadexactexistingpackedslotwrapper/kernelandthreeSHAchecks;allmatch.',artifacts=[N/'read_M21_consumer_binding_and_slot_helper_v40_166_v2.py',N/'M21-bind166-v40-v2.json']),
        dict(kind='review',status='pass',description='Mainmath/source review:ratio1maskcounterexample,physicalparity,draftcachehandoffgap,QLIcaptureguardand108chainreconciliation;noimplementationorclaimofmeasuredgain.',artifacts=[O/s for s in names.values()])]
    action=('继续Loop004，v40实质进展。先读M21-field-reconciliation-summary-v40.json、M21-slot-coordinate-draft-cache-and-QLI-consumer-contract-v40.json、F31-slot-fusion-domain-and-draft-cache-gap-proof-v40.json、F31-QLI-empty-query-and-capture-reader-review-v40.json、H02-CP-target-dummy-draft-publication-review-v40.json、M21-profile-chain-coverage-frontier-v40.json、module-coverage-gaps-v40.json。'
        '108partial66685TASK已按6类合同归档：compressedslot12/2276,slotpage/C2len64/47636,batch6/221,CP9/13960,RoPE10/2112,firstpass7/480；186selector0partial/0unassigned不是wholeM21。'
        'ExistingpackedslothelperSHA d1948b...有ratio1额外valid_endmask，legacy只slot>=0，例slots3/5,end1旧flat3/5新3/-1；不可blinddrop-in，adaptmask或证tail冗余。ratio2物理slot完成(s+1)%2和logicaloddpos都保，posNone/genericratio/outputstride/I32cast/statePADbranch单独。slotkeylong/index和lengths:c2 batchcache已存在不新计收益。'
        'Draft caller2560-2651只有sharedSASdict，无common_v41_batch_metadata；DSA build_for_drafting842丢kwargs。新共享需caller显式batch/fieldcache+消费接口/kwargsforward，单forward当前dict不足；pergroupwindow会改seq/table，组物理slot/operatorbank必须独立，只有完整sig相同field可共享。'
        'Worker3823 rootbatch/perKVgroup shared3850与CPglobalchild，在target/dummy同fullcaller成立；v13cachehitlengthtemporary无读者proof可结构扩展dummy后续hit，首miss/QSL/groupfields保，没新测400deadcount。Draftfreshmisssourceused别套2200targetdeadproof。'
        'QLIindexer181-298 physicalquery0或hostmaxCacheSeq0先return再blobread；否则INDEXERwait+QLIv3。Packedmetadata guard1228仅actual0&&!metadata_in_graph，legacy1254alwaysV2metadata。Logicalactual0不是capturedphysicalshape0，actualgraphbody有QLI仍读plan，未查M22不要skip。Packed/legacy/SMLA格式不混。'
        'DSpark226-357 seedprefix+zerotail/contextcopy+fusedcopyexpand→GPUQSLarange*Q/seq-oldreject+Q；保kernel旧输入和canonicalCPUclone扩展，uniformtemplates需B/Q/order/CP/epoch。RoPE use_cache仍写runtime/spec scratch，v34sameSHA已复用非pointer命中。'
        '8variants/'+str(len(conditions))+'conditions细化F31/H02/H06关联F29/F30/F34，5runs3pass2localerror：freeze缺helperprep失败，误启动dependentexecutor本地rc1未SSH；显式v2一次remoteCPU读/hashpass。35model7framework0module，26theoryINCOMPLETE/3136资源/849missingPMU/16unbound图；无model/torch/NPU/service/health,lastv12post。'
        '下一步M21原module/phase/branchmatrix与hostfields/capture/async/emptyobservers对selector的有限scope audit，别再重读108已闭合字段。随后M22完整80graph/16unbound/QLIactualbody/参数bank/inputready-lastreaderreuse/queue-transport，再M23-M25和global26module完成审计。未知实际gain/implementation不自动当未分析；Goalactive因原scope和图/process领域确实未审。')
    persist_refinement(version=40,previous=39,title='v40 M21 field reconciliation and draft cache handoff',summary=summary,
        changes=['108partialnativechains66685TASKreconciledwithsameSHApointsource/reusedcontracts;selectorhas0partialwithoutclaimingwholemodule.',
            'Existingpackedslothelperratio1maskmismatch andactualdraftkwargs/cacheinterfacegap bound;QLIphysicalemptyversuscapturedlogicalemptyreaders localized.',
            '8conditionalrefinements/27constraints;target/dummysharedcache-hitversusdraftfreshmissdomainseparated;twoinitiallocalerrorsretained,explicitv2remoteonepass.'],
        candidate_updates=model_updates,framework_updates=fw_updates,module_updates=module_updates,remaining=remaining,action=action,jobs=jobs,report=report,
        read_first=[names['contract'],names['proof'],names['qli'],names['cp'],names['coverage']],
        publication_rows={'F31':'| F31 | Metadata、slot、draft 共享及输入输出发布重组 | 条件 | v40完成186条选中native链局部字段归属；packed slot helper有更强ratio1尾mask，不可直接替换legacy。draft调用方缺共享batch接口且wrapper丢kwargs，需按完整字段签名与组bank所有权重组。QLI物理empty/zero-cache可绕过读者，逻辑empty不证明capture plan可省。原module/phase/branch与M22图体范围继续审计。 |'},
        gate_updates=dict(status='PROFILE_PARTIAL_FIELD_AND_QLI_READER_LOCAL_QUESTIONS_REVIEWED',answered_by=evidence,
            closed_scope='All108selectedpartialfieldinterfaces/slotmaskcounterexample/draftcachehandoff/CPscope/QLIphysicalemptyguardlocalquestions.',
            not_closed='OriginalM21modulephase/branch/observercoverageversusselectorandactualM22captured/transport/processdomains.',
            stop_rule='Advancefinitescopeauditratherthanrereadclosed108fieldsorimplementkernelmerelytorecordconditionalanalysis.'),
        scripts=[N/'read_M21_consumer_binding_and_slot_helper_v40_166_v2.py'],executor_prefix='zcode-M21-',main_script=N/'finalize_M21_field_reconciliation_v40.py')


if __name__=='__main__':
    main()
