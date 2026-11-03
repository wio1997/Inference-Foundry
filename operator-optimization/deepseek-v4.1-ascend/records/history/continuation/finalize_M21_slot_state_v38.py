from copy import deepcopy
from datetime import datetime,timezone
from collections import Counter
from finalize_origin_coverage_v9 import N,O,read,write
from finalize_Engram_whole_frontier_v30 import packet,graph
from persist_profile_refinement_v2 import persist_refinement


def main():
    assert not (O/'REVIEW-CHECKPOINT-v38.json').exists()
    stamp=datetime.now(timezone.utc).isoformat()
    a=packet('M21-slot166-v38-v2',True)
    b=packet('M21-state166-v38',True)
    native=packet('M21-native166-v37',False)
    error=read(N/'M21-slot166-v38.json');assert error['returncode']!=0 and 'compute_slot_mapping_fused_groups_adaptive_kernel' in error['stderr']
    pins={}
    for p in [a,b]:
        for f in p['result']['files']:
            previous=pins.setdefault(f['path'],f['sha256']);assert previous==f['sha256']
    assert pins['/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/utils.py']=='94a569fa2b54fcd206d6992c29e13a852ee5f6596d6d483fd4a33ca182c3f1f7'
    refs=[dict(receipt=str(N/(label+'.json')),remote=p['full_result'],sha256=p['result_sha256'])for label,p in [('M21-slot166-v38-v2',a),('M21-state166-v38',b),('M21-native166-v37',native)]]
    old=read(O/'M21-profile-chain-coverage-frontier-v37.json')
    positive={
        'ordinary_slot':dict(chain='9dcba1bc2f537f68199920e0f6a14959a68f181f6e1d9053afa9e92ee7fbad44',TASKs=24,token_extents={'6400':8,'6528':8,'6668':8}),
        'adaptive_slot':dict(chain='48f262051bfd7484de921099633ebd2ca2cad63b7bad9b6d4f50df0c9f3222f8',TASKs=64,token_extents={'12':16,'24':48}),
        'padded_selector':dict(chain='140f15f5cbce2071e43f8dfc37b70696a55765344ada4b1ccd7546ccced19746',TASKs=72,request_extents={'2':16,'4':56})}
    bychain={z['chain_key']:z for z in native['native_chain_headers']}
    for z in positive.values():assert bychain[z['chain']]['TASKs']==z['TASKs']
    contract='M21-slot-rejection-state-and-packing-contract-v38.json'
    proof='F31-rejection-count-state-remap-and-index-proof-v38.json'
    variants_file='M21-slot-state-variants-and-conditions-v38.json'
    coverage='M21-profile-chain-coverage-frontier-v38.json'
    branches='M21-slot-state-branch-ledger-v38.json'
    graphs='M21-slot-state-conditional-graphs-v38.json'
    write(O/contract,dict(generated_utc=stamp,source_pins=pins,evidence_refs=refs,profile_positive=positive,
        facts=[
            dict(grade='MEASURED',statement='Ordinary24/adaptive64 kernel calls have13-entry UINT64 block-table and slot-output pointer arrays. These are address arrays; their typed output is not a 13-element logical slot vector or measured slot-write byte count.'),
            dict(grade='DOCUMENTED',statement='BlockTable gate requires more than one active group and dcp_world_size==1 for every active group. Fused path also requires no compressed positions/request-index lists. Pointer arrays own distinct group tables/output addresses plus group strides/block sizes/circular flags; max output extent comes from first active table. Fallback per-group/Mamba/compressed branches are separate.'),
            dict(grade='DOCUMENTED',statement='Existing implementation already fuses groups in one launch and selects ordinary versus adaptive request tiles. Tile starts from capped1024, uses power-of-two average request length with minimum16; ordinary parallel tiles cap4 with special1/2-request policy. Adaptive runs one group/request program and chooses small tile versus1024 by actual request length. These are existing behavior, not new optimization gains.'),
            dict(grade='DOCUMENTED',statement='Both kernels reserve a program per group to write PAD_ID through [num_tokens,max_num_tokens). Active rows load QSL/positions, cast positionsINT32, compute group-specific block page/remainder, gather block numbers through a window, then write INT32 slot IDs to dereferenced group buffers. Circular groups force page0 and negative positions to PAD_ID.'),
            dict(grade='DOCUMENTED',statement='Block numbers are loaded asINT32, converted toFP32 for tl.gather, then converted backINT32 before block_size multiplication. Direct integer lookup must preserve baseline rounding or prove representable IDs. Window sizing assumes appropriate page-span/address bounds; no unsupported layout change or current bug is inferred.'),
            dict(grade='DOCUMENTED',statement='Adaptive helper has generic CP mapping, but its selected fused callers pass TOTAL_CP_WORLD_SIZE=1. CP>1 fallback mapping is not established by the selected fused ancestry and must retain its separate interleave/rank ownership domain.'),
            dict(grade='DOCUMENTED_PLUS_MEASURED',statement='Padded-input kernel72calls already emits both rejection count and selector from cumulative draft counts,INT64 valid counts,QSL. It does not compact rejected queries: original token arange,QSL,positions,slot mapping and sequence-length banks pass through; both CPU sequence-length mirrors are retained. The no-blocking-CPU comment does not make CPU shape/item work an NPU synchronization.'),
            dict(grade='DOCUMENTED',statement='Previous-batch correction gathers valid count,old computed state,previous draft count at max(prev_position,0), then uses participating=(prev_position>=0)&(previous_draft>0). It commits computed prefix and accepted counts only after these gather temporaries exist. A naive in-place fused row update can race with other old-row readers.'),
            dict(grade='DOCUMENTED',statement='Normal worker position path derives int64(computed[req_indices])+query_pos; seq_len=computed+scheduled and tail fill0. Existing CP rebuild/positions-ready alternatives bypass the normal position expression and cannot be silently replaced by it.'),
            dict(grade='DOCUMENTED',statement='Input-ID packing waits pending PP receive, preserves no-prev normal CPU input/optional embeds upload, current→previous row map,new/prefill rows,optional is_token_ids refresh,existing contiguous no-spec fastpath,live sampled IDs and flattened draft IDs. General path constructs four index arrays and scatters sampled before draft IDs withINT32 conversion.'),
            dict(grade='DOCUMENTED_PLUS_DERIVED',statement='Draft metadata performs consecutive input_ids[logits_indices] then that temporary[target_logits_indices+1]. On valid baseline index domains and stable input/indices, compose the row indices to avoid the intermediate gathered ID vector. Preserve dtype/order/bounds/error domain; count all checks/launch savings only after measurement.'),
            dict(grade='UNKNOWN',statement='Actual pointer/content equality across groups,old-state alias epochs,tail first/last readers,critical-path/physical traffic andnew tiling cost are not proved by these profiles. They are explicit conditions of the scoped opportunities, not automatic demands for implementing a kernel before recording analysis.')],
        local_questions_closed=['Existing group fusion/adaptive policy and typed pointer-array meaning','Exact fused slot active/tail/circular/dcp1 behavior','Padded rejection formula and retained input banks','Previous-batch remap read-before-write contract','General input-ID packing and composed draft gather'],
        no_implementation=True,whole_module_closed=False,overall_complete=False))
    write(O/proof,dict(generated_utc=stamp,evidence_grade='DERIVED_ON_VALID_BASELINE_DOMAINS',
        padded_formula=dict(draft_i='D[i] - (D[i-1] if i>0 else 0)',rejected_i='draft_i+1-count_i if draft_i>0 else 0',selector_i='QSL[i+1]-1-rejected_i',
            keep=['D is inclusive prefix without leading zero.','Preserve baseline signed arithmetic,INT32 output casts and mask;no new nonnegative clamp.','count is exact v36 sentinel/vocab-count result,not rightmost valid index.','No-draft branch rejects0 even if count0.','Keep all original padded queries and metadata/sequence mirrors;selector is not permission to compact noncausal query/KV rows.']),
        state_formula=dict(gather_i='max(prev_position_i,0)',participating_i='prev_position_i>=0 and old_draft[gather_i]>0',
            new_computed_i='old_computed[gather_i]+int32(old_valid_count[gather_i]) if participating_i else CPU_computed_i',
            new_accepted_i='old_valid_count[gather_i] if participating_i else old_accepted_i',
            snapshots='All gathered old values refer to the same pre-commit generation;copy_ retains its output dtype and alias semantics.'),
        in_place_counterexample=dict(old_computed=[100,200],old_valid_count=[1,2],old_draft=[1,1],prev_position=[1,0],baseline_result=[202,101],
            invalid_serial_row0_then_row1=[202,203],reason='Writing current row0 into old slot0 before row1 gathers it destroys the old100. Threads on separate programs need a real snapshot/two-phase protocol; a local barrier is insufficient across programs.'),
        compositional_opportunities=[
            'Valid-count producer may additionally emit padded rejection/selector if the same request order,D/QSL generation andoutput layout are ready. Preserve count/GPU/CPU last readers andall original sampled matrices/backup semantics.',
            'Remap correction may feed position/sequence/slot geometry in a composite pipeline. Snapshot old state before writing, keep CP rebuild path andstable published position/length banks required by other readers.',
            'On valid indices input_ids[logits_indices][target_logits_indices+1] equals input_ids[logits_indices[target_logits_indices+1]]. Skipping first-gather checks outside the selected subset is only legal on the original valid-input domain.'],
        native_validation=False,implemented=False,critical_path_gain_us=None,whole_module_closed=False,overall_complete=False))
    variants=[
        ('F31-v38-SLOT-COMMON-GEOMETRY','F31','Reuse common QSL/position and page-offset calculations across compatible group programs.','Keep each group table/output/stride/block-size/circular flag;reuse arithmetic by exact geometry,not physical slot values by shape. Existing group launch fusion is not new.'),
        ('F31-v38-SLOT-POSITION-PIPELINE','F31','Compose corrected position/sequence producers with slot mapping where readiness permits.','Snapshot old request state;keep normal versusCP-rebuilt path,all live published position/length arrays andgroup-specific readers;no cross-program producer race.'),
        ('F31-v38-SLOT-LOOKUP-LAYOUT','F31','Compare direct integer block lookup or alternate small-request tiling against existing window gather/adaptive policy.','Preserve baseline INT32→FP32→INT32 rounding orprove representability,all page-span/mask/overflow/circular semantics anddcp1 domain;measure cost rather than claim current adaptive policy absent.'),
        ('F31-v38-SLOT-TAIL-EXTENT','F31','Reduce or defer PAD tail publication only under exact consumer/generation extent proof.','Current tail write is a real full-capacity effect. Keep graph/capture addresses,all full-bank readers,shrink/grow transitions andold-generation clearing;no active-shape-only dead-tail claim.'),
        ('F31-v38-COUNT-TO-PADDED-SELECTOR','F31','Emit rejection count and sample selector alongside the existing exact valid-count producer.','Same request,D/QSL/counter generations,readiness andINT32outputs;inclusive-prefix/no-draft formula retained,not query compaction. Current padded kernel already emits both outputs.'),
        ('F31-v38-STATE-REMAP-SNAPSHOT','F31','Fuse remap gathers/predicates/updates with a snapshot-safe state pipeline.','All old-state gathers precede overwritten aliased slots;retain new-request CPU fallback,old accepted values,dtype copies andrequest permutation. Two-phase/out-of-place orone proven synchronization domain required.'),
        ('F31-v38-COMPOSE-DRAFT-GATHER','F31','Compose consecutive draft metadata index selections to avoid an intermediate ID gather.','Preserve valid baseline indices,INT32ID/INT64index domains,selected order,stable input snapshot andany observable errors;no measured removal ofall360TASK claimed.'),
        ('H02-v38-INPUT-ID-PACK-GEOMETRY','H02','Reuse immutable exact packing index templates or produce equivalent sampled/draft scatters directly.','Refines existingv13candidate;key fullprev/current map,scheduled/draft extents andstrides. IDs remain live;keep CPU-prefill/embeds/is-token-ids/PPwait andexisting contiguous fastpath.')]
    vv=[dict(id=i,parent=p,transformation=t,conditions=c,implemented=False,native_validation=False,critical_path_gain_us=None)for i,p,t,c in variants]
    requirements=[
        ('SCOPE','Only positive fused dcp1 slot,selected padded-input andnormal/state/packing source contracts are bound;no allbackend/CP/fallback closure.'),
        ('POINTERS','UINT64[13] are address arrays;preserve dereferenced INT32 group outputs andactual extents,not a13elementlogical slot result.'),
        ('OWNERS','Each group keeps its own table/output/stride/block size/circular ownership;equal blocksize/shape doesnotprove identical contents orwrite destination.'),
        ('EXISTING','Group launch fusion,adaptive small-request split andpadded dualoutput already exist;do not price them asnew.'),
        ('GATE','Keep activegroup dcp_world_size1 andcompressedposition/index-list exclusion;Mamba andCP>1fallback branches separate.'),
        ('CAPACITY','Fused baseline uses first active max extent;new layout must honoractual capacities andall target/padding writes before readers.'),
        ('INT32','Positions narrowedINT32,blockindex/remainder/slot math andsigned overflow/mask behavior preserved.'),
        ('FLOATGATHER','Direct integer lookup must reproducebaselineFP32roundtriporprove all IDs representable;no silent numerical change.'),
        ('WINDOW','Preserve window span,relative index validity,block-table stride andmasked loads;arbitrary sparse positions needtheir ownvalid-domain proof.'),
        ('CIRCULAR','Circular page0,negative-position PAD andoriginal modulo expression remain;do not merge circular/noncircular byblocksize.'),
        ('TAIL','PAD tail [num_tokens,max_num_tokens) remainsliveuntil allconsumer/epoch/captureextent evidence permitschange.'),
        ('TILING','Existing adaptive/parallel tile policy isbaseline;newlayoutmayreduce reuse oroccupancy,so operation countsare notgain.'),
        ('SNAPSHOT','All oldcomputed/count/draft gathers use precommit generation;naive perrow in-place fusion canrace onpermutations.'),
        ('SYNC','Use realtwo-phase/out-of-place orproven single synchronization domain;local blockbarrier cannot synchronizeotherprograms.'),
        ('NEWREQ','prevposition<0 andold_draft0 retainCPUcomputed fallback andoldaccepted value;clampedgather0 notnewrequest identity.'),
        ('CASTS','Preserve validcountINT64,int32computed addition andcopy_ destinationcasts;no assumed width equivalence.'),
        ('REQUESTS','Current/previousrequest map,duplicate/invalid rows andall state snapshot/last-readers explicit;B/cardinalitynotidentity.'),
        ('POSITION','Normalposition=corrected[req_idx]+querypos;CPpositionsready/rebuiltalternatives retained.'),
        ('LENGTH','Publishcorrected+scheduled seq_len andzero tail with original readiness;CPU upper mirrors arenot authoritative replacement.'),
        ('PADPREFIX','Draftcounts come from inclusive cumulativeD withoutleadingzero;preserve no-draft rejection0.'),
        ('PADFORMULA','rejected=draft+1-count ifdraft>0else0;selector=QSLend-1-rejected;no newclamp/rightmost-valid interpretation.'),
        ('PADREADY','Producerfusion requires same D/QSL/request/count epochandready inputs,plus stableoutputs andall countCPU/GPUreaders.'),
        ('NONCAUSAL','Padded inputs/position/slot/sequence banks pass through;selector doesnotprove rejectedquery/KVrowsdead.'),
        ('GATHERDOMAIN','Composed indices require validbaseline fullgather/error domain,stableindices/input,dtypes/order/bounds;not justsame shapes.'),
        ('PACKKEY','Template key includesfullprev/currentreqmap,cu_num_tokens,draftlengths,prev_num_spec_tokens,total/invalid/newrow geometry.'),
        ('LIVEIDS','Sampled/draft IDs stay live andpergeneration;packing-template reuse doesnotreuse tokenvalues.'),
        ('PACKORDER','Keep sampled-before-draft scatter semantics,disjointness/error domain andINT32draft conversion;not an automatic singleunconditionalcopy.'),
        ('OPTIONAL','Keep PP receivewait,CPU/prefilluploads,optional embeddings andis_token_ids refresh/normal-no-prev branch.'),
        ('FASTPATH','Existing exact common_indices_match/max_flattened condition remains;uniform drafts are not automatically contiguous.'),
        ('LIFETIME','Persistent/capturedaddresses,async H2Dpinned indexlifetime,bankreuse fences/firstready/lastreaders retained.'),
        ('OVERLAP','v36nextcount,v37emptyclone/countpublication andv13packing alreadyoverlap;newremovalsnotadditive.'),
        ('EVIDENCE','2952TASK/22chain localcontractcoverage isnot newwork/globalfunctionalbucket/physicaltraffic/causalgain;retain missingPMU/unboundgraphs.')]
    write(O/variants_file,dict(generated_utc=stamp,variants=vv,conditions=[dict(id='M21-v38-'+k,requirement=s,actual_current_legality_proved=False)for k,s in requirements],
        overlap=['Commongeometry/positionpipeline/directlookup/tail alternatives share slot work.','Count-to-selector andstate-remap share counts butdifferentiteration/request epochs.','Input packing refinesv13 andcount/output envelopesv36-v37;do not duplicategains.'],whole_module_closed=False,overall_complete=False))
    cs=deepcopy(old['selected_chains']);changed=[]
    for c in cs:
        leaf=c['leaf_project_frames'][0]
        state=None
        if c['state']=='PENDING_SLOT_MAPPING_KERNEL_CONTRACT':state='ANALYZED_CONDITIONAL_FUSED_SLOT'
        elif c['state']=='PENDING_REJECTED_INPUT_KERNEL_CONTRACT':state='ANALYZED_CONDITIONAL_PADDED_SELECTOR'
        elif c['state']=='UNANALYZED_FIELD_OR_CONSUMER':
            if ': update_num_computed_tokens_for_batch_change' in leaf:state='ANALYZED_CONDITIONAL_STATE_REMAP'
            elif ': _prepare_input_ids' in leaf:state='ANALYZED_CONDITIONAL_PACKING'
            elif ': _calc_spec_decode_metadata' in leaf:state='ANALYZED_CONDITIONAL_COMPOSED_GATHER'
            elif any('/model_runner_v1.py('+str(i)+')'in leaf for i in[1599,1604,1606,1645]):state='ANALYZED_CONDITIONAL_POSITION_FIELDS'
        if state:
            c['previous_state']=c['state'];c['state']=state;c['evidence']=[contract,proof,variants_file];changed.append(c)
    assert len(changed)==22 and sum(c['TASKs']for c in changed)==2952
    unknown=[c for c in cs if c['state']=='UNANALYZED_FIELD_OR_CONSUMER'];assert len(unknown)==39 and sum(c['TASKs']for c in unknown)==4280
    remain=[
        '30chains/3256TASK build_c2_metadata atdsa_v41.py1327-1368: elementwise/gather selectors andtheir cachewrite/read consumers stillunreviewed;do not label currentmetadata fusioncomplete.',
        '8chains/936TASK _get_block_table_and_slot_mapping3583/3584 fills plusworker1462 fill88:needexact initializedfield/fullbank/capture readers beforedead-write claim.',
        '101partialCPfieldchains/7partialproposerfieldchains andQLIempty-query eager/capturedreaders retain theirper-field question ledger;existing reuse doesnotautomaticallycloseallfields.',
        'Fusedslotlocalcontractanswered;compressedpositions/Mamba/CPfallback domains not promotedto currentfused proof. Taildeadness/groupvalueequality/epochs/cost areconditions ofregisteredvariants,not reasons to demandimplementation.',
        'Then M22 graph/queue/transport,M23-M25/full finite completionaudit;worker profiling doesnotclose EngineCore/frontend costs.']
    write(O/coverage,dict(generated_utc=stamp,selected_chains=cs,state_counts=dict(Counter(c['state']for c in cs)),
        newly_analyzed_scoped_chains=22,newly_analyzed_scoped_TASKs=2952,remaining_unassigned_chains=39,remaining_unassigned_TASKs=4280,
        genuine_unanalyzed_frontiers=remain,completion_policy='A conditional opportunity can be analyzed before implementation/performance validation. Local contracts here are closed;the genuinely unreviewedfields/consumers above keep wholegoal incomplete.',whole_module_closed=False,overall_complete=False))
    branch_rows=[
        ('Ordinary/adaptive fused groups','ANALYZED_CONDITIONAL','24ordinary+64adaptive,13pointerentries;existingpolicy retained.'),
        ('Circular/negative/tail','ANALYZED_LOCAL_CONTRACT','PADandINT32/FP32lookup domains bound;actualtaildeadness onlyconditional.'),
        ('dcp1 versus compressed/Mamba/CPfallback','FUSED_DOMAIN_BOUND_OTHER_DOMAIN_SEPARATE','Selectedfused gate dcp1;generichelperCPbody notborrowed asactiveproof.'),
        ('Padded rejection selector','ANALYZED_CONDITIONAL','72kernels;inclusiveD/validcount/QSL exactformula,noquerycompaction.'),
        ('Previous batch remap','ANALYZED_CONDITIONAL','1224TASK8chains;oldstate snapshot/correctedfallback andacceptedcopy bound.'),
        ('Normal versusCP-rebuilt positions','NORMAL_LOCAL_CONTRACT_CP_ALTERNATIVE_RETAINED','4chains488TASK inclseqtail/non-specselector;normalmathnotblanketCPsubstitution.'),
        ('Input ID packing','ANALYZED_CONDITIONAL','720TASK5chains;map/templates/liveIDs/optionalbranches andexistingfastpath bound.'),
        ('Consecutive draft gather','ANALYZED_CONDITIONAL','360TASK2chains,compose indexselection on validbaseline domain.'),
        ('C2/fills/otherfieldreaders','UNANALYZED_REMAINDER','39unassignedchains4280TASK pluspartialCP/proposer/QLIreaders remain.')]
    write(O/branches,dict(generated_utc=stamp,branches=[dict(branch=x,state=y,evidence=z)for x,y,z in branch_rows],whole_module_closed=False,overall_complete=False))
    g1=graph(['old_request_state','snapshot_gathers','current_request_map','corrected_computed_and_accepted','published_positions_and_seq_lens','group_specific_slot_outputs','attention_cache_consumers','reuse_release'],[
        ('old_request_state','snapshot_gathers','READ_BEFORE_COMMIT'),('current_request_map','snapshot_gathers','PREV_POSITION_CLAMP'),('snapshot_gathers','corrected_computed_and_accepted','EXACT_PARTICIPATING_OR_CPU_FALLBACK'),('corrected_computed_and_accepted','published_positions_and_seq_lens','NORMAL_OR_EXISTING_CP_ALTERNATIVE'),('published_positions_and_seq_lens','group_specific_slot_outputs','GROUP_TABLE_OWNERSHIP_AND_PAD_TAIL'),('group_specific_slot_outputs','attention_cache_consumers','PUBLISHED_READY'),('attention_cache_consumers','reuse_release','LAST_READER_FENCE')])
    g2=graph(['original_sampled_matrix','exact_valid_count','same_epoch_D_and_QSL','conditional_rejection_and_selector','unchanged_padded_inputs','draft_consumer'],[
        ('original_sampled_matrix','exact_valid_count','V36_LITERAL_RULE'),('exact_valid_count','conditional_rejection_and_selector','SAME_REQUEST_GENERATION'),('same_epoch_D_and_QSL','conditional_rejection_and_selector','MUST_BE_READY'),('unchanged_padded_inputs','draft_consumer','DO_NOT_COMPACT_QUERY_KV'),('conditional_rejection_and_selector','draft_consumer','INT32_SELECTOR_AND_REJECT')])
    write(O/graphs,dict(generated_utc=stamp,snapshot_safe_state_pipeline=g1,conditional_count_selector=g2,graphs_acyclic=True,
        note='Source/conditional dependency graphs,not measured invocation pairings or device criticalpath. Newglobal barrier/protocol mayoffset launchremoval.',implemented=False,whole_module_closed=False,overall_complete=False))
    evidence=[contract,proof,variants_file,coverage,branches,graphs]
    common=dict(contract=str(O/contract),proof=str(O/proof),conditions=str(O/variants_file),coverage=str(O/coverage),graphs=str(O/graphs),implementation=False,new_performance_result=False)
    model_updates={'F31':dict(review_v38=dict(common,variants=[v for v in vv if v['parent']=='F31']),variants_v38=[v for v in vv if v['parent']=='F31'],
        profile_review_state='POSITIVE_FUSED_SLOT_REJECTION_STATE_REMAP_AND_PACKING_LOCAL_CONTRACTS_REVIEWED;CONDITIONAL_UNPRICED',
        conclusion_changing_gap='C2 selectors/fullbank fill/readers/partialCP-proposer/QLIcapture remaingenuineunreviewedranges;registeredslot/state/counter opportunitiesconditional ratherthanunanalysedbecauseunimplemented.',
        next_profile_or_contract_action='Usev37nativefullchains tobind30C2metadatachains/8blockslotfillchains/worker1462 andtheir directconsumers;reuseexistingfieldproofs;no SDK/fullrepoaudit.',all_consumers_closed=False),
        'F29':dict(review_v38=dict(common,relation='Slot geometry reuse never merges differentphysical KVgroup tables/output ownership;keep perlayer/group writes andpadded/circular/capture banks.')),
        'F35':dict(review_v38=dict(common,relation='ExactvalidcountmayfeedrejectselectorwithsameD/QSLepoch;stillkeepsoriginalsampledIDs/snapshot/backup/countconsumersandliteralholebehavior.'))}
    fw_updates={'H02':dict(review_v38=dict(common,variants=[v for v in vv if v['parent']=='H02'],relation='Refinesv13indexgeometry/liveIDscatter andv36-v37count envelopes;PPwait/optionalCPU-payload/indexlifetime retained.')),
        'H06':dict(review_v38=dict(common,relation='Existingadaptive16..1024tiling/fused13groups,tailfullcapacityandcaptureaddressesarebaseline;newbucketing/layoutneedsactual910Ccost,noprofilesumgain.'))}
    notes={
        'M03':'Fusedslot/state/packing localcontracts pointread;hostcopy index-template reuse needs exactmap/generation,livenewIDs andoptionalfields.',
        'M04':'Previousstate correction isgather-before-write. Newreqfallback andoldacceptedvaluesnotreplacedbycount0;naivein-placefusioncanrace onrowpermutation.',
        'M10':'13-entrypointerarrays addresspergroupINT32slotbanks. Existingfused/ordinary/adaptive policyalreadyimplemented;groupownership/circular/PADtail retained.',
        'M17':'72paddedkernels exactinclusiveD/count/QSLrejectselector;count-1 helpersemantics andoriginalIDs/backup retained;selector notrightmostvalid orquerydeadness.',
        'M18':'Paddedmetadata retainsallquery/input/position/seqmirrors. Slots/selectorproducerfusion keepsnoncausaldecoderKV/queryrows,groupbanks andlastreaders.',
        'M20':'Count-to-selector composition conditionalonmatchingrequest/D/QSLgeneration;oldstatecorrection belongsnextiterationmap,notcardinalityidentity.',
        'M21':'22chains2952TASK nowlocalcontractanalyzed;39unassigned4280TASK pluspartialCP/proposer andQLIreaders remain. C2/fills next,not repeatpositive slot/reject aswhollyunknown.',
        'M22':'Tail clearing andstateout-of-place alternatives retainfixedcaptureaddresses/reusefences/allfrontiers. No crossprogramorderinferredfromsinglelaunch.',
        'M24':'Previous/currentstate snapshot withrequestpermutation isexplicit;pointer/shape equalitycannotcache tokenvalues oroverwrite oldreaders.',
        'M25':'LiveinputIDsample/draft scatter,PPwait andCPU/optionalpayload branchesbound;notcompleteEngineCore/frontend transportcoverage.'}
    module_updates={k:dict(finding=v,evidence=evidence,candidates=['F31']+(['F35']if k in['M17','M20']else[])+(['F29']if k in['M10','M18']else[]),gap='RemainingC2/fill/partialfield/QLIreader/M22-M25ranges explicit;localconditionalvariantimplementationandtimingnotrequiredforopportunityanalysis.')for k,v in notes.items()}
    remaining=[s for s in deepcopy(read(O/'module-coverage-gaps-v37.json')['remaining'])if not s.startswith('NextM21subfrontier:')]
    remaining.insert(17,'M21v38answeredpositiveordinary24/adaptive64slotmapping,paddedselector72,oldbatchremap1224TASK,inputpacking720TASK,draftgather360TASK,normalpositionfields488TASK:22chains2952TASK localcontractclosed. Existingfusion13groups/adaptivepolicy/dualoutputnotnewgain;groupownership/PADtail/FP32lookuprounding/stateprecommit snapshot/CPUfallback/paddedallqueries/PPandoptionalinputsretained.8conditionalrefinements/'+str(len(requirements))+'constraints;noimplementationgainclaim.')
    remaining.insert(18,'NextM21truefrontier:39unassignedchains4280TASK=30C2metadata3256+8blockslotfill936+worker1462fill88. Bind exactC2selector producers/cache consumers andfilledbank readers first;then101partialCP/7partialproposer/QLIempty-queryeager-capturedfieldcoverage/dummysharing. Reusev13-v38;do notrepeatslot/reject/stateformulas asunknownormakeimplementationprerequisite. M22-M25andfiniteglobalcompletionaudit remain.')
    summary=dict(filename='M21-slot-rejection-state-summary-v38.json',previous_turn_classification='PROGRESS:v37metadata/hostpublication',
        reused_native_APIs=60404,reused_native_TASKs=76033,reused_native_chains=186,ordinary_slot_calls=24,adaptive_slot_calls=64,padded_selector_calls=72,
        newly_analyzed_scoped_chains=22,newly_analyzed_scoped_TASKs=2952,remaining_unassigned_chains=39,remaining_unassigned_TASKs=4280,
        conditional_variants=len(vv),condition_rows=len(requirements),refined_model_candidates=['F31'],related_model_candidates=['F29','F35'],refined_framework_candidates=['H02'],related_framework_candidates=['H06'],new_model_candidates=[],
        remote_jobs=3,remote_pass=2,remote_retained_errors=1,
        remote_job_status='3remoteCPUreads terminal:2pass/1retained exact-name ASTguard error. Profileradaptive name omittedsourceunderscore;explicitv2correctsnameandreads directlaunchpolicy. Mainsemantic/mathreviewpass;no model/torch/NPU/service/new-health work.',
        evidence_refs=refs,no_new_model_NPU_service_or_health_work=True)
    report=f'''本轮继续按 profile 定点推进 M21。复用 v37 的 186 条完整调用链，24 次普通 slot mapping、64 次 adaptive slot mapping、72 次 prepare_inputs_padded 以及状态修正/输入 packing/索引 gather 的确切现场决定源码范围。没有扩大为仓库或 SDK 扫描。

slot mapping 已融合多组：profile 的两个 UINT64[13] 是指针数组，实际写的是各组的 INT32 slot bank。当前源码已有普通/自适应 tile 策略、每组尾部 PAD 程序和 circular 负位置处理；这些不算新增优化。新的条件方向包括公共位置/page-offset 算术复用、与位置 producer 重组、lookup/小请求布局调整，以及在完整尾部读者与代际证明后缩小 PAD 发布。各组 block table、stride、block size、circular 标志和写入所有权必须保留。现有 lookup 中 INT32 block number 经 FP32 gather 再转回 INT32，直接整数读取需保留原舍入或证明值域，不能暗改数值语义。选中 fused 路径要求 dcp1 且无 compressed position/index 列表；其他 fallback 不借此关闭。

拒绝采样 padded kernel 原本已经同时输出 rejected 和 selector：D 是无前导零的 inclusive draft 前缀，d=D[i]-D[i-1]；r=d+1-count（d>0），否则 r=0；selector=QSL[i+1]-1-r。可在同一 request、D/QSL/count 代际与 ready 边界下并入已有 count producer；保留 INT32 输出、无额外 clamp、全部 padded 输入和 metadata。不能从 selector 推出非因果 decoder 的 query/KV 行可删。现有 count 的 sentinel/vocab 与 count−1 语义仍沿用 v36。

跨迭代状态修正的关键是先 gather 旧状态再提交。对 prev_positions=[1,0]、old_computed=[100,200]、counts=[1,2]，正确结果是 [202,101]；若逐行原位写，第二行可能读到已覆盖的 202，得到 [202,203]。这是融合方案的反例，不是现有实现错误。应保留快照、分阶段或明确单一同步域。新请求/无 draft 的 CPU fallback、old accepted、INT64→INT32/copy_ 转换和 CP rebuilt positions 分支保留。公共位置/seq_len/slot pipeline 可作为条件重组，但不能凭一个 launch 消除跨 program 依赖。

输入 ID general packing 的四组索引可按完整请求映射和 schedule/draft 几何复用，token 内容始终是 live 值。保持 PP receive wait、新请求/prefill 上传、optional embeds/is_token_ids 和现有 contiguous 快路径。该方向细化既有 v13 候选，不重复计收益。两个相邻 draft-ID gather 可在合法 baseline 索引/错误域与稳定输入下组合索引，避免中间 ID gather；不把该范围全部 360 TASK 宣称为可删。

本轮将 22 条链、2952 TASK 的局部合同从未分析/待查转为条件分析，登记 {len(vv)} 项细化和 {len(requirements)} 项约束。仍有 39 条未归属链、4280 TASK：30 条 C2 metadata 链/3256 TASK，8 条 block/slot 初始化 fill 链/936 TASK，以及 worker1462 fill88；此外仍有部分 CP/proposer 字段、QLI 空 query 的 eager/captured 读者与 M22-M25。不能因此标记整体完成。条件方案未实现/未测速本身不等于未分析范围。

远端 CPU 点读 2 次通过；首次 AST 函数名 guard 因 profile 展示名少一个下划线失败，真实错误保留，显式 v2 修正后通过。加主审共 4 runs：3 pass / 1 error。全局保持 35 模型候选、7 框架候选、0 整模块关闭；26 理论比较 INCOMPLETE、3136 candidate 资源行、849 缺 PMU、16 phase-unbound 图不变。无模型/torch 导入、新 NPU 负载、服务更改或新健康检查。Goal active。
'''
    jobs=[
        dict(kind='review',status='error',description='First exact adaptive-name ASTguard failed before result write;profiler name lacked source underscore. Retained unmodified error;no model/device work.',artifacts=[N/'read_M21_slot_and_rejected_input_v38_166.py',N/'M21-slot166-v38.json']),
        dict(kind='review',status='pass',description='Explicitv2 exact fused slot/launchpolicy/blocktable owner/padded wrapper/calls source read.',artifacts=[N/'read_M21_slot_and_rejected_input_v38_166_v2.py',N/'M21-slot166-v38-v2.json']),
        dict(kind='review',status='pass',description='Direct adaptive helper/fusedbank stores/padded kernel/oldbatch state update/liveinput packing source read.',artifacts=[N/'read_M21_slot_helpers_and_rejection_state_v38_166.py',N/'M21-state166-v38.json']),
        dict(kind='review',status='pass',description='Main semantic/mathreview:old-state overwrite counterexample,exactrejectformula/packing/indexcomposition,22chain2952TASK scopedcoverage and39chain4280TASK truefrontier.',artifacts=[O/s for s in evidence])]
    action=('继续Loop004，v38实质进展。先读M21-slot-rejection-state-summary-v38.json、M21-slot-rejection-state-and-packing-contract-v38.json、F31-rejection-count-state-remap-and-index-proof-v38.json、M21-profile-chain-coverage-frontier-v38.json、module-coverage-gaps-v38.json。'
        '复用v37native60404API76033TASK186chains；24ordinary/64adaptive slots，13entryUINT64是指针数组实际各组INT32slots。已fusedgroups/自适应tile16..1024/parallel<=4/每组PADtail不是新gain；保dcp1且无compressed gate、group table/stride/blocksize/circular/outputowner、posINT32、lookupI32→F32gather→I32舍入、fullbank/capturetail。'
        '72paddedkernel已有dualoutput：D inclusive无leading0,d=diffD,r=(d+1-c)ifd>0else0,selector=QSLend-1-r；可在same req/D/QSL/count gen+ready并入countproducer，保I32/无新clamp/全部paddedqueryKV/metadataCPU mirrors。'
        '旧stateupdate1224TASK8chains先gatheroldcomputed/count/draft再commit，p=prevpos>=0&&olddraft>0；newcomputed=p?oldcomputed[g]+int(c[g]):CPU,newaccepted=p?c[g]:oldaccepted。反例prev[1,0],old[100,200],c[1,2]正确[202,101]，逐行原位可能[202,203]；需要snapshot/twophase/outofplace或真实单同步域，blockbarrier不能跨program。普通pos=computed[reqidx].I64+querypos/seq=computed+scheduled/seqtail0，保CPready/rebuilt替代。'
        'inputpack720TASK5chains保PPwait/CPUnewprefill/optionalembeds/isTokenIDs/contiguousexistingfastpath/samplethenDraftscatter/I32draftcast；模板复用完整prev-currentmap/schedule/draft/prevwidth，liveIDs不缓存，细化v13非新gain。calcmetadata360TASK2chains连续inputIDs[logits][target+1]可合法域组合index避免中间IDs，保错误domain/顺序/epoch，不称360全删。'
        '8variants/'+str(len(requirements))+'conditions细化F31/H02关联F29/F35/H06，22chains2952TASK局部合同closed，不是wholemoduleclosed。3remote2pass1ASTnameguarderror保留+mainpass=4runs3pass1error；35model7framework0module，26theoryINCOMPLETE/3136resources/849missingPMU/16unboundgraphs保持，无model/torch/NPU/service/health，lastv12post。'
        '下一步真正未分析M21：39unassignedchains4280TASK=30build_c2_metadata3256(at1327-1368)+8_get_block_table_and_slot_mapping fill936(3583/3584)+worker1462fill88；先nativecaller/shape复用后只读exactproducer/consumer。再核101partialCP/7partialproposer字段、QLI空queryeager/captured读者/dummy共享域；随后M22-M25/globalfiniteaudit。不要重读已知slot/reject公式或要求实现测速才算条件分析，Goalactive因上述真缺口。')
    persist_refinement(version=38,previous=37,title='v38 M21 fused slot ownership and snapshot-safe rejection state',summary=summary,
        changes=['Positive ordinary/adaptive slot mapping hasexisting13groupfusion/tilepolicy;pergroupbank/tail/circular/rounding domains bound.',
            'Exact padded rejection formula,old-state read-before-write counterexample andpacking/indexcomposition opportunities point-proved.',
            '8conditionalrefinements/32conditions and22chain2952TASK localcoverage;39chain4280TASK unreviewedC2/fill frontier retained.'],
        candidate_updates=model_updates,framework_updates=fw_updates,module_updates=module_updates,remaining=remaining,action=action,jobs=jobs,report=report,
        read_first=[contract,proof,variants_file,coverage],
        publication_rows={'F31':'| F31 | Metadata、索引计数、slot 与跨迭代状态发布重组 | 条件 | v38补齐24ordinary/64adaptive fused-slot、72padded selector及旧状态重排合同；已有13组融合/自适应策略不计新增收益。公共几何/position pipeline、count→selector、快照安全状态融合、组合draft gather条件登记；保group bank/PADtail/FP32lookup/CP分支/原padded queries。C2/fill/QLI读者等仍待分析。 |'},
        gate_updates=dict(status='PROFILE_SELECTED_SLOT_AND_REJECTION_LOCAL_CONTRACTS_REVIEWED',answered_by=evidence,
            closed_scope='Positive fusedslot/tile/ownership/paddedformula/oldstate snapshot/livepacking/indexcomposition localquestions.',not_closed='39unassignedC2/fill chains pluspartialCP/proposer/QLIcapturedreaders/remainingM22-M25;no wholemodule closure.',
            stop_rule='Advance actualunreviewedC2/fill/reader questions;do notrepeatlocalmath orrequire implementation/performancevalidation merely to recordconditionalanalysis.'),
        scripts=[N/'read_M21_slot_and_rejected_input_v38_166.py',N/'read_M21_slot_and_rejected_input_v38_166_v2.py',N/'read_M21_slot_helpers_and_rejection_state_v38_166.py'],executor_prefix='zcode-M21-',main_script=N/'finalize_M21_slot_state_v38.py')


if __name__=='__main__':
    main()
