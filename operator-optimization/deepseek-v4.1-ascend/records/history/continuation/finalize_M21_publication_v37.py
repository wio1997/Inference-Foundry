from copy import deepcopy
from datetime import datetime, timezone
from collections import Counter
from finalize_origin_coverage_v9 import N, O, read, write
from finalize_Engram_whole_frontier_v30 import packet, graph
from persist_profile_refinement_v2 import persist_refinement


def main():
    assert not (O/'REVIEW-CHECKPOINT-v37.json').exists()
    stamp=datetime.now(timezone.utc).isoformat()
    specs=[
        ('profile','M21-host166-v37','query_M21_global_host_copy_v37_166.py',False,
         'All32 immutable raw DBs/SHA/stat:64008selected host copy/scalar/sync records inside/outside prepare. Initial event counts valid; copy-parent matching subsequently refined.'),
        ('profile','M21-native166-v37','query_M21_native_metadata_v37_166.py',False,
         'Allrank metadata/input ancestry60404API/76033TASK/186chains plus positive typed-metadata fallback; no native-fullcaller gaps for529metadata executions.'),
        ('profile','M21-host166-v37-v2','query_M21_global_host_copy_v37_166_v2.py',False,
         'Same exact host population; match copy parents by canonical fullchain content rather than raw chain IDs. Capture outside primary payloads and queue-ID namespace multiplicity.'),
        ('review','M21-source166-v37','read_M21_SWA_count_and_metadata_v37_166.py',True,
         'Profile-selected SWA producer/localcount1116/SMLA/QLI closures/CP callers/executor.submit/index_fill990 read; producer already emits lengths.'),
        ('review','M21-count-forward166-v37','read_M21_existing_count_forwarding_v37_166.py',True,
         'Exact count uses/CP index-only forwarding/wrapper width and executor wait-release establish existing-length propagation and frontier obligations.'),
        ('review','M21-output-source166-v37','read_M21_async_sampled_output_v37_166.py',True,
         'SecondD2H is target sampled-ID matrix in AsyncGPUModelRunnerOutput, not draft IDs. Exact constructor/caller/event/sample-ID uses bind readiness and parsing.'),
    ]
    packets={label:packet(label,embedded)for _,label,_,embedded,_ in specs}
    hp=packets['M21-host166-v37'];h=packets['M21-host166-v37-v2'];n=packets['M21-native166-v37']
    hs,ns=h['summary'],n['summary']
    assert hs['name_inside_counts']==hp['summary']['name_inside_counts']
    assert (hs['selected_host_API_records'],hs['inside_prepare'],hs['outside_prepare'])==(64008,37280,26728)
    assert hs['primary_copy_names']=={'acl_memcpy_host_to_device':8744,'acl_memcpy_device_to_host':176}
    assert (ns['native_APIs'],ns['native_TASKs'],ns['native_chains'])==(60404,76033,186)
    assert ns['typed_metadata_without_native_fullcaller_TASKs']==0
    assert ns['metadata_type_counts']=={'SparseFlashMlaMetadata':321,'VllmAscendQuantLightningIndexerV2Metadata':208}
    focused=h['focused_callers']
    def host_count(name,scope):return sum(z['records']for z in focused if z['name']==name and z['scope']==scope)
    assert host_count('acl_memcpy_host_to_device','NEXT_ID_HELPER')==88
    assert host_count('acl_memcpy_device_to_host','VALID_COUNT_PUBLICATION')==88
    assert host_count('aten::clone','NEXT_ID_HELPER')==88 and host_count('aten::index_fill_','NEXT_ID_HELPER')==88
    fill=next(z for z in focused if z['name']=='aten::index_fill_')
    for z in fill['primary_shapes']:
        dims=[x.strip()for x in z['input_shapes'].split(';')]
        assert dims[2]=='0' and 'long int' in z['input_dtypes']
    zero_headers=[z for z in n['native_chain_headers']if any('/dsa_v41.py(1174)'in f for f in z['project_frames'])]
    zeros={z['contexts'][0]:z['TASKs']for z in zero_headers}
    assert zeros=={'OTHER_RETAINED':24,'DRAFT_PROPOSAL_ANCESTRY':19,'DUMMY_ANCESTRY':36}
    refs=[dict(label=k,receipt=str(N/(k+'.json')),full_result=v['full_result'],sha256=v['result_sha256'])for k,v in packets.items()]
    source_files={}
    for _,label,_,embedded,_ in specs:
        if not embedded:continue
        for f in packets[label]['result']['files']:
            prev=source_files.setdefault(f['path'],f['sha256']);assert prev==f['sha256']

    profile_file='M21-current-native-and-global-host-profile-v37.json'
    contract_file='M21-count-metadata-and-output-publication-contract-v37.json'
    count_file='F31-existing-SWA-length-propagation-proof-v37.json'
    empty_file='F31-empty-discard-readonly-alias-review-v37.json'
    output_file='H02-current-copy-payload-and-output-ready-review-v37.json'
    variant_file='M21-variants-and-condition-ledger-v37.json'
    coverage_file='M21-profile-chain-coverage-frontier-v37.json'
    branch_file='M21-branch-and-scope-ledger-v37.json'
    graph_file='M21-source-and-conditional-publication-graphs-v37.json'
    write(O/profile_file,dict(generated_utc=stamp,host=hs,native=ns,focused_host_callers=focused,
        outside_primary_copy_callers=h['outside_primary_copy_callers'],metadata_zero_callbacks=zeros,
        evidence_refs=refs,scope_notes=[
            'Old7392H2D/4936pin/312EventSync describe prepare-root scope. Current selected host scope additionally has1352primaryH2D and176primaryD2H outside prepare; old evidence remains valid within its scope.',
            'OutsideH2D1352=1088dummy-ancestry+88next-ID backup+88draftCP QSL+88draftCP suffix lengths. Dummy ancestry is not startup-only/removable.',
            '8744H2D primary APIs,8744enqueue and8744dequeue are different host layers, not3x measured DMA. Primary connection identifiers repeat while enqueue/dequeue identifiers differ; no equality/ordinal/nearest-time bridge is claimed.',
            'aten::item and _local_scalar_dense are nested API layers. Tensor device is not in these shape/dtype strings; do not call every scalar record an NPU barrier or double-count it.',
            '76033metadata/input ancestry TASKs overlap other modules and do not replace v20 M21 functional bucket76943certain+80optional.529metadata PMU records remain missing.',
            'SMLA321nonempty callbacks plus79zero-output callbacks are distinct: target240+24zero,dummy12+36zero,draft69+19zero. QLI208 remains a separate metadata format/consumer domain.',
            'No source-bound NPU collectives in this selector is not no CPU/Gloo communication. No phase/request/alias identity from counts or shape.'
        ],whole_module_closed=False,overall_complete=False))

    facts=[
        ('MEASURED','Global raw32DB primary H2D8744=7392insideprepare+1352outside; primaryD2H176outside. Backup88H2D isINT32[B] and valid-count88D2H isINT64[B],B2x32/B4x56, from same-fullchain/thread innermost host copy parents.'),
        ('MEASURED','The other88D2H isINT32[B,K] target sampled IDs,K1/B2x16,K6/B2x16,K6/B4x56. Fullcaller is basegpu322 constructor←Ascend2957sample_tokens, not draft-ID retrieval.'),
        ('MEASURED_PLUS_DOCUMENTED','All88host index_fill_ calls have an emptyINT64 index argument. DeviceOperator.index_fill990 directly calls tensor.index_fill_ and returns the same tensor. No current IndexFill ComputeInfo is implied by these host calls.'),
        ('DOCUMENTED','Next-ID helper clones sampled IDs solely before filling discard rows; subsequent predicate/count/gather reads are readonly and only next/count escape. Empty-index alias fastpath is a local value identity under a stable original input epoch. Empty discard does not imply positive valid count or dead backup.'),
        ('MEASURED_PLUS_DOCUMENTED','SWA global producer executes88times. It already returnsINT32 indices[N,1,C] andINT32 lengths[N,1]; kernel stores logical slots and raw visible_len. Do not claim emitting lengths as a new optimization.'),
        ('DOCUMENTED','Base builder1070 initializes ori_topk_length=None. Noncausal/no-precomputed-index branch obtains both outputs, copies them into stable max-token banks and returns both metadata fields1409/1410. CP165 only forwards global ori_sparse_indices[actual_start:actual_end], not its length field.'),
        ('MEASURED_PLUS_DOCUMENTED','Local noncausal builder1115/1116 therefore recomputes counts when lengthNone.69local draft count groups have276TASK:INT32->INT64 Cast,GreaterEqual,BOOL->INT32 Cast,ReduceSum. R2x27/R3x42,indices width256. This is a consumer-interface/forwarding question, not a missing producer.'),
        ('DOCUMENTED','Kernel length is raw V=L-max(L-Q-W,0), while stored nonnegative logical-slot count is clamp(V,0,C) under normal integer ranges. Wrapper enforces C>=W+configuredSpec. Equality of raw V/count additionally needs Q<=configuredSpec (or another V<=C proof); otherwise cap for the recount replacement. Physical-slot fallback is a separate domain.'),
        ('DOCUMENTED','SMLA metadata callback consumes QSL,sequence/compressed lengths/residuals,topk lengths,heads/dims/max lengths/compression/masks/windows/layout flags; it copies the returned blob into a fixed metadata bank. Actual sparse-index values are not a direct tensor argument, but purity/address binding of the opaque native format is not proved.'),
        ('DOCUMENTED','QLI metadata takes its own QSL/cache sequence/residual/heads/dims/topk/ratio/layout attributes and copies a different blob. Same metadata type/shape/group size does not establish identical plans; repeated .int() may already be a dtype no-op.'),
        ('DOCUMENTED_PLUS_MEASURED','SMLA num_actual_tokens0 zeros its metadata bank before callback completion.79positive zero callbacks remain real writes/frontiers. SourceQLI callback has no corresponding guard; whether its plan can be skipped requires actual eager/captured readers, not borrowing SMLA zero semantics.'),
        ('DOCUMENTED','DeviceMetadataExecutor.submit already sorts stage/group tasks, waits input-ready and reuse fence on a dedicated stream, runs tasks and records each frontier. Existing fullgraph batch descriptors require stable frontier sets. It rejects conflicting modes, unreleased resubmission and empty submissions.'),
        ('DOCUMENTED','wait deduplicates waits per stage/group; external events wait+reset. release records buffer_reusable unless captured producers are used, then clears in-flight state. CSE/lazy/direct-out changes must keep stage-ready and reuse protocols, including zero-query tasks.'),
        ('DOCUMENTED','AsyncGPUModelRunnerOutput retains the device sampled matrix, waits default stream on an output copy stream, copies IDs and optional logprobs/routed-experts/NaNs/fault fields, records a readiness event. get_output synchronizes before parsing; K1 uses tolist, wider output uses RejectionSampler.parse_output with invalid-request metadata.'),
        ('DOCUMENTED','Ascend caller2957 constructs that output after proposal work. Earlier ID-copy readiness or co-publication with valid counts needs a producer-specific event/generation envelope; it cannot simply remove the final event or forget optional output fields.'),
        ('UNKNOWN','Actual copy completion/critical-path gain, tensor pointer aliases, live mask/slot values, opaque metadata purity/address binding, precise copy queue bridges and complete EngineCore/frontend transport are not established by these selected sources.')]
    write(O/contract_file,dict(generated_utc=stamp,source_files=source_files,evidence_refs=refs,facts=[dict(grade=g,statement=s)for g,s in facts],
        reuse=['H02-field-and-synchronization-ledger-v13.json','H02-CP-reuse-contract-v13.json','F20-routing-ID-reuse-review-v29.json','M18-partition-bank-RoPE-and-slot-contract-v34.json','M19-embedding-HC-CP-row-and-consumer-contract-v35.json','M20-Markov-confidence-and-verification-contract-v36.json'],
        local_questions_closed=['Prepare-only versus outside copy visibility','Empty discard argument and clone read-only opportunity','Existing SWA lengths and exact CP loss/recount','SMLA/QLI metadata input signatures and existing event-frontier lifecycle','SecondD2H actual payload and async output readiness'],
        stop_rule='These local questions are answered. Continue unreviewed slot/rejection/empty-QLI/captured-reader metadata branches; do not read SDK merely to price already conditional opportunities.',
        implementation=False,whole_module_closed=False,overall_complete=False))

    write(O/count_file,dict(generated_utc=stamp,candidate='F31',evidence_grade='DERIVED_WITH_EXPLICIT_DOMAIN',producer_already_exists=True,
        source_gap='Global metadata contains ori_topk_length; CP passes only indices, and base unconditionally resets local lengthNone. A new explicit count-input consumer contract plus CP forwarding is needed; adding an unused kwarg alone is insufficient.',
        derivation=[
            'For request sequence lengthL,query lengthQ,windowW, start=max(L-Q-W,0),V=L-start. Logical kernel slot at columnc is start+c if c<V,otherwise-1; start>=0.',
            'Within safe integer bounds, among columns0..C-1 the number of nonnegative slots is clamp(V,0,C). This is the exact predicate used by local recount, including duplicates of logical values as separate columns.',
            'When 0<=Q<=configuredSpec and L>=Q and W>=0, V=min(L,W+Q)<=W+configuredSpec<=C by wrapper validation. Then existing raw length equals the recount without capping.',
            'Otherwise preserve the baseline recount by using the capped length only in this local replacement. Do not mutate existing global raw-length semantics for other consumers.',
            'Row slicing/selection commutes with column predicate count: local S(I>=0).sum=S(global_count), if CP makes no value-changing transformation. Apply exact actual_start/end,order,empty/padded-row andINT32[R,1] layout.',
            'Physical slot/block-table backend can contain negative mapped slots despite positive logical lengths; do not extend this logical-Triton proof to it without another predicate proof.'
        ],kernel_only_capacity_example=dict(W=4,Q=3,L=100,C=5,raw_visible_len=7,stored_nonnegative_columns=5,note='Illustrates required capacity condition/cap; not a claim current valid wrapper metadata violates its contract.'),
        profile_support=dict(global_SWA_kernel=88,local_count_groups=69,local_count_TASKs=276,width=256,local_rows={'2':27,'3':42}),
        stable_bank_conditions=['Keep existing global index/length banks and all other raw-length readers.','Propagated local count or capped view remains valid until metadata callback/consumer last read.','Preserve dedicated-stream input-ready/frontier/reuse and capture addresses.'],
        local_question_closed=True,implemented=False,native_numerical_validation=False,critical_path_gain_us=None,whole_module_closed=False,overall_complete=False))
    write(O/empty_file,dict(generated_utc=stamp,candidate='F31',related=['F35','H02'],evidence_grade='MEASURED_CONDITION_PLUS_LOCAL_VALUE_PROOF',
        measured=dict(host_clone_calls=88,host_index_fill_calls=88,index_dtype='INT64',index_extent=0,sampled_domains=['INT32[2,1]','INT32[2,6]','INT32[4,6]']),
        argument='For an empty valid index tensor, index_fill_ changes no sampled value. The remainder of the helper only reads the cloned matrix and returns next IDs/counts. Under a stable readonly original epoch, directly using the original matrix gives the exact same predicates,counts,count-1 selection and zero-count fallback, without a private clone/empty fill call.',
        guard='Shape/numel inspection of an already available empty-index argument can select this domain without a new data-dependent device .item(). Preserve normal dtype/dim/error conditions and input stability.',
        not_proved=['No native IndexFill kernel or physical clone-DMA count was measured by host-call counts.','Empty discard does not mean valid count>0: a row containing only -1 sentinels still needs backup.','All original sampled-ID response/logprob/count consumers and concurrent last-reader ordering remain.'],
        retained_formula='v36 literal(x!=-1)&(x<V),count-1,backup andINT32next/INT64count unchanged.',
        local_question_closed=True,implemented=False,critical_path_gain_us=None,whole_module_closed=False,overall_complete=False))

    logical_payloads=[]
    for B,K,calls in[(2,1,16),(2,6,16),(4,6,56)]:
        logical_payloads.append(dict(B=B,K=K,calls=calls,backup_INT32_bytes=B*4,count_INT64_bytes=B*8,sampled_INT32_bytes=B*K*4,note='Per-record logical tensor payload only; not measured DMA/wire/protocol/peak-memory bytes.'))
    write(O/output_file,dict(generated_utc=stamp,framework_candidates=['H02'],related=['H05','H06'],host_positive_payloads=logical_payloads,
        baseline=['BackupINT32[B]H2D during next-ID preparation.','CountINT64[B]D2H on its named count-copy stream/event before draft proposal finishes.','Target sampledINT32[B,K]D2H in AsyncGPUModelRunnerOutput after proposal, on an output-copy stream/event with optional output fields.'],
        opportunities=['Schedule a readonly sampled-ID snapshot after its actual sampler-ready event, avoiding an unnecessarily broad wait on later proposal work where source lifetimes permit.','Publish counts and target IDs in a matching mixed-dtype generation envelope if compatible, preserving their different CPU/GPU consumers and readiness.','Pack compatible CP QSL/suffix inputs into persistent views, or reuse immutable components under the existing exact geometry contract; do not re-count v13 dead suffix producers as a new saving.'],
        invariants=['SecondD2H is target sampled IDs, not draft proposal IDs or dynamic verify lengths.','Count-copy eventNone behavior,GPU reference/next-ID state and both copy last readers retained.','Async result final readiness covers optional logprobs/routing/NaN/fault data and invalid-request parsing, not only IDs.','Actual CANN copy completion/queue source bridge/frontend stall attribution not proven.','Packing may add copying/waiting/peak live storage; fewer host copies is not a speedup proof.'],
        implementation=False,whole_module_closed=False,overall_complete=False))

    variants=[
        ('F31-v37-PASS-SWA-LENGTH','F31','Carry existing producer lengths through the CP row map to avoid local index recount.','Use exact logical slot count/cap,INT32[R,1],new consumed precomputed-count ABI and existing bank/frontier lifecycle; producer already emits lengths.'),
        ('F31-v37-EMPTY-DISCARD','F31','Use a readonly original sampled matrix when the discard-index argument is empty.','Current empty-index argument measured88times. Preserve input epoch/dtype/dim,all original readers,exact next-ID formula and backup fallback.'),
        ('F31-v37-METADATA-PLAN','F31','Consider plan CSE/versioned reuse only for identical full metadata input/value/attribute signatures.','Do not merge groups by type/shape; purity/address binding of opaque format and stable output banks/frontier events remain conditions.'),
        ('F31-v37-METADATA-OUT','F31','Consider direct output into fixed metadata banks or compatible composite producer interfaces.','Keep distinct SMLA/QLI formats,zero-query output effects,all stage/group/frontier keys and input/reuse fences. Current submit is already asynchronous; no new overlap or measured copy removal claimed.'),
        ('H02-v37-EARLY-SAMPLED-ID','H02','Start the target sampled-ID host snapshot at its real readiness frontier where legal.','Constructor currently waits on default after proposal. New producer-specific event/envelope must retain matrix lifetime,other output fields and final consumer readiness.'),
        ('H02-v37-RESULT-ENVELOPE','H02','Co-publish compatible target sampled IDs and valid counts under one generation envelope.','MixedINT32matrix/INT64vector,request order,discard parsing,GPU count reference and distinct last readers preserved; not a draft-ID/verify-length substitution.'),
        ('H02-v37-CP-INPUT-PACK','H02','Pack compatible CP query-start/suffix arrays or reuse immutable pieces under exact geometry generations.','Positive draft88+88H2D and dummy416+416H2D are separate domains. Preserve QSL prefix/tail and seq-length values,static addresses and oldv13 sharing; padding/full republish may offset gains.')
    ]
    vv=[dict(id=i,parent=c,transformation=t,conditions=q,implemented=False,native_validation=False,critical_path_gain_us=None)for i,c,t,q in variants]
    requirements=[
        ('POPULATION','Selected raw host event population includes prepare and outside;notallprocesses/CPUwork/runtimeCANNevents.76033ancestryTASKs do not replace functionalPMU bucket.'),
        ('HIERARCHY','Primary/Enqueue/Dequeue and item/local_scalar are different nested/queue layers; do not add them as transfers or barriers.'),
        ('IDS','Raw connection/callchain identifiers have different namespaces; use canonical stack/thread containment for parent shape, never nearest-time/ordinal aliasing.'),
        ('EMPTY','Empty index argument,dtypeINT64/dim0 and normal sampled domain required; actual num_discarded parameter/value not inferred beyond recorded empty slice.'),
        ('READONLY','Original sampled matrix remains stable through all helper reads; preserve snapshot/readiness and all original response/logprob consumers.'),
        ('NEXTFORMULA','Preserve the literal v36 predicate (x!=-1)&(x<V), count-1 selection, holes, negative values other than -1, INT32 next IDs, INT64 counts and backup behavior.'),
        ('BACKUP','Empty discard is not positive validcount; no deletion of backupH2D without a separate exact no-fallback/generation proof.'),
        ('SCALAR','Shape/dtype lacks scalar tensor device; recordeditem does not proveNPU synchronization or freeCPU constant behavior.'),
        ('LOGICAL','Length propagation proof applies to actual logical-Triton slot branch;physical block-table negative slots need another proof.'),
        ('CAPACITY','Raw visible_len is clamped to[0,C] for recount unlessV<=Cproved,for exampleQ<=configuredSpec andwrapperC>=W+configuredSpec;safe integer range retained.'),
        ('ROWMAP','Apply exact CP actual_start/end/order/empty/padding and predicate-preserving transformations,not inferred rank chunks/count pairings.'),
        ('COUNTSHAPE','PreserveINT32[R,1] count layout and baseline counts for all stored columns;do not modify global raw lengths for other readers.'),
        ('CONSUMERABI','Base currently initializes lengthNone;pass a count only through an explicit consumed ABI,newinterfaceor equivalent protocol.'),
        ('BANKS','Existing global index/length and fixed SMLA/QLI bank addresses/capacity/tail/lifetime and other consumers retained.'),
        ('PLANKEY','FullQSL/seq/compressed/residual/count values and all scalar masks/layouts/windows/head/config/version inputs required;notpointer/shape-onlyplan reuse.'),
        ('OPAQUE','Native metadata purity/device/address binding and format compatibility are conditions;noSDKinternal proof orphysicalefficiency inferred.'),
        ('FORMATS','SMLA/QLI/cache-kind/ratio/group outputs differ;current `.int()` mayalready be dtype no-op,not new removable casts.'),
        ('ZERO','SMLA no-query zeros metadata and recordsfrontier;do not drop required bank writes/events or assume QLI can use the same zero format.'),
        ('FRONTIER','Submit stage/group sorting,input-ready,reusefence,stable externalfrontier set and perstage event records retained oncachehits/compositeoutputs.'),
        ('WAITRELEASE','wait externalreset/dedup and release buffer-reusable event/inflight mode preserved;existing asyncsubmit not counted as a new gain.'),
        ('OUTPUTABI','Direct-out/composite metadata needs valid fixed-bank native interface and all first-consumer readiness;API limitations do not prove semantic impossibility.'),
        ('SAMPLERREADY','Early IDcopy uses actual completed sampler-input epoch and producer event,not an unsupported change to broad streamwait timing.'),
        ('ASYNCFIELDS','Final output event covers optional logprobs/routed/NaN/fault fields,invalid-request handling and K1versuswiderparse behavior.'),
        ('DTYPES','Target sampled IDsINT32 andcountsINT64 distinct;transport packing retains both typed views/alignment and original sampled matrix values.'),
        ('COUNTEVENT','CountcopyeventNone/asyncGPUcountreference/prevIDstate and bothGPU/CPU last readers remain,not just outputvector equality.'),
        ('ENVELOPE','req_id/order/generation/snapshotcapacity stable until all consumers finish;bufferreuse cannot overwrite pending copies/corrections.'),
        ('CPFIELDS','Packed QSL andseqsuffix values/active extents/paddedtail/CPgroup policy retained;sameB does not mean identical content.'),
        ('OLDSHARING','Existing v13cache-hit/deadproducer sharing and slot-group packing are not new savings;packing immutable andmutable pieces can increase bytes.'),
        ('DUMMY','Dummy-ancestry operations mayserve runtimeDP/profile/captureallocation;do not delete byname orlabelstartup-only.'),
        ('COST','Hostwalls/counts/logicalbytes/counterratios notcausalcriticalpath orphysicalDMA;retain529missingmetadataPMU and all global uncertainty.')]
    write(O/variant_file,dict(generated_utc=stamp,variants=vv,conditions=[dict(id='M21-v37-'+k,requirement=q,actual_current_legality_proved=False)for k,q in requirements],
        overlap=['Empty-discard fastpath overlapsv36virtualdiscard fusion;count one combined removal.','Passing existing lengths replaces recount work;do not also claim generating lengths as new gain.','Earlycopy andco-publication can trade waiting/packing;their benefits are not additive.','Metadata cache/directout/composite and preexisting stream overlap sharebank work;price jointly.'],
        implementation=False,whole_module_closed=False,overall_complete=False))

    # Complete navigation of the selected186chains, without pretending every field is analyzed.
    chains=[]
    for z in n['native_chain_headers']:
        text='\n'.join(z['project_frames']);ops={x['signature']['op']for x in z['typed_shapes']}
        state='UNANALYZED_FIELD_OR_CONSUMER';refs_for_chain=[]
        if any('slot_mapping'in op for op in ops):
            state='PENDING_SLOT_MAPPING_KERNEL_CONTRACT';refs_for_chain=['v13blocktablepublicationisnotfused-slotkernelproof']
        elif 'prepare_inputs_padded_kernel'in ops:
            state='PENDING_REJECTED_INPUT_KERNEL_CONTRACT';refs_for_chain=['v36countconsumerhandoffonly']
        elif ': prepare_next_token_ids_padded'in text:
            state='ANALYZED_CONDITIONAL_LOCAL';refs_for_chain=['F31-next-ID-and-acceptance-summary-review-v36.json',empty_file]
        elif any(q in text for q in[': build_smla_metadata',': build_qli_metadata','dspark_swa_indices.py(72)','/dsa_v41.py(1116)']):
            state='ANALYZED_CONDITIONAL_INTERFACE';refs_for_chain=[contract_file,count_file]
        elif '/spec_decode/dspark_proposer.py('in text:
            state='REUSED_PARTIAL_FIELD_CONTRACT';refs_for_chain=['M18-partition-bank-RoPE-and-slot-contract-v34.json']
        elif 'dsa_v41_cp.py('in text:
            state='REUSED_PARTIAL_CP_FIELD_CONTRACT';refs_for_chain=['H02-CP-reuse-contract-v13.json',count_file]
        chains.append(dict(chain_key=z['chain_key'],TASKs=z['TASKs'],contexts=z['contexts'],ops=sorted(ops),leaf_project_frames=z['project_frames'][:3],state=state,evidence=refs_for_chain))
    write(O/coverage_file,dict(generated_utc=stamp,selected_chains=chains,state_counts=dict(Counter(z['state']for z in chains)),
        local_closed_questions=['Global selected copy visibility and payload identity','Empty-discard localvalueidentity','SWA existing length loss/recount andcapacity/rowmap conditions','Metadata interface/frontier and asyncsampled-output lifecycle'],
        genuine_unanalyzed_frontiers=['24ordinary+64adaptive fused slot-mapping kernels andtheir input/output ownership,notjust blocktablecopies.','72prepare_inputs_padded kernel/rejection-correction/position/slot/selector producer-consumer contract.','Other unassigned fieldchains anddummy CP shared-dictionary scope beyond existing prepare-rootproof.','QLI empty-query/captured-readers before claiming outputdead;SMLAzero rule cannot substituteopaqueQLIformat.','Copyqueue source bridge and complete transport/process coverage inM22-M25.'],
        completion_policy='ANALYZED_CONDITIONAL means the opportunity and its legality conditions have been analyzed,not implemented/measured. Unvalidated gain alone is not an unanalyzed range. Whole goal remains incomplete because actual field/branch/consumer frontiers above remain unreviewed,not merely because no kernel was implemented.',
        whole_module_closed=False,overall_complete=False))
    branches=[
        ('Prepare/outside host','MEASURED_SCOPED','64008selectedevents;7392+1352primaryH2D,176D2H;canonical copyparent refinement preserves population.'),
        ('Empty-discard next IDs','ANALYZED_CONDITIONAL','88emptyINT64index arguments andreadonly helper localproof;backup stilllive whenvalidcount0.'),
        ('Logical SWA count/CP','ANALYZED_CONDITIONAL','Existing raw lengths and index-only CP forwarding; clamp/width/row-view conditions apply to the current 69 recount groups / 276 TASKs.'),
        ('SMLA nonempty/empty','ANALYZED_INTERFACE','321metadataop+79zero callbacks;frontier/bank effects retained.'),
        ('QLI plan/empty/readers','INTERFACE_ANALYZED_READERS_PENDING','208opaqueplans;no copiedSMLAzero semantics orno-querydeadness claim.'),
        ('Captured/external producers','ANALYZED_PROTOCOL_LOCAL','Stage/group frontier set,eventwait/reset and bufferreusable lifecycle pinned;completeM22graph/aliasaudit pending.'),
        ('Target sampled-ID output','ANALYZED_CONDITIONAL_LOCAL','88matrixD2H and88countD2H have different streams/events/dtypes/consumers;notdraftIDs.'),
        ('Dummy metadata/copies','PARTIAL_REUSED','1088outsideH2D dummy ancestry retained;notstartup-only;per-group reuse scope needs itsown proof.'),
        ('Slot/reject/future cases','UNANALYZED_REMAINDER','Fused/adaptive slotkernels,prepare_inputs_padded andunreviewedfieldchains remain explicitnextwork.')]
    write(O/branch_file,dict(generated_utc=stamp,branches=[dict(branch=b,state=s,evidence=e)for b,s,e in branches],whole_module_closed=False,overall_complete=False))

    baseline=graph(['global_query_metadata','SWA_indices_and_rawlength','stable_global_banks','CP_index_slice_only','local_predicate_reduce','SMLA_callback','fixed_plan_bank','stage_ready','attention','reuse_release'],
        [('global_query_metadata','SWA_indices_and_rawlength','LOGICAL_TRITON'),('SWA_indices_and_rawlength','stable_global_banks','BOTH_OUTPUTS'),('stable_global_banks','CP_index_slice_only','ONLY_INDICES_CURRENT'),('CP_index_slice_only','local_predicate_reduce','LENGTH_RESET_NONE'),('local_predicate_reduce','SMLA_callback','INT32_COUNTS'),('SMLA_callback','fixed_plan_bank','COPY_OR_ZERO'),('fixed_plan_bank','stage_ready','FRONTIER_RECORD'),('stage_ready','attention','WAIT_READY'),('attention','reuse_release','LAST_READER_FENCE')])
    propagated=graph(['global_indices','global_rawlength','exact_CP_rowmap','capped_local_count','new_count_input_ABI','same_metadata_frontier'],
        [('global_rawlength','exact_CP_rowmap','SAME_ROW_GENERATION'),('global_indices','exact_CP_rowmap','PREDICATE_PRESERVING'),('exact_CP_rowmap','capped_local_count','CLAMP_IF_CAPACITY_UNPROVED'),('capped_local_count','new_count_input_ABI','INT32_Rx1'),('new_count_input_ABI','same_metadata_frontier','PRESERVE_OUTPUT_AND_EVENTS')])
    transport=graph(['sampler_original_ID_matrix','readonly_next_ID_helper','valid_count','count_copy_event','GPU_count_last_reader','proposal_work','Async_output_constructor','output_copy_event','CPU_parse'],
        [('sampler_original_ID_matrix','readonly_next_ID_helper','SAME_SOURCE_EPOCH'),('readonly_next_ID_helper','valid_count','EXACT_V36_RULE'),('valid_count','count_copy_event','COPY_STREAM'),('valid_count','GPU_count_last_reader','ASYNC_REFERENCE'),('readonly_next_ID_helper','proposal_work','NEXT_ID'),('proposal_work','Async_output_constructor','CURRENT_HOST_SOURCE_ORDER'),('sampler_original_ID_matrix','Async_output_constructor','RETAINED_TARGET_MATRIX'),('Async_output_constructor','output_copy_event','DEFAULT_WAIT_AND_OPTIONAL_FIELDS'),('output_copy_event','CPU_parse','SYNCHRONIZE_AND_PARSE')])
    write(O/graph_file,dict(generated_utc=stamp,source_count_publication=baseline,conditional_existing_length_forwarding=propagated,source_distinct_output_publications=transport,
        note='Graphs show source/data/frontier contracts,not a measured request pairing or devicecriticalpath. Earlier/sharedcopy variant needs new producer-specific readiness protocol.',graphs_acyclic=True,implementation=False,whole_module_closed=False,overall_complete=False))

    evidence=[profile_file,contract_file,count_file,empty_file,output_file,variant_file,coverage_file,branch_file,graph_file]
    common=dict(contract=str(O/contract_file),count_proof=str(O/count_file),empty_discard=str(O/empty_file),output=str(O/output_file),conditions=str(O/variant_file),coverage=str(O/coverage_file),graphs=str(O/graph_file),implementation=False,new_performance_result=False)
    candidate_updates={'F31':dict(review_v37=dict(common,variants=[z for z in vv if z['parent']=='F31']),variants_v37=[z for z in vv if z['parent']=='F31'],
        profile_review_state='GLOBAL_HOST_AND_NATIVE_METADATA_SCOPE_EXISTING_SWA_LENGTH_LOSS_EMPTY_DISCARD_AND_FRONTIERS_REVIEWED;CONDITIONAL_UNPRICED',target_source_gate=str(O/'PROFILE-CODE-QUESTION-GATE-v37.json'),
        conclusion_changing_gap='Local copy/count/frontier contracts answered. Slot/rejection/QLIcapturedreader/remainingfield scopes unreviewed;actualvalues/opaqueformat/nativeprecision/causalgain remain candidateconditions.',
        next_profile_or_contract_action='Continue actual unreviewedM21slot-mapping/adaptive/rejected-input andfieldconsumer contracts;do not repeat existing-length producer or prepare-onlyglobalcount confusion.',all_consumers_closed=False),
        'F35':dict(review_v37=dict(common,relation='Current next-helper discard index is empty andtargetsampledmatrix has its ownD2H/parse reader. Preserve original IDs,count protocol andexactv36predicate;emptydiscard does not prove backupdead.')),
        'F29':dict(review_v37=dict(common,relation='SWAquerymetadata length reuse concerns logicalindex/CP count domain;not per-layerKVvalue/physicalslot/cachewrite equivalence.'))}
    framework_updates={
        'H02':dict(review_v37=dict(common,variants=[z for z in vv if z['parent']=='H02'],current_primary_H2D=8744,prior_prepare_only_H2D=7392,outside_H2D=1352,outside_primary_D2H=176),implementation=False,new_performance_result=False),
        'H05':dict(review_v37=dict(common,relation='Worker-side AsyncGPUModelRunnerOutput readiness/sourcecopy nowbound;not causal attribution of frontendDecodeStream walltime orcompleteEngineCore process coverage.')),
        'H06':dict(review_v37=dict(common,relation='Metadatafrontiers/zero-querybank updates andlogicalSWAwidth/count relation constrain future bucket regrouping;no measuredphysicalpaddinggain.'))}
    notes={
        'M03':'Global selected hostcopy outsideprepare nowinventoried;queue layers/connectionnamespace andmissingCPUdevice scalar type do not proveextraDMA/barriers/criticalpath.',
        'M04':'88emptydiscard arguments support readonlyclone fastpath understableinput;backupH2D88 stillneededunlessseparate nofallback/generation proof.',
        'M09':'QLI208metadata interface bound;SMLA empty-queryzero cannot be borrowed to skipQLI without captured/eagerreader proof.',
        'M10':'SWA kernel88already emitsrawlength;CP forwardsindicesonly,69local276TASK recount canbe replacedconditionallybyrowmapped/cappedcounts,newconsumedABI.',
        'M17':'Target sampledIDmatrixD2H88isdistinctfromvalidcountD2H88;originalmatrix/parse/logprob readers preserved bynext-ID fusion.',
        'M18':'Actual draft globalSWAindices88/localSMLA69+19zero fields distinguished fromcontextKVprecompute andper-layercachewrites.',
        'M21':'76033nativeTASK/186chain scope and64008hostevents inside/outsideprepare added;7conditional refinements with explicitcoverage-frontier registry. Slot/adaptive/reject andotherfields stillunanalyzed.',
        'M22':'Existing metadata executor already asynchronouslyrecords allfrontiers and reusefences;cache/directout/zeroquery changes preserve fixedbatchdescriptor keys,wait/reset/release andcapture bank ABI.',
        'M23':'Source-bound NPUCOMM0inmetadata selector isnotnoCPUcommunication. Queue wrappers andsharedconnectionIDsnotadditionaltransfers orprovenflow links.',
        'M24':'Count/plan/backup/input/outputreuse requiresvalue/config/requestgeneration andlastreaders;reusedpointer orfieldshape doesnotdefineepoch.',
        'M25':'AsyncGPUModelRunnerOutput retains targetIDs,copy-waits default afterproposal andgatesparsewith event plusoptionalfields;worker snapshot analysisdoesnotclosefrontend/EngineCore lifecycle.'}
    module_updates={mid:dict(finding=text,evidence=evidence,candidates=['F31']+(['F35']if mid in['M04','M17','M25']else[])+(['F29']if mid in['M10','M18']else[]),gap='Listed genuine unreviewedfield/branch/consumer scopes remain;conditional candidate notimplemented/measured isnot itself an unanalyzedscope.')for mid,text in notes.items()}
    remaining=[x for x in deepcopy(read(O/'module-coverage-gaps-v36.json')['remaining'])if not x.startswith('NextwholecoveragefrontierM21')]
    remaining.insert(16,'M21v37localcontractsanswered:globalselectedhost8744primaryH2D(7392prepare+1352outside)/176D2H,88backup/88count/88targetmatrixcopies andemptydiscardindices;existingSWAproducerlength→CPindex-onlyloss→69recount276TASK;metadatafrontier/waitrelease/AsyncOutputpayloadbound.7scopedconditionalrefinementsregistered. Do notrepeatproducerlengthasmissingor0ComputeInfoasnobackup/clone/indexfill. Actualgain/aliases/opaqueformat remain candidateconditions,notautomaticanalysisgaps.')
    remaining.insert(17,'NextM21subfrontier:24ordinary+64adaptive fusedslotmapping and72prepare_inputs_padded/rejectioncorrection plus186-chain registryunreviewedfields. Reusev13tablecopy/v29ID/v34slot/v36nextcount/v37lengthandfrontier contracts. BeforeQLIno-queryskip bindactualeager/capturedreaders;SMLAzero79doesnotproveQLIplanunused. ExtenddummyCPdeadproduceronlyafterownshared-generationproof. ThenM22graph/queue/transport,M23-M25andfiniteglobalcompletionaudit. Do notmistakeunvalidatedconditionalgainforunanalyzedrange,orreportpolishforcompletion.')
    summary=dict(filename='M21-metadata-and-host-publication-summary-v37.json',previous_turn_classification='PROGRESS:v36confidence/nextIDhandoff',
        native_APIs=60404,native_TASKs=76033,native_chains=186,selected_host_APIs=64008,inside_prepare_host=37280,outside_prepare_host=26728,
        primary_H2D=8744,prepare_primary_H2D=7392,outside_primary_H2D=1352,primary_D2H=176,next_backup_H2D=88,valid_count_D2H=88,target_sampled_ID_D2H=88,
        empty_discard_index_calls=88,SMLA_metadata=321,QLI_metadata=208,SMLA_zero_callbacks=79,SWA_index_producers=88,local_count_groups=69,local_count_TASKs=276,
        conditional_variants=len(vv),condition_rows=len(requirements),refined_model_candidates=['F31'],related_model_candidates=['F35','F29'],refined_framework_candidates=['H02'],related_framework_candidates=['H05','H06'],new_model_candidates=[],
        remote_jobs=6,remote_pass=6,remote_retained_errors=0,
        remote_job_status='6remoteCPUjobs terminal/pass. First hostquery eventcountsvalid;canonicalparentv2refines relation. One initial localinline-preparation SyntaxError retained,no remote launch. Main semantic/math reviewpass;no model/torch/NPU/service/new-health work.',
        evidence_refs=refs,no_new_model_NPU_service_or_health_work=True)
    report=f'''本轮继续M21，先补profile范围再定点源码。32份raw DB经SHA和只读stat校验，选出64008条host copy/scalar/sync记录，其中37280在prepare内、26728在外。primary H2D总8744=原prepare7392+区间外1352，primary D2H176均在外。Enqueue/Dequeue另列，不当成额外DMA；它们的connection标识与重复primary标识不能直接相等配对。初版主事件计数有效，v2改用canonical完整栈内容+同线程host包含匹配copy父节点，修正raw callchainId不等于内容身份的问题。旧prepare台账仍按其原范围保留。

正向确认88次backup INT32[B]H2D、88次valid-count INT64[B]D2H，以及另一88次target sampled-ID INT32[B,1/6]D2H。后一项来自AsyncGPUModelRunnerOutput，不是draft IDs。额外H2D中1088条属于dummy ancestry，仍不解释成只在启动发生。item/local_scalar是嵌套API记录，缺少tensor device信息，不能统称NPU同步；传输记录/逻辑shape也不等于实际DMA字节或关键路径。

88次host index_fill_的INT64索引长度均为0，native没有因此出现IndexFill ComputeInfo。源码表明helper仅为discard fill先clone，后续对矩阵只读。因此可在空索引、合法dtype/dim和稳定输入代际下直接读原sampled矩阵，省私有clone/empty fill调用。它与v36虚拟discard单核重叠，不重复计收益。空discard不保证count>0，原!=−1且<V、count−1选择、零数量backup和所有原矩阵读者保留；不能据此删88次backup拷贝。

native metadata/input ancestry覆盖60404 API、76033 TASK、186条链。321次SMLA和208次QLI metadata与原529条缺PMU范围一致，不用它们声称物理低利用率。SMLA另有79次零输出callback（target24/dummy36/draft19），既有代码已对空query零化并维持frontier。QLI没有同样guard，其空query/captured-reader是否可省仍待核对，不能直接套用SMLA零格式。

SWA长度发现是本轮主要局部结论：88次索引kernel原本就生成indices与raw visible_len，复制进稳定bank并随global metadata返回；CP165只转发切片后的indices，base1070又把length置None，局部1116遂有69组、276TASK的有效列重算。应补明确消费接口并传已有长度，不是新造长度producer。对逻辑slot，完整列计数为clamp(visible_len,0,index_width)。wrapper的容量校验配合query_len不超过配置spec宽度可证明无需cap；未证明时要保持局部recount的cap语义，不能改全局raw长度，也不能扩展到可能有负physical slot的后端。CP rowmap、INT32[R,1]、bank和ready/last-reader代际均保留。

SMLA/QLI的全部调用输入和metadata executor局部合同已补齐。现有executor已经在专用stream上等待inputs/reuse fence，按stage/group运行并逐frontier记event；capture batch descriptor要求frontier集合稳定，wait外部event会reset，release记录可复用栅栏。metadata plan cache、direct-out或composite接口可以作为条件方案，但不能省这些事件，不能按type/shape合并不同group，也不能把已有async当新收益。当前opaque plan格式/地址绑定仍是合法性条件，未为登记候选而追SDK内部。

输出构造会保持device sampled矩阵，copy stream等待default，再拷IDs及可选logprobs/routing/NaN/fault字段，最后event约束CPU解析。可研究把只读IDs快照放到实际sampler就绪边界，或把IDs与counts按同一request/generation发布；它们dtype、消费者、last-readers不同，packing不保证更快，worker证据不证明frontend DecodeStream长等待由此造成。

本轮登记{len(vv)}项条件细化、{len(requirements)}项约束，更新F31及H02并关联F35/F29/H05/H06。186条profile链均进入导航台账，明确区分ANALYZED_CONDITIONAL、部分复用及真正未分析字段。仍有24普通/64adaptive slotmapping、72prepare_inputs_padded/reject correction及QLI captured-reader等范围待查，所以整体不完成；未实现/未测速是候选证据级别，不单独当成“未分析”的理由。

6次远端CPU作业均pass；最初本地inline SQL/Python引号冲突的准备错误保留，未启动远端，主审通过。35模型候选、7框架候选、0整模块关闭，26项理论比较INCOMPLETE、3136candidate资源行、849PMU缺失、16个phase-unbound图保持。无model/torch导入、新NPU负载、服务更改或新健康检查，最后健康仍v12post。下一批继续M21剩余slot/reject/字段合同，之后M22-M25与全范围有限完成审计。Goal active。
'''
    jobs=[dict(kind=kind,status='pass',description=desc,artifacts=[N/script,N/(label+'.json')])for kind,label,script,_,desc in specs]
    jobs.append(dict(kind='review',status='error',description='Initial local inline generator failed Python parsing due nested SQL quotes;no script/task/remote launch. Retained receipt,corrected with direct file patch and syntax validation.',artifacts=[N/'M21-local-preparation-guard-v37.json']))
    jobs.append(dict(kind='review',status='pass',description='Main review validates exact outside-copy scopes/canonicalparents/emptydiscard/SWA count-propagation proof/frontier and output lifecycle;7variants and explicitremaining186chainfrontier,no implementation/gain.',artifacts=[O/q for q in evidence]))
    action=('继续Loop004，v37实质进展。先读M21-metadata-and-host-publication-summary-v37.json、M21-count-metadata-and-output-publication-contract-v37.json、F31-existing-SWA-length-propagation-proof-v37.json、M21-profile-chain-coverage-frontier-v37.json、module-coverage-gaps-v37.json。'
        'Host64k selected=64008,prepare37280/out26728；primaryH2D8744=7392prepare+1352out，D2H176。outH2D1088dummy+88backup+88draftCPQSL+88suffix。backupI32[B]88/countI64[B]88D2H/targetsampleI32[B,K1or6]88D2H；第二路不是draftID。QueueEnq/Deq8744各不当extraDMA，conn namespace不等不能强配；item/local_scalar不能无device信息当NPUbarrier。hostv1人口有效,v2canonical内容+TID包含修正父copy匹配，rawchainID不是内容。'
        'Native60404API76033TASK186chain,321SMLA+208QLI全529missingPMU不变，无typed-onlygap；源SMLA空queryZero79=target24+dummy36+draft19，321op+79zero callbacks；QLI无同guard不能直接假定dead/可用zeroformat。'
        '88empty I64 index_fill参数，DeviceOp990原位fill，helper后段只读；稳定原矩阵代际可alias省privateclone+emptyfill，保留v36谓词/count-1/backup/dtypes，emptydiscard不等于c>0/backupdead。NoNPUIndexFill或物理cloneDMA数量/gain由hostcounts推出。'
        'SWAkernel88本来已输出indices与raw visible_len；globalmetadata1409/1410存两者，CP165只传indices[actual_start:end]，base1070长度None→69localrecount276TASK。新方案需显式consumerABI+CP传已有长度，非新producer；logical count=clamp(V,0,C)，V=min(seqL,W+Q)正常域。Q<=cfgSpec且C>=W+cfgSpec才rawcount相等，否则局部cap，别改globalrawreader/套physicalnegativeSlots；INT32[R,1]、rowmap、stablebanks/events保留。'
        'Executor已asyncsubmit，inputReady/reuseFence/stagegroupfrontier/fullgraphdescriptor固定集合/waitreset/release保留；metadataKey全输入value+scalarattrs/config/version，非type/shape/pointer。opaqueformat/purity/addressbindings条件，不为candidate登记追SDK。'
        'AsyncGPUOutputconstructorbase290-343 retains originalsampled,copyStreamwaitDefault后ids+optionalfields最后record,get_output351sync后K1tolist/更宽parse；worker2957在proposal后构造。Early sampler-ready readonlysnapshot或IDs+countsgenEnvelope条件方案，保CPU/GPUcount独立lastreader/alloptionalfields/invalidReq解析；不归因frontend长wall。'
        '6remoteallpass+localprepquoteerror1retained+mainpass=8runs7pass1error，35model7framework0wholemodule，7variants/'+str(len(requirements))+'conditions；F31/H02refine,F35/F29/H05/H06relation。No model/torch/NPU/service/newhealth,lastv12post；26theoryINCOMPLETE/3136candidate资源/849missingPMU/16unboundgraph保留。'
        '下一批继续真正未分析M21：24ordinary+64adaptive fusedslotmapping、72prepare_inputs_padded/reject correction和186chain registryremainingfields、QLI empty-query/capturedreader、dummyCPcache共享域；复用v13/v29/v34/v36/v37避免重读。随后M22graphqueue/transport、M23-M25/globalfiniteaudit。ANALYZED_CONDITIONAL不是实现验证，但未测速/未实现本身不算未分析；Goal保持active因确有上述未分析范围。')
    persist_refinement(version=37,previous=36,title='v37 M21 global host publication and existing SWA length reuse',summary=summary,
        changes=['Global selected host scope reconciles prepare/outside primarycopies andcanonicalparent shapes withoutqueue/scalar-layer inflation.',
            'Existing SWA length producer/CP loss/recount and empty-index readonly opportunity point-proved;metadatafrontiers andactualtarget-ID output lifecycle bound.',
            '7conditional refinements and complete186chain navigation with genuine remaining slot/reject/reader scopes recorded;unvalidatedgainnotconflatedwithunanalyzedrange.'],
        candidate_updates=candidate_updates,framework_updates=framework_updates,module_updates=module_updates,remaining=remaining,action=action,jobs=jobs,report=report,
        read_first=[contract_file,count_file,empty_file,output_file,coverage_file],publication_rows={'F31':'| F31 | Metadata、索引计数及输入输出发布重组 | 条件 | v37补齐prepare外1352H2D/176D2H；SWA producer已输出length，CP只传indices导致69组276TASK重计数，可按容量/rowmap传已有count；88empty-discard支持只读alias条件。metadataKey/direct-out和copy发布保持固定bank、frontier及输出全部读者；slot/adaptive/reject/QLI捕获读者仍待分析。 |'},
        gate_updates=dict(status='PROFILE_AND_NECESSARY_POINT_CONTRACTS_REVIEWED',answered_by=evidence,
            closed_scope='Global selected host-copy visibility/canonicalparents/emptydiscard/localSWAcount loss and capacity proof/metadata eventfrontier/actualsampled-ID D2H lifecycle.',
            not_closed='Slot/adaptive/reject kernels/remainingfieldchains/QLIcapturedreaders/dummysharing/completequeue-transport/processcoverage;actualalias/nativeformat/gain conditions retained.',
            stop_rule='Continue genuineM21remainingfield/branch contracts. Do not repeat producerlength invention or demand implementation/speedup merely to count analyzedconditionalopportunities.'),
        scripts=[N/script for _,_,script,_,_ in specs],executor_prefix='zcode-M21-',main_script=N/'finalize_M21_publication_v37.py')


if __name__=='__main__':
    main()
