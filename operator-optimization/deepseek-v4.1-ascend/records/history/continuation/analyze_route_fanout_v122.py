"""Recorded token/destination fanout and complete dispatch/return payload references."""
from pathlib import Path
from collections import Counter
from fractions import Fraction
import gzip,hashlib,json,struct
from finalize_origin_coverage_v9 import N,O,W,read,write

def pin(p):
    p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def section(p,a,b):return dict(**pin(p),first_line=a,last_line=b,source=Path(p).read_text().splitlines()[a-1:b])

def main():
    assert not(O/'REVIEW-CHECKPOINT-v122.json').exists()
    raw=read(O/'route-fanout-saved-records-v122.json');assert raw['snapshot_records']==1392 and not raw['Torch_imported']and not raw['new_server_files_written']
    assert not(N/'ssh166-route-v122.sock').exists()
    selected_indices={r['source_index']for r in raw['records']}
    old_records=[json.loads(line)for line in gzip.decompress((N/'MC2-V1-observer-source-v119/decoded-records.jsonl.gz').read_bytes()).splitlines()]
    selected_old=[r for r in old_records if r['index']in selected_indices]
    assert len(selected_old)==1392
    assert all(r['placement_strategy']=='linear'and r['redundant_experts']==0 and r['group_binding_complete']and r['group_ranks']==list(range(16))for r in selected_old)
    assert all(r['capacity_max_output_size']==65536 and r['receiver_count_sum']<=65536 and not r['receiver_count_sum_above_capacity_argument']for r in selected_old)
    root=W/'frozen-source-snapshot/vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_kernel'
    kernel=root/'dispatch_ffn_combine_kernel.hpp';header=root/'dispatch_ffn_combine.h'
    quant=root/'moe_init_routing_quant_v2/moe_v2_fullload_dynamic_quant.h'
    docs=[O/'route-fanout-saved-records-v122.json',O/'route-snapshot-source-index-v122.json',
        N/'route-sample166-v122.json',N/'route-fanout166-v122.json',N/'route-snapshot-source-v122/tensor-index.json',
        N/'route-snapshot-source-v122/sample.pt',N/'route-snapshot-source-v122/sample-metadata.json',
        N/'MC2-V1-observer-source-v119/decoded-records.jsonl.gz',N/'saved_tensor_archive_v122.py',N/'observer_draft_v3/wio_moe_observer.py',
        O/'MC2-receiver-work-and-communication-model-v56.json',O/'route-map-operator-source-contract-v96.json',
        O/'route-map-semantic-and-ceiling-contract-v96.json',O/'combine-v1-observer-peer-loads-v119.json']
    write(O/'route-fanout-inputs-v122.json',dict(inputs=[pin(p)for p in docs+[kernel,header,quant]],
        questions=['Howmanyrecordedtoken/destinationreferencesareactuallyduplicated?',
            'Doesametadata-completepacketreducebothdispatchandunchangedreturncontent?',
            'Whatquantizer/order/count/expansion/lifetimeconditionspreventfalsefusioncredits?'],
        previous_hitrate_question='OriginalobservercollectsMoEcounts/TopKids/ordinals/masks/maps,notEngramlookupIDs;noEngramcachehitrateestimated.',
        new_device_work=False,overall_complete=False))
    write(O/'route-fanout-source-contract-v122.json',dict(parts=[section(kernel,313,362),section(kernel,796,865),
        section(header,69,77),section(header,184,196),section(quant,222,309)],
        dispatch_direction='Currentreceiverpulls expert-expandedINT8recordsfromsenderpeermemviaCopyGMToGMPerToken;eachrecordstrideH+32andfirstFP32scaleisgatheredseparately.',
        input_row_semantics='SnapshotispostcallM×Ktopk_idsplusactualx_active_mask,notpre-maskrouteroutputorrequestID. v96frozennormaldomainpreservesactiveinputrowassociation;loadednative/alias/generationbindingremainsconditional.',
        source_quant_reuse='Full-load21000CopyOutXQuant1HcomputesonerowthenwritesitsoutLocaltomultipleexpandedpositionswithinthatpartition. Newfanouttransportdoesnotearnanewonce-per-tokenquantizationcredit. Otherkeys/partitionboundary/nativebitsremainexplicit.',
        source_smoothing='Selectedheaderinitializesscale/offsetnullandquantMode1. Noexpert-specificreplacementornewprecisionpolicyassumed;actualpackedbyteswerenotarchived.',
        assignment_preservation='Keeporiginalexpertassignmentcounts,expandedrow/inverse/probsmap,capacity/order and per-expertGMM/return. DistincttokendestinationDreplacesonlynewtransportpacketcounts,notGMMrowsA.',
        archived_placement_domain=dict(records=1392,placement='linear',redundant_experts=0,EP_group_members=list(range(16)),
            destination_rule='expert_global_id // E_local',recorded_capacity_argument=65536,max_recorded_receiver_assignment_rows=max(r['receiver_count_sum']for r in selected_old),
            capacity_clipping_in_recorded_count_model=False,
            limitation='Recorded metadata and locally retained sample owner-map support the selected contiguous logical-placement domain. All historical loaded maps/pointers/implementation and physical network topology are not independently attested.'),
        native_new_ABI_or_quantized_byte_equality_attested=False,overall_complete=False))

    cohorts={}
    for r in raw['records']:
        key=(r['role'],r['layer'],r['observer_ordinal']);cohorts.setdefault(key,[]).append(r)
        assert len(r['assignments_by_destination'])==len(r['token_destinations_by_destination'])==16
        assert sum(r['assignments_by_destination'])==r['K']*r['valid_recorded_rows']
    assert len(cohorts)==87 and all(len(v)==16 and {r['sender_EP_slot']for r in v}==set(range(16))for v in cohorts.values())
    groups=[]
    for role,g in raw['groups'].items():
        nc=sum(k[0]==role for k in cohorts);A=g['remote_assignments'];D=g['remote_token_destination_pairs']
        base=A*5152;ret=A*10240
        # Explicit32Bheader + six8B(localexpert,oldexpandedordinal)descriptors padded64B.
        data=D*5216
        # Chosenreplicatedcountplan: eachof16rankscontributes16uint32 Dcounts.
        extra_counts=nc*16*15*64
        new=data+extra_counts
        groups.append(dict(role=role,observer_cohorts=nc,rank_snapshots=g['records'],recorded_active_rows_across_layers_and_ranks=g['valid_recorded_rows'],
            A_remote=A,D_remote=D,structural_duplicate_references=A-D,
            existing_dispatch_record_payload_bytes=base,selected_fanout_packet_payload_bytes=data,
            extra_unique_D_count_matrix_content_bytes=nc*1024,
            extra_replicated_remote_D_count_content_bytes=extra_counts,
            selected_dispatch_and_extra_counts_bytes=new,unchanged_BF16_return_payload_bytes=ret,
            selected_dispatch_reduction_fraction=str(Fraction(base-new,base)),
            selected_dispatch_plus_return_reduction_fraction=str(Fraction(base-new,base+ret)),
            logical_dispatch_byte_delta_saved=base-new,
            if_no_groups_byte_equal_fallback_dispatch_data_bytes=A*5216,
            actual_network_bytes_or_latency_measured=False))
    t=next(g for g in groups if g['role']=='target');d=next(g for g in groups if g['role']=='draft')
    assert(t['A_remote'],t['D_remote'],t['logical_dispatch_byte_delta_saved'])==(2565204,2204823,1714345440)
    assert(d['A_remote'],d['D_remote'],d['logical_dispatch_byte_delta_saved'])==(731,727,-133440)
    layouts=[]
    for role,g in raw['groups'].items():
        K=6 if role=='target' else 3
        nc=sum(k[0]==role for k in cohorts);A=g['remote_assignments'];D=g['remote_token_destination_pairs'];base=A*5152;ret=A*10240
        for layout,stride in [('uniform_8B_descriptor_maxK6',5216),('K_specialized_8B_descriptor',5120+32+((8*K+31)//32)*32),
            ('packed_4B_descriptor_frame_epoch',5120+((16+4*K+31)//32)*32)]:
            for added in [0,nc*15360]:
                layouts.append(dict(role=role,K=K,layout=layout,record_stride=stride,
                    D_count_plan='replicatednewDmatrix'if added else'noincrementalpayload_onlyifexistingprotocolspaceandreadersproved',
                    extra_D_count_content_bytes=added,new_dispatch_reference_bytes=D*stride+added,
                    reduction_fraction=str(Fraction(base-D*stride-added,base)),
                    dispatch_plus_return_reduction_fraction=str(Fraction(base-D*stride-added,base+ret)),
                    zero_incremental_count_payload_does_not_mean_zero_compute_protocol_or_latency=True,
                    actual_protocol_packing_or_gain_attested=False))
    write(O/'route-fanout-packet-and-return-models-v122.json',dict(groups=groups,layout_and_count_plan_models=layouts,
        H=5120,existing_record_stride=5152,selected_packet_stride=5216,
        layout='INT8codesH +32Bheader(scaleFP32,source-rowI32,epochU64,fanoutcountU32,flagsU32,reserved8) +64Bdescriptorarea(up to6pairs(localExpertI32,exactoldExpandedRowOrdinalI32)). Selectedrawlayout,notminimalorinstalledABI.',
        descriptor_purpose='Preserveoldrow/expertplacementandmappingratherthanresortingbynewpacketarrival. Deriving/publishingtheexactordinalremainsnewworkandaregisteredABIgate.',
        extra_count_plan='KeepoldexpertassignmentAcounts;addD[sender,dest]. Model16uint32localcounts/rank=64B,globalmatrix1024B/cohortand16*15*64=15360Breplicatedremotereceivedcontent/cohort. CanbeintegratedinoldcountprotocolonlywithnewABI;notphysicalwirebytesoraforcednewcollectivelaunch.',
        earlier_reader_Hplus64_reference='ReaderretainsanoptimisticpackeddescriptorpayloadreferenceD*5184. Maincomparisonhereusesunpacked8Bexactordinaldescriptorsand5216Bfixedrecordspluscountpublication;neitherisactualwiremeasurement.',
        compact_header_alternative='Explicitselecteddomain:localexpert<256,oldexpandedordinal<2^24,sourceRow<2^32,K<=6. Packdescriptor(localexpert8bits,oldordinal24bits);baseheader16B(scale rawbits4,sourceRow4,fanout1,flags1,reserved6),epochboundbyexternalcallcontext. Headeralign32(16+4K) givesH+64forK6,H+32forK3. Thisrequiresnewexactmap/frame-epochABI;itdoesnotproveexistingpaddingavailable.',
        byte_equality_scope='Dcounts structuralreferencereuse,notobservedpackedpayloadidentity. Byteequalityofrequiredoriginalquantizedcode/scale andallpaddingreaders mustbeprovenorchecked;otherwiseeffectivepacketcountD_effcanrangeDtoA.',
        no_run_total='Sumsbelongtothese87savedcohorts/1392snapshots. TensorListcapturereplayscanreusefilenames;neithersnapshotmultiplicitynorobserverordinalisfullC8frequency/requestlatency.',
        no_return_or_compute_credit='Expertoutputsareweight-dependent. CurrentBF16returnperassignment,weightedunpermute,GMMFLOPs/activeweightreadsandv119V1returnrowworkretainA,notD.',
        theory_state='CONDITIONAL_CEILING',physical_speedup_MFU_or_whole_model_ceiling_proved=False,overall_complete=False))

    rows=[]
    for key,rs in cohorts.items():
        A=sum(r['remote_assignments']for r in rs);D=sum(r['remote_token_destination_pairs']for r in rs);base_current=A*5152;new=D*5216+15360
        rows.append(dict(role=key[0],layer=key[1],observer_ordinal=key[2],A_remote=A,D_remote=D,
            existing_dispatch_bytes=base_current,selected_dispatch_count_bytes=new,selected_byte_delta=base_current-new,
            layout_reference_only=True,no_later_TASK_join=True))
    write(O/'route-fanout-cohort-evidence-v122.json',dict(cohorts=rows,
        source_dataset=dict(run=raw['run_id'],records=1392,formal_cohorts=87,read_saved_bytes=raw['read_bytes'],archive_reader_elapsed_wall_seconds=raw['elapsed_CPU_process_seconds'],
            elapsed_field_interpretation='Legacy worker key elapsed_CPU_process_seconds uses time.monotonic(); it is elapsed wall time of the CPU-only archive reader, not measured CPU time.',
            SHA_checked_archive_and_metadata_files=2784,torch_imported=False,new_server_files_written=False),
        actual_data_fact='Recordedrowsareactive-maskfilteredandallper-experthistograms,receivercountsandordinals matchd6e787...decodedrecords. ActiveTopKexpertduplicates=0. Thisprovesrecordedrowfanout,notphysicaltraffic.',
        no_Engram_hitrate_or_later_profile_attribution=True,overall_complete=False))
    write(O/'route-fanout-variants-and-legality-v122.json',dict(
        packed_equality_counterexample=dict(same_recorded_source_row=0,same_destination=1,
            old_assignment_payloads_hex=['017f0000803f','027f0000803f'],
            conclusion='Identicaltoken/destinationlabelsalonedonotprovepackedbytesequality. Blindrepresentativereusechangesassignment2inthisallowedcounterexample;bitwisegroupingorfrozenproducerproofisrequired. Notanobservednativebug.'),
        variants=[dict(id='F21-TOKEN-DEST-PACK',graph='PostmasknativeIDs/oldroutingmap→per-rowdestinationfanout→onebyte-equalpacketperdest→newDcountpublish/pull→receiverreexpandoldrows/scales→unchangedGMM/return/unpermute.',
            legality='Alloriginalrow/scale/prob/epochidentitiesandFP16/BF16/INT8boundaries unchanged. NoTopKscore-sort/MC2expert-sortCSEorassignmentpruningcredit.',
            work='AtmostK6destinations/groupingperactiverow;chargegrouping/prefix/counts/pack,metadata,sourceandreceiverbuffers,expansionandresourcecompetition. K3draftprofilemakesselectedpayloadworseevenwithstructuralduplicate4.'),
            dict(id='F21-BYTE-CERTIFIED-PACK',graph='Compareoriginalpackedrecordsforeachproposedgroup,compressonlybyte-identicalrequiredcontents,elseemitseparatepreservedrecords.',
                legality='Bitwisecomparisoncanretainexactpayloadwithoutexpert-quantizationequivalenceassumption,butmetadata/rowmap/protocolandallreadersstillrequired.',
                cost='Comparisons/sourceGMreadsandstorageareextra;withretainedrepresentativesonchiponefullrecordreadperassignmentisapossiblesource-workmodel,notfreebandwidth. D_effnotobservedandworstcaseA,so5216*Ddoesnotapplyunconditionally.'),
            dict(id='F21-INDIRECT-INPUT',graph='ReceiverusesperexpertrowindirectiontosharedreceivedpacketsinsteadofreexpandingdenseGMinputs.',
                legality='NewGMMiterator/L1gather,exactroworder/scales/tileprecision/capacityandreadinessrequired;currentCATLASSdenseLayoutAdoesnotalreadyprovideit.',
                cost='Receiverdenseexpansionmaybedisplacedbyindirect/stagingwork,notfreeeliminated. Nocomputeorweightfactorassumed.')],
        required=['Noexpert-specificdifferenceinrepresentativepackedcodes/scalebitsunlessbyte-checkfallbackused;preserveoldunobservedpaddingreaders.',
            'UseeffectivepostmasknativeIDs/configepoch,retainexpertassignmentcountsandexactoldexpanded/inverse/probsmap;do notreplaceAwithDincapacity/GMM/return.',
            'CopyFP32scalepayloadasrawbitsalongwithINT8codes;do notrematerializeNaN/signedzero/scaleswhilepacking. Packedheaderindexrangesandexternalframeepochmustbeproved.',
            'NewcountDmatrix,packetoffsets,bounds,allrankparticipationandcompletioneventsagreed;localempty/zeroqueryranksmayreceivepositivework.',
            'Sourceoldsendbankmaystillbemandatory;newcompressedbankcanincreasepeakstorageuntiloldreadersretire. RetainA/C2alias/capturebank/queuedwriter/remote-pullcompletionepochs.',
            'No return aggregation or GMM reduction from input sharing;other proposed return/numeric changes need separate contracts.',
            'Countsarefromobserver-perturbedsavedsubsample,notclock-linkedtolaterhostprof. AllninehistoricalMemSet/BOOLgapsremain.'],
        source_reference_bytes='CurrentreceiverreferenceisA*(H+32). ChosennewpacketreferenceisD_eff*(H+96)+selectedcountpublication. Itisnotuniversalinformation/wirelowerbound:inputresidentstate,compression,multicast/otherrepresentationscanchangeit.',
        resource_bound='ForonechosenlegalrepresentationderiveactualnecessaryQ_by_resourceandtypedupperU,thenT>=max(Q/U,validcausalcuts). Donotadddispatch/returntimingsorconvertpayloadratiosintoMC2speedup.',
        new_device_work=False,new_native_promotions=0,no_whole_scope_closure=True,overall_complete=False))
    write(O/'route-fanout-local-graph-map-v122.json',dict(id='MC2_TOKEN_DEST_DISPATCH_V122',
        status='CONDITIONAL_CANDIDATE_SUBGRAPH_NOT_OBSERVED_RUNTIME',
        old_subnodes=['effective_native_IDs','original_routing_quant_and_A_counts','receiver_assignment_record_pull','GMM1_SwiGLU','GMM2_return','weighted_unpermute'],
        new_nodes=['effective_native_IDs','retained_probability_input','original_routing_quant_and_A_counts','token_destination_groups','original_map_fanout_descriptors',
            'byte_identity_proof_or_check','compressed_packet_stage','D_counts_publication','unique_packet_pull','old_row_scale_reexpansion','GMM1_SwiGLU','GMM2_return','weighted_unpermute'],
        edges=[dict(a=a,b=b,value=v)for a,b,v in [
            ('effective_native_IDs','original_routing_quant_and_A_counts','unchangedpostmaskIDs/mask/epoch'),
            ('effective_native_IDs','token_destination_groups','recordedrowanddestinationlabels'),
            ('original_routing_quant_and_A_counts','original_map_fanout_descriptors','exactoldexpanded/inversemapandAcounts'),
            ('token_destination_groups','original_map_fanout_descriptors','fanoutmembership'),
            ('original_routing_quant_and_A_counts','byte_identity_proof_or_check','oldINT8/scalerecordbits'),
            ('byte_identity_proof_or_check','compressed_packet_stage','certifiedidenticalpayloads'),
            ('original_map_fanout_descriptors','compressed_packet_stage','expert/oldrowordinal/probassociation'),
            ('token_destination_groups','D_counts_publication','newpacketcounts/offsets,notexpertrows'),
            ('compressed_packet_stage','unique_packet_pull','validpacketcontentandgeneration'),
            ('D_counts_publication','unique_packet_pull','count/offsetbounds'),
            ('unique_packet_pull','old_row_scale_reexpansion','allassignmentdescriptorsandcodes/scales'),
            ('old_row_scale_reexpansion','GMM1_SwiGLU','sameolddenserows/scaleindices/expertcounts/readiness'),
            ('original_routing_quant_and_A_counts','GMM1_SwiGLU','originalAexpertcounts/capacityprefix,notD'),
            ('GMM1_SwiGLU','GMM2_return','oldINT8activation/FP32scalesandFP16rounds'),
            ('GMM2_return','weighted_unpermute','oldBF16assignmentoutputs'),
            ('original_routing_quant_and_A_counts','weighted_unpermute','originalinverse/expandedrowmap'),
            ('retained_probability_input','weighted_unpermute','unchangedprobsbytes/epoch')]],
        retained_external_inputs='Originalinputactivation/mask/configepoch,expertweights/dequantscales/probs,allcache/capture/peerworkspacesandcountprotocolparameters remainunderoldcontracts.',
        mapping='Onlyoldassignment-recordtransportisreplaced;newpack/count/expandareadditionalwork. GMM/return/unpermuteandallassignmentidentitiesretained.',
        readiness='Useactualpacketandrequiredrow/tilecompletion,notinventedwhole-layerbarriers. Retainoldinterexpertflag/alias/crossrankcontractsorprovideaprovednewprotocol.',
        storage='Oldsourcequantbankandnewcompressedstagecanoverlap;retireonlyafterallremote-pull/readersandqueuedwriterscomplete. Global813nodecatalogisstillapartialseparateindex;thislocalvariantislinkedviaF21evidence,notnewobservednodes.',
        new_observed_native_or_global_catalog_nodes=0,overall_complete=False))
    write(O/'route-fanout-theory-and-measurement-v122.json',dict(
        measured_scope='Archived postmask routing IDs/masks/counts only; neither runtime packet bytes, packed code/scale identity nor MC2 internal phase latency is measured.',
        measured_recorded_row_counts=[dict(role=g['role'],A_remote=g['A_remote'],D_remote=g['D_remote'],structural_repeated_references=g['structural_duplicate_references'])for g in groups],
        selected_implementation_domain='Declared token/destination packet transport with unchanged assignment work, GMM, return and weighted unpermute. It is a candidate implementation domain, not a lower bound over arbitrary legal MoE algorithms.',
        conditional_payload_comparison='If all proposed groups have identical required old quantized bytes and the selected packet/count ABI is legal, compare A*(H+32) against D*selected_stride+selected_extra_count_content. These are logical record-content references, not physical wire/HBM necessities.',
        symbolic_timing='After proving actual resource cuts and necessary physical work W_r for a chosen domain, T >= max_r(W_r / C_r_upper). Shared-resource demands may sum only without overlap/double counting; logical payload totals here are not substituted for W_r without a transport/initial-state proof.',
        additional_work=['Grouping and exact old-ordinal descriptors', 'Byte-identity proof or explicit comparison with fallback', 'D-count/offset publication and peer readiness', 'Packet staging and receiver reexpansion or alternative indirect-GMM iterator', 'Capture-bank lifetime and possible overlapping old/new peak storage'],
        physical_cut_work_and_capacity_upper_bounds_bound=False,matching_internal_MC2_timing_bound=False,
        T_lower_numeric=None,throughput_upper_numeric=None,measured_to_ceiling_ratio=None,
        theory_state='SYMBOLIC_CEILING',payload_scenario_state='CONDITIONAL_REFERENCE',
        actual_latency_speedup_or_MFU_proved=False,new_native_promotions=0,overall_complete=False))
    print(json.dumps(dict(status='ANALYZED',archived_snapshots=1392,formal_cohorts=87,
        target_selected_dispatch_reference_fraction_saved=float(Fraction(t['selected_dispatch_reduction_fraction'])),
        target_selected_dispatch_plus_return_reference_fraction_saved=float(Fraction(t['selected_dispatch_plus_return_reduction_fraction'])),
        draft_selected_dispatch_reference_fraction_saved=float(Fraction(d['selected_dispatch_reduction_fraction'])),
        new_device_work=False,actual_speedup_proved=False)))

if __name__=='__main__':main()
