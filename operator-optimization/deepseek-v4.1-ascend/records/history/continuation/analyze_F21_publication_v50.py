"""Analyze static/dynamic placement boundaries and conditional multiplane publication."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import ast
import hashlib
import json

from finalize_origin_coverage_v9 import N, O, read, write
from finalize_Engram_whole_frontier_v30 import graph


def main():
    assert not (O / 'REVIEW-CHECKPOINT-v50.json').exists()
    for a in read(O / 'checkpoint-manifest-v49.json')['artifacts']:
        b = Path(a['path']).read_bytes()
        assert len(b) == a['bytes'] and hashlib.sha256(b).hexdigest() == a['sha256'], a['path']
    now = datetime.now(timezone.utc).isoformat()
    entry = read(O / 'F21-postload-adaptor-entry-source-v50.json')
    chain = read(O / 'F21-postload-updater-transfer-source-v50.json')
    old = read(O / 'EP-mapping-and-weight-contract-v19.json')
    files = {f['path']: f for f in chain['files']}
    frozen_inputs = []
    source_root = Path('/Users/wio/work/910c-deepinfer-control/installed-vllm-source')
    for f in files.values():
        b = Path(f['path']).read_bytes()
        assert hashlib.sha256(b).hexdigest() == f['sha256']
        frozen = N / 'F21-source-snapshot-v50' / Path(f['path']).relative_to(source_root)
        frozen.parent.mkdir(parents=True, exist_ok=True)
        if frozen.exists():
            assert hashlib.sha256(frozen.read_bytes()).hexdigest() == f['sha256']
        else:
            frozen.write_bytes(b)
        frozen_inputs.append(dict(original_local_path=f['path'], frozen_path=str(frozen),
            sha256=f['sha256'], bytes=len(b), historical166_hash_attested=f['historical166dependency_hash_attested']))
    assert len(files) == 6
    parts = {p['name']: p for f in files.values() for p in f['parts']}
    methods = {p['name']: p for p in entry['functions']}
    warmup = '\n'.join(parts['eplb_warmup/EPLB_entry']['source'])
    assert 'if self.dynamic_eplb and not self.is_eplb_warmuped:' in warmup
    assert 'VllmEplbAdaptor(model=self.model)' in warmup
    disable_filter = '\n'.join(parts['load_model/if self.eplb_enable']['source'])
    assert 'DefaultModelLoader._init_ep_weight_filter = mock_pass' in disable_filter
    eligibility = [p for p in files[next(p for p in files if p.endswith('/worker/model_runner_v1.py'))]['parts']
                   if p['name'] == '__init__/EPLB_eligibility']
    assert any('self.dynamic_eplb or (eplb_config.expert_map_path is not None)' in '\n'.join(p['source']) for p in eligibility)
    # Decode only the literal registry;never import QuantType or the model module.
    raw = Path(entry['path']).read_bytes();tree = ast.parse(raw)
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and
                      any(isinstance(a, ast.Name) and a.id == 'EPLB_EXPERT_WEIGHT_NAMES' for a in n.targets))
    registry = []
    for k, v in zip(assignment.value.keys, assignment.value.values):
        registry.append(dict(quant_type=k.elts[0].attr, fused_mc2=k.elts[1].value,
                             planes=[a.value for a in v.elts]))
    w8 = next(r for r in registry if r['quant_type'] == 'W8A8' and r['fused_mc2'])
    assert w8['planes'] == ['w13_weight_list', 'w2_weight_list', 'w13_weight_scale_fp32_list',
                            'w2_weight_scale_list', 'fused_w1_scale_list', 'fused_w2_scale_list']
    refs = ['F21-postload-adaptor-entry-source-v50.json', 'F21-postload-updater-transfer-source-v50.json',
            'EP-mapping-and-weight-contract-v19.json', 'F21-interface-branch-ledger-v19.json']
    provenance = [dict(path=f['path'], sha256=f['sha256'],
                       historical166_hash_attested=f['historical166dependency_hash_attested']) for f in files.values()]
    assert sum(r['historical166_hash_attested'] for r in provenance) == 3
    write(O / 'F21-source-snapshot-index-v50.json', dict(
        generated_utc=now, files=frozen_inputs,
        scope='Exactly six source inputs frozen locally for reproducibility. Copying local dependency bytes doesnotturn them into a historical/current166attestation. No file uploaded.',
        whole_module_closed=False, overall_complete=False))
    write(O / 'F21-static-dynamic-loading-contract-v50.json', dict(
        generated_utc=now, evidence=refs, source_provenance=provenance,
        evidence_grades=['DOCUMENTED_HISTORICAL166_HASH_MATCHED_SOURCE', 'DOCUMENTED_LOCAL_DEPENDENCY_REFERENCE', 'DERIVED', 'UNKNOWN'],
        findings=[
            dict(id='PL50-01', grade='DOCUMENTED_HISTORICAL166_HASH_MATCHED_SOURCE',
                finding='Runner577-578setsdynamic_eplbfromconfigand eplb_enable=dynamicOR(pathisnotNone). Runner4354-4360createsadaptor/setsloader/updator/warmuponlyifdynamicandnotalreadywarmed. Astaticfilealone doesnotselectthisdynamicwarmup.'),
            dict(id='PL50-02', grade='DOCUMENTED_HISTORICAL166_HASH_MATCHED_SOURCE',
                finding='Within load_model4408-4412,eplb_enable replacesDefaultModelLoader._init_ep_weight_filter withanoopclassfunction. That is a filter-change edge,not an explicit physical-slot weight permutation or payload reconciliation. Actual selected loader and any other hooks remain separate.'),
            dict(id='PL50-03', grade='DOCUMENTED_HISTORICAL166_HASH_MATCHED_SOURCE',
                finding='RoutedExperts init_eplb preserves upstream physical map duringcheckpointloading while Ascendexecution usesascend_expert_map. get_eplb_parameter returnsnamedattribute. W8A8postload transposes/formats weights,createsFP32/packed scales,andmaycloneper-expertlists/deleteoriginals. Binding stale pre-postload views is not generally valid.'),
            dict(id='PL50-04', grade='DOCUMENTED_LOCAL_DEPENDENCY_REFERENCE',
                finding='Local adaptor uses a class-global registered-layer list andlocalenumeration,firstlayerlocalcapacity,quant/fusedkey registry,andexplicitper-planemutableviews. This doesnotattest actual166registeredtarget/draftorder or dependencyfilehash.'),
            dict(id='PL50-05', grade='DERIVED_CONDITIONAL',
                finding='Staticplacement eligibility needs a positive loader/remap proof that eachdesiredslot containsitslogicalexpert payload afterquantpostload. Dynamicadaptor existence or disabling a filefilter cannot substitute for that proof. No claim that the complete installation has no other reconciliation hook.'),
            dict(id='PL50-06', grade='DERIVED_CONDITIONAL',
                finding='Aplacement-aware checkpoint plan may load directlyintodesiredslots/replicas andfilteronlytheactualneededlogicalexperts. It must preserve checkpoint shard order/quantplanes,corephysicalmap,Ascendlogicalmap,log2phy,andtarget/draftlayeridentity. Startup-only saving is not formalC8decode saving.'),
        ],
        static_postload_weight_pairing_proved=False, actual_dynamic_or_custom_map_enabled='UNKNOWN_FROM_THIS_REVIEW',
        no_new_model_import_remote_or_device_work=True, implementation=False,
        whole_module_closed=False, overall_complete=False))

    update = '\n'.join(parts['D2DExpertWeightLoader.update_expert_map_and_weight']['source'])
    assert update.index('req.wait()') < update.index('do_update_expert_map') < update.index('do_update_log2phy_map') < update.index('do_update_expert_weight')
    nodes = [
        'registered_payload_views_v', 'shared_buffer_layout_by_key', 'old_CPU_map_v',
        'build_send_and_recv_tasks', 'post_P2P_requests', 'wait_requests_call',
        'copy_CPU_map_v1', 'copy_log2phy_v1', 'enqueue_payload_plane_copies_v1', 'set_WAITING_v1',
        'all_old_readers_finished', 'all_new_planes_native_ready', 'all_rank_map_generation_agreed', 'next_reader_v1']
    edges = [
        ('registered_payload_views_v', 'build_send_and_recv_tasks', 'SOURCE_SENDS_FROM_BOUND_OLD_VIEWS'),
        ('shared_buffer_layout_by_key', 'build_send_and_recv_tasks', 'SOURCE_RECEIVES_TO_BUFFER_BANKS'),
        ('old_CPU_map_v', 'build_send_and_recv_tasks', 'SOURCE_GLOBAL_TO_LOCAL_SEND_LOOKUP'),
        ('build_send_and_recv_tasks', 'post_P2P_requests', 'STATE_WAITING_TO_READY'),
        ('post_P2P_requests', 'wait_requests_call', 'STATE_READY_TO_TRANSFERRING'),
        ('wait_requests_call', 'copy_CPU_map_v1', 'SOURCE_CALL_ORDER_NOT_NATIVE_COMPLETION_PROOF'),
        ('copy_CPU_map_v1', 'copy_log2phy_v1', 'SOURCE_CALL_ORDER'),
        ('copy_log2phy_v1', 'enqueue_payload_plane_copies_v1', 'SOURCE_CALL_ORDER'),
        ('enqueue_payload_plane_copies_v1', 'set_WAITING_v1', 'SOURCE_CALL_ORDER_NOT_ASYNC_COMPLETION_PROOF'),
        ('all_old_readers_finished', 'enqueue_payload_plane_copies_v1', 'REQUIRED_LAST_READER_EDGE_UNPROVED'),
        ('enqueue_payload_plane_copies_v1', 'all_new_planes_native_ready', 'REQUIRED_NATIVE_READY_EDGE_UNPROVED'),
        ('all_new_planes_native_ready', 'next_reader_v1', 'REQUIRED_PUBLICATION_EDGE_UNPROVED'),
        ('all_rank_map_generation_agreed', 'next_reader_v1', 'REQUIRED_DISTRIBUTED_GENERATION_EDGE_UNPROVED'),
        ('set_WAITING_v1', 'next_reader_v1', 'CONDITIONAL_HOST_SCHEDULE_EDGE_NOT_NATIVE_FENCE'),
    ]
    frontier = graph(nodes, edges)
    frontier.update(scope='One unrolled generation of the local-reference dynamic updater;not the globalmodel DAG or an actual timestamp/native event certificate.',
                    native_publication_proved=False, overall_complete=False)
    write(O / 'F21-unrolled-dynamic-publication-frontier-v50.json', frontier)
    # A concrete ownership witness;no expert model arithmetic/device copy is run.
    before = {0: tuple('e0/' + p for p in w8['planes']), 1: tuple('e1/' + p for p in w8['planes'])}
    old_snapshot = deepcopy(before)
    staged = {0: old_snapshot[1], 1: old_snapshot[0]}
    naive = deepcopy(before)
    naive[0] = naive[1]
    naive[1] = naive[0]
    assert staged[0] != staged[1] and naive[0] == naive[1]
    publication = dict(
        generated_utc=now, evidence=refs,
        dependency_scope='Adaptor/updator/D2Dloader are local-reference hashes only;current166identity/instances notattested.',
        plane_registry=registry, W8A8_fused_MC2_named_planes=w8['planes'],
        buffer_model=dict(
            allocation_rule='C=firstrealMoElayerlocalcapacity;for each distinct(quant_type,MC2==1)key allocateCbuffer tuples,oneempty_likeper namedplane. Reused acrosslayerswiththesamekeyonlyafterequal per-expertshapes assert.',
            logical_bytes_expression='C * sum_over_distinct_keys(sum_over_named_planes(logical_bytes(single_expert_plane)))',
            not_multiplied_by_layer_count=True,
            exclusions=['Not a physicalmemory/nzpadding/allocator bound;residentnamedplanesmayaliasandneedseparatestorageaccounting.',
                        'Pipeliningtwo updates thatuseonesharedbankrequiresnewbankcapacityorlast-useordering;layer-independentbufferreuseisnotprovenfreeoverlap.',
                        'No actualC/shape/storage/occupancy values recovered by this review.']),
        source_update_protocol=[
            'WAITINGacceptstask;sendviewslookupoldCPUmap,receivesuseper-keybuffers;otherstatesrejectnewtask.',
            'READYpostsbatchP2Pifnonempty,thenTRANSFERRINGevenforemptytasklist.',
            'TRANSFERRINGcallsreq.waitforpostedrequests;thenCPUmapcopy,log2phycopy,per-receivedexpertpayloadcopies;finallyclearsandsetsWAITING.',
            'Hostmethodreturn/stateWAITINGaloneisnotan independently attested native completion barrier for another stream/captured reader.'],
        iteration_protocol=dict(
            heat='H=expert_heat_collection_interval', algorithm='A=algorithm_execution_interval', layers='L=num_moe_layers',
            trigger='iteration==H-1', get_update_info='iteration==H+A-1', update_window='0<=iteration-(H+A)<L',
            commit_guard='forward_end commits only when update_flag and expert_map_record_path is None',
            next_counter='increment at end when iteration>=H-1 or heat_collection_status;reset exactly atH+A+L',
            actual_H_A_L_or_flags='UNKNOWN;no observedEPLBepochjoin'),
        lifetime_conditions=[
            'Every logicalexpert assignment must use one coherent generation of weights,FP scales,packed scales and anyotherliveexpertattributes;the six-fieldregistryisnotall-backend/adapterconsumerclosure.',
            'Construct/rebuildviewsafterpostloadclone/format and afterreload;mapobjectidentityaloneisnotpayloadgenerationidentity.',
            'Keepoldpayloadandsendbuffersuntilallcurrentreaders/transfersfinish;keepreceivebuffersuntilallcopiesandfuturebankreadersareready.',
            'Allranksagreeonlayer/slot/logicalexpert/replica/MC2groupandgenerationbeforeusingnewroute;registryordinalisnotautomaticallymodel-layerorconstructor-ID.',
            'In-placecopy_ preservesdestinationbindingsbyintent;actualalias/captureaddresses/nativecompletionareseparateevidence. Map-firstsourcecallorderdoesnotproveaconcurrentraceoratomictransaction.'],
        copy_cycle_witness=dict(scope='PURE_LABEL_DATAFLOW_COUNTEREXAMPLE;NOT_MODEL_NUMERICS_OR_TARGET_TEST',
            desired='Swapexpert0and1,keepingallsixplanespaired.', before=before,
            naive_forward_overwrite=naive, staged_from_old_snapshot=staged,
            inference='Anynewpermutationimplementationmustprotectcyclesandallplanes;currentreferenceP2Preceivebuffersarepositiveevidenceofstaging,notgenericpermissiontoeliminateit.'),
        native_ready_last_reader_or_all_rank_transaction_proved=False,
        implementation=False, critical_path_gain_us=None, whole_module_closed=False, overall_complete=False)
    write(O / 'F21-multiplane-buffers-and-publication-contract-v50.json', publication)
    variants = [
        dict(id='F21-v50-STATIC-CHECKPOINT-PLACEMENT', phase='INITIAL_LOAD;NOT_AUTOMATICALLY_FORMAL_C8_LATENCY',
            state='CONDITIONAL_INITIAL_PLACEMENT_AND_FILTER_REORGANIZATION',
            change='Generateoneversioneddesiredslot/replica/loadplanbeforecheckpointingest;loadrequiredlogicalexpertsdirectlyintotargetslotsandretaina placement-aware file/tensor filterinsteadofblindlydisablingpruning.',
            requires=['Positiveactual-loaderbindingandcheckpoint-to-physical-to-Ascendmapcommutation,includingredundant/target/draftlayercases.',
                      'Allquantplanes/postloadformatviewsandfutureconsumerssamegeneration;no class-globalfilterrestorationorper-modelscopeassumedfromoneblock.',
                      'InitialI/O/RAM/format/copycost must be measured in itsownstartupdenominator;ifclaimingsteady-stategain,useactualrouting/communicationevidence separately.'],
            implementation=False, measured_gain=None),
        dict(id='F21-v50-DYNAMIC-COHERENT-PUBLICATION', phase='DYNAMIC_REBALANCE_ONLY_WHEN_ENABLED',
            state='CONDITIONAL_MULTIPLANE_TRANSFER_AND_GENERATION_REORGANIZATION',
            change='Plancompatiblewholeexpertpayloadtransfersandpublishmap/payloadgenerationwiththeprecisefencesandshared-bankcapacityrequiredbyallreaders;considerbatching/coalescing/pipeliningonlywiththoseconstraints.',
            requires=['Confirmoriginaladaptor/updator/loaderhashesandactualdynamicactivation,registrylayerorder,shape/layoutandcoherentall-planeownership.',
                      'Preservecycle-staging,P2Pmatching,requestcompletionsemantics,oldlastreader/newready/captureaddressesandrankgenerationagreement.',
                      'Sharedbufferbanksareperquantkey,notperlayer;concurrencyrequiresprovenbanklifetimesornewcapacity.'],
            implementation=False, measured_gain=None),
    ]
    write(O / 'F21-placement-publication-variants-v50.json', dict(
        generated_utc=now, parent_candidate='F21', variants=variants,
        source_scope=str(O / 'F21-static-dynamic-loading-contract-v50.json'),
        exclusions=['Staticmapaloneisnotproofdynamicadaptorperformedpostloadreconciliation.',
                    'Sixnamedplanesarenotsixindependentresidentallocationsoracompleteallbackendpayloadproof.',
                    'Do notpriceEPLBpolicy/transferorstartupworkfromunrelatedformalgreedyComputeInfoTASKs.'],
        new_main_candidates=0, implementation=False, whole_module_closed=False, overall_complete=False))
    branch = deepcopy(read(O / 'F21-interface-branch-ledger-v19.json'))
    b8 = next(b for b in branch['branches'] if b['id'] == 'F21-B8-postload-and-capture')
    b8.update(finding='v50separatesstatic/dynamicpaths. Historicalrunneronlycreatesadaptorunderdynamicwarmup;staticfileenablesanoopDefaultModelLoaderfilter,notanexplicitweight-slotreorderproof. Local-referenceupdatecopiesallregisteredquantplaneswithsharedper-keybanks;actualstaticreconciliationandnativebank/capturegenerationsremainopen.',
              evidence=['F21-static-dynamic-loading-contract-v50.json', 'F21-multiplane-buffers-and-publication-contract-v50.json'])
    branch.update(generated_utc=now, parent=str(O / 'F21-interface-branch-ledger-v19.json'),
                  source_dependency_identity_scope='Threehistorical166matchedfiles;threelocal-reference-onlydependencies.',
                  whole_module_closed=False, overall_complete=False)
    write(O / 'F21-interface-branch-ledger-v50.json', branch)
    gate = read(O / 'PROFILE-CODE-QUESTION-GATE-v50.json')
    gate.update(status='STATIC_DYNAMIC_BOUNDARY_AND_REFERENCE_PAYLOAD_PROTOCOL_ANALYZED;STATIC_ACTUAL_RECONCILIATION_OPEN',
        answered_by=['F21-static-dynamic-loading-contract-v50.json', 'F21-multiplane-buffers-and-publication-contract-v50.json',
                     'F21-placement-publication-variants-v50.json'],
        closed_scope='Exacthistoricalrunnerstatic-vs-dynamiceligibility/filteredges;conditionalreferenceadaptor/updaterplane/buffer/state/call-ordersemantics.',
        not_closed='Positivecustomstaticpostloadslot/weightproof,threeoriginaldependencyhashes,actualflags/maps/weights/epochs/allreaders/nativefences/resources/physicalEPtopology.',
        stop_rule='Do notrepeatgenericABIabsenceorclaimstaticfileauto-reconcilesviaadaptor. Nextinspectonlyactualstaticloader/remapedgeorattestmissingoriginaldependencieswhenaccessreturns;advanceotherfinitecoverageindependently.')
    write(O / 'PROFILE-CODE-QUESTION-GATE-v50.json', gate)
    print(json.dumps(dict(source_files=6, historical166_hash_matched_files=3,
        local_reference_only_dependencies=3, W8A8_MC2_named_planes=6,
        frontier_nodes=len(frontier['nodes']), frontier_edges=len(frontier['edges']),
        conditional_variants=2, static_reconciliation_proved=False, native_publication_proved=False,
        whole_modules_closed=0, overall_complete=False)))


if __name__ == '__main__':
    main()
