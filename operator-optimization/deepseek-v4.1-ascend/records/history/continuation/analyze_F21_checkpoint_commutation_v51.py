"""Derive checkpoint-map compatibility without reconstructing absent historical tensor values."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

from finalize_origin_coverage_v9 import N, O, read, write
from finalize_Engram_whole_frontier_v30 import graph, packet


def main():
    assert not (O / 'REVIEW-CHECKPOINT-v51.json').exists()
    for a in read(O / 'checkpoint-manifest-v50.json')['artifacts']:
        b = Path(a['path']).read_bytes()
        assert len(b) == a['bytes'] and hashlib.sha256(b).hexdigest() == a['sha256'], a['path']
    now = datetime.now(timezone.utc).isoformat()
    main_source = read(O / 'F21-checkpoint-routing-source-v51.json')
    bridge = read(O / 'F21-mapping-import-bridge-source-v51.json')
    source_files = main_source['files'] + bridge['files']
    for f in source_files:
        assert hashlib.sha256(Path(f['frozen_path']).read_bytes()).hexdigest() == f['sha256']
    assert len(source_files) == 5 and sum(f['historical166_hash_matched'] for f in source_files) == 3
    parts = {p['name']: p for f in main_source['files'] for p in f['parts']}
    query = '\n'.join(parts['AscendDeepseekV41LLMForCausalLM.get_expert_mapping']['source'])
    actual = '\n'.join(parts['AscendDeepseekV41LLMForCausalLM._load_model_weights']['source'])
    assert 'num_redundant_experts=0' in query
    assert 'num_redundant_experts=self.num_redundant_experts' in actual
    assert 'expert_id=expert_id' in actual and 'return_success=True' in actual
    assert 'if success:' in actual and 'break' in actual
    core_mapping = '\n'.join(parts['RoutedExperts.build_expert_params_mapping']['source'])
    assert 'EplbState.build_initial_global_physical_to_logical_map(' in core_mapping
    assert 'physical_to_logical_map[expert_id]' in core_mapping
    callback = packet('EP-map-contract166-v19', False)
    ascend = packet('EP-map-callees166-v19', False)
    builders = packet('EP-map-builders166-v19', False)
    initializer = packet('EP-map-initializer166-v19', False)
    callback_excerpt = next(e for f in callback['result']['files'] if f['path'].endswith('/fused_moe/routed_experts.py')
                            for e in f['excerpts'] if e['start'] == 575)
    assert 'expert_id = self._map_global_expert_id_to_local_expert_id(global_expert_id)' in '\n'.join(callback_excerpt['lines'])
    selector = next(e for f in ascend['result']['files'] if f['path'].endswith('/routed_experts.py')
                    for e in f['excerpts'] if e['start'] == 638)
    assert 'topk_ids = self.log2phy[topk_ids]' in '\n'.join(selector['lines'])
    # Retain the initial P0 function as an opaque source result. Its body and
    # historical/current table values have NOT been recovered by reading its call.
    p0_scope = dict(producer='EplbState.build_initial_global_physical_to_logical_map(E,R)',
                    body_newly_read=False, actual_return_values_known=False,
                    scope='P0 is symbolic;do not replace it with desiredplacementorassumecurrenttablecontents.')
    refs = ['F21-checkpoint-routing-source-v51.json', 'F21-mapping-import-bridge-source-v51.json',
            'EP-mapping-and-weight-contract-v19.json', 'F21-static-dynamic-loading-contract-v50.json']
    provenance = []
    for f in source_files:
        provenance.append(dict(path=f['path'], frozen_path=f['frozen_path'], sha256=f['sha256'],
                               historical166_hash_matched=f['historical166_hash_matched']))
    manifests = [read(p) for p in O.glob('checkpoint-manifest-v*.json')]
    reused_inputs = []
    for label in ['EP-map-contract166-v19', 'EP-map-callees166-v19', 'EP-map-builders166-v19', 'EP-map-initializer166-v19']:
        p = N / (label + '.json');sha = hashlib.sha256(p.read_bytes()).hexdigest()
        assert any(a['path'] == str(p) and a['sha256'] == sha for m in manifests for a in m.get('artifacts', [])), label
        reused_inputs.append(dict(path=str(p), sha256=sha, historical_manifest_match=True))
    write(O / 'F21-checkpoint-source-index-v51.json', dict(
        generated_utc=now, files=provenance, reused_receipts=reused_inputs,
        source_hashes_are_not_actual_tensor_or_callback_identity=True,
        overall_complete=False))
    facts = [
        ('entry_and_stream', 'HISTORICAL166_HASH_MATCHED_SOURCE',
         'Thismodelclassload_weights invokes its private _load_model_weights,not genericRoutedExperts.load_weights. NoEngrampath filters .engram.;Engrampath usesmilestonefilter. Theclassmarkersareempty;prefixfiltercoversaligner/vision/image_/mtp. Targetprivatebodyseparatelyskipsspec-layerrecords andPP-missingparameters. Draftclasses/otherloadersarenotprovedbythisentry.'),
        ('metadata_vs_actual_load', 'HISTORICAL166_HASH_MATCHED_SOURCE',
         'get_expert_mapping1785-1796 passesR=0. Actualprivate_load1816-1823 passesR=self.num_redundant_experts. Bothincludeconditionalsharedlogicalexperts. Do notusethequerymethodaloneasactualreplicaloadcoverageproof;actualRvalueisnotrecoveredhere.'),
        ('helper_and_tuple_namespaces', 'HISTORICAL166_SOURCE_PLUS_QUALIFIED_REFERENCE_BRIDGE',
         'Frozenreferenceimportreexports layerhelper whichdelegatesRoutedExperts.make/buildmapping. Corebuilder namescheckpointrecordswithlogicalID P0[p] while tupleexpert_idispglobalphysicalID. ItconstructsP0fromE,R;noexplicitdesiredplacementargumentexistsonthissignature. P0body/actualvaluesandactualbridgebindingsremainunattested.'),
        ('target_param_callback', 'HISTORICAL166_HASH_MATCHED_SOURCE',
         'Private loader normalizesnames,findsmatchingtuple,paramandshard,then callsparam.weight_loader(...expert_id=p,return_success=True). Itbreaksafterthefirstsuccessfulmatchingtupleforachunk. Callbackobjectidentityandallquant-specificoverridesarenotmeasuredbythissourceinspection.'),
        ('ordinary_base_callback', 'HISTORICAL166_HASH_MATCHED_SOURCE_REUSED',
         'Inordinarybaseweight_loader597-608,pislookedupviaExpertMapManager.map_global_to_local. Managerusesitscanonical_expert_map[p],oridentitywhenNone. Nonlocal-1returnswithoutloadingunlessglobal-scaleexceptionapplies. ThisisaseparatemapfromAscendexecutionlogicalplacement;GPT-OSSspecialbypass/globalSFarenotcoveredbytheordinarycase.'),
        ('execution_composition', 'HISTORICAL166_HASH_MATCHED_SOURCE_REUSED',
         'Ascendinitializerbuildsplacementrows,logical-to-localmapsandlog2phy;selector638-691applieslog2phybeforefurtheroptionalshared/force-routingbranches. Thecorrectnessconditionmustusethecompletecurrentrouteincludingreplicachoice,nottreatonequery/executionmapasallofthem.'),
        ('replica_and_shared_load_boundaries', 'HISTORICAL166_HASH_MATCHED_SOURCE',
         'Alogicalrecordmaymatchseveralphysicalreplicatuples,butthetargetchunkstopsonthefirstsuccess. Multiplelocalreplicasthereforeneedcompatiblefanoutorproofofunique-local-success;loaded_params isbookkeeping,notper-replica/per-planevalueproof. SharedfusionrewritesappendedlogicalIDsandsplitsgate/uprows,down_proj.weightcolumns;otherquantmetadatanamesfollowtheirowncurrentrules.'),
        ('alternative_generic_interface', 'HISTORICAL166_HASH_MATCHED_SOURCE',
         'GenericRoutedExperts.load_weights isadifferententrywithinclude_fused=True. Somefusedmappingheadersreuseexpert_idasgate/updeconcatindex,thenenumerateexpertrows. SuchheaderIDsarenotuniformphysicalplacementkeys. Do notapplygeneric3Dheaderinterpretationtotheactualmanualtargetentryorviceversa.'),
        ('reload_namespace', 'DERIVED_CONDITIONAL_FROM_REUSED_SOURCE',
         'Ascendlegacyupdate_expert_map(new)assignsboth_ascend_expert_mapandmanager._expert_map. Ifsuchanupdateisusedbeforeareload,thephysical-IDlookupmustberevalidated/restoredorrebuilt;anexecutionlogicalmapisnotautomaticallyavalidphysical-loadingmap,especiallywithR>0. Noactualreloadorcalltothissetterasserted.'),
        ('source_scope_limit', 'UNKNOWN_ACTUAL_STATE',
         'NoactualP0/managerPhi/customG/replicaroute/allweightplanes,currentcallbackobject,postloadexternalhook,loadedbinaryorstreamgenerationvaluesarerecovered. Defaultloaderfilter-noopremainsafilteredgeandcannotaloneproveweight-slotreconciliation. Conditional source analysis doesnotrequireanimplementation/nativebenchmark,butactualeligibilityclaimdoes.'),
    ]
    contract = dict(generated_utc=now, evidence=refs, initial_map_producer_scope=p0_scope,
        findings=[dict(id='CK51-%02d' % (i + 1), topic=t, grade=g, finding=s) for i, (t, g, s) in enumerate(facts)],
        locally_answered=['QueryR=0versusactualRentry','LogicalcheckpointnameversusphysicalcallbackID',
                          'Canonicalmanagerlookupversusexecutioncompositeroute','Firstsuccess/replicacoverageandgenericentrydistinctions'],
        actual_param_callback_binding_proved=False, actual_static_load_execution_pairing_proved=False,
        all_loader_or_quantization_branches_analyzed=False, target_changed=False,
        no_model_import_remote_or_device_work=True, whole_module_closed=False, overall_complete=False)
    write(O / 'F21-checkpoint-map-commutation-contract-v51.json', contract)

    # Source-level counterexample:all records are available,but the old physical
    # mapper stillrejects a record on itsnew desiredrank. No model/kernel is run.
    p0 = [0, 1, 2, 3]
    phi = [{0: 0, 1: 1}, {2: 0, 3: 1}]
    desired = [[2, 1], [0, 3]]
    old_payload = [[0, 1], [2, 3]]
    routed_old_payload = {}
    for rank, row in enumerate(desired):
        for slot, logical in enumerate(row):routed_old_payload[logical] = old_payload[rank][slot]
    assert routed_old_payload[0] == 2 and routed_old_payload[2] == 0
    needed = [set(row) for row in desired]
    default_needed = [set(row) for row in old_payload]
    assert needed[0] - default_needed[0] == {2}
    rejected_even_if_record_read = phi[0].get(2, -1)
    assert rejected_even_if_record_read == -1
    desired_pi = [e for row in desired for e in row]
    rebuilt_payload = []
    for rank in range(2):
        row = [None, None]
        for physical, slot in phi[rank].items():row[slot] = desired_pi[physical]
        rebuilt_payload.append(row)
    assert rebuilt_payload == desired
    theorem = dict(generated_utc=now,
        evidence=['F21-checkpoint-map-commutation-contract-v51.json'],
        grade='DERIVED_SYMBOLIC_CONDITION_AND_DECLARED_TOY_MAPPING;NOT_CURRENT166_TABLE_VALUES',
        notation=dict(P0='physicalIDp->checkpointlogicalexpertID', Phi_r='loadingphysicalIDp->rankrlocalslots,or-1',
                      B_r='postloadedpayloadbank', Route='completeexecutionselectionlogicale->(rankr,slots),includingreplicas'),
        requirement='For every actualexecutionassignment(e,r,s),B_r[s]mustimplementthesamelogicalexpertE_eundertherequirednumericpolicy.',
        sufficient_map_condition='IfloadedpayloadB_r[Phi_r(p)]=E_P0[p]andallplanes/postloadgenerationsarecoherent,thenrequireP0[p]=eforanyphysicalpwithPhi_r(p)=sselectedbyRoute(e). Thisisnotprovedfrommapnames/shapeoravailablefiles.',
        new_load_plan='For desiredbankG[r][s]=e,choosecheckpointnameforphysicalpfromPi_star[p]=G[r][Phi_r(p)]andmaterializeallreplicapayloadsbeforetheexecutionroutepublishesit. ThepresenthelperdoesnotacceptPi_starexplicitly;anactualhook/newinterfaceorverifiedexistingreconciliationisrequired.',
        witness=dict(configuration='Toy4logicalexperts,2ranks,2slots/rank,R=0;notfixedC8/EP16measurement',
            P0=p0, loading_Phi=phi, old_payload_labels=old_payload, desired_payload_labels=desired,
            execution_obtains_from_old_bank=routed_old_payload,
            rank0_logical2_raw_record_available=True, rank0_ordinary_callback_slot_for_physical2=rejected_even_if_record_read,
            correctly_rebuilt_Pi_star=desired_pi, direct_load_payload_labels=rebuilt_payload,
            inference='DisablingfilepruningdoesnotalterP0orPhi:weight2maybereadbutstillrejectedonrank0. Eitherloadingandpayloadplacementorcompositeexecutionmappingmustbechangedcoherently.'),
        filter_condition='Neededlogicalrecordsderivefromtheactualload/replicaplanandallparameterconsumers,atfile/tensorgranularity. R=0metadataqueryoroldlinearownershipisnotageneralnewplacementfilter.',
        exclusions=['Thisprovesnoactualcustommapfailureoractualallrecordsread;P0/Phi/Gherearedeclaredhypotheticalinputs.',
                    'Expertpayloadlabelsabstractallweights/scales/postloadfunctionanddonotproveW8A8nativebitwiseparity.',
                    'Forcedrouting,sharedbranches,quantglobal-scaleexceptions,LoRA/capture/reloadmustretaintheirindividualcontracts.'],
        target_changed=False, implementation=False, native_test=False, critical_path_gain_us=None,
        whole_module_closed=False, overall_complete=False)
    write(O / 'F21-static-load-identity-witness-v51.json', theorem)
    nodes = ['checkpoint_stream', 'target_filter_and_name_rules', 'actual_target_private_loader',
             'actual_E_R_mapping_call', 'qualified_reference_helper_bridge', 'P0_constructor_opaque',
             'logical_name_physical_p_tuples', 'matching_parameter_callback', 'ordinary_base_callback_case',
             'canonical_loading_Phi', 'reject_nonlocal_ordinary_case', 'quant_payload_slot_write_abstract',
             'complete_execution_route', 'load_execution_pairing_requirement',
             'metadata_query_R0_separate_call', 'generic_layer_3D_loader_alternative']
    edges = [
        ('checkpoint_stream', 'target_filter_and_name_rules', 'DOCUMENTED_MODEL_ENTRY'),
        ('target_filter_and_name_rules', 'actual_target_private_loader', 'DOCUMENTED_TARGET_LOAD_METHOD'),
        ('actual_target_private_loader', 'actual_E_R_mapping_call', 'DOCUMENTED_SELF_REDUNDANT_COUNT'),
        ('actual_E_R_mapping_call', 'qualified_reference_helper_bridge', 'REFERENCE_BRIDGE_IDENTITY_NOT_ATTESTED166'),
        ('qualified_reference_helper_bridge', 'P0_constructor_opaque', 'SOURCE_INITIALIZER_CALL;BODY_VALUES_UNKNOWN'),
        ('P0_constructor_opaque', 'logical_name_physical_p_tuples', 'DOCUMENTED_CORE_TUPLE_CONSTRUCTION'),
        ('logical_name_physical_p_tuples', 'matching_parameter_callback', 'DOCUMENTED_NAME_MATCH_AND_PHYSICAL_ID_ARG'),
        ('matching_parameter_callback', 'ordinary_base_callback_case', 'CONDITIONAL_ACTUAL_CALLBACK_BINDING_NOT_PROVED'),
        ('ordinary_base_callback_case', 'canonical_loading_Phi', 'DOCUMENTED_MANAGER_LOOKUP'),
        ('canonical_loading_Phi', 'reject_nonlocal_ordinary_case', 'CONDITIONAL_PHI_MINUS1_NO_GLOBAL_SF'),
        ('canonical_loading_Phi', 'quant_payload_slot_write_abstract', 'CONDITIONAL_LOCAL_NORMAL_QUANT_CASE'),
        ('quant_payload_slot_write_abstract', 'load_execution_pairing_requirement', 'REQUIRED_LOGICAL_EXPERT_PAYLOAD_IDENTITY'),
        ('complete_execution_route', 'load_execution_pairing_requirement', 'REQUIRED_FULL_ROUTE_NOT_SINGLE_MAP'),
    ]
    flow = graph(nodes, edges)
    flow.update(scope='Source-levelconditionalmappingflow;metadataqueryandgenericlayerloaderareseparateentrynodes,notprovenactualtargetwriter. NotactualnativeDAG/captureproof.',
                actual_pairing_proved=False, overall_complete=False)
    write(O / 'F21-static-loading-source-flow-v51.json', flow)
    conditions = [
        'Bindtheactualtarget/draftmodelclass,parametercallback,helperbridgeandP0initializer;do notsubstituteaqueryinterfaceorgeneric3Dloader.',
        'UseactualE/R/shared/PP/expertname/shard/base_layerprefixrules,andphysicalpversuslogicalIDnamespaces.',
        'ProvefullRoute(e)->B_r[s]=E_ecommutation;actualP0/Phi/G/replicaroutevaluesarenotgivenbysourcehash.',
        'Materializeeveryrequiredlocalreplica:currenttargetfirstsuccessbreakrequiresuniquelocal-successorcorrectfanout;global-scaleexceptionsdiffer.',
        'Preserveallpostloadquantplanes/formatandnumericcontract,loaded_coverageerrors,sharedsliceaxes,targetspecskipandper-layeridentity.',
        'Deriveplacement-awarefile/tensorfilterfromactualplan;defaultEPfilterdisabledisnotstaticreorder.',
        'Re-establishphysicalloadingmapandpayloadviewsafterEPLB/mapupdates/reload;logicalexecutionsettercanchangecanonicalmanagerfields.',
        'InitialloadI/O/RAM/formatgain,dynamicrebalanceandsteadyformalC8latencyhaveowndenominators;no gain inferredfromcount/witness.']
    write(O / 'F21-static-loader-conditions-v51.json', dict(
        generated_utc=now, candidate='F21', refines='F21-v50-STATIC-CHECKPOINT-PLACEMENT',
        conditions=[dict(id='SL51-%02d' % (i + 1), condition=s, state='REQUIRED_CONDITIONAL;ACTUAL_BINDING_NOT_PROVED') for i, s in enumerate(conditions)],
        analyzed_source_scope='Namedcheckpoint-name/physicalID/slotandquery-versusactualentryquestions answeredatqualifiedfrozencodelevel.',
        unresolved_source_or_runtime_edges=['Actualparam.weight_loader binding/overridesandP0initializerbody/currentreturnvalues.',
            'Original166importbridgebytesandglobalrebinding,selectedDefaultModelLoaderorotherloader/postloadhooks.',
            'Actualmap/replica/weight/scale/capture/nativegenerationsandallreaders/resourceevidence.'],
        new_main_candidates=0, implementation=False, native_test=False,
        whole_module_closed=False, overall_complete=False))
    old_branch = deepcopy(read(O / 'F21-interface-branch-ledger-v50.json'))
    b = next(row for row in old_branch['branches'] if row['id'] == 'F21-B8-postload-and-capture')
    b['finding'] += ' v51locatesactualtargetprivate_load(selfR)vsquery(R0),physicalp/logicalcheckpointnametupleandordinarymanagerPhi. Fullroute/loadpayloadcommutationisrequired;filter-noopdoesnotestablishit. Firstsuccessreplicacoverageandseparategeneric3Dloader/reloadnamespacesnowdocumented;actualcallback/P0/bridge/staticpairingremainopen.'
    b['evidence'] += ['F21-checkpoint-map-commutation-contract-v51.json', 'F21-static-load-identity-witness-v51.json']
    old_branch.update(generated_utc=now, parent=str(O / 'F21-interface-branch-ledger-v50.json'))
    write(O / 'F21-interface-branch-ledger-v51.json', old_branch)
    gate = read(O / 'PROFILE-CODE-QUESTION-GATE-v51.json')
    gate.update(status='CHECKPOINT_QUERY_ACTUAL_ENTRY_AND_ID_NAMESPACE_QUESTIONS_ANSWERED;ACTUAL_STATIC_PAIRING_OPEN',
        answered_by=['F21-checkpoint-map-commutation-contract-v51.json', 'F21-static-load-identity-witness-v51.json',
                     'F21-static-loader-conditions-v51.json'],
        closed_scope='Threehistorical166hash-matchedsources+tworeferencebridgeedges;actualtargetquery/loadRdistinctionandconditionalphysicalID→manager→slot/executioncompatibilitycontract.',
        not_closed='Actualparamcallbackbinding/P0bodyandvalues/bridgeidentity,postloadreconciliation/globalhooks,weights/maps/replicas/native/capture/lifetime/resourcesandwholemodulecoverage.',
        stop_rule='Do notrepeatqueryR0asactualloadmappingorclaimallfilesreadimpliescorrectplacement. Nexttraceonlyremainingcallback/P0/actualhookedgeoruseoriginalrawdatawhenavailable;whole goal remainsopen.')
    write(O / 'PROFILE-CODE-QUESTION-GATE-v51.json', gate)
    print(json.dumps(dict(frozen_source_inputs=5, historical166_hash_matches=3, reference_bridge_inputs=2,
        ordinary_loader_namespace='global_physical_ID', source_flow_nodes=len(nodes), source_flow_edges=len(edges),
        conditional_static_load_requirements=len(conditions), actual_static_pairing_proved=False,
        new_main_candidates=0, overall_complete=False)))


if __name__ == '__main__':
    main()
