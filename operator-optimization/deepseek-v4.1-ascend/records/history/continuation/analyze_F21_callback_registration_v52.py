"""Analyze exact callback forwarding and distinguish initial parameters from runtime payloads."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import ast
import hashlib
import json

from finalize_origin_coverage_v9 import N, O, read, write
from finalize_Engram_whole_frontier_v30 import graph, packet


def main():
    assert not (O / 'REVIEW-CHECKPOINT-v52.json').exists()
    for a in read(O / 'checkpoint-manifest-v51.json')['artifacts']:
        b = Path(a['path']).read_bytes()
        assert len(b) == a['bytes'] and hashlib.sha256(b).hexdigest() == a['sha256'], a['path']
    now = datetime.now(timezone.utc).isoformat()
    source = read(O / 'F21-callback-registration-source-v52.json')
    by_role = {f['role']: f for f in source['files']}
    assert len(by_role) == 5 and sum(f['historical166_hash_matched'] for f in source['files']) == 3
    for f in source['files']:
        assert hashlib.sha256(Path(f['frozen_path']).read_bytes()).hexdigest() == f['sha256']
    core = by_role['CORE_INIT_CALLBACK']
    assert [p['start'] for p in core['parts']] == [123, 159, 171]
    inheritance = by_role['ASCEND_INHERITANCE']['metadata']
    assert inheritance['bases'] == ['RoutedExperts']
    assert not inheritance['own_weight_loader_defined'] and not inheritance['own_global_to_local_override']
    config = by_role['MODELSLIM_MOE_SELECTION']
    tree = ast.parse(Path(config['frozen_path']).read_bytes())
    outer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_make_modelslim_moe_weight_loader')
    inner = next(n for n in outer.body if isinstance(n, ast.FunctionDef))
    call = next(n.value for n in inner.body if isinstance(n, ast.Return))
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'weight_loader'
    forwarding = {k.arg: ast.unparse(k.value) for k in call.keywords}
    assert forwarding == dict(param='param', loaded_weight='loaded_weight', weight_name='mapped_weight_name',
                              shard_id='shard_id', expert_id='expert_id', return_success='return_success')
    assert not any(isinstance(n, ast.Name) and n.id == 'expert_id' and isinstance(n.ctx, ast.Store) for n in ast.walk(inner))
    name_assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and
                           any(isinstance(a, ast.Name) and a.id == '_MOE_WEIGHT_LOADER_NAME_MAP' for a in n.targets))
    aliases = ast.literal_eval(name_assignment.value)
    assert aliases == {'w13_scale_bias': 'w13_scale', 'w2_scale_bias': 'w2_scale'}
    assert set(aliases).isdisjoint(set(aliases.values()))
    adapter = by_role['MOE_PARAM_ADAPTER']
    adapter_text = '\n'.join(p for part in adapter['parts'] for p in part['source'])
    assert adapter_text.count('set_weight_attrs(param, extra_weight_attrs)') == 2
    assert 'extra_weight_attrs.update({"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})' in adapter_text
    scheme = by_role['W8A8_SCHEME_LAYOUT']
    scheme_tree = ast.parse(Path(scheme['frozen_path']).read_bytes())
    scheme_cls = next(n for n in scheme_tree.body if isinstance(n, ast.ClassDef) and n.name == 'AscendW8A8DynamicFusedMoEMethod')
    param_defs = []
    for fn in scheme_cls.body:
        if not isinstance(fn, ast.FunctionDef) or fn.name not in ['get_weight', 'get_dynamic_quant_param']:continue
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Subscript):
                sub = n.targets[0]
                if isinstance(sub.value, ast.Name) and sub.value.id == 'param_dict':
                    param_defs.append(dict(name=ast.literal_eval(sub.slice), producer=fn.name,
                        allocation_expression=ast.unparse(n.value),
                        callback_attribute_call='set_weight_attrs(param,extra_weight_attrs)',
                        actual_field_value_or_native_allocation_proved=False))
    initial_names = [p['name'] for p in param_defs]
    assert initial_names == ['w13_weight', 'w2_weight', 'w13_weight_scale', 'w13_weight_offset', 'w2_weight_scale', 'w2_weight_offset']
    assert not set(initial_names) & set(aliases)
    runtime = read(O / 'F21-multiplane-buffers-and-publication-contract-v50.json')['W8A8_fused_MC2_named_planes']
    assert len(runtime) == 6
    refs = ['F21-callback-registration-source-v52.json', 'F21-checkpoint-map-commutation-contract-v51.json',
            'F21-multiplane-buffers-and-publication-contract-v50.json']
    # Verify the reused profile receipt,without promoting itslinearapplyframes to
    # proof of aMoEconstructor/parameter callback.
    receipt = N / 'generic-callers166-v18.json';sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
    manifests = [read(p) for p in O.glob('checkpoint-manifest-v*.json')]
    assert any(a['path'] == str(receipt) and a['sha256'] == sha for m in manifests for a in m.get('artifacts', []))
    profile = packet('generic-callers166-v18', False)
    assert 'method_adapters.py(224): apply' in json.dumps(profile, ensure_ascii=False)
    write(O / 'F21-callback-source-index-v52.json', dict(
        generated_utc=now, files=[{k:f[k] for k in ['path', 'frozen_path', 'sha256', 'role', 'historical166_hash_matched']} for f in source['files']],
        reused_profile=dict(path=str(receipt), receipt_sha256=sha, historical_manifest_match=True,
            limit='Savedgenericlinearapplyframeonly;notMoEcallback/factoryidentity.'),
        actual_parameter_fields_or_factory_selection_proved=False, overall_complete=False))

    facts = [
        ('resolve_before_capture', 'HISTORICAL166_HASH_MATCHED_SOURCE',
         'CoreRoutedExperts resolvesquant_method123beforecapturing self.weight_loaderinmoe_quant_params159-169andcallingcreate_weights171. Ascend_get_quant_method588-591delegatestoquant_configwhenpresent. Classhasnoownweight_loaderormapperoverride,butinstanceassignmentduringquantresolutioncanstillreplacecallback.'),
        ('ModelSlim_instance_wrapper', 'LOCAL_REFERENCE_CONFIG_SOURCE',
         'UnderModelSlimMoEpredicateandquant_type!=None,config754rebindslayer.weight_loaderto_make_modelslim_moe_weight_loaderbefore returningtheMoEadapter. quant_typeNonechoosesunquantizedmethodanddoesnotinstallthiswrapperinthatbranch. Actualconfig/predicateandcurrent166configbytesarenotattested.'),
        ('wrapper_exact_forwarding', 'DERIVED_FROM_REFERENCE_WRAPPER_AST',
         'Wrapperonlyrewritesthelastweight_namecomponent: w13_scale_bias→w13_scale,w2_scale_bias→w2_scale. Itforwardsparam,loaded_weight,shard_id,expert_idandreturn_successunchangedtothecapturedloader,andreturnsitsresult. ThiswrapperdoesnotitselfrelocateexpertID,changebuffersorimplementstaticplacement.'),
        ('parameter_attribute_registration', 'LOCAL_REFERENCE_ADAPTER_SOURCE',
         'Adaptercallsset_weight_attrswiththecapturedextraattributesineachinitialweightloopanddynamicquantparamloop. DynamicquantparamsdefaultCHANNEL,withconditionalGROUPoverridebyfieldnameandgroup_size. TheattributehelperimplementationandactualParam.weight_loaderobjectsarenotmeasuredhere.'),
        ('initial_W8A8_parameters', 'HISTORICAL166_HASH_MATCHED_SCHEME_SOURCE',
         'W8A8schemereturns2INT8weighttensorsand4scale/offsettensorsinparams_dtype,allwithleadingnum_experts. Theseareinitialregistrationnames/layouts,notallruntimeEPLBplane identities orphysicalmemorycost.'),
        ('W8_name_identity', 'DERIVED_CONDITIONAL',
         'The sixinitialW8A8namesarenotkeysinthe2-entrybiasaliasmap. Forthosenames,thewrapperisname-identityandkeepsphysicalexpert_idp. Thev51ordinaryPhi(p)conditionthereforeisnotfixedbythiswrapper,assumingthedeclaredregistration/underlyingloaderpath.'),
        ('repeated_configuration_and_reload', 'DERIVED_CONDITIONAL',
         'Withthisfixedliteralaliasmap,name-rewriteisidempotent;constructingwrappersrepeatedlystillcreatesnewclosuresaroundthecurrentloader. Anewlayer.weight_loaderbindingdoesnotbyitselfproveexistingParamcallbacksorpayloadbanksupdated. Sourceparameterregistrationandpostloadgenerationsmustbematched;actualhelper/settersemanticsandobjectsremainunknown.'),
        ('scope_and_next_edge', 'UNKNOWN_ACTUAL_STATE',
         'Expectedregistrationpathisnowlocalized;actualModelSlimselection,adapter/config/helperversionandParamcallableidentitystillrequireevidence. P0initializerbody/actualtablevalues,otherpostloadhooks,allquantbackends/consumers/native/capture/resourcegapsremain. Do notrepeattheinheritance-onlyassumptionorclaimthisknownnamemapreconcilesstaticplacement.'),
    ]
    write(O / 'F21-quant-callback-forwarding-contract-v52.json', dict(
        generated_utc=now, evidence=refs,
        findings=[dict(id='CB52-%02d' % (i + 1), topic=t, grade=g, finding=s) for i, (t, g, s) in enumerate(facts)],
        expected_registration_scope_analyzed=True,
        wrapper_keyword_forwarding=forwarding, literal_name_map=aliases,
        name_map_idempotent_under_fixed_literal=True,
        changes_expert_id_in_this_wrapper=False,
        actual_parameter_callback_proved=False, actual_static_placement_reconciled=False,
        no_model_import_remote_or_device_work=True, implementation=False,
        whole_module_closed=False, overall_complete=False))

    producer_rows = [
        dict(initial='w13_weight', initial_layout='INT8[C,2I,H]', runtime=['w13_weight_list'],
             transformation='Transpose/contiguous;W8A8NZformatcast;optionalper-expertclone/list.', source='W8A8postload280-286/299-301'),
        dict(initial='w2_weight', initial_layout='INT8[C,H,I]', runtime=['w2_weight_list'],
             transformation='Transpose/contiguous;W8A8NZformatcast;optionalper-expertclone/list.', source='W8A8postload280-286/299-301'),
        dict(initial='w13_weight_scale', initial_layout='params_dtype[C,2I,1]',
             runtime=['w13_weight_scale_fp32_list', 'fused_w1_scale_list'],
             transformation='View;FP32conversionandpackedINT64scaleproducer;optionalclones.', source='W8A8postload287-295/302-310'),
        dict(initial='w2_weight_scale', initial_layout='params_dtype[C,H,1]',
             runtime=['w2_weight_scale_list', 'fused_w2_scale_list'],
             transformation='ViewandpackedINT64scaleproducer;optionalclones.', source='W8A8postload290-295/305-314'),
        dict(initial='w13_weight_offset', initial_layout='params_dtype[C,2I,1]', runtime=[],
             transformation='View;notnamedintheinspectedW8A8MC2EPLBregistry,notadeadnessproof.', source='W8A8postload289'),
        dict(initial='w2_weight_offset', initial_layout='params_dtype[C,H,1]', runtime=[],
             transformation='View;notnamedintheinspectedW8A8MC2EPLBregistry,notadeadnessproof.', source='W8A8postload291'),
    ]
    assert sorted(name for row in producer_rows for name in row['runtime']) == sorted(runtime)
    write(O / 'F21-initial-parameters-to-runtime-planes-v52.json', dict(
        generated_utc=now, initial_parameter_definitions=param_defs,
        runtime_W8A8_MC2_registry_names=runtime, producer_rows=producer_rows,
        evidence=['F21-callback-registration-source-v52.json', 'F21-postload-updater-transfer-source-v50.json'],
        both_counts_six_is_not_one_to_one_identity=True,
        lifecycle=['Quantmethodresolution/wrapperselection.', 'Initialparametersregisteredandloadcallbackattributesforwarded.',
                   'Checkpointdatawrittenundercorrectphysicalmapping.',
                   'Postloadtranspose/format/view/cast/packedscales;listmodeclonesanddeletessomerawattributes.',
                   'RuntimeEPLBbindsderivednamedplanesandbufferlifetimes;capturedreadersneedthesamegeneration.'],
        conditions=['List/mega/otherbackendbranchesandactualattributesmustbematched;thisisnotanall-backendautomaticABIproof.',
                    'DeletingrawattributesdoesnotproveTensorstoragefreed;existingexternalrefs/capture/readersmayretainit.',
                    'Do notuseinitialcheckpointcallbackorparameter-namecoverageasruntimepackedplane/generationcoverage.',
                    'Reloadmayneedrecreate/re-registerinitialparamsorbackend-specificruntime-loadinterface;do notassumetheoriginalparams_dictstillnamesdeletedrawweightattributes.',
                    'Fieldcounts/shapesarenotphysicalmemory/traffic/copylatencyorindependentallocationcounts.'],
        actual_aliases_or_native_generation_proved=False, implementation=False,
        whole_module_closed=False, overall_complete=False))
    nodes = ['resolve_quant_method', 'ModelSlim_nonNone_MoE_branch', 'install_instance_name_wrapper',
             'capture_layer_callback_in_kwargs', 'register_initial_weight_params', 'register_initial_quant_params',
             'set_weight_attrs_external_contract', 'actual_Param_callback_unknown',
             'name_alias_wrapper', 'underlying_ordinary_physical_mapper', 'checkpoint_payload_loaded',
             'postload_cast_clone_format_and_packing', 'runtime_planes_EPLB_views', 'next_native_reader_unknown']
    edges = [
        ('resolve_quant_method', 'ModelSlim_nonNone_MoE_branch', 'CONDITIONAL_REFERENCE_FACTORY_SELECTION'),
        ('ModelSlim_nonNone_MoE_branch', 'install_instance_name_wrapper', 'REFERENCE_CONFIG_INSTANCE_ASSIGNMENT'),
        ('install_instance_name_wrapper', 'capture_layer_callback_in_kwargs', 'HISTORICAL_CORE_SOURCE_ORDER'),
        ('capture_layer_callback_in_kwargs', 'register_initial_weight_params', 'REFERENCE_ADAPTER_KWARG_FLOW'),
        ('capture_layer_callback_in_kwargs', 'register_initial_quant_params', 'REFERENCE_ADAPTER_KWARG_FLOW'),
        ('register_initial_weight_params', 'set_weight_attrs_external_contract', 'SOURCE_ATTRIBUTE_HELPER_CALL'),
        ('register_initial_quant_params', 'set_weight_attrs_external_contract', 'SOURCE_ATTRIBUTE_HELPER_CALL'),
        ('set_weight_attrs_external_contract', 'actual_Param_callback_unknown', 'REQUIRED_ACTUAL_HELPER_OBJECT_EDGE_UNPROVED'),
        ('actual_Param_callback_unknown', 'name_alias_wrapper', 'CONDITIONAL_ACTUAL_REGISTERED_CALLBACK'),
        ('name_alias_wrapper', 'underlying_ordinary_physical_mapper', 'REFERENCE_WRAPPER_EXPERT_ID_UNCHANGED'),
        ('underlying_ordinary_physical_mapper', 'checkpoint_payload_loaded', 'CONDITIONAL_MAPPING_AND_ALL_PLANES_COHERENT'),
        ('checkpoint_payload_loaded', 'postload_cast_clone_format_and_packing', 'DOCUMENTED_POSTLOAD_TRANSFORMS'),
        ('postload_cast_clone_format_and_packing', 'runtime_planes_EPLB_views', 'CONDITIONAL_LAYOUT_AND_REGISTRY_BRANCH'),
        ('runtime_planes_EPLB_views', 'next_native_reader_unknown', 'REQUIRED_READY_LAST_READER_CAPTURE_GENERATION_UNPROVED'),
    ]
    flow = graph(nodes, edges)
    flow.update(scope='Qualifiedsource-registrationandproducerflow;notanactualparameterobject,nativecaptureorall-loadercertificate.',
                actual_callback_or_static_pairing_proved=False, overall_complete=False)
    write(O / 'F21-callback-and-plane-lifecycle-flow-v52.json', flow)
    refinements = [
        'Installanyplacement-awareloaderattherightresolution/registrationphase;preserveModelSlimleaf-namecompatibilityandallcallbackmetadata.',
        'SourceclassinheritancealonecannotidentifyParam.weight_loader:quantconfigurationmayreplaceinstancecallbackbeforecreation.',
        'Preserveexpert_idphysicalnamespaceandfullP0/Phi/Routecommutation;thisbias-namewrapperdoesnotimplementplacement.',
        'Matchinitialweights/scales/offsetswithpostloadFP32/packed/NZ/listproducers;6initialfieldsand6runtimeplanesarenotone-to-onetensoridentities.',
        'ExistingParamcallbacks/rawattributes/runtimebanksmustberevalidatedonreload;rebindinglayercallbackisnotaproofofupdatedregisteredparametersorpayloads.',
        'Keepactualhelper/config/adapteridentities,factorybranch,allconsumers/capture/nativeconditionsandseparateinitialload/dynamic/steadycostdenominators.']
    write(O / 'F21-static-loader-registration-refinements-v52.json', dict(
        generated_utc=now, candidate='F21', refines='F21-v50-STATIC-CHECKPOINT-PLACEMENT / SL51conditions',
        conditions=[dict(id='RG52-%02d' % (i + 1), condition=s,
                         state='CONDITIONAL_SOURCE_CONTRACT;ACTUAL_OBJECT_BINDING_UNPROVED') for i, s in enumerate(refinements)],
        new_main_candidates=0, implementation=False, native_test=False,
        whole_module_closed=False, overall_complete=False))
    branch = deepcopy(read(O / 'F21-interface-branch-ledger-v51.json'))
    b8 = next(r for r in branch['branches'] if r['id'] == 'F21-B8-postload-and-capture')
    b8['finding'] += ' v52referenceModelSliminstallsnamewrapperbeforekwargs/paramregistration;onlyscale_biastailnameschange,expert_idpunchanged. Bothinitialweightandquantparamloopsforwardattrs. SixinitialparametersandruntimeEPLBsixplanesareproducer-relatedbutnotone-to-one;actualobjects/helper/configandpostloadstaticpairingremainopen.'
    b8['evidence'] += ['F21-quant-callback-forwarding-contract-v52.json', 'F21-initial-parameters-to-runtime-planes-v52.json']
    branch.update(generated_utc=now, parent=str(O / 'F21-interface-branch-ledger-v51.json'))
    write(O / 'F21-interface-branch-ledger-v52.json', branch)
    gate = read(O / 'PROFILE-CODE-QUESTION-GATE-v52.json')
    gate.update(status='CONDITIONAL_MODELSLIM_CALLBACK_AND_PARAM_TO_RUNTIME_PLANE_FLOW_ANALYZED;ACTUAL_BINDING_OPEN',
        answered_by=['F21-quant-callback-forwarding-contract-v52.json', 'F21-initial-parameters-to-runtime-planes-v52.json',
                     'F21-static-loader-registration-refinements-v52.json'],
        closed_scope='Concretequalifiedregistrationpathandwrapperargumentforwarding;initialparameterdefinitionsversusruntimeplaneproducers.',
        not_closed='Originaladapter/config/helperidentityandactualfactory/Paramcallbacks;P0body/currenttables,otherhooks/staticreconciliation,allbackend/readers/native/resources.',
        stop_rule='Do notrepeatclassinheritanceasactualcallbackproofornamewrapperasplacementhook. Nextonlytheunresolvedoriginalversion/helper/P0/actualmapedgeorotherfinitecoverage;do notbroadenunrelatedrepo/SDKreading.')
    write(O / 'PROFILE-CODE-QUESTION-GATE-v52.json', gate)
    print(json.dumps(dict(frozen_inputs=5, historical166_hash_matches=3, reference_only_inputs=2,
        wrapper_expert_id_unchanged=True, name_aliases=len(aliases), initial_W8_parameters=6, runtime_EPLB_planes=6,
        counts_not_one_to_one=True, flow_nodes=len(nodes), flow_edges=len(edges),
        refined_conditions=len(refinements), actual_callback_or_pairing_proved=False,
        overall_complete=False)))


if __name__ == '__main__':
    main()
