"""Persist the qualified quant callback and initial/runtime payload distinctions."""
from copy import deepcopy
from pathlib import Path
import hashlib

from finalize_origin_coverage_v9 import N, O, read, write
from persist_profile_refinement_v3 import persist_refinement


def main():
    assert not (O / 'REVIEW-CHECKPOINT-v52.json').exists()
    source = read(O / 'F21-callback-source-index-v52.json')
    contract = read(O / 'F21-quant-callback-forwarding-contract-v52.json')
    planes = read(O / 'F21-initial-parameters-to-runtime-planes-v52.json')
    flow = read(O / 'F21-callback-and-plane-lifecycle-flow-v52.json')
    refs = read(O / 'F21-static-loader-registration-refinements-v52.json')
    assert len(source['files']) == 5 and sum(f['historical166_hash_matched'] for f in source['files']) == 3
    for f in source['files']:
        assert hashlib.sha256(Path(f['frozen_path']).read_bytes()).hexdigest() == f['sha256']
    assert not contract['changes_expert_id_in_this_wrapper'] and not contract['actual_parameter_callback_proved']
    assert contract['literal_name_map'] == {'w13_scale_bias': 'w13_scale', 'w2_scale_bias': 'w2_scale'}
    assert len(planes['initial_parameter_definitions']) == len(planes['runtime_W8A8_MC2_registry_names']) == 6
    assert planes['both_counts_six_is_not_one_to_one_identity']
    assert len(flow['nodes']) == len(flow['edges']) == 14 and len(refs['conditions']) == 6
    evidence = ['F21-quant-callback-forwarding-contract-v52.json', 'F21-initial-parameters-to-runtime-planes-v52.json',
                'F21-callback-and-plane-lifecycle-flow-v52.json', 'F21-static-loader-registration-refinements-v52.json',
                'F21-interface-branch-ledger-v52.json', 'F21-callback-source-index-v52.json',
                'F21-callback-registration-source-v52.json']
    candidate = dict(
        review_v52=dict(callback_contract=str(O / evidence[0]), initial_runtime_planes=str(O / evidence[1]),
            lifecycle_flow=str(O / evidence[2]), refined_conditions=str(O / evidence[3]),
            frozen_inputs=str(O / evidence[5]),
            relation='QualifiedModelSlimfactorywrapsinstanceweight_loaderbeforecorecaptureskwargs;classinheritancealoneisnotactualcallbackproof. Wrapperonlyaliasesw13/w2scale_biasleafnamesandforwardsphysicalexpert_idunchanged. InitialsixW8A8parametersarenotone-to-onesixruntimeEPLBplanes. Knownwrapperisnotstaticplacementreconciliation.',
            implementation=False, native_test=False, new_performance_result=False),
        profile_review_state='QUALIFIED_MODELSLIM_CALLBACK_REGISTRATION_AND_PAYLOAD_PRODUCERS_ANALYZED;ACTUAL_BINDING_OPEN',
        conclusion_changing_gap='ExpectedModelSlim/W8A8registrationandwrapperforwardingnowknownunderdeclared3historical166+2local-referenceinputs. Remainingactualconfig/predicate/adapter/helperversionandParamcallbackobjects,P0body/currenttables/otherpostloadhooks/staticP0-Phi-Routepayloadpairing/nativecapture/allreader/resourceevidence. Do not repeat the callback chain as wholly unlocated or treat no classoverride as no instancewrapper.',
        next_profile_or_contract_action='Firstuseknownsame-versionfrozen/receiptmaterialforP0/helperidentity;ifmissing,retainoriginaldata/versiondependencyratherthanborrowanunmatchedrepoas166. No further broadF21referenceexpansion. Advanceotheroriginalfiniteconsumer/branchquestionswhilewaitingfororiginalprofileaccess. Preservev51physical-IDcommutationandv50multiplanebufferconditions;noimplementation/nativebenchmarkrequiredmerelyforconditionalanalysis.')
    modules = {
        'M00': dict(finding='ConcretequalifiedModelSlimMoEregistration nowlocated:quantmethodresolutionbeforecallbackcapture;configcanwrapinstancecallbackbeforeparametersregistered. Noownclassloaderoverrideisnotcallbackidentityproof. ActualModelSlimselectionand2referencefile166identityremainunattested.',
            evidence=evidence, candidates=['F21'], gap='Actualconfig/predicate/adapter/helper/Paramcallbackobjectsandwholemodelselection/initialization/processcoverage remain;no startupcost/gain from sourcecallcount.'),
        'M12': dict(finding='Referencewrapperaliasesonly2scale_biasleafnames;physicalexpert_idunchanged,soitdoesnotreconcileplacement. SourceW8A8initial2INT8weights+4scaleoffsetparams andruntime6planesareseparateproducerdomains. DefaultCHANNEL/conditionalGROUPmetadataandallquantconsumersmustsurvivealoadplan.',
            evidence=evidence, candidates=['F21'], gap='Actualcallback/table/payloadidentity,otherquantbackends/hooks,completeconsumers/nativeprecision/currentmapcommutationremain;initialregistrationcountnotruntimeplane/nativepayloadproof.'),
        'M24': dict(finding='Callbackreferencecaptureandpostloadview/clone/packedplaneproduction nowexplicitlyseparated. Rebindinglayercallbackorcounting6fieldsdoesnotproveexistingParamcallbacks/runtimebanks refreshed;reloadmayneedrawparamregistrationorbackend-specificloadinterface.',
            evidence=evidence, candidates=['F21'], gap='Actualattributehelper/objects,reload/postloadhistory,captureaddresses/nativegeneration/lastread/readyandallobserverlifetimesstillunknown;no storagereclamationoraddress-onlyreuseclaim.'),
    }
    remaining = deepcopy(read(O / 'module-coverage-gaps-v51.json')['remaining'])
    matches = [(i, s) for i, s in enumerate(remaining) if s.startswith('F21v50static/dynamicboundary')]
    assert len(matches) == 1
    i, before = matches[0]
    after = before + ' v52qualifiesregistration:ModelSlimnonNoneMoEbranchrebindslayercallbackbeforecorekwargs/paramcreation;knownwrapperchangesonlyscale_biasleafnameandforwardsphysicalpunchanged. InitialW8A8sixparams≠runtimeEPLBsixplanesone-to-one. Expectedchain/producersanswered;actualconfig/adapter/helperbytesandParamobjectbindingremainunknown. No classoverridealoneisnolongeraproofassumption.'
    remaining[i] = after
    write(O / 'F21-callback-gap-reconciliation-v52.json', dict(parent_version=51, before=before, after=after,
        scope='Refine the exact callback/version/producer gap;do not retire actual static pairing or wholemodule/consumer/native/resource requirements.',
        overall_complete=False))
    remaining.append('F21v52fiveinput sourcecase:core/Ascend/W8A8scheme3historical166matched,ModelSlimconfig/adapter2referenceonly. Configfactorywrapsweight_loaderbeforecreate_weights;wrapperliteralw13_scale_bias→w13_scale,w2_scale_bias→w2_scaleandotherargs/expert_idunchanged. Adapterforwardsattrdictinweight+dynamicquantloops;actualset_weight_attrs/currentParamfieldsnotattested. W8A8initialsix=(w13,w2weights,2scales,2offsets),runtimeEPLBsix=(2weights,FP32/ordinaryscales,2packedscales);view/cast/format/clone/deletionproducerboundarypreserved,nosix-countidentity/dead-offset/storagefreeproof. SixregistrationconditionsrefineexistingF21,notnewcandidate. Knownwrapperdoesnotload-reorderstaticplacement;P0same-versionbody/currentmaps/callbackobjects/hooks/allreader/native/phasecostremain. Nextuseoriginalsame-versionmaterialoradvanceotherfinitegaps,notfurtherunmatchedF21referenceexpansion.')
    action = (
        '继续原Goal。v52是F21具体量化callback/参数到runtime生产者的实际定点进展。先读F21-quant-callback-forwarding-contract-v52.json、F21-initial-parameters-to-runtime-planes-v52.json、F21-static-loader-registration-refinements-v52.json、F21-callback-source-index-v52.json及module-coverage-gaps-v52.json。'
        '5输入冻结N/F21-callback-source-v52；coreRouted68c/AscendRoutedf9db/W8A8a676历史166hash匹配，method_adapters341d93.../ModelSlimconfig92c924...仅本地参考，不能冒充原166实际配置/Param字段。generic-callers-v18的method_adapters224是linearapply定位线索，不是MoE构造callback证明。'
        'Core先123resolvequant_method再159捕获self.weight_loader、171create_weights。Ascend类虽无ownweight_loader，但ModelSlimnonNone MoE分支754会先给instance安装_make_modelslim_moe_weight_loader。不能以class无override推断param直接用Base。'
        'wrapper只改尾名w13_scale_bias→w13_scale和w2_scale_bias→w2_scale；param/tensor/shard/expert_id/return_success原样转发。W8A8初始6参数名不在alias键中，p仍physical，故已知wrapper没有承担静态slot重排。set_weight_attrs helper实现/实际Paramcallback对象未核实。'
        '初始6张量是2INT8weight+2scale+2offset；runtimeEPLB6是2weight+FP32/普通scale+2packedscale。后处理view/cast/NZ/clone和某些raw属性删除必须入代际/所有者契约，不能按6=6配对或认offset死/存储已释放。reload不能只改layercallback假设旧Params/runtimebanks已更新。'
        '本轮6条件细化现有F21-v50STATIC，不增主候选不实现不native测试无收益结论。下一步先检查P0/helper同版本source是否已在冻结包/回执；没有则记为原资料/版本依赖，不再用其它git版本扩F21参考树。推进其它原范围明确未读branch/consumer，勿重复已答wrapper/registry/ID问题。v51P0/Phi/fullRoute条件与v50动态6plane/per-keybank仍有效。'
        '原入口待用户：193root/hostname已验，166/实际105新账号asset-no-match(SSHrc0非成功)，193已查旧路径无资料不等于全盘无。不要旧密码<REDACTED_CREDENTIAL>。恢复原入口后核caps/Eindex/model/input/serve/graphhash补读保存原数据。'
        '35模型7框架0整模块；849缺PMU/16zero-sidecarcandidate phase-unbound/3136模块歧义执行/2082regimes/26theoryINCOMPLETE保留。缓存16device24Cube48Vector/1.8GHz参考及28PMU字段已知勿重采。M21/M22语义族与v49fallback契约保持。局部条件分析完成不需实现/nativebenchmark；实际未读范围和关键证据不可因报告成稿关闭，Goal仍active。'
    )
    report = (
        '本轮定位并分析了一个具体的ModelSlim/W8A8参数注册路径。3份源码与历史166哈希匹配，ModelSlim配置和适配层2份依赖只有本地参考身份。保存调用栈中的量化适配层linear apply仅用于定位，没有冒充MoE加载回调对象证据。没有连接服务器或执行模型/设备工作。\n\n'
        '核心构造器先解析quant_method，再把self.weight_loader放进参数属性字典。参考ModelSlim配置会在这一解析过程中给layer实例安装包装器；因此，类没有重写weight_loader并不能证明参数直接持有原始Base回调。适配层在权重和动态量化参数两个注册循环中都转交该属性字典。实际属性helper与Param字段仍未核验。\n\n'
        '该包装器仅改写w13_scale_bias→w13_scale、w2_scale_bias→w2_scale尾名，expert_id等参数原样转发。它没有实施静态expert槽位重排；在所查W8A8初始参数名上，名称映射也为恒等。因此v51的物理ID、加载map与完整执行route一致性条件仍然需要证明。\n\n'
        '初始注册六张参数与runtime EPLB的六个字段不能一一配对：前者包括两张weight、两张scale和两张offset；后者含两张weight、FP32/普通scale及两张packed scale。中间经过view、cast、NZ格式、可选clone及部分raw属性删除。数量相同不证明张量、存储或代际相同，也不证明offset已死或存储释放。\n\n'
        '新增6项条件细化现有F21静态加载方案，覆盖正确hook时机、回调元数据、producer变换与reload边界。已保留3次本地检查/选择器错误并修正；源代码与条件分析结果仍需最终完整性校验。原166实际配置/依赖/回调对象、P0与映射表、静态权重对齐、全部消费者及资源证据仍待补齐。\n\n'
        '35模型候选、7框架候选、0整模块关闭。没有新增优化实现、native测试或实测收益；整体Goal继续进行。\n'
    )
    jobs = [
        dict(kind='review', status='error', description='Adhocreceipt-displayinspectioninitiallyhadelseNoneSyntaxError;correctedboundedinspection,no datamutationorremote work.',
             artifacts=[O / 'F21-callback-receipt-inspection-error-v52.json']),
        dict(kind='review', status='error', description='FirstcallbackselectoromittedAnnAssignandindirect**kwargscreate_weights;guardfailedbeforecompleteoutput. Correctedexactdictionaryandcallselection;notabsenceevidence.',
             artifacts=[O / 'F21-callback-reader-error-v52.json']),
        dict(kind='review', status='error', description='FactoryselectorlookedforliteralRoutedExpertsbutactualpredicateisis_fused_moe_layer;correctedactualMoEbranchandparentelsearms,no runtimefactoryclaim.',
             artifacts=[O / 'F21-callback-config-selector-error-v52.json']),
        dict(kind='review', status='pass', description='Exactregistration-pathsourcecapture/freeze:3historical166matchedinputs+2localreferenceinputs;coreorder,instancewrapper,2paramloopsandW8A8definitionsrecorded withoutimports/device/remote work.',
             artifacts=[O / evidence[5], O / evidence[6], N / 'read_F21_callback_registration_v52.py'] + [Path(f['frozen_path']) for f in source['files']]),
        dict(kind='design-check', status='pass', description='ASTprovesknownwrapperforwardsexpert_idandotherargsunchanged;literalbias-namealiasmapidempotent. Separate6initialparamsfrom6runtimeplaneproducers;14node14edgequalifiedflow,nativeedgesunknown.',
             artifacts=[O / evidence[0], O / evidence[1], O / evidence[2], N / 'analyze_F21_callback_registration_v52.py']),
        dict(kind='review', status='pass', description='Sixregistration/reloadconditionsrefineexistingF21;actualfactory/callback/helper/P0/staticpairing/allconsumer/resourcegapspreservedandoriginalscopeleftopen.',
             artifacts=[O / evidence[3], O / evidence[4], O / 'F21-callback-gap-reconciliation-v52.json']),
    ]
    persist_refinement(version=52, previous=51,
        title='F21 ModelSlim callback registration and initial/runtime payload producers',
        summary=dict(filename='F21-callback-registration-summary-v52.json',
            previous_turn_classification='PROGRESS:v51checkpoint_entry_namespace',
            frozen_source_inputs=5, historical166hash_matched_inputs=3, local_reference_inputs=2,
            literal_name_aliases=2, initial_parameter_fields=6, runtime_EPLB_fields=6,
            initial_runtime_are_one_to_one=False, wrapper_changes_expert_id=False,
            refined_registration_conditions=6, qualified_flow_nodes=14, qualified_flow_edges=14,
            actual_callback_or_static_pairing_proved=False, target_implementation_changed=False,
            native_test=False, original_profile_access_restored=False,
            remote_job_status='No remote operation/background job launched in v52. Localinspection/selector3errorscorrectedandretained;3source/design/reviewpasses;serverhealth/workloadnotpolled.'),
        changes=['Sourceclassinheritancealoneisnotcallbackproof:referenceModelSliminstallsinstancewrapperbeforecoreparamcreation.',
                 'Knownwrapperaliasesonly2scale_biasleafnamesandforwardsphysicalexpert_idunchanged;itdoesnotreconcilestaticplacement.',
                 '6initialW8A8paramsand6runtimeEPLBplanesaredifferentproducerdomains;6conditionsrefineF21withoutimplementation/gain/moduleclosure.'],
        candidate_updates={'F21': candidate}, module_updates=modules,
        remaining=remaining, action=action, jobs=jobs, report=report,
        read_first=evidence[:6] + ['original-profile-access-status-v47.json'],
        publication_rows={}, gate_updates=dict(overall_complete=False),
        scripts=[N / 'read_F21_callback_registration_v52.py', N / 'analyze_F21_callback_registration_v52.py'],
        executor_prefix='zcode-F21-callback', main_script=Path(__file__), source_read=True)


if __name__ == '__main__':
    main()
