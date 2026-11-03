"""Persist the checkpoint-loading namespace analysis,keeping actual pairing and full-scope gaps open."""
from copy import deepcopy
from pathlib import Path
import hashlib

from finalize_origin_coverage_v9 import N, O, read, write
from persist_profile_refinement_v3 import persist_refinement


def main():
    assert not (O / 'REVIEW-CHECKPOINT-v51.json').exists()
    contract = read(O / 'F21-checkpoint-map-commutation-contract-v51.json')
    witness = read(O / 'F21-static-load-identity-witness-v51.json')
    inputs = read(O / 'F21-checkpoint-source-index-v51.json')
    conditions = read(O / 'F21-static-loader-conditions-v51.json')
    flow = read(O / 'F21-static-loading-source-flow-v51.json')
    assert len(inputs['files']) == 5 and sum(f['historical166_hash_matched'] for f in inputs['files']) == 3
    for f in inputs['files']:
        assert hashlib.sha256(Path(f['frozen_path']).read_bytes()).hexdigest() == f['sha256']
    assert len(contract['findings']) == 10 and len(conditions['conditions']) == 8
    assert not contract['actual_param_callback_binding_proved'] and not contract['actual_static_load_execution_pairing_proved']
    assert not witness['native_test']
    assert len(flow['nodes']) == 16 and len(flow['edges']) == 13
    evidence = ['F21-checkpoint-map-commutation-contract-v51.json', 'F21-static-load-identity-witness-v51.json',
                'F21-static-loading-source-flow-v51.json', 'F21-static-loader-conditions-v51.json',
                'F21-interface-branch-ledger-v51.json', 'F21-checkpoint-source-index-v51.json',
                'F21-checkpoint-routing-source-v51.json', 'F21-mapping-import-bridge-source-v51.json']
    candidate = dict(
        review_v51=dict(contract=str(O / evidence[0]), identity_condition=str(O / evidence[1]),
            source_flow=str(O / evidence[2]), conditions=str(O / evidence[3]), frozen_source_index=str(O / evidence[5]),
            relation='TargetquerymethodpassesR0,butactualprivateloadersuppliesselfR. QualifiedreferencecoremappinguseslogicalcheckpointnameP0[p]andphysicalcallbackIDp;ordinarybasecallbackusescanonicalmanagerPhi. Fullcompositeexecutionrouteandloadedpayloadmustcommute. Filefilter-noop/queryR0/generic3Dentrycannotstandinforactualstaticpairing.',
            source_scope='3historical166matchedsourceinputs;2frozenreferencebridgeinputs. P0body/actualcallback/actualmapsandweightsnotrecovered.',
            implementation=False, new_performance_result=False),
        profile_review_state='CHECKPOINT_QUERY_ACTUAL_ENTRY_AND_PHYSICAL_ID_SCOPE_ANALYZED;ACTUAL_STATIC_PAIRING_OPEN',
        conclusion_changing_gap='Remaining actualparam.weight_loaderbinding/overrides,P0initializerbody/currentvalues,166bridgeidentity/otherpostloadhooks andactualP0/Phi/placement/replica/payload/nativegenerations. Query-vs-actualcount,firstsuccess,ordinarymapperandgeneric3Dalternative questions nowlocallyanswered;do not list them as whollyunread.',
        next_profile_or_contract_action='Nexttraceonlytheactualparametercallback/P0initializer/remainingpostloadhookedgewithversionidentity,orobtainactualsavedP0/Phi/placement/payloadwhenoriginalaccessreturns. PreservephysicalvslogicalIDs andmetadataR0vsactualR. Reusev50sixplane/buffer/lifetimecontracts;noimplementation/nativebenchmarkrequiredmerelytoregisterconditionalanalysis.')
    modules = {
        'M00': dict(finding='Targetmodelweightentryandquery/loadcountdifference nowlocated. Staticloadplan mustusetheactualprivate loader/selfR andfile/tensornamespace,notmetadataqueryR0orgenericlayer3Dentry. Initialloadcostandsteadyrunremainseparate.',
            evidence=evidence, candidates=['F21'], gap='Actualcallback/P0constructor/bridge/loaderpostloadhooks andallinitialization/process/selectioncoverage stillopen;noactualstartupgainorfullmodelreadproof.'),
        'M12': dict(finding='F21conditionalcheckpointflow nowseparateslogicalrecordP0[p],physicalcallbackIDp,canonicalmanagerPhi andfullAscendroute. Firstsuccessreplicacoverage/sharedsplit/generic3Dheadersemanticdistinctionsdocumented underqualifiedsourceidentity.',
            evidence=evidence, candidates=['F21'], gap='Actualparamcallback/P0/bodyandtablevalues/completequantpaths/payloadandnativeidentityremain;sourceflow/witnessarenotcurrentmapreconciliationornumericalperformanceproof.'),
        'M23': dict(finding='Staticdesiredownership/filter mustderivefromfullcurrentplacement/replicaplanandactualphysical-to-localmapper. Adeclared4experttoycaseprovesall-recordavailabilitydoesnotpreventoldmapperrejectingarecordonitsnewdesiredrank.',
            evidence=evidence, candidates=['F21'], gap='ToymapsarenotEP16runtimevalues. CurrentP0/Phi/placement/replicaroutes,commgrouptopology/nativeprotocol/timingremainopen;no row-balance ratio converted to latency.'),
        'M24': dict(finding='LegacyAscendupdate_expert_map(new)canreplacecanonicalmanagerfieldwithanexecutionmap;physical-loadingnamespaceandpostloadviewsneedrevalidationonreload. Noactualupdate/reloadasserted.',
            evidence=evidence, candidates=['F21'], gap='Actualreload/weightmapepochs/allreaders/capturebankandnativefencesremainunknown;noaddress-onlyreuse orsame-shapegenerationclaim.'),
    }
    remaining = deepcopy(read(O / 'module-coverage-gaps-v50.json')['remaining'])
    rows = [(i, s) for i, s in enumerate(remaining) if s.startswith('F21v50static/dynamicboundary')]
    assert len(rows) == 1
    i, before = rows[0]
    after = before + ' v51actualmodelprivate_load(selfR)vsquery(R0),physicalcallbackIDp/logicalcheckpointnameP0[p]/ordinarymanagerPhi/fullcompositeroute distinctionsnowbound. Firstsuccess/sharedsplit/generic3Dalternative/reloadnamespaceconditionsdocumented. Stillneedactualcallbackbinding,P0body/currentvalues,166bridgeandpositiveactualP0/Phi/G/payloadcommutation;donotrepeataloadingnamespaceorquerymethodaswhollyunlocated.'
    remaining[i] = after
    write(O / 'F21-checkpoint-gap-reconciliation-v51.json', dict(
        parent_version=50, before=before, after=after,
        scope='Retire only newly answered entry/ID semantics. Actual static reconciliation and whole module/quant/consumer/process/resource scope remain.',
        overall_complete=False))
    remaining.append('F21v51sourceflow16nodes13edges isconditionalcheckpointname→physicalp→managerPhi→payloadversusfullRoute,notactualnativeDAG. Sourcequeryget_expert_mapping usesR0,actual_load_model_weights usesselfR;firstsuccessfulparamcallbackbreaksperchunk. Somegenericfused3Dheadersuseexpert_idasdeconcatselectorbutarenotthismanualtargetentry. StaticloadrequiresB_r[s]=E_e for everyactualRoute(e)=(r,s),coherentallplanes/postloadgenerations/replicas. AtoyR0N4rank2mappingshowsfilefilter-noopcanreadarecordyetoldphysicalmapperrejectsitonnewrank;notactualC8tablefailure. RefinesexistingF21-v50STATICvariant8conditions,noF36.3historical166sourcehashes+2referencebridgefilesfrozen;P0constructorbody/globalcallbackbinding/actualtablevaluesandremaininghooksstillunproved.')
    action = (
        '继续原Goal。v51是F21检查点加载/执行映射的实际定点进展。先读F21-checkpoint-map-commutation-contract-v51.json、F21-static-load-identity-witness-v51.json、F21-static-loader-conditions-v51.json、F21-checkpoint-source-index-v51.json及module-coverage-gaps-v51.json。'
        '5输入已冻结N/F21-checkpoint-source-v51；model101f/coreRouted68c4/ExpertMapManager308c为历史166hash匹配，fused_moe __init__/layer两桥只有原冻结源码参考、未单独166核验。不要把sourcehash当实际权重/回调/表值。'
        '明确新事实：V41类load_weights调用private _load_model_weights；get_expert_mapping传R0但实际private加载传self.num_redundant_experts。普通Core mapping tuple用logical name=P0[p]而expert_id是globalphysicalp，普通callback经ExpertMapManager canonical_expert_map[p]落本地槽。读取R0查询/getter或关闭DefaultLoader文件过滤不能当实际加载正确性证明。'
        '目标private loader每chunk首个success即break；共享fused分片有独立name/axis规则。通用RoutedExperts.load_weights(include_fused)是另一个入口，3D headers的expert_id可暂作w1/w3 deconcat索引，不能混作physicalID。旧Ascendsetter可能改变manager字段，reload需重新验证physical加载map与payload代际。'
        'Source关系的P0来自EplbState.build_initial_global_physical_to_logical_map(E,R)，本轮未读该body/未恢复返回表，必须保留symbolic未知。actualparam.weight_loader对象/override未证明。下一步只补这些callback/P0/实际postload hook的明确边，或原资料入口恢复后读实际P0/Phi/placement/replica/payload。不得用当前git其它版本冒充166。'
        '静态正确性要求每个实际Route(e)=(r,s)都有B_r[s]实现同一E_e；toyN4/2rank反例只证明filter≠mapper及新loader plan的必要性，不是当前C8故障/native数值测试。8条件细化F21-v50STATIC，不增加主候选或实现。v50动态six-plane/per-keybuffer/sourcecallstate契约继续有效但3依赖仍参考。'
        '原profile入口待用户：193root/hostname已验，166/实际105新账号asset-no-match(SSHrc0是假成功)，193已查旧路径无资料不等于全盘无。不重试旧密码<REDACTED_CREDENTIAL>备。恢复后先核原caps/Eindex/graph/model/input/servehash。'
        '其余有限模块/branch/consumer分析可继续，不把未实现/nativebenchmark本身算未分析。35模型7框架0整模块，849缺PMU/16phasezero-sidecarcandidate/3136模块归属歧义执行/2082regimes/26theoryINCOMPLETE保持；缓存16device24Cube48Vector/1.8GHz参考/28PMU已知勿重复采。M21/M22语义族与v49四fallback契约未变。整体Goal未完成，不能以报告/局部source问题关闭。'
    )
    report = (
        '本轮继续专家静态placement加载链路，新增证据来自3份历史166哈希匹配源码及2份限定范围的冻结参考桥接源码，均本地冻结保存。未连接服务器、导入模型或运行设备工作。\n\n'
        '查询映射与实际加载并非同一入口：V41的get_expert_mapping固定传入零冗余专家，实际_load_model_weights则传入self.num_redundant_experts。只用查询接口不能证明副本加载覆盖。目标类实际调用自己的手写加载方法，不能用另一个通用3D专家加载入口代替。\n\n'
        '参考桥接调用链中，checkpoint名称使用逻辑expert P0[p]，回调参数expert_id却是全局物理ID p；普通base回调再用ExpertMapManager的canonical map寻找本地槽位。执行端还包含placement、log2phy和副本选择，正确性需要证明实际路由到的槽位装的是对应逻辑expert的完整同代payload。\n\n'
        '已用声明的4专家、2rank小例说明：即使某权重文件已读取，旧物理mapper仍可在新目标rank拒绝该记录。因此，关闭文件过滤不等于完成权重重排。这是条件映射推导，不是当前EP16/C8的实测错误。实际P0、回调对象、映射/权重值和外部postload钩子仍未核实。\n\n'
        '目标加载在每个chunk首次success后停止匹配，副本覆盖需与此相容；共享专家分片、通用3D header ID、reload后的物理/逻辑map也分别定界。新增8项条件细化现有F21静态直装/过滤候选，没有增加主候选、实现优化或推断时延收益。P0构造函数体尚未读取，其返回值保持未知。\n\n'
        '原始profile入口仍待补充。其余模块/分支/全部消费者、版本绑定、跨层/卡/迭代状态及资源证据没有因本报告关闭。35模型候选、7框架候选、0整模块关闭；整体Goal继续进行。\n'
    )
    jobs = [
        dict(kind='review', status='pass', description='Profile-gated checkpoint-path source read/freeze:3historical166matchedinputs+2qualifiedreferencebridges;reuse4historicallypinnedv19receipts. ActualqueryR0vsprivate_loadselfRandphysicalIDnamespace distinguished.',
             artifacts=[O / evidence[5], O / evidence[6], O / evidence[7], N / 'read_F21_checkpoint_routing_v51.py', N / 'read_F21_mapping_bridge_v51.py'] + [Path(f['frozen_path']) for f in inputs['files']]),
        dict(kind='design-check', status='pass', description='Derivefullroute/payloadcommutationanddeclaredtoyfilter-versusmappercounterexample;16node13edgeconditionalflow. P0body/actualparamcallback/tablesdeliberatelyunknown;no native/model/device test.',
             artifacts=[O / evidence[0], O / evidence[1], O / evidence[2], N / 'analyze_F21_checkpoint_commutation_v51.py']),
        dict(kind='review', status='pass', description='RefineexistingF21staticloadvariant8conditions:replica-firstsuccess,shared/fusedentry/reloadnamespaces andcorrectphasecost. Retainactualstaticpairing/wholemodule/resourcegaps.',
             artifacts=[O / evidence[3], O / evidence[4], O / 'F21-checkpoint-gap-reconciliation-v51.json']),
    ]
    persist_refinement(version=51, previous=50,
        title='F21 checkpoint entry, physical IDs and load/execution compatibility',
        summary=dict(filename='F21-checkpoint-commutation-summary-v51.json',
            previous_turn_classification='PROGRESS:v50static_dynamic_publication_boundary',
            frozen_source_inputs=5, historical166hash_matched_inputs=3, reference_bridge_inputs=2,
            source_flow_nodes=16, source_flow_edges=13, refined_static_conditions=8,
            actual_P0_values_or_callback_binding_proved=False, actual_static_pairing_proved=False,
            target_implementation_changed=False, native_test=False, original_profile_access_restored=False,
            remote_job_status='No remote operation/background job launched in v51. Three local source/conditionalmapping/review jobs finished;serverhealth/workloadnotpolled.'),
        changes=['QuerymappingR0differsfromactualprivate_loadselfR;actualtargetentryseparatedfromgeneric3Dloader.',
                 'Checkpointlogicalname/physicalcallbackID/canonicalloadingmapper/fullcompositeexecutionroutecompatibilitynowexplicitunderqualifiedsourceidentities.',
                 'Declaredtoycaseprovesall-recordavailabilityisnotplacementreconciliation;8conditionsrefineexistingF21whileP0/callback/actualpayload/native/wholemodulegapsremain.'],
        candidate_updates={'F21': candidate}, module_updates=modules,
        remaining=remaining, action=action, jobs=jobs, report=report,
        read_first=evidence[:6] + ['original-profile-access-status-v47.json'],
        publication_rows={}, gate_updates=dict(overall_complete=False),
        scripts=[N / 'read_F21_checkpoint_routing_v51.py', N / 'read_F21_mapping_bridge_v51.py', N / 'analyze_F21_checkpoint_commutation_v51.py'],
        executor_prefix='zcode-F21-checkpoint', main_script=Path(__file__), source_read=True)


if __name__ == '__main__':
    main()
