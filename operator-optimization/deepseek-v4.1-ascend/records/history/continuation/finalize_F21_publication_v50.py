"""Persist version-qualified placement/loading/publication evidence without changing the model."""
from copy import deepcopy
from pathlib import Path
import hashlib

from finalize_origin_coverage_v9 import N, O, read, write
from persist_profile_refinement_v3 import persist_refinement


def main():
    assert not (O / 'REVIEW-CHECKPOINT-v50.json').exists()
    loading = read(O / 'F21-static-dynamic-loading-contract-v50.json')
    publication = read(O / 'F21-multiplane-buffers-and-publication-contract-v50.json')
    snapshots = read(O / 'F21-source-snapshot-index-v50.json')
    frontier = read(O / 'F21-unrolled-dynamic-publication-frontier-v50.json')
    variants = read(O / 'F21-placement-publication-variants-v50.json')
    assert len(snapshots['files']) == 6 and sum(f['historical166_hash_attested'] for f in snapshots['files']) == 3
    for f in snapshots['files']:
        b = Path(f['frozen_path']).read_bytes()
        assert len(b) == f['bytes'] and hashlib.sha256(b).hexdigest() == f['sha256']
    assert len(publication['W8A8_fused_MC2_named_planes']) == 6
    assert publication['buffer_model']['not_multiplied_by_layer_count']
    assert not loading['static_postload_weight_pairing_proved']
    assert not publication['native_ready_last_reader_or_all_rank_transaction_proved']
    assert len(frontier['nodes']) == len(frontier['edges']) == 14
    assert len(variants['variants']) == 2 and variants['new_main_candidates'] == 0
    evidence = ['F21-static-dynamic-loading-contract-v50.json',
                'F21-multiplane-buffers-and-publication-contract-v50.json',
                'F21-unrolled-dynamic-publication-frontier-v50.json',
                'F21-placement-publication-variants-v50.json', 'F21-interface-branch-ledger-v50.json',
                'F21-source-snapshot-index-v50.json', 'F21-postload-adaptor-entry-source-v50.json',
                'F21-postload-updater-transfer-source-v50.json']
    update = dict(
        review_v50=dict(loading=str(O / evidence[0]), publication=str(O / evidence[1]),
            frontier=str(O / evidence[2]), variants=str(O / evidence[3]), frozen_inputs=str(O / evidence[5]),
            source_provenance='Threehistorical166matchedfiles(runner,routedexperts,W8A8producer);threeadaptor/updator/D2Ddependencieslocalreferenceonly.',
            relation='Staticfilealone doesnotselectrunnerdynamicadaptorwarmup. load_model disablesDefaultModelLoaderEPfilter,not a positiveslot-reconciliationproof. ReferenceW8A8MC2registryhas6namedplanes andper-keysharedbufferbanks. Referencewait→CPUmap→log2phy→plane-copy calls are not proven crossstream/native/allrankpublication.',
            implementation=False, new_performance_result=False),
        profile_review_state='STATIC_DYNAMIC_PATHS_AND_REFERENCE_MULTIPLANE_PROTOCOL_ANALYZED;ACTUAL_STATIC_PAIRING_OPEN',
        conclusion_changing_gap='Positive actual static checkpoint-to-physical-to-Ascend slot/payload reconciliation remains unproved;original166adaptor/updator/D2Dhashes and runtime layer/plane/alias/ready/last-reader/capture/generation,capacity/topology/timing stillneeded for actual eligibility. Conditional analysis of the named reference protocol is complete at its declared scope.',
        next_profile_or_contract_action='Do not assume staticfiles triggerdynamicadaptororrepeatthe genericABIabsence claim. Nextread only actualstaticloader/remap edge orattest the three original dependency hashes whenolddataaccessreturns;preservetarget/draftregistryindexdistinctions. Price initialload/dynamicrebalance/formalC8work separately;shared per-keybanks need capacity/fences forconcurrency.')
    modules = {
        'M00': dict(finding='F21initial-loadvariant nowdeclares own phase/denominator. Historicalrunner eplb_enable changesDefaultModelLoaderclassfilter;customstaticfiledoesnotbyitselfenableadaptorwarmup. No startupI/O/copy/time or actualloadereligibility assumed.',
            evidence=evidence, candidates=['F21'], gap='Actualchosenloader/staticpostloadslotpairing andwhole registration/selection/sourceprocess coverage remain;no 0ComputeInfo→0loadcost conclusion.'),
        'M12': dict(finding='F21 weight/map publication traced locally:historicalrunner/routedowner/W8A8producer matched;3newdependency snapshots referenceonly. SixnamedW8A8MC2planes,postload clone/formatgeneration,buffersandstatic-vs-dynamic path distinctions documented.',
            evidence=evidence, candidates=['F21'], gap='Actualcustomstaticreconciliation,originaldependencyhashes,full expertfunctionpayload/adapters/allconsumers,nativeprecision/aliases/maps/timing/topology/capacity remain;local reference proof isnotactual166deploymentproof.'),
        'M23': dict(finding='Reference D2Dtask usesoldCPUmapfor sends,sharedquant-keyreceivebanksandWAITING/READY/TRANSFERRINGgates;requestwait precedesmap/log2phy/weightcopycalls. Allrank/P2P/layer-generationpublication requirements nowexplicit,notmeasuredbandwidth/overlap.',
            evidence=evidence, candidates=['F21'], gap='Originaldependency/currentinstances,actualrank/layer/replica mappings andnativeP2Pcompletion/readerfences/physicalEPtopology/timing remain. ExistingCOMM/MC2/ComputeInfo populations notadded.'),
        'M24': dict(finding='F21same-generationpayload/maplifetime nowlinked to crossiteration scope. Local referenceper-keybuffers reusedacrosslayers;newconcurrencyneedsbanklifetime/capacity. Inplacecopycyclewitness excludesnaiveoverwriteandfrozengraph/reloadviewsneedrebindconditions.',
            evidence=evidence, candidates=['F21'], gap='Oldlastreader/newready/allrankgenerationandcaptureaddresses areunproved. Neither sourcecallorder norstateWAITINGestablishesnativecompletion;no blanket reuse or free pipeline asserted.'),
    }
    remaining = deepcopy(read(O / 'module-coverage-gaps-v49.json')['remaining'])
    target = 'F21 custom-map post-load weight/FP+packed-scale/capture-generation reconciliation, EP16 physical topology and same-call MC2 internal work/wait remain open.'
    assert remaining.count(target) == 1
    revised = ('F21v50static/dynamicboundary nowlocated in historical166runner:staticfileenablesEPfilter-noopbutnotdynamicadaptorwarmup. Conditionalreferenceadaptor/updator/D2Dpath resolves6namedW8A8MC2planes,per-keysharedbanks,state/requestwait/map/log2phy/copyorderanditerationguards;3dependenciesstillreference-only. Actualstaticpostloadslot/payloadpositiveproof andoriginaldependencyhashes,allreader/nativecapturegeneration,EP16physicaltopology/capacity/samecallMC2workwaitremainopen. Do notrepeatallpayloadnames/bufferallocationaswhollyunlocated orclaimstaticauto-reconcile.')
    remaining[remaining.index(target)] = revised
    write(O / 'F21-coverage-gap-reconciliation-v50.json', dict(previous_version=49,
        before=target, after=revised,
        scope='Only supersede the generic post-load/source detail gap with exact version-qualified subquestions;actualstaticreconciliation and wholemodule scope remain incomplete.',
        overall_complete=False))
    remaining.append('F21v50variants:initialcheckpointplacement/filteranddynamiccoherentpublicationrefineexistingF21,notnewmaincandidate. InitialloadcostisnotformalC8decodecost. InreferenceW8A8MC2payloadcontains2weights+FP32/ordinaryscales+2packedscales;namedplanesnotindependentphysicalbytes/allbackendpayloadclosure. BufferlogicalmodelC*sum_distinct_quantkeys(sum_namedplanes(singleexpertlogicalbytes)),notLtimesbufferperlayer;pipeliningneedsnewbanksorprovenlast-use. Nativepublicationedgesandactualdata/pointers/resourcesareUNKNOWN;sourcecallDAG14nodes14edgesandpure2-slot6-planeoverwritecounterexamplearenotglobalmodelclosure/nativebenchmarks. Allprevious35model7framework/0module/849PMU/16phase/3136ambiguity/26theory gaps retained.')
    action = (
        '继续原Goal，v50是F21加载/权重发布边界的实际定点分析进展。先读F21-static-dynamic-loading-contract-v50.json、F21-multiplane-buffers-and-publication-contract-v50.json、F21-placement-publication-variants-v50.json、F21-source-snapshot-index-v50.json及module-coverage-gaps-v50.json。'
        '6源码输入已本地冻结N/F21-source-snapshot-v50；runner568c/routedf9db/W8A8a676为历史166hash匹配，adaptor e372c2.../updator a734e8.../D2Dloader db1023...只有本地依赖参考，不能冒充当前/历史166 attestation。'
        '已核验runner:dynamic_eplb=config值，eplb_enable=dynamic或path不是None；eplb_warmup只在dynamic&&!warmed时构造adaptor。staticfile仅使load_model块把DefaultModelLoader._init_ep_weight_filter改成noop，这是filter证据，不是postloadslot重排证据。不要沿错误前提假定static也走adaptor。'
        '参考W8A8MC2六字段(w13/W2、FP/普通scale、两packedscale)及per-quantkey共享buffer布局已查；不能按每层各一套buffer计数，也不能把namedplanes当独立物理内存。后加载transpose/NZ/clone可能改view代际，adaptor须绑定完成后的同代对象并在reload后更新。'
        '参考D2D状态WAITING→READY→TRANSFERRING→WAITING，wait调用后CPUmap→log2phy→weightcopies；这不是native跨stream原子发布/完成证明。一代14node14edgeDAG与2slot6plane覆盖反例仅说明源序/必要条件，旧lastreader、新ready、allrankgen和capturebank仍需证据。H/A/L调度和record-onlycommitguard已源代码解释。'
        'F21两条件变体为placement-aware初始checkpoint直装/filter、dynamic多plane发布/搬迁重组；startup与动态rebalance与formalC8不同分母，无实现/设备测试/新收益。下一步只追实际staticloader/param physical→Ascend映射正证据，或旧入口恢复后核3依赖hash及bank/reader；不要重读已回答的参考registry/state/planes。也可继续其它模块的有限明确source缺口。'
        '原数据入口仍待用户答复：193root/hostname已验，166/实际105新账号asset-no-match(SSHrc0不是成功)，193已查旧路径无资料不等于全盘无。不重试旧密码<REDACTED_CREDENTIAL>备未执行；按原caps/Eindex/model/input/serve/graph哈希使用旧数据。'
        '35模型7框架0整模块，849缺PMU、16zero-sidecar-candidate phase-unbound、3136模块归属歧义执行、2082regimes、26theoryINCOMPLETE保留；缓存16device24Cube48Vector/1.8GHz参考和28PMU字段已知勿反复采。M21/M22语义族及v49四fallback契约保持。整体Goal未完成，不以报告/局部证明关闭。'
    )
    report = (
        '本轮推进专家重排候选F21的加载与发布契约。6个源码输入已本地冻结：runner、RoutedExperts与W8A8生产者和历史166哈希一致；adaptor、updator及D2D loader只有本地依赖参考，未冒充原166版本核验。没有上传本地资料或连接服务器。\n\n'
        '历史runner源码给出了关键区分：只有dynamic_eplb且尚未预热时，eplb_warmup才创建VllmEplbAdaptor。静态expert_map_path会使eplb_enable成立，并在load_model的一处将DefaultModelLoader的EP权重过滤方法替换为noop；这个过滤变化不是权重槽位已经重排的正证据。静态路径的检查点槽位与Ascend执行映射一致性仍须单独补证，不能假定动态adaptor自动处理。\n\n'
        '本地依赖参考明确了W8A8+MC2的六个命名字段：两份权重、两份FP/普通scale、两份packed scale。缓冲区按量化类型与MC2键共享，同键跨层形状需相等；并非每层独立一套。增加跨层搬迁并发需要额外bank容量或最后使用顺序，不能把它当免费重叠。字段数也不等于独立物理内存或所有后端payload闭包。\n\n'
        '参考D2D路径先用旧CPU map绑定发送与接收，等待请求后依次调用CPU map、log2phy和payload更新，最终回到WAITING。已建立单代14节点/14边调用关系及双槽位六字段覆盖反例。源码调用顺序不独立证明跨stream/native完成或全rank原子发布；仍需旧reader结束、新plane就绪、capture地址与代际一致性。初始加载、动态重平衡和formalC8稳态成本分别定界。\n\n'
        '这些结果细化了现有F21的两个条件方向：面向目标placement的检查点直装/过滤，以及完整payload的动态搬迁发布。没有新增主候选、实现优化或声称时延收益。原始profile入口与实际静态postload重排、三项依赖原始哈希、全部消费者及资源证据仍未补齐。35模型候选、7框架候选、0整模块关闭；整体Goal继续进行。\n'
    )
    jobs = [
        dict(kind='review', status='error', description='Initialpointreader soughtadaptorconstructorinside load_model andhitStopIteration;actualcalliseplb_warmup. CorrectedASTenclosingfunctionbeforeanysharedcheckpointchange;error retained.',
            artifacts=[O / 'F21-postload-reader-error-v50.json']),
        dict(kind='review', status='pass', description='Readexactadaptor/updator/transfer/registry andhistoricalrunner/owner/W8A8produceredges;freeze6inputs with3historical166matches and3local-reference-onlydependencies. Noimport/device/remote work.',
            artifacts=[O / evidence[5], O / evidence[6], O / evidence[7], N / 'read_F21_postload_entry_v50.py', N / 'read_F21_postload_chain_v50.py'] + [Path(f['frozen_path']) for f in snapshots['files']]),
        dict(kind='design-check', status='pass', description='Separate staticfilteredgefromdynamicwarmup;derive6plane/per-keybuffer model and14node14edge unrolledsource/publicationfrontier plus2slotpayloadcyclecounterexample. Nativeedges deliberatelyunproved.',
            artifacts=[O / evidence[0], O / evidence[1], O / evidence[2], N / 'analyze_F21_publication_v50.py']),
        dict(kind='review', status='pass', description='RefineexistingF21withtwophase-qualifiedconditionalvariants;retainpositiveactualstaticreconciliation/dependency/native/process/resourcegaps andpreserveoverallscope.',
            artifacts=[O / evidence[3], O / evidence[4], O / 'F21-coverage-gap-reconciliation-v50.json']),
    ]
    persist_refinement(version=50, previous=49,
        title='F21 static placement versus dynamic multiplane publication',
        summary=dict(filename='F21-placement-publication-summary-v50.json',
            previous_turn_classification='PROGRESS:v49M17fallback_contracts',
            frozen_source_inputs=6, historical166hash_matched_inputs=3, local_reference_only_inputs=3,
            W8A8_MC2_named_planes=6, conditional_variants=2, unrolled_frontier_nodes=14, unrolled_frontier_edges=14,
            actual_static_reconciliation_proved=False, native_generation_publication_proved=False,
            target_implementation_changed=False, original_profile_access_restored=False,
            remote_job_status='No new remote operation or background job in v50. Local pointread initialStopIteration corrected;3pass/1errorregistered. Currentserverhealth/workloadnotpolled.'),
        changes=['Historicalrunner provesstaticfiledoesnotselectdynamicadaptorwarmup;EPfilter-noopisnotpositiveweight-slotreconciliation.',
                 'Version-qualifiedreferenceprotocol binds6namedW8A8MC2planes,per-keysharedbanks,state/iteration andsourcecopyorder;3dependenciesremainunattestedfor166.',
                 'TwoconditionalF21variants/sourcefrontierpreserveactualstatic/native/lifetime/capacitygaps;no implementation/gain/newmaincandidate/moduleclosure.'],
        candidate_updates={'F21': update}, module_updates=modules,
        remaining=remaining, action=action, jobs=jobs, report=report,
        read_first=evidence[:6] + ['original-profile-access-status-v47.json'],
        publication_rows={}, gate_updates=dict(overall_complete=False),
        scripts=[N / 'read_F21_postload_entry_v50.py', N / 'read_F21_postload_chain_v50.py', N / 'analyze_F21_publication_v50.py'],
        executor_prefix='zcode-F21-publication', main_script=Path(__file__), source_read=True)


if __name__ == '__main__':
    main()
