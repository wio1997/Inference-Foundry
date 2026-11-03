"""Checkpoint restored original data, finite metadata census and output contracts."""
from copy import deepcopy
from pathlib import Path
import hashlib

from finalize_origin_coverage_v9 import N, O, read, write
from persist_profile_refinement_v4 import persist_refinement


def main():
    assert not (O / 'REVIEW-CHECKPOINT-v53.json').exists()
    cache = read(O / 'original-profile-full-cache-index-v53.json')
    frontend = read(O / 'original-frontend-trace-cache-index-v53.json')
    core = read(O / 'original-profile-core-cache-index-v53.json')
    source = read(O / 'original-source-cache-index-v53.json')
    metadata = read(O / 'saved-capture-config-metadata-census-v53.json')
    frontier = read(O / 'M25-semantic-coverage-frontier-v53.json')
    assert cache['completed_files'] == cache['indexed_files'] == 144
    assert cache['all_indexed_files_copied_and_hash_matched'] and frontend['all_raw_hashes_match']
    assert source['all_requested_files_read'] and source['all_six_historical_controls_match']
    assert metadata['profile_memory_false'] == metadata['record_op_args_false'] == 32
    assert len(frontier['semantic_families']) == 11 and not frontier['universe_exhaustive']
    assert read(O / 'original-profile-access-status-v53.json')['original_profile_access_restored']
    # v4 changes only the old EP-specific semantic-inventory display label.
    v3=(N/'persist_profile_refinement_v3.py').read_text()
    v4=(N/'persist_profile_refinement_v4.py').read_text()
    assert v4==v3.replace("bm[mid]['source_scope']='Semantic inventory with separate current/historical scopes; old EP3 entry hints preserved in '+str(history)",
        "bm[mid]['source_scope']='Scoped semantic inventory with explicit remaining gaps; prior chronological evidence preserved in '+str(history)")
    evidence = ['saved-capture-config-metadata-census-v53.json','restored-source-point-contracts-v53.json',
        'F21-initial-map-and-attribute-helper-contract-v53.json','M25-output-merge-and-terminal-contract-v53.json',
        'M25-conditional-output-reorganizations-v53.json','M25-semantic-coverage-frontier-v53.json',
        'original-profile-access-status-v53.json','original-profile-full-cache-index-v53.json']
    f21 = dict(review_v53=dict(source=str(O/evidence[1]), initial_map_helper=str(O/evidence[2]),
        source_cache=str(O/'original-source-cache-index-v53.json'),
        relation='Original166stage files now match the previously reference-only ModelSlim andEPLB dependencies. P0 is identity then round-robin redundant copies;set_weight_attrs asserts absence,may sync-wrap weight_loader,then captures callable. Inspected adapter postload delegates. Actual deployed objects/placement/payload pairing remains conditional.',
        implementation=False,native_test=False,new_performance_result=False),
        profile_review_state='ORIGINAL_SOURCE_P0_AND_ATTRIBUTE_HELPER_REVIEWED;STATIC_PAIRING_CONDITIONAL',
        conclusion_changing_gap='P0/helper bodies and current166stage dependency bytes are answered. Remaining actual chosen config/platform/callback objects,current placement/maps/fullRoute/payload pairing,all readers/native events/capture generations andresource costs. Do not repeat missing P0 body or unmatched five reference dependencies as current gaps.',
        next_profile_or_contract_action='Use cached original32databases/full callchains/PMU and bounded saved deployment records for actual guards/values. Do not infer historical loaded objects from currentstage bytes or expand broad F21source trees.')
    h05 = dict(review_v53=dict(contract=str(O/evidence[3]),variants=str(O/evidence[4]),coverage=str(O/evidence[5]),
        relation='Existing collector merge/chunk/yield/native prompt priming are baseline. Per-request owner/order/stop/terminal/error/metadata contracts bound;four conditional reorganizations refine H05 without a new maincandidate.',
        implementation=False,native_test=False,new_performance_result=False),
        candidate_graph='Conditional decoder preparation,per-request serial offload/owner publication,post-semantic emission batching or compatible stop matcher. Preserve current state/terminal/structured-output boundaries;no equal-work gain attested.',
        minimal_missing_evidence=['Actual backend preparability/thread/GIL behavior only if needed to choose offload/preparation.',
            'Actual optional output fields/modes/aliases and all parent/cancel/resumable/transport consumers.',
            'Producer-ready/consumer timing and measured latency if implementation/gain is pursued. Conditional analysis does not require native implementation.'])
    modules = {
        'M00':dict(finding='Original166stage callback/EPLB dependency bytes are now available;P0/helper/platform-sync-wrapper source contracts resolved. Actual selected objects/config and historical loaded code remain separate.',evidence=evidence[:3],candidates=['F21'],gap='Whole initialization/model/op selection and actual config/callback/placement/payload consistency.'),
        'M12':dict(finding='P0[p]=p for p<E,else(p-E)%E;known helper captures supplied or platform-synced callback without overwriting existing fields. Adapterpostload onlydelegates in selected method. Static placement still needs all-plane route/payload commutation.',evidence=evidence[1:3],candidates=['F21'],gap='Actual map/route/config/Param callback/payload generations and other backend consumers;no actual static failure claim.'),
        'M24':dict(finding='Tensor callback attributes are captured at registration;later layer callback rebind does not rewrite existing attributes. Output offload also needs request-generation/owner guards after cancel/cleanup.',evidence=evidence[1:6],candidates=['F21'],gap='All state/capture/storage/request generations,late results,last readers and native publication evidence.'),
        'M25':dict(finding='11partial semantic families replace10chronological review rows. Existing pending merge/chunk/yield,mode/interval,stop ordering,structured-field merge,abort/resumable/cleanup and selected completion SSE observers reviewed;four conditional H05variants recorded.',
            evidence=evidence[3:6],candidates=[],gap='Inventory is explicitly nonexhaustive;parent body,remaining tokenizer/API/transport/error/IPC consumers,actual guards/objects and currentEngineCore causal costs remain.',
            profile_scope='Original166frontend traces39544Python records remain separate from16workerPIDs and historical169EngineCore. Current166stage source matches prior localreference hashes;only6historicalcontrols directly matched.',
            scope_audit=str(O/'M25-semantic-coverage-frontier-v53.json')),
    }
    remaining = deepcopy(read(O/'module-coverage-gaps-v52.json')['remaining'])
    reconciled=[]
    for i, text in enumerate(remaining):
        old=text
        if text.startswith('Nextoriginal-profile access:'):
            text='Original166 access restored v53 using authorized xinyufeng@root@10.55.0.166;hostname105/root and fourhistoricalcoreindex hashes verified.144indexed original files plus4coreindices and2frontendtraces cached with historicalhashes. No original-data access question remains pending. Other unindexed original artifacts are not claimed fully copied.'
        elif text.startswith('v48evidence-capability audit:'):
            text='v48capability audit remains useful;v53executes the prepared immutable savedmetadata query on original166. All32configs enable shapes/stack and disable memory/opargs;CANN9.2/torch-npu2.10.post7 confirmed. All32DB stats unchanged;5metadata table schemas/counts present32times,META_DATAvalues intentionally unread. Neither flags nor rank/session clocks establish aliases/versions,physicalpeaks or phasebinding. Preserve849missingPMU,16zero-sidecar-candidate graphs and finite branch/consumer/process/theory gaps.'
        elif text.startswith('F21v50static/dynamicboundary'):
            text+=' v53supersedes dependency/P0/helper absence:five reference dependency hashes match current original166stage;P0 constructor andset_weight_attrs body read;adapterpostload delegate read. P0 is identity+round-robin redundancy,helper may platform-sync-wrap. Currentstagebytes are not historical Paramobject/config/route/payload or complete native publication proof.'
        elif text.startswith('F21v51sourceflow'):
            text=text.replace('P0constructorbody/globalcallbackbinding/actualtablevaluesandremaininghooksstillunproved.', 'P0constructorbody answered v53;globalcallbackbinding/actualtablevaluesandremaininghooksstillunproved.')
        elif text.startswith('F21v52fiveinput sourcecase:'):
            text='F21v52registration/producer analysis retained. v53nowreadsoriginal166stage ModelSlimconfig/adapterand3EPLBfiles with hashes matching prior references;P0/helper/adapterpostload source questions answered. set_weight_attrs asserts no overwrite andmay wrap weight_loader by platform before setattr;actual platform flag and Paramobject callback still unproved. P0 identity then round-robin redundancy does not consume staticfile. No known namewrapper/selectedadapter postload staticreconciliation. Keep actual maps/fullRoute/payload consistency,allbackend hooks/readers/native generations andphase costs.'
        elif text.startswith('Current166 EngineCore/producer-ready/'):
            text='Current166 EngineCore/producer-ready/prefix-hash CPU costs remain absent from the analyzed saved surfaces. Continue original saved data first;v53does not authorize or perform a new capture/health/process query. A decisive missing process surface must remain explicit rather than borrowing historical169cost or zero-filling.'
        if text!=old:
            reconciled.append(dict(index=i,before=old,after=text));remaining[i]=text
    assert len(reconciled)>=5
    remaining.append('M25v53:11partial semantic families,4conditional H05variants;collector DELTA merge and snapshot replacement/chunk-yield/priming alreadybaseline. Stop earliest-end/list-order ties and per-chunk stop boundary answered;END/ENDx abstractwitness gives same visibletext but3vs4IDs if blindprefusion. MatchedDELTA add doesnot explicitly refresh finish-only sampling_mask/routed_experts/spec_decode_metrics;conditional enabled-field/pendingmerge witness,not actualC8failure. One-shot promptlogprobs,emptydelta/terminal,parent/resumable/cancel/error/owner andselectedcompletionSSEusage/DONE observers bound. Wholefamily/sourceguard/consumer/API/transport/currentEngineCore coverage remainsopen;no native test/implementation/gain required for conditional proposals.')
    write(O/'gap-reconciliation-v53.json',dict(changes=reconciled,scope='Retire access/P0/helper/currentstageidentity and full32config-census gaps only;do not close originalwholemodule or actualobject/native requirements.',overall_complete=False))
    action=(
        '继续原Goal，保持profile优先、不能由profile定案才定点源码。v53已用用户授权xinyufeng账号恢复10.55.0.166/actualhostname105的root入口；原caps/Eindex/graph/Mindex4哈希匹配。'
        '本地N/original-profile-cache-v53有144个indexed原资料(32rawDB含完整调用栈+32native+32PMU+32config+16hostmetadata)、4coreindices、2frontendtrace共150文件，逐个历史hash匹配；不等于原目录所有文件已复制。优先使用本地immutable数据库分析，避免重复登录/重采。'
        '读取saved-capture-config-metadata-census-v53：32份CANN9.2/torch-npu2.10.post7、shapes/stack开、memory/opargs关已完整确认，准备态查询现已实际完成。5小metadata表面已读schema/count/boundedrows，META_DATA值没读。不能以此声称alias/generation/nativephysicalpeak/16unboundphase问题已解。'
        'N/original-source-cache-v53缓存20文件，6historicalcontrol匹配；其余是当前stage来源，不能冒充historicalloadedParam对象。F21先读initial-map-and-attribute-helper-contract-v53：P0身份+轮转冗余；helper禁止覆盖、可platformsyncwrap、再setattr；adapterpostload只delegate。5旧reference依赖与当前166stagehash匹配。勿重复P0/helper/已知wrapper/registry问题；实际config/platform/Paramcallback/maps/fullRoute/payload/nativepublication仍条件。'
        'M25先读output-merge-and-terminal-contract-v53、conditional-output-reorganizations-v53和semantic-coverage-frontier-v53。11个部分语义族已替换10历史时间行并归档；不是穷尽branch/guard/consumer。基线已含collector合并/chunk-yield/fastget/nativepromptpriming。DELTA合并字段、mode/offset、stop最早结束/同终点列表顺序、逐chunk语义边界和取消/owner/metadata/SSE观察者已答，4条件方案细化H05。'
        '下一步用完整原DB/调用栈和缓存metadata补真正未覆盖guard/consumer/跨层跨卡跨迭代机会及26模块theory上界；必要时选剩余有限M25 parent/IPC/API路径，不重复已答contract或泛读整库。长nativeDecodeStream.step的Rust/GIL/onCPU原因仍没被wallprofile定案，不先归因。'
        '35模型7框架0整模块关闭；849缺PMU/16phase-unbound/3136模块歧义ComputeInfo执行/2082regimes/26theoryINCOMPLETE保留。M21/M22语义族与v49fallback契约保持。无模型/torch/NPU/新profile/服务health/process查询/kill/实现；后续优先读现有资料。源数据/条件局部分析可完成，不因无实现nativebenchmark而一律开着，也不得因报告成稿关闭实际未分析全范围，Goal active。'
    )
    report=(
        'xinyufeng已成功连接10.55.0.166，root主机为bm-jn-zs-zone1-910C-64G-10-105。此前fanyichao可登录堡垒机但无166匹配资产，不能据SSH返回0认定登录目标成功；这段失败及菜单查找保留为历史记录。原profile四个关键索引的哈希与历史台账一致。\n\n'
        '已缓存150个原资料文件：32份原始数据库及完整调用栈、32份native执行记录、32份全部PMU导出、32份采集配置、16份host metadata、4个关键索引和2份frontend trace，逐文件与历史哈希核对。这里只覆盖指定索引内文件，没有宣称原目录全部复制。32份配置均为CANN9.2.0、torch-npu2.10.0.post7，开启shape/stack、关闭memory/opargs；补齐了此前只读首份配置的证据缺口。\n\n'
        '读取20个定点原166 stage容器源码文件，6项历史源码控制哈希匹配，未启动容器进程。此前参考的5个ModelSlim/EPLB依赖与当前stage内容一致。P0构造和属性helper已定位：P0为原expert后接轮转冗余；helper禁止覆盖、可包装同步loader后绑定。所查MoE适配层后处理只委托quant_method。实际配置、历史加载对象、部署map/route/payload和native事件仍需单独证据。\n\n'
        '输出模块以11个部分语义分支族取代10条时间线记录，原记录已归档。确认基线已有结果合并、分块处理、让出事件循环和prompt priming；新增优化不能重复计数。4个条件方案细化现有H05：预备decoder状态、请求内串行卸载后由owner发布、语义处理后的输出批量化、保持停止规则的可选matcher。\n\n'
        '两个局部反例限定重组边界：停止检查前盲目拼接chunk，即使可见文本相同，也会改变token-ID历史；已有DELTA合并并不显式复制所有结束时才生成的字段，实际是否触发取决于开关、pending状态和对象共享。它们不是当前C8负载故障证明。取消、请求代际、单次prompt logprobs、parent/resumable输入、usage和[DONE]等观察者均已登记约束。\n\n'
        '35项模型候选、7项框架候选、0个整模块关闭。仍有真实分支、消费者、进程、资源上界与跨边界范围未完成；Goal保持进行中。未启动模型、Torch、NPU、新采集或服务/进程检查，没有优化实现或新增收益测试。\n'
    )
    cache_paths=[Path(x['local_path'])for x in cache['files']+core['files']+frontend['files']]
    source_paths=[Path(x['local_path'])for x in source['files']]
    prelim=read(O/'M25-output-lifecycle-source-v53.json')['files']+read(O/'M25-output-terminal-dependency-source-v53.json')['files']
    scripts=[N/x for x in ['probe_original166_new_account_v53.py','inspect_new_account_bastion_menu_v53.py',
        'probe_original166_xinyufeng_v53.py','mirror_saved_profile_core_v53.py','mirror_saved_profile_full_v53.py',
        'read_original_source_archive_v53.py','read_M25_output_lifecycle_v53.py','read_M25_output_dependencies_v53.py',
        'analyze_restored_profile_and_outputs_v53.py','persist_profile_refinement_v4.py']]
    jobs=[
        dict(kind='review',status='error',description='Fresh fanyichao exact166asset lookup failed despiteSSHrc0;no hostname/profile returned. Retained application failure,not authenticated target success.',artifacts=[N/'original166-new-account-recheck-v53.json',O/'original-profile-access-before-xinyufeng-v53.json']),
        dict(kind='review',status='pass',description='Authorized account/menu recheck and xinyufeng original166root identity succeeded;owned failed mux closed;no other session or server process affected.',artifacts=[N/'original166-xinyufeng-recheck-v53.json',O/'bastion-new-account-asset-menu-v53.json',O/'original166-recheck-owned-mux-cleanup-v53.json',O/evidence[6]]),
        dict(kind='profile',status='pass',description='Saved-only original data cache:150files,all historicalhashmatches;includes32full rawDB/callchains,32native,32PMU,32configs,16metadata,4indices,2frontendtraces. No recollection.',artifacts=[O/evidence[7],O/'original-profile-core-cache-index-v53.json',O/'original-frontend-trace-cache-index-v53.json',O/'original-profile-cache-plan-v53.json']+cache_paths),
        dict(kind='profile',status='pass',description='Immutable original32DB metadata and32config census executed with sourcehash/stat guards;all32memory/opargs off,shape/stack on;no phase/alias/peak overclaim.',artifacts=[N/'saved-metadata-config166-v53.json',O/evidence[0],N/'query_saved_metadata_and_capture_flags_v48.py']),
        dict(kind='review',status='pass',description='Exactexisting166stage docker-cp-only source read:20files,6historicalcontrol hashes match;no docker exec/container start/model import. P0/helper/postload andoutput directcontracts selected.',artifacts=[N/'original-source-files166-v53.json',O/'original-source-cache-index-v53.json',O/evidence[1],O/evidence[2]]+source_paths),
        dict(kind='review',status='error',description='M25stop helper first looked inengine/utils.py;corrected to localdetokenizer function. Retained selector error,not absence proof.',artifacts=[O/'M25-stop-callee-reader-error-v53.json']),
        dict(kind='review',status='error',description='M25streaming_input initially selected as method;corrected actualfield/constructor/update paths. Retained local selectorerror.',artifacts=[O/'M25-stream-input-selector-error-v53.json']),
        dict(kind='design-check',status='pass',description='Selected output lifecycle contracts:baseline merge/chunk/yield,perchunk stop andfield-merge counterexamples,mode/interval/abort/owner/SSE observers. Independent abstract witnesses,not model/native tests.',artifacts=[O/evidence[3],O/'M25-output-lifecycle-source-v53.json',O/'M25-output-terminal-dependency-source-v53.json']+list({Path(x['frozen_path'])for x in prelim})),
        dict(kind='review',status='pass',description='11partial M25semantic families and4conditional H05variants;originalaccess/config/P0/helper gaps reconciled without wholemodule closure. Generic semantic displaylabel helper correction only.',artifacts=[O/evidence[4],O/evidence[5],O/'gap-reconciliation-v53.json',O/'F21-pinned-source-edge-availability-v53.json',N/'persist_profile_refinement_v4.py']),
    ]
    persist_refinement(version=53,previous=52,title='Original166 access and full saved-profile cache;output lifecycle contracts',
        summary=dict(filename='restored-profile-and-output-summary-v53.json',previous_turn_classification='PROGRESS:v52quant_callback_and_plane_producers',
            original_profile_access_restored=True,cached_original_files=len(cache_paths),cached_original_bytes=sum(x['bytes']for x in cache['files']+core['files']+frontend['files']),
            source_files=20,historical_source_controls=6,full_capture_config_census=32,M25_semantic_families=11,H05_conditional_variants=4,
            target_implementation_changed=False,native_test=False,new_capture=False,
            remote_job_status='Original166saved-file copy/immutable SQL/dockercp completed. No new model/Torch/NPU/capture/container process/service health/process query or kill. Local selector2errorsand fanyichaoassetfailure retained;no analysis worker left running.'),
        changes=['Original166root access restored withxinyufeng;150savedfiles cached withhistoricalhashes,including32full rawDB/callchains.',
            'Full32config census and missing originalstage P0/helper/adapter contracts nowanswered;actualobject/native gaps retained.',
            'M25now11partial semanticfamilies,4conditional H05variants;existing merging/chunk-yield excluded fromnewgains,stop/terminal/metadata/owner boundaries refined.'],
        candidate_updates={'F21':f21},framework_updates={'H05':h05},module_updates=modules,
        semantic_branches={'M25':frontier['semantic_families']},remaining=remaining,action=action,jobs=jobs,report=report,
        read_first=evidence,publication_rows={},gate_updates=dict(overall_complete=False),scripts=scripts,
        executor_prefix='zcode-restored-profile',main_script=Path(__file__),source_read=True)


if __name__=='__main__':
    main()
