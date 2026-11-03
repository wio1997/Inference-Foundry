"""Persist request provenance, parent/sentinel consumers and saved group metadata."""
from copy import deepcopy
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v5 import persist_refinement


def main():
    assert not(O/'REVIEW-CHECKPOINT-v54.json').exists()
    params=read(O/'frozen-request-parameter-provenance-v54.json')
    src=read(O/'request-resolution-selected-source-v54.json')
    inputs=read(O/'request-resolution-inputs-v54.json')
    groups=read(O/'saved-logical-group-and-env-provenance-v54.json')
    m01=read(O/'M01-semantic-coverage-frontier-v54.json')
    m25=read(O/'M25-semantic-coverage-refinement-v54.json')
    assert len(params['parameters'])==22 and len(src['parts'])==15
    assert len(m01['semantic_families'])==12 and len(m25['semantic_families'])==11
    assert groups['all_group_rank_maps_agree_with_saved_rank_device_map']and len(groups['duplicate_env_rows_differ_in_databases'])==32
    v4=(N/'persist_profile_refinement_v4.py').read_text();v5=(N/'persist_profile_refinement_v5.py').read_text()
    assert v5==v4.replace("scope='Prior chronological rows retained as provenance,not active semantic branches.'",
        "scope='Prior branch inventory retained as provenance; the current semantic rows supersede this inventory.'")
    e=['frozen-request-parameter-provenance-v54.json','request-resolution-selected-source-v54.json',
       'request-guard-saved-profile-v54.json','M25-parent-and-sentinel-contract-v54.json',
       'saved-logical-group-and-env-provenance-v54.json','M01-semantic-coverage-frontier-v54.json',
       'M25-semantic-coverage-refinement-v54.json']
    f35=dict(review_v54=dict(parameters=str(O/e[0]),selected_source=str(O/e[1]),profile=str(O/e[2]),
        relation='Source-selected formalC8 requests set temperature0;greedy init normalizes top_p1/top_k0/min_p0. Requestlogprob fieldsNone/neutralpresence-frequency nowresolved;internal probability/entropy consumers andactual inputmetadata remain separate. Repetition penalty model-dependent;watermark fieldTrue butprocessor unproved.',implementation=False,native_test=False,new_performance_result=False),
        profile_review_state='REQUEST_PARAMETER_PROVENANCE_REFINED;POSTPROCESSING_AND_INTERNAL_CONSUMERS_CONDITIONAL',
        conclusion_changing_gap='Do not repeat omitted requestlogprobs/topfilter/defaultn as whollyunknown. Stillneedactualgenerationrepetitiondefault,watermark/otherprocessor consumption,worker/dummy metadata andall probability/entropy/rawlogit/validity/layout/generation/nativeprecision consumers forF35variants.',
        next_profile_or_contract_action='Use finite saved model-generation metadata and direct sampler-input processor guard consumers only if they change a specific variant conclusion. Do not infer no processors from3recordedframe kinds or greedyargmax.')
    fw={
      'H03':dict(review_v54=dict(parameters=str(O/e[0]),relation='Requestcache_salt/sessionNone and initialpromptlogprob cache-skipFalse source-resolved. No cache-namespace,H,actualhashCPUtime,allocation/generation orcross-requestalias claim.',implementation=False,new_performance_result=False)),
      'H05':dict(review_v54=dict(parameters=str(O/e[0]),parent_sentinel=str(O/e[3]),coverage=str(O/e[6]),
          relation='C8source pathn1/DELTA/include_usageTrue;stopstrings empty,so matcher variant isalternative. Savedengine disables perrequestspecmetrics/routedexpert return. Parent seed/index/duplicatefinal contracts and singleton-empty-vs-pending terminal semantics reviewed;all owner/abort/metadata/transport guards retained.',implementation=False,native_test=False,new_performance_result=False),
          minimal_missing_evidence=['Backend preparability/thread/GIL/onCPU evidence if needed to choose decoder preparation/offload.',
            'Actual serverstreaminterval/samplingmask/othermetadata guards,all later request mutations/consumer lifetimes andtransport/error boundaries.',
            'Producer-ready/consumer latency only for implementation/gain;conditional analysis itself does not require native tests.'])}
    modules={
      'M01':dict(finding='22parameter provenance rows and12partial semantic families replace7chronological rows. Frozenclient body plus originalstage protocol/init resolves temperature0/topfilter normalization,n1,logprobNone,stopstringsEmpty andDELTA/includeUsage;modeldependent repetition andwatermark consumer remainexplicit.',evidence=e[:3]+e[5:6],candidates=[],gap='Finite all-sourceguard/consumer inventory notexhaustive;remaining rendering/length/defaultgeneration/processor/mutation/admission/failure cases remain.',profile_scope='OriginalC8text completion client/profiler data;source-selected defaults distinguished fromhistorical object snapshots.',scope_audit=str(O/e[5])),
      'M02':dict(finding='Initialsource pathpromptlogprobNone makes skip_reading_prefix_cacheFalse;requestsalt/sessionNone. This refines eligibility,not runtimehit/hash/extra namespace orlifetime.',evidence=[e[0],e[1]],candidates=[],gap='H,otherextra keys,actualhashCPUcost,allalloc/recycle/generation/crossrequest consumers.'),
      'M03':dict(finding='Source-selectedn1 bypasses ParentRequest fanout;alternativeparent stream/final-only/seed+index/duplicatefinal contracts nowreviewed. AsyncLLM stilluses possibly updatedrequest.params,soactual inputprocessingmutation remains a separate surface.',evidence=[e[0],e[1],e[3]],candidates=[],gap='Current166EngineCoreCPU/producer-ready/IPC plus all scheduling/mutation/lifecycle branches;historical169 evidence notsubstitute.'),
      'M17':dict(finding='Greedyinit normalizes topfilters aftervalidation;no externalrequest logprobs/neutralpresence-frequency byselectedsource. This doesnoteliminate internal rejection probability/entropy,modeldependent repetition orunknown watermark consumer.',evidence=e[:3],candidates=['F35'],gap='Actualsampler/input metadata andfullprocessor/RNG/alternative/non-greedy consumers,numerics/maps/lifetimes/resources.'),
      'M23':dict(finding='Saved32DB parallelgroupmetadata validates2TP8 membershipsets andEP/MC216 ranks with matching group_rank;physicaltopology/events notimplied. Two conflicting ENV_VARIABLES rowsperDB preserved instead of last-wins.',evidence=[e[2],e[4]],candidates=[],gap='FullphysicalEP16 capacity/links/MC2work-wait andallnative readiness/generation/communication ownership consumers.'),
      'M25':dict(finding='ParentRequest body andSTREAM_FINISHED singleton are no longer missing. Emptybuffer sentinel isidentity-suppressed;pendingpayload+sentinel yields finalpayload. C8n1/DELTA/finalusage source path and2disabled finishmetadata flags refined.',evidence=[e[0],e[1],e[3],e[6]],candidates=[],gap='AllalternateAPI/transport/currentEngineCore/late result/error/cancel/resumable consumers andactual samplingmask/interval/objects remain;11families stillnonexhaustive.',profile_scope='Existing166frontendprofile andsource-selected normaltext completion;no newcapture or historical loaded-object claim.',scope_audit=str(O/e[6])),
    }
    remaining=deepcopy(read(O/'module-coverage-gaps-v53.json')['remaining']);reconciled=[]
    for i,s in enumerate(remaining):
        old=s
        if s.startswith('M17v28nowbinds'):
            s+=' v54request/source provenance nowresolves formaltemperature0,n1,no externalrequestlogprobs,andgreedy-normalizedtopfilters. Originalperstepmaxlogprob/processor/entropy objects stillnotfullyattested;modeldefault repetition andwatermark actualconsumer remain. Do notrepeat requestdefault/provenanceabsence as the samegap.'
        elif s.startswith('M25v53:'):
            s+=' v54ParentRequest selectedbodies andSTREAM_FINISHED definition/empty-vs-pending behavior answered. FormalC8 n1/DELTA/includeUsageTrue;stopstrings empty makes STOPmatcher alternative. Savedconfigspecmetricsnone androutedexpertFalse refineactualmetadata conditions;samplingmask/interval/otherconsumers stayopen.'
        if old!=s:remaining[i]=s;reconciled.append(dict(index=i,before=old,after=s))
    remaining.append('M01v54:12partial semanticfamilies and22request-parameter provenance rows nowrecorded. OriginalclientSHA matcheshistorical;4savedcapturefiles+6currentstage dependencies pinned. Source-selectedC8logprobsNone,n1,no stopstrings/beam/echo,DELTA/finalusage;temperature0 resets top_p1/top_k0/min_p0 aftervalidation. RequestseedNone differs fromengineseed0. No blanket neutralprocessor/localtop1 removal:repetitionmodeldefaults/watermark consumer,serverinterval,samplingmask andlaterrequest mutations remain. Saved32DBlogicalgroup maps consistent;duplicateENVrows differall32,soempty/lastrow not worker-env proof. Module/consumer/theory coverage stillincomplete,not merely missingimplementation/nativebenchmark.')
    write(O/'request-gap-reconciliation-v54.json',dict(changes=reconciled,scope='Retire precise requestdefault/parent/sentinel gaps;retain real originalfullscope/consumer/resource gaps.',overall_complete=False))
    action=(
      '继续原Goal。上轮v53恢复入口/缓存150原资料是PROGRESS，本轮v54补了真正参数/consumer证据，不是重复状态。保持profile优先。'
      '先读frozen-request-parameter-provenance-v54、M01-semantic-coverage-frontier-v54、M25-parent-and-sentinel-contract-v54、saved-logical-group-and-env-provenance-v54。'
      '4个捕获时保存文件(serve/serve-host/log/client)历史hash匹配；6个当前166stage依赖docker-cp读取，serving控制hash匹配v53，不是historical loadedobject快照。32本地immutable DB META_DATA与关键词stack census已读，只有3个logits head/gather frame种类；没有watermarkframe不证明禁用。'
      '22项参数来源：formalC8completiontext,temp0→greedytop_p1/top_k0/min_p0,n1/logprobsNone/stop[]/min0/ignoreEOStrue/DELTA/include_usageTrue。prefixcache初始skipFalse/salt-sessionNone。默认watermarking字段True不等于processor执行；repetition_penalty依赖modelgenconfig，streaminterval依赖server。engine日志perrequestspecmetricsnone、returnedroutedexpertsFalse已明确，别继续把这2个flag全当未知；samplingmask仍单列。'
      'M01改12个非穷尽语义族，M25保留11族并修复parent/sentinel旧缺口。Parent streaming duplicatefinal抑制、FINAL_ONLY按index聚合、seedNone缓存/有seed加index已知；C8n1源路径不走fanout。STREAM_FINISHED在vllm.outputs不是engineinit：空buffer保存singleton→identity过滤；pendingpayload则add终止状态→yieldpayload。重组不能复制哨兵后假设identity仍成立或丢pendingpayload。'
      '32DB的logicalTP为[0..7]/[8..15]各16条，EP/MC2[0..15]各32条，group_rank均匹配rankmap；不证明physicalEP16链路/nativeevent。每DB2个ENV_VARIABLES行都有差异，不能dict最后覆盖推断worker真实env。'
      '下一步选择真正未答有限consumer/branch：优先从缓存原资料定位model生成默认repetition与worker采样metadata/水印字段消费，只有决定F35等变体才点读对应调用函数；否则推进M00-M20/M23-M24尚缺原有限分支/consumer库存和26模块资源上界，不重复F21/P0/helper/M25baseline。保持native概率/entropy和v49backend区分，不能以当前greedy永久排除所有请求/算法分支。'
      '35model7framework0整模块；M21/M22语义族不变，849PMU缺/16unbound/3136歧义ComputeInfo/2082regimes/26theoryINCOMPLETE保留。无实现/新modelTorchNPU/profile/health/process/kill；本地原DB与已授权保存文件足够继续。条件分析不因无实现benchmark自动open，实际原范围未分析不得因成稿完成，Goal active。')
    report=(
      '本轮补齐请求参数与两个直接输出消费者的证据。已读取4个历史哈希匹配的捕获记录和6个定点原166 stage源码文件，并查询全部32份本地数据库的相关META_DATA与调用栈字符串。没有模型、设备、新采集、服务/进程操作。\n\n'
      '22项参数来源中，正式C8请求显式temperature=0；所查初始化会在贪心模式下把top-p/top-k/min-p归一化为1/0/0。n=1、logprobs为空、stopstrings为空、DELTA输出以及最终usage要求已通过客户端和默认参数源码连接起来。engine日志明确关闭逐请求spec metrics及routed-expert返回。请求seed为空不能当成引擎seed=0；内部拒绝采样概率和entropy也不能因外部不返回logprobs而删除。\n\n'
      '保留三个会影响方案的条件：repetition penalty依赖模型生成默认值；watermarking字段默认True但实际processor消费未证实；流式间隔仍取服务器设置。字段值、保存栈和实际参数对象分别记录，没有把缺少watermark调用栈当成禁用证明。\n\n'
      'M01由7条历史时间线改为12个部分语义分支族。M25仍为11族，补读ParentRequest及结束哨兵：parent按模式、child索引和seed处理；空缓冲的结束单例被identity检查过滤，有待发内容时则合并finished并发送payload。复制单例或丢弃所有终止插入会改变行为。当前C8源路径n=1，不走多子请求fanout。\n\n'
      '32份保存元数据确认两个TP8逻辑组及EP/MC216-rank组，group_rank均与rank映射一致；这不证明物理拓扑和native事件。每份数据库两条环境记录值不同，不能按最后一条覆盖后声称是worker实际环境。\n\n'
      '35项模型候选、7项框架候选、0个整模块关闭。仍有真实分支、消费者、进程及资源上界缺口；本轮没有优化实现或收益测试，整体Goal继续。\n')
    input_paths=[Path(x['local_path'])for x in inputs['saved_files']+inputs['source_files']]
    source_paths=list({Path(x['path'])for x in src['parts']})
    jobs=[
      dict(kind='review',status='error',description='Initialboundeddisplayselectedwindows[0] forwarmupwith emptywindows;correctedphaseiteration. Not absent requests/capture proof;no data mutation.',artifacts=[O/'request-window-empty-selection-error-v54.json']),
      dict(kind='review',status='pass',description='4historicalhash-matched savedcapturefiles and6known166stage sourcefiles read;no containerstart/exec/model/device/service query. Source version scope explicit.',artifacts=[O/'request-resolution-inputs-v54.json',N/'request-resolution-source166-v54.json',N/'read_request_resolution_originals_v54.py']+input_paths),
      dict(kind='profile',status='pass',description='Immutable32cachedDB META_DATA env/group rows andrecorded processor-frame census;duplicates retained. Definition incidences are notinvocation counts/CPUtime;missingframe notdisabledproof.',artifacts=[O/e[2],O/e[4],N/'query_request_guard_profile_v54.py']),
      dict(kind='design-check',status='pass',description='22source-parameter provenance rows and15selected sourceparts;greedy normalization/defaults/engineflags separated fromactual object/worker guard proof.',artifacts=[O/e[0],O/e[1],N/'analyze_request_resolution_v54.py']+source_paths),
      dict(kind='design-check',status='pass',description='Parent childindex/seed/mode/duplicatefinal andsingleton-empty-vs-pending termination reviewed. Abstract state witnesses,not native/runtime tests.',artifacts=[O/e[3],O/e[1]]),
      dict(kind='review',status='pass',description='M01twelvepartial families,M25elevenrefined families;precise gap reconciliation andF35/H03/H05 refinement,withoutwholemodule closure ornewmaincandidate. Archivecaption generalized only.',artifacts=[O/e[5],O/e[6],O/'request-gap-reconciliation-v54.json',N/'persist_profile_refinement_v5.py']),
    ]
    persist_refinement(version=54,previous=53,title='Frozen request defaults, parent/sentinel consumers and saved logical groups',
      summary=dict(filename='request-resolution-summary-v54.json',previous_turn_classification='PROGRESS:v53original_access_full_saved_cache_and_output_contracts',
        parameter_provenance_rows=22,selected_source_parts=15,saved_capture_files=4,source_files=6,immutable_metadata_databases=32,
        M01_semantic_families=12,M25_semantic_families=11,original_profile_access_restored=True,
        target_implementation_changed=False,new_performance_result=False,
        remote_job_status='Only saved originalcapture files/dockercp source read completed. Local32DB query completed;no live model/Torch/NPU/capture/containerexec/health/process/kill/backgroundanalysis. One local empty-window selector error retained.'),
      changes=['22parameter provenance rows distinguish C8explicit/default/greedy-normalized fields andactualsavedengineflags fromunknownprocessors.',
        'M01now12partialsemantic families;M25parent/sentinel missing-body gaps resolved withmode/seed/identity/pendingpayload contracts.',
        'Saved32DB logicalTP8/EP16/MC216 maps verified;duplicateenvvalues preserved withoutphysicaltopology/worker-envoverclaim.'],
      candidate_updates={'F35':f35},framework_updates=fw,module_updates=modules,
      semantic_branches={'M01':m01['semantic_families'],'M25':m25['semantic_families']},remaining=remaining,action=action,jobs=jobs,report=report,
      read_first=e,publication_rows={},gate_updates=dict(overall_complete=False),
      scripts=[N/x for x in ['read_request_resolution_originals_v54.py','query_request_guard_profile_v54.py','analyze_request_resolution_v54.py','persist_profile_refinement_v4.py','persist_profile_refinement_v5.py']],
      executor_prefix='zcode-request-resolution',main_script=Path(__file__),source_read=True)


if __name__=='__main__':main()
