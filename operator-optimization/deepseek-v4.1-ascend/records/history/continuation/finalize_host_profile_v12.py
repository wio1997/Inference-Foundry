from datetime import datetime, timezone, timedelta
from copy import deepcopy
import hashlib, json
from finalize_origin_coverage_v9 import N,O,R,W,T,L,read,write,ctl

def receipt(name):
    r=read(N/name);assert r['returncode']==0,(name,r['returncode']);return json.loads(r['stdout'])

def main():
    assert not (O/'REVIEW-CHECKPOINT-v12.json').exists()
    now=datetime.now(timezone.utc);stamp=now.isoformat()
    live=read(O/'v12-live-run-registration.json');assert live['state']=='CAPTURED_AND_RESTORED'
    inv=receipt('host-profile-inventory166-v12-v2.json');origins=receipt('prepare-field-origins166-v12.json')
    phase=receipt('prepare-phase-cohorts166-v12.json');cp=receipt('cp-metadata-shapes-source-pin166-v12.json')
    outputs=receipt('saved-output-cohort-checks166-v12.json');post=receipt('server-readonly-health166-v12-post.json')
    cores=read(O/'device-core-attribute-summary-v12.json');tokens=read(O/'frozen-token-identity-summary-v12.json')
    flags=receipt('host-profile-inventory166-v12.json')['profiler_info_flags']
    assert len(flags)==32 and all(f['flags']['config.common_config.with_stack'] is True and f['flags']['config.common_config.record_shapes'] is True and f['flags']['cann_version']=='9.2.0' for f in flags)
    assert inv['summary']['database_files']==inv['summary']['databases_with_callchains']==32
    s=origins['summary'];assert s['prepare_roots']==s['exact_label_and_pid_tid_matches']==s['prepare_roots_with_unique_same_thread_marker']==96
    assert s['selected_records']==s['selected_with_full_chains']==s['selected_with_project_frames']==96960
    assert s['active_method_counts']=={'dspark':96} and s['active_aux_flag_counts']=={'True':96}
    assert cp['result']['matches_cached_source']
    assert phase['summary']['nonempty_schedule_position_cohorts']==phase['summary']['cohorts_with_unique_eight_ep']==11
    assert outputs['result']['all11_expected_TP8_group_membership']
    assert post['result']['http']['/health']['status']==200 and post['result']['observed_NPU_processes_all_owned_by_base']
    cp_groups=cp['result']['CP_source_groups']
    copies=[g for g in cp_groups if g['name']=='acl_memcpy_host_to_device'];pins=[g for g in cp_groups if g['name']=='aten::_pin_memory']
    assert len(copies)==len(pins)==2 and all(g['records']==2288 and g['roots']==88 for g in copies+pins)
    variants=[
      dict(id='H02_CP_HOST_STAGING',scope='Observed LegacyDSV4.1 DSACP builder131/140;otherPCP/draft paths separate',
           graph='Repeated CPU qsl/suffix construction +pin -> per-builder H2D becomes shared immutable per-call pinned staging -> original per-builder H2D/destinations.',
           keeps='Original fixed GPUquery addresses and group-specific seq_lens computation/consumer order.',
           conditions=['Prove identical complete CPUvalue/dtype/partition key,not only shape or callsite','Keep shared pinned contents immutable until all async DMA readers complete;generation and allocator/event lifetime','Preserve causal/noncausal branch and actual token interval/padding'],status='CONDITIONAL_CANDIDATE_NOT_IMPLEMENTED'),
      dict(id='H02_CP_DEVICE_SUFFIX',graph='Identical CPU suffix +pin +H2D across repeated builds becomes one shared device suffix per matching model-call key;retain group-specific device seq_lens subtraction/clamp/where.',
           conditions=['Same query CPUmap,actual_start/end,num_actual_tokens,causal,dtype/device','Read-only shared suffix and complete stream readiness/lifetime','Device seq_lens remains authoritative after rejection;CPUmirror is not a substitute'],status='CONDITIONAL_CANDIDATE_NOT_IMPLEMENTED'),
      dict(id='H02_CP_DEVICE_RESULTS',graph='If complete producer/version and readers match,share query_start_loc or derived local_seq_lens across repeated builds.',
           conditions=['Prove all common metadata inputs and GPUseq_lens versions match across the26invocations','All aliases/readers and graph parameter updates','Stable query_start_loc address is explicitly required atsource106-107;fresh per-iteration allocation is not justified','Check existing common_v41/cp_global/local batch cache interactions before calling work absent or redundant'],status='STRONGER_SEPARATE_CONDITIONAL_CANDIDATE'),
      dict(id='H02_CP_CROSS_ITERATION',graph='Reuse a previously materialized qsl/suffix only when its complete input/value and generation key is unchanged.',
           conditions=['Request-order/position andactual token interval keys;no shape-only cache','Do not reuse changing post-rejection device seq_lens','Graph bucket,address ownership and all in-flight readers','Requires the relevant host/device reuse contract above;not an independent additive saving'],status='DEPENDENT_CONDITIONAL_CANDIDATE')]
    contract=dict(generated_utc=stamp,profile=dict(nonempty_prepare_roots=88,invocations_per_site_per_nonempty_root=26,
      CP_H2D_records=4576,all_H2D_records=7392,CP_pin_records=4576,all_pin_records=4936,all_sync_records=312),
      source=dict(path='/vllm-workspace/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_v41_cp.py',sha256='039ec9d18e0a8a4aa39f9d5b4bea15204cea3f397a3fce97a5097b266bc9822e',installed_matches_cached=True,
      narrow_cached_reads='25-188:class/allocation/global-sharing guard,two build sites and direct returned local/global fields;no other module bodies read for this question.',
      fields={'131':'query_start_loc.copy_(qsl.pin_memory(),non_blocking=True),persistent INT32graph replay buffer','140':'(authoritative device seq_lens - suffix.pin_memory().to(device,non_blocking=True)).clamp_min(0),then where(query_start_loc difference>0,local_seq_lens,0)'},
      existing_optimizations='cp_global common/batch sharing87-95 andsourceRoPEview reuse156-163 already exist;not counted as new missing work.'),
      shapes=cp_groups,variants=variants,
      relationships=['Host/device variants overlap on the same producers;their hypothetical savings cannot be added.','Device-result andcross-iteration variants require stronger input/version/reader/address proof thanhost staging reuse.'],
      missing_conclusion_changing_evidence=['Across repeated build invocations:complete commonCPUinput andGPUseq_lens generation identity andexisting local batch-cache key/return consumers','Pinned/device storage andall async readers/graphupdate readiness','Unprofiled per-function CPUcost/critical-path contribution;call counts alone do not price the opportunity'],
      legal_fusion_proved=False,critical_path_gain=None,implementation=False,whole_modules_closed=0,overall_complete=False)
    write(O/'H02-CP-reuse-contract-v12.json',contract)
    gate=read(O/'CP-METADATA-CODE-QUESTION-GATE-v12.json');gate.update(answer=contract['source'],conditional_variants=[v['id'] for v in variants],review_complete_for_stated_fields=True,full_consumer_or_alias_closure=False);write(O/'CP-METADATA-CODE-QUESTION-GATE-v12.json',gate)
    framework=deepcopy(read(O/'framework-candidate-ledger-v11.json'));framework.update(generated_utc=stamp,parent_ledger='framework-candidate-ledger-v11.json',overall_complete=False)
    for c in framework['candidates']:
        if c['id']=='H02':
            c['status']='FRESH_ALL16_SAME_CALL_SOURCE_STACKS_SHAPES_BOUND; CP_REUSE_VARIANTS_CONDITIONAL'
            c['fresh_profile_v12']=dict(roots=96,selected_API_records=96960,full_chains=96960,canonical_chains=99,source_groups=121,phase=phase['summary'],source_result=origins['full_result'],sha256=origins['result_sha256'])
            c['CP_reuse_contract_v12']=contract
            c['minimal_missing_evidence']=contract['missing_conclusion_changing_evidence']
            c['profile_evidence'].append('Current166:96exact label/pid/tid-bound prepare scopes;all96960selected APIs retain full chains;7392H2D/4936pin/312sync.131/140 CPsites each2288H2D andpin=26/site in88nonempty roots;integer shapes2/3and4/5. Not all copies redundant.')
        if c['id']=='H03':
            c['status']='CONDITIONAL_UNPRICED; CURRENT_TOKEN_PREFIX_IDENTITY_PROVED; HASH_COST_NAMESPACE_OPEN'
            c['token_identity_v12']=tokens
            c['minimal_missing_evidence']=['Actual hash function calls/CPUcost andresolved H,parent-root/hash-extra namespace for the same workload.','Existing cross-request reuse/invalidation only if measured cost makes it conclusion-changing;do not rediscover per-request incremental hashing.']
            c['profile_evidence'].append('Current166/tokenizeAPI withactual completion special-token default:8distinct32768token prompts share29440token prefix;three share128more;sum seed LCP235904 matchescasehitcounter asconsistency only. No inference orhash CPUcost measured.')
    write(O/'framework-candidate-ledger-v12.json',framework)
    candidates=deepcopy(read(O/'candidate-profile-ledger-v11.json'));candidates.update(generated_utc=stamp,parent_ledger='candidate-profile-ledger-v11.json',review_v12_utc=stamp,candidate_universe_complete=False,overall_complete=False)
    for c in candidates['candidates']:
        if c['id']=='F26':c['runtime_guard_v12']=dict(population='New166samefixedcase',method_dspark_and_aux_true_calls=96,meaning='Supports inspected Ascend v1 direct content-use exclusion;not global dead-buffer proof orhistorical169runtime attestation.',evidence=origins['full_result'])
    write(O/'candidate-profile-ledger-v12.json',candidates)
    module_notes={
      'M00':'Actual marker-bound96calls reportmethoddspark/auxTrue;source-boundCP131/140 isLegacyDSV4.1DSACP builder,not blanketPCP branch proof.',
      'M02':'Current fixedserver inputs nowhaveexact tokenIDs/prefixes;H,hash namespace/cost andlive cross-request reuse remainunpriced.',
      'M04':'All96freshprepare roots bindexact callID/pid/tid;96960selected APIs/fullchains.8zero-token,24containprefillcontext,64all-scheduled-context/spec5decodecompatible.',
      'M05':'CurrenttokenizeAPI usingcompletiondefault reproduces8x32768plus2x29696seed tokens;puretext fixedinputs,not historicalbuffer byteattestation.',
      'M07':'CPquery offsets/device-authoritative local lengths publication hasconditional cross-build producer reuse;per-group consumers/version proof pending.',
      'M08':'Preserve replicatedglobal cache metadata vsCP local query views andactual slot/ring generations;newCPproducer variants do not closecache-state readers.',
      'M09':'CPquery/length metadata feeds multiple cache/index groups;existing commonmetadata andRoPEview sharing retained;onlynewconditional local producers registered.',
      'M10':'CPquery_start_loc hasfixedgraphaddress contract;shared suffix/local lengths requiresall reader/input/version proof andno CPUmirror substitution.',
      'M15':'Runtime96currentcalls dspark+auxTrue supportsF26 directdraftercontent guard;normalized hidden/other getter observers/lifetime stillopen.',
      'M18':'Aux-output flag isTrueon96marker-boundcurrentcalls;CPtargetbuilder source doesnot implyreplicateddraftsameCPpath.',
      'M21':'7392H2D/4936pin/312sync source-linked;two CPsites contribute4576eachH2D/pin. Utilitycopy_to_gpu2064calls stillneedfullcaller field interpretation.',
      'M22':'32profileDBs+32analysisDBs exportedonSHA-verified copies;rawunchanged.96markers/96roots;actualstack+shape flagsTrue;CPquery fixedaddressesandstream/event generationsretained.',
      'M23':'11nonemptycohorts each8uniqueEPandexpectedTP8members byexactrequestIDs/scheduledcounts/CPUpositions/dp;no time/ordinaljoin.16visibledevices report24Cube/48Vector;notpeakrate/utilization.',
      'M24':'Conditionalper-call/cross-iterationCPproducerreuse includesfullvalue/causal/partition/generation key;oldper-request incrementalhashingalreadyexists;tokenprefixidentitynowsupported.'}
    modules=deepcopy(read(O/'module-profile-ledger-v11.json'));modules.update(generated_utc=stamp,parent_ledger='module-profile-ledger-v11.json',whole_modules_closed=0,overall_complete=False)
    matrix=deepcopy(read(O/'module-phase-branch-matrix-v11.json'));matrix.update(generated_utc=stamp,parent_matrix='module-phase-branch-matrix-v11.json',whole_modules_closed=0,overall_complete=False,all_phases_closed=False,all_conditions_closed=False)
    for m in modules['modules']:
        m['phase_branch_record']='module-phase-branch-matrix-v12.json#'+m['id']
        if m['id']in module_notes:m['review_v12']=dict(finding=module_notes[m['id']],evidence=['host-profile-summary-v12.json','H02-CP-reuse-contract-v12.json'],whole_module_closed=False)
    for m in matrix['modules']:
        if m['id']in module_notes:m['branches'].append(dict(branch='Fresh166same-call source/shape/phase andconditional producer reuse',state='SCOPED_PROFILE_AND_TARGETED_CONTRACT_REVIEWED; NOT_WHOLE_MODULE_CLOSED',evidence=module_notes[m['id']],gap='See finite remaining CPfield/consumer/alias/ready/resource andframeworkcost questions.'))
    write(O/'module-profile-ledger-v12.json',modules);write(O/'module-phase-branch-matrix-v12.json',matrix)
    remaining=['H02:interpretall99canonicalfullchains/121sourcegroups intoexactfields;CPcommon input/generation identity,localbatchcache keys,asyncreaders/fixedgraphaddresses;keepfour syncgates untiltheir contracts settle.',
      'H01EngineCoreonCPU/producer-ready/read andH03actualhash/frontendfunction costs remain;newworkerprofile doesnotmeasurethoseotherprocesses.',
      'Use new166all16cohorts toassociate nativegraph/MC2/COMM andmodule resources;GPUvalid masks/routes andcrossrank clockcausality remainunproven. Do not transferold169timings/observervalues.',
      'All26module/branch/cross-operator/layer/card/iteration scan andsame-domain utilization/ceilings/causal lowest-module assessment remainincomplete.']
    overhead=dict(profiler_off_baseline='Temporary hook installed butARMEDfile absent:entry filesystem check thenoriginal function;no per-call metadata/range emission.',
      profiled_case='ARMEDbeforeclient seeds;per-call metadataandCPUrange emission throughoutthecase,includingoutside thetwo shortprofiler windows.',
      conclusion='Timing difference includes both profiler anddifferent diagnostic bookkeeping,plusobservedscheduling/speculation differences;not a cleanprofileroverhead estimate ormodelgain.',
      pinned_inventory_note='Earlierinventory phrase wrapperoverhead inboth means onlytheinactive wrappercheck;it doesnotassert equalmetadata/logging work. Originalpinnedinventory remainsunaltered.')
    write(O/'capture-overhead-contract-v12.json',overhead)
    summary=dict(generated_utc=stamp,status='PROGRESS',run_id=live['run_id'],CANN='9.2.0',raw_capture_bytes=3853367778,
      raw_profiler_infos=32,profile_DBs=32,helper_analysis_DBs=32,all32_actual_with_stack_and_record_shapes=True,
      profiler_off_and_profile_full_case_counters_valid=True,all8_visible_text_equal=outputs['result']['all8_visible_text_equal'],
      exact_same_call_profile=origins['summary'],phase_cohorts=phase['summary'],CP_contract='H02-CP-reuse-contract-v12.json',
      hardware_attributes=cores['interpretation'],current_token_identity=tokens['summary'],
      restoration=dict(status='PASS_ORIGINAL_BASELINE_RESTORED',posthealth_receipt='server-readonly-health166-v12-post.json',posthealth_sha256=post['result_sha256'],health=200,all17NPU_PIDsbaseowned=True,prefix_cache_cold=True),
      saved_server_results=[dict(path=v['full_result'],sha256=v['result_sha256']) for v in [inv,origins,phase,cp,outputs,post]],
      instrumentation_overhead_contract=overhead,remaining=remaining,optimization_implementation=False,performance_gain_claimed=False,whole_modules_closed=0,overall_complete=False)
    write(O/'host-profile-summary-v12.json',summary)
    write(O/'module-coverage-gaps-v12.json',dict(generated_utc=stamp,remaining=remaining,whole_modules_closed=0,overall_complete=False))
    report=f'''本轮取得实质进展：已补采现有数据缺失的输入准备调用栈与shape，原模型服务已恢复，整体分析仍未完成。

当前166在相同模型、镜像、CANN9.2.0和C8/P90/32K/1K输入下，完成预热、关闭profiler对照和采集三轮，三轮固定负载计数均有效。初始及后段decode各最多3次worker迭代，原始数据3853367778B。已在校验副本上离线生成32份workerDB和32份辅助analysisDB，原始数据前后SHA不变。32份实际导出配置均为with_stack=True、record_shapes=True。对照与采集8条可见输出全部一致：{outputs['result']['all8_visible_text_equal']}；这是一轮可见输出检查，不是中间值或全局正确性证明。对照轮只有未激活hook检查；采集轮ARMED后在全case写metadata和标记，即使短profiler窗口之外也有这些开销，两轮耗时差不是纯profiler开销或模型收益。

96个prepare input区间全部按精确标记名称、PID和线程ID对应同调用metadata。选中96960条复制/锁页/同步/to嵌套API，全部保留完整调用栈；按完全相同的有序帧内容归约为99种完整链，仍保留原DB、rowid、chainID和shape/dtype。共有7392次H2D、4936次锁页、312次同步，分121个最内层项目来源组，数量与嵌套时长不能相加当CPU成本或收益。

用每个已调度请求的CPUcomputed位置重新分相：8个零调度token区间、24个含prefill上下文区间、64个所有调度请求上下文至少32768且符合spec5decode条件的区间。非空区间按运行时请求ID、调度数、CPU位置和dp组成11组，每组恰为EP0..7或EP8..15的8卡。没有按邻近时间或ordinal强配，也没有把CPU位置当GPU有效mask。

最具体的新线索在LegacyDSV4.1DSACP的两处局部元数据：131行query_start_loc和140行suffix，各有2288次H2D及锁页，等于88个非空准备区间中每处各26次。对应源文件SHA与冻结缓存一致。定点阅读说明query缓冲区需要固定图地址，而local_seq_lens必须使用拒绝修正后的设备值；cp_global共享和RoPE切片复用已经存在。台账新增独立条件变体：复用不可变锁页暂存、共享设备suffix、更强的共享设备结果，以及完整key不变时的跨迭代复用。尚未证明各次build的完整输入/版本相同，也未闭合所有消费者、别名和事件，不能直接删除26份工作或宣称提速。

另补两项原来缺失的事实。16个可见设备的SDK只读查询均报告24Cube/48Vector，这只是当前运行时资源属性，不是峰值吞吐或模块利用率；UBUF接口虽然返回成功但值为0，不能用作零容量结论。当前模型tokenizer对服务器内8条固定请求得到8个不同的32768token序列，共同前缀29440token，其中3条再共享128token；两条seed各29696token。逐请求seed前缀之和235904与本次前缀命中计数一致，作为一致性检查。Hash的实际CPU成本、H和父hash/extra namespace仍未测得。

原BASE容器、镜像和serve脚本校验通过，health/models均200，17个NPU进程全部归属BASE，诊断容器已退出。恢复后前缀缓存冷启动。下一步直接读取已保存99种完整链和CP合同，核对common字段、现有局部batch缓存、同步门和图地址消费者；再使用新16卡cohort分析模块资源及通信。H01/H03其他进程的CPU函数成本和完整模块/理论界仍有缺口，Goal保持active。
'''
    (O/'PROFILE-FINDINGS-v12.md').write_text(report)
    # Complete the actual live collection Run only after case,source/shape andrestoration checks.
    artifacts=[N/'launch-host-profile166-v12.json',N/'host-profile-poll166-v12-07.json',N/'host-profile-inventory166-v12-v2.json',N/'server-readonly-health166-v12-post.json',O/'profiling-plan-v12.json',O/'host-profile-summary-v12.json']
    args=['finish-run','--run-dir',live['task_run_dir'],'--status','pass','--correctness','not-applicable','--summary','Finiteall16stack+shape capture complete;3fixedcase roundsvalid;96same-call roots and96960fullchain APIs;originalbase restored. No candidate implementation/gain proof.']
    for p in artifacts:args.extend(['--artifact',str(p)])
    ctl(*args)
    jobs=[
      ('review','pass','Fresh166ownership/health guard;three isolated lazyCPU preflights andexactinstalledprofiler constructor review exposedhardcodedwith_stack/omittedshape. Temporaryfactory adapts2collectionkwargs only;32exportedflagsverifyTrue.',[N/'server-readonly-health166-v12.json',N/'host-profile-preflight166-v12c.json',N/'npu-profiler-constructor166-v12.json',O/'PROFILE-CODE-QUESTION-GATE-v12.json']),
      ('profile','pass','Text/raw output requires offline DB export;all32rawprofiles copied andSHA checked,analyse max4/DBmode returns64DBs(32API/callchains+32helper);originalrawunchanged. No new capture/service restart.',[N/'launch-offline-host-profile-db166-v12.json',N/'offline-host-profile-db-poll166-v12.json',N/'offline_host_profile_db_worker_v12.py',N/'host-profile-inventory166-v12-v2.json']),
      ('review','error','Firstread-onlycorequery SDKchildreturned0butemittedtrailingruntime stdout;outerstrictjson.loadsfailedaftervalidJSON. Originalfailure retained;no guessed attributes.',[N/'readonly-core-attributes166-v12.json',N/'query_readonly_core_attributes_v12_166.py']),
      ('review','pass','v2preservesrawstdout/stderr andparsesuniqueJSONresult;16visibledevicesreport24Cube/48Vector/IS_VIRTUAL0. UBUF0notusablecapacity. No init/setdevice/inferenceexplicitcalls;posthealth17baseownedPIDs.',[N/'readonly-core-attributes166-v12-v2.json',O/'device-core-attribute-summary-v12.json',N/'server-readonly-health166-v12-post.json']),
      ('profile','pass','Full96prepare roots/96960selectedAPIs bindexactlabels,pid,tid andhavecomplete chains.99canonicalchains/121sourcegroups;7392H2D/4936pin/312sync;notCPUtime/gain.',[N/'prepare-field-origins166-v12.json',N/'query_prepare_field_origins_v12_166.py']),
      ('review','pass','StrictCPUcomputed-position phase:8zero,24containprefill,64decode-compatible.11nonemptycohorts each8unique expectedTP8EPmembers. Nocrosscapture,nearest-time orordinaljoin;CPUpositionsnotGPUmask.',[N/'prepare-phase-cohorts166-v12.json',N/'saved-output-cohort-checks166-v12.json']),
      ('review','pass','CurrenttokenizeCPUAPI exactfrozen8prompts+2seeds,actualcompletiondefault:29440common tokens plus128for3requests;8distinctfullIDs. SeedLCPsum235904isconsistentwithhitcounter;hashcost/H/namespaceopen.',[N/'frozen-token-identity166-v12.json',O/'frozen-token-identity-summary-v12.json']),
      ('review','pass','TwoCPsites each26copies+pins in88nonemptyroots identified;exactsourceSHA matchescached. Targetedclass/allocation/build/returnreviewbindsQSL/suffix,authoritativeGPUlengthsandfixedgraphaddress;4conditionalproducerreusevariants,existingglobal/RoPEsharingretained.',[N/'cp-metadata-shapes-source-pin166-v12.json',O/'H02-CP-reuse-contract-v12.json',O/'CP-METADATA-CODE-QUESTION-GATE-v12.json']),
      ('review','pass','v12ledgers updatefiniteprofile/phase/source/hardware/token facts;32modelcandidates,26modulesretained withopenalias/consumer/resource questions;0wholemodulesclosed,goalactive.',[O/'host-profile-summary-v12.json',O/'framework-candidate-ledger-v12.json',O/'candidate-profile-ledger-v12.json',O/'module-profile-ledger-v12.json',O/'module-phase-branch-matrix-v12.json',O/'module-coverage-gaps-v12.json',O/'PROFILE-FINDINGS-v12.md'])]
    runs=[dict(run_dir=live['task_run_dir'],status='pass',description='Actual finite v12capture andrestoration Run')]
    for i,(kind,status,description,arts) in enumerate(jobs):
        rid='run-'+(now-timedelta(seconds=len(jobs)-i)).strftime('%Y%m%dT%H%M%SZ')
        run=ctl('new-run','--loop-dir',str(L),'--run-id',rid,'--case-key','c8-input32k-output1k-prefix90','--kind',kind,'--command','Preparedv12evidence execution;retrospective registration: '+description,'--code-ref','Frozen model/frameworkfiles unchanged;independentclone temporarycapturehook only;originalrestored;hashesrecorded','--working-directory',str(N))
        args=['finish-run','--run-dir',run,'--status',status,'--correctness','not-applicable','--summary',description]
        for p in arts:assert p.exists(),p;args.extend(['--artifact',str(p)])
        ctl(*args);runs.append(dict(run_dir=run,status=status,description=description))
    action=('继续Loop004。上一轮v12是实质进展：新166有限全16卡stack/shape补采完毕且原BASE健康恢复，禁止重跑模型或再复制raw。原始3.853GB/32份profilerinfo均实际with_stack/record_shapesTrue，CANN9.2.0；CPU工作副本离线64DB中32有API/fullchains，原raw前后SHA一致。'
      '优先读host-profile-summary-v12.json和H02-CP-reuse-contract-v12.json。保存全证据prepare-field-origins-v12.json SHA c23b75f3fbd3ae3d3d7152f6882524e899bade53001ea2f1befa09ce943f091f：96roots/96exactlabel+pid+tid匹配，96960选中API全有source/fullchains，99canonicalchains/121sourcegroups；每API仍保留DB/rowid/chainID及shape。'
      '用完整saved衍生数据定点读99链/121组而非重查32DB。10个H2D最内来源不是10字段，尤其utils.copy_to_gpu2064次需看完整caller；4同步来源312次分别1495(88)、1867(72)、gpu3849(96)、gpu4987(56)，只补关键field/producer/event合同，不按时长合计或删除。'
      '当前96metadata均dspark+auxTrue，支持F26直接内容读者排除，但不全局删buffer。Phase新文件SHA55ab5a2ebb4775f0af61039b2c2ba6e82457f97dcec79fc0033b30c733f77563：8zero、24含prefill、64全scheduledctx>=32768且spec5decodecompatible；11非空cohort按真实requestIDs+scheduledcounts+CPUcomputed位置+dp，各恰EP0..7或8..15。不得用时间/ordinal配对或把CPU位置视GPU有效mask，不能修补旧169其他rank历史身份。'
      '关键CP新线索：LegacyAscendDSAV41CPMetadataBuilder.build131/140各2288H2D+pin=88非空root每处26次；CPfileSHA039ec9d18e0a8a4aa39f9d5b4bea15204cea3f397a3fce97a5097b266bc9822e已安装匹配，shapeQSL3/5、suffix2/4，int。已注册hoststaging、devicesuffix、strongerdeviceresults和跨迭代4条件变体；不是已实现优化或证实冗余。'
      '下一步定点查共同CPU querymap/CP分区key、实际GPU seq_lens代际及local common_v41_batch_metadata缓存key/返回消费者，判断26次build是否同源/已有memo命中后的死生产者。CPglobal共享87-95和RoPEview156-163已经实现不能重报；QSL106-107固定graph地址，GPU长度在拒绝后为真值，禁止CPU镜像替代。必要才补Legacyhelper的具体小函数，不整库/整元数据模块审计。'
      '然后使用新11TP8cohort的所有32DB关联nativegraph/MC2/COMM/模块窗口与资源，以全范围同口径分析最低模块；newcoreSDK16visibleIDs各24Cube48Vector，只有当前运行时资源数，UBUF返回0不可作0容量，不能当峰值或旧169证明。'
      'H03当前输入token身份证据已补：8不同32768tokens，公共29440、其中3多128，2seed29696，seedLCP和235904一致；仍缺actualhashCPUcost/H/root/extrakeynamespace及真实reuse，勿用HTTPtokenize耗时当hashcost。H01EngineCoreonCPU/ready与前端函数成本另需最小CPU证据，当前workerprofile不覆盖。'
      '所有v12远程任务均terminal；原BASE8757dcc...后健康200且17NPU PIDs全归属，source/serve不改，缓存冷。无新活CPU解析或采集，后续新设备诊断仍先复核占用，已有授权不重复问。源数据都在166自身P/analysis/.../hostprof-20261030T212812Z-360e0c72/analysis；0wholemodulesclosed，Goalactive，未完成全模块/条件/跨边/资源界不标完。')
    checkpoint=dict(status='PROGRESS',previous_turn_classification='PROGRESS:v11outer compute/copy andscopedF26consumer refinements',current_verified_changes=[
      'Freshsamecaseall16stack+shape capture andfullbaseline restoration;32actualprofilerconfigsverified,rawunchangedafterofflineDBcopy/export.',
      '96same-call roots/96960fullchain selectedAPIs/99canonicalchains;11exact schedule-positionTP8cohorts;zero/prefill/decode-compatiblebranches separated.',
      'CPquery/suffix26repeatedmaterializations pernonemptyroot localized andtargeted sourcefixedaddress/device-lengthcontract reviewed;4conditionalvariantsregistered.',
      'Current16device24Cube/48Vectorattribute facts andexact8requesttokenprefix identities supplementhardware/input evidence withoutpeak/gain/hashcostclaim.'],
      summary=str(O/'host-profile-summary-v12.json'),candidate_ledger=str(O/'candidate-profile-ledger-v12.json'),framework_candidate_ledger=str(O/'framework-candidate-ledger-v12.json'),module_ledger=str(O/'module-profile-ledger-v12.json'),branch_matrix=str(O/'module-phase-branch-matrix-v12.json'),report=str(O/'PROFILE-FINDINGS-v12.md'),source_gate=str(O/'CP-METADATA-CODE-QUESTION-GATE-v12.json'),coverage_gaps=str(O/'module-coverage-gaps-v12.json'),
      pending_full_scope_domains=remaining,remote_job_status='All v12capture/CPUexport/query handles terminal;originalbasehealthy.1SDKstdoutparseerrorpreservedandcorrected. No optimization implementation.',next_action=action,whole_modules_closed=0,overall_complete=False,updated_utc=stamp)
    write(O/'REVIEW-CHECKPOINT-v12.json',checkpoint)
    if not (O/'profile-summary-before-v12.json').exists():(O/'profile-summary-before-v12.json').write_bytes((R/'profile-summary.json').read_bytes())
    prof=read(R/'profile-summary.json');prof['supplementary_all16_host_profile_v12']=dict(summary=checkpoint['summary'],population='New166samefrozeninput,CANN9.2.0;separatefromold169',prepare_roots=96,API_fullchain_records=96960,TP8schedule_position_cohorts=11,source_CP_reuse_conditional=True,whole_modules_closed=0,overall_complete=False);write(R/'profile-summary.json',prof)
    for p in [L/'state.json',T/'index.json']:
        d=read(p);assert d['status']in ['EVALUATING','ACTIVE'];d.update(next_action=action,updated_at=stamp);write(p,d)
    for p in [L/'status.md',T/'status.md']:
        ls=p.read_text().splitlines();ls=[('- 下一步：'+action) if line.startswith('- 下一步：') else ('- 更新时间：`'+stamp+'`') if line.startswith('- 更新时间：') else line for line in ls];p.write_text('\n'.join(ls)+'\n')
    live.update(state='CAPTURED_ANALYSED_AND_RESTORED',all_offline_jobs_terminal=True,checkpoint=str(O/'REVIEW-CHECKPOINT-v12.json'));write(O/'v12-live-run-registration.json',live)
    for p in [N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
        d=read(p);d.update(latest_profile_first_review=checkpoint,live_host_profile_v12=live,overall_complete=False,whole_modules_closed=0,updated_utc=stamp);d['next_read_only_action' if p==R/'STATE.json' else 'next_action']=action
        if p==W/'STATE.json':d.update(profile_candidate_ledger=checkpoint['candidate_ledger'],profile_module_ledger=checkpoint['module_ledger'])
        if p in [N/'RECOVERY.json',R/'PARALLEL-RESUME-PACK.json']:d['read_first']=[checkpoint['summary'],str(O/'H02-CP-reuse-contract-v12.json'),str(O/'REVIEW-CHECKPOINT-v12.json'),checkpoint['coverage_gaps']]
        if p==N/'RECOVERY.json':d['original_service']='v12finitecapture completed andoriginalbase restored;posthealth2026-10-30T21:56:36UTChealth/models200,17NPU PIDsallbaseowned,serveSHAunchanged,prefixcachecold;allCPUexport/queryjobsnowterminal.'
        write(p,d)
    compact=ctl('compact','--task-dir',str(T));validation=ctl('validate','--task-dir',str(T));write(O/'task-run-registration-v12.json',dict(runs=runs,compact_output=compact,validation_output=validation,whole_modules_closed=0,overall_complete=False))
    paths=[O/n for n in ['host-profile-summary-v12.json','H02-CP-reuse-contract-v12.json','capture-overhead-contract-v12.json','profiling-plan-v12.json','PROFILE-CODE-QUESTION-GATE-v12.json','CP-METADATA-CODE-QUESTION-GATE-v12.json','device-core-attribute-summary-v12.json','frozen-token-identity-summary-v12.json','framework-candidate-ledger-v12.json','candidate-profile-ledger-v12.json','module-profile-ledger-v12.json','module-phase-branch-matrix-v12.json','module-coverage-gaps-v12.json','PROFILE-FINDINGS-v12.md','REVIEW-CHECKPOINT-v12.json','task-run-registration-v12.json','v12-live-run-registration.json','profile-summary-before-v12.json']]
    paths += [N/n for n in ['server-readonly-health166-v12.json','launch-host-profile166-v12.json','host-profile-poll166-v12-07.json','host-profile-inventory166-v12.json','host-profile-inventory166-v12-v2.json','launch-offline-host-profile-db166-v12.json','offline-host-profile-db-poll166-v12.json','readonly-core-attributes166-v12.json','readonly-core-attributes166-v12-v2.json','frozen-token-identity166-v12.json','prepare-field-origins166-v12.json','prepare-phase-cohorts166-v12.json','cp-metadata-shapes-source-pin166-v12.json','saved-output-cohort-checks166-v12.json','server-readonly-health166-v12-post.json']]
    paths += [N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json',R/'profile-summary.json',L/'state.json',T/'index.json']
    write(O/'checkpoint-manifest-v12.json',dict(generated_utc=stamp,artifacts=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],whole_modules_closed=0,overall_complete=False))
    print(json.dumps(dict(checkpoint=str(O/'REVIEW-CHECKPOINT-v12.json'),runs=len(runs),statuses=[r['status'] for r in runs],validation=validation,all8_text_equal=outputs['result']['all8_visible_text_equal'],whole_modules_closed=0,overall_complete=False),ensure_ascii=False))

if __name__=='__main__':main()
