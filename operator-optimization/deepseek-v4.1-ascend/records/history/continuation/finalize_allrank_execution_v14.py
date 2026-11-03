from datetime import datetime,timezone,timedelta
from copy import deepcopy
import json,hashlib
from finalize_origin_coverage_v9 import N,O,R,W,T,L,read,write,ctl

def receipt(name):
    r=read(N/name);assert r['returncode']==0,(name,r['returncode']);return json.loads(r['stdout'])

def main():
    assert not (O/'REVIEW-CHECKPOINT-v14.json').exists()
    now=datetime.now(timezone.utc);stamp=now.isoformat()
    native=receipt('native-execution-identity166-v14.json');chains=receipt('native-fullchains166-v14-v2.json');graph=receipt('saved-graph-resources166-v14.json')
    sidecar=receipt('graph-sidecar-candidates166-v14.json');unbound=receipt('unbound-graph-host-context166-v14.json');surfaces=receipt('saved-profile-surfaces166-v14.json')
    front=receipt('saved-frontend-trace166-v14.json');parents=receipt('frontend-parent-work166-v14.json');source=receipt('frontend-exact-contracts166-v14.json')
    partition=receipt('execution-partition166-v14.json')
    orphan=receipt('unassigned-compute166-v14.json')
    assert orphan['summary']['unassigned_compute_records']==10
    assert all(len([a for a in r['native_CANN_APIs']if a['connectionId']==r['task']['connectionId']])==1 for r in orphan['records'])
    assert partition['summary']['totals']==dict(compute=286950,native_only=154940,graph_only=132000,both=0,unassigned=10)
    assert native['summary']['compute_executions']==286950 and native['summary']['graph_contained_compute_executions']==132000
    assert native['summary']['graph_compute_multiple_containers']==0
    assert chains['summary']['native_linked_compute_API_records']==chains['summary']['records_with_fullchains']==127920
    assert chains['summary']['canonical_fullchains']==486
    assert graph['summary']['PMU_assigned_TASK_rows']==graph['summary']['PMU_unique_native_id_and_interval_groups']==286101
    assert graph['summary']['PMU_on_noncompute_TASK_rows']==0
    assert graph['summary']['compute_missing_PMU_rows']==849
    assert graph['summary']['marker_sequence_pass']==80 and graph['summary']['marker_blocks']==graph['summary']['blocks_with_one_attention_one_MoE']==3200
    assert graph['summary']['bound_graph_cohorts']==8
    assert sidecar['summary']['previously_unbound_with_unique_candidate']==0 and unbound['summary']['enclosing_scope_count_hist']=={'0':16}
    assert front['summary']['category_counts']['python_function']==39544 and parents['summary']['parent_violations']==parents['summary']['Python_id_duplicate_groups']==0
    refs=[dict(path=v['full_result'],sha256=v['result_sha256'])for v in [native,chains,graph,sidecar,unbound,surfaces,front,parents,source,partition,orphan]]
    write(O/'allrank-execution-evidence-v14.json',dict(generated_utc=stamp,native_summary=native['summary'],execution_partition=partition['summary'],execution_partition_evidence=refs[-2],unassigned_native_context=orphan,fullchain_summary=chains['summary'],graph_PMU_MC2_summary=graph['summary'],MC2_regimes=graph['MC2_regimes'],missing_PMU_ops=graph['missing_PMU_ops'],sidecar_negative_check=sidecar['summary'],unbound_host_scope_negative_check=unbound['summary'],saved_evidence=refs[:6],resource_metric_scope='Native definition+unique timestamp interval associates counters with executions. Reported MAC/vector proxies remain unnormalized; no physical MFU or global-module ranking.',no_optimization_implementation=True,whole_modules_closed=0,overall_complete=False))
    candidates_front=[
      dict(id='H04',title='前端长提示 tokenization 的精确复用、排队与批处理',modules=['M01'],status='MEASURED_FUNCTION_WORK;CONDITIONAL_REUSE_OR_REORGANIZATION;UNPRICED',evidence_kind=['MEASURED','DOCUMENTED'],
        profile_evidence=['Initialfrontendwindow:8_tokenize_prompt records500319.83us inclusive;8nativeencode_batch records478411.39us,all observed ononeworker thread. Parent IDs unique andallresolvededges same-thread/contained.','Laterwindowhasno_tokenize_prompt occurrences; it samples output handling. These are boundedprofiled windows,notfullcaseCPUcost.'],
        source_answer='Renderer582-597 gets tokenizer andencode kwargs/offset option, calls tokenizer(text,**kwargs), thenbuildsTokensPrompt.666-690already applies pre/posttokenization andawaits_tokenize_prompt_async;124-125already usesmake_async withself._executor. ThreadPoolExecutor exists; observedoneencodingthread doesnotattest configuredworker count. No prompt-result cache is visible in these selected bodies; tokenizer/backend caches remainunresolved.',
        candidate_graph='Under a complete immutable tokenization key, reuse identical complete encodings; separately assess whether existing async worker submission/batch encoding can reduce queue exposure. Shared text-prefix reuse is a stronger, separate algorithm requiring boundary correctness.',
        constraints=['Tokenizer/model/revision and actualbackend namespace','Exact text afterpreprocessing;encodekwargs/specialtokens/truncation/offset mapping;postprocessing andfresh request outputs','Concurrency andownership of sharedtokenizer/output buffers;pendingwork/readiness beforechangingpool policy','BPE/normalization boundary semantics for prefix reuse; do not splice pre-encoded prefixes blindly'],
        rejected_overclaims=['Async tokenization is absent.','Existing encode_batch calls prove a missing batch API.','One observed worker means executor max_workers==1.','478ms is request gain or unprofiled CPU time.'],
        minimal_missing_evidence=['Exact existing tokenizer cache/backend andpool_workers resolution,only ifthat distinction changes candidate applicability.','Same-case queue/ready andunprofiledCPUcost for pricing;no mandatory implementation for an explicitlyconditional analysis.'],across=['request admissions','frontend worker threads','repeat requests']),
      dict(id='H05',title='前端 DecodeStream 早段长调用与输出就绪流水',modules=['M25'],status='MEASURED_LONG_MAIN_THREAD_STEPS;CAUSE_AND_REORGANIZATION_CONDITIONAL',evidence_kind=['MEASURED','DOCUMENTED'],
        profile_evidence=['Initialwindow45nativeDecodeStream.step records85366.69us,including8above10ms onmainoutput thread;fullPythonparent chains reachdetokenizer.update→process_outputs→output_handler.','Laterwindow201steps709.56us total,noneabove1ms. Differentphases/work are not an equal-work speed experiment.','Parent graphhas0duplicatePythonIDs/0invalidedges;firstprocess_outputs87627.2us inclusive but279.56us recordedexclusivewall. Thedecoder child,not genericPythonoutput bookkeeping,containsmostobservedwall.'],
        source_answer='Detokenizer169-209already constructs native DecodeStream(ids=request.prompt_token_ids,skip_special_tokens=...).update96-143processeseachnewtoken,appendsdecodedtext andchecksstopcriteria.224-248usesnative.step andretainsinvalidUTF8reset/retry/errorbehavior. Existingnativepromptpriming mustnotbe calledmissing. Longstep cause/laziness/GIL/CPUpreemption notsettled by Pythonwall profile.',
        candidate_graph='Assess preparable decoder-state work and per-request ready publication/off-thread processing, preserving ordered generated tokens and text/stop behavior. Do not remove prompt priming or split UTF8 history without a backend contract.',
        constraints=['Per-request stream/token_ids/output_text mutation order andin-flight results','Same skip_special_tokens/special-spacing/min_tokens/stop-string semantics','Identical invalid-prefix UTF8 recovery andexceptions','Actualbackend/GIL/CPUcost/readiness; movingwork maynot reduceprocessing orrequestlatency'],
        rejected_overclaims=['Native prompt priming is absent.','All8longsteps are independentlyattested firsttokensof8requests.','Rust/DecodeStream lazyprefill is proved.','Exclusive recordedwall isthreadCPUtime orremovablelatency.'],
        minimal_missing_evidence=['Actualnativebackend andearly-step context/cost ifnecessary tochoosebetweenstate-preparation/offload/nochange.','Outputready/consumer timing andsame semantic text/stop contract beforeanimplementationgain claim.'],across=['worker output→frontend','successive accepted-token chunks','frontend event loop andworker threads'])]
    write(O/'frontend-function-contract-v14.json',dict(generated_utc=stamp,profile_summary=front['summary'],parent_validation=parents['summary'],selected_function_measurements=parents['selected'],installed_exact_source=source['result'],variants=candidates_front,source_gate='PROFILE-CODE-QUESTION-GATE-v14.json',saved_evidence=refs[6:9],no_per_function_onCPU_claim=True,no_optimization_implementation=True,whole_modules_closed=0,overall_complete=False))
    framework=deepcopy(read(O/'framework-candidate-ledger-v13.json'));framework.update(generated_utc=stamp,parent_ledger='framework-candidate-ledger-v13.json');assert len(framework['candidates'])==3
    framework['candidates'].extend(candidates_front)
    for c in framework['candidates']:
      if c['id']=='H01':c['frontend_window_caution_v14']='Newpid1frontendmonitor/wait spans are a separateprocess fromoldEngineCoreleaderonCPUdata; do not use them toresolve H01.'
      if c['id']=='H03':c['frontend_window_caution_v14']='Frontendtokenization/detokenization functioncost is nowvisible, butgeneric__hash__ doesnotmeasureEngineCoreprefixhash. H03hashcost/namespace remains separate.'
    write(O/'framework-candidate-ledger-v14.json',framework)
    model=deepcopy(read(O/'candidate-profile-ledger-v13.json'));model.update(generated_utc=stamp,parent_ledger='candidate-profile-ledger-v13.json',review_v14_utc=stamp)
    for c in model['candidates']:
      c['fresh_allrank_evidence_v14']='allrank-execution-evidence-v14.json; current166populationseparatefromold169'
      if c['id']in ['F20','F21','F22','F23']:c['MC2_regimes_v14']=dict(executions=4472,shape_regimes=5,regimes=graph['MC2_regimes'],caution='3/816/834 targetrows and2/3draftrows arestaticinputs,not actualroutes/validmasks/expertcountvalues. Nativecompoundinternalsremainunpriced.')
    write(O/'candidate-profile-ledger-v14.json',model)
    modules=deepcopy(read(O/'module-profile-ledger-v13.json'));matrix=deepcopy(read(O/'module-phase-branch-matrix-v13.json'))
    notes={
      'M00':'32currentDBs/83computeoptypes inventoried; runtimeimplementation selectorsstillrequiresemanticfullchain/configreview,notname-only absence.',
      'M01':'Frontend2savedCPUtracesgive8realtext tokenizationcalls andbuiltinencode_batchcost;existingasyncpathconfirmed, H04conditional reuse/queue variants added.',
      'M04':'Current96prepare scopesremainexact-callbound;newwholewindow127920nativecomputeAPIs/fullchains includeotherworker/dummy/proposalcontexts,notjustprepare.',
      'M06':'All80graph sequences pass40four-mHC-marker blocks;3200blocks eachoneattention/oneMoE. Temporalblocksarestructuralpartition,notallresidualconsumerproof.',
      'M09':'Old169EP3vector242CSVidentitygap wasresolvedinv10;current166nativeTASK/fullchain census alsoretainsallgraph/eagerexecutions. Key/indexreusegenerationremainopen.',
      'M10':'All3200graphblocks containoneSparseFlashMla;native metadata PMUis missing321records andcannotbecalledzeroactivity.',
      'M12':'Currentall16MC24472executions/5shaperegimes includeM3/E24/K6graph3200,M816target640,M834target320,draftM3/E8/K3 216andM2/E8/K3 96. Keepregimesspecific;nativeinternalroutes/validrowdataremainopen.',
      'M18':'Fullcurrentnativecallerframesbind312sharedKV/store chains acrossdummy/proposalcontexts;exactper-callsource/phaseattributionpending,not312formalmodelsteps.',
      'M19':'Currentfullchainsretaindspark/dummy/proposalancestryandnativeper-phasecompute;graphreplayanddraftsourcecontextsremainseparate.',
      'M20':'CurrentfullchainsbindmergeddraftMarkov/postprocesscallers;old65-stepnumericalcontractpreserved. Do notinferparallel/dynamic/vocabflagsbycounts.',
      'M21':'MissingPMU849metadataexecutions splitPadV3 320/SMLAmetadata321/QLImetadata208;nativeexecutionidentitiesretained. PreviousscopedCPdeadproducerproofunchanged.',
      'M22':'80nativeCANN/controlpairscover132000graphexecutions with0duplicatecontainers;64exactdecodecallbindings/8TP8cohorts.16othergraphs have0same-threadscope/sidecarcandidatesandremainphaseunbound.',
      'M23':'All1108212TASKrecordsretained includingnotify/event/SDMA/COMM-like tasks;83computeoptypesareaseparatelayer. COMMoperation/group/order/exposureassociationstillneeded.',
      'M25':'TwoexistingfrontendCPUtraces39544Pythonfunctions nowparsedwithuniqueparentIDs/0invalidedges. DecoderchildworkandJSON/outputfunctionsvisible;H05early-stepreadyvariantadded,notonCPUorlatencygainproof.'}
    for m in modules['modules']:
      m['phase_branch_record']='module-phase-branch-matrix-v14.json#'+m['id']
      if m['id']in notes:m['review_v14']=dict(finding=notes[m['id']],evidence=['allrank-execution-evidence-v14.json','frontend-function-contract-v14.json'],whole_module_closed=False)
      if m['id']=='M01':m['candidate_links']=list(dict.fromkeys(m.get('candidate_links',[])+['H04']))
      if m['id']=='M25':m['candidate_links']=list(dict.fromkeys(m.get('candidate_links',[])+['H05']))
    for m in matrix['modules']:
      if m['id']=='M04':m['phase']='Historical24rootspluscurrent16696exact-call roots:8zero/24prefill-containing/64decode-compatible'
      for b in m['branches']:
        if m['id']=='M09'and'242' in str(b.get('gap','')):b['gap']='Historical242CSVidentitygapresolvedinv10;currentgraphdefinitionsemanticownershipandkey/indexreuse/consumer/generationstillopen.'
        if m['id']=='M25'and b.get('state')=='FUNCTION_PROFILE_MISSING':b.update(state='NOW_OBSERVED_IN_TWO_CURRENT_CPU_TRACE_WINDOWS',evidence='39544Pythonfunctionevents;parentIDsunique/0invalidedges;tokenization,detokenization,JSON/outputpathsretained.',gap='ProfiledwallisnotCPUorrequestgain;fullcasebranch/exposureandnativeearlydecodermechanismremainopen.')
        if m['id']=='M01'and b.get('branch')=='text prompt input':b['gap']='Eightcurrenttokenizationfunctioncallsmeasured;fulltokenizernamespace/cache/readinessandunprofiledcoststillopen.'
      if m['id']in notes:m['branches'].append(dict(branch='Currentall16nativeexecution/resources andsavedfrontendfunctionevidence',state='SCOPED_MEASURED_AND_DERIVED;NOT_WHOLE_MODULE_CLOSED',evidence=notes[m['id']],gap='Semanticmodule/consumer/branchclosureandcriticalresourcequestionsremain;implementationisnotrequiredmerelytoregisteraconditionalanalysis.'))
    modules.update(generated_utc=stamp,parent_ledger='module-profile-ledger-v13.json',framework_candidates=['H01','H02','H03','H04','H05']);matrix.update(generated_utc=stamp,parent_matrix='module-phase-branch-matrix-v13.json')
    write(O/'module-profile-ledger-v14.json',modules);write(O/'module-phase-branch-matrix-v14.json',matrix)
    remaining=[
      'Assign486fullcanonicalnativecallchains andgraphdefinition/outertasks tosemantictarget/draftmodule/branch/cross-boundaryroles. Namesandtemporalblocksarenotallconsumerproof.',
      '10computeexecutionsoutsidebothnativePYTORCH/graphrelationshaveexactCANNlaunchandbackendDequeuecontext;followqueueconnection/EnqueuecounterpartswithoutmixingIDnamespacesorforcingasourcechain.',
      'COMMoperation/group/rank/dtype/count/order andnativeMC2internalwork/actualexpert/validrowdata remainunassociated;staticM3/816/834andM2/3mustnotmix.',
      'PMUcounterdefinitions,units,averaging/frequency/coredenominators andsame-domainmoduleceilings/causallowestrankingremainopen.849metadataexecutionslackPMU,notzeroutilization.',
      '16graphsubmissionshave0namedmarker/hostscope/sidecarcandidates;theiractualclientphasecannotberecoveredfromcurrentrelations. Keepseparatefrom64exactdecodegraphs.',
      'FrontendCPUfunctionvisibilityisnowpresent;H04/H05 conditionalreuse/queue/nativeearlystep-readinessquestionsremain. Existingasyncencoding/nativepromptprimingmustberetained.',
      'H01EngineCoreonCPU/producerready andH03actualprefix-hashcost/namespaceareotherprocessquestions;frontendmonitor/generichashrecordsdonotsettlethem.',
      'All26modules/allfiniteconditions/cross-operator/layer/card/iterationalternatives requirecompletionaudit. Candidateimplementationisnotapreconditionforhonestlyconditionalanalysis;nogainclaimwithoutappropriatevalidation.']
    summary=dict(generated_utc=stamp,status='PROGRESS',previous_turn_classification='PROGRESS:v13CPdeadproducerandfieldcontracts',workflow='SAVED_PROFILE_FIRST_THEN_TWO_EXACT_FRONTEND_FILES',native=native['summary'],execution_partition=partition['summary'],fullchains=chains['summary'],graph_resources=graph['summary'],frontend=front['summary'],frontend_parent_validation=parents['summary'],new_framework_candidates=['H04','H05'],negative_graph_phase_evidence=dict(sidecar=sidecar['summary'],hostscope=unbound['summary']),saved_evidence=refs,remaining=remaining,no_new_device_or_service_work=True,optimization_implementation=False,whole_modules_closed=0,overall_complete=False)
    write(O/'allrank-profile-summary-v14.json',summary);write(O/'module-coverage-gaps-v14.json',dict(generated_utc=stamp,remaining=remaining,whole_modules_closed=0,overall_complete=False))
    report='''本轮取得实质进展：完成当前166全部16卡、两窗口的计算执行与PMU对应清点，并找回已保存的前端CPU函数profile。整体分析仍未完成。本轮没有新设备、模型或服务操作。

32份DB中保留1,108,212条TASK记录和286,950次计算执行。80次图提交全部有原生CANN/control pair，对应132,000次计算执行，无重复容器分配；154,940次计算保留原生API关联，另10次缺少上述两种归属。逐条核对发现它们均有同连接键的CANN launch及后台Dequeue包围范围，但尚无Python源码链；混合ID命名空间带来的额外候选未被采用。127,920条相关提交API全部补齐完整调用链，按完整有序帧归约为486种链、229个最内项目位置。曾因多次分块扫描触及145秒SQLite界限，v2改为每DB一次精确ID集合查询并保存分片，未丢弃身份或源码帧。

286,101组PMU均按同一globalTaskId及唯一包含其时间戳的TASK区间对应到独立计算执行，没有靠最近时间配对，也没有把重复图定义去重。另849次计算缺PMU：PadV3 320、SparseFlashMlaMetadata 321、QLI metadata 208。缺测不能写成零利用率。当前仅给出原始cycle加权MAC/Vector代理；平均、单位、资源分母和物理MFU尚未闭合。

80次图执行均满足40组四个mHC标记的结构，每组恰有一次SparseFlashMla和一次DispatchFFNCombine，共3,200个时间区间。64次图执行精确绑定decode兼容调用，组成8个完整TP8 cohort。其余16次图提交没有同线程包围范围，也没有3,712对完整sidecar调用中的候选；保留阶段未定，不能按相邻次序补成正式decode、dummy或warmup。时间区间也不等于所有张量消费者或源码层对象的证明。

全量MoE为4,472次、5种形状：target图M3/E24/K6有3,200次；target M816和M834分别640和320次；draft M3/E8/K3和M2/E8/K3分别216和96次。不能只用小decode形状分析全部专家流水，也不能把静态shape视为实际有效行、路由或专家接收数量。该证据已补到F20—F23候选。

文件清点发现两份原来未解析的前端CPU trace，现已解析39,544条Python函数记录。全部Python ID唯一，已解析父边均同线程且时间包含成立。初始窗口8次_tokenize_prompt共500.32ms inclusive，底层8次encode_batch共478.41ms；实际观察到同一worker线程，不能据此声称配置只有一个worker。现有代码已经make_async并使用ThreadPoolExecutor，候选转向完整key下的编码复用、排队或批处理；BPE前缀不能直接切拼。

初始窗口45次DecodeStream.step共85.37ms，其中8次超过10ms，完整父链到detokenizer.update→process_outputs→output_handler。后续窗口201次step共0.710ms，工作和阶段不同，不能拿差值当实验收益。process_outputs初始inclusive 87.63ms，减去已验证直接子区间后仅约0.280ms记录内exclusive wall，说明应先查decoder子工作。定点源码确认原生DecodeStream(ids=prompt_token_ids) priming已存在，且保留special-token/stop/min_tokens/UTF8错误恢复合同；不能重报缺失，也未证明长step的内部原因。新增H04编码复用/排队与H05早段decoder就绪流水两个条件候选，均无提速承诺。

前端等待、monitor及inclusive/exclusive wall均不是线程CPU占用；前端generic hash也不是EngineCore prefix-hash。H01/H03其他进程因果/函数成本仍有缺口。下一步直接复用486条完整链、全部TASK及8个cohort，完成模块语义和通信、专家内部与资源分母分析。26模块、跨边及全部条件仍未闭合，Goal保持active。'''
    (O/'PROFILE-FINDINGS-v14.md').write_text(report+'\n')
    gate=read(O/'PROFILE-CODE-QUESTION-GATE-v14.json');gate.update(answered_by='frontend-function-contract-v14.json',source_paths_and_SHA=[dict(path=r['path'],sha256=r['sha256'])for r in source['result']],existing_async_and_native_priming_confirmed=True,whole_backend_or_cache_closure=False);write(O/'PROFILE-CODE-QUESTION-GATE-v14.json',gate)
    jobs=[
      ('review','pass','SavedTASKsetspartition286950=154940nativeonly+132000graphonly+10unassigned,0overlap. Explicit10recordgapretained. Identitiesratherthancountcoincidenceseparatepopulations.',[N/'execution-partition166-v14.json',N/'verify_saved_execution_partition_v14_166.py']),
      ('review','pass','10outsidepartitioncomputeexecutionsqueriedbyoriginalnativeconnectionandsamethreadenclosinghostscopes;allcandidatesretainedwithoutforcinglink.',[N/'unassigned-compute166-v14.json',N/'query_unassigned_compute_v14_166.py']),
      ('profile','pass','All32currentDB native census:1108212TASK/286950compute,80nativegraphcontrolpairs/132000graphexecutions;PMUtimestampcandidateidentitiesretained.',[N/'native-execution-identity166-v14.json',N/'query_native_execution_identity_v14_166.py']),
      ('profile','error','600-ID repeated callchain scans hit145sSQLitebound. No finalderivedoutput;errorretained.',[N/'native-fullchains166-v14.json',N/'query_native_fullchains_v14_166.py']),
      ('profile','pass','SingleexactIDset/DBv2plusverifiedparts completes127920nativecomputeAPIfullchains/486canonicalchains/229leaflocations.',[N/'native-fullchains166-v14-v2.json',N/'query_native_fullchains_v14_166_v2.py']),
      ('profile','pass','Savedrecordsassociate286101PMUgroupstouniquecomputeexecutions;849metadataPMUmissing;80graphsequences/3200blocks/8TP8decodecohorts and5MC2regimes4472rows.',[N/'saved-graph-resources166-v14.json',N/'derive_saved_graph_resources_v14_166.py']),
      ('review','pass','3712completeexecute-sidecarpairs/96epochenvelopevalidations;all64knownmarkerbindingsagree,16unboundgraphsget0candidate. No inventedphase.',[N/'graph-sidecar-candidates166-v14.json',N/'query_saved_graph_sidecar_candidates_v14_166.py']),
      ('review','pass','Explicitsame-threadenclosing-scopequeryreturns0scopesforall16unboundCANNsubmissions;negativeevidencesaved,notdummy/warmupproof.',[N/'unbound-graph-host-context166-v14.json',N/'query_unbound_graph_host_context_v14_166.py']),
      ('review','pass','Runtimefileinventoryfinds2existingfrontendCPUgztraces;no rawlog/requestargsread.',[N/'saved-profile-surfaces166-v14.json',N/'inventory_saved_profile_surfaces_v14_166.py']),
      ('profile','pass','TwosavedCPUtraces39544Pythonfunctions;tokenization/detokenization/output/JSONvisible;whitelistedrelationsonly,notCPU/gain.',[N/'saved-frontend-trace166-v14.json',N/'query_saved_frontend_trace_v14_166.py']),
      ('review','pass','ExplicitPythonparentIDsunique/0invalidedges;fullselectedparentchainsandexclusive-recorded-wallunioncalculationsretainwaiting/phasecaveats.',[N/'frontend-parent-work166-v14.json',N/'derive_frontend_parent_work_v14_166.py']),
      ('review','pass','Profilegated2installedfiles/exactmethodsandSHAs:asynctokenizationalreadyexists,nativeDecodeStreampromptprimingalreadyexists;cache/backend/early-stepcauseunresolved.',[N/'frontend-exact-contracts166-v14.json',N/'read_frontend_exact_contracts_v14_166.py',O/'PROFILE-CODE-QUESTION-GATE-v14.json']),
      ('review','pass','Mainsemanticreviewreconcilesallranknative/PMU/shapeevidenceandfrontendH04/H05conditionalcandidates;historicalstale242gapcorrected;0wholemodulesclosed.',[O/'allrank-profile-summary-v14.json',O/'allrank-execution-evidence-v14.json',O/'frontend-function-contract-v14.json',O/'PROFILE-FINDINGS-v14.md'])]
    runs=[]
    for i,(kind,status,description,arts)in enumerate(jobs):
      rid='run-'+(now-timedelta(seconds=len(jobs)-i)).strftime('%Y%m%dT%H%M%SZ');run=ctl('new-run','--loop-dir',str(L),'--run-id',rid,'--case-key','c8-input32k-output1k-prefix90','--kind',kind,'--command','Preparedv14evidence;retrospectiveregistration: '+description,'--code-ref','Current166savedprofileCPUqueries;2exactfrontendfilequestions;nooptimization/device/servicework','--working-directory',str(N));args=['finish-run','--run-dir',run,'--status',status,'--correctness','not-applicable','--summary',description]
      for p in arts:assert p.exists(),p;args.extend(['--artifact',str(p)])
      ctl(*args);runs.append(dict(run_dir=run,status=status,description=description))
    action=('继续Loop004。v14是实质进展：当前166全部32DB原生执行/PMU/完整调用栈和2份前端CPUprofile已补齐，禁止重跑采集/raw复制。先读allrank-profile-summary-v14.json、allrank-execution-evidence-v14.json、frontend-function-contract-v14.json。'
      'native-execution-identity-v14/index.json SHA a3aa47cdf9170785432ecbc18117a489f3fba37d89959f110f4fcbd8890c7dc4 保存32个DB衍生parts路径/SHA，合计1108212TASK/286950compute=154940nativeAPI关联eager+132000graph+10unassigned，80CANN/controlpairs无重叠。'
      'native-compute-fullchains-v14-v2.json SHA6c7bf4bd3a7dcf557327aeadc4a37ec4b9b23cb1ba675353bbd6e5d56a4e1ca0：127920API全有完整链，486canonical/229leaf，保留DB/rowid/callchainID。v1分块多扫描145s中断已保留，v2一次ID集合+parts完成，不重查32DB。'
      'graph-blocks-PMU-and-MC2-v14.json SHA a035c4de08191656141ad0727e8305a6444cc77e6184d4951f86136021af0991：286101PMU组都唯一同globalTaskId+timestamp落TASK区间且1:1 compute；849缺PMU是PadV3320/SMLAmetadata321/QLImetadata208，勿作0利用率。80图各1650计算/40四mHCmarker区间，3200块各1SparseFlashMla+1MoE；64exactdecode图8完整TP8cohort；另10计算有exactCANNlaunch+backendDequeue但缺Python链，下一步按queueconnection查Enqueue/原CPU父链（CANN与CONNECTION_IDS命名空间不混）；16图0sidecar/0hostscope已核查，不再凭时间/ordinal强配或重查同缺口。'
      'MC24472/5regime：targetgraphM3E24K6 3200，targetM816 640/M834 320；draftM3E8K3 216/M2E8K3 96，静态shape不是真实路由/validmask。优先复用486完整链做target/draft模块语义、跨边与nativeCOMMoperation/group/order/MC2内部成本关联，再定点查SDK counter单位/averaging/core/freq定义，勿把rawMAC/Vectorproxy当MFU/全局最低模块。'
      '前端gz确实存在，已解析39544Pythonfunctions，不能继续说frontend函数profile缺失。frontend-CPU-trace-inventory-v14.json SHA4da035667b2a4dde966b884590ab8d8d0668bf408258ee172be04310329a09d3；frontend-parent-work-v14.json SHAc05481d7acb79ccc6bcf92ec41bde76ee2b9248e5002a69cc27c0af8d19815ee，parentID无重复/0违例。8encode_batch478411.39us同一观察worker；初始45DecodeStreamstep85366.69us含8个>10ms主线程长步，后段201步709.56us。inclusive/exclusive_recorded_wall都非onCPU或收益。'
      'frontend-exact-contracts-v14.json SHAf259edc09966b0028d12fe45cf7051cd525c11232feed0751597cba5b14f14f8：rendererSHA4db44560..已make_async/ThreadPoolExecutor，pool_workers实际值尚未查；2个小tokenize函数没有结果cache但backend缓存未知。detokenizerSHA213d71cf..已DecodeStream(ids=prompt_token_ids)原生priming，update逐token并保留stop/min_tokens/special/UTF8恢复，不能说缺priming或已证明lazy机制。H04编码完整key复用/排队、H05早段decoder就绪流水均条件候选未定价；若关键才定点查pool赋值/TokenizeParams/实际backend或做CPU-only原语诊断，避免全源码audit。'
      'H01EngineCore CPU/producerready与H03prefixhash成本是其他进程，frontendmonitor/generic__hash__不替代。此前CP2200H2D+11000unusedGPUtask结构候选保留，不重新审计；候选实现不是conditionalanalysis登记前置条件，未验证不声称收益。全部v14任务terminal，无新设备/服务操作，lasthealth仍v12post。全26模块/条件/跨层卡迭代/资源界未闭合，Goalactive，0wholemodulesclosed。')
    checkpoint=dict(status='PROGRESS',previous_turn_classification='PROGRESS:v13CPdeadproducer',current_verified_changes=['All32native286950compute/TASKand127920APIfullchains/486canonicalchainscensused.','286101PMU1:1executionbinding/849missing,80graphs/3200blocks/8TP8decodecohorts,5MC2regimes.','Twoexistingfrontendtraces39544Pythonfunctions/parentvalidation;2exactfilesconfirmexistingasync/nativepriming;H04/H05conditionalvariants.'],summary=str(O/'allrank-profile-summary-v14.json'),candidate_ledger=str(O/'candidate-profile-ledger-v14.json'),framework_candidate_ledger=str(O/'framework-candidate-ledger-v14.json'),module_ledger=str(O/'module-profile-ledger-v14.json'),branch_matrix=str(O/'module-phase-branch-matrix-v14.json'),report=str(O/'PROFILE-FINDINGS-v14.md'),source_gate=str(O/'PROFILE-CODE-QUESTION-GATE-v14.json'),coverage_gaps=str(O/'module-coverage-gaps-v14.json'),pending_full_scope_domains=remaining,remote_job_status='Allv14jobs terminal;1boundedSQLinterruptionretainedandcorrected. CPUonly;2exactsourcefiles;noNPU/serviceoperation.',next_action=action,whole_modules_closed=0,overall_complete=False,updated_utc=stamp)
    write(O/'REVIEW-CHECKPOINT-v14.json',checkpoint)
    for p in [L/'state.json',T/'index.json']:
      d=read(p);assert d['status']in ['EVALUATING','ACTIVE'];d.update(next_action=action,updated_at=stamp);write(p,d)
    for p in [L/'status.md',T/'status.md']:
      ls=[('- 下一步：'+action)if l.startswith('- 下一步：')else('- 更新时间：`'+stamp+'`')if l.startswith('- 更新时间：')else l for l in p.read_text().splitlines()];p.write_text('\n'.join(ls)+'\n')
    if not (O/'profile-summary-before-v14.json').exists():(O/'profile-summary-before-v14.json').write_bytes((R/'profile-summary.json').read_bytes())
    prof=read(R/'profile-summary.json');prof['supplementary_allrank_and_frontend_v14']=dict(summary=checkpoint['summary'],compute_executions=286950,fullchain_APIs=127920,PMU_groups=286101,frontend_Python_functions=39544,whole_modules_closed=0,overall_complete=False);write(R/'profile-summary.json',prof)
    for p in [N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
      d=read(p);d.update(latest_profile_first_review=checkpoint,overall_complete=False,whole_modules_closed=0,updated_utc=stamp);d['next_read_only_action'if p==R/'STATE.json'else'next_action']=action
      if p==W/'STATE.json':d.update(profile_candidate_ledger=checkpoint['candidate_ledger'],profile_module_ledger=checkpoint['module_ledger'])
      if p in [N/'RECOVERY.json',R/'PARALLEL-RESUME-PACK.json']:d['read_first']=[checkpoint['summary'],str(O/'allrank-execution-evidence-v14.json'),str(O/'frontend-function-contract-v14.json'),checkpoint['coverage_gaps']]
      write(p,d)
    compact=ctl('compact','--task-dir',str(T));validation=ctl('validate','--task-dir',str(T));write(O/'task-run-registration-v14.json',dict(runs=runs,compact_output=compact,validation_output=validation,whole_modules_closed=0,overall_complete=False))
    paths=[O/f for f in ['allrank-profile-summary-v14.json','allrank-execution-evidence-v14.json','frontend-function-contract-v14.json','PROFILE-CODE-QUESTION-GATE-v14.json','candidate-profile-ledger-v14.json','framework-candidate-ledger-v14.json','module-profile-ledger-v14.json','module-phase-branch-matrix-v14.json','module-coverage-gaps-v14.json','PROFILE-FINDINGS-v14.md','REVIEW-CHECKPOINT-v14.json','task-run-registration-v14.json','profile-summary-before-v14.json']]
    paths += [N/f for f in ['unassigned-compute166-v14.json','execution-partition166-v14.json','native-execution-identity166-v14.json','native-fullchains166-v14.json','native-fullchains166-v14-v2.json','saved-graph-resources166-v14.json','graph-sidecar-candidates166-v14.json','unbound-graph-host-context166-v14.json','saved-profile-surfaces166-v14.json','saved-frontend-trace166-v14.json','frontend-parent-work166-v14.json','frontend-exact-contracts166-v14.json']]
    paths += [N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json',R/'profile-summary.json',L/'state.json',T/'index.json']
    write(O/'checkpoint-manifest-v14.json',dict(generated_utc=stamp,artifacts=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in paths],whole_modules_closed=0,overall_complete=False))
    print(json.dumps(dict(checkpoint=str(O/'REVIEW-CHECKPOINT-v14.json'),runs=len(runs),statuses=[r['status']for r in runs],validation=validation,overall_complete=False),ensure_ascii=False))

if __name__=='__main__':main()
