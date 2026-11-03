from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, copy

N=Path(__file__).parent
O=N/'profile-first-20261031'
R=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
W=R/'whole-scope-20261030'
def read(p): return json.loads(p.read_text())
def write(p,d):
    assert not p.exists(),('refuse_overwrite',str(p))
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def receipt(name):
    d=read(N/name); assert d['returncode']==0,(name,d['stderr']); return json.loads(d['stdout'])

def main():
    now=datetime.now(timezone.utc).isoformat()
    audit=read(O/'framework-profile-audit-v8.json'); s=audit['summary']
    assert s['prepare_roots']==24 and s['partition_checks_pass']==24
    assert s['original_recomputed_gap_match_disagreements']==0
    assert s['independent_trace_gap_prepare_checks']==s['independent_trace_gap_prepare_pass']==22
    source=receipt('framework-source-slices166-v8.json')
    stacks=receipt('archived-framework-speedscope-callsites166-v8.json')
    request=receipt('framework-request-contract166-v8-v2.json'); q=request['result']
    assert q['hash_equal_to_framework_result'] and q['source_stats_unchanged']
    assert len(q['request_rows'])==8 and len(q['seed_rows'])==2
    assert all(r['kind']=='text' for r in q['request_rows'])
    assert len({r['prompt_sha256'] for r in q['request_rows']})==8
    assert read(N/'framework-request-contract166-v8.json')['returncode']==1
    client=R/'diagnostic-20261029/official_profile_client.py'
    assert hashlib.sha256(client.read_bytes()).hexdigest()=='42a6d9e54ea0a922843e8ab37f4096e52c54fb4dbcdd23b77bb60d69293f3131'
    controller=R/'diagnostic-20261029/official-analysis/enginecore_framework_controller.py'
    request_summary=dict(input_transport='text',semantic_input_tokens_each=32768,requests=8,prefix_seed_requests=2,
        question_char_lengths=[r['length'] for r in q['request_rows']],distinct_full_prompt_hashes=8,
        common_text_prefix_chars=min(q['question_prefix_matrix'][i][j] for i in range(8) for j in range(8) if i!=j),
        original_framework_prompt_sha256=q['input_question_list_json_sha256'],hash_binds_original_framework_run=True,
        formal_DP_header='not supplied by the frozen request rows; actual dispatch must come from runtime evidence',
        seed_DP_headers=[r['dp_rank'] for r in q['seed_rows']],
        client_body_contract={'temperature':0,'ignore_eos':True,'max_tokens':1024,'stream':True,'stream_options':{'include_usage':True},
                              'logprobs':'not explicitly requested','top_p':'not explicitly supplied','seed':'not explicitly supplied'},
        client_contract_sources=[dict(path=str(client),sha256=hashlib.sha256(client.read_bytes()).hexdigest(),lines='36-41,133-135'),
                                 dict(path=str(controller),sha256=hashlib.sha256(controller.read_bytes()).hexdigest(),lines='168-172,216-219,256-259')],
        fixture_seed_metadata=q['seed_fixture_declared_fields'],
        limits=['Character prefix does not prove a token-prefix boundary or identical hash extras.',
                'Fixture seed shared_prefix_tokens=29568 is declared metadata, not measured per-request cache hits.',
                'Body fields are the frozen diagnostic client contract; runtime function timing and unsupplied defaults remain unproven.',
                'Existing Task shape/dtype describes tokenized semantic input, not the HTTP transport type.'])

    candidates=[
      dict(id='H01',title='响应等待路径的轮询开销与重复等待工作',modules=['M03','M25'],
        status='SUPPORTED_ACTIVE_PATH; CPU_COST_CAUSALITY_AND_LATENCY_VALUE_UNRESOLVED',evidence_kind=['MEASURED','DOCUMENTED'],
        profile_evidence=['Original sampled full stacks: step_with_batch_queue706 -> FutureWrapper.result98 -> _wait_for_response103 -> get_response433 -> shm.dequeue889 -> acquire_read786 -> wait204.',
                          'Both EngineCore leader threads consumed15.12CPU seconds over15.43385706s (~0.979664 cores);76/78 leader state observations were R.',
                          'Samples also reach timeout_ms/should_warn/check/memory_fence/sched_yield. Capture sample weights include idle tail and are not per-function CPU time.'],
        source_answer='Existing SHM wait already combines a busy/yield branch with idle poller notifications and cancellation. check() is defined inside the retry loop; FIFO Future draining and optional output_rank selection are present.',
        candidate_graph='Retain response publication/read/ack semantics while reviewing repeated readiness/deadline/check bookkeeping and the existing busy/idle policy as separate possible causes.',
        rejected_overclaims=['Event-driven waiting is absent.','70.9/77.8% sample weight equals removable latency or CPU time.','Every call waits for all ranks.','The scheduler itself is proven slow.'],
        constraints=['memory-fence and written/read flag publication order','per-slot generation and reader ack','Future FIFO and selected output rank','deadline/warning policy','shutdown/cancel wakeups','response latency and throughput protection'],
        competing_explanations=['Busy waiting may spend CPU while the worker is legitimately running.','Other on-CPU functions and profiler overhead may contribute.','A shorter busy interval may save CPU yet delay a ready response.'],
        resource='Host CPU occupancy is observed. No NPU utilization or end-to-end gain is assigned.',
        minimal_missing_evidence=['Time-aligned on-CPU attribution and producer-ready/read-complete boundaries for the same requests.','Resolved busy interval/output_rank and queue identities for the observed run.'],
        source_evidence='framework-source-slices166-v8.json; three fixed-image files and narrow ranges',
        across=['worker->EngineCore','request/response queues','successive Future generations']),
      dict(id='H02',title='输入准备中剩余小拷贝、锁页和元数据工作的合并重组',modules=['M04','M21','M22','M24'],
        status='REPEATED_WORK_OBSERVED; FIELD_BINDING_READINESS_AND_CRITICAL_PATH_OPEN',evidence_kind=['MEASURED','DOCUMENTED'],
        profile_evidence=['24 roots on ranks0/8; each6956 descendants,86 H2D calls,57 _pin_memory calls and4 Event::synchronize calls.',
                          'Full113-name/166944-record aggregate;24 exclusive partitions and22 same-trace prepare/gap intersections match.',
                          'Median recorded prepare wall45.926825ms(rank0)/44.87814ms(rank8); uncovered wall16.589875/15.403085ms is not CPU-active time.'],
        source_answer='Persistent batch buffers, early block-table commit, GPU accepted-token correction, optimistic CPU length correction and avoidance of an extra seq_lens D2H/sync already exist. Prepare scope includes persistent-state update and other conditional work, not only _prepare_inputs.',
        candidate_graph='Review remaining field-compatible metadata production/copy groups and repeated allocation/pinning under the existing persistent-state and graph-buffer contract; retain early-ready copies and late-dependent corrections.',
        rejected_overclaims=['All86 H2D calls are redundant.','All buffers are fresh allocations.','Async corrections are missing.','Four synchronizations can be deleted.','Host/device exported timestamp overlap proves causal hiding or removable gap.'],
        constraints=['exact field dtype/layout/padding and meaningful extents','graph-captured buffer addresses and lifetime','request ID/index remapping and newly admitted/removed requests','accepted/rejected token generations','block allocation and slot mapping','early-ready overlap vs late field readiness','zero-token DP coordination and connector semantics'],
        competing_explanations=['Many copies may carry different layer/metadata fields.','Recorded wait wall can be device/previous-generation dependency rather than preparation compute.','Apparent uniform roots need phase association before generalizing.'],
        resource='Host API counts and recorded wall only. No physical copy byte total, indispensable bandwidth floor or inference gain is established.',
        minimal_missing_evidence=['Copy/allocation/event -> exact field/source callsite mapping, plus producer-ready and all consumers.','Prepare-root phase/request association and calibrated dependency/stream boundaries.'],
        source_evidence='Existing prepare-input-analysis.json.sources worker SHA568c3049...; question-gated slices only',
        across=['scheduler->persistent batch','host->device','metadata->many layers','previous sample->next prepare','prepare->graph replay']),
      dict(id='H03',title='前缀块哈希的跨请求重复工作',modules=['M02','M24'],
        status='CONDITIONAL_UNPRICED_HYPOTHESIS; NO_FUNCTION_PROFILE_COST_OR_TOKEN_IDENTITY',evidence_kind=['MEASURED','DOCUMENTED','UNKNOWN'],
        profile_evidence=['Validated original run has235904/262144 prefix hits,8 successful requests and0 recorded preemptions.',
                          'Frozen8 text requests have8 distinct full hashes and share at least117286 characters;2 prefix seeds are outside formal statistics.',
                          'Frontend exported traces contain only profiler-window markers; no hash/tokenizer function costs.'],
        source_answer='The existing request block hasher already computes only new complete blocks and chains each block to its previous prefix hash. Incremental hashing within a request is already implemented.',
        candidate_graph='A conditional cross-request reuse edge may avoid recomputing an exactly identical hash prefix; this is distinct from already implemented per-request append hashing.',
        rejected_overclaims=['Full requests are identical.','Character prefix proves token/block identity.','Hashing is a measured bottleneck.','Historical resolved hash block size is established as128.','Zero preemptions means zero evictions.'],
        constraints=['exact token IDs and stable prefix boundary','hash function/version and resolved hash block size','parent hash and full extra-key namespace including multimodal/LoRA/cache_salt','model/tokenizer version as relevant','collision-safe identity, immutable generation, eviction/invalidation and ownership'],
        competing_explanations=['Hash cost may be negligible.','Existing host caches or request construction may already avoid some repeated work.','Shared text prefixes can tokenize differently at a boundary.'],
        resource='No hash CPU cost, retained cache bytes or same-domain time bound is known; do not rank by prefix-hit fraction.',
        minimal_missing_evidence=['Actual hash call counts/cost and repeated exact token/hash-extra identities under the same resolved H.','Existing reuse/invalidation path only if measured cost makes that question conclusion-changing.'],
        source_evidence='Reused stage-kv-hash-source.json SHA2666c9f1... slices812-869 and block-size resolution; no new target file read',
        across=['requests','prefix hash chain->KV lookup','seed->formal request','cache ownership/lifetime'])
    ]
    edge_rows=[
      dict(id='HE01',producer='worker response publication',consumer='EngineCore dequeue/read/ack',links=['H01'],observed='Sampled full call stack and source protocol.',open='Publication/ready/ack timestamps and slot generations absent; preserve fences/cancel/Future FIFO.'),
      dict(id='HE02',producer='previous sampled token acceptance',consumer='scheduler bookkeeping and next persistent batch',links=['H02'],observed='Existing GPU and CPU optimistic correction paths in source.',open='Exact active branch, event generation, accepted counts and copy-to-field timing not bound to each root.'),
      dict(id='HE03',producer='host input metadata fields',consumer='graph replay and attention/indexer layers',links=['H02'],observed='24 roots show repeat copy/pin/metadata operations; persistent buffers exist.',open='Per-field producer readiness, aliases, addresses, all consumers and layer-specific values remain open.'),
      dict(id='HE04',producer='token IDs plus hash namespace',consumer='chained prefix hash/KV lookup',links=['H03'],observed='Prefix hits and text overlap; hasher contract already incremental.',open='Exact token prefix/hash extras/H/current reuse costs are unknown.'),
      dict(id='HE05',producer='KV allocation/recycle/preemption handling',consumer='block table and slot mapping',links=['H02','H03'],observed='Source orders preemption handling before _update_states and commits block table early.',open='0 preemption counter does not cover eviction/recycle/cancellation or actual physical-slot generations.'),
      dict(id='HE06',producer='zero-scheduled-token rank/DP state',consumer='dummy coordination/collective order',links=['H02'],observed='Dummy path appears in sampled ancestry; source external-launcher+DP zero-token branch calls dummy coordination.',open='Do not equate all dummy samples with this particular branch; exact trigger/participants unknown.'),
      dict(id='HE07',producer='sampled output/RPC result',consumer='frontend detokenization/serialization/client SSE',links=['H01'],observed='Output counters and217-224 nonempty lines for1024 output tokens in separate worker validation.',open='No detailed frontend function profile, aligned output generations or per-token SSE mapping; line gap is not TPOT.'),
      dict(id='HE08',producer='termination/cancellation/free',consumer='later slot and buffer reuse',links=[],observed='8 successful fixed-length responses; SHM cancel wakeup exists in source.',open='Application cancellation/error/early termination and free/reuse generations not profiled in this case.')
    ]
    modules=[
      dict(id='M00',phase='startup/selection and runtime selected implementation',branches=[
        dict(branch='initial registration/model/platform replacement',state='OUTSIDE_RECORDED_STEADY_STATE_WINDOW; TIMING_UNKNOWN',evidence='Requests follow warmup; no startup function profile.',gap='Selected implementation/alternative selection reconciliation remains scope-relevant.'),
        dict(branch='runtime native/custom choices',state='PARTIAL_PROFILE_BINDINGS',evidence='v7 selected MC2 source/task bindings and existing model candidate reviews.',gap='Not all selectors or alternative branches are closed.')],candidate_links=[],closed=False),
      dict(id='M01',phase='formal request admission/preprocessing and sampling contract',branches=[
        dict(branch='text prompt input',state='FROZEN_INPUT_CONTRACT_BOUND_TO_PROFILE_RUN',evidence='8 string prompts; aggregate SHA equals original framework result.',gap='Tokenizer/admission CPU timing and actual caching path not profiled.'),
        dict(branch='already-tokenized HTTP input',state='NOT_SELECTED_BY_THIS_FROZEN_BODY',evidence='All8 prompt values are strings.',gap='This says nothing about semantic internal token-ID tensors.'),
        dict(branch='temperature=0, ignore_eos, max_tokens=1024, stream',state='REQUESTED_CONTRACT_WITH_OUTPUT_COUNT_VALIDATION',evidence='Frozen client body and8x1024 output token validation.',gap='Resolved processors/defaults and exact sampler branches refer to model profile ledger, not body inference alone.'),
        dict(branch='nonzero temperature, explicit top_p/seed/logprobs',state='NOT_EXPLICITLY_REQUESTED; RESOLVED_DEFAULTS_NOT_AUDITED',evidence='Frozen body sets temp0 and omits these optional fields.',gap='Absence of a key is not proof every corresponding implementation branch is inactive.')],candidate_links=[],closed=False),
      dict(id='M02',phase='seed/formal cache lookup, allocate/recycle',branches=[
        dict(branch='prefix seeding and cache hits',state='MEASURED_COUNTERS_AND_BOUND_FIXTURE',evidence='2 per-DP seeds outside formal stats;235904/262144 formal cache hits.',gap='Per-request hit IDs, H, block/token identities, hash CPU cost and KV generations unknown.'),
        dict(branch='preemption',state='ZERO_RECORDED_COUNTER_IN_THIS_RUN',evidence='num_preemptions_total delta0.',gap='Not a statement about evictions, all allocation/recycle work or unobserved branches.'),
        dict(branch='incremental append hashing',state='EXISTING_SOURCE_CONTRACT; COST_UNKNOWN',evidence='Hasher812-869 computes uncomputed full blocks only.',gap='No hash-function timing or exact cross-request reuse namespace.'),
        dict(branch='allocation, eviction/free, connector/error transitions',state='NOT_CLOSED',evidence='Prepare source has block-table/slot/preemption ordering.',gap='No complete lifetime profile or active-branch identities.')],candidate_links=['H03'],closed=False),
      dict(id='M03',phase='EngineCore request-window and response queue; sampled idle tail separate',branches=[
        dict(branch='step_with_batch_queue response wait',state='SAMPLED_ACTIVE_PATH_AND_WINDOWED_CPU',evidence='Full raw stacks;15.12s leader CPU/15.43385706s each.',gap='No exact sampled request-time crop or per-function CPU attribution.'),
        dict(branch='busy/yield and idle poll',state='SOURCE_IMPLEMENTED; BOTH_SAMPLED_LOCATIONS_PRESENT',evidence='wait204 yield and poll.py leaves; source wait191-206.',gap='Resolved busy interval and aligned per-branch costs not known.'),
        dict(branch='Future FIFO and output_rank queue subset',state='SOURCE_CONTRACT; DYNAMIC_ARGUMENTS_UNKNOWN',evidence='FutureWrapper91-108, executor422-448.',gap='Cannot assert all-rank response head-of-line cost.'),
        dict(branch='dummy work / scheduler function',state='PARTIAL_SAMPLES',evidence='Response-wait dummy ancestors exist; scheduler sampling near zero.',gap='Sparse scheduler samples do not exclude scheduler cost or prove dummy branch cause.')],candidate_links=['H01'],closed=False),
      dict(id='M04',phase='24 retained prepare scopes, client phase association incomplete',branches=[
        dict(branch='recorded prepare input',state='FULL_AGGREGATE_FOR_RETAINED_ROOTS_ONLY',evidence='24 roots,113 names,166944 descendants;24/24 partitions;22/22 gap coverage.',gap='Not all ranks/phases/iterations and not independently calibrated host/device causality.'),
        dict(branch='persistent state and async speculative corrections',state='EXISTING_SOURCE_CONTRACT; PER_ROOT_BRANCH_UNKNOWN',evidence='Worker1299-1306,1435-1467,1491-1543,1603-1627,2287-2311.',gap='4 event waits not mapped to4 independent removable dependencies.'),
        dict(branch='zero-token/DP, connector, EC, GDN, DBO/cascade alternatives',state='CONDITIONAL_SOURCE_BRANCHES_NOT_CLOSED',evidence='Worker2323-2385 and GDN buffer branch1443.',gap='No per-root resolved flags; do not label inactive from no named profile scope.')],candidate_links=['H02'],closed=False),
      dict(id='M25',phase='RPC output, frontend/SSE and terminal lifecycle',branches=[
        dict(branch='RPC response acquire',state='SAMPLED_ACTIVE_SUBSET',evidence='Same H01 response stack.',gap='Only worker->core subset; no detailed frontend cost.'),
        dict(branch='streamed visible output',state='OBSERVED_COARSE_CLIENT_BOUNDARIES',evidence='8 successful outputs; worker validation217-224 nonempty SSE lines per1024 tokens.',gap='Separate run, no one-line/one-token contract, detokenization/serialize/network split unknown.'),
        dict(branch='frontend tokenization/detokenization/SSE functions',state='FUNCTION_PROFILE_MISSING',evidence='2 frontend exported traces each have1 profiler X marker.',gap='Missing scopes do not prove idle or absent work; code cannot recreate historical timing.'),
        dict(branch='cancel/error/early EOS/free/reuse',state='NOT_CLOSED; NOT_EXERCISED_BY_SUCCESS_COUNTS_ALONE',evidence='ignore_eos requested,8 successes, SHM source cancel exists.',gap='Request cancellation and slot-generation behavior not measured.')],candidate_links=['H01'],closed=False)
    ]
    universe=read(O/'module-profile-ledger-v7.json')
    old_candidates=read(O/'candidate-profile-ledger-v7.json')
    assert len(universe['modules'])==26 and len(old_candidates['candidates'])==32
    for m in universe['modules']:
        m['inherited_review_from']='module-profile-ledger-v7.json'
        extra=[c['id'] for c in candidates if m['id'] in c['modules']]
        m['candidate_links']=list(dict.fromkeys(m['candidate_links']+extra))
        if m['id'] in {x['id'] for x in modules}:
            m['review_state']='PARTIAL_FRAMEWORK_PROFILE_BRANCH_REVIEW; NO_WHOLE_MODULE_CLOSURE'
            m['framework_branch_record']='framework-module-branch-matrix-v8.json#'+m['id']
            m['profile_entry_points']+=['framework-profile-audit-v8.json','framework-summary-v8.json']
        if extra: m['framework_candidate_evidence']='framework-candidate-ledger-v8.json'
        m['no_candidate_links_meaning']='No supported candidate attached yet; not proof of no opportunity or no historical work.' if not m['candidate_links'] else None
        assert not m['whole_module_closed']
    universe.update(generated_utc=now,scope='Inherited26-module model intake plus bounded6-domain framework phase/branch audit; provisional universe, no exhaustive closure.',
                    framework_candidates=3,existing_model_candidates=32,new_device_work=False,new_source_read=True)
    old_candidates.update(generated_utc=now,parent_ledger='candidate-profile-ledger-v7.json',framework_candidate_ledger=str(O/'framework-candidate-ledger-v8.json'),
                          candidate_universe_complete=False,overall_complete=False)
    evidence_manifest=[]
    for p in [O/'framework-profile-audit-v8.json',O/'framework-profile-summary-v8.json',N/'archived-framework-speedscope-callsites166-v8.json',
              N/'framework-source-slices166-v8.json',N/'framework-request-contract166-v8-v2.json',client,controller,R/'agents/diagnosis/stage-kv-hash-source.json']:
        evidence_manifest.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    fsummary=dict(generated_utc=now,request_contract=request_summary,prepare=s,engine=audit['engine'],
       framework_candidates=[dict(id=c['id'],status=c['status']) for c in candidates],
       reviewed_framework_domains=[m['id'] for m in modules],existing_model_candidates=32,new_framework_hypotheses=3,
       source_snapshots=[dict(path=x['container_path'],sha256=x['sha256'],ranges=[[z['start'],z['end']] for z in x['excerpts']]) for x in source['sources']],
       raw_speedscope_result=dict(path=stacks['full_result'],sha256=stacks['result_sha256']),
       request_contract_result=dict(path=request['full_result'],sha256=request['result_sha256']),
       evidence_manifest=evidence_manifest,source_read_is_timing_evidence=False,whole_modules_closed=0,new_device_or_inference_work=False,overall_complete=False,
       mathematical_boundary='CPU core occupancy is not NPU utilization. No same-domain indispensable work/capacity bound or causal gain is supplied by observed wall/call counts.',
       corrections=['First overlap comparison omitted gap_ns and falsely reported22 mismatches; corrected result0, all22 independent checks pass.',
                    'First request audit incorrectly required8 seeds; fixture has8 formal requests and2 seeds. Failed receipt retained, v2 resolved input hash and shape.'])
    gaps=dict(generated_utc=now,module_universe_status='PROVISIONAL26; completeness remains to be checked against retained profile/source boundaries',
      framework_audit='Six domains reviewed partially; not six closed modules.',
      function_profile_gaps=['M01 preprocessing/tokenization','M02 hash/allocation/eviction/lifetime','M25 detokenization/serialization/SSE/cancel/free'],
      causal_gaps=['H01 same-window on-CPU and producer-ready/consume alignment','H02 exact field/copy/event mapping, phase and calibrated overlap','H03 exact token/hash namespace and measured cost'],
      remaining_scope=['All26-module finite phase/branch requirements under the frozen workload, including startup vs steady-state and inactive/unknown branches.',
                       'Retained formal/seed/dummy/capture/proposal phase identity and source consumers across ranks/iterations.',
                       'Cross-operator/layer/card/iteration consumer and alias generations for existing32model candidates and framework hypotheses.',
                       'Same-domain resource bounds/ranking and critical-path cause; NPU-lowest-utilization conclusion remains distinct from host CPU occupancy.'],
      unsupported_completion_metrics=['32 model candidate intakes','3 framework hypotheses','6-domain branch table','report ready','source/AST enumeration'],
      whole_modules_closed=0,overall_complete=False)
    write(O/'framework-summary-v8.json',fsummary)
    write(O/'framework-candidate-ledger-v8.json',dict(generated_utc=now,candidates=candidates,relationships=[
        dict(a='H01',b='H02',relation='interacts',scope='Host contention and request/response pipeline; not measured; no additive gains.'),
        dict(a='H02',b='F31',relation='overlaps',scope='Existing draft/target metadata preparation; field mapping needed before deduplication.'),
        dict(a='H02',b='H03',relation='interacts',scope='Both touch KV lookup/ownership and subsequent metadata. Neither candidate requires implementing the other; not a demonstrated performance interaction.')],
        no_implementation=True,no_speedup_claim=True,whole_modules_closed=0,overall_complete=False))
    write(O/'framework-module-branch-matrix-v8.json',dict(generated_utc=now,modules=modules,phase_status_is_not_timing=True,whole_modules_closed=0,overall_complete=False))
    write(O/'framework-consumer-edges-v8.json',dict(generated_utc=now,edges=edge_rows,scope='Framework boundaries only; complements model consumer evidence throughv7, not complete edge universe.',all_edges_closed=False,overall_complete=False))
    write(O/'module-profile-ledger-v8.json',universe)
    write(O/'candidate-profile-ledger-v8.json',old_candidates)
    write(O/'module-coverage-gaps-v8.json',gaps)
    gate=read(O/'FRAMEWORK-CODE-QUESTION-GATE-v8.json')
    gate['review_answers']={c['id']:dict(answer=c['source_answer'],still_missing=c['minimal_missing_evidence']) for c in candidates}
    gate['case_transport_answer']=request_summary
    write(O/'FRAMEWORK-CODE-QUESTION-REVIEW-v8.json',gate)
    report='''# Profile 主导的框架范围补充审查 v8

这轮补上 M00/M01/M02/M03/M04/M25 的阶段和分支记录，形成3个分级框架假设。原32个模型候选沿用 v7 证据。整体分析仍未完成，0个整模块关闭；候选数和报告成稿不代表覆盖率。

## 已确认的事实

- 原始 EngineCore run 的输入哈希与服务器冻结输入一致。8条正式请求发送的是文本，各全文不同，至少共享117286字符；2条前缀 seed 各指定一个 DP rank，不计正式统计。字符前缀不能代替 tokenizer 后的 token 边界或块哈希身份。
- 两条 EngineCore 主线程在15.43385706秒观察窗内各消耗15.12 CPU秒，约0.98个核。完整采样栈定位到 `step_with_batch_queue → FutureWrapper → get_response → SHM acquire_read/wait`。采样包含空闲尾段，70.9%/77.8%的 response-wait 样本占比不能当 CPU占比或可省时延。
- 24个保留的输入准备窗口，各含6956条子记录、86次H2D调用、57次 `_pin_memory`、4次 Event同步。全部113类名称、166944条子记录均参与聚合。24个区间分账成立，22个同 trace 的 prepare/间隙交集与旧汇总一致。
- rank0/rank8 的 prepare 记录墙钟中位数分别45.926825/44.87814ms；未被子范围覆盖的中位数16.589875/15.403085ms。它们不是 CPU活跃时长，也不是可直接删除的间隙。窗口尚未全部绑定正式/捕获/其他阶段，跨主机/设备时钟交集不构成因果证明。

## 三个框架假设

| ID | 证据支持的范围 | 当前不能确定的结论 |
| --- | --- | --- |
| H01 | 响应等待路径反复经过 readiness、deadline、yield/poll 等处理；同时观测到高主线程 CPU占用 | 等待路径具体消耗多少 on-CPU 时间、是否拖慢推理、调整是否有净收益 |
| H02 | 输入准备中存在大量小拷贝、pin、元数据操作与等待 | 哪些字段可以合法合并/复用、4次同步各依赖什么、是否处于关键路径 |
| H03 | 不同请求有共同文本前缀且已有实际 KV前缀命中；块哈希源合同可约束跨请求重用 | 实际 token/hash-extra 身份、历史 H、重复哈希 CPU成本及现有重用路径 |

定点代码回答修正了三个容易误报的方向：SHM 已有忙等后阻塞及通知/取消机制；worker 已有持久缓冲、提前 block-table copy 和异步接受数修正；块哈希已经按请求增量计算新完整块。不能把这些现有能力再次登记为“尚未实现”。H01只是现有等待机制的成本问题，H02只审查剩余工作，H03只保留条件性的跨请求重复工作。

## 覆盖和消费者约束

`framework-module-branch-matrix-v8.json` 逐项保存六个范围的观测、未请求条件、源合同和未知分支。文本预处理不能因旧case dtype写token IDs而排除；未显式请求logprobs/top_p/seed也不能替代解析默认值审查。0次preemption只约束该计数窗口，不证明无驱逐/回收。前端两条导出trace只有profiler窗口标记，无法定位 tokenizer、detokenizer 和 SSE函数时间。另一轮 worker py-spy 未产生调用栈；客户端每1024token对应217–224条非空SSE行，不能用行间隔当逐token TPOT。

`framework-consumer-edges-v8.json` 覆盖 worker发布/读取确认、Future FIFO、上轮接受数到下轮输入、图捕获缓冲地址、跨层元数据、KV槽重用、零token DP协调和输出终止。所有边都保留代际、别名、就绪时间或阶段证据缺口。H02与F31范围交叠，候选收益不能相加。

CPU占核指标不能用于判断最低 NPU利用率模块。框架必要工作量、同域硬件容量上界和依赖下界未齐，保持符号/未知，不从观察到的拷贝次数、墙钟或等待比例推导理论提速。

## 本轮错误与修正

首轮比较脚本漏写 `gap_ns` 字段，误报22处结构不一致；修复后0处不一致，22/22独立核对通过。首轮请求脚本假设8条seed，实际fixture是8条正式请求、2条seed；失败回执保留，v2固定原运行SHA并成功核对。两者均为分析脚本假设错误，不是原始profile冲突。

下一步先把26模块的已有profile映射到有限的阶段/条件和消费者要求，复核模块集合是否完整；对结论仍有影响的缺口区分“已有原始数据可再查”与“需要有限补采”。只有profile无法回答语义时才定点读代码。没有目标实现修改、服务重启或新的推理/NPU作业。
'''
    p=O/'PROFILE-FINDINGS-v8.md'; assert not p.exists();p.write_text(report)
    print(json.dumps(dict(status='PROGRESS',framework_domains=6,framework_hypotheses=3,existing_model_candidates=32,request_transport='text',input_hash_bound=True,
        partition_pass=24,gap_check_pass=22,whole_modules_closed=0,overall_complete=False,summary=str(O/'framework-summary-v8.json')),ensure_ascii=False))

if __name__=='__main__': main()
