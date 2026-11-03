from pathlib import Path
from datetime import datetime, timezone, timedelta
from copy import deepcopy
import hashlib, json
from finalize_origin_coverage_v9 import N,O,R,W,T,L,read,write,ctl

def receipt(name, expected=0):
    r=read(N/name)
    assert r['returncode']==expected,(name,r['returncode'])
    return json.loads(r['stdout']) if expected==0 else r

def main():
    assert not (O/'REVIEW-CHECKPOINT-v13.json').exists()
    now=datetime.now(timezone.utc);stamp=now.isoformat()
    consumers=receipt('saved-prepare-consumers166-v13.json')
    cp=receipt('cp-dead-producer-device-tasks166-v13-v2.json')
    ctor=receipt('metadata-replace-contract166-v13-v2.json')
    block=receipt('blocktable-copy-contract166-v13-retry.json')
    events=receipt('event-resource-schema166-v13.json')
    resources=receipt('saved-events-resources166-v13-v2.json')
    assert consumers['summary']['later_suffix_H2D_after_root_batch_init']==2200
    assert consumers['summary']['utility_H2D_records']==2064 and consumers['summary']['utility_full_caller_chains']==13
    assert cp['summary']['dead_producer_compute_executions']==11000
    assert cp['summary']['ambiguous_execution_links']==0
    assert cp['summary']['per_root_dead_task_hist']=={'125':88}
    assert cp['summary']['dead_producer_op_type_counts']=={op:2200 for op in ['Sub','ClipByValueV2','Greater','Fill','SelectV2']}
    resultrefs=[dict(path=v['full_result'],sha256=v['result_sha256']) for v in [consumers,cp,ctor,block,events,resources]]
    source_pins={
      'CP_builder':'039ec9d18e0a8a4aa39f9d5b4bea15204cea3f397a3fce97a5097b266bc9822e',
      'DSA_builder':'3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd',
      'Ascend_worker':'568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c',
      'legacy_CP_helper':'7e8dd93eeda988866138342523ecfa7cc9c42ce67669d4b1507ee8c5f3452305',
      'CommonAttentionMetadata':'01ac364372afa75fdb22486aa3a9c7cf9aebb742e3471dfb9e0f9d9d36b734bc',
      'AscendCommonAttentionMetadata':'67e1c2e33efce8616a6845d605d2ed2522c2e10e91ad4b7752d5084da5813405',
      'block_table':'215305c699a9df9c118f47c6f7b71d8457ef54e98b5b8571536fd373de7b21ed'}
    new_variant=dict(id='H02_CP_SUFFIX_ON_BATCH_MISS',status='STRUCTURALLY_SUPPORTED_IN_OBSERVED_FIXED_PATH; NOT_IMPLEMENTED_OR_TIMED',evidence_kind=['MEASURED','DOCUMENTED','DERIVED'],
      scope='Fresh166 fixed model/image/CANN9.2.0; LegacyAscendDSAV41CPMetadataBuilder local branch with same-call shared batch dictionary. Other CP/draft/backend paths remain separate.',
      graph='Move CPU suffix/pin/H2D and device Sub/Clip/Greater/Fill/Select production behind the existing local batch-cache miss; on a hit preserve cached authoritative lengths and all live per-builder fields.',
      observed=dict(nonempty_prepare_roots=88,CP_builds_per_root=26,CP_LOCAL_batch_initialization_copies_per_root=1,CP_GLOBAL_batch_initialization_copies_per_root=1,post_local_initialization_suffix_H2D_per_root=25,post_local_initialization_suffix_H2D=2200,source_bound_native_compute_tasks=11000,tasks_per_root=125,op_type_counts=cp['summary']['dead_producer_op_type_counts']),
      proof=[
        'Worker3823 creates common_v41_batch_metadata once per build; per-KV cm=copy(cm_base) is shallow and CP common fields are retained. Per-group block/slot state stays separate.',
        'CP global helper87-95 uses a cp_global child batch dictionary. The local builder receives the root batch dictionary; first local DSA903 copy is observed once in every nonempty root.',
        'DSA964-967 builds batch coordinates only on a cache miss. Later common.seq_lens content is not read;945 is shape metadata only. Returned and deferred consumers use cached coordinate seq_lens.',
        'Metadata.replace is dataclasses.replace; both metadata classes have no custom init/post_init/getattr/seq_lens property. Constructing local_common introduces no hidden length-content reader.',
        'Native CONNECTION_IDS links, then exact same-thread root/build containment, map all125 later-branch kernels per root without ambiguous execution links. NULL host connections are retained but not matched.',
        'For this unused-result elimination, equality of all later input values is not required; the first cache miss and live query offsets remain required.'],
      must_keep=[
        'First local cache miss and its authoritative post-rejection GPU seq_lens computation; never substitute optimistic CPU lengths.',
        'Per-builder fixed-address query_start_loc publication: group-specific slots/SMLA/QLI/C2 callbacks directly read common.query_start_loc.',
        'CPU local query metadata/max values and group-specific physical KV/slot/ring data; do not skip the whole CP helper.',
        'Existing cp_global cache, RoPE views, shared C2 lengths, graph addresses, stream readiness and host/device lifetimes.',
        'Original path when shared batch context is missing or cache misses; no blanket model/backend change.'],
      no_content_reader_closed_for='Later local GPU length temporary in the inspected fixed program path; not all metadata or all modes.',
      pricing=dict(profiled_kernel_interval_union_ns_per_root=resources['summary']['CP_dead_value_kernel_union_ns_per_root'],critical_path_gain=None,unprofiled_CPU_cost=None,attainable_speedup=None),
      implementation=False,performance_verified=False)
    contract=deepcopy(read(O/'H02-CP-reuse-contract-v12.json'))
    contract.update(generated_utc=stamp,parent_contract='H02-CP-reuse-contract-v12.json',source_pins=source_pins,scoped_dead_producer_variant=new_variant,saved_evidence=resultrefs,
      missing_conclusion_changing_evidence=['Unprofiled CPU/function/critical-path cost and implementation validation for the scoped dead producer.','Remaining query-offset sharing/fixed-address/stream contracts are separate; do not wait for them to recognize unused lengths.','Full consumer/lifetime and complete key proofs for geometry templates, dirty block publication, lookback sentinel reuse and optional GDN buffer publication.'])
    contract['variants'].append(new_variant)
    contract['relationships'].append('Batch-miss-only length production supersedes suffix reuse on the same25 cache-hit producers; do not add their hypothetical savings. Query-offset host staging is a separate remaining subgraph.')
    write(O/'H02-CP-reuse-contract-v13.json',contract)
    # Complete, disjoint H2D census by first source and full utility callers.
    rows=[
      ('num_accepted_tokens',88,'Ascend1500','DYNAMIC_ACCEPTANCE_STATE','Keep previous-step acceptance readiness and correction; not geometry-only.'),
      ('query_start_loc_initial',88,'Ascend1437','MUTABLE_QUERY_BOUNDARIES','Full persistent publication; later GPU tail fill1462 changes device contents.'),
      ('query_start_loc_padding_publication',64,'Ascend1105','CONDITIONAL_NARROW_OR_SKIP','16 actual2req/12token and48 actual4req/24token cases; all publish padded4req QSL. CPU equality alone does not preserve GPU tail.'),
      ('prev_positions',72,'Ascend1354','DYNAMIC_PREVIOUS_REQUEST_MAP','Preserve request permutation, generation and async length semantics.'),
      ('discard_request_mask',88,'Ascend1490','DYNAMIC_REQUEST_MASK','Preserve actual acceptance/discard behavior.'),
      ('prev_num_draft_tokens',72,'Ascend1521','DYNAMIC_PREVIOUS_SPEC_STATE','May be uniform only under complete invariant proof, not shape alone.'),
      ('KV_group_block_tables',1144,'block_table299 ->521 -> Ascend1304','DIRTY_ROW_OR_VERSION_PUBLICATION_CANDIDATE','13 tables per nonempty root. Early commit already exists; commit copies active-row prefixes. Inspected append/add/clear/move/swap methods have no dirty/generation tracking. Preserve all physical mappings and changed/padded rows.'),
      ('num_scheduled_tokens',88,'Ascend1542','GEOMETRY_TEMPLATE_CONDITIONAL','Full scheduled counts and request ordering key required.'),
      ('lookback_token_ids',88,'baseGPU1088','LOOKBACK_SENTINEL_REUSE_UNPROVEN','Generated positions use -1 sentinel but consumer writes and depth/position eligibility not closed. _preprocess is outside synchronize_input_prep guard.'),
      ('req_indices',88,'Ascend1537','GEOMETRY_TEMPLATE_CONDITIONAL','Exact row order/shape and consumers/lifetimes required.'),
      ('num_decode_draft_tokens',72,'Ascend1679','DEVICE_PUBLICATION_CONSUMERS_UNRESOLVED','Inspected workers show CPU GDN consumers; no direct GPU read found there. Requires exact installed-symbol consumer/alias inventory before elimination.'),
      ('input_ids',24,'baseGPU1872','PROMPT_OR_MIXED_INPUT_PUBLICATION','Token contents vary; do not cache by shape.'),
      ('query_pos',88,'Ascend1540','GEOMETRY_TEMPLATE_CONDITIONAL','Exact per-request scheduled geometry and row mapping required.')]
    utility=[dict(field=f,H2D=n,source=s,status=st,contract=c) for f,n,s,st,c in rows]
    assert sum(v['H2D'] for v in utility)==2064
    families=[dict(family='CP_query_offsets',H2D=2288,status='LIVE_FIXED_ADDRESS_PUBLICATION'),dict(family='CP_suffix',H2D=2288,status='88_FIRST_MISS_REQUIRED;2200_CACHE_HIT_DEAD_PRODUCERS'),dict(family='utility_copy_to_gpu',H2D=2064,status='13_CALLER_FIELDS_PARTITIONED'),dict(family='spec_five_index_vectors',H2D=360,status='GEOMETRY_TEMPLATE_CONDITIONAL;LIVE_DRAFT_IDS_RECOMPUTED'),dict(family='sample_draft_input_packing_indices',H2D=288,status='STABLE_UNIFORM_SPEC_STRIDED_PACKING_CONDITIONAL'),dict(family='computed_token_tensor_CPU',H2D=88,status='DYNAMIC_POST_REJECTION_POSITION_STATE'),dict(family='computed_token_fallback',H2D=16,status='CONDITIONAL_CPU_TO_DEVICE_FALLBACK')]
    assert sum(f['H2D'] for f in families)==7392
    fields=dict(generated_utc=stamp,population='96 roots /32 worker DBs; counts are profiled API occurrences, not CPU cost or savings.',H2D_total=7392,H2D_families=families,utility_H2D=2064,utility_full_caller_classes=13,utility_fields=utility,
      additional_candidates=[dict(id='H02_SPEC_INDEX_GEOMETRY_TEMPLATE',scope='72 spec metadata calls /360 index-vector H2D and pins',conditions='Exact num_draft_tokens and cumulative scheduled counts/dtype/device; immutable cached vectors and stream lifetime. Draft token IDs remain live GPU gathers. Source already keeps pinned tuple alive via event/deque.'),dict(id='H02_UNIFORM_SPEC_INPUT_PACKING',scope='Four scatter-index H2Ds per72 calls',conditions='Exact stable prev/current request-row map, scheduled/draft geometry, padding, invalid rows, cast and aliases. Existing contiguous no-spec fast path already exists; uniform-spec striding does not generally satisfy it.')],
      synchronization=dict(sites={'Ascend1495_num_accepted_ready':88,'Ascend1867_valid_counts_ready':72,'baseGPU3849_input_prep_lifetime':96,'baseGPU4987_valid_counts_read':56},total=312,corrected_order=resources['summary'],generation_closure=False,removable_waits_proved=0,
      guard='Input prep guard covers update/prepare/pad/attention metadata and their CP DMA. It ends before _preprocess/lookback; no automatic lookback-lifetime proof.'),
      source_pins=source_pins,saved_evidence=resultrefs,no_optimization_implementation=True,whole_modules_closed=0,overall_complete=False)
    write(O/'H02-field-and-synchronization-ledger-v13.json',fields)
    write(O/'resource-profile-inventory-v13.json',dict(generated_utc=stamp,summary=resources['summary'],first_database_metrics=resources.get('first_database_metrics'),saved_result=resultrefs[-1],interpretation='Inventory of names/raw values/identity multiplicities only; no normalized utilization, peak FLOPS or MFU claim. Complete saved per-DB census remains on166. The summary frequency_value_counts counts per-DB frequency groups, not raw sample records.',whole_modules_closed=0,overall_complete=False))
    framework=deepcopy(read(O/'framework-candidate-ledger-v12.json'));framework.update(generated_utc=stamp,parent_ledger='framework-candidate-ledger-v12.json')
    h=next(c for c in framework['candidates'] if c['id']=='H02')
    h.update(status='SCOPED_CP_CACHE_HIT_DEAD_PRODUCER_SUPPORTED;OTHER_FIELDS_AND_SYNC_CONDITIONAL;UNPRICED',CP_reuse_contract_v13='H02-CP-reuse-contract-v13.json',field_and_synchronization_ledger_v13='H02-field-and-synchronization-ledger-v13.json',minimal_missing_evidence=contract['missing_conclusion_changing_evidence'])
    h['profile_evidence'].append('Native-linked11000compute tasks plus2200suffix H2D occur after the existing local batch-cache initialization across88 roots; later length values have no content reader in inspected fixed path. Query buffers remain live.')
    h['source_answer']+=' v13: The local batch cache already shares the first length result, but CP outer suffix/GPU-length producers execute on25 later cache-hit builds. Metadata constructors do not introduce hidden length reads. Preserve live query buffers and all group-specific fields.'
    write(O/'framework-candidate-ledger-v13.json',framework)
    candidates=deepcopy(read(O/'candidate-profile-ledger-v12.json'));candidates.update(generated_utc=stamp,parent_ledger='candidate-profile-ledger-v12.json',review_v13_utc=stamp,overall_complete=False)
    write(O/'candidate-profile-ledger-v13.json',candidates)
    modules=deepcopy(read(O/'module-profile-ledger-v12.json'));matrix=deepcopy(read(O/'module-phase-branch-matrix-v12.json'))
    notes={'M04':'All7392H2D sources and13utilitycaller fields partitioned;4sync contracts distinguished. Native11000CP length kernels are later-cache-hit unused producers in fixed path, not a priced module result.', 'M21':'CP local batch first-miss length shared already;25 later suffix/GPU length producers per88 roots remain structurally removable. Fixed-address query buffers and per-KV slots/rings stay live. Spec geometry-template and uniform-spec input packing remain conditional.', 'M22':'Per-builder fixed graph query addresses remain live; no fresh-address sharing justified. Existing spec pinned tuple event/deque lifetime retained; new geometry-template/storage changes require their own graph and stream contract.', 'M24':'Accepted/valid token-count state and input-buffer lifetime events separated. Correct actual sync order saved; no wait removed without generation/source-recorder closure.'}
    for m in modules['modules']:
      m['phase_branch_record']='module-phase-branch-matrix-v13.json#'+m['id']
      if m['id'] in notes:m['review_v13']=dict(finding=notes[m['id']],evidence=['H02-CP-reuse-contract-v13.json','H02-field-and-synchronization-ledger-v13.json'],whole_module_closed=False)
    for m in matrix['modules']:
      if m['id'] in notes:m['branches'].append(dict(branch='Fresh166 exact-field and CP cache-hit producer analysis',state='SCOPED_STRUCTURAL_REVIEW;NOT_WHOLE_MODULE_CLOSED',evidence=notes[m['id']],gap='Implementation correctness/performance, remaining consumers/lifetimes and whole-module resource/cross-boundary closure remain open.'))
    modules.update(generated_utc=stamp,parent_ledger='module-profile-ledger-v12.json');matrix.update(generated_utc=stamp,parent_matrix='module-phase-branch-matrix-v12.json')
    write(O/'module-profile-ledger-v13.json',modules);write(O/'module-phase-branch-matrix-v13.json',matrix)
    remaining=[
      'H02 scoped dead-producer candidate is structurally supported, but implementation/performance and remaining query/geometry/block/optional-field consumers/lifetimes are not validated. No source optimization edits in this pass.',
      'Sync generation and recorder/producer-ready contracts remain open; corrected56pair ordering is not proof a wait is removable.',
      'Use new16611exactTP8cohorts and32DBs for native forward/graph/MC2/COMM/module/resource association. Raw PMU names/values/frequency inventory does not settle normalized utilization or lowest-module causes.',
      'H01 EngineCore on-CPU/producer-ready attribution, H03 actual hash/frontend costs and namespace remain unmeasured by worker-only profile.',
      'All26module finite branches/consumers, cross-operator/layer/card/iteration alternatives, same-domain hardware ceilings and causal ranking remain incomplete.']
    summary=dict(generated_utc=stamp,status='PROGRESS',workflow='PROFILE_FIRST;EXACT_CODE_QUESTIONS_ONLY',new_device_or_service_work=False,source_optimization_implementation=False,
      CP_cache_hit_dead_producer=new_variant,field_ledger='H02-field-and-synchronization-ledger-v13.json',resource_inventory=resources['summary'],saved_evidence=resultrefs,remaining=remaining,whole_modules_closed=0,overall_complete=False)
    write(O/'profile-contract-summary-v13.json',summary);write(O/'module-coverage-gaps-v13.json',dict(generated_utc=stamp,remaining=remaining,whole_modules_closed=0,overall_complete=False))
    report=f'''本轮取得实质进展：从保存的32份 profile 和完整调用链定位出 CP 的缓存命中后死生产者；整体分析仍未完成。本轮只做 CPU 离线查询和定点源码核对，没有启动设备任务、修改服务或实现优化。

88个非空输入准备区间中，CP局部元数据各构建26次，但现有DSA batch缓存每区间只初始化一次。后25次仍各上传suffix并执行Sub、ClipByValueV2、Greater、Fill、SelectV2。原生CONNECTION_IDS关联到11,000个实际计算执行，各类型2,200个，88个区间均为125个，无歧义连接；另有2,200次suffix H2D及对应锁页工作。保留全部原DB/API/任务执行身份，未用邻近时间或算子名补配，未把重复定义当单次执行。

定点核对回答了profile不能单独回答的读者问题：worker每次attention构建创建一个共享batch字典，CP global使用独立子字典；local DSA只在首次缺失时复制seq_lens并构造坐标。后续返回值和延迟回调使用缓存的长度，common.seq_lens只另有形状查询。两层metadata均为dataclass，replace没有隐藏的长度内容读取。因此在当前固定路径中，可把suffix和设备长度计算放到现有cache miss分支，保留首次计算。消除未被读取的结果不要求证明后25份输入值相等。

这个结论不覆盖整个CP build。固定图地址query_start_loc仍直接服务于分组slot/SMLA/QLI/C2消费者；CPU局部query元数据、max值、每KV组物理映射和首次拒绝修正后的设备长度必须保留。cp_global共享、C2长度共享、RoPE切片复用原本已经存在。候选尚未实现或计价，profile中的算子时间不能作为请求收益。

7,392次H2D已分成互斥来源族，其中copy_to_gpu的2,064次对应13类完整caller字段。块表1,144次=每非空区间13份，commit已提前执行；已查的小型行更新/commit方法没有脏行版本跟踪，后续候选是保留物理映射语义的脏行/版本发布。Spec的五个索引向量可评估按完整几何key复用，实际draft token IDs仍实时GPU取得；源码已有pinned tuple加event/deque生命周期保护。稳定uniform-spec输入可另评估带步长打包，已有连续无spec快路不能重复当缺失优化。

query_start_loc的第二次上传不能简单删除：两次之间有GPU尾部fill，64次padding发布里16次实际为2请求/12token，48次为4请求/24token，均发布4请求的padding形状。CPU值或shape相同不保证GPU状态相同。lookback上传位于input-prep同步guard之外，不能继承该guard的暂存生命周期结论。num_decode_draft_tokens设备副本的完整消费者尚未闭合，保留为待定项。

312次同步分为accepted计数就绪88、valid计数就绪72、input-prep复用屏障96、valid计数读取56。最初事件查询把两个valid同步的先后方向写反，相关零writer推论已作废并保留；v2用实际API时间重新排序，保留原记录与源位置。没有仅凭没有观察到record而宣称同代际，也没有删除同步。

32份DB的PMU指标名称、原值范围、TASK身份重数与频率样本已另存为资源清单。它们仍需对应采样/平均、单位、硬件和模块窗口，不能直接换算MFU。H01/H03其他进程成本、全模块/条件/跨边和理论界仍有缺口；Goal保持active，关闭的整模块为0。
'''
    (O/'PROFILE-FINDINGS-v13.md').write_text(report)
    gate=read(O/'PROFILE-CODE-QUESTION-GATE-v13.json');gate.update(answered_by=['H02-CP-reuse-contract-v13.json','H02-field-and-synchronization-ledger-v13.json'],scoped_length_content_reader_question_closed=True,whole_consumer_or_alias_closure=False,source_optimization_implementation=False);write(O/'PROFILE-CODE-QUESTION-GATE-v13.json',gate)
    jobs=[
      ('profile','pass','Saved full chains resolve2064utilityH2D into13callers and observe oneCPglobal/local batch init plus25 later suffixH2D per88roots.',[N/'saved-prepare-consumers166-v13.json',N/'query_saved_prepare_consumers_v13_166.py']),
      ('review','error','Initial metadata contract query stopped because utils.py reexports the base metadata class; no guessed constructor result.',[N/'metadata-replace-contract166-v13.json',N/'read_metadata_replace_contract_v13_166.py']),
      ('review','pass','Exact reexport resolution and installedSHA verification: base/Ascenddataclasses have no hidden seq_lens reader in init/post_init/replace.',[N/'metadata-replace-contract166-v13-v2.json',N/'read_metadata_replace_contract_v13_166_v2.py']),
      ('profile','error','Initial CP native query stopped sorting NULL host connection IDs; no result or synthetic identity.',[N/'cp-dead-producer-device-tasks166-v13.json',N/'query_cp_dead_producer_device_tasks_v13_166.py']),
      ('profile','pass','v2 retains NULL APIs and matches only actual native connections:11000cache-hit dead length tasks,125per88roots,0ambiguous executions.',[N/'cp-dead-producer-device-tasks166-v13-v2.json',N/'query_cp_dead_producer_device_tasks_v13_166_v2.py']),
      ('review','error','One read-only blocktable source query failed SSH authentication before execution; original rc255 retained.',[N/'blocktable-copy-contract166-v13.json']),
      ('review','pass','Bounded retry succeeds:13group tables use existing early commit; exact active-row copy and append/add/clear/move/swap methods reviewed, no dirty/generation tracking in these methods.',[N/'blocktable-copy-contract166-v13-retry.json',N/'read_blocktable_copy_contract_v13_166.py']),
      ('review','pass','32DB event/resource schemas and rawsource events saved. Original reversed early/late pair analysis is invalidated; inventory remains valid.',[N/'event-resource-schema166-v13.json',N/'query_event_resource_schema_v13_166.py']),
      ('profile','error','First resource metric membership query stopped at145sSQLite progress bound on correlated EXISTS; no derived output.',[N/'saved-events-resources166-v13.json',N/'query_saved_event_order_and_resource_metrics_v13_166.py']),
      ('profile','pass','Equivalent IN membership v2 completes32DB metric/frequency/definition inventory and actual chronological56sync pairs. No generation/normalized utilization claim.',[N/'saved-events-resources166-v13-v2.json',N/'query_saved_event_order_and_resource_metrics_v13_166_v2.py']),
      ('review','pass','Main-agent semantic review persists scoped dead-producer proof, conserved7392H2D/13field partition, conditional geometry/block candidates and unresolved whole-module questions.',[O/'profile-contract-summary-v13.json',O/'H02-CP-reuse-contract-v13.json',O/'H02-field-and-synchronization-ledger-v13.json',O/'PROFILE-FINDINGS-v13.md'])]
    runs=[]
    for i,(kind,status,description,arts) in enumerate(jobs):
      rid='run-'+(now-timedelta(seconds=len(jobs)-i)).strftime('%Y%m%dT%H%M%SZ')
      run=ctl('new-run','--loop-dir',str(L),'--run-id',rid,'--case-key','c8-input32k-output1k-prefix90','--kind',kind,'--command','Preparedv13 evidence;retrospective registration: '+description,'--code-ref','Frozen target unchanged; saved166profile CPU queries and exact source questions only; no fresh NPU/service work','--working-directory',str(N))
      args=['finish-run','--run-dir',run,'--status',status,'--correctness','not-applicable','--summary',description]
      for p in arts:assert p.exists(),p;args.extend(['--artifact',str(p)])
      ctl(*args);runs.append(dict(run_dir=run,status=status,description=description))
    action=('继续Loop004，v13是实质进展，不重跑v12采集或raw复制。先读profile-contract-summary-v13.json、H02-CP-reuse-contract-v13.json和H02-field-and-synchronization-ledger-v13.json。'
      '当前166固定路径CP cache-hit死生产者已结构闭合：88非空root各26build，local/global903初始化copy各1次，后25次suffixH2D共2200和5类GPUkernels各2200=11000实际执行，经nativeCONNECTION_IDS绑定无歧义。'
      'DSAbatchcache首次计算共享，后续local_common.seq_lens无内容读者；dataclass构造/replace无隐藏reader。下一候选是仅batchmiss生产suffix/GPUlength，保留首次GPU权威长度、固定query地址/全部CPUquery和每组slot/ring。不需为死结果再证明25份值相等；勿重复消耗源码审计。尚无实现/速度收益。'
      '全7392H2D互斥分账，utility2064分13完整callers；块表1144已有earlycommit，dirtyrow/version发布条件候选；spec5索引360可几何模板，draftIDs仍live且event/deque保命已存在；4scatterindices×72可稳定uniformspec打包条件候选。'
      'QSL第二上传64不能blindskip：16actual2req12token、48actual4req24token，且中间GPUtailfill；lookback在inputprepguard外；numDecodeDraftGPU消费者未全闭合。同步4sites312不删。'
      '事件v1早晚方向误写已撤销，由saved-event-order-and-resource-metrics-v13-v2.json按真实时间修正56pair。资源PMU名/原值/freq/TASK重数已存，未归一化。v1相关SQL145s中断已保留，v2等价IN完成；nativeNULL排序、reexport、一次SSH255也保留且修正。'
      '优先用新166全部32DB/11精确TP8cohort闭合forward/graph/MC2/COMM和各模块资源窗口，保留graph重复执行定义身份，不用nearesttime/ordinal替代native关系。资源metrics清单不是MFU或最低模块排序；如缺定义只定点查SDK exporter。'
      'H01 EngineCore onCPU/producerready与H03hash/frontend成本不在workerprofile中，仍需最小有界证据；先用现有资料，不先全代码扫描。所有v13任务terminal，无新NPU/推理/服务改动；lasthealth仍v12post2026-10-30T21:56:36UTC，勿冒充新鲜健康。全26模块和条件/跨边/理论界未完，Goalactive，0wholemodulesclosed。')
    checkpoint=dict(status='PROGRESS',previous_turn_classification='PROGRESS:v12full16stack-shape capture',current_verified_changes=['CPbatch-hit11000nativekernels/2200suffixH2D unused-result proof in fixed path.','7392H2D conserved families and2064utility/13field contracts;conditional geometry/block/packing candidates.','Corrected actual valid-sync ordering and32DBPMU/frequency/identity inventory;no wait removal or normalized utilization claim.'],summary=str(O/'profile-contract-summary-v13.json'),candidate_ledger=str(O/'candidate-profile-ledger-v13.json'),framework_candidate_ledger=str(O/'framework-candidate-ledger-v13.json'),module_ledger=str(O/'module-profile-ledger-v13.json'),branch_matrix=str(O/'module-phase-branch-matrix-v13.json'),report=str(O/'PROFILE-FINDINGS-v13.md'),source_gate=str(O/'PROFILE-CODE-QUESTION-GATE-v13.json'),coverage_gaps=str(O/'module-coverage-gaps-v13.json'),pending_full_scope_domains=remaining,remote_job_status='Allv13jobs terminal; four execution errors and one invalid event-order analysis retained/corrected. CPUreadonly only;lastlivehealthv12post.',next_action=action,whole_modules_closed=0,overall_complete=False,updated_utc=stamp)
    write(O/'REVIEW-CHECKPOINT-v13.json',checkpoint)
    for p in [L/'state.json',T/'index.json']:
      d=read(p);assert d['status'] in ['EVALUATING','ACTIVE'];d.update(next_action=action,updated_at=stamp);write(p,d)
    for p in [L/'status.md',T/'status.md']:
      lines=[('- 下一步：'+action) if line.startswith('- 下一步：') else ('- 更新时间：`'+stamp+'`') if line.startswith('- 更新时间：') else line for line in p.read_text().splitlines()];p.write_text('\n'.join(lines)+'\n')
    if not (O/'profile-summary-before-v13.json').exists():(O/'profile-summary-before-v13.json').write_bytes((R/'profile-summary.json').read_bytes())
    prof=read(R/'profile-summary.json');prof['supplementary_CP_contract_v13']=dict(summary=checkpoint['summary'],dead_value_native_kernels=11000,dead_value_suffix_H2D=2200,optimization_implementation=False,whole_modules_closed=0,overall_complete=False);write(R/'profile-summary.json',prof)
    for p in [N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
      d=read(p);d.update(latest_profile_first_review=checkpoint,overall_complete=False,whole_modules_closed=0,updated_utc=stamp);d['next_read_only_action' if p==R/'STATE.json' else 'next_action']=action
      if p==W/'STATE.json':d.update(profile_candidate_ledger=checkpoint['candidate_ledger'],profile_module_ledger=checkpoint['module_ledger'])
      if p in [N/'RECOVERY.json',R/'PARALLEL-RESUME-PACK.json']:d['read_first']=[checkpoint['summary'],str(O/'H02-CP-reuse-contract-v13.json'),str(O/'H02-field-and-synchronization-ledger-v13.json'),checkpoint['coverage_gaps']]
      if p==N/'RECOVERY.json':d['original_service']='No v13 device/service work or fresh health assertion; last verified v12post2026-10-30T21:56:36UTC:originalbasehealthy,17NPUPIDsbaseowned.'
      write(p,d)
    compact=ctl('compact','--task-dir',str(T));validation=ctl('validate','--task-dir',str(T));write(O/'task-run-registration-v13.json',dict(runs=runs,compact_output=compact,validation_output=validation,whole_modules_closed=0,overall_complete=False))
    paths=[O/f for f in ['profile-contract-summary-v13.json','H02-CP-reuse-contract-v13.json','H02-field-and-synchronization-ledger-v13.json','resource-profile-inventory-v13.json','PROFILE-CODE-QUESTION-GATE-v13.json','candidate-profile-ledger-v13.json','framework-candidate-ledger-v13.json','module-profile-ledger-v13.json','module-phase-branch-matrix-v13.json','module-coverage-gaps-v13.json','PROFILE-FINDINGS-v13.md','REVIEW-CHECKPOINT-v13.json','task-run-registration-v13.json','profile-summary-before-v13.json']]
    paths += [N/f for f in ['saved-prepare-consumers166-v13.json','metadata-replace-contract166-v13.json','metadata-replace-contract166-v13-v2.json','cp-dead-producer-device-tasks166-v13.json','cp-dead-producer-device-tasks166-v13-v2.json','blocktable-copy-contract166-v13.json','blocktable-copy-contract166-v13-retry.json','event-resource-schema166-v13.json','saved-events-resources166-v13.json','saved-events-resources166-v13-v2.json']]
    paths += [N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json',R/'profile-summary.json',L/'state.json',T/'index.json']
    write(O/'checkpoint-manifest-v13.json',dict(generated_utc=stamp,artifacts=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],whole_modules_closed=0,overall_complete=False))
    print(json.dumps(dict(checkpoint=str(O/'REVIEW-CHECKPOINT-v13.json'),runs=len(runs),statuses=[r['status'] for r in runs],validation=validation,whole_modules_closed=0,overall_complete=False),ensure_ascii=False))

if __name__=='__main__':main()
