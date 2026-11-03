from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import json,hashlib,copy

N=Path(__file__).parent;O=N/'profile-first-20261031'
R=Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
W=R/'whole-scope-20261030'
def read(p):return json.loads(p.read_text())
def write(p,d):
 assert not p.exists(),('refuse_overwrite',str(p));p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def receipt(name):
 d=read(N/name);assert d['returncode']==0,(name,d['stderr']);return json.loads(d['stdout'])

def main():
 now=datetime.now(timezone.utc).isoformat()
 identity=receipt('profile-task-identity166-v9.json');origin=receipt('profile-all-compute-origins166-v9-v2.json')
 audit=receipt('profile-origin-source-audit166-v9-v2.json');post=receipt('profile-postprocess-csv-kinds166-v9.json')
 assert origin['summary']['compute_tasks']==36521 and origin['summary']['compute_definition_count']==20021
 assert origin['summary']['linked_compute_tasks']==18371 and origin['summary']['unlinked_compute_tasks']==18150
 assert audit['summary']['exact_unique_csv_compute_matches']==36279 and audit['summary']['ambiguous_matches']==0
 assert audit['summary']['identity_strength_counts']=={'FULL_FOUR_FIELDS':36279}
 assert post['summary']['selected_add_argmax_copy_counts']=={'Slice':195,'Add':65,'Cast':65,'ArgMaxV2':65,'ViewCopy':143}
 assert sum(len(v) for v in audit['project_leaf_locations'].values())==228
 old=read(O/'aggregate-recheck.json');old_types={r['type'] for r in old['old_capture']['tables']['operator_types']}
 db_types={r['op_type'] for r in origin['type_counts']}
 assert db_types-old_types=={'IndexFill'}
 assert old_types-db_types=={'allgatherAicpuKernel','allreduceAicpuKernel','alltoallAicpuKernel','reduce_scatterAicpuKernel'}
 kinds=post['summary']['unmatched_csv_kind_audit']
 comm=sum(r['exact_one_COMM'] for r in kinds);aicpu=sum(r['exact_one_TASK'] for r in kinds)
 unresolved=sum(r['rows'] for r in kinds if not r['exact_one_COMM'] and not r['exact_one_TASK'])
 assert (comm,aicpu,unresolved)==(1849,278,242)
 assert 36279+comm+aicpu+unresolved==audit['summary']['csv_rows']==38648
 table_partition=dict(csv_rows=38648,exact_TASK_COMPUTE_records=36279,exact_COMM_intervals=comm,exact_AICPU_TASK_records=aicpu,
                      unresolved_vector_rows=unresolved,partition_conserved=True,durations_additive=False,
                      unresolved_types=[r for r in kinds if not r['exact_one_COMM'] and not r['exact_one_TASK']])
 # These two frozen source questions are downstream of the full origin census.
 src=read(R/'continuation-20261030/prepare-input-analysis.json')['sources']
 proposer=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py'
 dspark=W/'frozen-source-snapshot/vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py'
 assert hashlib.sha256(proposer.read_bytes()).hexdigest()=='458fb50e6839f3a353ddc7181ace4fdae02e35093d8ccd1c4327dc24ae5fff31'
 assert hashlib.sha256(dspark.read_bytes()).hexdigest()=='70d216de21875e679d510153ebdb0f0ccf130875a0e644f0673d60b67b6b633c'
 f30=dict(profile_scope='All selected postprocessing source ancestry in one EP3 diagnostic DB; native-task identity retained server-side.',
  native_counts=post['summary']['selected_add_argmax_copy_counts'],postprocess_signature_groups=post['summary']['postprocess_signature_groups'],
  greedy_branch='MEASURED_AT_65_NATIVE_ARGMAX_CALLS_FROM_LINE1599; does not establish every request/iteration or exclude other branches globally',
  phase_records=[dict(host_context='DUMMY_ANCESTRY',B=4,step_records=5),dict(host_context='PROPOSAL_ANCESTRY',B=3,step_records=5),dict(host_context='PROPOSAL_ANCESTRY',B=4,step_records=55)],
  direct_dense_model=dict(V=129280,embedding_width=256,step_row_counts=[3,4],BF16_bias_bytes_per_row=129280*2,
    BF16_full_dense_transition_table_bytes=129280*129280*2,direct_bias_MAC_per_step={str(b):b*129280*256 for b in [3,4]},
    meaning='Logical uncompressed BF16 table size and current direct-dense MAC counts only. Not compulsory HBM traffic, an unconditional work lower bound, hardware throughput or a claimed gain.'),
  source_answers=['Per-step current code mutates logits[:,idx] with BF16 bias, selects argmax, optionally maps token space and copies the next ID into a stable buffer.',
    'Each next Markov embedding consumes the preceding selected token. Direct evaluation cannot batch unknown future inputs without additional lookahead/precomputation; this is not a lower bound over all algorithms.',
    'Probability path calls _sample_draft_from_logits and retains draft_probs; selected65 argmax records establish the greedy else branch for those calls.',
    'Dynamic DSpark confidence update consumes last_hidden_states and draft_token_ids, not the full summed logits argument. Actual dynamic_spec value remains unverified; initializer activates it when dynamic_spec_config.method == dspark.',
    'DSpark initializer explicitly sets use_cuda_graph=False. This applies to draft execution, not target graph replay.',
    'Early return depends on num_speculative_tokens==1 or parallel_drafting; actual resolved parallel_drafting/vocab-map flags are not attested by source alone.'],
  variants=[
   dict(id='F30-STRIDED-ADD-ARGMAX',status='CONDITIONAL_LOCAL_SEMANTIC_CANDIDATE_WITH_MEASURED_BRANCH',
     graph='Read the current strided logits slice and bias, preserve BF16 add rounding and optional writeback, reduce with existing argmax semantics, produce INT64 IDs and keep token remap/copy order.',
     supports='65 Add+ArgMax+Cast records plus195 Slice/143 ViewCopy records; source binds each logical step.',
     constraints=['BF16 materialization/rounding before compare','NaN/tie/index semantics','same logits writeback and any external aliases','output INT64 and vocabulary remap','row padding/mask and stable buffer generation','per-step token recurrence and existing stream/event dependencies'],
     open='Historical stride/storage ownership and native Slice/ViewCopy semantics are not attested. A shape reporting the full base allocation does not prove all bytes are physically copied. Score-write deletion requires stronger all-consumer proof.'),
   dict(id='F30-VOCAB-OWNERSHIP',status='CONDITIONAL_CROSS_CARD_REORGANIZATION',
     graph='Keep base-head and compatible Markov work in matching vocabulary partitions through greedy selection; preserve global tie/index semantics and publish the same selected IDs.',
     constraints=['matching vocabulary partitions/group/rank ownership','same bias and add numerical boundary','token-space mapping and selected ID publication','probability/entropy/logprobs paths need their full distribution contract','other model/head consumers and graph lifetime'],
     open='This does not prove an AllGather can be removed. No per-rank ownership/complete consumer or end-to-end benefit is established.'),
   dict(id='F30-MARKOV-ROW-REUSE',status='CONDITIONAL_CROSS_ITERATION_COMPUTE_MEMORY_TRADEOFF; UNPRICED',
     graph='Reuse an exactly computed Markov bias row for a repeated Markov input ID under frozen embedding/projection parameters; partial cache and full precomputation are distinct domains.',
     constraints=['actual Markov input token space','immutable parameter/version namespace','matching GEMM reduction/rounding and inference numeric mode','no stochastic/stateful embedding/projection behavior','cache invalidation/capacity/lookup overhead and device ownership'],
     open='Actual repeated-ID frequency and memory headroom are absent. A full dense BF16 table is33426636800 logical bytes before indices/replication; this is not a feasible-capacity or speedup claim.')],
  source_snapshots=[dict(path=str(proposer),sha256=hashlib.sha256(proposer.read_bytes()).hexdigest(),ranges=[[1200,1262],[1510,1705]]),
                    dict(path=str(dspark),sha256=hashlib.sha256(dspark.read_bytes()).hexdigest(),ranges=[[50,78]])],
  closed=False)
 h02=dict(status='REFINED_EXISTING_H02; NO_GAIN_OR_REDUNDANCY_CLAIM',
  measured_origins=['CP build140-142:350/350/1050 native API records in this DB.','DSA build1008-1028 and C2 metadata1327-1368 fallback primitives have source-linked records.'],
  existing_optimizations=['Batch metadata dictionary is already reset once per build and shared independent of physical block IDs.',
    'Slot mapping coordinate/flat buffers already share within a framework KV group using compression ratio and storage block size keys.',
    'C2 lengths share through batch_shared[lengths:c2]; SMLA/QLI metadata have staged publication.',
    'C2 ring metadata publication stays group-local; skip-update and actual token/request limits are semantic fields.',
    'CP local RoPE reuses slices of global cos/sin to avoid overwriting still-referenced process-wide buffers.'],
  remaining_candidate='Remaining CP local offset/device-seq-length suffix operations and fallback field transforms may be fused/reorganized only under their exact group/query/version/readiness and graph-address contract.',
  exclusions=['Do not claim cross-group metadata sharing is absent.','Do not replace authoritative device seq_lens with optimistic CPU lengths after rejection.','Do not hoist group-local physical slots or ring owners as batch-global fields.','Do not infer repeated350source records represent350 identical tensors or removable work.'],
  source_snapshots=[dict(path=k,sha256=src[k]['sha256']) for k in ['vllm_ascend/attention/dsa_v41.py','vllm_ascend/attention/context_parallel/dsa_v41_cp.py']],
  missing=['Per-build shared dictionary hits/misses, common-field and slot generations, all graph consumers and actual copy/event linkage.'])
 # Finite module requirements are stated against the frozen case; an observed
 # source location is not used to silently close an unobserved condition.
 definitions={
 'M05':('embedding and pure-text wrapper',['standard token embedding and SP mask/padding','vision/multimodal branches not requested by frozen text-only HTTP body; implementation selection still conditional'],['ops/vocab_parallel_embedding.py','layers/vocab_parallel_embedding.py'],'Differentiate target/draft masks and token ownership; source wrapper name vl_model does not prove image execution.'),
 'M06':('mHC pre/post, residual and cross-layer edges',['HcPre/HcPost helper source and target/draft occurrences','coefficient/accumulator consumers and cross-layer fusion boundary'],['models/deepseek_v41/model.py'],'Helper records combine target/draft; preserve all coefficient/residual consumers and numeric boundaries.'),
 'M07':('Q/KV projections, norm, quantization and RoPE',['selected W8A8/BF16 projection and native rotary paths','different row domains, batch layouts and global/local RoPE views'],['ops/rope_dsv4.py','ops/cv_linear.py','quantization/methods/w8a8/w8a8_dynamic.py'],'Generic linear/norm source cannot alone assign a projection role or exact tensor edge; reuse v3/v4 role bindings.'),
 'M08':('compressor and persistent ring/latent state',['retained C1/C2 roles and source/native compressor paths','skip-ring/actual-token/padded-row state, write ownership and rounding'],['models/deepseek_v41/compressor.py','ops/triton/compressor/compressor_triton.py'],'C2 metadata and ring buffers are group-specific; graph replay and rollback generations remain open.'),
 'M09':('indexer projection, quantization, scoring and key/index state',['select_projected/project_weights/update_keys and metadata paths','cross-layer key/index reuse and per-request position validity'],['models/deepseek_v41/indexer.py','ops/triton/quantize_indexer_query.py','ops/triton/prepare_indexer_indices.py'],'242 CSV identity gaps include indexer Triton records; do not invent matching source or layer phase.'),
 'M10':('SWA/compressed KV and sparse attention',['source-linked _forward_attention/scatter and sparse-index selection','SWA/long/index cache planes, valid lengths, C1/C2 and CP views'],['attention/dsa_v41.py'],'Metadata builder and attention compute are separate scopes. KV physical slots/lifetime and complete active consumers remain open.'),
 'M11':('attention output head restore and projections',['restore_tp_heads/_forward_o_proj and selected F14-F16/F32 chains','alltoall/transpose/reduce and different shape/row ownership'],['attention/context_parallel/dsa_cp.py'],'Communication missing from COMPUTE cannot be declared absent; group and consumer contract remain scoped.'),
 'M12':('MoE routing/dispatch/experts/combine',['router/rms/gating/native fused_experts paths','target/draft shapes and selected nonMega native branch; per-rank expert counts'],['ops/fused_moe/fused_moe.py','ops/fused_moe/router/fused_topk_router.py','ops/fused_moe/moe_comm_method.py'],'Native compound remains opaque internally.166count observations and old169timing are separate.'),
 'M13':('shared experts and routed/shared merge',['shared expert gate-up/act/down source paths','Tensor/legacy tuple return and scaling/merge consumers'],['ops/fused_moe/shared_experts.py'],'Retain fixed-source Tensor contract and alias/numerical merge boundaries; legacy alternative is not assumed active.'),
 'M14':('Engram host/UVA/hash/gate',['hash/cache/write/request-index and host-gather-dequant paths','gate ordinary primitives and parameter/WKV/copy boundaries'],['models/deepseek_v41/engram/'],'Hash-cache helpers already exist; actual per-request reuse, page/memory cost and phase/lifetime are not all observed.'),
 'M15':('final collapse/norm and MTP auxiliary state',['full-row norm vs selected logits rows','auxiliary/full residual/confidence consumers'],['models/deepseek_v41/model.py'],'hc_collapse is a shared helper and API records are not invocation count. Do not delete full-state consumers from a logits-only view.'),
 'M16':('target/draft vocabulary head and reconstruction',['bound base-head logits/AllGather and target selected rows','vocab sharding, ties, mapping, other score consumers'],['model_executor/layers/logits_processor.py'],'F30ownership variant is conditional; target bonus/logprobs and draft Markov consumers differ.'),
 'M17':('sampling/rejection/bonus/logprobs',['source-linked greedy_sample and rejection_greedy Triton','probability/entropy/raw-logprob/bonus and RNG alternatives'],['v1/sample/sampler.py','sample/sampler.py','sample/rejection_sampler.py','ops/triton/reject_sample.py'],'Greedy native records support selected calls, not absence of every probabilistic or logprob branch.'),
 'M18':('DSpark context projection and KV publication',['39 context projection/store chains fromv7','standard SWA store, slot masking/INT32 pairs and context cosine views'],['models/deepseek_v41/dspark.py'],'Slot/position values and alias lifetime remain unobserved; context can be shared only with complete consumers.'),
 'M19':('DSpark decoder/HC/embedding and SP',['dummy/proposal repeat and three-layer decoder source contexts','SP-before-repeat variant, padding/mask/residual/clone consumers'],['models/deepseek_v41/dspark.py','models/common/ops/sequence_parallel.py'],'DSpark eager contract is explicit. Generic dflash/qwen parent file names do not select another algorithm by themselves.'),
 'M20':('DSpark Markov/logits/selection/confidence',['65 source-linked greedy Markov add/argmax/cast steps','probabilistic fallback, vocab mapping, dynamic confidence and ID recurrence'],['spec_decode/llm_base_proposer.py','model_executor/models/qwen3_dspark.py'],'F30 refined with three conditional variants; dynamic/parallel/vocab flags and full aliases still need runtime attestation.'),
 'M21':('draft/target metadata and input publication',['prepare_next_token_ids_padded/prepare_inputs_padded and slot mapping','one IndexFill [3,1] in proposal context; independent step buffers and rejected-token correction'],['spec_decode/dspark_proposer.py','spec_decode/llm_base_proposer.py','ops/triton/compute_slot_mapping.py','ops/triton/spec_decode/dspark_swa_indices.py','ops/fused_moe/prepare_finalize.py'],'IndexFill is a real observed branch beyond the earlier87-type set; it is not evidence of a high-value standalone optimization.'),
 'M22':('capture/replay/parameter update and streams',['1650 graph definitions each have11 executions;18371 singly-executed definitions retain native source links','recorded host ancestry vs replay phase, persistent addresses and metadata stages'],['compilation/acl_graph.py','compilation/decorators.py'],'18150 graph execution records lack PYTORCH_API links in this relation. Do not deduplicate them or label them all dummy/formal.'),
 'M23':('TP/DP/EP/CP communication and ownership',['1849 COMM intervals and278 AICPU records strictly located; physical hcom compute rows are a separate level','group/rank/dtype/count/order and native MC2 internal communication'],['distributed/','models/common/ops/sequence_parallel.py'],'COMM intervals and nested task durations overlap; record counts, MAC proxy and end-to-end exposure are different metrics.'),
 'M24':('cross-iteration state and reuse',['source-linked acceptance-count correction and existing persistent hash/cache paths','request remap, rejected rollback, slot/ring generation and candidate row-cache invalidation'],['spec_decode/utils.py','models/deepseek_v41/engram/hash_state.py'],'Source contracts do not supply historical generation/alias values. F30 row reuse requires exact model/token/numeric namespace.')}
 oldmods=read(O/'module-profile-ledger-v8.json');framework=read(O/'framework-module-branch-matrix-v8.json')
 matrix={m['id']:copy.deepcopy(m) for m in framework['modules']}
 leaves=audit['project_leaf_locations'];allfiles=origin['project_file_api_counts']
 for mid,(phase,conditions,selectors,gap) in definitions.items():
  locations=[dict(source_file=k,**r) for k,vs in leaves.items() if any(s in k for s in selectors) for r in vs]
  files={k:v for k,v in allfiles.items() if any(s in k for s in selectors)}
  matrix[mid]=dict(id=mid,phase=phase,branches=[dict(branch=conditions[0],state='PROFILE_PATHS_OBSERVED_IN_RETAINED_SCOPE',evidence='Full origin census plus existing model role evidence throughv8.',gap=gap),
    dict(branch=conditions[1],state='CONTRACT_REVIEW_PARTIAL; ACTUAL_CONDITION_OR_CONSUMERS_NOT_CLOSED',evidence='Existing scoped contracts and question-gated v9 source answers.',gap=gap)],
    source_file_API_record_counts=files,leaf_API_record_locations=locations,
    source_selector_is_module_identity_proof=False,source_scope='One EP3 recorded host/native origin census; patterns are entry-point hints and overlap across modules.',
    source_record_count_is_function_call_count=False,closed=False)
 assert set(matrix)=={m['id'] for m in oldmods['modules']} and len(matrix)==26
 matrix['M20']['specific_review']='F30-proposer-review-v9.json';matrix['M04']['metadata_refinement']='input-metadata-review-v9.json'
 for m in oldmods['modules']:
  m.update(phase_branch_record='module-phase-branch-matrix-v9.json#'+m['id'],whole_module_closed=False,
           review_state='PARTIAL_PHASE_BRANCH_ORIGIN_REQUIREMENTS_REVIEWED; NOT_WHOLE_MODULE_CLOSED')
  m['profile_entry_points']=list(dict.fromkeys(m['profile_entry_points']+['origin-coverage-summary-v9.json','module-phase-branch-matrix-v9.json']))
 oldmods.update(generated_utc=now,whole_modules_closed=0,overall_complete=False,scope='26-domain finite branch/phase requirements plus recorded origin evidence; module universe remains provisional outside the retained source census.')
 cs=read(O/'candidate-profile-ledger-v8.json');c=next(c for c in cs['candidates'] if c['id']=='F30')
 c.update(profile_review_state='MODEL_AND_PROPOSER_GREEDY_POSTPROCESS_BOUND; PHASE_FLAGS_ALIAS_AND_RESOURCE_OPEN',
          conclusion_changing_gap='Observed65greedy steps are source-bound, but formal-client phase, actual dynamic/parallel/vocab-map flags, external logits aliases and same-domain resource benefit remain open.',
          target_source_gate=str(O/'PROFILE-CODE-QUESTION-GATE-v9.json'))
 c['scoped_detail']['proposer_postprocess_v9']=f30;c['evidence_files'].append('F30-proposer-review-v9.json')
 cs.update(generated_utc=now,parent_ledger='candidate-profile-ledger-v8.json',candidate_universe_complete=False,overall_complete=False,review_v9_utc=now)
 hc=read(O/'framework-candidate-ledger-v8.json');h=next(c for c in hc['candidates'] if c['id']=='H02');h['metadata_refinement_v9']=h02;h['source_answer']+=' Batch/group metadata sharing and CP RoPE view reuse also already exist; remaining work must be reviewed within their scopes.'
 hc.update(generated_utc=now,parent_ledger='framework-candidate-ledger-v8.json')
 oldfamilies={r['type'] for c in cs['candidates'] for rs in c['family_name_hits_not_role_proof'].values() for r in rs}
 summary=dict(generated_utc=now,status='PROGRESS',origin=origin['summary'],CSV_audit=audit['summary'],CSV_partition=table_partition,
  new_observed_type_vs_prior_sets=['IndexFill'],prior_types_absent_from_COMPUTE=sorted(old_types-db_types),
  source_leaf_locations=228,source_leaf_files=len(leaves),source_ancestry_files=65,
  selected_proposer_postprocess=post['summary'],old_family_index_types=len(oldfamilies),old_types_without_legacy_family_links=sorted(old_types-oldfamilies),
  stale_family_index_meaning='Initial family tags are incomplete indexing, not proof that every omitted generic type lacked subsequent scoped review.',
  reports=[str(O/'F30-proposer-review-v9.json'),str(O/'input-metadata-review-v9.json'),str(O/'module-phase-branch-matrix-v9.json')],
  saved_server_results=[{k:r[k] for k in ['full_result','result_sha256']} for r in [origin,audit,post]],
  no_new_device_or_inference_work=True,whole_modules_closed=0,overall_complete=False,
  remaining=['242 vector CSV rows lack exact TASK identity even though they share four type names with DB records; inspect original keys/flows next without nearest-time repair.',
    '18150 graph executions have no native PYTORCH_API association; actual replay/client phases and graph-definition-to-module evidence remain partial.',
    '26-module universe completeness outside this retained source census, frontend functions, dynamic branch flags, all consumers/generations and same-domain resource ceilings/lowest-utilization evidence.'])
 write(O/'origin-coverage-summary-v9.json',summary)
 write(O/'F30-proposer-review-v9.json',f30);write(O/'input-metadata-review-v9.json',h02)
 write(O/'module-phase-branch-matrix-v9.json',dict(generated_utc=now,modules=[matrix[k] for k in sorted(matrix)],
    module_universe='PROVISIONAL26; all recorded source locations are retained, but source paths/overlapping selectors are not complete semantic module attribution.',
    all_phases_closed=False,all_conditions_closed=False,whole_modules_closed=0,overall_complete=False))
 write(O/'module-profile-ledger-v9.json',oldmods);write(O/'candidate-profile-ledger-v9.json',cs);write(O/'framework-candidate-ledger-v9.json',hc)
 write(O/'module-coverage-gaps-v9.json',dict(generated_utc=now,remaining=summary['remaining'],
    next_original_data_query='For the242 vector rows, inspect original CSV stream/task/model/start/end and exact DB identity candidates/graph execution metadata; retain any exporter mismatch explicitly. Then audit graph definition/capture/replay source association before collecting new device data.',
    unresolved_framework_costs=['H01 same-window on-CPU / producer-ready','H02 per-field/group generation and readiness','H03 exact token/hash namespace and cost'],
    whole_modules_closed=0,overall_complete=False))
 report='''# 全量执行来源与阶段审查 v9

本轮以一个固定 EP3 原始数据库为完整查询域，补上逐执行记录、API和完整调用栈的归属审计，并复核同目录全部CSV记录。它扩展已有16卡汇总和候选证据，不把单卡结果替代全卡、全阶段覆盖。整体仍未完成，0个整模块关闭。

## 记录身份与完整性

- 36521条计算执行共享20021条计算定义：18371条定义各执行一次，另1650条定义各执行11次。按 `TASK.rowid` 保留执行次数；`globalTaskId` 不能作为单次执行唯一键。
- 18371条执行关联14821条原生API，全部14821条调用链均有完整帧，合计356826条帧记录、501个不同帧、65个项目来源文件、228个最内层项目位置。18150条重复图执行未关联到原生PyTorch API，继续保留来源和阶段缺口。
- 原CSV共38648条、88种类型。36279条与TASK/COMPUTE按stream/task/model/start四字段严格唯一匹配，duration差全部为0；1849条与COMM按原name/start/end严格匹配；278条与AICPU TASK严格匹配；余242条向量记录身份未闭合。四类计数完整守恒。COMM区间与其内部任务可能重叠，不能把两层时长相加。
- 242条未匹配记录来自quantize_indexer_query88、prepare_indexer_indices88、compressor pool33、cache_update33。数据库也有相应类型，类型和数量相同不能替代执行身份。下一步先核对原键和graph元数据，不作最近时间补配。
- 与旧87类型集合相比，观察到1条 `IndexFill`：INT32的[3,1]输入/输出，调用链定位到 `prepare_next_token_ids_padded`。这是一个已运行的分支，不据此声称它是高收益热点。旧family索引遗漏若干通用类型，但后续局部审查已覆盖其中一些；不能把标签缺失当作完全未分析。

## F30：Markov后处理

全量来源审计补上了此前DSpark模型帧筛选之外的proposer路径：65次BF16 Add、65次ArgMaxV2、65次INT32→INT64 Cast、195次Slice、143次ViewCopy。按记录调用栈分为dummy的5个B4步骤、proposal的5个B3步骤和55个B4步骤；proposal不等于已证明的正式客户端阶段。

定点代码确认：每一步先计算当前token的Markov embedding/bias，再修改该步logits，argmax后发布下一token。因而可继续分析三个有区别的变体：保留BF16舍入/回写的strided Add+ArgMax边界；在词表所有权一致时审查跨卡logits/Markov/greedy选择重组；对完全相同Markov输入ID考虑跨迭代bias行复用。它们都未证明净收益。

不能无条件并行合并所有Markov步骤，因为直接求值时下一步输入取决于上一步选择；若允许预计算或额外推测工作，则需另立算法/资源域。当前V129280、宽度256的直接bias GEMM，每步B3/B4分别为99287040/132382720 MAC；完整未压缩BF16转移表为33426636800字节，每行258560字节。这些只是声明域内的工作/存储模型，不是必要HBM流量、全算法下界、可用显存或可达提速。

所见65个argmax调用证明这些样本进入greedy分支。概率分支仍有完整分布/RNG约束；动态confidence的源码消费者是hidden states和draft IDs，但实际开关尚未绑定。词表映射、parallel_drafting、外部logits别名与score写回生命周期继续保留。ViewCopy报告基张量形状，不能据此认定它每次真的搬运整个基张量。

## H02：元数据机会收紧

输入准备的来源已细化到CP local seq_lens/offset、DSA slot坐标、C2 ring/完成掩码等函数。源码复用审查发现，batch metadata、C2 lengths和组内slot坐标已经共享；设备元数据已有分阶段发布；CP局部RoPE也已通过global切片复用，以免覆盖仍被引用的缓冲。

剩余机会只能针对确实重复的字段变换和准备工作：保持组内物理槽与ring所有权，保留拒绝后权威设备seq_lens修正，以及捕获地址、actual/padded边界和代际。不能把已共享的字段再次列为“尚未共享”，也不能由350条同源记录推断350份相同数据。

## 覆盖状态

26模块均已有有限phase/condition要求与对应的profile入口，六个框架范围沿用v8并补入本轮证据。表中保留已观测路径、未请求条件、源码合同和未定分支；共享helper的API数不是函数调用数，文件名包含qwen/dflash/vl也不自动证明另一模型、算法或图像分支被启用。

未关闭项包括242条向量记录身份、图定义与replay阶段/模块对应、前端函数profile、动态开关、完整消费者/别名/状态代际，以及同域资源上界、利用率排名和关键路径原因。单卡来源覆盖、26行表或报告成稿均不是整体完成条件。无目标实现修改、服务操作或新推理/NPU负载。

本轮保留两次查询失败：首轮误把globalTaskId当唯一执行；首轮CSV解析未处理身份字段N/A。原始记录审计后分别修订为执行行身份及显式缺失字段。成功CSV匹配实际全部为完整四字段，没有使用缺失model的备用匹配。
'''
 p=O/'PROFILE-FINDINGS-v9.md';assert not p.exists();p.write_text(report)
 print(json.dumps(dict(status='PROGRESS',compute_executions=36521,source_linked=18371,CSV_partition=[36279,1849,278,242],F30_greedy_steps=65,module_requirements=26,whole_modules_closed=0,overall_complete=False,summary=str(O/'origin-coverage-summary-v9.json')),ensure_ascii=False))

if __name__=='__main__':main()
