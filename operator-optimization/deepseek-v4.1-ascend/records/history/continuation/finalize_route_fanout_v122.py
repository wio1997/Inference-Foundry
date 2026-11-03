"""Record saved per-token fanout evidence; no device work or native closure."""
from copy import deepcopy
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v90 import persist_refinement


def main():
    assert not (O/'REVIEW-CHECKPOINT-v122.json').exists()
    preflight=read(O/'route-fanout-preflight-v122.json')
    assert preflight['status']=='PASS' and preflight['checks']==7071
    names=['route-fanout-inputs-v122.json','route-fanout-source-contract-v122.json',
        'route-fanout-packet-and-return-models-v122.json','route-fanout-cohort-evidence-v122.json',
        'route-fanout-variants-and-legality-v122.json','route-fanout-local-graph-map-v122.json',
        'route-fanout-theory-and-measurement-v122.json']
    detail=dict(evidence=names,saved_snapshot_records=1392,formal_cohorts=87,
        source_archive_and_metadata_SHA_checks=2784,archive_bytes_read=35632142,
        target_remote_assignment_references=2565204,target_remote_token_destination_groups=2204823,
        target_structural_repeated_references=360381,draft_remote_assignment_references=731,
        draft_remote_token_destination_groups=727,draft_structural_repeated_references=4,
        selected_target_dispatch_logical_bytes_saved=1714345440,
        selected_draft_dispatch_logical_bytes_saved=-133440,
        packet_count_layout_models=12,independent_descriptor_clip_cases=6400,
        existing_relation_refined='REL73-07',existing_group_refined='HG73-MOE',
        new_pairs=0,new_groups=0,new_native_promotions=0,new_device_work=False,
        native_quantized_byte_identity_proved=False,actual_network_traffic_or_speedup_proved=False,
        whole_requirement_closed=False)

    rel=deepcopy(read(O/'candidate-relation-clauses-v121.json'))
    clause=next(r for r in rel['clauses'] if r['id']=='REL73-07')
    assert {clause['a'],clause['b']}=={'F20','F21'}
    clause['evidence']=list(dict.fromkeys(clause['evidence']+names))
    clause['token_destination_fanout_v122']=dict(**detail,contract=(
        'Postmask routing IDs determine structural token/destination fanout in the archived linear EP16 domain. '
        'A metadata-complete packet may share identical original INT8 codes/scale while retaining old expert A counts, '
        'expanded/inverse/probability maps, receiver row order/capacity, GMM and BF16 return. '
        'D packet counts are a separate new protocol. Grouping/byte checks/packing/counts/reexpansion and lifetime costs remain. '
        'No router TopK/MC2 grouping CSE or extra once-per-token quantization credit.'))
    group=next(g for g in rel['higher_order_groups'] if g['id']=='HG73-MOE')
    group['token_destination_fanout_v122']=dict(**detail,contract=(
        'Select one packet layout/count plan and one dense-reexpansion or indirect-GMM variant. '
        'Retain the existing quantizer, original assignment rows and shared/routed expert rounding. '
        'No GMM, return, weighted-unpermute or full-model latency reduction follows from D<A. '
        'Packet staging, old source banks, shared memory budgets and remote-reader retirement are joint costs.'))
    selected={tuple(sorted((r['a'],r['b']))) for r in rel['clauses']}
    assert len(selected)==135 and len(rel['higher_order_groups'])==26
    rel.update(parent_relation_index='candidate-relation-clauses-v121.json',
        new_pair_clauses=0,new_higher_order_groups=0,token_destination_fanout_v122=detail,overall_complete=False)
    write(O/'candidate-relation-clauses-v122.json',rel)

    cov=deepcopy(read(O/'candidate-relation-coverage-v121.json'))
    pair=next(p for p in cov['pairs'] if {p['a'],p['b']}=={'F20','F21'})
    pair['token_destination_fanout_v122']=deepcopy(detail)
    cov.update(parent='candidate-relation-coverage-v121.json',
        parent_relation_index='candidate-relation-clauses-v122.json',
        scope='135 scoped conditional pairs /26 groups /35 model candidates;460 central pair reconciliations remain. Existing F20/F21 relation gains archived per-token fanout evidence, with no all-variant closure.',
        token_destination_fanout_v122=detail,overall_complete=False)
    write(O/'candidate-relation-coverage-v122.json',cov)

    nav=deepcopy(read(O/'candidate-pair-navigation-v121.json'))
    pair=next(p for p in nav['pairs'] if {p['a'],p['b']}=={'F20','F21'})
    pair['token_destination_fanout_v122']=deepcopy(detail)
    pair['evidence']=list(dict.fromkeys(pair.get('evidence',[])+names))
    nav.update(parent='candidate-pair-navigation-v121.json',
        parent_footprints='candidate-operator-footprints-v122.json',
        parent_relation_index='candidate-relation-clauses-v122.json',token_destination_fanout_v122=detail)
    write(O/'candidate-pair-navigation-v122.json',nav)

    footprints=deepcopy(read(O/'candidate-operator-footprints-v121.json'))
    for r in footprints['candidates']:
        if r['id'] in ['F20','F21']:
            r['token_destination_fanout_v122']=dict(evidence=names,
                local_candidate_subgraph='MC2_TOKEN_DEST_DISPATCH_V122',
                new_observed_or_global_catalog_nodes=0,new_local_candidate_nodes=13,
                original_assignment_work_and_return_retained=True)
    footprints.update(parent='candidate-operator-footprints-v121.json')
    write(O/'candidate-operator-footprints-v122.json',footprints)
    op=deepcopy(read(O/'operator-analysis-coverage-ledger-v121.json'))
    op['token_destination_fanout_v122']=detail
    write(O/'operator-analysis-coverage-ledger-v122.json',op)

    obligations=deepcopy(read(O/'coverage-current-obligations-v121.json'))
    for r in obligations['obligations']:
        if r['id']=='G04':
            r['remaining']=(
                'Raw C65536 binds 3200 later target records;1272 other later capacities remain unknown. '
                'Original observer joint counts and now per-token destination fanout are recovered:1392 archived rank/layer snapshots in87 formal EP16 cohorts, '
                'linear/no redundant experts, recorded C65536 with no count clipping. Target A2565204/D2204823 and draft A731/D727 are recorded-domain totals. '
                'Need historical loaded byte/scale identity, actual packet/count/placement ABI and old-row maps/allocator/capture/retirement. '
                'These archives are not joined to later TASK timings; no later native packet traffic or phase latency is proved.')
        if r['id'] in ['G04','G05','G06','G07','G08','G09','G10','G11','G13']:
            r['token_destination_fanout_v122']=deepcopy(detail)
        if r['id']=='G13':
            r['remaining']=(
                'v122 recovers recorded per-token MoE fanout, not Engram lookup IDs/hitrate. '
                'Do not reread the full-load quantizer as a new saving:it already shares per-row quantization within partitions. '
                'Need original packed code/scale equality or explicit byte-certified fallback, exact old maps/rounding, D-count publication/readiness, '
                'selected metadata layout, grouping/packing/reexpansion costs and overlapping storage/remote-reader retirement. '
                'Main layout saves12.97% target dispatch logical content but increases draft3.54%;compact draft outcomes depend on count-plan ABI. '
                'No physical wire/time or all-module ceiling follows. v121 F24/F34 one-budget result and v119 peer counts are prior answers;actual cache hits/budget remain unknown. '
                'v120 one-shot Pad authorization is complete/consumed;no equivalent retry or pending permission.')
    obligations.update(parent='coverage-current-obligations-v121.json',new_native_or_whole_closures=0,overall_complete=False)
    write(O/'coverage-current-obligations-v122.json',obligations)

    audit=deepcopy(read(O/'whole-scope-requirement-audit-v121.json'))
    audit.update(previous_goal_turn='PROGRESS:v121 explicit F24/F34 same-device budget relation;postcheck PASS647',
        input_checkpoint=121,followup_checkpoint=122,token_destination_fanout_v122=detail)
    for r in audit['requirements']:
        if r['id'] in ['MODEL_VERSION_HARDWARE_LOAD','ALL_MODULE_CONDITIONAL_BRANCHES',
                'CURRENT_FINITE_GRAPH_LEDGER','CROSS_OPERATOR_LAYER_CARD_ITERATION',
                'THEORY_AND_MEASURED_COMPARISON','CRITICAL_EVIDENCE_AND_FINAL_CLOSURE']:
            r.setdefault('evidence',[]).extend(names)
    write(O/'whole-scope-requirement-audit-v122.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v122.json',dict(
        status='SAVED_MOE_ARCHIVE_FANOUT_THEN_ONLY_EXISTING_QUANT_MAP_PACKET_AND_RETURN_CONTRACTS',
        questions=read(O/names[0])['questions'],
        abandoned_hitrate_estimate='Saved observer has no Engram lookup row IDs;hitrate remains unknown.',
        existing_source_and_profile_contracts_reused=True,new_target_source_files_requested=False,
        saved_tensor_data_read=True,remote_action='CPU-only archived-file reader finished and SSH explicitly closed.',
        Torch_or_NPU_or_model_loaded=False,new_device_work=False,
        Pad_authorization='GRANTED_AND_CONSUMED;no equivalent probe',overall_complete=False))
    write(O/'route-fanout-validation-history-v122.json',dict(events=[
        'v121 revalidated PASS647/manifest58 before work. Prior-turn progress was the explicit F24/F34 relation, not a new physical-memory proof.',
        'Original observer records lack Engram lookup IDs, so no cache hitrate is estimated. Per-token MoE TopK/mask archives can answer a different named transport-grouping question.',
        'Read exact authorized source-server archives only:1392 saved tensors plus1392 metadata files,35632142 bytes;SHA checked against the original tensor index. No Torch/NPU/model or new server data/source writes.',
        'One local SSH socket expired before a remote worker submission;local assertion stopped that attempt. Reconnected for the authorized archive read and closed the resulting socket. No device authorization used.',
        'Restricted stdlib ZIP/pickle decoder returns bounded CPU DTOs,imports no Torch. Locally retained real sample validates counts/epoch/mask/expert histogram and contiguous owner map.',
        'All1392 selected metadata records are linear EP16/no redundant experts/group members0..15/C65536,with receiver counts below capacity. Fanout counts conserve original A matrices/histograms/epochs;no duplicate active TopK experts.',
        'Target80 cohorts/1280 snapshots:Aremote2565204,Dremote2204823,repeated360381. Draft7/112:A731,D727,repeated4. Counts are summed captured rows across layers/ranks,not unique requests or full-run executions.',
        'Baseline dispatch is already INT8 H+32,not BF16;selected H+96 metadata-complete packet and replicated D counts reduce target logical dispatch1714345440B/12.9718%,increase draft133440B/3.5432%. BF16 return stays per assignment A.',
        'Twelve layout/count-plan cases retain alternative compact header/ordinal and external frame-epoch ABI. Compact draft can save20608B only under the zero-incremental-count-content contract;replicated counts reverse its sign. No universal draft impossibility claim.',
        'Full-load source already quantizes one row then copies it into expanded assignments inside partitions. Source IDs do not attest quantized byte/scale identity across all partitions/loaded keys. Byte-certified grouping needs its own comparisons/fallback/new counts.',
        '6400 independent small descriptor/order/capacity references pass. New thirteen-node conditional candidate DAG preserves original A counts/inverse/probs, GMM/return and packet/tile readiness,without adding observed/global catalog nodes.',
        'Legacy elapsed_CPU_process_seconds field uses time.monotonic():8.58s is CPU-only reader elapsed wall time,not measured CPU usage or NPU latency.',
        'Preflight7071 checks PASS. Existing REL73-07/HG73-MOE refined only:135 relations/26 groups/460 pending,356 semantic rows/26 modules. Original9/1352 regimes and all whole-scope gaps remain;Pad grant consumed.'],overall_complete=False))

    gap=(
        'Need loaded quantized code/scale raw-bit identity, actual linear placement/maps, exact old expanded/inverse/probability and clipping order, '
        'selected packet and D-count ABI plus offsets/participation/readiness. Count A remains expert GMM/return work;D is only new packet work. '
        'Compare grouping/identity checks/packing/count publication/reexpansion or indirect-GMM iterator, shared memory peak and all remote readers/queued writers/capture epochs. '
        'Saved postmask IDs do not prove physical network bytes or new implementation latency. No Engram lookup IDs/hitrate, later TASK join, physical capacity bound, actual MFU or speedup is established.')
    remaining=[r['id']+': '+r['title']+'. '+r['remaining'] for r in obligations['obligations']]
    remaining.append('v122 F20/F21 per-token fanout evidence:87 saved EP16 cohorts/1392 snapshots,not new native TASK promotions. Target structural repeated references360381;metadata-complete selected logical packet comparison12.97% dispatch/4.34% dispatch+return,not latency. Draft layout/count costs can reverse sign.135/26/460 and356/26 unchanged;all-module closure open.')

    report='''本轮围绕 MoE dispatch 的一个具体重组机会补齐了原始数据：同一输入 token 的多个专家若在同一 EP 目标上，能否只传一份激活，再按原来的专家行映射展开。先复查已有 profile 和观测索引，再只读取原服务器保存的 tensor/metadata 归档；没有加载 Torch、模型或 NPU，没有 replay，没有运行新的设备探针。先前批准的 Pad 探针已在 v120 完成，本轮没有重复使用该授权。

来源是原 observer 的 87 个正式 EP16 cohort、1392 份 rank/layer 快照及配套 metadata，共核对 2784 个文件的 SHA，读取 35,632,142B。快照含 post-call TopK IDs、实际 active mask、计数及映射。用标准库受限解码器分析，逐份核对了 epoch、专家 histogram、receiver counts 和既有 sender/receiver 矩阵。1392 份记录均声明 linear placement、无冗余专家、EP 成员 0..15；记录的 capacity 参数均为 65536，最大 receiver assignment count 为 14852，计数模型中没有截断。保留的实际样例 owner map 也符合连续分区。历史加载对象、全部实际指针/placement 与网络拓扑仍不由这些元数据单独证明。

target 的 80 个 cohort/1280 份快照中，远端 assignment 引用 A=2,565,204；按“源记录行、目标 EP”分组 D=2,204,823，结构性重复引用 360,381。draft 的 7 个 cohort/112 份快照为 A=731、D=727，仅重复 4 次。这里累计的是跨层、跨 rank 的记录行，不是去重请求数、全部 C8 token 数或执行次数。capture/replay 可覆盖归档文件，observer ordinal 不能直接作请求编号。这批 MoE 观测没有 Engram lookup IDs，不能据此估计设备行缓存命中率。

当前 dispatch 已经传 INT8 code，逻辑 record 是 H+32，其中 H=5120；不能以 BF16 激活作为基线夸大压缩空间。候选主布局明确计入 H bytes code、32B header（保留 FP32 scale raw bits、source row、epoch、fanout count 等）及 64B 的最多六个 8B 描述符区域，每条描述符为 local expert 与原 expanded-row ordinal。固定 stride 为 5216B，这是一种声明布局，不是最小 ABI，也没有宣称现有 padding 可以直接复用。

还保留原来的每专家 assignment counts A，并单独发布 unique-packet counts D。选定计数方案为每 rank 16 个 uint32、每 cohort 全矩阵 1024B；若全部其他 15 个参与者各接收一份，每 cohort 的逻辑远端内容为 15360B。这个量不是物理链路字节，也不强制新增独立 collective；若复用现有协议空间，必须证明新 ABI、所有读者、地址/offset 与发布完成条件。

在所有拟合并组的原量化 code 与 scale 均可按要求共用、上述布局/计数协议合法的条件下，target 样本的 dispatch 逻辑内容从 13,215,931,008B 变为 11,501,585,568B，减少 1,714,345,440B，即 12.9718%。BF16 返回仍按 A 发生，参考量 26,267,688,960B 保持；dispatch 加该返回参考量的减少比例为 4.3419%。这两个比例都不是 MC2 或请求耗时收益，也不是全模型可达加速比。

同一主布局对 draft 反而使逻辑内容从 3,766,112B 增至 3,899,552B，增加 133,440B，即 3.5432%。因此不能直接把 target 判断套给 draft。也不能因此否定所有 draft 布局：本轮另列 K 专用布局和 4B packed descriptor、外部 frame epoch 的紧凑布局，连同两种计数计划共 12 个角色/布局/协议模型。紧凑 draft stride 可为 5152B；仅在新增计数内容为零且新协议成立时，它有 20,608B 的微小正节省，加入选定 replicated count 后则变为负值。整数位宽、旧行 ordinal 范围、外部 epoch 和同调用边界都是显式条件。

分组 D 是结构性复用，不等于量化字节一致性已证。归档没有保存 code/scale，不能从“同 token、同目的地”单独推出原 assignment 内容相同。冻结源码中的 full-load CopyOutXQuant1H 已在分区内对一行量化后复制到多个 expanded positions；因此新方案不能再领一份 K 倍量化计算减少。其它 key、分区边界、加载实现及 scale 位模式仍需证明。候选可以采用 producer 合法性证明，也可采用 byte-certified grouping、比较失败逐条回退；后者的有效 packet 数可介于 D 与 A，比较、读取、计数、packing 和分组本身都不是免费。

保持语义的局部候选图已登记 13 个节点：实际 postmask IDs、原量化与 A counts、token/destination grouping、精确原行描述符、字节证明或检查、新 packet stage、D counts、peer pull、原行/scale 展开及原 GMM1/SwiGLU、GMM2/return、weighted unpermute，另保留概率输入边。A counts 与原 inverse/probs 直接连到相应原消费者。描述符必须恢复原专家/源 rank/行顺序和 capacity 前缀，不能由 packet 到达顺序替代；router score TopK 和 MC2 专家分组也不是可直接消除的同一个 sort。

接收端可以选择恢复现有 dense rows，或另行选择间接 GMM iterator/L1 gather 的候选；后者不等于当前 CATLASS 已自动支持，不能同时领两种路径的收益。专家计算和返回是权重相关的 assignment 结果，不能把 v119 的行工作、GMM FLOPs、权重读取或返回量中的 A 一律替换为 D。所有旧 round、scale/probs raw bits、alias、副作用、全 rank 参与、capture bank、remote pull 完成与旧读者/queued writer 退役都保留。新压缩 stage 与旧量化 source bank 可能重叠，峰值存储可能增加；只能按实际 packet/所需 tile readiness 分析，不能假设整层串行屏障。

理论对照独立标为 SYMBOLIC_CEILING：只有证明选定域的实际必要物理工作 W_r 与容量上界 C_r 后，才可用 T>=max_r(W_r/C_r)。本轮逻辑 record bytes 没有直接代入物理网络/HBM 必要量，新增分组/比较/packing/计数/展开也尚无同调用内部时延；数值时间下界、吞吐上界、实测/上界比均为 null。没有从这些载荷比例推导硬件利用率或最低利用率模块。

6400 个独立小规模 descriptor/旧序/capacity 截断参考用例通过，前置 7071 项核对通过。现有 REL73-07 与 HG73-MOE 补入证据，没有新增根候选、中央 pair、联合组或 observed 图节点。台账保持 35 项模型候选、7 项历史框架记录、356 条语义记录/26 模块、135 条条件关系/26 联合组/460 待核对 pair；2073 类/285598 次条件审阅、9 类/1352 次待补原生字段、849 缺 PMU、16 phase 未绑定图和 3136 含糊归属不变，整模块关闭仍为 0。源服务器只读任务完成、SSH 已关闭；整体目标继续，不能因本报告成稿而标记全部优化机会已经分析完。
'''
    action=(
        'Continue original Goal;v122 PROGRESS from newly recovered saved per-token MoE fanout,not native work. '
        'Read route-fanout summary/source-contract/packet-and-return/variants/local-graph/theory/current-obligations-v122 first;verify_route_fanout_v122 must PASS. '
        'v121 predecessor PASS647/manifest58;v122 preflight7071 plus6400 descriptor/clip refs. '
        '1392 original archived rank/layer snapshots in87 formal EP16 cohorts,2784 SHA checks,35632142B CPU-only read,SSH closed;no Torch/NPU/model/replay. '
        'Recorded linear/no-redundant/group0..15/C65536/no count clipping domain. Target A2565204 D2204823 repeated360381;draft A731 D727 repeated4. '
        'Selected H5120 INT8 baseline stride5152;new5216 +replicated D counts15360B/cohort saves target1714345440B/12.9718% dispatch or4.3419% dispatch+unchangedBF16return;draft grows133440B/3.5432%. '
        'Not full-run/wire/HBM/timing/speedup.12 layouts/count branches retain compact draft smallpositive20608B only with legal zero-incremental count content;not universal draft rejection. '
        'Archived IDs lack quantized code/scale;D structural only. Need loaded producer byteidentity or certified-check/fallback D_eff in[D,A];full-load baseline already shares row quantization. '
        'Retain A expert counts,old expanded/inverse/probs/clipping/rounding,GMM+return+unpermute. New grouping/packing/D-count/readiness/reexpansion or indirect-GMM costs and old+new storage/remote-reader/capture epochs remain. '
        '13-node local conditional candidate DAG preserves direct A/inverse/probs edges;not new global observed graph. Existing REL73-07/HG73-MOE refined only;135/26/460 and356/26 unchanged. '
        'SYMBOLIC physical ceiling;numeric T/MFU/lowest null. Original9/1352 pending849PMU16phase3136ambiguity and0whole unchanged. '
        'MoE observer has no Engram lookupIDs;hitrate remains unknown. v121 rowcache/C2 budget and v119 V1 peer counts are prior answers,not new unread code. '
        'v120 Pad grant GRANTED_AND_CONSUMED;no pending reply/equivalent retry/full-model capture permission. All jobs finished;do not edit sealedv122 or rerun finalizer. '
        'Proceed with genuine remaining operator branches/cross-candidate relations or call-bound evidence;Task ACTIVE,Loop004 EVALUATING,Goal ACTIVE.')
    inputs=[Path(r['path']) for r in read(O/names[0])['inputs']]
    scripts=[N/'analyze_route_fanout_v122.py',N/'verify_route_fanout_v122.py',
        N/'saved_tensor_archive_v122.py',N/'derive_route_fanout_saved_v122.py',
        N/'read_route_sample166_v122.py',N/'read_route_fanout166_v122.py',N/'persist_profile_refinement_v90.py']
    persist_refinement(version=122,previous=121,title='Saved MoE token-destination fanout and metadata-complete transport comparison',
        summary=dict(filename='route-fanout-summary-v122.json',
            previous_turn_classification='PROGRESS:v121 explicit F24/F34 same-device budget relation;PASS647',
            **detail,refined_candidates=2,refined_modules=3,
            branch_records=356,semantic_modules=26,relation_clauses=135,higher_order_groups=26,
            pending_central_pair_reconciliation=460,partial_graph_node_rows=813,
            partial_graph_edge_rows=1297,structural_pair_hints=46,
            local_candidate_graph_nodes=13,scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,
            pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
            new_archive_tensor_data_read=True,new_target_source_files_requested=False,
            implementation=False,native_test=False,physical_theory_state='SYMBOLIC_CEILING',
            actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,
            Pad_authorization='GRANTED_AND_CONSUMED',
            remote_job_status='CPU-only saved archive read completed;SHA/descriptor/layout checks finished;SSH closed. No Torch/NPU/model/replay/target implementation;completed Pad probe not rerun.'),
        changes=[
            'Recover exact recorded active token/destination fanout from1392 archived tensors and metadata;conserve original histograms/counts/epochs and preserve declared placement/capacity scope.',
            'Compare metadata-complete packet and count plans with unchanged per-assignment BF16 return:target selected logical dispatch reduction12.97%,draft selected increase3.54%;retain12 layout/count-plan branches.',
            'Link13-node conditional transport variant to existing F20/F21 relation and HG73-MOE;retain byteidentity,old maps/rounding/A counts,GMM/return,packing/readiness/lifetime and symbolic physical-capacity gaps.'],
        candidate_updates={c:dict(review_v122=deepcopy(detail),conclusion_changing_gap=gap) for c in ['F20','F21']},
        module_updates={
            'M12':dict(candidates=['F20','F21'],
                finding='Archived postmask token/destination fanout now bounds a concrete transport-repacking candidate. Keep original quantizer/A counts/maps/GMM/return and price all new packet/count/expand work;byte identity is unproved.',evidence=names,gap=gap),
            'M23':dict(candidates=['F20','F21'],
                finding='Selected linear EP16 archived domain has exact A/D counts;metadata-complete logical transport comparison depends on layout/count protocol and differs for target/draft. No physical topology/wire/latency binding.',evidence=names,gap=gap),
            'M24':dict(candidates=['F21'],
                finding='Token/destination packet reuse is scoped to one source row and call epoch;request IDs/observer ordinal cannot substitute. Retain old source bank plus packet staging until all remote pulls/queued writers retire.',evidence=names,gap=gap)},
        remaining=remaining,action=action,
        jobs=[dict(kind='profile',status='pass',
            description='CPU-only read of1392 archived tensors and1392 metadata files;SHA/epoch/histogram/count/linear-placement checks recover per-token EP fanout,no new device work.',
            artifacts=[O/'route-fanout-saved-records-v122.json',O/'route-snapshot-source-index-v122.json',O/'route-fanout-cohort-evidence-v122.json']+inputs),
            dict(kind='design-check',status='pass',
            description='Twelve packet/count-plan models,6400 independent descriptor/old-order/capacity references and7071 preflight checks;existing relation/group refined,physical ceiling and all whole scope remain open.',
            artifacts=[O/n for n in names]+[O/'route-fanout-preflight-v122.json',O/'route-fanout-validation-history-v122.json',O/'coverage-current-obligations-v122.json',O/'whole-scope-requirement-audit-v122.json'])],
        report=report,read_first=names+['coverage-current-obligations-v122.json'],
        publication_rows={},gate_updates=dict(overall_complete=False),scripts=scripts,
        executor_prefix='route-fanout',main_script=Path(__file__),source_read=False)


if __name__=='__main__':
    main()
