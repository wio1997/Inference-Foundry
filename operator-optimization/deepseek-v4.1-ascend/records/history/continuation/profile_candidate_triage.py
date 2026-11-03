"""Build a profile-first intake ledger, preserving unknown invocation identity."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import hashlib
import statistics
import re

R = Path('/Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41')
N = Path(__file__).parent
OUT = N / 'profile-first-20261031'

# Each row specifies a profile question, not permission to scan the source tree.
QUESTIONS = {
 'F01': ('HcPre', '复合边界已出现；内部阶段和系数消费者没有由任务标签展开。', '在现有target/draft窗口分别检查HcPre输入输出与相邻调用，不把复合调用再计新增融合。'),
 'F02': ('HcPre MatMulV2', 'embedding行与首层四路mHC输入尚无同调用内容/身份绑定。', '先定位首层scope中的embedding、初始化、系数投影及norm记录，区分启动、prefill和decode。'),
 'F03': ('HcPre HcPost', '跨层残差版本、多消费者和生命周期没有由相邻任务证明。', '连接同rank相邻层post/pre窗口，查Engram、aux及tail交叉消费者的已有scope/flow。'),
 'F04': ('RmsNormCast', '已观察到norm/cast复合边界；尚未证明内部重新组织等价。', '对target/draft各自对应的RmsNormCast记录复核shape/dtype与后继，不重复计已有复合段。'),
 'F05': ('RmsNorm DynamicQuant QuantBatchMatmulV3', '同形量化记录尚不等于同一Q归一化值；indexer旁路消费者须保留。', '复用F05/F09已存host-scope和量化关联，逐调用区分Q_b与index query的消费。'),
 'F06': ('DynamicQuant MatMulV2 QuantBatchMatmulV3', '共享输入及CP行域未由family汇总绑定。', '先查Q/KV两投影scope的输入shape、stream和gather前后行域，核对已存双投影host-flow样本。'),
 'F07': ('QuantBatchMatmulV3 InplacePartialRotaryMul ScatterNdUpdateSk', '所选记录有交叠和事件标签，但缺event对象配对及Q/RoPE实际tensor身份。', '已读1920窗及9窗27事件记录；复用现有局部等待/别名合同，历史事件身份继续保留为运行时缺口。'),
 'F08': ('MatMulV2 RmsNorm InplacePartialRotaryMul ScatterNdUpdateSk', '1280个选定三任务scope不是tensor身份、实际有效槽或全部持久消费者证明。', '已读全部选定三任务和320个T4760交叠分类；复用局部cache合同，实际slot/alias属于运行时缺口。'),
 'F09': ('QuantBatchMatmulV3 InplacePartialRotaryMul DynamicQuant', 'query的行/head归约域及scale消费者尚未由汇总绑定。', '先核对已存shared-QR量化调用scope，定位投影→RoPE→量化的flow与head形状。'),
 'F10': ('MatMulV2 Cast Mul', 'FP16舍入和query-scale分叉不能由算子名或时长确定。', '先在已存indexer提交栈与selected scope中绑定weight投影、cast/scale和query scale消费者。'),
 'F11': ('RmsNorm InplacePartialRotaryMul DynamicQuant ScatterNdUpdateSk', '分页key/scale状态和实际stride、rounding消费者没有由时间相邻闭合。', '先核对key侧任务与cache写入的host-scope、shape和分支，复用既有F11绑定审计。'),
 'F12': ('VllmAscendQuantLightningIndexerV2', '两个提交栈可定位调用位置，不能证明输出ID→prepare输入实际tensor边或跨层行版本。', '已读CONNECTION_IDS和两个callchain；下一步复用prepare/发布的局部合同，真实长度和运行值保留缺口。'),
 'F13': ('SparseFlashMla', '复合attention任务已经存在，内部稀疏/SWA分支和所有可观察输出未由标签展开。', '先核对所有选定attention scope的shape、索引和cache角色，不重复计已有复合调用。'),
 'F14': ('SparseFlashMla InplacePartialRotaryMul Transpose TransposeBatchMatMul', 'attention后同形RoPE的角色、layout及投影消费者仍需逐调用关联。', '利用已有pre/post RoPE筛选，查post-attention候选到布局恢复和输出投影的scope/flow。'),
 'F15': ('TransposeBatchMatMul hcom_alltoall_ hcom_reduceScatter_', 'CP padding/恢复、有效行和通信所有权尚未由类型汇总证明。', '先关联已有CP投影scope、restore任务及collective记录，保留有效行与padding差别。'),
 'F16': ('TransposeBatchMatMul MatMulV2 hcom_reduceScatter_', '两级投影之间的数值边界与逐块collective可见性不由记录时长推出。', '先复核已存两投影host-flow-device样本、通信组/count/连接，再区分受限dense模型与通用融合域。'),
 'F17': ('MatMulV2 MatMulV3 Cast', 'Compressor双投影的实例归属、行stride和ring消费者尚未由GEMM同形证明。', '先按已有compressor scope绑定两投影、FP32路径、pool和ring更新，避免按全局MatMul计量。'),
 'F18': ('RmsNorm', '状态池化与norm的tensor版本、历史页和舍入语义没有被RmsNorm标签证明。', '先定位已存pool/norm的同调用scope及状态写入，查全部导出消费者和phase。'),
 'F19': ('MatMulV2 ScatterNdUpdateSk', 'latent两路分叉及跨层持久发布没有由family名称证明。', '先定位compressor/index-key/压缩KV的分支scope与写入，核对现有跨片段发布映射。'),
 'F20': ('PadV3 Equal Fill SelectV2 MoeGatingTopKHash DispatchFFNCombine', '已存320组pad/where/gate栈支持提交位置；ID内容、全消费者和历史binary身份仍缺。', '先复核320组pad/gate/MC2键和已有栈审计；已有整数映射合同可复用，运行值缺口不能靠再读源码补出。'),
 'F21': ('DispatchFFNCombine', '外部复合任务已存在；内部资源争用、ready flag与实际分阶段耗时未由一个区间展开。', '复用MC2到达、其他流事件及166同调用计数证据；各采集域分开，禁止旧profile时长与新计数同调用拼接。'),
 'F22': ('QuantBatchMatmulV3 DequantSwigluQuant', '激活量化段已是复合调用；gate/up半区的块就绪与scale发布没有由类型名证明。', '先核对shared expert投影→DequantSwigluQuant→down的同scope签名和既有消费者审计。'),
 'F23': ('ForeachAddList HcPost hcom_reduceScatter_', 'shared/routed归约域、scale和BF16边界没有由add或collective标签闭合。', '先关联merge、scaling/add和HC的实际通信分支scope，区分两分支归约和重复工作。'),
 'F24': ('', '四个精确Engram kernel只覆盖局部，不覆盖全部host/UVA生命周期和物理流量。', '先复用精确kernel→CANN连接、host查找与ready/event记录，查投影和后续消费者的已有scope。'),
 'F25': ('MatMulV2 MatMulV3', 'Engram gate/residual与全局GEMM同名不能建立模块归属或残差版本。', '先用Engram相关已存scope/flow绑定投影、gate与residual，不把全部同形MatMul归给Engram。'),
 'F26': ('RmsNorm HcPost', 'tail collapse/norm和MTP条件消费者、独立buffer生命周期未由decoder窗口覆盖。', '先查尾部及MTP的既有scope、copy和消费者记录，区分启动分配、写入与数值消费。'),
 'F27': ('MatMulV2 RmsNorm hcom_allGather_', '辅助上下文投影的token/output分片所有权不能由shape推断。', '先核对DSpark context投影、通信group和行域的已有scope，查off-diagonal信息来源。'),
 'F28': ('HcPre', 'draft embedding/初始化是否重复及状态消费者尚未逐调用绑定。', '先结合strict draft窗口和已有first-draft scope，定位embedding、HC init与decoder输入。'),
 'F29': ('MatMulV2 RmsNorm InplacePartialRotaryMul ScatterNdUpdateSk', 'context buffer的dtype、slot容量和跨层消费者不由target形状候选替代。', '先按draft/context独立scope核对投影、cache写入、metadata copy及空请求/padding记录。'),
 'F30': ('ArgMaxV2 MatMulV2 Cast', 'target/draft词表、bonus和拒绝分支未由全局ArgMax/GEMM名称区分。', '先绑定sampling/rejection与logits的已有CPU栈、词表shape和请求temperature；区分greedy与概率分支。'),
 'F31': ('Fill Cast', 'metadata全消费者、跨步alias及空行分支没有由small-op汇总证明。', '先关联已存prepare-input、slot/index/length和跨迭代发布scope，逐buffer区分host与device读者。'),
 'F32': ('hcom_allGather_ MatMulV2 QuantBatchMatmulV3', 'gather与Q/KV实例的所有权、CP/PCP恢复次序及多消费者未由shape证明。', '先复用8个host-flow-device投影样本和已有group元数据，检查gather前后有效行与消费者。'),
}

MODULE_LINKS = [
 [], [], [], [], [], ['F02','F28'], ['F01','F02','F03'], ['F05','F06','F07','F08','F09','F10','F11'],
 ['F17','F18','F19'], ['F09','F10','F11','F12'], ['F08','F13'], ['F14','F15','F16','F32'],
 ['F20','F21'], ['F22','F23'], ['F24','F25'], ['F26'], ['F30'], ['F30'], ['F27','F29'],
 ['F28'], ['F30'], ['F31'], ['F07','F31'], ['F06','F15','F16','F20','F21','F23','F27','F29','F32'],
 ['F03','F12','F19','F24','F26','F28','F29','F31'], [],
]


def load(rel):
    return json.loads((R / rel).read_text())


def stats(values):
    return {'n': len(values), 'min': min(values), 'median': statistics.median(values), 'max': max(values)}


def main():
    aggregate = json.loads((OUT / 'aggregate-recheck.json').read_text())
    q = load('agents/fusion/q-scatter-16rank-overlap.json')
    rope = load('agents/fusion/q-scatter-rope-16rank.json')
    key = lambda x: (x['rank'], x['run'], x['layer'])
    qmap, rmap = {key(x): x for x in q['samples']}, {key(x): x for x in rope['samples']}
    assert len(qmap) == len(rmap) == len(q['samples']) == len(rope['samples']) == 1920
    assert qmap.keys() == rmap.keys() and q['input_csv_sha256'] == rope['input_csv_sha256']
    slot = []
    for k, a in qmap.items():
        b = rmap[k]
        assert a['scatter_count'] == 1 and a['overlap_ns'] > 0
        assert b['q_end_to_rope_start_ns'] == a['scatter_tail_after_q_ns'] + b['scatter_end_to_rope_start_ns']
        slot.append(a['scatter_tail_after_q_ns'] - b['rope_duration_ns'])
    events = load('agents/fusion/f07-event-object-evidence.json')
    event_rows = [e for m in events['event_trace_matches'] for e in m['matched_trace_events']]
    assert len(event_rows) == 27 and all(m['match_count'] == 1 and not m['truncated'] for m in events['event_trace_matches'])
    triples = load('agents/fusion/f08-actual-triples.json')
    assert len(triples['records']) == triples['unique_scopes'] == 1280
    f08 = load('agents/fusion/f08-4760-rank-overlap.json')
    assert len(f08['records']) == 320
    pad = load('agents/diagnosis/int64-pad-gating-scope.json')
    mc2 = load('agents/diagnosis/pad-gap-mc2-rank.json')
    pkey = lambda x: (x['ep'], x['layer'], x['scope'])
    pmap, mmap = {pkey(x):x for x in pad['rows']}, {pkey(x):x for x in mc2['rows']}
    assert len(pmap) == len(mmap) == 320 and pmap.keys() == mmap.keys()
    indexer = load('agents/diagnosis/ep2-indexer-callchain.json')
    bound_indexer = [x for x in indexer['links'] if len(x['pytorch_api']) == 1]
    detail = {
        'F07': {'joined_windows': len(qmap), 'overlap_windows': len(qmap),
                'slot_minus_rope_ns': stats(slot), 'event_rows': len(event_rows),
                'event_id_or_id2_present': sum(e.get('id') is not None or e.get('id2') is not None for e in event_rows),
                'event_numeric_arg_keys': sorted(set(k for e in event_rows for k in e.get('numeric_identifier_args', {}))),
                'event_connections': sorted(set(e.get('numeric_identifier_args', {}).get('connection_id') for e in event_rows)),
                'decision': 'Exported overlap exists and simple additive timing is invalid. A geometrically large slot does not prove legal scheduling or releasable time.'},
        'F08': {'selected_three_task_scopes': triples['unique_scopes'], 'selected_row_counts': triples['rows_counts'],
                'T4760_scope_count': len(f08['records']), 'class_counts': dict(Counter(x['class'] for x in f08['records'])),
                'long_with_quant_matmul_overlap': sum(x['class'] == 'long' and 'aclnnQuantMatmulWeightNz_QuantBatchMatmulV3_QuantBatchMatmulV3' in x['other_track_overlap_names'] for x in f08['records']),
                'short_without_other_selected_track_overlap': sum(x['class'] == 'short' and not x['other_track_overlap_names'] for x in f08['records']),
                'decision': 'Same exported signature can coexist with very different durations. Overlap correlation does not prove contention or equal effective workload.'},
        'F12': {'launch_api_links': len(bound_indexer),
                'api_names': [x['pytorch_api'][0]['name'] for x in bound_indexer],
                'callchain_ids': [x['pytorch_api'][0]['callchainId'] for x in bound_indexer],
                'decision': 'Indexer and prepare submission stacks are bound to selected launches. This is not a tensor edge, Dequeue stack or whole-request causal path.'},
        'F20': {'unique_pad_gate_mc2_scope_keys': len(pmap), 'ep_count': len({k[0] for k in pmap}),
                'layers': len({k[1] for k in pmap}), 'pad_gate_reported_stats': {k: pad[k] for k in ['pad_duration_us','gate_duration_us','gap_us','pad_longer_count']},
                'stack_binding_status': '320/320 reported in retained review; underlying per-call stack result is not present in this local input set and has not been independently rebuilt here.',
                'decision': 'Selected record association is consistent; do not turn reported stack coverage into proof of historical tensor identity, ID values or fusion savings.'},
    }
    datasets = {'old_all_ranks': aggregate['old_capture']['tables']['operator_types']}
    datasets.update({'official_window_' + w: rows for w, rows in aggregate['official_capture']['tables_by_window'].items()})
    inventory = (R/'publication-draft/FUSION-INVENTORY.md').read_text()
    names = {m[0]: m[1].strip() for m in re.findall(r'^\| (F\d\d) \| ([^|]+) \|', inventory, re.M)}
    assert set(QUESTIONS) == set(names) == {f'F{i:02}' for i in range(1,33)}
    candidates = []
    for cid, (labels, gap, query) in QUESTIONS.items():
        selected = set(labels.split())
        hits = {dataset: [{k: row[k] for k in ['type','sample_count','all_sample_call_count']} for row in rows if row['type'] in selected]
                for dataset, rows in datasets.items()}
        checked = cid in detail
        state = 'SELECTED_PROFILE_SLICE_REVIEWED' if checked else 'FAMILY_INTAKE_DONE_SCOPE_STACK_REVIEW_PENDING'
        if cid in ['F01','F04','F13']:
            state = 'EXISTING_COMPOSITE_OBSERVED_FURTHER_BOUNDARIES_OPEN'
        candidates.append({'id': cid, 'name': names[cid], 'profile_review_state': state,
                           'family_name_hits_not_role_proof': hits, 'scoped_detail': detail.get(cid),
                           'conclusion_changing_gap': gap, 'next_profile_or_contract_action': query,
                           'target_source_gate': 'REUSE_EXISTING_LOCAL_CONTRACT_FOR_SPECIFIC_SEMANTIC_QUESTION' if checked else 'DEFER_UNTIL_RELEVANT_PROFILE_SCOPE_AND_STACK_REVIEWED',
                           'runtime_gap_policy': 'Code cannot recover historical values, storage identity or event generations absent from the capture.',
                           'legal_fusion_proved': False, 'candidate_closed': False,
                           'critical_path_gain_us': None, 'critical_path_gain_status': 'UNKNOWN_NOT_SUM_OF_TASK_TIMES'})
    modules = load('whole-scope-20261030/module-coverage-ledger.json')['surfaces']
    assert len(modules) == len(MODULE_LINKS) == 26
    module_rows = []
    for i, module in enumerate(modules):
        cid = f'M{i:02}'
        assert module['id'] == cid
        refs = MODULE_LINKS[i]
        evidence = ['candidate-profile-ledger.json'] if refs else []
        if i in [1,2,3,4,21,22,24,25]: evidence += ['publication-draft/framework-cpu.json', 'continuation-20261030/worker-validation.json', 'continuation-20261030/prepare-input-analysis.json', 'continuation-20261030/graph-executions.json', 'continuation-20261030/iteration-gaps.json']
        module_rows.append({'id':cid, 'name':module['name'], 'candidate_links':refs,
                            'profile_entry_points':evidence,
                            'review_state':'PARTIAL_CANDIDATE_LINKS; module/phase/branch closure not proved',
                            'no_candidate_links_meaning':'Framework/state opportunity domain remains open; not absence of optimization.' if not refs else None,
                            'branch_coverage':'NOT_CLOSED', 'cross_iteration_consumers':'NOT_CLOSED',
                            'whole_module_closed':False})
    inputs = ['agents/fusion/q-scatter-16rank-overlap.json','agents/fusion/q-scatter-rope-16rank.json',
              'agents/fusion/f07-event-object-evidence.json','agents/fusion/f08-actual-triples.json',
              'agents/fusion/f08-4760-rank-overlap.json','agents/diagnosis/int64-pad-gating-scope.json',
              'agents/diagnosis/pad-gap-mc2-rank.json','agents/diagnosis/ep2-indexer-callchain.json',
              'publication-draft/FUSION-INVENTORY.md','whole-scope-20261030/module-coverage-ledger.json']
    provenance = [{'path':str(R/p), 'sha256':hashlib.sha256((R/p).read_bytes()).hexdigest()} for p in inputs]
    common = {'generated_utc':datetime.now(timezone.utc).isoformat(), 'overall_complete':False,
              'new_source_read':False, 'new_device_work':False, 'whole_modules_closed':0,
              'scope':'Profile intake for all existing candidates/surfaces, selected slice review only. Not exhaustive candidate or branch closure.'}
    (OUT/'selected-slice-recheck.json').write_text(json.dumps({**common,'details':detail,'provenance':provenance},ensure_ascii=False,indent=2)+'\n')
    (OUT/'candidate-profile-ledger.json').write_text(json.dumps({**common,'candidates':candidates,'count':len(candidates)},ensure_ascii=False,indent=2)+'\n')
    (OUT/'module-profile-ledger.json').write_text(json.dumps({**common,'modules':module_rows,'count':len(module_rows)},ensure_ascii=False,indent=2)+'\n')
    lines = ['# 候选的 Profile 证据分级', '', '32个既有候选全部登记，26个暂定模块关联。只是完整登记和部分切片复核；未证明全范围分析完成。', '', '|候选|本轮profile状态|仍影响判断的缺口|下一步|', '|---|---|---|---|']
    for c in candidates: lines.append('|'+ '|'.join([c['id'],c['profile_review_state'],c['conclusion_changing_gap'],c['next_profile_or_contract_action']])+'|')
    lines += ['', 'F01/F04/F13已观察到复合边界；F21和F22也包含既有复合段，不计作尚未发生的独立融合收益。family命中仅是类型出现，不证明模块角色或同调用生产消费边。', '', 'F07/F08/F12/F20的本地切片结果见selected-slice-recheck.json。未加入新目标源码审阅或设备采集。历史事件/张量/有效值的缺口不能通过静态源码补成实测。', '', '其余候选保留scope/stack查询，只有相关profile查尽仍无法回答具体语义时才定点补源码。当前Loop继续，0整模块关闭。']
    (OUT/'CANDIDATE-PROFILE-TRIAGE.md').write_text('\n'.join(lines)+'\n')
    summary = {**common,'candidate_count':len(candidates),'module_surface_count':len(module_rows),
               'review_states':dict(Counter(c['profile_review_state'] for c in candidates)),
               'selected_slice_candidates':list(detail),'F07_joined_windows':len(qmap),
               'F08_selected_scopes':1280,'F20_unique_scope_keys':len(pmap),
               'next_action':'继续所有scope/stack未复核条目；F07/F08/F12/F20先复用已有局部合同，保留实际运行身份缺口。'}
    (OUT/'triage-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__ == '__main__':
    main()
