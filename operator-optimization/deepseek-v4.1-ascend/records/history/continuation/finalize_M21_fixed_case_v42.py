"""Persist v42 fixed configuration, conditional PCP/eager-plan opportunities and open gaps."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json

from finalize_origin_coverage_v9 import N, O, R, read, write
from finalize_Engram_whole_frontier_v30 import packet
from persist_profile_refinement_v3 import persist_refinement


def main():
    assert not (O/'REVIEW-CHECKPOINT-v42.json').exists()
    for a in read(O/'checkpoint-manifest-v41.json')['artifacts']:
        from pathlib import Path
        b = Path(a['path']).read_bytes()
        assert len(b) == a['bytes'] and hashlib.sha256(b).hexdigest() == a['sha256'], a['path']
    cfg = packet('M21-config166-v42', True)
    assert read(N/'M21-predicates166-v42.json')['returncode'] == 255
    sem = read(O/'M21-semantic-branch-inventory-v42.json')
    audit = read(O/'M21-original-requirement-completion-audit-v42.json')
    guards = read(O/'M21-adjacent-guard-crosswalk-v42.json')
    variants_file = 'M21-fixed-case-conditional-variants-v42.json'
    variants = read(O/variants_file)
    assert len(sem['branches']) == 18 and len(sem['static_branch_sites']) == 268
    assert len(guards['guard_dataflow_reconciled']) == 27
    assert len(guards['remaining_specific_guard_crosswalk']) == 36
    assert len(audit['requirements']) == 8 and not any(r['completion_proved'] for r in audit['requirements'])
    assert len(variants['variants']) == 4 and len(variants['conditions']) == 20
    assert read(O/'F32-PCP-row-projection-and-restore-proof-v42.json')['finite_integer_examples'] == 682
    evidence = [
        'M21-captured-config-and-selected-dispatch-contract-v42.json',
        'M21-PCP-alternative-case-and-consumer-contract-v42.json',
        'F32-PCP-row-projection-and-restore-proof-v42.json',
        'F31-eager-empty-query-plan-write-review-v42.json',
        'M21-adjacent-guard-crosswalk-v42.json', variants_file,
        'M21-fixed-case-conditional-graphs-v42.json',
        'M21-semantic-branch-inventory-v42.json',
        'M21-original-requirement-completion-audit-v42.json',
    ]
    common = dict(configuration=str(O/evidence[0]), pcp=str(O/evidence[1]),
                  conditions=str(O/variants_file), guard_crosswalk=str(O/evidence[4]),
                  implementation=False, new_performance_result=False)
    candidate_updates = {
        'F31': dict(review_v42=dict(common, relation='Constructor/group and selected main/draft configuration separated; PCP coordinate/RoPE and eager no-query SMLA value-write refinements recorded.'),
            variants_v42=[v for v in variants['variants'] if v['parent']=='F31'],
            profile_review_state='FIXED_CONFIG_GROUPS_AND_27_NEIGHBOR_GUARDS_RECONCILED;CONDITIONAL_NOQ_PLAN_AND_PCP_PRODUCER',
            conclusion_changing_gap='Exact enable_pcp/decode-shard/packed predicates unread after auth failure; 36 neighbor guards and actual main graph/QLI bank/all-reader/process/resource ranges remain.',
            next_profile_or_contract_action='Use existing neighbor evidence to reconcile 36 named guards, then actual M22 graph/bank/phase audit. Keep current draft outer eager distinct from main capture.'),
        'F32': dict(review_v42=dict(common, relation='PCP project-before-gather/restoration row identity derived; SWA KV and compressed-source/multistream consumers must all be included or original hidden gather remains.'),
            variants_v42=[v for v in variants['variants'] if v['parent']=='F32'],
            profile_review_state='CONDITIONAL_PCP_ROW_PROJECTION_RESTORE_IDENTITY;ALL_GLOBAL_HIDDEN_CONSUMERS_REQUIRED',
            conclusion_changing_gap='Actual weights/quant layouts, all consumers, physical padded row/restore map and native GEMM precision/readiness/cost remain conditions; no measured PCP benefit.'),
        'F16': dict(review_v42=dict(common, relation='PCP full/shard output-weight and activation scheduling alternative includes warmup alias, pending handles and finally switch-back. Source does not establish per-iteration weight transfer.'),
            variants_v42=[v for v in variants['variants'] if v['parent']=='F16']),
        'F30': dict(review_v42=dict(common, relation='dspark algorithm selector and dynamic_spec_config adaptive-budget method are separate. Default None is not a live object snapshot; anchor default and 20/24 constructor capacities remain separate from positive repeat-row counts.')),
    }
    fw_updates = {
        'H06': dict(review_v42=dict(common, relation='Eager no-query SMLA plan zero-write elision overlaps F31 and earlier initializer/dirty-range alternatives; preserve task/frontier and do not add duplicate gains.')),
        'H02': dict(review_v42=dict(common, relation='Source can bind draft outer wrapper but cannot fill saved producer CPU/transport timing. No new health/inference was performed.')),
    }
    module_updates = {
        'M21': dict(finding='Fixed saved config/group constructors separated from normalized live predicates; 27 of63 S13 guards reconciled,36 explicitly named. PCP alternative interfaces/cases analyzed; 4 conditional refinements/20 constraints; original18 families/268 static sites and8 open gates preserved.',
            evidence=evidence, candidates=['F31','F32','F16','F30'],
            gap='Actual graph/QLI/parameter-bank/phase/all-readers,36 neighbor guards, exact dispatch predicates, full-process transport and same-denominator bounds remain real gaps.',
            scope_audit=str(O/evidence[8]), profile_scope=str(O/'M21-original-profile-gap-reconciliation-v41.json')),
        'M19': dict(finding='Current draft exact direct CP input caller and no-query global-cache/output work reused. PCP alternative project-before-gather must include SWA KV, not only compressed KV/score.',
            evidence=evidence, candidates=['F32','F16','F31'],
            gap='No new actual PCP degree/row-owner/alias/native numerical/cost observations; source conditional rewrite is not a measured execution.'),
        'M20': dict(finding='Current dspark5 algorithm and eager draft settings are saved facts; separate adaptive-budget method default None, cached config and anchor alternatives explicitly retained.',
            evidence=evidence, candidates=['F30'], gap='Actual dynamic_spec method/HF anchor and full phase/consumer coverage remain unbound where decisive.'),
        'M22': dict(finding='Draft outer ACLGraph wrapper excluded by constructor plus saved draft enforce_eager; effective main sizes[8,16,24]/warmups1 kept distinct. Conditional eager no-query plan write does not remove publication events.',
            evidence=evidence, candidates=['F31'], gap='Actual main/nested graph/QLI body/parameter bank/replay ownership/all observers and16 phase-unbound graphs still require audit.'),
        'M23': dict(finding='PCP alternative global/local decision, restore selector, local-query/global-cache separation and weight handle/switch epochs analyzed as conditional cross-card alternatives.',
            evidence=evidence, candidates=['F32','F16'], gap='Exact PCP/decode selector definition, topology/group/weight and actual carrier/readiness/cost are not inferred from interface or config alone.'),
    }
    remaining = deepcopy(read(O/'module-coverage-gaps-v41.json')['remaining'])
    retired = [s for s in remaining if s.startswith(('M21v41','NextM21'))]
    write(O/'M21-remaining-history-before-v42.json', dict(previous_version=41, retired_progress_paragraphs=retired,
          reason='v42 replaces fixed-dispatch/PCP/neighbor progress with precise resolved and unresolved domains; prior full proof remains archived.', overall_complete=False))
    remaining = [s for s in remaining if s not in retired]
    remaining[16:16] = [
        'M21v42: saved DP2/TP8/mainDCP1 and main warmups1/capture[8,16,24] separated from eager dspark5 draft. Constructor/group logical blocksize/capacity contracts and PCP global/local/empty-query/RoPE/weight-state interfaces analyzed.27/63 S13 guards reconciled;36 named in crosswalk.18semantic families/268static sites/8original gates remain partial.4conditional F31/F32/F16 variants20constraints; no new gain or resource bucket.',
        'NextM21: reconcile36 specific S13 guard records using existing M07-M11/M19 source proofs; then M22 actual graph/QLI body, parameter banks/input-ready/last-reader/reuse and16 phase-unbound graphs, S16 same-denominator critical bounds, S18 full EngineCore/transport with M23-M25/global26module audit. Exact enable_pcp/decode-shard/normalized helper/packed device predicates unread because166 authentication failed; no retry without credential update. Observed draft direct CP is not integer PCP degree for main/all phases. Conditional implementation/benchmark absence alone is not an analysis gap.',
    ]
    action = (
        '继续Loop004，v42是固定配置/PCP替代接口/空query plan分析进展，不是整体完成。先读M21-fixed-case-analysis-summary-v42.json、M21-captured-config-and-selected-dispatch-contract-v42.json、M21-PCP-alternative-case-and-consumer-contract-v42.json、M21-adjacent-guard-crosswalk-v42.json和M21-original-requirement-completion-audit-v42.json。'
        '保持profile优先，仅不能定案再定点代码；禁止整库/整SDK扫描、实现或部署、新torch/model/NPU/service/health工作。35模型7框架0整模块，26理论INCOMPLETE/3136资源/849缺PMU/16未绑定图不变。'
        '当前main有效DP2/TP8/PP1/DCP1、warmups1/capture[8,16,24]/copyInputsFalse；parser早期0/[1,2,4,8,16,24]不是有效配置。dspark5 draft nested enforce_eagerTrue且ctor use_cuda_graphFalse，outer ACLGraph不选；不要当main或nested编译body无图。'
        'spec.methoddspark不是dynamic_spec_config.method，后者默认None但无live object/cached mutation快照；sample_from_anchor默认True但HF字段未绑定，Q5/6与maxbank20/24并列保留。'
        'PCP工厂与legacyDSACP互斥，观察到draft完整callerCP192→base701且无PCPforward490，支持此观察域directCP，不测得main/draft PCP整数。Exactenable_pcp/common_cp decode-shard/packed device helper在166rc255鉴权失败没有结果，已问凭据是否变化，勿自动重试。此前cfg保存资料CPU读取已成功。继续本地可用证据，不设blocked。'
        'PCP反事实小方法所有接口/分支已审：globalPrefill OR symbolicDecodeShard决定global owner；emptyQ保留global cache和通信，replicated decode无gather；independent localQ/globalKV RoPE banks既有dedup不是新收益；C2global/localowner和dummy skip不可混；Oproj warmup aliases/handles/wait/finally switch回不能假定每轮allgather。'
        'F32 row-affine projection与concat/index_select可交换(682整数示例，仅数学identity)；必须包括SWA KV+compressed-source+multistream所有globalHidden消费者，否则原hidden gather仍活。C2pool/history/ring更新留在restored rowspace；量化/tiling/native精度/cost未证。'
        'F31 exclusive eager noQ可条件省SMLA value zero_，保留task/frontier/inputReady/reuse/release、global cache/output AllToAll/ReduceScatter；next positive callback全bank copy后才读，所有alias/debug/version/captured reader需隔离。19draft zero与19noQ pattern不按数量配对；QLI/main logical0graph不借此结论。'
        'v42 27/63具体S13guard已dataflow对照，36余项逐条列名，不再把全部63当未知。保18semanticfamilies/268导航site不造新branch。下一步用既有M07-M11/M19证据对36guard，而后M22实际80graph/QLIbody/parambank/firstlastreader/producer modes/16unboundphase，S16资源同分母关键bounds，S18/M03/M25全进程回传，再M23-M25/global26module审计。'
        '2remote jobs=1cfgPASS/1authERROR；local missingfreeze、longloader reader guard和finalizer字段名precheck各1ERROR均保留，定点source/proof/formalaudit3PASS，共8runs4PASS4ERROR。所有作业terminal无server后台工作，lasthealth仍v12post。使用persist_profile_refinement_v3，勿修改旧helper以免破坏manifest。Goal active因真实覆盖缺口，非仅条件变体未实现。'
    )
    report = '''本轮继续采用 profile 定位 → 明确未定问题 → 定点代码补证。没有修改模型、启动推理、导入 torch、检查当前服务器健康或改变服务。

保存配置的事实已经分清：DP2、TP8、EP，主模型有效日志为 PP1、DCP1、warmup1、capture sizes[8,16,24]、copy_inputs=False；早期 parser 的 warmup0、[1,2,4,8,16,24] 不代替有效配置。dspark 草稿为5步、嵌套 enforce_eager=True，构造器另将 use_cuda_graph=False，因此草稿外层 ACLGraphWrapper 不选；主模型和嵌套编译体需要各自取证。

算法开关 method=dspark 与自适应预算 dynamic_spec_config.method 分开。后者源码默认 None；缓存配置和 live object 未快照，不能仅凭默认值宣称实际关闭。sample_from_anchor 的源码默认也不是实际 HF 字段，query bank20/24两种构造域保留。metadata 与 draft group 的 cache kind、逻辑 blocksize、backend/spec 分组、每组独立 bank 和 dtype/capacity 已分析。

PCP 作为跨卡替代方案分析了全局/局部 metadata、空 query、恢复 row map、RoPE、C2 状态所有权与输出权重切换。当前 v35 完整草稿调用栈是直接 CP 输入→base forward，支持该观察域使用 legacy CP；它不测得主模型或所有阶段的 PCP 整数。工厂里的 target/draft 注释也不是本次配置证明。enable_pcp、decode-sharding 等精确谓词读取遇到堡垒机鉴权失败，没有产物，仍明确保留缺口。

4项条件细化与20条约束写入既有候选，没有新造候选计数：
1. F32：将所有必要的逐行投影移到 PCP gather/restore 前。逐行仿射映射与 concat/index_select 可交换，682个含重复行/物理 PAD 的整数示例核对通过；这只证明行映射恒等式。SWA KV 也是全局 hidden 消费者，仅搬 compressed KV/score 不能消除原 gather；C2 相邻池化、历史和 ring 更新须保留正确恢复顺序。native GEMM/量化舍入没有因此得到证明。
2. F16：比较 PCP 输出投影的权重/activation 调度与分片重组，保留 warmup aliases、pending handles、等待与 finally switch-back。不把可能缓存的权重 gather 当作每轮必发通信。
3. F31：坐标与 RoPE table 索引 producer 可研究直接写入各自固定 bank；已有 local/global 分离和表指针 dedup 属于基线，不能重复计收益。
4. F31：在所有读者绕过且 bank 专有的 eager 空 query 路径，可条件省去 SMLA plan 的 value zero_。保留 task/frontier、input-ready、reuse/release 和全局 cache/output 通信；下一次正 query 前必须完整覆盖 bank。19次 draft zero callback 与19个 no-query pattern没有按计数配对；QLI 和主模型 logical-zero graph 不借用此结论。

原 S13 的63个静态 guard 中，27个具体位置已与 metadata 选择、cache update、preprocess、attention、forward 的数据流对照；36个余项按原 ID/owner/condition 明确列出，下一步复用 M07-M11/M19 证据核对。18个稳定语义范围、268个导航 site 和8项原始 mandatory gate 保留，8项仍未证明完整关闭。实际主图/QLI body、参数 bank、所有读者与跨代复用、16个未绑定图、同分母资源上界、完整进程/回传仍未分析完。

本轮8项登记为4 pass、4 error：远端保存配置读取成功；远端谓词读取鉴权失败；本地缺冻结文件、长 loader 读取守卫和持久化前字段名校验三次错误保留，字段名修正发生在共享状态更新前；定点源码、数学/guard 分析与持久化审计通过。失败没有被标记为成功，也没有自动重试登录。全部作业已终止，35模型候选、7框架候选、0整模块关闭，26理论比较 INCOMPLETE、3136资源行、849缺 PMU、16未绑定图保持。Goal active，未因报告或条件变体成稿标记完成。
'''
    gate = read(O/'PROFILE-CODE-QUESTION-GATE-v42.json')
    gate['allowed_files'] += [
        'Same-SHA DSAImpl._get_layer_metadata/_update_caches/_preprocess/_forward_attention/forward for exact S13/no-query reader question',
        'Same-SHA legacy SMLA producer1161-1207 only; compare zero_ branch with complete positive copy_ destination publication',
        'Same-SHA AscendConfig dynamic_spec/default loader selected statements; base draft runnable wrapper guard only',
    ]
    gate['additional_profile_led_questions'] = [
        'Does eager no-query execution bypass all SMLA plan value readers while retaining cache/output protocol?',
        'Which global hidden consumers must move with a PCP project-before-communication alternative?',
    ]
    write(O/'PROFILE-CODE-QUESTION-GATE-v42.json', gate)
    jobs = [
        dict(kind='profile', status='pass', description='Saved capture script/log/env whitelist SHA/stat query on166: main/draft parallel/spec/eager and effective graph settings separated.', artifacts=[N/'query_M21_fixed_configuration_v42_166.py',N/'M21-config166-v42.json']),
        dict(kind='review', status='error', description='Local exact parallel predicate read stopped because attention/utils.py absent from frozen snapshot/zip; no result artifact emitted.', artifacts=[O/'M21-local-predicate-freeze-gap-v42.json']),
        dict(kind='review', status='error', description='Local loader reader span guard rejected >100line method; replaced with exact selected assignments, preserving initial error.', artifacts=[O/'M21-local-config-loader-span-guard-v42.json',O/'M21-config-default-loader-source-v42.json']),
        dict(kind='review', status='error', description='Prepared remote exact predicate job failed SSH authentication rc255; no predicate result/source execution, no automatic retry.', artifacts=[N/'read_M21_dispatch_predicates_v42_166.py',N/'M21-predicates166-v42.json',O/'M21-predicate-remote-authentication-failure-v42.json']),
        dict(kind='review', status='error', description='Finalizer precheck used incorrect integer example field name; KeyError occurred before shared state mutation. Corrected to finite_integer_examples.', artifacts=[O/'M21-finalizer-field-precheck-error-v42.json']),
        dict(kind='review', status='pass', description='Same-SHA frozen constructor/group/factory/PCP small-method review; main/draft modes, adaptive selector/default domain and all global-hidden consumers separated.', artifacts=[O/'M21-selected-constructor-group-and-PCP-alternative-source-v42.json',O/'M21-frozen-AscendConfig-dispatch-fields-v42.json',O/'M21-captured-config-and-selected-dispatch-contract-v42.json',O/'M21-PCP-alternative-case-and-consumer-contract-v42.json']),
        dict(kind='review', status='pass', description='682 pure integer row-affine examples plus pooling counterexample, conditional no-query reader proof,27exactguard crosswalk and4variants20constraints; no native precision/performance claim.', artifacts=[N/'build_M21_fixed_case_analysis_v42.py']+[O/p for p in evidence[2:7]]),
        dict(kind='review', status='pass', description='Original8gate/18semanticfamily inventory retained with explicit gaps, previous57file manifest verified and candidate/module/framework persistence audit.', artifacts=[O/'M21-semantic-branch-inventory-v42.json',O/'M21-original-requirement-completion-audit-v42.json',N/'persist_profile_refinement_v3.py']),
    ]
    summary = dict(filename='M21-fixed-case-analysis-summary-v42.json', previous_turn_classification='PROGRESS:v41originalscopeaudit',
        original_static_sites=268, semantic_branch_families=18, original_mandatory_gates=8, completion_audit='NOT_COMPLETE',
        neighbor_S13_sites=63, reconciled_neighbor_guards=27, remaining_neighbor_guards=36,
        expanded_native_chains_unchanged=209, expanded_native_TASKs_unchanged=84297,
        conditional_variants=4, condition_rows=20, integer_row_examples=682,
        refined_model_candidates=['F31','F32','F16'], related_model_candidates=['F30'], new_model_candidates=[],
        refined_framework_candidates=['H06','H02'], remote_jobs=2, remote_pass=1, remote_retained_errors=1,
        retained_local_errors=3, authentication_question_pending=True,
        remote_job_status='All jobs terminal: saved-config remotePASS, exact-predicate remoteAUTH_ERROR(rc255); 2local reader errors and1finalizer precheck schema error preserved/corrected before shared mutations. Frozen point/source/proof/formal reviewPASS. No new model/torch/NPU/service/health work; no live jobs.',
        evidence_refs=[dict(receipt=str(N/'M21-config166-v42.json'), full_result=cfg['full_result'], sha256=cfg['result_sha256'])],
        no_new_model_NPU_service_or_health_work=True)
    persist_refinement(version=42, previous=41, title='v42 M21 fixed-case dispatch and PCP/eager-plan conditional analysis',
        summary=summary, changes=[
            'Pinned saved main DP2/TP8/DCP1/effective graph settings separated from eager dspark5 draft; constructors/groups/default/live predicates distinguished.',
            'PCP global/local/cache/restore/RoPE/weight-state alternatives analyzed;4conditional F31/F32/F16 refinements20constraints,682 row identity examples only.',
            '27/63 neighbor guards reconciled;36 named guards plus graph/bank/phase/resource/process gaps remain. Exact helper remote read failed authentication and is not claimed complete.'],
        candidate_updates=candidate_updates, framework_updates=fw_updates, module_updates=module_updates,
        semantic_branches={'M21':sem['branches']}, source_read=True,
        remaining=remaining, action=action, jobs=jobs, report=report,
        read_first=['M21-captured-config-and-selected-dispatch-contract-v42.json','M21-PCP-alternative-case-and-consumer-contract-v42.json','M21-adjacent-guard-crosswalk-v42.json','M21-original-requirement-completion-audit-v42.json'],
        publication_rows={
            'F31':'| F31 | Metadata、slot、draft共享与输入输出发布重组 | 条件 | v42分清有效main配置与eager draft；27/63邻接guard对照、36仍待核对。PCP坐标/RoPE producer与eager空query SMLA value零写省略已列条件，保持完整事件/cache/output协议；主图/QLI参数bank/所有读者与生命周期未闭合。 |',
            'F32':'| F32 | SP/PCP输入gather与逐行Q_a/KV投影重排 | 条件 | v42补PCP投影→gather/restore数学合同；必须覆盖SWA KV、compressed-source和multistream所有global hidden消费者，否则原gather仍需保留。682整数例仅验证行恒等式，native舍入/实际rowmap/拓扑/收益未证。 |',
            'F16':'| F16 | 两级输出投影与reduce-scatter重组 | 条件 | v42补PCP full/shard权重与activation调度域；保留warmup alias、handles、wait及finally切回，不能假定权重每轮all-gather。实际代际与通信成本待绑定。 |',
        },
        gate_updates=dict(status='FIXED_CONFIG_AND_PCP_INTERFACE_ANALYZED;EXACT_PREDICATES_AUTH_GAP', answered_by=evidence,
            closed_scope='Saved main/draft settings, bounded constructors/group contracts, PCP counterfactual interfaces and27 exact neighbor guard dataflows; conditional eager no-query value-write analysis.',
            not_closed='Exact helper definitions after auth failure,36 neighbor guards, main/nested graph/QLI/parameter bank/all readers, resource/process/global26module coverage.',
            stop_rule='Do not reopen answered local contracts; use existing neighbor proof and actual saved graph evidence before any more point code. No whole-module or goal completion.'),
        scripts=[N/'query_M21_fixed_configuration_v42_166.py',N/'read_M21_dispatch_predicates_v42_166.py',N/'build_M21_fixed_case_analysis_v42.py'],
        executor_prefix='zcode-M21-', main_script=N/'finalize_M21_fixed_case_v42.py')


if __name__=='__main__':
    main()
