"""Persist M22 lifetime, producer placement and honest original-scope audit."""
from copy import deepcopy
from pathlib import Path
import hashlib

from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v3 import persist_refinement


def main():
    assert not(O/'REVIEW-CHECKPOINT-v44.json').exists()
    for a in read(O/'checkpoint-manifest-v43.json')['artifacts']:
        b=Path(a['path']).read_bytes();assert len(b)==a['bytes']and hashlib.sha256(b).hexdigest()==a['sha256'],a['path']
    sem=read(O/'M22-semantic-branch-inventory-v44.json')
    scope=read(O/'M22-original-scope-and-guard-reconciliation-v44.json')
    audit=read(O/'M22-original-requirement-completion-audit-v44.json')
    variantsfile='M22-graph-generation-conditional-variants-v44.json';variants=read(O/variantsfile)
    placement=read(O/'M22-named-metadata-graph-placement-derivation-v44.json')
    assert len(sem['branches'])==16 and len(sem['static_branch_sites'])==1101
    assert len(scope['guard_dataflow_reconciled'])==41 and len({r['id']for r in scope['guard_dataflow_reconciled']})==41
    assert len(audit['requirements'])==8 and not any(r['completion_proved']for r in audit['requirements'])
    assert len(variants['variants'])==3 and len(variants['conditions'])==17
    assert sum(r['DIRECT']for r in placement['derived_named_metadata_origins'].values())==529
    assert not any(r['GRAPH']for r in placement['derived_named_metadata_origins'].values())
    oldmatrix=read(O/'module-phase-branch-matrix-v43.json');old=next(r for r in oldmatrix['modules']if r['id']=='M22')
    assert len(old['branches'])==32 and old.get('branch_schema')!='SEMANTIC_BRANCHES_V1'
    evidence=[
        'M22-named-metadata-graph-placement-derivation-v44.json',
        'M22-capture-replay-input-output-and-update-contract-v44.json',
        'M22-positional-address-check-domain-proof-v44.json',
        'M22-capture-lifetime-point-source-v44.json',
        'M22-runner-update-and-enpu-binding-point-source-v44.json',
        'M22-original-scope-and-guard-reconciliation-v44.json',
        'M22-semantic-branch-inventory-v44.json',variantsfile,
        'M22-graph-generation-protocol-v44.json',
        'M22-original-requirement-completion-audit-v44.json',
    ]
    common=dict(placement=str(O/evidence[0]),capture_contract=str(O/evidence[1]),address_check=str(O/evidence[2]),
        conditions=str(O/variantsfile),original_scope_audit=str(O/evidence[9]),implementation=False,new_performance_result=False)
    candidates={
        'F31':dict(review_v44=dict(common,relation='321SMLA+208QLI named metadata TASKs derived DIRECT/0GRAPH from complete same-capture censuses and identity classifier. Three placement/generation/sync variants preserve capture scalars/kwargs banks/native parameter epochs and all readiness/last readers.'),
            variants_v44=variants['variants'],
            profile_review_state='NAMED_METADATA529_DIRECT_0GRAPH_DERIVED;41_GRAPH_LIFETIME_GUARDS_RECONCILED',
            conclusion_changing_gap='Named metadata placement is answered; actual captured plan/parameter bank identities, ENPU_ENABLE/backend/option eligibility, native update epochs/all reader generations,16phase-unbound graphs and original worker/registry/resource/process coverage remain.',
            next_profile_or_contract_action='Point-reconcile28remaining graph-file registry/weak-workspace helpers using current placement and fixed-bank questions, then actual saved graph/native/bank associations and neighboring M23-M25/global scopes. Do not repeat named529placement or M21S13.'),
        'F07':dict(review_v44=dict(common,relation='Rope/producer fusion crossing graph boundary must publish into captured kwargs/nested Tensor banks; wrapper positional debug check is not complete address evidence. CPU scalar capture fields and native parameter epochs remain separate.')),
        'F26':dict(review_v44=dict(common,relation='Captured outputs/workspaces use weak-reference protocol; first capture returns output, later replay returns stored output. Cross-boundary norm/head composition must retain all output readers before bank replay overwrite; weak refs are not new device copies.')),
        'F24':dict(review_v44=dict(common,relation='Runner prepares live Engram routing before replay or graph input banks during capture; existing history/DP/mask/generation contracts remain. Graph reuse does not make route/cache values reusable across accepted-history iterations.')),
    }
    fw={
        'H02':dict(review_v44=dict(common,relation='Generation-aware metadata producer placement/bank pipeline may refine broad waits only with correct GPU ready/last-reader and CPU native parameter-mutator host completion. ENPU_ENABLE differs from enable_npugraph_ex; no actual sync-cost saving inferred.')),
        'H06':dict(review_v44=dict(common,relation='Descriptor×bank graph instances carry pool retention/capture/weak-output memory cost and globally consistent rank decisions; finite bank pipeline is not free bucket or memory reuse.')),
        'H07':dict(review_v44=dict(common,relation='Ordinary runner passes model inputs through kwargs, so wrapper positional address list can be empty even in DEBUG. Embedding/input producer changes must preserve actual captured kwargs banks; source debug assertion alone cannot validate them.')),
    }
    modules={
        'M22':dict(finding='Original2files142units1101static navigation sites retained; old32chronological rows archived as history and replaced by16 stable semantic families.41core guard dataflows reconciled,28other graph-file helper sites need finite reconciliation. Named metadata321+208 derived529DIRECT/0GRAPH;3conditional variants17constraints, all8 originalgates stillnotcomplete.',
            evidence=evidence,candidates=['F31','F07','F26','F24'],
            gap='Actual bank/graph/native update epochs/all readers and16unbound phases; exact ENPU/backend/options where decisive;28registry/workspace helper sites plus adjacent worker scope; resource/process/globalmodule evidence remain.',
            scope_audit=str(O/evidence[9]),profile_scope=str(O/evidence[0])),
        'M21':dict(finding='S13 finite63guard reconciliation retained. M22 now derives named SMLA321/QLI208 metadata allDIRECT, not graph; current plan bank/capture-scalar/native generation/all-consumer proofs still pending. No new resource counts or captured producer assumption.',
            evidence=evidence,candidates=['F31'],gap='M21S01/S14exacthelper mappings, S15actualbanks/graphs/generations, S16same-denominator bounds and S18completeprocess remain; named producer placement is no longer wholly unknown.'),
    }
    remaining=deepcopy(read(O/'module-coverage-gaps-v43.json')['remaining'])
    retired=[s for s in remaining if s.startswith('NextM21')]
    write(O/'M22-remaining-history-before-v44.json',dict(previous_version=43,retired_progress_paragraphs=retired,
        reason='Named metadata placement and M22core lifetime guards now answered; preserve actual bank/native epoch/source-scope/process/resource gaps precisely.',overall_complete=False))
    remaining=[s for s in remaining if s not in retired]
    remaining[17:17]=[
        'M22v44: original2files142units1101static source-association sites→16semantic families;32chronological rows archived.41core guard dataflows reconciled;28other graph-file helper sites stillfinitepending, mostworker sites need neighbor-ownership/case crosswalk, not blanketgraphidentity. Same-capture exhaustive metadata+disjoint origin audit derives321SMLA+208QLI namedmetadata529DIRECT/0GRAPH, not bank/capture-ABI proof. kwargs model invocation can make positional DEBUGaddress list empty. ENPU_ENABLE C-env independent of enable_npugraph_ex; source update/replay order/scalar capture/weak outputs/exception release preserved.',
        'NextM22: point-reconcile28 graph parameter-registry/weak-workspace/proxy helper sites, then actual graph/bank/native task-update epoch/all readers/ready/reuse/16phase-unbound binding and worker graph lifecycle/DP dispatch neighbor crosswalk. Current compact receipts lack full80bodies; full pinned results remain on166 and authentication question is pending(no retry). Continue other available original M23-M25/global26 evidence where independent. S16same-denominator critical bounds/S18full EngineCore/transport remain. Do not re-read M21S13 or treat529named placement as unknown, and do not require conditional optimization implementation as an analysis gate.',
    ]
    action=(
        '继续Loop004。v44为M22实际进展：529namedmetadata执行位置推导+41coreguard生命周期/更新合同+原始scope审计。先读M22-graph-lifetime-analysis-summary-v44.json、M22-named-metadata-graph-placement-derivation-v44.json、M22-capture-replay-input-output-and-update-contract-v44.json、M22-original-scope-and-guard-reconciliation-v44.json、M22-semantic-branch-inventory-v44.json、M22-original-requirement-completion-audit-v44.json。'
        'M22原2文件142units1101静态site(acl_graph55/worker1046)都是source关联，不是所有都属于graph；旧32历轮进展行archive为history，active16稳定semanticfamilies；41coreguard对应wrapper ctor/call/updatable replay和runner modelForward/genericUpdate已逐项核对；其余graphfile28helper和worker邻接scope仍需有限case/evidence对照。8原gate仍partial/incomplete，不因分配1101site或报告关闭。M21的18families/63S13语义完成保留不重做。用v3 semantic保持helper，勿改pinned旧helper。'
        '当前namedSMLA321+QLI208共529metadata在v37全32TASK typedfallback0，v20同capture完整native identity classifier断言DIRECT与GRAPH集合不交；历史query脚本SHA核验，所以推导529DIRECT/0GRAPH。不是新rawidentityjoin，不证明opaque/fused内部metadata无工作、plan地址/事件/phase或所有读者，529PMU依旧缺失。不要再将这两类producer位置当完全未知。'
        'Graph contract: runtime NONE/mismatch走原callable，descriptor entry无graph才capture；replay忽略新args/kwargs并回storedoutput。地址DEBUG只看顶层positional Tensor；runner partial(self.model,**model_inputs)→run_model()无位置参数，这条ordinarypath地址list可以空，不能拿assert过当kwargs/nested bank证据。globalpool/weakworkspace/weakoutput/firstcapture返回原output/后replay返回stored refs不是devicecopy。保offloader预同步和copy stream join。'
        'enable_enpu来自C getenv ENPU_ENABLE而非enable_npugraph_ex；actualenv仍unknown。Wrapper非ENPU FULL且非draftEAGLE时主stream.sync；updatable要backend subclass AscendAttentionBackend，localutils predicateSHA5f88fd...未新166attest。目标ContextSource和draftSharedSource不同；ENPU update→replay，非ENPU replay→update，勿Python顺序猜native参数epoch。Runner genericUpdate又受FULL&&!capturing&&!use_sparse&&!use_compress限制，enableENPU内sync；use_compress=hfcfg有compress_ratios，sparse=enable_sfa，别靠SparseFlashMla名字推liveflags。'
        'Runner ENPU时先genericUpdate后model，反之model后update；finally释放inflightMetadata，即失败也释放，优化保异常恢复。Engram每replay前按current IDs/positions/lookback路由，capture用graph inputs，不可跨迭代地址CSE旧值。计划producer还含CPU maxlen/req/head/layout标量，move insidegraph需更新标量或证明固定安全bound，不能只刷新GPU长度tensor。'
        '3个F31条件变体17约束：namedproducer graph内外放置；全同步细化为代际ready/lastreader及必要CPU hostcompletion；独立bank+graph代际pipeline(含descriptor×bank捕获/内存成本、反馈输入就绪和所有rank一致)。设备wait不阻CPU参数mutator，必须host完成或独立native对象。不能直接删sync/跨步复用plan或相加收益，未测任何加速。'
        '下一步先graph-file剩28registry/weak-workspace/proxy小helper按具体task/存储问题查证，再M22实际80graph/16unboundphase/QLI-SMLA参数bank/nativehandles/allreader/ready/reuse及worker图生命周期/DP mode。full v14 graphresult仍在166SHAa035c4...；身份/源码能local推进就继续，不用old16911graph样本替代current166。v42SSHrc255凭据问题仍pending，不自动重试；没有remote活job。随后S16同分母资源critical bounds、S18全进程transport及M23-M25/global26审计。'
        '本轮5runs4PASS1ERROR(初reader误把imported predicate当本地函数StopIteration，未输出sourceartifact；修正import并保留错误)。无remote/模型/torch/NPU/service/health工作，lasthealth仍v12post。35模型7框架0整模块，26理论INCOMPLETE/3136资源/849缺PMU/16phase-unbound保持。Goal active因上述真覆盖缺口，非因条件变体未实现。'
    )
    report='''本轮推进 M22 graph/capture/replay 生命周期，采用已保存 profile → 明确未知边界 → 有限源码补证，未重复 M21 的63个邻接 guard。

两类命名 metadata 的执行位置已获得明确推导。v37 全32份 TASK census 会保留所有 metadata 类型，非 native fallback 数为0，SMLA321、QLI208。v20 的同一快照完整身份分类覆盖154940个 DIRECT、132000个 GRAPH、10个 QUEUE，并逐身份断言 DIRECT 不在 GRAPH 集合；对应历史脚本 SHA 已核验。因此这529个命名 metadata TASK 均为 DIRECT、这两类在 GRAPH 集合中为0。该推导没有重新查询远端，也不证明 opaque/fused内部没有 metadata 工作，或plan bank已绑定；529个metadata PMU缺口仍保留。

定点合同说明了几项会改变优化合法性的边界：

- wrapper 按 runtime mode 与 batch descriptor 分派，首次 capture 消费当时参数，之后 replay 使用已捕获图并返回已存输出，新 args/kwargs 不会重新执行 Python callable。
- DEBUG 地址列表仅含顶层位置参数 Tensor。runner 普通路径通过 partial(self.model,**model_inputs) 调用，位置参数可为空，因此地址 assert 可以在 kwargs Tensor 地址变化时仍通过；它不是全部输入固定地址的证明。两个纯数据反例只说明检查范围，不是运行时 Tensor 实验。
- enable_enpu 来自 C 环境 ENPU_ENABLE，不能与已保存 enable_npugraph_ex 配置等同。其实际值未恢复。wrapper 和 runner 的同步/更新条件不同；updatable backend、主模型 ContextSource、draft SharedSource、ENPU update→replay 与非ENPU replay→update 都保留。不能仅凭 Python顺序推断 native 参数代际。
- generic update 只在 FULL、非capturing、非sparse、非compress 时进入。内核叫 SparseFlashMla 不足以证明 live Python flag。model forward finally 会释放在途 metadata submission，异常路径也必须保留。
- capture 的 offloader sync/join、global graph pool、弱输出/工作区、首次返回与 replay 返回的持有关系都约束 bank 生命周期。弱引用操作不被算成待消除的设备拷贝。Engram 的每轮真实路由与 capture 输入 bank 沿用既有历史/DP合同。
- SMLA metadata producer 除 GPU长度 tensor 外还有 CPU最大长度、请求数、head/layout 等标量。移入 graph 需要正确更新这些标量或证明固定捕获上界安全；只刷新 GPU tensor 内容不构成完整证明。

本轮记录3个 F31 条件细化、17条约束：producer 在图内外的放置；以完整代际协议细化全流同步；独立 bank/graph 的有限代际流水。GPU stream wait 不会阻止 CPU 改写 native descriptor，必要的 host完成或独立native参数对象必须保留。还需真实反馈输入就绪、全部rank一致、捕获 bank地址及最后读者、descriptor×bank内存/捕获成本。没有删除同步、修改实现或宣称实测收益，三项也不相加。

同时对照原始M22范围：2文件142units、1101静态导航site（graph文件55、worker1046）全部保留。它们是 source association，不等于1101个graph语义分支。旧矩阵32条历史进展完整归档，改为16个稳定语义范围；41处核心guard有逐项数据流说明。graph文件另28个registry/weak-workspace/proxy等helper site和worker邻接责任范围仍待有限核对，原8项mandatory gates仍partial/incomplete。没有把导航分配或报告成稿当成整体完成。

一次初始 reader 把 imported use_updatable_graph 当成本文件函数而 StopIteration；发生在source artifact写出前，错误已保留，按明确import修正后成功读取小函数。本轮5项登记4 pass、1 error；无远端登录、模型/torch/NPU/服务/健康操作。166前轮鉴权问题仍待恢复，full graph结果尚在166，16次phase仍未绑定。

35模型候选、7框架候选、0整模块关闭；26理论比较INCOMPLETE、3136资源行、849缺PMU、16phase-unbound保持。下一步是28个具体graph helper、实际graph/native任务/bank/所有读者/代际，以及同分母资源、完整进程transport和其余模块范围。Goal active；未实现条件优化本身不被当成分析未完成的理由。
'''
    jobs=[
        dict(kind='review',status='error',description='Initial bounded reader expected imported use_updatable_graph inside acl_graph.py and hit StopIteration before source artifact; explicit import resolution corrected, initial error retained.',artifacts=[O/'M22-imported-predicate-local-reader-error-v44.json']),
        dict(kind='review',status='pass',description='Same-SHA bounded wrapper/runner lifetime/ENPU/update source; local imported backend predicate marked not fresh166attested; no entire runner/SDK audit.',artifacts=[N/'read_M22_capture_lifetime_points_v44.py',O/evidence[3],O/evidence[4]]),
        dict(kind='review',status='pass',description='Historical immutable query SHA and same-capture complete classification prove529namedmetadataDIRECT/0GRAPH; original1101site16family scope/41guard crosswalk retains open8gates.',artifacts=[N/'build_M22_graph_lifetime_analysis_v44.py',N/'query_M21_native_metadata_v37_166.py',N/'aggregate_resource_roles_v20_166_v3.py',N/'M21-native166-v37.json',O/'resource-coverage-summary-v20.json']+[O/x for x in [evidence[0],evidence[5],evidence[6],evidence[9]]]),
        dict(kind='review',status='pass',description='Positional-address counterexamples and graph-generation DAG;3conditionalvariants17constraints preserve CPU scalar/native mutator/ready/lastreader/feedback/rank/memory/exception requirements, no measured gain.',artifacts=[O/x for x in [evidence[1],evidence[2],evidence[7],evidence[8]]]),
        dict(kind='review',status='pass',description='Previous56file manifest verified; old32M22progressrows archived into separate history,16semanticfamilies and original8open gates persisted without wholemodule closure.',artifacts=[O/evidence[9],N/'persist_profile_refinement_v3.py']),
    ]
    summary=dict(filename='M22-graph-lifetime-analysis-summary-v44.json',previous_turn_classification='PROGRESS:v43neighbor63guardreconciliation',
        named_metadata_DIRECT=529,named_metadata_GRAPH=0,origin_derivation_not_fresh_identity_join=True,
        original_source_files=2,original_indexed_units=142,original_static_sites=1101,graphfile_sites=55,worker_sites=1046,
        local_core_guard_sites_reconciled=41,remaining_graphfile_helper_sites=28,semantic_branch_families=16,prior_chronological_M22rows=32,
        original_mandatory_gates=8,completion_audit='NOT_COMPLETE',conditional_variants=3,condition_rows=17,
        refined_model_candidates=['F31'],related_model_candidates=['F07','F26','F24'],new_model_candidates=[],
        related_framework_candidates=['H02','H06','H07'],remote_jobs=0,remote_pass=0,remote_retained_errors=0,retained_local_errors=1,
        authentication_question_pending=True,
        remote_job_status='No remote jobs or current server handles. Local5runs4PASS/1import-readerERROR preserved/corrected. Prior166authentication failure remains pending,no retry. No model/torch/NPU/service/health work.',
        no_new_model_NPU_service_or_health_work=True)
    persist_refinement(version=44,previous=43,title='v44 M22 named metadata placement and graph generation lifetime',summary=summary,
        changes=['Same-capture exhaustive metadata and disjoint identity classifier derive529namedmetadataDIRECT/0GRAPH; no plan-bank or micro-op absence claim.',
            '41core capture/replay/update guards reconciled;kwargs-only runner can make positional debug check empty;ENPU_ENABLE differs from enable_npugraph_ex;CPU scalar/native epoch/weak output/release constraints explicit.',
            'M22 original2files142units1101navigation sites retained in16semantic families,32chronologicalrows archived;3conditionalvariants17constraints,8originalgates remain incomplete.'],
        candidate_updates=candidates,framework_updates=fw,module_updates=modules,semantic_branches={'M22':sem['branches']},source_read=True,
        remaining=remaining,action=action,jobs=jobs,report=report,
        read_first=['M22-named-metadata-graph-placement-derivation-v44.json','M22-capture-replay-input-output-and-update-contract-v44.json','M22-original-scope-and-guard-reconciliation-v44.json','M22-semantic-branch-inventory-v44.json','M22-original-requirement-completion-audit-v44.json'],
        publication_rows={
            'F31':'| F31 | Metadata、slot、draft与graph发布重组 | 条件 | v44由完整同快照census推导SMLA321+QLI208命名metadata全529DIRECT/0GRAPH。41图生命周期guard已核对；图内producer、同步细化和多bank代际流水需正确CPU标量、kwargs固定bank、native参数epoch及全部ready/last-reader/异常释放。原M22范围与资源/进程仍未闭合，无实测收益。 |',
            'F07':'| F07 | Q投影与RoPE的producer组合 | 条件 | v44补graph边界：replay使用捕获参数，runner kwargs输入不由位置参数DEBUG检查完整覆盖；需保持实际kwargs/nested bank、标量代际、事件与所有读者，不能仅以地址assert通过证明合法。 |',
            'F26':'| F26 | 最后层norm、有效行选择与logits消费者重组 | 条件 | v44补captured输出/工作区weak-ref和first-capture/replay持有合同；保全部输出读者先于下一次bank/replay覆盖，不把弱引用当成设备拷贝或内容跨迭代缓存。 |',
        },
        gate_updates=dict(status='NAMED_METADATA_PLACEMENT_AND_41_CORE_LIFETIME_GUARDS_ANALYZED;FULL_SCOPE_OPEN',answered_by=evidence,
            closed_scope='529namedmetadata recorded placement and41local wrapper/runner guard cases; finite original source inventory/8gate audit.',
            not_closed='28graph helpers,adjacentworker case ownership,actual bank/native epoch/allreaders/16phases,live ENPU/backend/options,resource andfullprocess/globalmodule scope.',
            stop_rule='Do not infer parameter epochs or remove host synchronization from source order/npugraph flag. Advance finite helper and actual saved graph/bank evidence; no SDK-wide audit or automatic goal completion.'),
        scripts=[N/'read_M22_capture_lifetime_points_v44.py',N/'build_M22_graph_lifetime_analysis_v44.py'],
        executor_prefix='zcode-M22-',main_script=N/'finalize_M22_graph_lifetime_v44.py')


if __name__=='__main__':main()
