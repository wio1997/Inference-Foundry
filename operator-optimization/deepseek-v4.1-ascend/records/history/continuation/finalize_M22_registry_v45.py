"""Persist completed local graph-helper cases without claiming whole-module closure."""
from copy import deepcopy
from pathlib import Path
import hashlib

from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v3 import persist_refinement


def main():
    assert not(O/'REVIEW-CHECKPOINT-v45.json').exists()
    for a in read(O/'checkpoint-manifest-v44.json')['artifacts']:
        b=Path(a['path']).read_bytes();assert len(b)==a['bytes']and hashlib.sha256(b).hexdigest()==a['sha256'],a['path']
    sem=read(O/'M22-semantic-branch-inventory-v45.json');scope=read(O/'M22-original-scope-and-guard-reconciliation-v45.json')
    audit=read(O/'M22-original-requirement-completion-audit-v45.json');cross=read(O/'M22-registry-guard-crosswalk-v45.json')
    truth=read(O/'M22-update-and-sync-mutual-exclusion-proof-v45.json')
    conditionsfile='M22-registry-refined-generation-conditions-v45.json';conditions=read(O/conditionsfile)
    assert len(sem['branches'])==16 and len(sem['static_branch_sites'])==1101
    assert scope['local_reconciled_sites']==69 and cross['remaining_graphfile_helper_sites']==0
    assert len(cross['graphfile_reconciled_site_ids'])==55 and len(cross['new_reconciled_guards'])==28
    assert truth['boolean_valuations']==256 and truth['both_update_paths_selected']==truth['both_explicit_syncs_selected']==0
    assert len(conditions['refined_existing_variant_ids'])==3 and conditions['new_variants']==[] and len(conditions['refinements'])==14
    assert len(audit['requirements'])==8 and not any(r['completion_proved']for r in audit['requirements'])
    evidence=[
        'M22-registry-workspace-point-source-v45.json',
        'M22-registry-guard-crosswalk-v45.json',
        'M22-registry-and-update-alternative-contract-v45.json',
        'M22-registry-key-completeness-and-alias-proof-v45.json',
        'M22-update-and-sync-mutual-exclusion-proof-v45.json',conditionsfile,
        'M22-registry-generation-obligations-v45.json',
        'M22-semantic-branch-inventory-v45.json',
        'M22-original-scope-and-guard-reconciliation-v45.json',
        'M22-original-requirement-completion-audit-v45.json',
    ]
    common=dict(registry_contract=str(O/evidence[2]),key_proof=str(O/evidence[3]),alternative_proof=str(O/evidence[4]),
        refined_conditions=str(O/conditionsfile),original_scope_audit=str(O/evidence[9]),implementation=False,new_performance_result=False)
    candidates={
        'F31':dict(review_v45=dict(common,relation='Remaining28 graph-helper cases answered; all55graphfile static guards plus14runner guards locally reconciled. Namespace/bucket completeness and alternative update/barrier paths refine existing graph generation candidates without new variants or gains.'),
            refined_variant_ids_v45=conditions['refined_existing_variant_ids'],
            profile_review_state='ALL_55_GRAPHFILE_GUARDS_PLUS14_RUNNER_CORE_LOCALLY_RECONCILED;REGISTRY_EPOCHS_CONDITIONAL',
            conclusion_changing_gap='Finite graph-file registry/workspace/proxy/error cases answered. Actual graph/native task/layer/event/handle/workspace generations,live backend/ENPU/mode,16unbound phases,worker adjacent case scope,resource/fullprocess/global26 coverage remain.',
            next_profile_or_contract_action='Advance M22S04 DP/mode/padding using saved root schedule2x12/4x24, effective capture[8,16,24] and existing DP contracts; only then point-read unresolved dispatcher branches. Do not redo55graphfile cases or529namedmetadata placement.'),
        'F26':dict(review_v45=dict(common,relation='Workspace weak-ref conversion only replaces workspace references and skips None; it does not free proven-dead output/storage or erase other registry planes. Preserve all cross-replay norm/head output readers and native owner generations.')),
    }
    fw={
        'H02':dict(review_v45=dict(common,relation='Common stable backend selects either wrapper-updatable or backend-impl update; common ENPU flag selects alternative explicit wrapper/runner syncs. Do not double-count either as additive current work; actual host cost remains unmeasured.')),
        'H06':dict(review_v45=dict(common,relation='Capture bucket reorganization needs main/draft/prefill namespace and all graph/event/handle/param planes, not workspace key alone. One-time setters reject reinit; workspace-only updates can add incomplete keys and pre-init writes no-op. Python map/list initialization is not GPU workspace allocation.')),
    }
    modules={'M22':dict(
        finding='Remaining28graph-file guard cases reconciled; cumulative55/55graphfile+14runner=69local guard dataflows. Three namespace/four-plane registry invariants and256boolean valuations establish source-conditional update/sync mutual exclusion.16semantic families/1101original source-navigation sites retained;3v44variants receive14refinements,0newvariant or gain.',
        evidence=evidence,candidates=['F31','F26'],gap='Actual bank/native event/handle/layer/workspace/graph epochs; live mode/backend/ENPU;16phase-unbound graphs; original worker relevant/adjacent scope and resource/process/globalmodule gates remain.',
        scope_audit=str(O/evidence[9]),profile_scope=str(O/'M22-named-metadata-graph-placement-derivation-v44.json'))}
    remaining=deepcopy(read(O/'module-coverage-gaps-v44.json')['remaining'])
    retired=[s for s in remaining if s.startswith(('M22v44','NextM22'))]
    write(O/'M22-remaining-history-before-v45.json',dict(previous_version=44,retired_progress_paragraphs=retired,
        reason='Remaining28graph-file helper frontier is now answered. Keep precise live epoch/worker/phase/resource/process gaps and avoid repeating55local cases.',overall_complete=False))
    remaining=[s for s in remaining if s not in retired]
    remaining[17:17]=[
        'M22v45: all55graph-file static guard cases nowlocally analyzed,plus14runnercore→69;0remaining graph-file helper case.16stable semantic families/1101source-navigation sites retained with original8gates open.Three namespaces/fourmap planes,one-time init/no-op pre-init/workspace-only key and weakref ownership contracts reviewed.256boolean valuations under common stable backend+ENPU prove wrapper/backend updates and these explicit wrapper/runner syncs alternative,not additive work.529namedmetadataDIRECT/0GRAPH v44 result retained.',
        'NextM22: S04 worker graph-mode/DP/padding finitecase audit using recorded root schedule2req12tokens/4req24tokens,effective capture[8,16,24],DP2/TP8/spec5 and existing v16DP contracts before unresolved pointcode; reconcile relevant capture/warmup/source-owner neighbors without whole-runner read. Actual graph/task/layer/event/handle/workspace/native parameter epochs and16unbound phases need full saved data/decisive evidence.166auth question pending,no retry; no live jobs. S16same-denominator resource bounds/S18full EngineCore-transport andM23-M25/global26remain. Do not redo55graphhelper cases or require unimplemented conditional variants as analysis gaps.',
    ]
    action=(
        '继续Loop004。v45实质进展是剩余28个graph helper guard全部对照；累计graph文件55/55和runner14，共69localcases完成。先读M22-registry-reconciliation-summary-v45.json、M22-registry-and-update-alternative-contract-v45.json、M22-registry-key-completeness-and-alias-proof-v45.json、M22-update-and-sync-mutual-exclusion-proof-v45.json、M22-semantic-branch-inventory-v45.json及M22-original-requirement-completion-audit-v45.json。'
        '保持profile优先，只有结论缺口才定点源码；不做整库/SDK/whole runner审计或新model/torch/NPU/service/health工作。M21S13的63guard、M22graphfile55guard、namedSMLA321+QLI208全529DIRECT/0GRAPH均已回答，不再列为未分析。'
        'GraphParams四张num_tokens map(events,workspaces,handles,attn_params)属于main/draft/draft-prefill三个命名空间；同token数不是同graph/任务/层/参数/代际，独立Pythonlist不是独立nativehandle/显存证明。初始化只是dict/[]/None，不报GPU allocation收益。Setter已有非None就raise；workspace updater未初始化时no-op，已初始化可新增workspace-only key，不能拿workspace存在当完整bucketready。weak_ref_workspaces只换非None workspace引用，不碰events/handles/tuple，不是copy/clear/释放证明。'
        'update_full_graph_params在use_updatable_graph为True直接return，否则在current-vllm-config作用域解析impl_cls并转发当前stream/context/tokens/config/spec/draftmetadata。与v44wrapper同一稳定backend时两种更新互斥；wrapper显式sync要!ENPU，runner显式sync要ENPU，common flag时两处也互斥。256布尔赋值仅源码条件代数，不是实际调用/耗时统计，也不排除nestedwrapper/native/event其它屏障。保持CPU mutator hostcompletion/native epoch/lastreader条件，不能直接删sync。'
        '3个v44generation变体只补14约束、0新variants/候选/收益。Proxy委托当前runnable，不缓存跨owner替换的boundmethod；DEBUG只改异常信息；错误识别只是文本分支无实际报错/升级建议。所有nativeepoch、actualENPU/backend/predicate部署SHA、workspaceref/事件/handle快照仍条件未证，别用源码默认None/empty代实际case。'
        '下一步真正范围M22S04图mode/DP/padding：先用v13root观察(2req12tokens或4req24tokens、QSLshape5)、v42有效main[8,16,24]/warmup1、DP2TP8/spec5和v16sync合同核对；不能以B*6/本rankshape直接断定全DP graphbucket或按ordinal拼16unbound。再只对未回答_determine_batch_execution_and_padding/_check_and_update_cudagraph_mode等分支补点源码，既有_sync_metadata_across_dp无需整段重读。然后capture/warmup/worker邻接原scope以及actual80graph/task/native参数bank/allreader/reuse/16phase未绑定。完整graph数据在166，鉴权v42rc255的问题已问凭据仍pending，不自动重试；本轮0remote/无livehandles。'
        '原M22两文件142units1101静态导航site保留16semanticfamily，worker其余source关联要审相关性/责任模块，而非全部误认graphbranch或默默排除；8原mandatorygates仍partial/incomplete。随后同分母resource/criticalbounds、全EngineCore/transport和M23-M25/global26审计。35model7framework0整module，26理论INCOMPLETE/3136resource/849缺PMU/16unbound图保持。'
        '本轮3runs全部PASS：同SHA14helper/声明定点读、28guard+registry模型+256逻辑赋值+协议DAG、原scope/8gate一致持久化审核。无新错误/模型/torch/NPU/service/health作业，lasthealth仍v12post。使用persist_profile_refinement_v3，别修改已pinned旧helper。Goal active因真实scope/证据缺口，不因条件实现没做。'
    )
    report='''本轮完成剩余28处 registry、workspace、proxy 和诊断分支的定点数据流核对。累计 graph 文件55处静态 guard 均已本地分析，加上前轮 runner 的14处核心分支，共69处；不再把这28处作为未分析范围。原 M22 两文件142units、1101个 source 关联导航位置和16个语义范围保留，整体8项mandatory gates仍未关闭。

关键结论是复用键和持有关系不能简化成 token 数：GraphParams 的 events、workspaces、handles、attn_params 四张 map，分别属于 main、draft、draft prefill 三个命名空间。相同 num_tokens 不代表相同图、任务、层、参数或代际；独立的 Python 容器也不证明不同显存或 native handle。初始化只建立 dict、空列表与 None，不在这些 helper 中分配 GPU workspace，因此没有将其算作待消除的显存分配。

workspace updater 在 registry未创建时静默不操作；创建后可新增一个 workspace key，却不建立对应 events/handles/attn_params 或 graph entry。纯Python模型构造了这一不完整 bucket 反例，以及不同命名空间引用同一存储的反例；重复 initializer 会报错。动态 bucket、双 bank 或跨阶段共享需要完整初始化、配对及生命周期协议，不能只看 workspace key。weak_ref_workspaces 只替换非None workspace 引用，不清空其他表，也不证明内存可释放或存在设备copy收益。

又排除了两种重复收益计算。在同一个稳定 backend 的普通 FULL wrapper/runner 组合中，use_updatable_graph 为真时 legacy update helper立即返回；两种更新路径是互斥选择。在构造后 ENPU flag一致的条件下，wrapper 的显式 stream同步需要非ENPU，runner 的显式同步需要ENPU，也不同时选中。256种布尔赋值均验证该逻辑；它们不是实际调用计数，不说明其它 nested/native/event同步不存在，更没有得到同步删除收益。必要的 CPU参数改写 host完成、native参数epoch和GPU ready/last-reader要求均保留。

本轮没有新增候选或变体，仅为前轮3项 graph-generation 条件变体增加14条约束。proxy 委托当前 runnable，DEBUG影响缺属性的错误信息；文本错误识别也不说明实际发生过硬件或stream故障。没有提出升级或变更服务。

验证登记3项全部pass：同SHA的14个小函数/声明、28处分支及registry模型/256逻辑赋值/协议图、原范围和8gate一致性审计。本轮没有远端登录、模型/torch/NPU/服务/健康操作。前轮529次命名 metadata DIRECT/0GRAPH 的推导保留；实际 bank、任务/层/event/handle配对及native代际仍未获得运行时证明，16次图阶段仍未绑定，166鉴权问题仍待恢复。

下一步用保存的2请求12tokens/4请求24tokens、有效capture[8,16,24]、DP2/TP8/spec5与已有DP同步合同核对 worker 的图模式/全局padding选择，再仅补未解分支代码。不能只凭本rank行数推断全DP图bucket或按执行序号补16次未知阶段。之后继续capture/warmup、原worker邻接责任范围、实际graph参数bank/全部读者、同分母资源上界、完整进程transport及其余模块。

35模型候选、7框架候选、0整模块关闭；26理论比较INCOMPLETE、3136资源行、849缺PMU、16phase-unbound保持。Goal active，未以局部源码核对或报告成稿代替全范围完成。
'''
    jobs=[
        dict(kind='review',status='pass',description='Same-SHA14small functions/GraphParams/global declarations reconcile remaining graph registry/workspace/proxy/error cases; no native SDK/whole runner/remote read.',artifacts=[N/'read_M22_registry_points_v45.py',O/evidence[0]]),
        dict(kind='review',status='pass',description='28exactguard cases plus registry namespace/incomplete-key/reference models and256Boolean valuations;3existing variants receive14constraints,0newvariant/gain,protocol DAG valid.',artifacts=[N/'build_M22_registry_reconciliation_v45.py']+[O/x for x in evidence[1:7]]),
        dict(kind='review',status='pass',description='Previous52file manifest verified; cumulative55graphfile+14runner69local guards and16semanticfamilies/1101navsites retained,original8gates remainopen.',artifacts=[O/x for x in evidence[7:]]+[N/'persist_profile_refinement_v3.py']),
    ]
    summary=dict(filename='M22-registry-reconciliation-summary-v45.json',previous_turn_classification='PROGRESS:v44placement41guardaudit',
        newly_reconciled_guard_sites=28,total_local_core_guards=69,graphfile_guard_sites=55,all_graphfile_static_guards_locally_reconciled=True,
        remaining_graphfile_helper_sites=0,semantic_branch_families=16,original_static_sites=1101,original_mandatory_gates=8,completion_audit='NOT_COMPLETE',
        boolean_valuations=256,new_variants=0,refined_existing_variants=3,condition_refinements=14,
        refined_model_candidates=['F31'],related_model_candidates=['F26'],related_framework_candidates=['H02','H06'],new_model_candidates=[],
        remote_jobs=0,remote_pass=0,remote_retained_errors=0,retained_local_errors=0,authentication_question_pending=True,
        remote_job_status='No remote/current server jobs. Local3reviews3PASS,no newerror; prior166authenticationfailure pending without retry. No model/torch/NPU/service/health work.',
        no_new_model_NPU_service_or_health_work=True)
    persist_refinement(version=45,previous=44,title='v45 M22 registry/workspace invariants and update alternatives',summary=summary,
        changes=['All28 remaining graph-helper guard cases reconciled; cumulative55graphfile+14runner69localcases,0remaining graphfilehelper frontier.',
            'Three namespaces/fourplane bucket completeness/init/weakref contracts and256Boolean valuations exclude duplicate update/barrier savings under stated source conditions.',
            'No newvariants/candidates/gain;3existing generation variants refined14constraints. Original16family1101site8gate scope remainsopen for actual native epochs/worker/phase/resource/process coverage.'],
        candidate_updates=candidates,framework_updates=fw,module_updates=modules,semantic_branches={'M22':sem['branches']},source_read=True,
        remaining=remaining,action=action,jobs=jobs,report=report,
        read_first=['M22-registry-and-update-alternative-contract-v45.json','M22-registry-key-completeness-and-alias-proof-v45.json','M22-update-and-sync-mutual-exclusion-proof-v45.json','M22-semantic-branch-inventory-v45.json','M22-original-requirement-completion-audit-v45.json'],
        publication_rows={
            'F31':'| F31 | Metadata、slot、draft与graph发布重组 | 条件 | v45累计graph文件55guard及runner14处本地合同已审。main/draft/prefill四plane registry不能只按token数复用；workspace-only key不等于bucket就绪。两套更新和显式同步在规定backend/ENPU条件下互斥，不重复计收益。529命名metadataDIRECT结论保留，actual bank/native epoch/全部读者、worker原scope和资源/进程仍未闭合。 |',
            'F26':'| F26 | 最后层norm、有效行选择与logits消费者重组 | 条件 | v45补workspace引用转换与namespace/owner合同；weak-ref既不证明copy也不证明内存可释放。跨replay保全部hidden/output读者及native代际，实际bank alias和最后读者仍待绑定。 |',
        },
        gate_updates=dict(status='ALL_55_GRAPHFILE_GUARD_SOURCE_CASES_ANALYZED;ACTUAL_GENERATIONS_OPEN',answered_by=evidence,
            closed_scope='Remaining28registry/workspace/proxy/error guards; complete55graphfile static guard localcases;namespace/bucket/source update/sync alternatives.',
            not_closed='Actual registry/native bank/event/handle/layer/graph epochs,live mode/backend/ENPU,worker relevant-case/phase/resource/fullprocess/global26 scope.',
            stop_rule='Do not redo55graphfile cases or price dict/None/weakref as device work. Advance profile-led worker DP/mode/padding then actual native generation and remaining modules.'),
        scripts=[N/'read_M22_registry_points_v45.py',N/'build_M22_registry_reconciliation_v45.py'],executor_prefix='zcode-M22-',main_script=N/'finalize_M22_registry_v45.py')


if __name__=='__main__':main()
