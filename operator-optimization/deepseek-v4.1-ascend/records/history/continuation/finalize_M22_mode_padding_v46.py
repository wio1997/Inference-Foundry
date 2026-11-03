"""Persist finite DP/mode/padding cases and conditional mode-aware capture alignment."""
from copy import deepcopy
from pathlib import Path
import hashlib
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v3 import persist_refinement


def main():
    assert not(O/'REVIEW-CHECKPOINT-v46.json').exists()
    for a in read(O/'checkpoint-manifest-v45.json')['artifacts']:
        b=Path(a['path']).read_bytes();assert len(b)==a['bytes']and hashlib.sha256(b).hexdigest()==a['sha256'],a['path']
    sem=read(O/'M22-semantic-branch-inventory-v46.json');cross=read(O/'M22-mode-DP-guard-crosswalk-v46.json')
    scope=read(O/'M22-original-scope-and-guard-reconciliation-v46.json');audit=read(O/'M22-original-requirement-completion-audit-v46.json')
    proof=read(O/'M22-DP-mode-padding-and-skip-domain-proof-v46.json');variantfile='H06-mode-padding-reorganization-conditions-v46.json';variant=read(O/variantfile)
    assert len(sem['branches'])==16 and len(sem['static_branch_sites'])==1101
    assert cross['new_exact_local_guards']==41 and scope['local_reconciled_sites']==110
    assert len({r['id']for r in scope['guard_dataflow_reconciled']})==110
    assert proof['DP_valuations']==81 and proof['skip_valuations']==512
    assert len(variant['conditions'])==8 and len(audit['requirements'])==8 and not any(r['completion_proved']for r in audit['requirements'])
    evidence=['M22-mode-padding-point-source-v46.json','M22-mode-padding-and-DP-contract-v46.json','H06-mode-aware-alignment-domain-proof-v46.json',
        'M22-DP-mode-padding-and-skip-domain-proof-v46.json','M22-mode-DP-guard-crosswalk-v46.json',variantfile,
        'M22-semantic-branch-inventory-v46.json','M22-original-scope-and-guard-reconciliation-v46.json','M22-original-requirement-completion-audit-v46.json']
    common=dict(mode_contract=str(O/evidence[1]),alignment=str(O/evidence[2]),DP_proof=str(O/evidence[3]),conditions=str(O/variantfile),
        original_scope_audit=str(O/evidence[8]),implementation=False,new_performance_result=False)
    candidates={
        'F31':dict(review_v46=dict(common,relation='Actual registry keys derive from resolved descriptors after conditional LCM prealignment; Engine startup list is not registry snapshot. Source mode/DP/force/LoRA/backend-support cases now reconciled.'),
            profile_review_state='110_M22_CORE_GUARDS_LOCALLY_RECONCILED;FINAL_WORKER_KEYS_AND_NATIVE_EPOCHS_PENDING',
            conclusion_changing_gap='Source mode/DP/alignment cases and imported utils historical166stage SHA support answered. Actual final per-worker descriptors/mode,DPskip/rankwise counts/native bank epochs,16unbound phases,adjacent capture/process/resource scope remain.'),
        'F32':dict(review_v46=dict(common,relation='Projection/query-domain repartition must retain global DP/EP carrier,modes,feedback and row map. TP-only16row legality is not unchanged FULLuniform6/TP8 legality; cannot infer fewer communication rows or work from local query count.')),
        'F20':dict(review_v46=dict(common,relation='Routing ID/padding reuse keeps DP synchronized mode,full carrier and correct logical-vs-physical mask/row mapping;24targetrows are not blanket one-third savings when requests2 schedule12.')),
    }
    fw={
        'H06':dict(review_v46=dict(common,relation='Uniform6+TP8 necessary24 bound retained; worker prealignment and post-resolution registry chronology now explicit. Mode-aware alignment alternative targets genuinely supported nonuniform/PIECEWISE branches; fixed FULLuniform case has no proved16row drop-in or measured gain.'),
            variants_v46=[variant['new_conditional_variant']],implementation=False,new_performance_result=False),
        'H02':dict(review_v46=dict(common,relation='DP packed all_reduce gathers disjoint column fields,then max tokens/min mode. Force eager does not eliminate coordination;skip predicate recomputed with model-role/comm/recompute context,not memoized by config id.')),
    }
    modules={'M22':dict(finding='41DP/mode/alignment source guards reconciled(37originalS04+4helperreclassifications),110local core guard dataflows total. FouroriginalsiteIDs moved fromadjacent families toS04 with all1101retained. Actualregistry sourcechronology,DPmodeMIN/tokenMAX/skipbranches and historical utils SHA bound;1conditionalH06variant8constraints, no measured gain.',
        evidence=evidence,candidates=['F31','F32','F20'],gap='Actual final worker descriptors/DPskip/rankwise field values,all native bank/handle/event/task epochs,16phase-unbound graphs,capture/warmup/workerneighbor relevantcase/resource/fullprocess/global26 scope remain.',
        scope_audit=str(O/evidence[8]),profile_scope=str(O/'M22-named-metadata-graph-placement-derivation-v44.json'))}
    remaining=deepcopy(read(O/'module-coverage-gaps-v45.json')['remaining']);retired=[s for s in remaining if s.startswith(('M22v45','NextM22'))]
    write(O/'M22-remaining-history-before-v46.json',dict(previous_version=45,retired_progress_paragraphs=retired,
        reason='Mode/DP/alignment finitecases and Engine-to-worker key chronology now answered; retain actual live branch/value/native generation and original remaining module scope.',overall_complete=False))
    remaining=[s for s in remaining if s not in retired]
    remaining[17:17]=[
        'M22v46: mode/DP/alignment41newsourceguard cases(37originalS04+4helper IDs)→110local M22coreguards,all55graphfile cases retained.16families1101originalnavigationIDs intact. Underalignmentguard initialEngine[8,16,24]/q6TP8/max24 rounds[24]beforemoderesolution;registrycomesfromresolveddescriptors,notEngine list. Actualfinalkeys/mode/DPskip stillunbound. DPpackedcolumnallreduce→minmode/maxcount,conditionaluniformcarrier andfullskipdecisiontree analyzed;81/512mathvaluations notactualcasefrequency.',
        'NextM22: move to profile-led capture/warmup/worker-neighbor ownership using saved current graph/metadata/native evidence before further source;do not read entire387line_dummy_run or re-do41DP/55graphfile cases. Actual final worker keys/DP skip/rankwise mode/token metadata and nativegraph/task/bank/event/handle/readiness/last-reader/16unboundphase need decisive saved evidence. Fullresults remainon166;authenticationquestion pending,no retry/no livejobs. Utils5f88... now has verified historical166stage receipt support; actualbackend/ENPU/nativeepochs remainunknown. Then S16same-denominator resourcebounds/S18fullEngineCoretransport andM23-M25/global26audit; no implementation-as-analysisgate.',
    ]
    action=('继续Loop004。v46是DP/mode/padding实际分析进展。先读M22-mode-padding-analysis-summary-v46.json、M22-mode-padding-and-DP-contract-v46.json、H06-mode-aware-alignment-domain-proof-v46.json、M22-DP-mode-padding-and-skip-domain-proof-v46.json、M22-mode-DP-guard-crosswalk-v46.json及M22-original-requirement-completion-audit-v46.json。'
        'M22累计110localguards：既有55graphfile+14runnercore，再37mode/DP原S04及4alignmenthelper；4原ID只改family分配，全1101ID/16semanticfamilies不丢。M21S13的63、M22graph55、DPmode41、529namedmetadataDIRECT/0GRAPH不再当未知或整段重读。'
        'Profile既有64decodegraphs中16B2/T12与48B4/T24仍physical24；3712CPUeligibility是另一人群，16unbound未补配。Engine启动日志有效sizes[8,16,24]不是最终worker registry。新source_check_and_update_mode按backend/spec最弱support聚合，若_should_align(初decodeFULL且CP/SP/passSP任一true)，先LCM6/8=24把list变[24]并resolverTP改1，再resolve mode/initkeys；registry取resolved capturedescs。不能将条件推导当live最终keys快照，或以Engine[8,16]可见就认有可用16rowFULLuniform图。'
        'Uniform默认需hasInitialState&&maxScheduled=qLen&&tokens=max*reqs，DCP/hybrid/recompute有prefill条件；forceUniform/forceEager/LoRA/encoder/cascade分开。Dispatcher FULLkey优先、relaxedPIECEWISE次之、NONEfallback，DPredispatch只允许syncedMode并assertagreedcount。_padSP在SP或DSACP时TProundup；uniform6+TP8仍必须24，只有新非uniform/PIECEWISE接口或域重组才可能16，不能只改setting。'
        'DP_sync非skip时CPU2×DPpackedtensor每rank写自身column，allreduceSUM只汇集独立column，之后min graphmode、max token，不是把tokens加总。any rankNONE→allNONE；graph/draft/ALLGATHER/MC2/适用MegaMoe/finegrained任一true时padallmax，否则可保不同eagerrows。ForceEager不取消DPsync。Skiphelper hierarchy或mega_supported先false，dense跳过，普通MoE非KVconsumer不跳，KVconsumer还受CANNmega/decodeMC2/prefillMC2/separate/recompute限制；每call重算，不按idmemo。ActualKVtransfer/skip/rankwisebits未知，不能凭MC2或本rankshape断言skip。'
        '81DPtoken/mode例和512skip布尔赋值只是source域数学，不是实测timing/频率。H06只新增条件variant MODE-AWARE-CAPTURE-ALIGNMENT：先resolvebackendmode再只对实际FULLuniform施LCM，保TP/metadata/keys/DPcarrier；源TODO支持此替代但当前fixedFULLuniform无测得16row收益。8constraints，不能按少1/3rows报1/3EP或时延。'
        'v16旧receipt全文含timestamp/envelope而stdout仅source子集；v46现代compact embeddedhash校验失败已记录，改核历史manifest原receipt SHA+源码SHA，不伪造fullresult rehash。padding-config旧receipt验证utils5f88fd...与冻结一致，已有166stage sourcehash证据，故v44/v45该predicate部署源码SHA历史缺口补齐；不等于livebackend/ENPU/registry/nativeepoch。'
        '下一步真正未证范围：profile先核capture/warmup/workerneighbor责任和当前80graph/metadata/native可用字段，不能整读_dummy_run387lines或SDK；实际workerfinalkeys/DPskip/rankwiseflags/native任务/层/handle/event/workspace/bank/所有读者/reuse/16phase绑定仍需保存资料。鉴权v42rc255已问凭据仍pending，不自动重试；本轮0remote无livehandles，无model/torch/NPU/service/health，lasthealthv12post。之后same-denominatorresource criticalbounds、fullEngineCoretransport和M23-M25/global26审计。'
        '本轮5runs3PASS2ERROR：首inventory误用remote basename作本地文件、旧receipt moderncompact hashguard失败均保留；legacy SHA/sourcepoints、41guard+LCM/DPskip域数学、formal audit三个PASS。35model7framework0整模块，26理论INCOMPLETE/3136resource/849缺PMU/16unbound保持；8原M22gates仍partial，Goalactive因真实覆盖缺口，不因条件实现没做。Usepersist_profile_refinement_v3，勿改pinned旧helper。')
    report='''本轮以保存的64次decode图为起点：其中16次调度2请求/12tokens、48次4请求/24tokens，记录的目标图物理域仍为24行。没有把窗口频率当全部负载频率，也没有补配16次未知图阶段。

补核了37处原mode/DP分支及4处对齐helper分支，累计M22核心本地数据流记录110处；全部1101个原静态导航ID及16个语义范围保留。新增源码只涉及明确问题的方法/小helper，DP同步、dispatcher和旧配置片段直接复用。

捕获尺寸的时序得到澄清：Engine启动日志的[8,16,24]不等于最终worker registry键。worker先按各backend/cache spec的最低支持聚合；满足FULL-decode且CP/SP/pass-SP条件时，会在mode resolver前按LCM(6,8)=24预对齐，并把resolver用TP改为1。该分支下初始列表变为[24]；之后还会resolve mode、建立descriptor，registry取其实际descriptor尺寸。当前尚无最终registry快照，因此没有把条件推导当成全部运行时值。

Uniform decode还需要初始状态、逐请求最大宽度匹配和总tokens=宽度×请求数；DCP/hybrid/recompute、force参数、LoRA、encoder/cascade各有分支。dispatcher优先FULL key，其次放宽的PIECEWISE key，最后NONE fallback。保持FULLuniform6+TP8接口时16行无效；16行只满足TP约束，不能靠改capture-size设置替换24行。

DP非skip路径的CPU all_reduce汇集每rank写入的独立列，之后显式取最小graph mode和最大token数。任一rank为NONE就协调为NONE；图回放、draft或规定的MoE/finegrained通信条件仍可能要求统一最大carrier。force_eager只限制图分派，不取消DP协调。skip策略按模型角色、KV-consumer、hierarchy/MegaMoe、prefill/decode通信与recompute等条件每次重算，不能按配置对象id或MC2名称推断。81个DP例、512个skip逻辑赋值只验证源码条件域，不是实测耗时或实际rank配对。

H06补充一个条件变体：先确定backend最终mode，再仅对实际FULLuniform域施加joint alignment。它可用于评估真正支持非uniform/PIECEWISE的分支；现有源码TODO也指出预对齐早于mode降级的限制。保留8条约束，包括metadata/descriptor、TP、DP/EP carrier、实际row map和bank/native代际。没有宣称当前固定FULLuniform路径可降到16，或少1/3行就少1/3MoE工作/延迟。

另补齐历史源码证据：v16 immutable receipt中的utils.py SHA与当前冻结文件一致，给前两轮imported predicate提供历史166stage hash支持；实际backend、ENPU、registry和native epoch仍未快照。旧receipt的full-result有未返回的timestamp/envelope，不能用现代子集compact serializer重算其全文SHA；本轮保留该失败，改核原receipt的历史manifest SHA与源码SHA，未掩盖差异。

登记5项：3 pass、2 error。两次错误分别为把远端结果名误作本地文件、旧receipt格式不适用新hash校验；均在分析阶段保留并修正。无远端登录、模型/torch/NPU/服务/健康操作。

整体仍未完成：实际worker最终keys、DP skip及rankwise字段、graph/task/参数bank/事件/handle代际与全部读者、16次未知阶段、capture/warmup和worker邻接责任、资源上界及完整EngineCore/transport仍待证。35模型候选、7框架候选、0整模块关闭，26理论比较INCOMPLETE、3136资源行、849缺PMU、16phase-unbound保持。Goal active。
'''
    jobs=[
        dict(kind='review',status='error',description='Initial inventory confused remote C1-and-padding result basename with local artifact; no source result emitted, corrected to original pinned receipt.',artifacts=[O/'M22-padding-local-reference-error-v46.json']),
        dict(kind='review',status='error',description='Modern compact embedded-result guard failed on legacy timestamp/envelope result format; immutable original receipt SHA/source SHA verified instead, no full-result digest fabricated.',artifacts=[O/'M22-legacy-receipt-compact-hash-guard-error-v46.json']),
        dict(kind='review',status='pass',description='Bounded mode/DP/alignment source plus legacy immutable receipt/source hash verification;historical utils166stage SHA support restored,no remote/model imports.',artifacts=[N/'read_M22_mode_padding_points_v46.py',O/evidence[0]]),
        dict(kind='review',status='pass',description='41exactguard cases,LCM/bucket chronology,81DP and512skip logical cases,conditionalH06mode-aware variant8constraints;no observed gain.',artifacts=[N/'build_M22_mode_padding_analysis_v46.py']+[O/x for x in evidence[1:6]]),
        dict(kind='review',status='pass',description='Previous43filemanifest verified;110coreguard IDs and all1101originalsite IDs/16families preserved;8mandatory gates remain open.',artifacts=[O/x for x in evidence[6:]]+[N/'persist_profile_refinement_v3.py']),
    ]
    summary=dict(filename='M22-mode-padding-analysis-summary-v46.json',previous_turn_classification='PROGRESS:v45registry28guardreconciliation',
        new_guard_dataflows=41,total_local_core_guards=110,original_static_sites=1101,semantic_branch_families=16,original_mandatory_gates=8,completion_audit='NOT_COMPLETE',
        DP_example_valuations=81,skip_valuations=512,conditional_new_framework_variant=1,condition_rows=8,
        Engine_startup_sizes_not_final_registry=True,final_registry_actual_snapshot=False,utils_historical_stage_hash_supported=True,
        refined_framework_candidates=['H06'],related_framework_candidates=['H02'],related_model_candidates=['F31','F32','F20'],new_model_candidates=[],
        remote_jobs=0,remote_pass=0,remote_retained_errors=0,retained_local_errors=2,authentication_question_pending=True,
        remote_job_status='No remote/current server jobs. Local5reviews3PASS/2legacy-reference-formatERROR preserved/corrected. Prior166auth failure remains pending,no retry. No model/torch/NPU/service/health work.',no_new_model_NPU_service_or_health_work=True)
    persist_refinement(version=46,previous=45,title='v46 M22 DP graph mode and capture-padding domains',summary=summary,
        changes=['41mode/DP/alignment guard cases reconciled;110localcore total,1101originalsite IDs/16families retained.',
            'Worker conditionalLCM prealignment and capture-descriptor registry chronology distinguished from Engine startup list;DP minmode/maxcount and full skip policy analyzed.',
            '1conditionalH06mode-aware alignment variant8constraints,81DP/512skip mathematical cases,no measured gain;utils historical166stage hash evidence recovered;original8gates stillopen.'],
        candidate_updates=candidates,framework_updates=fw,module_updates=modules,semantic_branches={'M22':sem['branches']},source_read=True,
        remaining=remaining,action=action,jobs=jobs,report=report,
        read_first=['M22-mode-padding-and-DP-contract-v46.json','H06-mode-aware-alignment-domain-proof-v46.json','M22-DP-mode-padding-and-skip-domain-proof-v46.json','M22-mode-DP-guard-crosswalk-v46.json','M22-original-requirement-completion-audit-v46.json'],
        publication_rows={'F31':'| F31 | Metadata、slot、draft与graph发布重组 | 条件 | v46累计110处M22核心分支已本地核对。registry从worker最终capture descriptors建立，不能用Engine启动[8,16,24]替代；FULLuniform6/TP8对齐域仍为24，DP需协调模式/载荷。实际keys/rankwise字段/native bank与全进程、资源仍未闭合，无实测收益。 |'},
        gate_updates=dict(status='41_MODE_DP_ALIGNMENT_CASES_ANALYZED;ACTUAL_FINAL_VALUES_OPEN',answered_by=evidence,
            closed_scope='37originalS04+4alignment source cases,DPpolicy domain and conditional capture-size chronology;historical utils source provenance.',
            not_closed='Actual finalkeys/rankwiseflags/DPskip/native bank epochs/16phases,capture-warmup and original neighbor/resource/process/globalmodule coverage.',
            stop_rule='Do not re-do41mode cases or infer16rowFULLuniform legality from Engine list. Use saved profile before next bounded capture/warmup questions; no whole dummy/SDK audit.'),
        scripts=[N/'read_M22_mode_padding_points_v46.py',N/'build_M22_mode_padding_analysis_v46.py'],executor_prefix='zcode-M22-',main_script=N/'finalize_M22_mode_padding_v46.py')


if __name__=='__main__':main()
