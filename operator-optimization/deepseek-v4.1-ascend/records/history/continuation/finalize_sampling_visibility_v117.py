"""Persist profiler non-observability evidence without promoting runtime branch coverage."""
from pathlib import Path
from copy import deepcopy
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v90 import persist_refinement

def main():
    assert not(O/'REVIEW-CHECKPOINT-v117.json').exists()
    assert read(O/'sampling-visibility-preflight-v117.json')['status']=='PASS'
    names=['sampling-visibility-inputs-v117.json','sampling-visibility-source-contract-v117.json',
        'sampling-visibility-profile-projection-v117.json','sampling-visibility-proof-and-boundaries-v117.json',
        'sampling-visibility-consequences-v117.json']
    detail=dict(evidence=names,installed_shared_descriptor_cap=5,
        same_prefix_flag_witnesses=12,greedy_executions_reused=72,hash_executions_reused=88,
        historical_loaded_launcher_bound=False,actual_runtime_flags_recovered=False,
        new_pairs=0,new_groups=0,new_native_promotions=0,whole_requirement_closed=False)
    for base in ['candidate-relation-clauses','candidate-relation-coverage','candidate-pair-navigation',
        'candidate-operator-footprints','operator-analysis-coverage-ledger']:
        d=deepcopy(read(O/(base+'-v116.json')));d['sampling_visibility_v117']=deepcopy(detail)
        if base=='candidate-relation-clauses':
            group=next(g for g in d['higher_order_groups']if g['id']=='HG73-TAIL')
            group['sampling_visibility_v117']=dict(**detail,
                contract='Use positive per-call guard and loaded ABI evidence before selecting a tail graph. Missing BOOL/synthetic inputs in a saturated five-entry descriptor packet cannot remove a random/entropy/RNG branch or its observers.')
        write(O/(base+'-v117.json'),d)
    cov=read(O/'candidate-relation-coverage-v117.json')
    obligations=deepcopy(read(O/'coverage-current-obligations-v116.json'))
    for g in obligations['obligations']:
        if g['id']=='G10':
            g['remaining']=f"Central{cov['scoped_conditional_pairs']} conditional relations/{cov['higher_order_groups']} groups leave{cov['remaining_pair_relations']} pairs to reconcile;"+g['remaining'].split(';',1)[1]
        if g['id']in ['G01','G02','G05','G07','G09','G13']:
            g['sampling_visibility_v117']=deepcopy(detail)
    obligations.update(parent='coverage-current-obligations-v116.json',new_native_or_whole_closures=0,overall_complete=False)
    write(O/'coverage-current-obligations-v117.json',obligations)
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v116.json'))
    audit.update(previous_goal_turn='PROGRESS:v116 explicit MC2return guards and mask-gated Cast work;postcheckPASS735',input_checkpoint=116,followup_checkpoint=117)
    audit['sampling_visibility_v117']=deepcopy(detail)
    for r in audit['requirements']:
        if r['id']in ['MODEL_VERSION_HARDWARE_LOAD','ALL_MODULE_CONDITIONAL_BRANCHES','THEORY_AND_MEASURED_COMPARISON','CRITICAL_EVIDENCE_AND_FINAL_CLOSURE']:
            r.setdefault('evidence',[]).extend(names)
    write(O/'whole-scope-requirement-audit-v117.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v117.json',dict(
        status='72_GREEDY_AND_88_HASH_TASK_DESCRIPTORS_THEN_ONLY_ARGUMENT_REPORTING_CAP_AND_OPTIONAL_GUARDS',
        questions=read(O/names[0])['questions'],installed_Triton_package='3.6.0+dev20260923230212',
        source_scope='Exact installed driver, two helper sources and CANN profiler ABI headers; purelib metadata query uses stdlib only. No Triton/Torch/CANN module imported.',
        retrieval='Read-only docker archive files; one guessed runtime/rt.h path absent. All successful files SHA-pinned, existing duplicate acl_rt checked byte-identical during local extraction recovery. SSHclosed.',
        no_historical_loaded_ABI_or_flag_promotion=True,no_model_or_device_context_capture_replay=True,
        native_Pad_probe='PREPARED_NOT_EXECUTED;separate pending reply',overall_complete=False))
    write(O/'sampling-visibility-validation-history-v117.json',dict(events=[
        'Latest v116 independently revalidated PASS735 before current work; complete checkpoint not edited or finalizer rerun.',
        'Reused v65 three greedy geometries/72 executions and v114 hash geometries/88 executions. Presence of greedy kernel was already known not to imply a wholly greedy batch.',
        'One stdlib-only installed package/path metadata query and bounded exact docker archive reads. No target library imported, no NPU context/capture/replay, no model inference or target code/environment change.',
        'A reused local archive extraction assertion stopped on an already existing acl_rt control copy after the remote read completed. Preserved receipt, verified duplicate byte identity and extracted remaining files locally; no duplicate remote read for that receipt.',
        'Guessed installed experiment/runtime/runtime/rt.h path returned READ_ERROR; saved as absent, not synthesized. Direct prof_api→prof_common→aprof_pub chain supplies the actual5-entry/rank8 constants.',
        'Installed dataTypes generator handles first five pointer signatures, descriptor loop shares one cap across input/output and can emit two records for INPUT_OUTPUT. No historical driver/source/SO key bound to the72 calls.',
        'Three no-empty-shape geometries each admit four indistinguishable all-greedy/mixed × ordinary/synthetic descriptor prefixes. Twelve source-model witnesses refute absence inversion, not recover old flags.',
        'Hash first five inputs fit the same cap; source output is tensor argument14 and N/A is not no-write/no-reader/CPU proof. Existing value interfaces and epochs retained.',
        'Independent shared-budget expansion checks78 input/output boundary cases. PreflightPASS235. Counts and all14 obligations retained; no native/candidate/pair/branch completion promoted.',
        'SSHmaster explicitly closed. Separate v111 Pad44B probe remains prepared and unexecuted; original v109 one-shot authorization was already consumed.'],overall_complete=False))
    gap=('Installed profiler ABI permits only5 total tensor descriptors,shared by input and output. The greedy output+four inputs fill the packet before optional is_greedy BOOL/uniform/rates. '
        'All-greedy/mixed and ordinary/synthetic source-compatible calls are indistinguishable on these fields;missing input cannot proveNone/false/zero work. '
        'Historical loaded launcher/sourceSO key, per-call effectiveflags and request/metadataepochs remainunbound. Preservepre-returnRNG/entropy/error/raw/logprobreaders,Fill-1suffix,precision/aliases/state. '
        'No physicaltime lowerbound/MFU/criticalpathgain or wholeclosure established.')
    remaining=[g['id']+': '+g['title']+'. '+g['remaining']for g in obligations['obligations']]
    remaining.append('v117:installedsharedtensorcap5 makes72greedytaskdescriptors insufficienttoinvertall_greedy/SYNTHETIC_MODE;12source-modelwitnesses,88hashoutputN/Aexplainedconditionally. Actualhistoricalflags/launcherbindingstillopen.355semanticrows/26modules,134relations/26groups/461pendingunchanged;Pad44Bnotrun.')
    report='''本轮只追查一个影响算子删除结论的问题：已有 profile 的 greedy kernel 没有列出 is_greedy BOOL，能否据此判定全批次都是 greedy？结论是不能。

只读取得当前166原容器的 Triton Ascend 3.6.0+dev20260923230212 launcher 文本与 CANN9.2 profiler 头文件。aprof_pub.h 明确 MSPROF_GE_TENSOR_DATA_NUM=5、shape rank 上限8；driver 的上报循环把输入和输出放进同一个5条预算，INPUT_OUTPUT还可能占两条。它不是“最多5个输入、再加5个输出”。数据类型也只为前5个指针签名生成。shape收集会跳过 None、size失败或空shape，因此一般也不能把shape表当完整参数表。这里的反例不依赖空shape、DCE或潜在错位。

拒绝采样kernel的参数顺序是：output、cu_num_draft_tokens、draft_token_ids、target_argmax、bonus_token_ids，然后才是可选 is_greedy，后面还有synthetic uniform/rates。现有3类、72次greedy执行恰好记录一个INT32输出及前四个输入 INT32/INT32/INT64/INT32。按当前上报规则，前五项已经填满报文；全greedy与混合批次、普通与synthetic模式，四种源码兼容状态都能产生完全相同的这份描述。三种几何共12个等投影参考核对通过，不能从缺少BOOL推出None，也不能从缺少uniform/rates推出synthetic关闭。

这也解释了一项已有现象：88次Engram _hash_ids_kernel只记录input_ids/token_map/dead_mask/positions/block_table五个输入，输出为N/A。源码输出指针是第14个张量参数，前五项上报已饱和。N/A不能证明没有写hash IDs、没有下游读者或在CPU执行。v114的历史因子、48个INT32 ID接口、commit/rollback和参数epoch合同保持原有条件，本轮没有新增融合机会。

上述结论分两层：当前安装文件的5条上限有精确源码和SHA证据；它们与已有profile前缀一致，给出不能反推标志的具体反例。但当前launcher文本和SO没有被逐任务绑定到历史72次调用，CANN控制头SHA一致也不能替代这种绑定。因此不宣称已重建历史内部标志。现有固定客户端温度0意图证据继续保留，与实际逐调用元数据区分。

对融合和理论界的影响是具体的：F35随机/混合/synthetic分支继续保留条件；不能把未列出的参数当作不存在的读者、零随机副作用或可删除分支，更不能把条件预算当作本负载确定可节省的HBM字节。仍保留Fill的-1未接受后缀、提前返回之前的RNG/entropy/error/raw/logprob观察及数值、别名和跨迭代状态要求。已有纯ID和保留raw接口的必要工作模型不因上报截断而改变。本轮没有获得物理能力上界或新的数值时间下界、MFU、提速倍数。

真正能改变分支结论的是与原任务绑定的launcher/source或SO digest、编译specialization key/常量、all_greedy/all_random与请求/元数据epoch，以及可选指针和提前返回前副作用的正面证据。应先复用已保存的缓存或日志；无绑定的另一枚小算子探针不能证明旧模型调用的采样标志。此项限定不会阻止继续分析其他未展开的算子关系。

本轮仅stdlib路径/包元数据查询和精确只读文件复制，没有导入Triton/Torch/CANN、创建NPU上下文、capture、replay、模型推理或修改目标。一个猜测的runtime头路径不存在，已如实记录；一次本地提取遇到重复控制文件后停止，随后只在本地核对字节并提取余下文件。SSH已关闭。独立复核前置235项通过，另有78个共享输入/输出预算边界参考。v111的44B原生Pad探针仍未执行，等待其单独明确回复。

覆盖数保持355条语义分支、26个暂定模块、134条条件关系、26个联合组和461对待核对；9类/1352次原生执行关键字段、849缺PMU、16个未绑定phase图及3136项含糊归属仍待补齐。0个整模块关闭，整体目标继续进行；本轮报告成稿不代表全部优化点分析完成。
'''
    action=('继续原Goal，v117PROGRESS。先读sampling-visibility-summary/source-contract/profile-projection/consequences/current-obligations-v117；verify_sampling_visibility_v117须PASS。'
        'v116manifestPASS78files/postcheck735，v117preflightPASS235。仅已有72greedy+88hashTASK描述与installedlauncher的参数可见性问题；CANNaprof_pub固定cap5总IO/rank8，driver前5pointerdtype/共享tensorCount。'
        'Greedy前5是output+cu+draft+target+bonus，BOOLis_greedy第6，uniform/rates更后；3几何×allgreedy/mixed×ordinary/synthetic=12相同报文参考。缺BOOL不等于None/全greedy，缺RNG不等于syntheticoff；实际historicalflags仍UNKNOWN。'
        'Hashoutput张量第14、N/A不等于no-write/no-reader/CPU；复用v114，不新增fusion。当前installeddriver/SO未逐历史task绑定，aclSHA相同不能升级为loadedABI。'
        '精确stdlib/archives结束，SSHclosed，无Triton/Torch/CANNimport/NPUcontext/capture/replay/model或目标变更。v111Pad44Bprobe仍PREPARED_NOT_EXECUTED/PENDING_ASYNC_REPLY，原v109一次授权已用，不可重用。'
        '不要再泛查框架/MRO或把缺参数当flags。优先真正未展开operatorrelation/branch或现有saved且call-bound的关键操作数/flags/capacity证据；继续台账未分析范围。'
        '355/26semantic,134relations26groups461pending,813/1297/46graph,2073/285598conditional,9/1352pending,849PMU16phase3136ambiguity,0whole/noallMFU不变；14G00-G13义务全open。'
        'TaskACTIVE Loop004EVALUATING GoalACTIVE；不改已封存v117，不重跑finalizer。')
    inputs=[Path(r['path'])for r in read(O/names[0])['inputs']]
    persist_refinement(version=117,previous=116,title='Sampling operator branch evidence and five-entry profiler observability',
        summary=dict(filename='sampling-visibility-summary-v117.json',previous_turn_classification='PROGRESS:v116 MC2return guard/mask clauses;PASS735',
            greedy_executions_reused=72,hash_executions_reused=88,installed_shared_tensor_descriptor_cap=5,
            equivalent_flag_projection_witnesses=12,independent_shared_budget_cases=78,
            actual_runtime_flags_or_historical_loaded_launcher_bound=False,
            refined_candidates=1,refined_modules=1,new_pairs=0,new_groups=0,new_semantic_rows=0,
            branch_records=355,semantic_modules=26,relation_clauses=134,higher_order_groups=26,pending_central_pair_reconciliation=461,
            partial_graph_node_rows=813,partial_graph_edge_rows=1297,structural_pair_hints=46,
            scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
            new_native_promotions=0,new_device_work=False,implementation=False,native_test=False,
            native_Pad_probe='PREPARED_NOT_EXECUTED;separate pending reply',actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,
            remote_job_status='Exact CPU-only package/path and archive reads finished;SSHclosed. No target module import/device/model/context/capture/replay;Padprobeunexecuted.'),
        changes=['Pin current installed CANN five-entry total tensor reporting cap and operator argument order.',
            'Demonstrate12 indistinguishable all-greedy/mixed/ordinary/synthetic projections for three existinggreedy geometries;missingBOOL cannot select one actual tail graph.',
            'Retain historicalloadedABI/flag gates and all fusion/theory coverage counts;hashN/A is not absence of writes or consumers.'],
        candidate_updates={'F35':dict(review_v117=deepcopy(detail),conclusion_changing_gap=gap)},
        module_updates={'M17':dict(candidates=['F35'],finding='Five-entry profile descriptor saturation prevents inversion of all_greedy/SYNTHETIC_MODE from missing optional tensors;source-compatible positive flags remain needed before deleting or costing a branch.',evidence=names,gap=gap)},
        remaining=remaining,action=action,
        jobs=[dict(kind='profile',status='pass',description='Reuse72greedy and88hashTASK descriptors;pin exact installed operator-reporting source/header and negative inference limits without historicalABIpromotion.',artifacts=[O/n for n in names]+inputs),
            dict(kind='design-check',status='pass',description='Twelve equivalent flag projections and78independent shared-IO budget references;allruntimeguard/native/wholecoverage gates remainopen.',artifacts=[O/'sampling-visibility-preflight-v117.json',O/'sampling-visibility-validation-history-v117.json',O/'coverage-current-obligations-v117.json',O/'whole-scope-requirement-audit-v117.json'])],
        report=report,read_first=names+['coverage-current-obligations-v117.json'],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/n for n in ['analyze_sampling_visibility_v117.py','verify_sampling_visibility_v117.py','read_sampling_profile_paths166_v117.py',
            'read_sampling_launch_source166_v117.py','read_sampling_launch_headers166_v117.py','read_sampling_launch_profiler166_v117.py','read_sampling_launch_limit166_v117.py','persist_profile_refinement_v90.py']],
        executor_prefix='sampling-visibility',main_script=Path(__file__),source_read=True)

if __name__=='__main__':main()
