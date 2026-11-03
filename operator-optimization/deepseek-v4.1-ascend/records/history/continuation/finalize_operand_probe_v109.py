"""Register the bounded user-authorized diagnostic and its exact evidence limits."""
from pathlib import Path
from copy import deepcopy
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v109 import persist_refinement

def main():
    assert not(O/'REVIEW-CHECKPOINT-v109.json').exists()
    assert read(O/'operand-probe-preflight-v109.json')['status']=='PASS'
    names=['operand-probe-inputs-v109.json','operand-probe-observations-v109.json','operand-probe-capability-gates-v109.json']
    for base in ['candidate-relation-clauses','candidate-relation-coverage','candidate-operator-footprints','candidate-pair-navigation','operator-analysis-coverage-ledger']:
        d=deepcopy(read(O/(base+'-v108.json')));d['authorized_operand_probe_v109']=dict(evidence=names,
            own_graph_tasks=4,own_kernel_GetParams_success=1,DEFAULT_GetParams_107000=3,
            new_saved_model_attributions=0,new_pairs=0,prior_pending_probe_authorization_resolved=True)
        write(O/(base+'-v109.json'),d)
    obligations=deepcopy(read(O/'coverage-current-obligations-v108.json'))
    for r in obligations['obligations']:
        if r['id']in ['G01','G03','G04','G09']:
            r['authorized_operand_probe_v109']=dict(evidence=names,own_controlled_graph_only=True,
                original_model_operands_recovered=False,probe_authorization='GRANTED_AND_CONSUMED',
                no_equivalent_probe_retry_needed=True)
    obligations.update(parent='coverage-current-obligations-v108.json',new_native_or_whole_closures=0,overall_complete=False)
    write(O/'coverage-current-obligations-v109.json',obligations)
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v108.json'))
    audit.update(previous_goal_turn='PROGRESS:v108 scoped MinTokens/force/argmax fusion and symbolic bounds;postcheck PASS149',input_checkpoint=108,followup_checkpoint=109)
    audit['authorized_operand_probe_v109']=dict(evidence=names,user_granted_minimal_probe=True,
        predevice_output_failure_retained=True,one_actual_capture_attempt=True,new_device_work=True,
        loaded_model=False,model_graph_replays=0,original_model_operand_gaps_closed=0,whole_requirement_closed=False)
    write(O/'whole-scope-requirement-audit-v109.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v109.json',dict(status='EXACT_APPROVED_TINY_GRAPH_CAPTURE_ONLY',
        questions=['Does the exact installed graph getter/dump expose typed fill ranges and BOOL semantic fields?',
            'Can controlled API-fill task types be identified with historical native MemSet?'],
        newly_read_SDK_error_header=1,full_framework_or_model_source_read=False,new_device_work=True,
        new_model_load_or_replay=False,authorization='User explicit allows minimal probe;now consumed',
        followup_not_implied='Full-model loading/replay/inference is outside this narrow authorization.',overall_complete=False))
    remaining=[r['id']+': '+r['title']+'. '+r['remaining']for r in obligations['obligations']]
    remaining.append('Current obligations-v109;minimal probe completed under explicit authorization. One kernel metadata packet succeeded;two controlled fills are MEMCPY_ASYNC and return107000,no ranges. This does not bind the original7MemSet+2BOOL24 roles. No pending authorization question for the already finished probe;no equivalent retry or full-model run implied.')
    gap=('Controlled own graph metadata is now measured:4tasks,1ZerosLikeKERNEL GetParams success(args64/opInfo136/isHostArgs0),3DEFAULT107000. '
        'BothAPI fills becomeMEMCPY_ASYNC;not historicalMemSet equivalence. No typednativefillrange/semanticBOOLfield/allreaders/historicalcall/storageepoch orMC2counts recovered. '
        'Do not dereference opaque deviceargs or infer lengths fromargsSize/repeated24. Fullmodeldiagnostic notauthorized byminimalprobe. Allnine savedregimes remain open.')
    mods={m:dict(candidates=[],finding='One user-authorized own-graph probe establishes generic getter/dump capability and its limits;no historical model operand attribution. Controlled byte/D32 fills are MEMCPY_ASYNC with invalid-parameter getter results;BOOL24 exposes an opaque kernel packet only.',evidence=names,gap=gap)for m in ['M05','M12','M19','M21']}
    report='''最小探针已获得用户明确授权并完成；没有加载模型、replay图、发送推理或修改既有任务。

开始前检查原166（实际host105）的16个设备AICore均为0、8张物理卡均无运行进程，原容器running/PID2032500。第一次Python提交在创建/data/tiankuan/wio输出目录时因只读挂载退出；尚未加载NPU库或导入Torch。保留该失败后，只把输出限制改为自有可写/tmp/wio-operand-probe-v109-*目录，其余探针代码保持一致，完成了一次实际NPU捕获尝试。未启动、重启或停止原容器。

成功探针耗时约6.55秒（进程墙钟，不是算子延迟），自有应用张量8240B，另有驱动/上下文/分配器开销。安装头文件SHA与历史控制一致；当前磁盘libascendcl.so解析到cann-9.2.0并记录SHA。这些仅是新受控诊断的环境证据，不证明历史模型进程的加载ABI。

自有图共4个任务、1条流：两次受控提交aclrtMemsetAsync(count=64B)和aclrtMemsetD32Async(N=16)在dump中均呈现MEMCPY_ASYNC；随后为ZerosLike内核和NOTIFY_RECORD。3个DEFAULT任务的GetParams均返回107000，精确安装头文件定义为ACL_ERROR_RT_PARAM_INVALID，没有返回填充范围。该返回码未给出根因，不能据此认定所有DEFAULT任务永久不支持，更不能把受控MEMCPY路径等同于原profile的ConstantPadNd-owned MemSet。

BOOL[24] ZerosLike的GetParams成功，报告argsSize=64、opInfoSize=136、isHostArgs=0、numBlocks=1。dump渲染8个uint64 word，其中已知输出地址出现两次、数值24出现三次、已知输入地址未出现。它证明可以取得这一个内核的通用包信息；重复地址/数值并不解释每个字段的ABI，也不识别原模型中BOOL[24]的语义字段或全部读者。没有直接解引用opaque参数或读取张量内容。

图未replay。dump中的ts/dur不用作设备实测时间；argsSize64也不能当作实际填充64字节，和两条受控fill恰好都写64B无关。没有计算物理带宽、MFU或提速。

退出后8张卡仍无NPU运行进程，原容器PID不变，SSH控制连接已关闭。两份小型原始输出及失败/资源/源码记录均已保存。本轮把“接口能力未知”推进到上述受控实测，但原7类MemSet/1272次和2类BOOL[24]/80次执行仍缺字段；MC2同call计数及其他关键证据未获得。不能因此批准删除/融合这些算子或缩小缓冲。

不再重复等价的小图探针。后续必须绑定原生provider/ABI和同call/图/更新/存储epoch；完整模型诊断会加载或replay模型，超出本次最小授权，需要另外准备具体有限计划。其他工作继续围绕算子融合与理论界，已有profile和定点源码分析不受影响。

35项模型候选、353条语义记录/26暂定模块、132条条件关系/25组/463对待核对以及9类/1352次执行缺口保持不变。整模块关闭数仍为0；全模块最低MFU及整体物理理论上限尚未建立。
'''
    action=('继续原Goal,v109PROGRESS：先读operand-probe-summary/observations/capability-gates及current-obligations-v109。v108PASS149。'
        '用户已明确允许最小probe，已完成且授权不再pending。原container2032500running，16设备util0/8卡noprocessguard。'
        'v109首进程因/data只读mkdir失败，发生在CDLL/Torch前；v109b只改own/tmp输出root，实际一次NPUcapture，无model/noReplay。'
        '4tasks同流seq0/1MEMCPY_ASYNC、2ZerosLikeKERNEL、3NOTIFY。3DEFAULT GetParams107000，安装errorheader映射PARAM_INVALID，根因未知非所有DEFAULTunsupported证明。'
        'BOOL24GetParams成功args64/opInfo136/isHostArgs0/blocks1；debug8words输出ptr重复2、24重复3、输入ptr无；无ABIfield/allreaders/historicalmodelcorrelation。Controlledfill≠savedMemSet。'
        '磁盘CANN9.2libsha1c06b...已记录非historicloadedABI；debug坐标不是时间，6.55s是进程wall非op。原9/1352全部remain，MC2counts未恢复。'
        'after8卡noprocess/containerPIDunchanged/SSHclosed。无livejob，不重跑equivalent tinyprobe；完整model诊断需另有具体plan/授权，不能从最小授权推断。'
        '下一步继续真正operator分析：已定点读cachedpenalty/bincount主体但尚未形成持久结论，v108M17OP9具体MinTokens已解范围不要重复。'
        'penalty可审prompt只需presence、output需count并castFLOAT，mask(count>0)可与consumer组合；spec histories=base+precedingdraftprefix，可按request共享，注意zeroDraft/ragged/rollback/slotepochs/I32overflow/rounding及新representation成本。先核prioranswers，再补证。'
        '35/353/26、132/25/463、813/1297/46、2073/285598conditional、9/1352pending849PMU16phase3136ambiguity0whole/noallMFU不变。'
        'GoalACTIVE TaskACTIVE Loop004EVALUATING；verify_operand_probe_v109必须PASS；不重跑completedfinalizers/改完成artifact。')
    inputs=[Path(r['path'])for r in read(O/names[0])['inputs']]
    persist_refinement(version=109,previous=108,title='Authorized own-graph operand metadata capability probe',
        summary=dict(filename='operand-probe-summary-v109.json',previous_turn_classification='PROGRESS:v108 MinTokens/force/argmax contract and symbolic bounds;postcheck PASS149',
            own_capture_attempts=1,predevice_failed_process_attempts=1,captured_tasks=4,successful_kernel_params=1,DEFAULT_params_107000=3,
            probe_status='COMPLETED_NO_REPLAY',probe_authorization='GRANTED_AND_CONSUMED',new_device_work=True,model_loaded=False,model_graph_replays=0,
            new_saved_model_native_promotions=0,new_pairs=0,new_groups=0,branch_records=353,semantic_modules=26,
            relation_clauses=132,higher_order_groups=25,pending_central_pair_reconciliation=463,
            partial_graph_node_rows=813,partial_graph_edge_rows=1297,structural_pair_hints=46,
            scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
            actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,implementation=False,native_test=True,native_test_scope='Owned tiny graph metadata capability only',
            remote_job_status='Authorized tiny capture terminal;afterward8physicalcards have noNPUprocesses,originalcontainerPID2032500unchanged,SSHcontrolclosed. No model load/replay or service changes.'),
        changes=['Execute the approved minimal metadata probe after idle-device guard;retain initial pre-device read-only-output failure.',
            'Establish one KERNEL GetParams success and three DEFAULT107000 observations;controlled fills map to MEMCPY_ASYNC.',
            'Keep original MemSet/BOOL and MC2 gaps open;packet/dump metadata do not provide historical semantic identity or physical timing.'],
        candidate_updates={},module_updates=mods,remaining=remaining,action=action,
        jobs=[dict(kind='profile',status='error',description='First Python attempt stopped at read-only output-directory mkdir before library/Torch/NPU initialization;retained and corrected only output root.',artifacts=[N/'operand-probe166-v109.json',N/'operand-probe-output-path-check-v109.json',N/'operand-probe-output-recovery-v109.json']),
            dict(kind='profile',status='pass',description='User-authorized one own-graph capture on idle device0;4task metadata/no replay;one kernel generic packet,threeDEFAULT107000;no saved-model operands recovered.',artifacts=[O/n for n in names]+inputs),
            dict(kind='design-check',status='pass',description='Independent CPU checks of source-only output recovery,resourceguards,typed returncodes/opaque-packet limits and preserved whole-scope gaps.',artifacts=[O/'operand-probe-preflight-v109.json',O/'whole-scope-requirement-audit-v109.json',O/'coverage-current-obligations-v109.json'])],
        report=report,read_first=names+['coverage-current-obligations-v109.json'],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/'analyze_operand_probe_v109.py',N/'verify_operand_probe_v109.py',N/'persist_profile_refinement_v109.py'],
        executor_prefix='operand-probe',main_script=Path(__file__),source_read=True,new_device_work=True)

if __name__=='__main__':main()
