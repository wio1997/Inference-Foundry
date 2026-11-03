"""Seal a compact evidence/workflow delta, reusing unchanged operator ledgers."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import hashlib, json
from finalize_origin_coverage_v9 import N, O, R, W, T as TASK_DIR, L as LOOP_DIR, read, write, ctl
from analyze_evidence_batches_v155 import pin


def main():
    checkpoint_path=O/'REVIEW-CHECKPOINT-v155.json';assert not checkpoint_path.exists(),'seal once only'
    predecessor=read(O/'REVIEW-CHECKPOINT-v154.json');pre_manifest=read(O/'checkpoint-manifest-v154.json');preflight=read(O/'raw-evidence-batch-preflight-v155.json')
    assert preflight['status']=='PASS'and preflight['preflight']and not preflight['old_unchanged_finite_case_suites_reexecuted']
    # Validate predecessor before any shared pointer/state/publication mutation.
    for row in pre_manifest['artifacts']:assert pin(row['path'])==row,row['path']
    now=datetime.now(timezone.utc);stamp=now.isoformat()
    protocol=O/'coverage-compression-protocol-v155.json';queue=O/'coverage-work-queue-v155.json';availability=O/'raw-field-availability-and-reuse-v155.json';hardware=O/'archived-hardware-pool-contract-v155.json'
    remaining=deepcopy(predecessor['pending_full_scope_domains'])
    remaining.append('G155: Legacy383unreconciled rows remain UNKNOWN coverage,not383individualpairtasks. Complete named rawstream census establishes unavailable input/address fields,never independent buffers/noopportunity. Reuse shared field/mode/resource/observer proofs with exact premises and delta checks. Full26module358row inventory still nonexhaustive;actual9/1352,849PMU,16phase,3136assignments,G14copy/control and physicalMFU/lowest/broader gaps remain open.')
    action=('继续原整体目标与Loop004，优先读coverage-work-queue-v155.json、raw-field-availability-and-reuse-v155.json和coverage-compression-protocol-v155.json。'
            '以具体数据流/状态/通信/资源切面的问题为工作单元，共用约束按联合图计一次，复用已验证证明仅核对前提差异；不再按候选星形/全配对或重复有限用例推进。'
            '原383项保留UNKNOWN覆盖索引，不是383个逐项任务，空白不能判独立。'
            '32原始Torch完整流无5/6/7字段、64HCCL流地址全0、80tensor流只含声明格式/类型/shape，除新增文件/采集/格式或反证外，不为每个配对重扫。'
            '从已有profile绑定具体物化切面及全部值/别名/代次/外部读者/分支，只有能改变算子融合或上限结论时定点查码或另一类保存资料。'
            '24AIC/48AIV与197份8板16芯片存档映射是元数据，runtime逻辑device到物理chip/board、实际执行精度/容量上界仍未闭合；1800报告频率不是fmax。'
            '所有26模块/358语义行及跨层/卡/迭代范围保留；9类1352次、849PMU、16phase、3136归属及G14/全部broader gaps不变；整模块关闭0，GoalACTIVE。'
            '本轮155采用DELTA_CHECKPOINT_V1，未变更的候选/模块/分支/关系权威复用154路径，后续从checkpoint字段取路径，勿假设每个版本都有整份新ledger。'
            '无新NPU/模型/capture/replay授权；只读历史资料与停止容器文件系统，不启动容器。')
    base_ob=read(O/'coverage-current-obligations-v154.json')
    obligations=[]
    for row in base_ob['obligations']:
        item={k:deepcopy(row[k])for k in['id','title','remaining','requirements','closed']if k in row}
        assert not item.get('closed',False)
        item['inherited_full_record']=str(O/'coverage-current-obligations-v154.json')+'#'+row['id']
        item['evidence_delta_v155']=dict(availability=str(availability),hardware=str(hardware),work_queue=str(queue),no_original_requirement_closed=True,missing_fields_do_not_prove_independence=True)
        obligations.append(item)
    write(O/'coverage-current-obligations-v155.json',dict(schema='OVERLAY_AND_INHERITED_OBLIGATIONS_V1',parent=pin(O/'coverage-current-obligations-v154.json'),obligations=obligations,inherited_records_retained=True,overlay_rule='Annotate only;all broad original remaining text and historical proof records retain their force. Dispatch is superseded by155questionbatches,notpaircounters.',legacy_pairs_are_navigation_not_tasks=True,new_native_or_whole_closures=0,overall_complete=False))
    write(O/'module-coverage-gaps-v155.json',dict(generated_utc=stamp,parent=pin(Path(predecessor['coverage_gaps'])),remaining=remaining,work_queue=str(queue),whole_modules_closed=0,overall_complete=False))
    audit=read(O/'whole-scope-requirement-audit-v154.json')
    write(O/'whole-scope-requirement-audit-v155.json',dict(schema='OVERLAY_AND_INHERITED_REQUIREMENTS_V1',parent=pin(O/'whole-scope-requirement-audit-v154.json'),requirements=[dict(id=x['id'],state=x['state'],base_requirement=str(O/'whole-scope-requirement-audit-v154.json')+'#'+x['id'],delta='No completion promotion;155evidence availability and compressed routing apply.')for x in audit['requirements']],complete_original_scope_preserved=True,legacy_pair_counts_are_not_completion_metric=True,new_native_or_whole_closures=0,whole_modules_closed=0,overall_complete=False))
    write(O/'PROFILE-CODE-QUESTION-GATE-v155.json',dict(status='PROFILE_CRITICAL_FIELD_AVAILABILITY_AND_OPERATOR_CUT_BATCHES',questions=['Canexistingarchivedrawstreamsrecoverconclusion-changingaddress/length/workspace/inputidentityfieldsomittedfromnormalexports?','Whichrecordeddevice/core/frequency/chip-boardfactsarevalid,andwhichphysical/math-modebridgesremainunknown?','Howcanproofreuseandgraphcutbatchingeliminateunrelatedpairworkwithoutdiscardingunknownscope?'],source_scope='Only exactstoppedcaptureimage TorchTLV/CANNrecord format,bean,decoder andfilename routing files;no target framework/module research. Complete archived rawTLV census and local cached rawrecords.',record_format_source_read=True,target_operator_source_read=False,no_model_operator_implementation=True,no_remote_Torch_CANN_NPU_model_or_replay=True,no_container_start_exec_or_attach=True,Pad_authorization='GRANTED_AND_CONSUMED',overall_complete=False))
    report='''已按用户要求压缩分析流程：以具体数据流、物化接口、状态、通信和资源切面的问题为工作单元。相同前提的证明与字段审计只做一次，新场景只核对差异；全局共享资源作为联合约束处理。旧配对表继续保留未知范围，不再逐对派工，也不作为完成率。空白字段不能证明独立或没有融合机会。

本轮完整扫描全部32份原始Torch记录，3,214,148,270字节、1,448,456条；没有记录输入tensor、tensor list、scalar三个TLV字段。匹配9.2.0采集镜像中的格式文件表明，普通导出虽可能跳过这些字段，本次原始采集也确实未保存。另64份HCCL原始文件共11,864条，源和目标地址字段全部为0，只能视为不可用元数据。80份tensor信息共297,976条的声明字段是输入/输出、格式、类型和shape，不含地址、stride或workspace；保留字节没有被猜解为操作数。物理文件清单也没有匹配的runtime memcpy_info流。以上均为指定存档及格式的有范围结论，不能推导真实buffer互不相交、任务零工作或所有证据渠道均不存在。

这些结果已写入共同的字段可用性证据，后续不为每个候选重复搜查同一批文件。旧图的每条来源边只归入一个producer导航切面；分组不证明同一次调用、数值相同、分支共执行或融合合法，更不生成新的逐组派工清单。实际工作队列只有图切面定点绑定、物理资源映射和共享操作数缺口三类问题。既有证明复用其哈希和前提，没有重跑未变更的大量有限用例。

32份保存的device信息报告CANN9.2.0、24AIC/48AIV。197份历史npu-smi样本稳定显示8个板号、16个芯片及各自PCI标识，但runtime逻辑device与物理芯片/板号仍缺显式桥接，报告频率也不是认证fmax。宿主机的8.5解码器没有被用于9.2采集；格式资料只从原采集的停止容器中只读提取，没有启动或exec容器。

整体目标仍进行中：26模块、358语义行的清单未证明穷尽；9类1352次原生操作数、849项PMU、16个phase图、3136项归属及G14拷贝/控制等原有缺口保留。没有新增独立性证明、原生融合/别名绑定、整模块关闭或物理MFU/最低利用率结论，也没有新增设备、模型、capture或replay工作。

本轮封存的是证据和流程增量。候选、模块、分支和关系的未变更权威直接复用v154文件，由v155 checkpoint中的实际路径引用；不再为了版本号复制整套台账。字段完整性和工作路由已独立校验，原先的本地打包/兼容性失败作为失败记录保留，不记作native算子失败。
'''
    report_path=O/'PROFILE-FINDINGS-v155.md';report_path.write_text(report)
    summary=read(O/'raw-evidence-batch-summary-v155.json');summary.update(generated_utc=stamp,previous_turn_classification='PROGRESS:v154operatorcontractssealed;v155sharedcriticalevidenceandworkflowcompression,zero newpairproofs',remaining=remaining,next_action=action,remote_job_status='All155readonlyraw/schema/cache/censusjobs terminal;no live device/model/replay job. No fresh server health or occupancy conclusion.',unchanged_operator_authorities_reused=True,operator_authority_baseline_checkpoint=154,checkpoint_schema='DELTA_CHECKPOINT_V1');write(O/'raw-evidence-batch-summary-v155.json',summary)
    jobs=[dict(kind='design-check',status='error',description='Retain actual155local syntax/filecount/socket-idle guards and remotePython3.9read-only tar packaging incompatibility;corrected fetch passed. These are not native/device failures.',artifacts=[O/'raw-evidence-local-corrections-v155.json',N/'raw-field-cache-receipt-v155.json',N/'cache_saved_raw_evidence_v155a_failed.py']),dict(kind='profile',status='pass',description='Read-only original166->105/root archive inventory/cache and exactstoppedCANN9.2/Torchschemas;full32TorchTLVcensus and64HCCL/80tensorfields with scopedavailability;no new nativeoperand bindings.',artifacts=[O/'raw-evidence-batch-inputs-v155.json',O/'raw-torch-field-census-v155.json',O/'raw-cann-field-census-v155.json',availability,hardware,N/'original166-xinyufeng-recheck-v155b.json',N/'raw-field-cache-receipt-v155b.json']),dict(kind='design-check',status='pass',description='Independent155delta preflight passed. Reuse oldproofhashes,all595navigation rows forbid individualpairdispatch/empty-independence. Oldfinitecase suites not rerun;fullscope and3questionbatches retained.',artifacts=[O/'raw-evidence-batch-preflight-v155.json',protocol,queue,O/'coverage-empty-evidence-audit-v155.json',O/'source-cut-navigation-v155.json'])]
    registered=[]
    for index,job in enumerate(jobs):
        rid='run-'+(now-timedelta(seconds=len(jobs)-index)).strftime('%Y%m%dT%H%M%SZ')
        run=ctl('new-run','--loop-dir',str(LOOP_DIR),'--run-id',rid,'--case-key','c8-input32k-output1k-prefix90-166','--kind',job['kind'],'--command','Prepared155retrospectiveregistration: '+job['description'],'--code-ref','Archivedprofile/recordformatonly;target model/operators unchanged;noNPU/context/replay','--working-directory',str(N))
        args=['finish-run','--run-dir',run,'--status',job['status'],'--correctness','not-applicable','--summary',job['description']]
        for p in job['artifacts']:assert p.exists();args+=['--artifact',str(p)]
        ctl(*args);registered.append(dict(run_dir=run,status=job['status'],description=job['description']))
    cp=deepcopy(predecessor);cp.update(checkpoint_schema='DELTA_CHECKPOINT_V1',previous_checkpoint=str(O/'REVIEW-CHECKPOINT-v154.json'),status='PROGRESS',previous_turn_classification=summary['previous_turn_classification'],current_verified_changes=['ReplaceCartesianpair/star dispatch with3questionbatches and7scopedprooffamilies;unknowncoveragepreserved,emptyfieldsneverindependence.','Complete archived32TorchTLV/64HCCL/80tensorfieldcensus;reuse source-hash-bounded unavailability result across candidates.','Bind32device-info and197archivedchip-board metadata with explicit physicalmapping/clock/mode limits.','Reuse unchanged154operatorledger paths and existingproofhashes;no unchangedfinitecase reruns/newpair/native/wholeproofs.'],summary=str(O/'raw-evidence-batch-summary-v155.json'),report=str(report_path),source_gate=str(O/'PROFILE-CODE-QUESTION-GATE-v155.json'),coverage_gaps=str(O/'module-coverage-gaps-v155.json'),coverage_obligations=str(O/'coverage-current-obligations-v155.json'),requirement_audit=str(O/'whole-scope-requirement-audit-v155.json'),pending_full_scope_domains=remaining,next_action=action,remote_job_status=summary['remote_job_status'],updated_utc=stamp,unchanged_operator_authorities_reused=True,operator_authority_baseline_checkpoint=154,evidence_overlays=[str(availability),str(hardware)],analysis_workflow=str(protocol),work_queue=str(queue),relation_coverage=str(O/'candidate-relation-coverage-v154.json'),whole_modules_closed=0,overall_complete=False)
    write(checkpoint_path,cp)
    pointer=read(N/'CURRENT-ANALYSIS-WORKFLOW.json');pointer.update(effective_checkpoint=155,work_queue=pin(queue),previous_shared_state_validated_before_authority_advance=True,prior_operator_evidence_unchanged=True);write(N/'CURRENT-ANALYSIS-WORKFLOW.json',pointer)
    shared=[]
    for path in[N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
        data=read(path);data.update(latest_profile_first_review=cp,overall_complete=False,whole_modules_closed=0,updated_utc=stamp);data['next_read_only_action'if path==R/'STATE.json'else'next_action']=action
        if path==W/'STATE.json':data.update(profile_candidate_ledger=cp['candidate_ledger'],profile_module_ledger=cp['module_ledger'])
        if path in[N/'RECOVERY.json',R/'PARALLEL-RESUME-PACK.json']:data['read_first']=[cp['summary'],str(protocol),str(queue),str(availability),str(hardware),cp['coverage_obligations']]
        write(path,data);shared.append(path)
    for path in[LOOP_DIR/'state.json',TASK_DIR/'index.json']:
        data=read(path);assert data['status']in['EVALUATING','ACTIVE'];data.update(next_action=action,updated_at=stamp);write(path,data);shared.append(path)
    for path in[LOOP_DIR/'status.md',TASK_DIR/'status.md']:
        path.write_text('\n'.join(('- 下一步：'+action)if line.startswith('- 下一步：')else('- 更新时间：`'+stamp+'`')if line.startswith('- 更新时间：')else line for line in path.read_text().splitlines())+'\n');shared.append(path)
    pub=R/'publication-draft/FUSION-INVENTORY.md';(O/'FUSION-INVENTORY-before-v155.md').write_bytes(pub.read_bytes());pub.write_text(pub.read_text()+'\n\nv155证据/流程增量：'+str(report_path)+'。未变更的算子台账继续引用v154；当前整体权威 '+str(checkpoint_path)+'。以算子切面问题批量补证，旧配对数不作任务数或完成率；空白不判独立，整模块仍open。\n');shared.append(pub)
    profile=read(R/'profile-summary.json');profile['supplementary_v155']=dict(summary=cp['summary'],analysis_workflow=str(protocol),work_queue=str(queue),operator_authority_baseline_checkpoint=154,unchanged_operator_authorities_reused=True,whole_modules_closed=0,overall_complete=False);write(R/'profile-summary.json',profile);shared.append(R/'profile-summary.json')
    compact=ctl('compact','--task-dir',str(TASK_DIR));validation=ctl('validate','--task-dir',str(TASK_DIR));write(O/'task-run-registration-v155.json',dict(runs=registered,compact_output=compact,validation_output=validation,whole_modules_closed=0,overall_complete=False))
    paths={Path(x['path'])for x in pre_manifest['artifacts']}
    paths.update(p for p in O.glob('*v155*')if p.is_file()and not p.name.startswith('checkpoint-manifest')and'postcheck'not in p.name)
    paths.update(Path(x['path'])for x in read(O/'raw-evidence-batch-inputs-v155.json')['inputs'])
    for job in jobs:paths.update(job['artifacts'])
    paths.update(N/name for name in['build_coverage_compression_v155.py','inspect_saved_raw_profile_v155.py','cache_saved_raw_evidence_v155.py','read_profile_record_schemas_v155.py','scan_raw_torch_tags_v155.py','read_cann_field_schemas_v155.py','read_cann_field_formats_v155.py','analyze_evidence_batches_v155.py','verify_evidence_batches_v155.py','finalize_evidence_batches_v155.py','CURRENT-ANALYSIS-WORKFLOW.json'])
    paths.update(shared)
    write(O/'checkpoint-manifest-v155.json',dict(schema='DELTA_CHECKPOINT_V1',generated_utc=stamp,predecessor_manifest=pin(O/'checkpoint-manifest-v154.json'),predecessor_current_state_validated_before_shared_writes=True,artifacts=[pin(p)for p in sorted(paths)],old_unchanged_proofs_reused=True,whole_modules_closed=0,overall_complete=False))
    print(json.dumps(dict(status='PROGRESS',checkpoint=str(checkpoint_path),manifest_files=len(paths),runs=len(registered),retained_error_runs=1,unchanged_operator_authorities_reused=True,old_finite_case_suites_reexecuted=False,validation=validation,whole_modules_closed=0,overall_complete=False),ensure_ascii=False))


if __name__=='__main__':main()
