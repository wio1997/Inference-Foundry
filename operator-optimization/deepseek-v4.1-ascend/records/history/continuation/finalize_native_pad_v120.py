"""Persist the consumed one-shot Pad diagnostic and bounded source-compatible packet evidence."""
from pathlib import Path
from copy import deepcopy
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v109 import persist_refinement

def main():
    assert not(O/'REVIEW-CHECKPOINT-v120.json').exists()
    assert read(O/'native-pad-preflight-v120.json')['status']=='PASS'
    names=['native-pad-inputs-v120.json','native-pad-observations-v120.json','native-pad-packet-and-store-contract-v120.json','native-pad-consequences-v120.json']
    detail=dict(evidence=names,authorization='GRANTED_AND_CONSUMED',actual_captures=1,guard_only_stops=1,
        own_tasks=6,native_MemSet_GetParams_success=True,native_MemSet_argsSize=152,
        observed_inline_flowtable_offset=24,source_compatible_own_requested_zero_bytes=32,
        advertised_workspace_metadata_bytes=1536,actual_allocator_extent_or_HBM_measured=False,
        original_model_operand_gaps_closed=0,new_native_promotions=0,new_pairs=0,new_groups=0,whole_requirement_closed=False)
    for base in ['candidate-relation-clauses','candidate-relation-coverage','candidate-pair-navigation','candidate-operator-footprints','operator-analysis-coverage-ledger']:
        d=deepcopy(read(O/(base+'-v119.json')));d['authorized_native_Pad_probe_v120']=deepcopy(detail);write(O/(base+'-v120.json'),d)
    obligations=deepcopy(read(O/'coverage-current-obligations-v119.json'))
    for g in obligations['obligations']:
        if g['id']in ['G01','G03','G05','G07','G09','G13']:g['authorized_native_Pad_probe_v120']=deepcopy(detail)
        if g['id']=='G03':g['remaining']=('SevenhistoricalMemSet/1272executionsstillneedownprovider/packet/destination/length/overlap/workspace-rolebinding;twoBOOL24/80stillneedsemanticfield/allreaders. '
            'CompletedownPadprobevalidatesnativeMemSetGetParamsandoneboundedinline96Brecordinside152Bpacket,source-span32B;notthesehistoricaloperands.')
        if g['id']=='G13':g['remaining']=('SingleauthorizedPadprobecompleted;donotwaitforreplyorrepeatequivalentdeviceprobe. '
            'Oldstrict128Bpacketpackinghypothesisisrejectedby152Bcapture;observedpointer+24bounds96Bflowtable,theadditional24zero bytesremainuntyped. '
            'Needoriginalmodelargument/epoch/provider/readersandtypedcapacityevidence,notmoreuntargetedsource. v119V1observerjointcountsandv118Engrammaskdemandareprioranswers.')
    obligations.update(parent='coverage-current-obligations-v119.json',native_Pad_probe_authorization='GRANTED_AND_CONSUMED',new_native_or_whole_closures=0,overall_complete=False)
    write(O/'coverage-current-obligations-v120.json',obligations)
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v119.json'))
    audit.update(previous_goal_turn='PROGRESS:v119V1rowepilogue and4736rawobserverrecordjointcounts;postcheckPASS4776',input_checkpoint=119,followup_checkpoint=120)
    audit['authorized_native_Pad_probe_v120']=dict(**detail,new_device_work=True,model_loads=0,graph_replays=0)
    for r in audit['requirements']:
        if r['id']in ['NATIVE_OPERATOR_REGIMES','MODEL_VERSION_HARDWARE_LOAD','CRITICAL_EVIDENCE_AND_FINAL_CLOSURE']:
            r.setdefault('evidence',[]).extend(names)
    write(O/'whole-scope-requirement-audit-v120.json',audit)
    write(O/'native-pad-authorization-consumption-v120.json',dict(status='GRANTED_AND_CONSUMED',user_reply='允许本次 Pad 探针',
        approval_record='native-pad-authorization-v119.json',actual_captures=1,guard_only_stops=1,
        probe_source_unchanged_from_v111=True,model_loads=0,graph_replays=0,tensor_payload_D2H=0,opaque_device_argument_deref=False,
        equivalent_retry_or_full_model_diagnostic_authorized=False))
    write(O/'PROFILE-CODE-QUESTION-GATE-v120.json',dict(status='ONE_EXPLICITLY_AUTHORIZED_OWN_NATIVE_PAD_CAPTURE_COMPLETE',
        questions=['CanthenativeMemSetgetterreturnapacket?','Whatdoesboundedrenderedflowtableandsourceproveaboutthisownworkspace?','Doesitbindhistoricalmodelcalls?'],
        new_device_work=True,model_loads=0,graph_replays=0,target_source_modifications=0,
        authorization='GRANTED_AND_CONSUMED',no_original_model_correlation=True,overall_complete=False))
    write(O/'native-pad-validation-history-v120.json',dict(events=[
        'v119sealedandpostverifiedPASS4776beforetheauthorizeddevicework;completedcheckpointnotedited.',
        'UserexplicitlyapprovedonePad44Bprobe. Exactv111scriptSHAf48726...reviewedandretained;runwrapperusesdurableattemptreceiptsand45secondown-processlimit.',
        'FirstwrappermisparsedchipindexascardindexandstoppedatGUARD_ONLYbeforeCDLL/Torch/device,actualcapturecount0. Preserveoriginalwrapper/receipt;correctonlyPhyID0/1guardinversionb,do notclaimserverwasoccupied.',
        'FreshSDK/library/providerdiskSHAchecks,visibledevices,mapandcard0idle/noprocessguardpass. Oneactualcapturecompleted6.5447sprocesswall. ApplicationIO44Bplusruntimeoverhead;notkernelperformance.',
        'OwnstreamhasCast,Cast,MemSet,PadV3,Cast,NOTIFY. FiveKERNELGetParams0,oneDEFAULT107000. NativeMemSetselected1a686...block48,args152/opInfo48;PadV3f466...block1.',
        'RenderedMemSetargword1equalsreportedargs_address+24and96sourcebytesfitinside152B. Oldwholepacketprediction3ptr*8+opPara104=128isrejected;trailing24zerobytesremainUNKNOWN,notnamedpadding.',
        'Source-recordneed_core_num1/core_num48,8FLOAT32elements,1tailburst,gap0,defaultzero. Source-requestedstoreisworkspace+[0,32);noGPUdereference,tensorreadback,replay,PMUorphysicalHBMstoremeasurement.',
        'FirstworkspacepointermatchesPadword5,distinctfromownBOOLbuffers;pinnedPadmetadatahasoneFLOAT32zeroedworkspaceadvertised1536B/TIKsync_workspace. Actualallocatorcapacitynotqueried;1536metadataisnotmeasuredallocationorwrittenextent.',
        'Afterexitall16AICoreutil0and8cardsnoprocess,containerPID2032500unchanged;SSHclosed. Oneactualcaptureauthorizationconsumed,noequivalentretryorfullmodelauthorization.',
        'Preflight109checkspass. No historicalMemSet/BOOLroleorallreaderpromotion;9regimes/1352executionsand356branches/26modules/134relations26groups461pendingunchanged.'],overall_complete=False))
    gap=('CompletedcontrolledPadcaptureestablishesnativeMemSetGetParamsandboundedsource-compatibleflowtabledecode. '
        'Own152Bpacketcontains96Brecordatobserved+24,source-requested32Bzeroing;1536metadataworkspaceisnotactualallocation/HBMextent. '
        'Originalmodelprovider/packet/address/length/epoch/allreadersstillunbound;BOOL24semanticfieldnotrecovered. '
        'Old128Bwholepacketpredictionrejected,extra24bytesuntyped. Do notdeletehistoricalMemSet,shrinkworkspaceorclaimphysicalspeedupfromthisowncase. '
        'Oneactualcaptureauthorizationnowconsumed;model/noReplaylimitsmet.')
    remaining=[g['id']+': '+g['title']+'. '+g['remaining']for g in obligations['obligations']]
    remaining.append('v120Pad44Bdiagnosticcomplete,oneactualcapture/onepredeviceguardstop;nativeMemSetGetParamsargs152,observedinline+24/source96record→requested32Bvsmetadata1536.Original9/1352stillopen;authorizationconsumed,SSHclosed,noretryimplied.')
    report='''用户新批准的Pad最小探针已完成一次实际捕获。没有加载模型、replay图、读取张量内容、解引用opaque设备参数或修改既有模型任务。应用输入/输出为BOOL[20]→BOOL[24]、右侧补4个True，共44B；另有上下文、分配器和内部转换开销。

第一次wrapper的空闲表格解析把chip号当成卡号，检查在设备初始化前停止，实际捕获次数为0；该代码和记录已保留。修正为物理ID0/1检查后，重新核对容器、CANN头/库及MemSet/Pad metadata和binary SHA、可见设备映射和卡0空闲状态，使用完全相同的已批准探针脚本，完成一次捕获。进程墙钟约6.54秒，不是算子延迟。没有重启或启动原容器。结束后16个AICore利用率为0、8张卡均无运行进程，原容器PID2032500不变，SSH已关闭。

自有图有1条流、6个任务：输入Cast、常量Cast、原生MemSet、PadV3、输出Cast、NOTIFY。5个内核GetParams均成功，NOTIFY的DEFAULT查询返回107000。选中的MemSet_1a686...为48 blocks、argsSize152/opInfo48；PadV3_f466...为1 block。这次得到的是原生MemSet内核，区别于v109的byte/D32 API→MEMCPY路径，但没有把新图与原模型执行关联。debug dump的ts/dur只是捕获图展示坐标，不作实测时间。

参数包还纠正了旧假设。metadata的3个pointer槽和opParaSize104曾给出128B整包预测；实际GetParams和dump一致为152B，旧严格helper因此拒绝它。保留该失败，不能宣布旧打包ABI已验证。新包中argword1恰好为报告的args_address+24，唯一落在已渲染包内部且能容纳96B源码record的prefix指针，据此只对已经渲染在CPU上的96B解码。128B之后额外24个零字节的语义仍未知，不称padding或其它字段。

这个受控record给出need_core_num=1、core_num=48、last_elems=8、tail_burst=1、gap=0、FLOAT32零值。按对应源码条件，只有writer0提交workspace+[0,32)的零写，其余worker不进入该行写入分支。它是源码推导的请求范围，既没有replay验证存储内容，也不是PMU/HBM交易字节。launch48不能解释为48个writer；args152、record96也都不是填充长度。

MemSet首workspace输入指针与Pad参数中workspace位置的指针一致，且不同于已知应用BOOL输入/输出地址。选中Pad metadata声明一个FLOAT32零初始化workspace、size1536，匹配TIK sync_workspace的用途；该1536是metadata声明范围，不是本次已查询的allocator容量或已写字节。本次source请求32B，与1536声明容量必须区分，不能用静态声明加地址差推出分配重叠、越界或可缩小缓冲。完整native参数ABI、实际分配范围和物理store仍有边界。

融合判断也保持范围：原整个Pad/initializer可在保留BOOL前缀、True后缀、mask读者、别名、capture及错误/同步语义的条件下研究替换；不能只删除原MemSet或根据这个单block小图删除其它多block Pad的barrier初始化。更不能把该32B套到原1272次MemSet，或据此识别原80次BOOL[24]的语义字段。原7类MemSet及2类BOOL的9类/1352次执行缺口保持。

本轮前置109项核对通过；台账仍为356条语义记录/26个模块、134条件关系/26联合组/461对待核对，整模块关闭0。单次授权已完成并消耗，不再等待本次回复、不重复等价探针；完整模型采集/replay或额外NPU诊断没有由本次授权自动获得。其他未分析算子范围和关键证据继续推进，整体目标未完成。
'''
    action=('继续原Goal，v120PROGRESS；先读native-pad-summary/observations/packet-and-store-contract/consequences/current-obligations-v120，verify_native_pad_v120须PASS。'
        'v119PASS4776/manifest64后执行新用户Pad授权。一次actualcapture完成；firstwrapperchip/card误解析guardstop0次，保留v120失败后v120b仅修PhyIDguard，sameprobeSHAf48726，44BBOOL20→24valueTrue/noModel/noReplay。'
        'FreshCANN头62d127/lib1c06b/provider3pairsSHAs通过；6tasks:CastCastMemSetPadCastNotify，5kernelGetParams0/Notify107000，MemSet1a686block48args152/opInfo48,Padf466block1。'
        '旧strict3ptr24+opPara104=128整包hypothesis被152反证；不得放宽后称其验证。已渲染word1=args_address+24且96Brecordbounds内，need1/core48/last8Float32/tail1/gap0/zero，source-requestedworkspace[0,32)。Trailing24zero字节语义UNKNOWN。无GPUderef/张量D2H/实测store/PMU/HBM/time。'
        'MemSetfirstworkspaceptr=Padword5，非BOOLappinput/output；metadata workspace1536/TIKsync用途被owncase支持，但actualallocator未查，1536≠written32，不推aliasbug/缩buffer。'
        '6.54s是processwall，debugts/dur不是device时间。After16AICore0/8cardsnoprocess/containerPID2032500unchanged/SSHclosed，alljobsdone。'
        '授权GRANTED_AND_CONSUMED，禁止等价tinyprobe重跑，不能复用本次授权做模型/更多NPU采集；不再等待reply。'
        '原9/1352历史keyfields仍未恢复，no newnativepromotion；356/26semantic,134relations26groups461pending,813/1297/46graph,2073/285598conditional,849PMU16phase3136ambiguity0whole/noallMFU不变。'
        '继续其他真正未分析operator关系/条件分支，或原已保存且call-bound的provider/packet/epoch/readers/typedcapacity证据。v119原observerjointcounts已恢复，不要说仍只有margins；v116V2/v117reportcap/v118Engrammask是prioranswers。'
        'TaskACTIVE Loop004EVALUATING GoalACTIVE；不改封存v120或重跑finalizer。')
    inputs=[Path(r['path'])for r in read(O/names[0])['inputs']]
    persist_refinement(version=120,previous=119,title='Authorized nativePad capture and bounded MemSet workspace packet evidence',
        summary=dict(filename='native-pad-summary-v120.json',previous_turn_classification='PROGRESS:v119V1row/jointcounts;PASS4776',
            **detail,new_device_work=True,model_loads=0,graph_replays=0,refined_candidates=0,refined_modules=1,
            branch_records=356,semantic_modules=26,relation_clauses=134,higher_order_groups=26,pending_central_pair_reconciliation=461,
            partial_graph_node_rows=813,partial_graph_edge_rows=1297,structural_pair_hints=46,
            scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
            actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,implementation=False,native_performance_test=False,
            remote_job_status='OneapprovedownPadcapturefinished,guard-onlyfailurepreserved;afterallidle/noprocess/PIDunchanged,SSHclosed. Authorizationconsumed,noretrypending.'),
        changes=['Executeoneuser-authorizedownnativePadcapturewithfreshidle/container/CANN/providercontrols;preservepredeviceguardparserfailureandoneactualattempt.',
            'NativeMemSetGetParamssucceeds,args152;rejectold128wholepackinganddecodeonlyboundedrendered96Brecordatobserved+24,source-requested32Bversusmetadata1536.',
            'Preservehistorical9/1352operands/readers/capacitygaps;markauthorizationconsumedandnoequivalent/fullmodelrunimplied.'],
        candidate_updates={},module_updates={'M19':dict(candidates=[],finding='ControllednativePadexposesMemSetpacket/source-workspacecontractanditslimits;own32Bsource-spanisnotallhistoricalMemSetorBOOLfieldproof. OldwholepacketABIhypothesisrejected.',evidence=names,gap=gap)},
        remaining=remaining,action=action,
        jobs=[dict(kind='review',status='error',description='Initialcardguardmisparsedchipfieldandstoppedbeforedevice;captureattempt0,preservedthenonlyguardcorrected.',artifacts=[N/'native-pad-probe166-v120.json',N/'native-pad-submission-v120.json',N/'run_native_pad_probe166_v120.py']),
            dict(kind='profile',status='pass',description='OneexplicitlyapprovedownPadcapture,44BapplicationIO,noModel/noReplay;freshcontrolsandafterexitidle/PID/SSHverified.',artifacts=[O/n for n in names]+inputs),
            dict(kind='design-check',status='pass',description='Audit152Bpacket/bounded+24record/source32Bzeroingversusmetadata1536;preserveoldpackingrejectionandallhistoricalgaps.',artifacts=[O/'native-pad-preflight-v120.json',O/'native-pad-validation-history-v120.json',O/'native-pad-authorization-consumption-v120.json',O/'coverage-current-obligations-v120.json',O/'whole-scope-requirement-audit-v120.json'])],
        report=report,read_first=names+['coverage-current-obligations-v120.json','native-pad-authorization-consumption-v120.json'],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/'analyze_native_pad_v120.py',N/'verify_native_pad_v120.py',N/'run_native_pad_probe166_v120.py',N/'run_native_pad_probe166_v120b.py',N/'persist_profile_refinement_v109.py'],
        executor_prefix='native-pad',main_script=Path(__file__),source_read=False,new_device_work=True)

if __name__=='__main__':main()
