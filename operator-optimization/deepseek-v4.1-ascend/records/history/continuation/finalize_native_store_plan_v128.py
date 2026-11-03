"""Persist a conditional native disk scalar certificate, not a hardware result."""
from copy import deepcopy
from pathlib import Path
from finalize_origin_coverage_v9 import N,O,read,write
from persist_profile_refinement_v90 import persist_refinement


def main():
    assert not (O/'REVIEW-CHECKPOINT-v128.json').exists()
    pre=read(O/'native-store-preflight-v128.json');assert pre['status']=='PASS'and pre['checks']==8573
    names=['native-store-inputs-v128.json','native-static-disassembly-index-v128.json','native-store-ELF-controls-v128.json',
        'native-store-own-scalar-traces-v128.json','native-store-synthetic-control-validation-v128.json','native-store-fusion-consequences-v128.json']
    detail=dict(evidence=names,native_disk_objects=3,native_instruction_words_checked=8081,
        saved_own_block_certificates=48,own_output_store_sites=1,own_source_matched_request_bytes=32,
        own_MemSet_CCE_KernelArgSize_field=16,own_saved_packet_bytes=152,source_record_bytes=96,
        synthetic_records=147,synthetic_block_traces=7056,actual_loaded_instruction_bytes_attested=False,
        scalar_control_only=True,actual_native_execution=False,physical_HBM_or_time_measured=False,
        original_pending_operands_recovered=0,new_native_promotions=0,new_fusion_findings=0,
        new_pairs=0,new_groups=0,new_device_work=False,whole_requirement_closed=False)
    for base in ['candidate-relation-clauses','candidate-relation-coverage','candidate-pair-navigation','candidate-operator-footprints','operator-analysis-coverage-ledger']:
        obj=deepcopy(read(O/(base+'-v127.json')));obj['native_store_scalar_plan_v128']=deepcopy(detail)
        if base!='operator-analysis-coverage-ledger':obj['parent']=base+'-v127.json'
        if base=='candidate-relation-clauses':obj.update(parent_relation_index=base+'-v127.json',new_pair_clauses=0,new_higher_order_groups=0,overall_complete=False)
        if base in ['candidate-relation-coverage','candidate-pair-navigation']:obj['parent_relation_index']='candidate-relation-clauses-v128.json'
        if base=='candidate-pair-navigation':obj['parent_footprints']='candidate-operator-footprints-v128.json'
        write(O/(base+'-v128.json'),obj)
    obligations=deepcopy(read(O/'coverage-current-obligations-v127.json'))
    for row in obligations['obligations']:
        if row['id']in ['G01','G03','G05','G09','G11','G13']:row['native_store_scalar_plan_v128']=deepcopy(detail)
        if row['id']=='G03':row['remaining']=(
            'Pinned disk MemSet entry reads firsttwo64bit addresses;its CCEKernelArgSize field16 is distinct from own-v120 packet152,record96,workspace declaration1536 andsource request32. '
            'For the saved own zero record,scalar derivation over48blocks reaches one output site onblock0;no hardware replay or loaded instruction SHA attestation occurred. '
            'Original1272MemSet address/length/workspace/allocation/alias/readers remain unbound;neither the own record nor the installed disk object may be substituted. '
            '80capturedBOOL tasks still require creation/value/storage/update/allreaders evidence. All9/1352 remain pending.')
        if row['id']=='G13':row['remaining']=(
            'CPU adapterv127 and native disk scalar derivationv128 are complete in stated diagnostic scopes;do not repeat generic infrastructure work. '
            'The installed vendor CPU objdump decoded3pinnedobjects;8081instructionwords match cached ELF bytes. Unsupported instructions/dtypes failclosed;vector/cache/pipeline/DMA/physicaltiming andloadedcode attestation remain outside the model. '
            'Real runtime/provider interposition/fullidentity/errorstate/operand/lifetime gates remain;fresh device/model/capture/replay authorization has not been granted. '
            'Continue remaining operator relation analysis using profile-backed demand/epoch/storage constraints;F30-F31 is still centrally unreconciled,not ruledout by empty navigation fields. '
            'Allscope/precision/physicalcapacity/criticalpath/MFU/lowest-time gates remain open. v120Pad grant consumed;no pending approval.')
    obligations.update(parent='coverage-current-obligations-v127.json',new_native_or_whole_closures=0,overall_complete=False)
    write(O/'coverage-current-obligations-v128.json',obligations)
    audit=deepcopy(read(O/'whole-scope-requirement-audit-v127.json'))
    audit.update(previous_goal_turn='PROGRESS:v127 Linux CPU adapter23cases/4sanitizer cases and nested-task/bypass evidence;postcheck PASS596',
        input_checkpoint=127,followup_checkpoint=128,native_store_scalar_plan_v128=detail)
    for row in audit['requirements']:
        if row['id']in ['MODEL_VERSION_HARDWARE_LOAD','NATIVE_OPERATOR_REGIMES','CRITICAL_EVIDENCE_AND_FINAL_CLOSURE']:row.setdefault('evidence',[]).extend(names)
    write(O/'whole-scope-requirement-audit-v128.json',audit)
    write(O/'PROFILE-CODE-QUESTION-GATE-v128.json',dict(
        status='EXACT_INSTALLED_NATIVE_MEMSET_PAD_DISK_OBJECTS_AND_SAVED_OWN_PARAMETER_RECORD',
        questions=read(O/names[0])['questions'],source_scope='Three already-pinned operator objects,SDK decoder and bounded standalone CPU scalar-control derivation;no framework survey.',
        no_target_model_operator_implementation=True,CANN_or_Torch_runtime_loaded=False,
        CPU_remote_objdump_used=True,Pad_authorization='GRANTED_AND_CONSUMED',additional_device_authorization=False,new_device_work=False,overall_complete=False))
    write(O/'native-store-validation-history-v128.json',dict(events=[
        'Predecessorv127 revalidated PASS596;previous goal turn is substantive CPUadapter progress,not a wait or originaloperand recovery.',
        'Four legacy guessed compiler tool paths were absent. One bounded installed SDK directory listing located latest/bin/llvm-objdump. No configcontents or framework source survey read.',
        'VendorLLVM15.0.5/hiipu64 CPU tool disassembled3SHA-pinned cached operator objects from unchanged existingcontainerPID2032500. No runtime library,Torch,NPU,model,capture orreplay invoked.',
        'All8081disassembled instruction words independently match little-endian cached ELF text bytes. MemSettext6828B/arg-size field16;two INT8Padobjects text12748B each/same text SHA/different fullobject SHA/arg-size field56.',
        'Saved own-v120 packet152/inlineoffset24/record96 supplied to explicit scalar-only model;48block certificates and conservative COREID taint check. One outputinstruction at0x478,block0,workspacebase/rawcontrol0x200001;source compatible32B.',
        'Loopfree147synthetic records(K0..48,gap0/4/28),7056blocktraces match sourceplan and rawdescriptor arithmetic;tilerlegality/vector/cache/pipeline/physicaltransactions/timing not proven.',
        'First verifier compared in-memory tuple with JSON list for rawMOVX8 metadata. Preserved source/representation-error receipt;canonical JSON comparison fixes representation only. No native data orsealed predecessor files changed.',
        'Allstaticreadcontrollers finished. ControlPersistexpired before explicitclose;exit returned missing socket. Independent process/socket audit confirms no ownedSSHmaster remains.',
        '8573preflight checksPASS. No originaloperand,physicalcapacity,MFU,relationcoverage orwholemodule promotion;all9/1352 andglobalgaps remain.'],overall_complete=False))
    gap=(
        'Pinned disk ISA/source agree on own saved zero-record scalar plan,not actual executed/loaded instruction bytes ororiginal fields. '
        'Native entry16B address field !=packet152 !=source record96 !=workspace1536 !=source-compatible32Bstore request. '
        'One outputstore site/48launchedblocks doesnotgive cycles,physicalHBM,or legal isolatedhelper deletion. '
        'Needoriginalnativeprovider/params/dst/count/allocation/alias/workspace/readers,plus capturedBOOL creation/update/readers. Wholemodules andphysicalMFU/timebounds remain open.')
    remaining=[r['id']+': '+r['title']+'. '+r['remaining']for r in obligations['obligations']]
    remaining.append('v1283diskobjects/8081words/48ownblockcertificates/147syntheticrecords7056traces reconciled only inexplicit scalar domain.9/1352pending,358/26semantic,854/1346/48partialgraph,135/26/460relations,849PMU/16phase/3136ambiguity and0whole unchanged.')
    action=(
        'Continue original Goal;v128 PROGRESS is targeted native disk scalar-control evidence,not execution,loadedcodeattestation oroldoperand recovery. '
        'Read native-store summary/ELF-controls/own-scalar-traces/synthetic-control-validation/fusion-consequences/current-obligations-v128;verify_native_store_plan_v128 mustPASS. '
        'v127postPASS596. InstalledvendorLLVM15hiipu64CPUobjdump read3exactSHAobjects;8081wordsmatchELFtext. No realCANN/Torch/NPU/model/capture/replay orserviceschanges;containerPID2032500stable. SSHmaster/socket nowabsentconfirmedafterControlPersistexpiry;do not reconnect just toclose. '
        'MemSet .text6828B,CCEKernelArgSize16;entryat0x40LDPloadsarg0dst,arg1flow. Ownpacket152,flow+24,record96,workspaceadvertised1536,source-compatiblewrite32 areseparate. Thirdpacketword/unknowntail semantics notinferred. '
        'Ownzero-record scalartrace48blocks:allpasscore_num48guard,3inputtransferinstruction sites each;onlyblock0outputat0x478,workspacebase/rawcontrol0x200001. Sitesvisited180forblock0,140forothers;notcyclecounts ornecessarysemanticwork. '
        '147syntheticloopfreeFLOAT32zero records K0..48 gap0/4/28=7056blocktraces matchsourceplan+rawdescriptorarithmetic. COREIDtaintdoesnotreachguard/address onownpaths. Modelnotsimulator:recordUBbindingpremise,rawMOVX8undecoded,vector/cache/pipeline/DMA/physicaltraffic/relocation-runtimeattestation outside. '
        'BothINT8Padcontiguityvariants .text12748BsameSHA,differentfullobjects/metadata;notruntimeplanidentity. NoisolatedMemSetdeletionorfusionproved. Retainedinitialverifier tuple/listrepresentationerror,notkerneldataerror;prePASS8573. '
        'Return to remaining operator-fusion relation coverage rather than more adapter infrastructure. F30-F31 (DSpark confidence/verification vsmetadata construction/publication) is centrally pending;review savedprofile demand/selectedrow/epoch/storage/publish constraints for same-iteration,graph-capture andnext-iteration cases. Empty navigation atoms/edges do notproveindependence. Usecachedtargetedsource onlyifprofilecannotdecide. '
        'All9/1352pending,2073/285598conditional,358/26semantic,854/1346/48partialgraph,135/26/460relations,849PMU16phase3136ambiguity0whole/noallmodulephysicalMFU/lowest unchanged. RealCANNfullidentity/operand/lifetime andcapturedBOOL80gates remain. No freshNPU/model/capture/replay authorization;v120Padgrantconsumed,no pendingapproval. '
        'Do noteditsealedv128/rerunfinalizer. Versionedpostcheckauthority;CURRENT-POSTCHECKoverlayvalidonlymatchingRECOVERY. TaskACTIVE Loop004EVALUATING GoalACTIVE.')
    report='''本轮把 MemSet 的源码请求模型推进到了原生磁盘二进制的标量控制核对，未启动 NPU。对3份此前已固定SHA的MemSet/INT8 Pad对象，用安装SDK的LLVM15.0.5 hiipu64 objdump做只读反汇编。逐一核对了8081个指令字与缓存ELF的.text字节，原对象没有修改，现有容器PID2032500保持不变。没有加载真实CANN/Torch运行库、运行模型或capture/replay。

参数大小必须分开解释：MemSet的__CCE_KernelArgSize段内容为16；原生入口0x40的LDP读取前两个64位地址。自有v120图展示的整包为152B，第二个地址指向包内+24位置，那里有96B源码记录。1536B是选中Pad metadata的workspace声明，32B是该自有记录与源码相容的清零请求。16、152、96、1536、32不可互相替代；没有把第三个包字或剩余尾部称为无用字段，也没有把编译段大小当成HostArgs总长或写入量。Pad对象相应字段56也不同于其自有图320B参数包。

在明确限定的零FLOAT32、无完整块循环域，CPU模型只推导标量分支与地址/原始描述符算术。对已保存的自有参数，48个block都通过core_num=48的外层检查，每个路径都访问3个输入传输指令位置；只有block0经过输出存储指令0x478。该指令目标为记录中的workspace基址，描述符原值0x200001，与源码请求32B相符。block0访问180个指令位置，其余各140个；这些是当前实现的静态路径计数，不是运行周期、必要工作下界或可累加的关键路径时间。

另用147份合成记录覆盖need_core_num=0..48和gap=0/4/28，共7056条block路径；输出位置、地址及描述符算术与独立源码区间模型一致。核验还检查了COREID的保守标量数据流未进入自有路径的分支条件或传输地址，并拒绝未知指令及非零FLOAT32域。模型不是NPU模拟器：96B记录进入标量UB视图是显式前提，MOVX8传输编码保留为未通用解码值；向量内容、cache、pipeline、DMA物理行为和时序均未模拟。自有图此前只捕获、没有replay，本轮也没有取得实际加载代码的字节SHA或硬件执行证据。

两份INT8 Pad的contiguity变体虽然完整对象SHA和metadata不同，12748B的.text完全相同。因而相同机器码也不能证明相同attrs、flowtable、存储布局、生命周期或读者集合。对原模型，原生函数名称和48个launched blocks仍不足以证明有效writer数、workspace角色或真实读写范围。

融合方向因此更明确但尚未定案：该类小Pad helper需要考虑通用分发和元数据处理开销，不能只按输出带宽评价。整段Pad/初始化替换仍须保留BOOL前缀、True后缀、全部mask读者、别名、epoch/capture和同步语义。单独删除MemSet，或把单个小图的同步区结论推广到多block Pad，仍没有合法性证明。更没有把32B外推到原1272次eager MemSet，或据此给80次捕获BOOL24绑定语义字段。

第一次本地核验遇到tuple与JSON list的表示比较差异，已保留错误记录和原核验脚本；改为统一JSON表示后通过8573项前置核验，算子数据未改变。静态读取均已结束。SSH ControlPersist在尝试显式退出前已到期；独立核查确认socket及本任务SSH master进程均不存在，无需重连清理。

本轮不是全局完成：35项模型候选、7条历史框架记录、358条语义分支/26模块、135条条件关系/26联合组/460待核对pair、854节点/1346边/48提示的部分图保持原状。原9类/1352次操作数字段缺口、849个PMU缺口、16个phase未绑定图、3136项归属歧义均保留，整模块关闭0，全模块物理MFU及最低时间下界未建立。后续回到尚未分析的算子组合关系，优先检查F30与F31的需求裁剪、行选择、存储epoch及发布约束；不继续泛化研究框架。
'''
    inputs=[Path(r['path'])for r in read(O/'native-store-inputs-v128.json')['inputs']]
    updates={cid:dict(review_v128=dict(evidence=names,
        pinned_disk_scalar_plan=True,own_zero_record_output_store_sites=1,own_source_compatible_store_bytes=32,
        CCE_KernelArgSize_field_is_not_packet_or_write_bytes=True,actual_loaded_instruction_bytes_attested=False,
        actual_model_native_operands_bound=False,isolated_helper_deletion_proved=False,new_device_work=False))for cid in ['F20','F28','F31']}
    persist_refinement(version=128,previous=127,title='Native disk MemSet scalar guard and operand-size reconciliation',
        summary=dict(filename='native-store-summary-v128.json',previous_turn_classification='PROGRESS:v127 Linux CPU adapter and nested-task/bypass evidence;PASS596',
            **detail,refined_candidates=3,refined_modules=2,branch_records=358,semantic_modules=26,
            relation_clauses=135,higher_order_groups=26,pending_central_pair_reconciliation=460,
            partial_graph_node_rows=854,partial_graph_edge_rows=1346,structural_pair_hints=48,
            scoped_reviewed_regimes=2073,scoped_reviewed_executions=285598,pending_regime_reconciliation=9,pending_execution_reconciliation=1352,
            model_operator_implementation=False,native_test=False,actual_MFU_numeric_physical_ceiling_or_speedup_proved=False,
            Pad_authorization='GRANTED_AND_CONSUMED',remote_job_status='CPU-only vendor objdump reads complete;containerPIDstable,no CANN/Torch/NPU/model/replay. SSHmaster/socket authoritatively absent afterControlPersist expiry.'),
        changes=['Decode3pinnednativeobjects withvendorCPUtool;8081instructionwords matchELFbytes anddistinctargument-size concepts reconciled.',
            'For savedownrecord,derive48scalarpaths/oneoutputsite and147syntheticrecords7056blocktraces;noexecution/physicaltraffic/oldoperand claims.',
            'Retain allnative/fullscopegaps andredirect nextwork toremainingoperatorrelation/demand/epoch/publication analysis.'],
        candidate_updates=updates,module_updates={
            'M12':dict(candidates=[],finding='Pinned native MemSet scalar entry and writer guards support own-record32Bsource request;48launchedblocks do not imply48writers. OriginalPad helperoperands unbound.',evidence=names,gap=gap),
            'M19':dict(candidates=[],finding='Compiler arg-sizefield,fullpacket,source record,workspace declaration andstore request are distinct;actual code/control binding and readers remain prerequisites for initialization fusion.',evidence=names,gap=gap)},
        remaining=remaining,action=action,
        jobs=[dict(kind='design-check',status='error',description='Retain initial local verifier representation error:tuple versus JSONlist forrawMOVX8;not native arithmetic/data failure.',artifacts=[O/'native-store-preflight-attempt-v128a.json',N/'verify_native_store_plan_v128a.py']),
            dict(kind='profile',status='pass',description='Read-only vendorCPUobjdump of3pinnedobjects/8081wordreconciliation;no CANN/NPU,containerPIDstable,SSHscope terminal.',artifacts=[O/'native-static-disassembly-index-v128.json']+inputs),
            dict(kind='design-check',status='pass',description='48ownscalarpathcertificates,147syntheticrecords7056traces,COREIDtaint/failclosed guards and8573preflight checksPASS;nooriginalnative/wholepromotion.',artifacts=[O/'native-store-preflight-v128.json',O/'native-store-validation-history-v128.json']+[O/n for n in names]+[O/'coverage-current-obligations-v128.json',O/'whole-scope-requirement-audit-v128.json'])],
        report=report,read_first=names+['coverage-current-obligations-v128.json'],publication_rows={},gate_updates=dict(overall_complete=False),
        scripts=[N/'read_native_static166_v128.py',N/'native_memset_scalar_model_v128.py',N/'analyze_native_store_plan_v128.py',N/'verify_native_store_plan_v128.py',N/'persist_profile_refinement_v90.py'],
        executor_prefix='native-static',main_script=Path(__file__),source_read=False)


if __name__=='__main__':main()
