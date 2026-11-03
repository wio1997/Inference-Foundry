"""Seal current physical-pool and scoped cycle/topology analysis once."""
import json
import sys
from copy import deepcopy
from pathlib import Path
sys.path.insert(0,'/Users/wio/work/166-vllm-analysis-continuation-20261030')
from finalize_origin_coverage_v9 import N,O,read,write
from analyze_evidence_batches_v155 import pin
from persist_delta_checkpoint_v163 import persist_delta
H=Path(__file__).resolve().parent;old=read(O/'REVIEW-CHECKPOINT-v170.json');prefix='physical-pool-cuts'
pool=read(H/'current-171-physical-pool-bridge.json');clock=read(H/'clock-field-primary-semantics-and-bound-status.json');cycle=read(H/'current-arch-cycle-and-reference-capacity-cuts.json');cut=read(H/'TP8-board-ingress-lower-bound-and-padding-domain.json');ind=read(H/'cycle-and-topology-independent-review.json');receipt=read(H/'final-health-and-remote-artifacts.json');acl=read(H/'ACL-device-metadata-query.json')['review'];dcmi=read(H/'dcmi-frequency-observation.json');bridge=read(H/'cohort-device-bridge-inputs.json')
checks={
 'previous_goal_turn_progress':read(O/'typed-operator-cuts-summary-v170.json')['status']=='PROGRESS',
 '16ACL_devices_init_finalize_success':acl['aclInit_rc']==acl['aclFinalize_rc']==0 and acl['device_count']=={'rc':0,'value':16},
 'all6mapping_roundtrips_all16success':len(pool['rows'])==16 and len({r['driver_physical_device_id'] for r in pool['rows']})==16,
 'arch2201_24Cube_48Vector_allnonvirtual':all(r['arch']==2201 and r['CUBE_cores']==24 and r['Vector_cores']==48 and not r['is_virtual'] for r in pool['rows']),
 '8NPUcards_two_compute_devices_each':pool['observed_NPU_cards']==8 and all(sum(r['NPU_card_id']==b for r in pool['rows'])==2 for b in range(8)),
 'different_chip_namespaces_explicit':len({r['ACL_reported_PHY_CHIP_ID'] for r in pool['rows']})==8 and len({r['SMP_ID'] for r in pool['rows']})==8,
 '16samecontainer_CANN9p2_capture_devices_bound':len(read(H/'physical-mapping-and-capture-metadata.json')['captured_device_info'])==16 and len(read(H/'captured-group-device-rows.json')['rows'])==16,
 '2TP8_recorded_groups_fourcards_each':len(pool['TP_placements'])==2 and all(len(r['NPU_cards'])==4 and r['directed_SIO_pairs']==8 and r['directed_HCCS_SW_pairs']==48 for r in pool['TP_placements']),
 'EP16_MC2_membership_static_not_globalcall_identity':len(pool['EP_MC2_memberships'])==2 and all(len(r['global_ranks'])==16 for r in pool['EP_MC2_memberships']),
 '16SIO_224HCCSSW_directed_topology_pairs':pool['topology_pair_counts']=={'SIO_directed':16,'HCCS_SW_directed':224},
 'UB_api_zero_retained_not_zero_capacity_proof':all(r['UB_attribute_raw_value']==0 for r in pool['rows']),
 'DCMI16rated1800_idle800_vectorunsupported':len(dcmi['review']['rows'])==16 and not dcmi['handle_live'] and clock['actual_queries']['rated_enum9_MHz']==1800 and clock['actual_queries']['idle_current_enum7_MHz']==800 and clock['actual_queries']['Vector_current_enum12_return_code']==-8255,
 'official_A3_rates_NOT_absolute_fmax':clock['physical_peak_frequency_upper_Hz'] is None and clock['official_source']['url'].endswith('/A3DCMI/dcmia3_049.html'),
 '6mode_rate_reference_cases':len(cycle['mode_cases'])==6 and all(r['current_reported_CUBE_cores']==24 for r in cycle['mode_cases']),
 'INT8_180cycle_per_assignment_and3role_exact_reference_checks':ind['MAC_per_assignment_180_equivalent_cycles'] and ind['cycle_time_reference_values_crosschecked_as_exact_rationals'] and len(cycle['cases'])==3,
 'AllGather_full_M3_ingress24S_not48S_wirefloor':cut['full_M3_example']['required_aggregate_ingress_bytes']==737280 and cut['full_M3_example']['direct_rank_delivery_crosscard_reference_bytes']==1474560,
 'padding20reduces_aggregate_not_maxcard_cut':cut['known_padding_n20_p8_example']['required_aggregate_ingress_bytes']==614400 and cut['known_padding_n20_p8_example']['required_max_card_ingress_bytes']==184320 and ind['padding_example_maxcut_unchanged'],
 '65finitegeometrychecks_not_formalverification':cut['independent_finite_geometry_checks']==65 and not ind['is_general_formal_verification'],
 'read_only_query_wrappers_errors_preserved':len(read(H/'acquisition-errors-and-scope.json')['errors'])==3 and len(bridge['auxiliary_info_markers'])==16,
 'DCMI_library_pinned_and_process_terminal':len(bridge['dcmi_library_pins'])==1 and not dcmi['handle_live'],
 'original_service200_unchangedStartedAt_no_applicationkernels_reload':receipt['health_http']==200 and receipt['running'] and receipt['started_at']=='2026-11-02T19:01:54.649699535Z' and receipt['new_application_kernel_model_replay_or_model_reload_calls']==0,
}
assert all(checks.values()),checks
write(O/(prefix+'-preflight-v171.json'),{'status':'PASS','preflight':True,'checks':checks,'basis':'CurrentreadonlySDK/managementmetadata,all16capturedgroup/device joins,declaredcycle/cutmodelsandindependentreview. No physicalfmax/wholemodulecompletioninferred.'})
write(O/(prefix+'-inputs-v171.json'),{'inputs':[pin(p) for p in sorted(H.rglob('*')) if p.is_file() and '__pycache__' not in p.parts],'overall_complete':False})
summary={'status':'PROGRESS','previous_goal_turn':'PROGRESS','overall_complete':False,'new_native_operand_or_phase_promotions':0,'new_whole_module_closures':0,'current_device_mappings_reconciled':16,'NPU_cards':8,'reported_CUBE_per_compute_device':24,'reported_Vector_per_compute_device':48,'actual_arch':2201,'TP_placements':pool['TP_placements'],'actual_clock_queries':clock['actual_queries'],'new_absolute_frequency_upper_bounds':0,'F21_cycle_reference_cases':cycle['cases'],'AllGather_full_vs_known_padding_ingress_bytes':{'full_M3':737280,'n20_to24_conditional':614400,'max_card_both':184320},'new_per_native_global_call_identity_or_alias_proofs':0,'new_independence_proofs':0,'whole_modules_closed':0,'new_application_kernel_launches_model_replays_or_reloads':0,'ACL_metadata_child_queries':2,'DCMI_metadata_query_jobs':1}
write(O/(prefix+'-summary-v171.json'),summary)
action=(
 '继续完整目标，当前checkpoint171。v170和v171为实质PROGRESS。直接读PROFILE-FINDINGS-v171、physical-pool-cuts-operator-overlay-v171、coverage-work-queue-v171和shared-profile-analysis-v171/physical-pool-analysis-question-review；不重读全部旧框架/source或重跑成功完整Case。'
 '当前171硬件映射阶段已取得16user/logic/driverphysicaldevice正向/逆向/直接映射，8NPU卡各2compute设备；全部arch2201/Cube24/Vector48/nonvirtual0。16实际CANN9.2profile的pid/nativeDeviceId/sourceEP/DP/TP和parallel_group_info已关联。TP全局0..7在卡0..3，8..15在卡4..7，各56非self pair中8SIO/48HCCS_SW；EP/MC2全16。应用当前映射于capture时保留无中间driver-remap前提，同容器/可见性/核心数一致。旧166不回填。'
 '命名空间已分开：driverphysicaldevice16个；ACL_PHY_CHIP_ID和SMP_ID各8组；MAINBOARD_ID全20，不是NPU卡号。AICPU每device报6但SMP成对，不擅自算16个独立AICPU池。UB attr204全部rc0/value0，不能当0UB或融合不可能；L2报192MiB/查询单位、不合并自由池；globalmemory报总暴露量，不是优化可用free。'
 '频率查询已结束，不再反复问fmax：16DCMI enum9=1800MHz，enum7闲置800，HBM1600，Vector12返回-8255；本机header叫DCMI_FREQ_AICORE_MAX但官方A3文档定义9为额定频率，不能升为绝对上界。保留physicalfmaxNone，不让它卡所有算子结论；下一步用已匹配arch的周期下界和明确额定频率参考域。'
 '已有6个Cube模式的percycle表和实际24cores匹配；directINT8源算法每assignment35389440MAC需>=180整device等效cycles。F21draft/targetsmall/prefill三域累计cycles为14588100/462153600/1136937600；在1800MHz参考下compute floor/完整kernel累计时间比约0.30696%/0.49423%/6.64652%，只是参考MAC比例，不是physicalMFU或可达提速。不同资源下界取max，未证实共同issue simplex不能相加多模式峰值。'
 '跨卡模型已在实际4卡TP8放置下重定价：任意单owner未压缩required rawpayload总G、卡b已有C_b，必要ingress>=G-C_b；总(k-1)G。均匀8rank各S、4卡每卡2rank，至少每卡6S/总24S；48S是rank直接投递参考，不是通用wire下界。F26/F32/F33/F31复用此共同cut，不做无关配对。未知带宽和真正同一端口的竞争仍保留。'
 '已知20->24padding条件示例需额外证明collective输入就是该tensor且边界padding仍常量，不能凭BOOLmaskshape套用native。此例总ingress720KiB降600KiB，但最忙卡仍180KiB，均匀card带宽下此cut不变。说明bytes比例不等于时延收益；不据此断言无法改善。初态已复制/压缩/移除raw需求改变cut前提。'
 '下一步直接转统一模块利用率与固定Case分支审计。先读取本仓MODULE_DIAGNOSIS.md确定一致主指标/适用域，再复用v165无observer32profile、v170资源/typed角色与现有operator work合同。不同cohort计数和时间不可硬配；AICPU/Vector/Matrix不可统一塞CubeMFU。给原26模块一份明确已证/条件/未知/不适用路由，并逐项补会改变fusion或最低利用率结论的缺证。'
 '若现有profile不能定案实际标志/参数，按当前215callsite挑最少字段准备一次合并小探针，先利用已存Case payload/config/CPUstack，避免逐候选探针。用户已授权容器推理进程和探针，不重复问许可。只在结论确需时动现有服务；不为日志解析问题重采集完整Case。四个v170公开算子条件分析和v169SPmask已分析域直接复用，不以缺实现/私有workspace packet作为其门槛。'
 '完整35+7候选、26模块358语义行、全部条件分支和跨算子/层/卡/迭代融合拆分重组不缩减。allreader/numerics/状态/copy/control/实际物理时频带宽与最低模块利用率仍有缺口；没有整模块闭合，也不将旧9/1352等历史forensic gap当所有新域通用前提。171服务health200且StartedAt未变；本轮只有metadata查询，显式应用kernel/model/replay/reload0。')
gate={'version':171,'parent':pin(Path(old['source_gate'])),'policy':'Resolve current device/semantic clock namespace once;use declared cycle/cut bounds and move to module comparison. No repeated fmax/ABI/framework archaeology.',
 'resolved_questions':[{'question':'Does current171 have an explicitrank/device/physical/card bridge?','answer':'16mappingroundtrips andsame-containercapturedpid/device/groupmetadata reconcile8cardsx2computedevices;TP8groups4cards each. Keepcapture-to-currentstablemapping assumption anddo notbackfill166.'},{'question':'DoesenumAICORE_MAX=9prove anactualclockupper?','answer':'No.OfficialA3definitionisratedMHz;query1800,currentidle800. Retainabsoluteupperunknown andusecycle/reference domains.'},{'question':'Are sameSMP/PHY_CHIP labels the16deviceIDs,orUBrc0/value0 azerocapacityproof?','answer':'No. TheseareseparatenamespacesandzeroUBfieldisunusableasphysicalbound. AICPUphysicalpoolownershipstillrequiresseparateevidence.'},{'question':'What crosscardpayloaddemandis necessary forrawTP8AllGather?','answer':'Inthesingleowner/uncompressed/fullraw-demanddomainonfourcards:atleast6S/card,24Stotal. 48Srankpairbudgetisnotuniversalwirefloor. Knownpaddingcanreduceaggregatewithoutreducingmaxcardcut.'}],
 'next_targeted_questions':['UseMODULE_DIAGNOSISandprofiles todefine uniform modulemetrics withoutcrosscohort mixing orforcingCubeMFUonAICPU/Vector.','Audit original26module/358semanticrows for fixedCase applicability andconclusion-changing missingfields.','Onlyifneeded,collectonecombinedsmallfieldprobe;no repeatedfullCase or per-unrelated-pair loops.'], 'all_missing_fields_remain_UNKNOWN':True,'overall_complete':False}
overlay={'version':171,'parent':pin(Path(old['evidence_overlays'][-1])),'affected_candidates':['F21','F26','F31','F32','F33'],'affected_modules':['M07','M11','M12','M13','M15','M19','M20','M21','M23'],'physical_pool':pin(H/'current-171-physical-pool-bridge.json'),'clock_field_contract':pin(H/'clock-field-primary-semantics-and-bound-status.json'),'cycle_reference_capacity':pin(H/'current-arch-cycle-and-reference-capacity-cuts.json'),'TP_board_cut':pin(H/'TP8-board-ingress-lower-bound-and-padding-domain.json'),'math_review':pin(H/'cycle-and-topology-independent-review.json'),'current_profile_group_rows':pin(H/'captured-group-device-rows.json'),'new_current_device_mappings':16,'new_absolute_frequency_upper_bounds':0,'new_pair_clauses':0,'new_independence_proofs':0,'new_native_operand_or_phase_promotions':0,'historical_native_or_phase_promotions':0,'new_whole_module_closures':0,'whole_modules_closed':0,'overall_complete':False}
queue=deepcopy(read(Path(old['work_queue'])));queue.update(version=171,parent=pin(Path(old['work_queue'])))
for row in queue['work']:
 if row['id']=='Q155-PHYSICAL-POOL':row.update(state='CURRENT16DEVICE_MAPPING_AND_ARCH_BOUND_CLOCKRATED_NOT_ABSOLUTEUPPER',action='Current171mapping/corearchstagecompleteinrecordeddomain;use24Cube/48Vector/2201andexistingtypedcyclerates. Stoprepeatingenum9queryforfmax:itmeansrated. Preserveclock/transportupperunknownwhileadvancingmodulecomparisons.',remaining='Capture-to-query no-remap premise;historical166mapping notbackfilled. AICPUsharedpool/actualinstructionmode/clockupper/HBMlinkuppers remainquestionsonlywhereconclusionchanging.',evidence=[str(H/'current-171-physical-pool-bridge.json'),str(H/'clock-field-primary-semantics-and-bound-status.json'),str(H/'current-arch-cycle-and-reference-capacity-cuts.json')])
 if row['id']=='Q166-COMM-NEIGHBORHOODS':row.update(state='ACTUAL_TP_GROUP_PLACEMENT_AND_BOARD_INGRESS_CUT_REVIEWED',action='Reuse424typedneighborhoodsandexistingF26/F32/F33clauses. Actual4cardTP8domainhas24Saggregateingressfloor;48Sisrankdirectdeliveryreference. RecomputeG/C_bafterconsumerfusion,constantpadding,initialreplicasorownershipchanges;no newunrelatedpairs.',remaining='Actualnativeinput/epochs/allrequiredrawreaders,wireprotocolandbandwidth/portsharing remainconditions. Staticmembershipisnotper-callglobalordinalidentity.',evidence=[str(H/'TP8-board-ingress-lower-bound-and-padding-domain.json'),str(H/'captured-group-device-rows.json')])
queue['work'].insert(0,{'id':'Q171-UNIFORM-MODULE-DIAGNOSIS','state':'CURRENT_HARDWARE_AND_OPERATOR_DOMAINS_READY_FOR_SCOPE_CORRECT_MODULE_REVIEW','action':'ReadrepoMODULE_DIAGNOSIS.mdforprimarymetricandapplicability. Usev165unobserverprofiles,v170resource/typedfrontierandcurrentarchcyclecuts;do notmixcohortcounts/time. Audit all26modulesand358semanticrows forfixedCasebranchapplicability,andresolveonlydecision-changingmissingfields;reuseconditionaloperatoranalysisalreadydone.','remaining':'Wholemodulelowestutilizationandfullbroadbranchclosure notestablished. Ifneededprepareonecombinedsmallargument/flagprobe,notper-pairprobes oranotherfullCase forbookkeeping.','evidence':[str(H/'physical-pool-analysis-question-review.json'),str(H/'current-171-physical-pool-bridge.json'),str(H.parent/'shared-profile-analysis-v170/resource-domain-routing-review.json')]})
queue['new_evidence']=['16ACLdeviceIDmappingroundtrips +16capturedpid/nativeDevice/groupmetadata join8NPUcardsx2computeunits;arch2201/24Cube/48Vector/nonvirtual forall.','TwoactualTP8groups on4cardseach;SIOwithinpaireddevices,HCCS_SWacrosscards. Distinguish16driverphysicaldeviceIDs from8PHY_CHIP_ID/SMPgroups.','DCMIcurrentqueries1800MHz enum9/800MHzidle enum7/HBM1600;officialA3defines9asrated,notfmax. Vectorqueryunsupported andUBrc0/value0 retainedasunknowncapacity.','Frequency-freeINT8Cube directsourcecyclecuts and6modereferencecapacitiesnowmatchcurrentarch/corecounts;1800MHzfractionsarereferenceonly.','FourcardTP8singleownerrawAllGatherrequires6S/card/24Stotal;rankdirect48Sisnotuniversalwirefloor. Known20to24paddingreduces720to600KiBaggregatebutmaxcard180KiBunchanged.','65geometrychecks andexactrationalcyclechecks;readonlymetadatawrapperserrorspreserved. Originalservice200/StartedAtunchanged,zeroapplicationkernels/modelreplay/reload.']
recs=deepcopy(read(O/'current-obligation-reconciliations-v170.json')['reconciliations'])
for row in recs.values():row.update(v171_evidence_delta={'operator_overlay':str(O/(prefix+'-operator-overlay-v171.json')),'scope':'Current171device/card/group/corearch bridge;frequencyfieldratednotfmax;scopedcycleandboardingresscuts. Alloriginalmodules/branchesretained;no historical/nativeoperand/wholemoduleclosure.'},whole_domain_closed=False)
write(O/'current-obligation-reconciliations-v171.json',{'parent':pin(O/'current-obligation-reconciliations-v170.json'),'reconciliations':recs,'does_not_delete_or_narrow_original_requirements':True,'historical_forensic_gap_not_universal_prerequisite':True,'overall_complete':False})
report='''# v171：本机设备映射、周期下界与跨卡数据下界

本轮上一目标回合v170为实质PROGRESS。本轮补的是会影响算子容量和跨卡结论的硬件证据；没有重跑完整Case、加载模型或运行应用kernel，也没有重启服务。171 health200且StartedAt未变。原始框架和算子实现未改。

当前CANN9.2接口对16个user device完成了user→logic→driver physical的正向、直接和逆向查询，均成功且往返一致。NPU-smi记录8个NPU卡号，每卡ChipID0/1两个计算设备；同卡对经SIO，其他设备对标为HCCS_SW。16个设备均报告arch2201、24Cube/48Vector核、非虚拟设备。16个实际profile的PID、native replay deviceID、源EP/DP/TP rank、captured parallel_group_info与设备记录完成关联。两个TP8组分别为global0..7和8..15，各占4张卡；EP/MC2组含全部16rank。每个TP组56个非self有向endpoint pair中有8个SIO、48个HCCS_SW。

应用当前映射到先前采集区间保留“期间未发生driver重映射”的环境前提；同一容器身份、可见性、CANN版本和采集核心数一致。这里不回填旧166历史。静态组成员不等于某次native通信的全局调用序号。

几种名称不能混用：driver physical deviceID有16个，ACL属性PHY_CHIP_ID和SMP_ID各只有8组，MAINBOARD_ID全为20。容量表按“一个暴露的计算设备”计数，不能因chip/card名词再翻倍。AICPU每设备查询报6，但SMP成对，未证明16个独立AICPU池。L2查询报192MiB，不能直接合并成自由共享缓存；globalmemory是API暴露总量，不是可供优化新用的free。UB属性204全部返回rc0/value0，此字段不能当物理零UB，也不能据此宣判某种融合不可能。

频率字段已定案：本机16个DCMI enum9均返回1800MHz，enum7空闲时800MHz，HBM字段1600MHz，Vector current查询返回不支持。尽管安装头文件把enum9命名为DCMI_FREQ_AICORE_MAX，官方A3文档将其定义为TDP条件下持续运行的额定频率。因此1800不能直接升为最高频率上界，HBM频率也不是带宽。保留真实fmax未知，停止为同一个字段反复查询；该缺口不再阻止其他算子域的条件分析。[A3 DCMI接口定义](https://www.hiascend.com/doc_center/source/zh/HDK/2610/A3/A3DCMI/dcmia3_049.html)

现有A3_2201指令表与本机架构/24Cube核现在匹配。在声明的direct-dense INT8 Cube源算法域内，必要corecycles≥Q_MAC/8192，整设备等效cycles≥Q_MAC/(24×8192)。每个receiver assignment的35389440MAC至少对应180个整设备等效cycles。该推导不依赖实际时钟上界，不把最快实测或可行调度倒数当上界。应用到当前计数仍保留返回计数stream/source算法合同；不是loaded binary中每条指令的证明。

| F21域 | 必要累计整设备等效cycles | 1800MHz参考计算时间 | 参考计算时间/累计完整kernel时间 |
| --- | ---: | ---: | ---: |
| Draft | 14588100 | 0.0081045s | 0.30696% |
| Target小行数 | 462153600 | 0.256752s | 0.49423% |
| Target prefill | 1136937600 | 0.631632s | 6.64652% |

时间来自v170同域完整聚合，跨rank/调用累计；不是请求walltime。这些百分比仅是额定时钟参考下、有效INT8 MAC周期下界占完整F21累计时间的比例，不能称真实物理MFU或剩余可达提速空间。其他必要访存、Vector和通信约束可能远强于该Cube下界。6个Cube模式已形成明确的每cycle容量/额定参考表；不同资源下界取max，未证明共同issue关系时不把多个模式/资源时间简单相加。

跨卡下界也已按实际放置重算。对一个4卡TP8完整raw AllGather，设每rank初始单独持有S字节，所有rank都需要全部raw数据，且没有预存远端副本、压缩或移除需求。每卡已有2S，总数据8S，至少需从卡外进入6S；4卡总入口必要payload至少24S。允许卡内转发/组播后，按48个跨卡rank pair计算的48S只是直接投递参考，不是通用wire下界。更一般地，总必要数据G、卡b初有C_b，则入口至少G−C_b，总入口至少(k−1)G。它是必要cut证明，没有将一个可行执行时间倒数成上界。

若入口容量上界为B_b，则覆盖全部所需跨界传输的时间窗至少max_b((G−C_b)/B_b)。这不强迫完整producer先完成才通信，也不把GEMM和通信下界串行相加。实际链路带宽、共享端口和协议仍未确定；HCCS_SW标签不表示独立全带宽。TP和EP若同时占用同一卡入口，就必须共用预算。

条件padding示例进一步说明字节比例不能直接解释时延：若一个BF16/H5120的AllGather输入确实由20行补到24行、padding到边界仍为已知常量，8rank有效行数为[3,3,3,3,3,3,2,0]。4卡已有有效行[6,6,6,2]，各卡需要远端[14,14,14,18]行。总入口下界从完整24行时的720KiB降为600KiB，最大单卡入口却仍为180KiB。相同卡入口容量下，此瓶颈cut未变；这不排除其他瓶颈改善。BOOL mask shape本身不能证明native AllGather的输入就是该tensor，已将此额外条件保留。65个几何边界例和精确有理数周期核算通过，不称通用形式化验证。

F26/F32/F33/F31复用同一拓扑cut，并在消费者融合、初始副本、投影后payload、row ownership或压缩改变时重新推导G/C_b。已经存在的HC局部collapse路径不算新优化。新算术、局部常量重建、存活期和通信需在同一候选图中计价，避免重复收益或免费重叠。

取证封装有三项错误已如实记录：第一次info.json.*也读了16个辅助marker；第一次ACL输出在JSON后追加运行时footer导致严格JSON解析失败，初次raw未保存，第二次只读metadata查询已先存raw再解析；另一次主机脚本字符串拼接SyntaxError发生在执行任何查询前。均已修正且没有因此重跑负载或重载模型。当前完整API结果、unsupported字段、头文件/动态库SHA及11份远端派生文件已保留。显式setDevice/context/allocation/model/kernel调用为0；不对SDK内部初始化行为作额外假设。

下一步转向统一模块诊断与固定Case分支审计，先按MODULE_DIAGNOSIS确定主指标和适用域，复用无observer窗口、当前资源/typed调用点和已完成的公开算子条件分析。只补影响候选判断或最低利用率结论的字段，必要时合并最小探针，不按无关候选配对重复取证。完整35+7候选、26模块358语义行、全部条件分支和跨算子/层/卡/迭代范围保持。实际时频/带宽上限、物理模式、all-reader/数值/状态/copy/control及最低模块结论仍有缺口；整体PROGRESS，未完成。
'''
jobs=[{'kind':'review','status':'error','description':'Three metadatawrappererrors preserved:auxiliaryinfo markers,ACLruntimefooter parsing,andpre-executionstringSyntaxError. ACLmetadataonlyquery repeatedonce withraw-firstpersistence;no inference/modelreload.','artifacts':[H/'acquisition-errors-and-scope.json',H/'cohort-device-bridge-inputs.json']},
 {'kind':'review','status':'pass','description':'16ACLuser/logic/driverphysicalmappingroundtrips andall16same-containerCANN9.2pid/device/groupjoins;8cardsx2nonvirtualcomputeunits,arch2201/24Cube/48Vector. Distinct8groupPHY_CHIP/SMPnamespaces retained.','artifacts':[H/'ACL-device-metadata-query.json',H/'captured-group-device-rows.json',H/'current-171-physical-pool-bridge.json']},
 {'kind':'review','status':'pass','description':'CurrentDCMI16devices:enum9=1800MHzratedbyofficialA3definition,enum7idle800,HBM1600,Vector12unsupported. No absolutefmaxclaim orsetter;currentdriverheader/library pinned.','artifacts':[H/'dcmi-frequency-observation.json',H/'clock-field-primary-semantics-and-bound-status.json',H/'topology-platform-frequency-query.json']},
 {'kind':'review','status':'pass','description':'Currentarch/corebindinginstantiate6modecyclecapacityreferences anddirectINT8F21>=180deviceequivalentcycles/assignment. Exactrationalreferencechecks,notphysicalMFU/timeupper.','artifacts':[H/'current-arch-cycle-and-reference-capacity-cuts.json',H/'cycle-and-topology-independent-review.json']},
 {'kind':'review','status':'pass','description':'Actual4cardTP8placement yieldsconditional6S/card24StotalAllGatheringresscut;48Sdirectreference notuniversalwirefloor. Known20to24paddingexample600vs720KiBaggregatebut180KiBmaxcardunchanged;65geometrychecks.','artifacts':[H/'TP8-board-ingress-lower-bound-and-padding-domain.json',H/'cycle-and-topology-independent-review.json']},
 {'kind':'review','status':'pass','description':'Original171servicehealth200/unchangedStartedAt;metadataqueriescompleted,no applicationkernel/model/replay/reloadcalls.11remoteartifacts pinned.','artifacts':[H/'final-health-and-remote-artifacts.json']}]
persist_delta(version=171,previous=170,prefix=prefix,changes=queue['new_evidence'],report=report,action=action,
 remaining_delta='G171:Current171device/card/group/archbridge andscopedcycle/boardingressmodelsreviewed. Enum9israted,notphysicalfmax;stoprepeatingthatquery. Uniformmodulemetrics/fixedCasebranchaudit andcriticalreaders/numerics/state/copy/control/actualphysicalcapacity/lowestmodule remainopen.',
 source_gate=gate,operator_overlay=overlay,queue=queue,jobs=jobs,scripts=list(H.glob('*.py')),obligation_reconciliations=recs,
 remote_status='ReadonlyACL/DCMImetadataqueries complete;DCMIexactPIDterminalverified. No applicationkernel/model/replay/reloadwork. Original171servicehealth200/StartedAtunchanged;11remoteoutputs pinned. Fulloriginalgoalactive.')
post={'status':'PASS','checkpoint':171,'checks':checks,'manifest_artifacts':len(read(O/'checkpoint-manifest-v171.json')['artifacts']),'predecessor_pins_verified_before_shared_writes':True,'complete_original_scope_preserved':True,'overall_complete':False}
write(O/(prefix+'-postcheck-result-v171.json'),post);write(N/'CURRENT-POSTCHECK.json',post)
write(N/'LIVE-SHARED-COHORT.json',{'effective_checkpoint':171,'status':'READ_ONLY_METADATA_AND_LOCAL_BOUND_ANALYSIS_TERMINAL','receipt':str(H/'final-health-and-remote-artifacts.json'),'no_live_device_or_analysis_job':True,'overall_complete':False})
write(N/'NEXT-EVIDENCE-PENDING.json',{'effective_checkpoint':171,'status':'UNIFORM_MODULE_METRIC_AND_FIXEDCASE_BRANCH_AUDIT_NEXT','next_action':action,'work_queue':str(O/'coverage-work-queue-v171.json'),'overall_complete':False})
print(json.dumps(post,ensure_ascii=False))
