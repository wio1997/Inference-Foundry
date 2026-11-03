"""Seal typed operator/ref-capacity progress, preserving the full active scope."""
import json
import sys
from copy import deepcopy
from pathlib import Path
sys.path.insert(0, '/Users/wio/work/166-vllm-analysis-continuation-20261030')
from finalize_origin_coverage_v9 import N, O, read, write
from analyze_evidence_batches_v155 import pin
from persist_delta_checkpoint_v163 import persist_delta
H=Path(__file__).resolve().parent
old=read(O/'REVIEW-CHECKPOINT-v169.json');prefix='typed-operator-cuts'
typed=read(H/'typed-frontier-observation.json');resource=read(H/'resource-domain-routing-review.json');agg=read(H/'mc2-aggregate-work-review.json')['review'];ind=read(H/'MC2-aggregate-independent-review.json');operator=read(H/'typed-operator-question-review.json');receipt=read(H/'final-service-jobs-and-artifact-receipt.json');engram=read(H/'Engram-hidden-cast-fork-semantic-cut.json');argmax=read(H/'draft-argmax-public-boundary-refinement.json');integer=read(H/'metadata-integer-public-boundary-refinement.json');embed=read(H/'Markov-embedding-strided-ID-refinement.json')
checks={
 'previous_goal_turn_is_progress':read(O/'profile-operator-frontier-summary-v169.json')['status']=='PROGRESS',
 'typed120groups148regimes15520calls21088executions':typed['review']['callsite_groups']==120 and typed['review']['typed_regimes']==148 and typed['review']['public_CPU_calls']==15520 and typed['review']['native_executions']==21088,
 'typed_all16profiles_terminal4p39seconds':typed['status']['state']=='RETURNED' and typed['status']['completed_profiles']==16 and not typed['handle_live'],
 'first_parser_failure_preserved_decimalTEXT_diagnosed':read(H/'first-job-terminal-receipt.json')['state']=='ERROR' and read(H/'CPU-time-field-diagnostic.json')['st']=='text',
 'all32resource_profiles_211operator_resources':resource['profile_count']==32 and resource['aggregate_operator_resource_rows']==211,
 'startup54336_runtime12188947compute_rows':resource['phase_census']['startup']['TASK_rows_with_COMPUTE_info']==54336 and resource['phase_census']['runtime']['TASK_rows_with_COMPUTE_info']==12188947,
 'runtime_untypedTASK14088140_not_zero_work':resource['phase_census']['runtime']['TASK_rows_without_COMPUTE_info']==14088140,
 'metadata_actual_AICPU_separate_capacity':all(x['exported_compute_resource']=='AI_CPU' and x['native_task_type']=='KERNEL_AICPU' for x in resource['actual_metadata_AICPU']),
 'argmaxBF16_indexCast64_160calls':argmax['actual_profile']['calls']==160 and argmax['actual_profile']['native_sequence'][-1]['outputDataTypes']=='INT64',
 'argmax_logical_IO_reference3102784_newfloor1034272':argmax['logical_array_model']['reference_bytes_per_call']==3102784 and argmax['logical_array_model']['combined_required_bytes_per_call']==1034272,
 '960quotientcalls5regimes405edgechecks':integer['actual_quotient_calls']==960 and integer['actual_quotient_typed_regimes']==5 and integer['integer_edge_checks']==405,
 'embedding_selected4IDs_private_budget64B':embed['actual_geometry']['selected_IDs']==4 and embed['logical_array_model']['private_ID_materialization_budget_bytes_per_call']==64,
 'Engram32pairs368640interface245760lifetime_bytes':engram['logical_value_and_lifetime_model']['calls_with_each_observed_site']==32 and engram['logical_value_and_lifetime_model']['one_removed_conversion_interface_bytes']==368640 and engram['logical_value_and_lifetime_model']['additional_live_FP32_storage_at_most_bytes_per_call']==245760,
 'aggregate128complete_role_shape_buckets_all158928calls':ind['per_process_role_M_buckets']==128 and ind['bucket_cardinality_equal'] and ind['native_calls_covered']==158928,
 'all_target_MAC_work_matches_prior_seal':ind['same_target_Q_matches_v169'] and ind['captured_Q_time_match_v169'],
 'aggregate_draft11088_prefill2560_small145280':agg['aggregate']['draft']['native_records']==11088 and agg['aggregate']['target_prefill']['native_records']==2560 and agg['aggregate']['target_small']['native_records']==145280,
 'aggregate_ref_efficiency_prefill_over_small13p448':abs(agg['target_prefill_over_small_row_direct_MAC_time_efficiency_ratio']-13.448235113599978)<1e-12,
 'no_new_eager_per_call_pairings_or_globalFIFOclaim':agg['new_per_call_eager_associations']==0 and ind['role_M_aggregate_proof_does_not_need_crossrank_iteration_matching'],
 'all_owned_CPU_jobs_terminal_service_unchanged_healthy':all(not x['exact_process_live'] and x['state'] in ['RETURNED','ERROR'] for x in receipt['jobs']) and receipt['running'] and receipt['health_http']==200 and receipt['started_at']=='2026-11-02T19:01:54.649699535Z',
 'no_new_device_work_or_model_reload':receipt['new_device_work_or_model_reloads']==0,
}
assert all(checks.values()),checks
write(O/(prefix+'-preflight-v170.json'),{'status':'PASS','preflight':True,'checks':checks,'basis':'Existing profile typed relations, exact aggregate-set coverage and declared operator semantic/reference-capacity domains. Not physicalMFU/whole-scope closure.'})
write(O/(prefix+'-inputs-v170.json'),{'inputs':[pin(p) for p in sorted(H.rglob('*')) if p.is_file() and '__pycache__' not in p.parts],'overall_complete':False})
summary={'status':'PROGRESS','previous_goal_turn':'PROGRESS','overall_complete':False,'new_native_operand_or_phase_promotions':0,'new_whole_module_closures':0,'new_per_call_eager_associations':0,'operator_domain_refinements':4,'typed_profile_counts':{k:typed['review'][k] for k in ['callsite_groups','typed_regimes','public_CPU_calls','native_executions']},'resource_census':resource['phase_census'],'MC2_complete_aggregate':agg['aggregate'],'reference_capacity_efficiency_order':ind['ordering_low_to_high'],'target_prefill_over_small_ratio':agg['target_prefill_over_small_row_direct_MAC_time_efficiency_ratio'],'Engram_cast_interface_bytes':368640,'Engram_extended_lifetime_bytes':245760,'new_independence_proofs':0,'whole_modules_closed':0,'new_device_work_or_model_reloads':0}
write(O/(prefix+'-summary-v170.json'),summary)
action=(
 '继续完整目标，当前checkpoint170。v169和v170都是实质PROGRESS。直接读PROFILE-FINDINGS-v170、typed-operator-cuts-operator-overlay-v170、coverage-work-queue-v170和shared-profile-analysis-v170/typed-operator-question-review，不重放全部旧源代码。'
 '本轮已用一次4.39秒indexedCPU pass补120callsite/148typedregime/15520publiccalls/21088native执行，四个公开算子域已条件分析完成：BF16argmax+INT32索引cast64、960次INT32->INT64floorDiv->INT32、选4ID的Markovembedding、Engram两次hidden.float的局部fork。不要把缺少实现或内部私有workspace参数再次当这四个条件结论的门槛。'
 'Engram只读一个已冻结27行函数，samehidden未修改，预旋转FP32值可供rotation和最终residual共用；不能拿旋转后的original替代residual。368640B/call转换接口预算，对应最多245760B/call额外FP32存活，按共同内存预算计，不重复算idealGate/wholefusion收益。'
 '32实际DB资源清点已完成：startup54336和runtime12188947条COMPUTE关联，后者另14088140TASK无COMPUTE元数据，非零工作。SparseFlashMlaMetadata与QLImetadata导出AI_CPU/KERNEL_AICPU，不用Cube峰值；通信AivKernel是KERNEL_AIVEC，不能与Vector重算假定免费并行；同名QuantMatmul可分AI_CORE/MIX_AIC，dtype/name不等于完整执行mode。全Case含大量observer开销，不能以该总表排名生产模块MFU或把4倍F21计数的Transpose仅凭算术相等判定observer。'
 '新聚合方法替代无必要的eager逐对关联：16pid的(role,M)共128桶source/native全覆盖且每桶数量相等，native W1/W2验证targetE24/draftE8且H5120/D2304，合计158928调用。可分别求和必要source-domain receiver MAC和native时间，不必配每个eager调用，也不依赖跨rankiteration对齐。targetsmall145280条Q90863094988800MAC/T51.949889895s；prefill2560条Q223531027660800/T9.503194703s；draft11088条Q2868137164800/T2.640268027s。以上T是跨rank/call累计kernel时间，不是Case时延。'
 '共同常数每device参考P_ref下，direct-MAC/full-F21时间效率draft<targetsmall<prefill，prefill/small=13.448235。物理MFU/wholemodule最低/真实提速仍未证，但聚合比较已经有效，不再因13648个eager未逐条配对卡这项结论。源算法/返回计数stream正确性和共同参考容量为声明前提，未冒充loadedbinary指令证明。'
 '下一步优先补当前171 actualrank/logicaldevice到physicalchip/board和actualmode/capacity的关键桥：用已存runtime/currentSDK查询及同可见性环境的只读device映射，不重复归档decoder/框架研究。以新当前cohort为域，不伪回填旧166。可先给清楚的参考mode条件上下界/模块比较，不因缺绝对fmax而阻止其他已定算子域收口。'
 '结合v165无observer窗口、v166通信邻域与当前已索引callsite推进全26模块统一必要工作/理论下界/利用率对照；保持采集域分离，不把v170计数硬配到旧production时间。只为影响结论的typedrow/operand/generation/readiness条件再查profile。完整35+7候选、26模块358语义行、全部条件分支和跨算子/层/卡/迭代融合拆分重组不缩减；未知allreader/numerics/状态/copy/control/物理容量和最低MFU仍保留。历史9/1352等不伪补，也不是每个新公开语义域的通用门槛。'
 'firstCPU查询因时间字段TEXT减法失败已保留，确证旧PID终止后int解析修复，成功4.39秒完成16profile；资源清点155.91秒、F21分桶4.24秒，均terminal。171原服务health200且StartedAt未变，无NPU负载/reload。本用户已授权探针和容器推理进程操作，不重复询问。')
gate={'version':170,'parent':pin(Path(old['source_gate'])),'policy':'Use typed profile/callsite evidence, wholepublicsemantic boundaries and complete-set aggregation before per-call forensic pairing. No newfullCase or frameworksurvey.',
 'resolved_questions':[{'question':'Does draft argmax convert score planes toFLOAT?','answer':'No inthese160calls:BF16Slice->BF16ArgMax->INT32index->INT64indexCast. Preserve scorebank/readers and analyze this publicoperator boundary independently of ID-onlywholeMarkov fusion.'},{'question':'Are wholepublicinteger/embedding/privateplanes obligatory?','answer':'The960quotientcalls materialize64bitplanes andselectedIDembedding materializes4IDs. Wholepublicsemantic replacements maykeep arithmetic/registerprecision andremove mandatory globalplanes underexpliciterror/storageguards;individualnative deletion notclaimed.'},{'question':'Are twohidden casts the same residual value afterrotation?','answer':'Both read originalhidden,butrotatedoriginal isdifferent. Retainpre-rotationFP32fork andprice extendedlive245760B.'},{'question':'Must every eager execution be paired before aggregate work/time comparison?','answer':'No.128complete(pid,role,M)buckets coverall158928source/nativecalls. Separate sums yield a valid declared-domain aggregate;globalcallordinalmatching is unnecessary for thisquestion.'},{'question':'Can every COMPTASK time be normalized byCubepeak?','answer':'No.32DBexportedtaxonomy separatesAICPU/Vector/Core/Mixed/COMM;labels are not independentcapacitypools anddiagnostic totals are notproductionMFU.'}],
 'next_targeted_questions':['Bind current171 physicaldevice pool/mode/capacity via currentSDK/query metadata,not identicalnumericlabels alone.','Build uniform module work/reference-capacity comparison using alreadyavailable operator contracts andscope-correct profiles.','Resolveonlyremainingconclusion-changingconditions acrossoriginalmodule/branch inventory.'], 'all_missing_fields_remain_UNKNOWN':True,'overall_complete':False}
overlay={'version':170,'parent':pin(Path(old['evidence_overlays'][-1])),'affected_candidates':['F21','F25','F30','F31','F35'],'affected_modules':['M12','M14','M17','M19','M20','M21','M23'],'typed_frontier':pin(H/'typed-frontier-observation.json'),'operator_domain_review':pin(H/'typed-operator-question-review.json'),'argmax_boundary':pin(H/'draft-argmax-public-boundary-refinement.json'),'integer_boundary':pin(H/'metadata-integer-public-boundary-refinement.json'),'selectedID_embedding':pin(H/'Markov-embedding-strided-ID-refinement.json'),'Engram_cast_fork':pin(H/'Engram-hidden-cast-fork-semantic-cut.json'),'resource_taxonomy':pin(H/'resource-domain-routing-review.json'),'complete_aggregate_work_time':pin(H/'mc2-aggregate-work-review.json'),'aggregate_math_review':pin(H/'MC2-aggregate-independent-review.json'),'new_operator_domain_refinements':4,'new_per_call_eager_associations':0,'new_pair_clauses':0,'new_independence_proofs':0,'new_native_operand_or_phase_promotions':0,'historical_native_or_phase_promotions':0,'new_whole_module_closures':0,'whole_modules_closed':0,'overall_complete':False}
queue=deepcopy(read(Path(old['work_queue'])));queue.update(version=170,parent=pin(Path(old['work_queue'])))
for row in queue['work']:
 if row['id']=='Q169-OPERATOR-FAMILIES':row.update(state='FOUR_ACTUAL_TYPED_PUBLIC_OPERATOR_DOMAINS_REVIEWED',action='Reuse fourconditionalreviewedcuts;do notreread/rederiveunlesspremiseschange. Continue otheroriginalmodulebranches andphysical/referencecapacity comparison. Engramforkisactivationlifetime tradeoff,notnewparametercache;embedding64Binterfaceislowprioritybybytes.',remaining='Wholevariant/nativeimplementation/allreader/physicalceilings remainseparate. Existingmetadata/Markov/Engramclauses reused,notnovelcounted.',evidence=[str(H/'typed-operator-question-review.json'),str(H/'Engram-hidden-cast-fork-semantic-cut.json')])
 if row['id']=='Q165-MOE-TIMING':row.update(state='ALL158928_AGGREGATE_ROLE_WORK_TIME_COMPLETE_IN_DECLARED_DOMAIN',action='Use128complete(pid,role,M)buckets;aggregateQ/time doesnotneedeagerper-callmatching orcrossrankiterationmatching. CommonP_refefficiencyorder:draft<targetsmall<prefill,prefill/small13.448235. Onlydoadditionalpercallassociationifitchangesawaiting/causeconclusion.',remaining='Physicalmode/peak/clock andproductionMFU/wholemodulelowest notbound;globalcallidentity onlyneededforquestionsthatusejointper-iterationgroups.',evidence=[str(H/'mc2-aggregate-work-review.json'),str(H/'MC2-aggregate-independent-review.json')])
 if row['id']=='Q155-PHYSICAL-POOL':row.update(state='CURRENT32DB_RESOURCE_TAXONOMY_BOUND_ACTUAL_PHYSICAL_QUERY_NEXT',action='Prioritize current171 rank/logicaldevice-to-physical mapping in matchingvisibility/currentSDK. UseexactreportedAICPU/Core/Vector/Mixed/COMMroles;do notassumeseparatepools/freeoverlap orCubepeakforinteger/comparison. State reference-mode conditions whereenough,notuniversalcertifiedfmaxasgateforeveryoperator.',remaining='Currentphysicalmapping/math-mode/capacityupper andwholemoduleMFU remainopen;historical166notbackfilled.',evidence=[str(H/'resource-domain-routing-review.json'),str(H/'resource-taxonomy-observation.json')])
 if row['id']=='Q167-JOINT-COUNT-READY':row.update(state='ALL_F21_ROLE_SHAPE_AGGREGATES_VALIDATED_NO_RECOLLECTION',action='Reusecomplete128bucketcoverage andallsourcecounter/nativeaggregates. Per-callassociationsremain145280;the13648eagergap no longerblocksaggregate directMACtimecomparison.',remaining='No originalCase reprofile/re-export neededforcurrentsuccessfulanalysis. Currentdiagnosticwork/costisnotproductiontiming.')
queue['new_evidence']=['120actualcallsitegroups/148typedregimes/15520publiccalls/21088nativeexecutions in4.39CPU seconds;firstTEXTparserfailure preserved/corrected afterterminalverification.','Fourpublicoperatordomainrefinements:BF16argmax/indexcast,960INT32quotientcalls,4IDMarkovembedding,and32pairedhidden.floatsiteswith245760Bpotentialextendedlifetime.','32DBtaxonomy:54336startup and12188947runtimeCOMPUTEassociatedtasks;AICPUmetadata/COMM-on-AIV/samename-multiresource facts prevent wrongcapacitynormalization.','128complete(pid,role,M)bucketscoverall158928F21calls;aggregateQ/time validwithouteagerpercallorcrossrankordinalmatching.','Commonreferencecapacity directMACtimeefficiency:draft<targetsmall<prefill;prefill/small13.448235,notphysicalMFU/requestthroughput/speedup.','AllCPUjobsnowterminal;171originalservicehealth200/unchangedStartedAt,no newNPUload/modelreload;16remoteoutputfilespinned.']
# Physical binding now has the highest conclusion-changing priority.
queue['work'].sort(key=lambda r:0 if r['id']=='Q155-PHYSICAL-POOL' else 1)
recs=deepcopy(read(O/'current-obligation-reconciliations-v169.json')['reconciliations'])
for row in recs.values():row.update(v170_evidence_delta={'operator_overlay':str(O/(prefix+'-operator-overlay-v170.json')),'scope':'Fourtypedpublicoperatorsemanticrefinements,32DBresourceclassification,andcompleteF21aggregatecomparison. Eagerpercall/globalordinalmatchingnotrequiredforaggregate;physicalMFU/allreader/wholemoduleclosuresnotinferred.'},whole_domain_closed=False)
write(O/'current-obligation-reconciliations-v170.json',{'parent':pin(O/'current-obligation-reconciliations-v169.json'),'reconciliations':recs,'does_not_delete_or_narrow_original_requirements':True,'historical_forensic_gap_not_universal_prerequisite':True,'overall_complete':False})
report='''# v170：实际 dtype、公开算子边界和完整集合聚合

本轮复用现有 profile，没有新增 NPU 负载、模型重载或完整 Case。171 原服务 health200，StartedAt 未变。上一轮 v169 是实质进展。本轮同样 PROGRESS，完整目标尚未完成。

一次索引查询在4.39秒内补齐120个实际调用点组、148个dtype/shape执行分支、15,520次公开CPU调用、21,088条原生执行。复用已存调用栈与AST索引，没有重读52个源文件。首个查询因CPU时间字段为十进制TEXT、直接相减而退出；已保留错误与旧代码，并确认旧PID结束后改为int解析。成功查询不是模型或采集重试。

四项算子域得到具体补充：

1. Draft argmax：160次精确公开调用均为BF16 Slice→BF16 ArgMax→INT32索引→INT64索引Cast，没有输入FLOAT平面升级。单独替换该公开argmax边界可以保留整个已修正score bank及其其他消费者，不要求所有raw/confidence/logprob消费者死亡。这不同于已分析的整条Markov ID-only融合。给定M4,V129280，在声明的全扫描、冷外部数组、fresh输出模型中，三kernel边界最低逻辑数组访问参考为3,102,784B，组合后的必要外部接口为1,034,272B；不是HBM测量或可达提速。同Python行还有160次独立Slice，没有凭数量相等偷偷与本公开调用相配或加进预算。保持BF16比较、tie/NaN、早先bias/Add的BF16舍入和所有输入输出语义。

2. Metadata整数：960次实际INT32商调用分为5种typed regime，都展开为INT32→INT64、INT64 floorDiv、INT64→INT32。对INT32输入和正INT64除数，数学floor商必在INT32范围，解析边界和405个整数边界样例已核验。整体公开算子内部的全局Cast平面不是语义必需，计算内部仍可保持INT64；未宣称存在更快INT32原生指令。负数floor不能改成截断，INT32_MIN的abs或q*d溢出不能忽略，INT64 slot/position不能仅凭maxlen缩窄。声明数组模型40N→必要接口8N，不是物理带宽。联合q/r仍需输入值、除数和generation相同的既有条件；shape相同不证明这些条件。

3. Markov embedding：160次实际调用只选择4个ID，输出4×256 BF16；权重形状129280×256不表示必须读取全表。消除公开调用内部ID Slice的逻辑接口预算仅64B/次，不能因“两kernel”而列为高收益。冷表必要权重读量取决于不同ID数U，模型为8B+2UR+2BR；实际ID重复和缓存未知。依赖的后续循环ID和confidence embedding输出保留。

4. Engram转换分叉：profile显示同一函数line62和72各32次BF16[3,4,5120]→FLOAT。只定点读取已冻结的27行函数，确认hidden在函数内未修改；可让同一预旋转FP32值供旋转/门控和最终原始残差分支使用。旋转后的original不能直接用于残差；[1,2]交换旋转、gate×value=0即给出反例。该局部复用预算是368,640B/次转换接口，同时可能延长245,760B/次即240KiB的FP32激活存活。必须和其他候选共用内存预算，保留完整norm/dot/mask/FP32乘加/BF16最终舍入，不能与既有idealGate或整函数融合重复记账。q/k参数和rotation权重缓存是旧结论，不计新发现。

32份实际数据库的资源清点也已完成，耗时155.91秒：启动域54,336条、完整诊断负载12,188,947条TASK有COMPUTE关联；运行域另有14,088,140条TASK没有此元数据，不能记为零工作。导出资源共六组：Core、Vector、AICPU、MIX_AIC、MIX_AIV、COMM。SparseFlashMlaMetadata和QLI metadata实际导出AI_CPU/KERNEL_AICPU，不能用INT8/BF16 Cube峰值归一化。通信AivKernel的native类型为KERNEL_AIVEC，不能假设通信与Vector重算资源独立或免费重叠。同名QuantMatmul也同时出现AI_CORE与MIX_AIC，名称/dtype不足以证明完整执行模式。导出标签不是互相独立的容量池证书。

该全Case是带观察器的诊断域。新增pack/write会增加大量Vector任务并改变等待，整表不能直接用于生产模块MFU排名。InplaceScatterTranspose计数635712恰为158928次F21的四倍，只是污染警示，未把算术相等当逐native观察器身份。TASK累计时间包含等待、跨rank和潜在重叠；不是请求关键路径。

另外，一项不必要的配对门槛已移除：为计算同域聚合工作量/时间，不需要给每个eager调用逐条找到GPU计数。使用16个进程的(role,M)共128桶，源记录与native记录的键集合、每桶数量完全相等。native W1/W2验证target24个本地专家、draft8个本地专家，H5120/D2304一致；不存在额外或缺失桶。v168已有完整计数、无缺行/overflow且capacity65536未裁剪。此处在当前direct-dense源算法与返回计数stream正确性的域内，将receiver行数乘35389440 MAC，并分别汇总同一完整桶的原生时间即可；不要求逐调用配对或跨rank iteration身份。源/二进制匹配和实际物理mode仍是单独条件。

| F21域 | 调用数 | direct MAC | 累计完整native内核时间 | MAC/累计内核秒 |
| --- | ---: | ---: | ---: | ---: |
| Draft | 11088 | 2868137164800 | 2.640268027s | 1.0863 TMAC/s |
| Target小行数 | 145280 | 90863094988800 | 51.949889895s | 1.7491 TMAC/s |
| Target prefill | 2560 | 223531027660800 | 9.503194703s | 23.5217 TMAC/s |

全部158,928次调用均进入完整聚合。目标总MAC与v169相等，captured小行数Q和T与v169逐条关联结果相等。独立整数交叉乘法核验排序。共同常数每device参考容量P_ref下，eta_ref=Q/(P_ref×累计完整F21时间)，其效率排序为draft<target小行数<prefill，prefill/小行数为13.448235；未知公共P_ref在比值中消去。它说明该诊断域的完整F21 direct-MAC时间效率差异，不是实际物理MFU、整个M19/M12模块排名、集群/请求吞吐或可达提速。每个eager调用的等待原因仍可另查，但13648条未逐条关联不再阻塞此聚合结论。跨rank分组身份只在使用逐迭代联合组的其他问题中仍保留为前提。

下一步优先补当前171逻辑device到physical chip/board、执行模式和容量口径的关键桥，再继续全26模块一致理论/利用率对照。使用现有SDK和匹配可见性环境的只读查询，条件参考模式足够时先给条件结论，不把缺绝对fmax或历史私有字段当所有算子分析的门槛。当前量化计数不能硬配到旧无observer窗口时间。四项公开语义条件分析已可复用；缺实现或benchmark本身不是重新开展分析的理由。

完整35+7候选、26模块358语义行、全部条件分支和跨算子/层/卡/迭代融合、拆分、重组继续保留。all-reader、数值、状态/回滚、copy/control、实际物理容量及最低模块MFU仍开放；未伪补旧9/1352、849PMU、16phase、3136归属，也不让旧缺口支配所有新域。所有本轮CPU作业已结束，16份远端派生文件有SHA256。整体未完成。
'''
jobs=[{'kind':'review','status':'error','description':'FirstCPU-onlytypedquery failed becausePYTORCH_API startNs/endNsareTEXT;oldPIDterminalverified,oldsource/errorpreserved. No device/model retry.','artifacts':[H/'typed-frontier-error-first-run.json',H/'first-job-terminal-receipt.json',H/'CPU-time-field-diagnostic.json']},
 {'kind':'review','status':'pass','description':'Correctedindexedtypedquery completes16profiles in4.39seconds:120callsitegroups,148typedregimes,15520publiccalls,21088nativeexecutions;no sourcesurvey/devicework.','artifacts':[H/'typed-frontier-observation.json',H/'typed-operator-question-review.json']},
 {'kind':'review','status':'pass','description':'Fourpublicoperatordomainrefinements:BF16argmax/indexcast64,960integerquotientcalls,4IDembedding,andEngramunchangedhiddenconversionforkwithpricedlifetime. Conditionalnumeric/storageguards explicit;no kernel implementation.','artifacts':[H/'draft-argmax-public-boundary-refinement.json',H/'metadata-integer-public-boundary-refinement.json',H/'Markov-embedding-strided-ID-refinement.json',H/'Engram-hidden-cast-fork-semantic-cut.json']},
 {'kind':'review','status':'pass','description':'32DBactualresourceinventory:54336startup/12188947runtimeCOMPUTEassociatedtasks;AICPUmetadata,COMM-on-AIV andsame-name mixed/Core roles;no productionMFU inference.','artifacts':[H/'resource-taxonomy-observation.json',H/'resource-domain-routing-review.json']},
 {'kind':'review','status':'pass','description':'All158928F21calls aggregate128complete(pid,role,M)buckets;work/timing independentofeagerpercallorcrossrankordinalmatching;commonreferenceP_refefficiencyorder and13.448235prefill/smallratio reviewed.','artifacts':[H/'mc2-aggregate-observation.json',H/'mc2-aggregate-work-review.json',H/'MC2-aggregate-independent-review.json']},
 {'kind':'review','status':'pass','description':'AllownedCPUjobs terminal;original171servicehealthy200/unchangedStartedAt;no newdevicework/reload;16derivedremoteartifacts pinned.','artifacts':[H/'final-service-jobs-and-artifact-receipt.json']}]
persist_delta(version=170,previous=169,prefix=prefix,changes=queue['new_evidence'],report=report,action=action,
 remaining_delta='G170:Fourtypedpublicdomains/32DBresourceclassification/completeF21aggregatework-time comparison reviewed. Actualphysicaldevice/mode/capacity andwholemodulelowestMFU,otheroriginalbranches/readers/numerics/state/copy/control remainopen. Eager percall/globalordinalmatching no longerblockscompleteaggregatecomparison.',
 source_gate=gate,operator_overlay=overlay,queue=queue,jobs=jobs,scripts=list(H.glob('*.py')),obligation_reconciliations=recs,
 remote_status='AllCPUanalysis terminal afteroneexplicitTEXTparsercorrection;no newNPUload/modelreload/profilecapture. Original171servicehealth200/StartedAtunchanged;16remoteartifacts pinned. Fulloriginalgoalactive.')
post={'status':'PASS','checkpoint':170,'checks':checks,'manifest_artifacts':len(read(O/'checkpoint-manifest-v170.json')['artifacts']),'predecessor_pins_verified_before_shared_writes':True,'complete_original_scope_preserved':True,'overall_complete':False}
write(O/(prefix+'-postcheck-result-v170.json'),post);write(N/'CURRENT-POSTCHECK.json',post)
write(N/'LIVE-SHARED-COHORT.json',{'effective_checkpoint':170,'status':'EXISTING_COHORT_CPU_ANALYSIS_TERMINAL','receipt':str(H/'final-service-jobs-and-artifact-receipt.json'),'no_live_device_or_analysis_job':True,'overall_complete':False})
write(N/'NEXT-EVIDENCE-PENDING.json',{'effective_checkpoint':170,'status':'CURRENT_PHYSICAL_POOL_AND_UNIFORM_MODULE_COMPARISON_NEXT','next_action':action,'work_queue':str(O/'coverage-work-queue-v170.json'),'overall_complete':False})
print(json.dumps(post,ensure_ascii=False))
