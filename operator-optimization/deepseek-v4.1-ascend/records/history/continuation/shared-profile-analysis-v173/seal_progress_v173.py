"""Seal actual branch/variant ledgers and the bounded operator-only review."""
import json
import sys
from copy import deepcopy
from pathlib import Path
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H));sys.path.insert(1,str(H.parent))
from finalize_origin_coverage_v9 import N,O,read,write
from analyze_evidence_batches_v155 import pin
from persist_operator_review_delta import persist_delta
old=read(O/'REVIEW-CHECKPOINT-v172.json');prefix='operator-scope-reconciliation'
receipt=read(H/'final-health-and-remote-artifacts.json')
checks={**read(H/'lifecycle-review-checks.json')['checks'],**read(H/'joint-review-checks.json')['checks'],**read(H/'current-ledger-preservation-checks.json')['checks']}
checks.update(original_service_healthy_unrestarted=receipt['health_http']==200 and receipt['running'] and receipt['started_at']=='2026-11-02T19:01:54.649699535Z',CPU_query_terminal=not receipt['same_analysis_handle_live'],new_device_work0=receipt['new_application_kernel_model_replay_or_model_reload_calls']==0)
assert all(checks.values()),checks
write(O/(prefix+'-preflight-v173.json'),{'status':'PASS','preflight':True,'checks':checks})
write(O/(prefix+'-inputs-v173.json'),{'inputs':[pin(p) for p in sorted(H.rglob('*')) if p.is_file() and '__pycache__' not in p.parts],'overall_complete':False})
changes=[
'All358originalsemanticrows screened withcurrentevidence:330explicitconditionalreviews reused,12requestfamilies operator-boundaryreviewed,2lifecyclefamilies refined,13partiallabels mapped toconcrete analysis/evidencequestions,1excludedlegacy reference retained. Labels are not runtime applicability or exhaustiveproof.',
'Existing91unreviewed lifecycle sites reviewed once in25operator-effectdomains fromoneidentical cachedworkerbody,drivenbyactualruntime_dummyprofile evidence. Metadata/caches/positions/DPpadding/PP/LoRA/logits/EPLB/draft/capturecleanup effects preserved;900othernavigation associations not silently closed.',
'6.45sCPU query ofcached32profiles returned120actualBOOL24zero executions:56exactCPUcreatorroles(16valid,16C2,16Engramtail,8SP) and64withoutcreator.56MaskedFillINT32outputswithBOOLinput masks were separatelyretained,notzerowrites. Old16680taskidentitynotpromoted.',
'16sourceconnectedINT32[24]dummyindexzerofills confirm96Blogicalvaluewrites/call,1536Btotal. Added conditional no-demand allocation and identical-row head-work reuse variants underexistingF30/F31;zero positiveheadcallbacks doesnotprove nonexecution or current gain.',
'Original358semanticrows preserved withoutoldfieldchanges;twoexplicitvariants extend360,26modules/35model+7frameworkcandidates retained. Currentcandidate,module,branchreview authorities published instead of leaving onlyoldv154labels as thecurrentledger.',
'One12node15edge composedoperatorgraph coversop/layer/card/iteration boundaries and19candidatefamilies. Routed/sharedquantizers keptdistinct;existingF21fusionnotrecounted;timing/resource cuts usemax andnonduplicatework,TP/EPingress/lifetimes notfree;cross-iterationedges haveexplicitoffsets.',
'Unmeasuredhostcost/privateABI/runtimebankbindings remainnamedconditions ratherthanuniversalglobalMFU/implementation gates. Stillneed remainingoperator-effectscope andall42familycompositioncoverageaudit;no wholemodule/goalcompletion.',
'Currentservicehealth200/StartedAtunchanged,CPUqueryterminal,newNPUcontext/inference/replay/reload0. Remoteoriginalqueryandbroadresult preserved;offlineMaskFillcorrection andfirstunassignedLoRAguard check recorded.'
]
write(O/(prefix+'-summary-v173.json'),{'status':'PROGRESS','previous_goal_turn':'PROGRESS','overall_complete':False,'new_native_operand_or_phase_promotions':0,'new_whole_module_closures':0,'original_semantic_rows':358,'current_semantic_rows':360,'module_universe':26,'model_candidates':35,'framework_candidates':7,'new_lifecycle_guard_sites_reviewed':91,'operator_effect_domains':25,'current_BOOL24_zero_source_bound':56,'current_BOOL24_zero_no_creator':64,'new_explicit_conditional_variants':2,'joint_graph_nodes':12,'joint_graph_edges':15,'joint_candidate_families':19,'new_application_kernel_model_replay_or_model_reload_calls':0})
action=(
'继续完整目标，checkpoint173。先读PROFILE-FINDINGS-v173、shared-profile-analysis-v173/13-partial-labels-current-routing、joint-operator-graph-and-resource-review、all358-current-analysis-screen。当前candidate/module/branch权威已更新v173：原358行不改旧字段地保留，新增M21-S19/M16-OP-11两个F31/F30下条件变体，共360；35+7候选/26模块不缩减。'
'v172模块诊断与v173为实质PROGRESS。12MAC模块中M12两个窗口最低、20Vector模块M23低算术活动但承担通信；26P5路由保留。不得重跑完整Case、重复问权限、重读6份相同采样源码或91已审guard来维护记账。'
'原358行screen330继承明确条件分析、12request算子边界新复核、2lifecycle细化、13partial旧标签已具体路由、1legacy排除保留；这些是分析状态，不是实际全部flags/nativegraph attestation。13partial里多数是私有实现/历史epoch/定量性能缺口，不能当所有公开语义候选的统一前置。'
'91生命周期静态点集中一个已匹配SHA的worker体，已按25类算子效应审完。保留metadataNone与FULL/force、request/padding/LoRA/GDN/circular cache/PP/input/copy/position127后reset0/EPLB/draft/metadatawriterrelease/exceptioncleanup。torch profiler开启不等于_dummy_run is_profile=True；dummy可能有remoteexpert work，不能整体跳过。'
'当前profile120次BOOL24zero中56已绑定valid16/C2mask16/Engramtail16/SP8，64无creator仍unknown。另56个MaskedFill的BOOL是输入mask、输出INT32，不是BOOLzero。旧16680prefix身份不回填；相同shape不同角色，不能blanketdelete。16个INT32[24]dummyindices是96B/call共1536Blogicalplane，不是HBM/速度收益；只有需求guard明确才删除。重复row0head reuse为新条件变体，保留全部通信/输出/precision，zero callback witness不是非执行证明，无当前gain信用。'
'组合图12node15edge已审19候选及op/layer/card/iteration：保留QR/QS不同rounding，不默认量化CSE；E已有dispatch/GMM/SwiGLU/return融合，receiver工作用实际receiver分配。跨层HC/EngramFP32fork、跨迭代prefetch/rollback、TP/EP共用ingress和24Cube/48Vector及livebank统一计价，不乘加独立理想speedups。'
'下一步真正覆盖frontier是旧900个runner静态导航关联，它们只是source-site索引，不是900个已证明算子分支，也不应变成900项框架研究。先用当前profile调用点和既有operator合同，对其中会改变算子输入/输出/精度/rowowner/状态/RNG/readiness的控制选择作一次聚合effect审计；仅导出那些未被现有合同覆盖的具体调用/写入/guard。保持原无关控制流，不研究所有API网络异常。未匹配profile不等于不可达；也不能靠文件名路由宣称全审完。'
'再把全部35+7候选族纳入共同guard/资源/生命周期/互斥模板，复用已有212clauses，补剩余23族中真正未审的组合约束而非笛卡尔配对。必要时对会改变所选候选结论的实际字段做一次合并探针，已获授权，不重复确认。保留完整范围，仍有未分析时不能因报告完成标Goal complete。'
'本轮仅缓存profile CPU查询和本地定点source/图分析，health200、StartedAt不变、所有分析terminal。Goal ACTIVE/PROGRESS。')
gate={'version':173,'parent':pin(Path(old['source_gate'])),'policy':'PROFILE_FIRST_OPERATOR_EFFECT_ONLY_CACHED_IDENTICAL_SOURCE','answered_frontier':str(H/'91-guard-operator-dataflow-review.json'),'current_roles':str(H/'BOOL24-current-source-callsite-bindings.json'),'named_partial_questions':str(H/'13-partial-labels-current-routing.json'),'next_source_gate':'One explicit operator input/output/state/precision/ownership/readiness question from the remaining navigation frontier or remaining family composition;do not reread reviewed bodies or infer impossibility/independence from absence.','no_new_device_work':True,'no_cartesian_pairs':True,'overall_complete':False}
overlay={'version':173,'changes':changes,'updated_authorities':read(H/'current-ledger-preservation-checks.json')['authorities'],'source_effect_review':str(H/'91-guard-operator-dataflow-review.json'),'current_semantic_rows':360,'all_original_rows_preserved':True,'joint_graph':str(H/'joint-operator-graph-and-resource-review.json'),'new_variants':str(H/'two-dummy-demand-variant-cuts.json'),'runtime_applicability_of_unknowns_not_promoted':True,'new_whole_module_closures':0,'overall_complete':False}
queue=deepcopy(read(O/'coverage-work-queue-v172.json'));queue.update(version=173,parent=pin(O/'coverage-work-queue-v172.json'),semantic_rows=360,new_evidence=changes)
queue['work']=[
{'id':'Q173-REMAINING-OPERATOR-EFFECT-FRONTIER','state':'BOUNDED_NAVIGATION_TO_OPERATOR_EFFECT_AUDIT_NEXT','action':'Use currentprofile callsites andcachedsameSHAAST toaggregate original900remaining runnernavigation associations intooperator inputs/outputs/precision/ownership/state/RNG/readiness selections. Export only previouslyuncovered concreteeffects;preserve unrelatedframeworkcontrolflow. Do not re-open91reviewedlifecycle guards or infer unreachability from absent profileframes.','evidence':[str(H/'91-guard-operator-dataflow-review.json'),str(H/'all358-current-analysis-screen.json'),str(H/'13-partial-labels-current-routing.json')],'remaining':'Currentfinitegraph/branchuniverse exhaustiveness notproved;needthisactualoperator-effectaudit ratherthan morestatusoverlays.'},
{'id':'Q173-ALL42-COMPOSITION-COVERAGE','state':'19FAMILIES_IN_SELECTED_JOINTGRAPH_REMAINING_FAMILY_TEMPLATES_TO_RECONCILE','action':'Mapremaining23families tocommonrequiredwork/resource/guard/lifetime/alternative templates usingexisting212clauses. Examineonlyuncoveredinteractionpremises;noCartesianpairs or sumofidealspeedups.','evidence':[str(H/'joint-operator-graph-and-resource-review.json')],'remaining':'Onejointgraph doesnotclaimall42candidatefamiliesreviewed together.'}
]+[x for x in queue['work'] if x['id']!='Q172-FIXEDCASE-BRANCH-AND-COMPOSITION-AUDIT']
base=read(Path(old['coverage_obligations']))['obligations'];recs={x['id']:{'current_dispatch_policy':'Operator-effect/guard/resourcefamily audit,not genericframework research or Cartesianpairs. Reuse current reviewedconditional answers.','current_effect_frontier':str(H/'91-guard-operator-dataflow-review.json'),'current_named_questions':str(H/'13-partial-labels-current-routing.json'),'whole_domain_closed_by_this_increment':False,'original_broad_scope_preserved':True} for x in base}
for k in ['G00','G02','G09','G13']:recs[k].update(current_semantic_rows=360,all358_original_rows_preserved=True,screen=str(H/'all358-current-analysis-screen.json'),remaining='Remaining900staticnavigation associations require operator-effect aggregation;notall runtimeapplicabilitybound.')
for k in ['G05','G06','G07','G08','G10','G14']:recs[k]['joint_nonduplicate_work_resource_lifetime_review']=str(H/'joint-operator-graph-and-resource-review.json')
recs['G12'].update(reuse_current_declared_grouped_diagnosis=True,universal_physical_MFU_not_protocol_prerequisite=True,authority=str(N/'shared-profile-analysis-v172/all26-module-metric-and-P5-routes.json'))

report='''本次继续算子范围审计，整体 PROGRESS，未完成全部目标。

原 358 条语义行已逐条保留并筛查：330 条已有明确条件分析，12 类请求语义补成算子边界约束，2 类生命周期范围补充分析，13 个旧 partial 标签逐项写出已有结论与真正缺证，另 1 条旧路径排除记录保留。这个分类不等于实际运行 flags、所有 native 参数或图覆盖均已证明。

此前未审的 91 个生命周期静态点集中于少数函数，本次沿 profile 里实际出现的 runtime dummy 算子，只读一个 hash 未变的 worker 片段，归约为 25 类输入形状、metadata、缓存、position、DP/PP、LoRA、logits、EPLB、draft、同步和清理效应。每个旧点恰好映射一个分析域，保留条件两侧的算子后果及融合/拆分/重组限制，没有研究整个框架。

关键区别包括：torch profiler 正在采集不等于 _dummy_run 的 is_profile=True；metadata 的 base predicate 实际由 force_attention 或 FULL 决定。Dummy 仍可做真实跨卡 receiver 专家工作、发布 KV/slot/状态，不能整体删除。设备和 CPU position127 的后续 reset0 需保留消费与恢复顺序；metadata executor 的 queued writer 与旧 reader 都可能阻止 bank 重用。capture finally 的状态清理也不能丢失。

6.45 秒的既有 profile CPU 查询取得 120 次 BOOL[24] 清零。其中 56 次有精确 CPU creator，分别是 valid mask16、C2 mask16、Engram 尾 mask16、SP mask8；其余64次保持 unknown。相同形状并不表示同一个字段。另有56个 MaskedFill 的 BOOL 是输入 mask，输出是 INT32，已单列为消费者，不能误算 BOOL 清零。初版宽筛记录和修正均保留，没有为了修正筛选重跑模型；当前来源也不反向宣称旧166的80个 prefix 身份已识别。

本次明确补入两个既有 F30/F31 下的条件变体：一是 dummy logits 没有真实需求时省去零索引准备；二是在所有 index 恒为0、算术逐行可分且保留全部通信/输出/精度协议时，复用重复隐藏行的 head 计算。16次 source-connected INT32[24] 初始化对应每次96B、合计1536B逻辑值写入，属于低优先级小平面，不是 HBM 或提速。当前没有正向 dummy head callback witness，不能由这个空白判断未执行，更不计实际收益。R行重复 head 的 direct 源算法工作由RKV变成KV加输出复制/原通信，仅是条件工作比较。

覆盖台账现已实际更新，而非只追加叙述：原358行旧字段完整保留，加入 M21-S19 与 M16-OP-11 后共360行；26模块、35模型候选、7框架候选不变。新的 candidate/module/branch 三份台账均有当前审计、P5指标路由和缺证条目，旧 v154 保留为不可变祖先。

已审一张12节点15边的保守组合图，涉及19个候选族并覆盖算子、层、卡、迭代。QR/QS保留各自不同量化舍入；不能因读同一hidden就合并INT8值。F21已融合dispatch、quant、双GMM、SwiGLU与return，不重复计新融合收益。跨卡重分配、同expert行调度、V2 pad优化是有不同成本/前提的选择，不能相乘或叠加理想提速。跨层HC和Engram预旋转FP32 fork、跨迭代accepted/rollback/prefetch与metadata banks置于同一资源/存活期账本。

组合下界按必要且不重复工作和真实共享池重算：每设备24Cube/48Vector；不同mode时间在无共同issue证明时取max；TP与EP同卡入口不免费，必要信息可能转发/复用，未证不重复字节时不能把cut直接相加。物理fmax/带宽未知处保留符号，1800MHz仅参考；总显存不是可用free，extended FP32/预取/旧bank和queued writer存活必须一起计。不同cohort的receiver work和PMU时间不硬配。

仍有真正未完成范围：旧900个runner静态导航关联尚未按算子效应证明覆盖，不能仅凭文件名路由宣布全审完，也不应变成900项框架研究。下一步用当前profile调用点与现有合同，一次聚合出仍会改变算子值、rows、精度、所有权、状态、RNG或就绪的具体选择；无关控制流保持原样。还需把组合图之外的23个候选族映射到共同约束模板并补未审前提，复用原有关系条款，不做笛卡尔配对。

未知 native operand、历史 epoch、私有 helper 删除和未测 host 因果成本均有具体适用边界。它们不能自动成为所有公开语义替换的前置，也不由空白推出独立。必要的新探针按已获授权直接执行，但只收会改变所选候选结论的字段。

服务health200，StartedAt未变；本次仅缓存profile查询、本地定点源码和组合分析，无NPU探针、模型推理、replay、重启或重载。查询进程终止。全部目标继续ACTIVE。
'''
jobs=[
{'kind':'review','status':'pass','description':'91existinglifecyclefrontiersites reviewedin25operator-effectdomains;all358semanticrowsscreened and13partiallabels explicitlyrouted;operator scopepreserved.','artifacts':[H/'91-guard-operator-dataflow-review.json',H/'all358-current-analysis-screen.json',H/'13-partial-labels-current-routing.json',H/'lifecycle-review-checks.json']},
{'kind':'review','status':'pass','description':'Cachedprofiles bind56of120BOOL24zerostofourroles;64unknownretained.16INT32index24zerowitnesses;two conditional F30/F31 variants,no currentgain/no absence proof.','artifacts':[H/'lifecycle-profile-witnesses.json',H/'BOOL24-current-source-callsite-bindings.json',H/'two-dummy-demand-variant-cuts.json']},
{'kind':'review','status':'error','description':'Preserve broadquery candidateMaskedFill classification correctedoffline56consumers,andinitialmissingLoRAguard caughtby91sitebijection check;no repeatedmodelrun.','artifacts':[H/'lifecycle-profile-candidate-witnesses-first.json',H/'query_lifecycle_profile_witnesses-first.py',H/'lifecycle-review-checks.json']},
{'kind':'review','status':'pass','description':'12node15edge jointgraph covers19families acrossop/layer/card/iteration withnonduplicatework,distinctquantizers,sharedphysicalresources/lifetimes andnextiterationoffsets.','artifacts':[H/'joint-operator-graph-and-resource-review.json',H/'joint-review-checks.json']},
{'kind':'review','status':'pass','description':'Threecurrentreviewauthorities updated;all358oldbranchfields preserved,onlytwoextravariants,total360/26modules/35+7candidates.Original171servicehealthy,CPUqueryterminal,newdevicework0.','artifacts':[H/'current-ledger-preservation-checks.json',O/'candidate-profile-ledger-v173.json',O/'module-profile-ledger-v173.json',O/'module-phase-branch-matrix-v173.json',H/'final-health-and-remote-artifacts.json']}
]
persist_delta(version=173,previous=172,prefix=prefix,changes=changes,report=report,action=action,
remaining_delta='G173:All358rowscurrent-screened,91lifecycleguardsreviewedin25operator-effectdomains,new2variants=>360currentledger. CurrentBOOL24roles56bound64unknown. Onejointgraph19families;remaining900navigationoperator-effectfrontier and23familycompositioncoverage stillopen;do notreplaceactualauditwithmorestatusoverlays.',
source_gate=gate,operator_overlay=overlay,queue=queue,jobs=jobs,scripts=list(H.glob('*.py')),obligation_reconciliations=recs,
remote_status='Cachedprofilequery6.45sCPUterminal;noNPUinference/reload. Originalservicehealth200/unchangedStartedAt. Fullgoalactive.',authority_overrides=read(H/'current-ledger-preservation-checks.json')['authorities'])
manifest=read(O/'checkpoint-manifest-v173.json');assert all(pin(x['path'])==x for x in manifest['artifacts'])
cp=read(O/'REVIEW-CHECKPOINT-v173.json');assert all(cp[k]==v for k,v in read(H/'current-ledger-preservation-checks.json')['authorities'].items())
post={'status':'PASS','checkpoint':173,'checks':checks,'current_authority_paths_verified':True,'manifest_artifacts':len(manifest['artifacts']),'all_manifest_pins_verified_after_seal':True,'complete_original_scope_preserved':True,'overall_complete':False}
write(O/(prefix+'-postcheck-result-v173.json'),post);write(N/'CURRENT-POSTCHECK.json',post)
write(N/'LIVE-SHARED-COHORT.json',{'effective_checkpoint':173,'status':'CACHED_CPU_OPERATOR_EFFECT_REVIEW_TERMINAL','receipt':str(H/'final-health-and-remote-artifacts.json'),'no_live_device_or_analysis_job':True,'overall_complete':False})
write(N/'NEXT-EVIDENCE-PENDING.json',{'effective_checkpoint':173,'status':'REMAINING_OPERATOR_EFFECT_AND_ALL42_FAMILY_COMPOSITION_FRONTIER','next_action':action,'work_queue':str(O/'coverage-work-queue-v173.json'),'overall_complete':False})
print(json.dumps({'status':'PASS','checkpoint':173,'manifest_artifacts':len(manifest['artifacts']),'current_semantic_rows':360,'current_authorities_verified':True,'overall_complete':False}))
