from copy import deepcopy
from datetime import datetime,timezone,timedelta
import json,hashlib
from finalize_origin_coverage_v9 import N,O,R,W,T,L,read,write,ctl

def main():
 assert not(O/'REVIEW-CHECKPOINT-v23.json').exists()
 now=datetime.now(timezone.utc);stamp=now.isoformat()
 labels=['attrInfo-localization166-v23-v2','export-route166-v23-v2','attrInfo-route-leaf166-v23','attrInfo-route-leaf166-v23-v2']
 hashes=['025c33f4a88c91327a4434f188c0f0f79c0f1c0780a4dc718c1e37459786efcc','815649f17092d0ab895e2375f356d424845c96da7992a55acd2f29e26b65a2f9','c93c2feaca3712f8c02896b5406bc42e500e9bbeb72dfb18741baed7dee48882','2f53dc556d5db0b378acd461a64b72a5419d7046f67b97b30f05a7fd4775c17d']
 data={};refs=[]
 for label,sha in zip(labels,hashes):
  r=read(N/(label+'.json'));assert r['returncode']==0 and not r['stderr'];d=json.loads(r['stdout']);assert d['result_sha256']==sha;data[label]=d;refs.append(dict(receipt=str(N/(label+'.json')),path=d['full_result'],sha256=sha))
 for label in ['attrInfo-localization166-v23','export-route166-v23']:assert read(N/(label+'.json'))['returncode']==1
 assert 'FileNotFoundError'in(N/'zcode-export-route-v23-v2-stderr-20261031.txt').read_text()
 q=read(O/'attribute-schema-primary-reference-v23.json');route=data[labels[1]];leaf=data[labels[3]]['result']['one_hop_msprof_entry'];pairs=route['adjacent_nonempty_same_attrInfo_op_pairs']
 assert route['graph_compute_executions']==132000 and len(pairs)==18
 assert sum(p['executions']for p in pairs)==14480
 assert sum(p['executions']for p in pairs if p['op']!=p['previous_op'])==13840
 assert leaf['line_count']==28 and any('msinterface.msprof_entrance'in s for s in leaf['bounded_entry_source'])
 findings=[
  dict(id='C23-1',grade='DOCUMENTED',finding='Official CANN9.1.0-beta.1 DB schema explicitly maps attrInfo through STRING_IDS and describes it as operator shape/custom-parameter metadata. This supports the decoding convention, not an exact installed9.2 writer/graph ownership binding.',evidence=q['schema_reference']),
  dict(id='C23-2',grade='MEASURED',finding='Saved132000 graph compute executions contain14480 adjacent nonempty same-attribute-ID pair occurrences in18 op-pair groups;13840 cross operator types. Adjacency here is chronological(startNs,endNs,TASKrowid), not proven producer/parent/API association or causal order across streams. Repeated literal values may legitimately share string IDs.',evidence=refs[1]),
  dict(id='C23-3',grade='MEASURED',finding='The already exact3200Foreach->FFN_HC pairs reuse one attribute ID/text. Additional chronological cross-type equality includes InplacePartialRotaryMul->ScatterNdUpdateSk3520, reverse3200, MatMulV2->Cast640, Cast->Mul640, Indexer->prepare_indices640. These do not establish wrong metadata or native attribute values.',evidence=[refs[1],'merge-to-HC-contract-v22.json']),
  dict(id='C23-4',grade='DOCUMENTED',finding='Bounded literal attrInfo/COMPUTE_TASK_INFO localization in installed Torch/CANN profiler Python analysis found only the Torch table constant and a task-info reader. No writer is identified by these literal matches. This is a bounded negative search result, not proof all writers are absent.',evidence=refs[0]),
  dict(id='C23-5',grade='DOCUMENTED',finding='Installed Torch CANNExportParser.run selects an external msprof --export=on --type=db branch for Db export. The exact CANN9.2 msprof.py entry is a28line dynamic importer of msinterface.msprof_entrance.MsprofEntrance().main(). Neither observed branch source alone attests the exact historical export invocation or attrInfo assignment.',evidence=[refs[1],refs[3]]),
  dict(id='C23-6',grade='UNKNOWN',finding='The precise current attrInfo population/ownership rule, why graph HC shares preceding metadata, and any relationship to loaded kernel scalar/tiling values remain open. No exporter bug, copied-parent rule or actual HC parameter is concluded.',evidence='PROFILE-CODE-QUESTION-GATE-v23.json')]
 contract=dict(generated_utc=stamp,status='RAW_DECODE_CONVENTION_SUPPORTED;OWN_OP_SEMANTICS_OPEN',findings=findings,
  observed_graph_attribute_adjacency=pairs,chronological_adjacency_is_causal_proof=False,
  actual_own_operator_parameter_values_proved=False,loaded_binary_or_tiling_proved=False,
  new_model_source_read=False,new_device_or_service_work=False,whole_module_closed=False,overall_complete=False)
 write(O/'profile-attribute-quality-contract-v23.json',contract)
 gate=read(O/'PROFILE-CODE-QUESTION-GATE-v23.json');gate.update(answered_scope='Official contextual schema validates STRING_IDS convention; exact installed Torch export delegation and28line msprof entry bound.',
  question_closed=False,precise_remaining_question='How does the relevant installed msprof export component populate COMPUTE_TASK_INFO.attrInfo and attach it to graph tasks?',
  next_allowed_symbol='msinterface/msprof_entrance.py:MsprofEntrance.main, selected by literal import in the already pinned entry; only if an active candidate needs a parameter value from attrInfo.',
  stop_rule='Do not chase all SDK internals merely to decode every field. If raw attribute ownership is not needed to decide a conditional candidate, retain its unknown values and advance other profile-raised opportunities.',
  no_broad_source_audit=True,overall_complete=False);write(O/'PROFILE-CODE-QUESTION-GATE-v23.json',gate)
 model=deepcopy(read(O/'candidate-profile-ledger-v22.json'))
 for c in model['candidates']:
  c['raw_attribute_evidence_policy_v23']='profile-attribute-quality-contract-v23.json;STRING_IDS convention supported by contextual schema;own-op scalar/tiling values need independent binding. Existing source-contract conclusions are separate.'
 model.update(generated_utc=stamp,parent_ledger='candidate-profile-ledger-v22.json');assert len(model['candidates'])==33;write(O/'candidate-profile-ledger-v23.json',model)
 framework=deepcopy(read(O/'framework-candidate-ledger-v22.json'));framework.update(generated_utc=stamp,parent_ledger='framework-candidate-ledger-v22.json');write(O/'framework-candidate-ledger-v23.json',framework)
 mods=deepcopy(read(O/'module-profile-ledger-v22.json'));matrix=deepcopy(read(O/'module-phase-branch-matrix-v22.json'))
 for m in mods['modules']:
  m['phase_branch_record']='module-phase-branch-matrix-v23.json#'+m['id']
  if m['id']in ['M00','M06','M12','M13','M22']:
   m['attribute_quality_v23']=dict(finding='Raw attribute decode convention supported; graph cross-type equality and exporter delegation reviewed. Own-op meaning still open and not used as native scalar/tiling proof.',evidence='profile-attribute-quality-contract-v23.json',whole_module_closed=False)
 for m in matrix['modules']:
  if m['id']in ['M00','M06','M12','M13','M22']:m['branches'].append(dict(branch='Raw profile attribute quality and exporter association',state='SCOPED_EVIDENCE_QUALITY_REVIEW;OWN_OP_SEMANTICS_OPEN',evidence='profile-attribute-quality-contract-v23.json',gap='Do not infer actual native attributes from unbound metadata; exact writer only if a candidate conclusion needs it.'))
 mods.update(generated_utc=stamp,parent_ledger='module-profile-ledger-v22.json');matrix.update(generated_utc=stamp,parent_matrix='module-phase-branch-matrix-v22.json');write(O/'module-profile-ledger-v23.json',mods);write(O/'module-phase-branch-matrix-v23.json',matrix)
 remaining=deepcopy(read(O/'module-coverage-gaps-v22.json')['remaining'])
 remaining[1]='Raw attrInfo lookup is now supported by official contextual STRING_IDS schema;graph attribute adjacency18groups/14480pairs and Torch->msprof dynamic entry bound. Own-op writer semantics remain unknown. Only point-read MsprofEntrance.main if an active candidate must use an attribute value; otherwise retain unknown and advance other profile-led questions.'
 write(O/'module-coverage-gaps-v23.json',dict(generated_utc=stamp,remaining=remaining,whole_modules_closed=0,overall_complete=False))
 summary=dict(generated_utc=stamp,status='PROGRESS_ON_EVIDENCE_QUALITY;OWNERSHIP_QUESTION_OPEN',previous_turn_classification='PROGRESS:v22 merge-HC finite boundary/rounding/row/schedule contracts',
  graph_compute_executions=132000,adjacent_same_ID_pair_groups=18,adjacent_same_ID_pair_occurrences=14480,cross_op_pair_occurrences=13840,
  unchanged_FFN_HC_exact_attribute_ID_reuse=3200,own_op_semantics_closed=False,
  source_scope='4 exact SDK/Torch files including2 literal-field localization hits,1 export-route file and1 dynamic entry; no model source audit.',
  official_schema_version='9.1.0-beta.1 contextual;exact9.2 URL unavailable',
  all_jobs_terminal=True,new_model_import_or_device_work=False,new_service_changes=False,
  model_candidates=33,framework_candidates=7,whole_modules_closed=0,overall_complete=False,evidence_refs=refs,remaining=remaining)
 write(O/'attribute-quality-summary-v23.json',summary)
 report='''本轮继续复用profile，没有重采、运行模型/NPU或变更服务。新增内容是属性证据质量核对，整体仍未完成，属性归属问题也未宣称关闭。

此前取回的attrInfo字符串不是凭字段名猜测解码：官方CANN9.1.0-beta.1表结构明确给出attrInfo通过STRING_IDS映射，内容用于算子shape及自定义参数描述。这支持解码约定；该文档不是已安装9.2导出器的精确实现证明，尝试的9.2同路径页面不可访问，已记录而未反复重试。[官方DB表结构](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta1/devaids/Profiling/atlasprofiling_16_1144.html)

对保存的132,000次图计算记录新增相邻属性ID核对，共18种op-pair、14,480次非空同ID相邻配对，其中13,840次跨算子类型。这里“相邻”只表示按startNs/endNs/TASKrowid排序后的记录，不能当作跨流生产依赖或API父子关系。相同字符串合法地共用ID也很常见，不能仅据ID相同判定错误。原先由结构精确绑定的3,200个ForeachAddList->FFN HcPost仍全部同ID；另有rotary/scatter、matmul/cast、indexer/prepare等跨类型复用。因此在绑定具体归属前，不将这些文本直接升级成自身算子的native参数、tiling或二进制证据。

本轮代码范围保持定点：按attrInfo/COMPUTE_TASK_INFO两个字面量，在两个已安装profiler Python analysis目录做有界定位，只找到表名常量和读取函数，没有定位到writer。这不证明其他写入路径不存在。随后仅查导出入口：Torch的CANNExportParser.run在Db分支调用外部msprof --export=on --type=db；精确CANN9.2的msprof.py只有28行，通过msinterface.msprof_entrance.MsprofEntrance().main()动态进入下一层。这是源码分支契约，不自动证明历史具体导出命令或writer规则。

属性为何复用、真正的own-op参数归属和loaded native算术仍保留UNKNOWN。没有断言exporter bug或“必然复制前一API属性”。如后续某个候选必须依赖属性值，下一步仅允许沿已确定的MsprofEntrance.main点查；若不影响候选定性，则保留未知并推进其他profile问题，避免扩展成整个SDK源码审计。v22的alpha默认1、pre-add缩放位置、Tensor分支、行域与BF16边界来自独立的固定源码/shape契约，不受这批原始属性文本替代。

初次定位因容器无rg停止，已保留失败记录并使用现有grep；另一次复核因主控写错保存字段名停止，已按原结构修正；一个本地任务路径输入错误发生在远程执行前。失败及修正均登记，没有安装软件或更改目标环境。新证据、报告、候选与模块台账已归档；仍为33模型候选、7框架候选，26整模块没有宣称闭合。Goal保持active，最后健康观测仍是旧v12post。
'''
 (O/'PROFILE-FINDINGS-v23.md').write_text(report)
 action=('继续Loop004，v23是证据质量方面的实质进展，own-op属性归属仍未闭合。先读attribute-quality-summary-v23.json、profile-attribute-quality-contract-v23.json、merge-HC-summary-v22.json和module-coverage-gaps-v23.json。'
  '官方9.1.0beta1schema支持attrInfo->STRING_IDS convention，不是9.2writer证明；9.2同路径URL不可访问一次已记录。'
  '132000保存graph compute按(startNs,endNs,TASKrowid)排序新增18类14480同非空attrID邻接配对，13840跨op；不是同stream/producer/parent关联证明。exact3200Foreach->FFNHC共享ID事实保留，但不宣称bug或强制parent-copy。'
  '两个安装Pythonanalysis目录仅field字面搜索找到constant/reader，无writer；Torch _cann_export.py SHA a0f6ed83ec4af76caba3c3d64d0fd1da820639d45b1d02a32ab358d3e2244b85 的run55-108在Db分支调msprof --export=on --type=db。CANN9.2 msprof/msprof.py SHA f110cac92111fb3bb36904422f6e66469d434f9dc13f57b41c892caa941d8bdb仅28行，dynamic import msinterface.msprof_entrance.MsprofEntrance().main()。不证明历史export invocation或attrwriter。'
  '只有某活跃候选必须依赖attr值时才沿msinterface/msprof_entrance.py:MsprofEntrance.main点查一跳；否则保留own-op值UNKNOWN并推进其他profile消费者/条件，不为解码每个字段追扫全部SDK。'
  'v22默认alpha1/plainTensoradd、条件pre-addscale、320M800vsrawM816/834、BF16边界、current-order epilogue cycle只在原域成立，16条件/6F23子域结论保留，不重复推未知。下一批可推进F33/H07/compressor/ring/rollback实际消费者或F21postloadscale/weights/generation；先查已有profile/完整stack再按具体缺口点查。'
  'v23server结果PMU目录：attrInfo-exporter-field-localization-v23-v2 SHA025c33f4a88c91327a4434f188c0f0f79c0f1c0780a4dc718c1e37459786efcc；export-route-localization-and-graph-attribute-reuse-v23-v2 SHA815649f17092d0ab895e2375f356d424845c96da7992a55acd2f29e26b65a2f9；attrInfo-export-route-leaf-source-v23-v2 SHA2f53dc556d5db0b378acd461a64b72a5419d7046f67b97b30f05a7fd4775c17d；原leafv1 SHA c93c2feaca3712f8c02896b5406bc42e500e9bbeb72dfb18741baed7dee48882保留。'
  '初定位缺rg、首reuse字段TASK_rowid错误、一个本地task路径误写均error保存，修正版pass；无重复运行原始DB/模型/NPU/服务。33model/7framework/0wholemodule，wholefalse，Goalactive；3136resourcecandidate/849missingPMU/16unbound/26theoryINCOMPLETE保留。当前EngineCore等成本若必要补采需先既有授权下freshhealth/ownership/frozenworkload/clone/restore，不假设空闲，lasthealthv12post。')
 jobs=[
  ('review','error','Initial field localization stopped because rg was unavailable in stage; no result, software installation or environment change.',[N/'locate_attrInfo_exporter_binding_v23_166.py',N/'attrInfo-localization166-v23.json']),
  ('review','pass','Existing grep bounded literal-field localization retained2 matching SDK/Torch files; no writer semantics claimed.',[N/'locate_attrInfo_exporter_binding_v23_166_v2.py',N/(labels[0]+'.json')]),
  ('review','error','Initial adjacency review stopped on main-authored wrong TASK_rowid field; original script/receipt retained.',[N/'read_export_route_and_attribute_reuse_v23_166.py',N/'export-route166-v23.json']),
  ('review','error','Local executor launch stopped before remote execution because task-file date directory was mistyped.',[N/'zcode-export-route-v23-v2-stderr-20261031.txt',N/'zcode-export-route-v23-v2-output-20261031.json']),
  ('review','pass','Corrected saved132000graph chronological attribute-ID review and exact installed Torch export-route localization; no original DB/device work.',[N/'read_export_route_and_attribute_reuse_v23_166_v2.py',N/(labels[1]+'.json')]),
  ('review','pass','Stored Torch export run functions and one exact CANN msprof entry keyword contexts reviewed without imports/export execution.',[N/'read_attrInfo_export_route_leaf_v23_166.py',N/(labels[2]+'.json')]),
  ('review','pass','Same28line entry top-level dynamic-import body retrieved; literal MsprofEntrance.main next symbol bound, writer question remains open.',[N/'read_attrInfo_export_route_leaf_v23_166_v2.py',N/(labels[3]+'.json')]),
  ('review','pass','Main review separates documented schema mapping, observed metadata adjacency, source export delegation and unknown own-op/native semantics; scope not closed.',[O/'profile-attribute-quality-contract-v23.json',O/'attribute-schema-primary-reference-v23.json',O/'PROFILE-FINDINGS-v23.md'])]
 runs=[]
 for i,(kind,status,desc,arts)in enumerate(jobs):
  rid='run-'+(now-timedelta(seconds=len(jobs)-i)).strftime('%Y%m%dT%H%M%SZ');run=ctl('new-run','--loop-dir',str(L),'--run-id',rid,'--case-key','c8-input32k-output1k-prefix90-166','--kind',kind,'--command','Preparedv23 retrospective evidence registration: '+desc,'--code-ref','Saved166profile/exactSDK export-source slices; no model/NPU/service work','--working-directory',str(N));args=['finish-run','--run-dir',run,'--status',status,'--correctness','not-applicable','--summary',desc]
  for p in arts:assert p.exists(),p;args+=['--artifact',str(p)]
  ctl(*args);runs.append(dict(run_dir=run,status=status,description=desc))
 cp=dict(status='PROGRESS',progress_domain='EVIDENCE_QUALITY;OWN_OP_ATTRIBUTE_QUESTION_STILL_OPEN',previous_turn_classification=summary['previous_turn_classification'],current_verified_changes=['Official contextual schema supports raw STRING_IDS decode convention.','132000graph chronological metadata-ID adjacency18groups/14480pairs reviewed without causal/own-op inference.','Exact Torch export delegation and28line CANN dynamic entry bound; further writer internals only if an active candidate needs them.'],summary=str(O/'attribute-quality-summary-v23.json'),candidate_ledger=str(O/'candidate-profile-ledger-v23.json'),framework_candidate_ledger=str(O/'framework-candidate-ledger-v23.json'),module_ledger=str(O/'module-profile-ledger-v23.json'),branch_matrix=str(O/'module-phase-branch-matrix-v23.json'),report=str(O/'PROFILE-FINDINGS-v23.md'),source_gate=str(O/'PROFILE-CODE-QUESTION-GATE-v23.json'),coverage_gaps=str(O/'module-coverage-gaps-v23.json'),pending_full_scope_domains=remaining,remote_job_status='All6prepared remote attempts terminal:2error/4pass;1local launch error before remote execution;no model/NPU/service/health work.',next_action=action,whole_modules_closed=0,overall_complete=False,updated_utc=stamp);write(O/'REVIEW-CHECKPOINT-v23.json',cp)
 for p in[L/'state.json',T/'index.json']:
  d=read(p);assert d['status']in['EVALUATING','ACTIVE'];d.update(next_action=action,updated_at=stamp);write(p,d)
 for p in[L/'status.md',T/'status.md']:p.write_text('\n'.join(('- 下一步：'+action)if x.startswith('- 下一步：')else('- 更新时间：`'+stamp+'`')if x.startswith('- 更新时间：')else x for x in p.read_text().splitlines())+'\n')
 (O/'profile-summary-before-v23.json').write_bytes((R/'profile-summary.json').read_bytes());prof=read(R/'profile-summary.json');prof['supplementary_attribute_quality_v23']=dict(summary=cp['summary'],raw_lookup='CONTEXT_SCHEMA_SUPPORTED',own_op_semantics='UNKNOWN',whole_modules_closed=0,overall_complete=False);write(R/'profile-summary.json',prof)
 for p in[N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json']:
  d=read(p);d.update(latest_profile_first_review=cp,overall_complete=False,whole_modules_closed=0,updated_utc=stamp);d['next_read_only_action'if p==R/'STATE.json'else'next_action']=action
  if p==W/'STATE.json':d.update(profile_candidate_ledger=cp['candidate_ledger'],profile_module_ledger=cp['module_ledger'])
  if p in[N/'RECOVERY.json',R/'PARALLEL-RESUME-PACK.json']:d['read_first']=[cp['summary'],str(O/'profile-attribute-quality-contract-v23.json'),str(O/'merge-HC-summary-v22.json'),cp['coverage_gaps']]
  write(p,d)
 compact=ctl('compact','--task-dir',str(T));validation=ctl('validate','--task-dir',str(T));write(O/'task-run-registration-v23.json',dict(runs=runs,compact_output=compact,validation_output=validation,whole_modules_closed=0,overall_complete=False))
 arts=[p for p in O.glob('*v23*')if p.is_file()and not p.name.startswith('checkpoint-manifest')]+[N/(l+'.json')for l in labels]+[N/'attrInfo-localization166-v23.json',N/'export-route166-v23.json',N/'zcode-export-route-v23-v2-stderr-20261031.txt',N/'zcode-export-route-v23-v2-output-20261031.json']
 arts +=[p for p in N.glob('*v23*py')if p.is_file()]+[N/'RECOVERY.json',W/'STATE.json',R/'STATE.json',R/'PARALLEL-RESUME-PACK.json',R/'profile-summary.json',L/'state.json',T/'index.json']
 write(O/'checkpoint-manifest-v23.json',dict(generated_utc=stamp,artifacts=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())for p in arts],whole_modules_closed=0,overall_complete=False))
 print(json.dumps(dict(checkpoint=str(O/'REVIEW-CHECKPOINT-v23.json'),runs=len(runs),pass_runs=5,error_runs=3,validation=validation,overall_complete=False),ensure_ascii=False))

if __name__=='__main__':main()
