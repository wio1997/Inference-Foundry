"""Seal the pre-pause read-only evidence delta; preserve full operator scope."""
from pathlib import Path
from copy import deepcopy
from finalize_origin_coverage_v9 import N,O,read,write
from analyze_evidence_batches_v155 import pin
from persist_delta_checkpoint_v159 import persist_delta

assert not (O/'REVIEW-CHECKPOINT-v159.json').exists()
old=read(O/'REVIEW-CHECKPOINT-v158.json')
action=('继续原完整全模块目标与Loop004，按共享算子证据问题推进，不把383旧关系行当383逐对任务。'
 'v159已核对142个检查点分片header、143324个索引声明和143330个物理条目；80主层/6MTP的WO_A/B均为BF16，FLOAT量化类别不是FP32存储。6个Engram同名条目在基础/extra分片均出现，值相等和实际load胜出者尚未证明；下一步优先定点核对这6个值域，避免无关框架研究。'
 '精确声明TorchNPU提交和3份已安装/缓存Python字节匹配显示：正常CPU回调内，输入细节需要record_shapes&&(with_stack||with_modules)&&profile_memory；record_op_args不在此Torch细节门内，但另有CANN配置转发。复用内置TensorImpl/存储基址/类型/元素大小/shape/stride/device字段，不重写全量tensor logger；storage_offset、实际storage范围/格式、分配和更新代次、实际标量/列表/掩码值、原生任务桥和全部读者仍需补证。'
 'P3先核对实际CANN9.2 provider/头文件/ELF及v127 ABI与bypass边界，再准备有界验证；旧109/120单次探针授权不扩展成新的诊断授权。80capturedBOOL的creation/update证据与1272eagerHostArgs入口不可替代。'
 '171已单独完成迁移，当前普通模型启动/单次生成由本次继续指令授权；其正常初始化及图运行不是v159字段取证，也不是新诊断profile或完整C8输入32k输出1k前缀90%测量。原169/166-105 profile继续保留原主机和run身份，迁移和新单次生成不回填旧9/1352。'
 '全部35+7候选、26模块358行及分支穷尽性、849PMU、16phase、3136归属、G14 copy/control、原生数值/读者/物理容量与最低MFU结论仍未闭合；不能因报告/清单成稿标记完成。')

gate={'version':159,'parent':pin(Path(old['source_gate'])),'policy':'Profile and shared evidence first; source reads only answer operator-fusion/ceiling questions; no Cartesian pair sweep','answered_in_stated_scope':[
 {'question':'Configured output projection storage and available global weight groups','answer':'All86 selected WO_A/B headers bound;BF16 storage and full global group metadata. No loaded-residency or historical-value proof.','evidence':'output-weight-checkpoint-coverage-v159.json'},
 {'question':'Does simply enabling record_op_args populate Torch input detail fields?','answer':'Not the source gate for TLV5/6/7;CPU callback+shapes+(stack/modules)+profile_memory are required. record_op_args has separate CANN forwarding.','evidence':'TorchNPU-input-field-producer-contract-v159.json'},
 {'question':'Can these basic fields alone prove nonaliasing or actual scalar values?','answer':'No;storage offset/extent/epochs and actual scalar values are absent;counterfactual overlap and value worlds remain possible.','evidence':'profiler-field-ambiguities-v159.json'}],
 'next_targeted_questions':[
 {'id':'Q159-ENGRAM-CONTENT','question':'Are the six duplicate BF16 parameter byte ranges equal in a validated current checkpoint cohort? If different, which operator-specific deferred load path selects them?','allowed':'Read only those already indexed/offset-bounded payloads and only necessary Engram loader edges. Do not survey the framework.'},
 {'id':'Q159-PRODUCER-RUNTIME','question':'Does the actually installed compiled producer implement the declared field gate and preserve argument-role association for this case?','allowed':'First static exact-image ABI/producer audit; runtime evidence must carry its own scope and case.'},
 {'id':'Q159-PROVIDER','question':'Do real CANN9.2 provider symbols/signatures/packet ownership match v127?','allowed':'Read exact headers and provider ELF;no new adapter/model diagnostic execution implied.'}],
 'all_missing_fields_remain_UNKNOWN':True,'no_new_independence_or_whole_closure':True,'overall_complete':False}

overlay={'version':159,'parent':pin(Path(old['evidence_overlays'][-1])),'changed_shared_facets':['E0','E2','E3'],
 'checkpoint_inventory':pin(O/'checkpoint-all-header-index-v159.json'),'weight_storage':pin(O/'output-weight-checkpoint-coverage-v159.json'),
 'weight_loading_source_contract':pin(O/'output-weight-loading-contract-v159.json'),'profiler_field_contract':pin(O/'TorchNPU-input-field-producer-contract-v159.json'),
 'offset_scalar_ambiguity':pin(O/'profiler-field-ambiguities-v159.json'),'startup_vs_steady_reference':pin(O/'output-weight-initialization-alternatives-v159.json'),
 'current_dtype_and_available_content_not_physical_work_or_capacity':True,'six_duplicate_names_not_native_failure_or_corruption':True,
 'new_pair_clauses':0,'new_independence_proofs':0,'new_native_operand_or_phase_promotions':0,'new_whole_module_closures':0,'whole_modules_closed':0,'overall_complete':False}

queue=deepcopy(read(Path(old['work_queue'])));queue['version']=159;queue['parent']=pin(Path(old['work_queue']))
for row in queue['work']:
 if row['id']=='Q155-GRAPH-CUTS':
  row.update(state='BUILTIN_METADATA_REUSE_AND_TARGETED_CHECKPOINT_DELTA',action='Reuse all26module358row routing and existing cut proofs. Prefer the declared-version builtin metadata producer; supplement only missing offset/extent/generation/value/role/read-retirement facts. Resolve the six real Engram duplicate-content questions as one batch,then point-read only conclusion-changing operator load edges.',remaining='Full branch/effect universe remains open;global checkpoint metadata is neither loaded replication nor historical values. No fourth pair star or repeated output-cut math.')
  row['evidence']+=['checkpoint-all-header-index-v159.json','TorchNPU-input-field-producer-contract-v159.json','output-weight-initialization-alternatives-v159.json']
 elif row['id']=='Q155-NATIVE-OPERANDS':
  row.update(state='BUILTIN_SOURCE_GATE_BOUND_REAL_PROVIDER_AND_CAPTURE_BRIDGES_OPEN',action='Reuse exact-declared-version gate and serialized-field audit; do not build a duplicate all-tensor logger. Read exact CANN9.2 provider headers/ELF before real v127 adapter testing. Keep original1272eagerHostArgs and80capturedcreation/update routes distinct;ordinary171 bring-up is not diagnostic hook evidence.',remaining='Actual compiled producer/bypass coverage,offsets/storage extents/epochs/scalar values/native task bridges/all readers remain open. Old missing fields remain unavailable;normal service generation does not backfill original tasks.')
  row['evidence']+=['TorchNPU-input-field-producer-contract-v159.json','profiler-field-ambiguities-v159.json','evidence-readiness-delta-v159.json']
queue['new_evidence']=['142fullheadercohort/143324indexed143330occurrences','86BF16WO_A_Bstorageheaders','exactdeclaredTorchproducer8sourcefiles/3Pythonmatches','sixduplicateEngramload/contentquestions']

reconciliations=deepcopy(read(O/'current-obligation-reconciliations-v158.json')['reconciliations'])
for row in reconciliations.values():
 facets=row['current_evidence_facets'];d=[]
 if 'E0' in facets:d.append('v159configuredcheckpointindex/header/quantizationandselectedloadingcontractbound;historicalactualobject/value/replicaidentitynotproved;sixduplicatenameprecedenceopen')
 if 'E2' in facets:d.append('v159builtininputfieldsourcecontractbound;storagebaseisnotviewaddress;offset/extent/epoch/readerandscalarvaluefactsremainunknown')
 if 'E3' in facets:d.append('v159exactdeclaredTorchgateclarified;actualCANN9.2providerABI/bypassandv127realforwardingstillunverified')
 if 'E6' in facets:d.append('v159BF16checkpointstoragedoesnotattestnativeMmadmode,layout,physicaltraffic,frequencyorcapacityupper')
 row.update(v159_evidence_delta=d,whole_domain_closed=False,original_broad_remaining_and_history_preserved=True)
write(O/'current-obligation-reconciliations-v159.json',{'parent':pin(O/'current-obligation-reconciliations-v158.json'),'reconciliations':reconciliations,'does_not_delete_or_narrow_original_requirements':True,'all_missing_evidence_remains_unknown':True,'overall_complete':False})

report='''# v159：检查点与内置 profiler 字段证据增量

本轮封存暂停前完成的只读取证。原完整目标、35+7候选、26模块358语义行、跨算子/层/卡/迭代范围全部保留；这些清单仍不是穷尽性证明。整模块闭合0，整体任务未完成。171迁移及本次普通启动是单独运行，不能填补原历史profile字段。

实际配置检查点的142个分片header已全部缓存并独立核对：143324个索引声明均在指定分片中存在；物理header有143330个条目，差值来自6个Engram同名参数在基础与extra分片各出现一次。索引指向extra不自动证明实际loader胜出者；本次未读权重payload，未证明这6组值相同，不判定损坏或原生失败。初次本地聚合错误地要求跨文件名称唯一；已保留失败记录并纠正为“索引声明”和“物理条目”两层计数，没有重新读取已缓存header。

全部40主层及3MTP的86个WO_A/B参数均已核对。WO_A为BF16[8192,4096]，WO_B为BF16[5120,8192]；描述中的FLOAT是量化类别，不能当FP32存储。全局检查点包含所有组，并不证明各rank已驻留所有组。精确镜像中的线性层切分、reshape及RL再加载路径仅建立源合同，不是实际对象/回调执行证明。

这也补充了初始化域：候选实现可以在加载时从全局检查点填充所选复制权重，不必仅因“原rank只持有分片”而强制每轮AllGather。对于已声明的TP8逻辑内容模型，方案B额外56MiB/层/rank，方案C额外126MiB；若选40主层，分别2240/5040MiB。初始I/O/H2D、实际分配/格式、缓存、workspace、代次和退役成本仍要计价，MTP不能自动套用同一TP域。稳定权重代次之后，参考图B的直接输出值交付量为A的2/3；C仍要求原分组BF16 partial及SUM语义，不能替换成一次完整WO_B点积。以上均非实际wire/HBM量、时间下界、显存可行性或可达加速证明。

精确镜像声明torch_npu 2.10.0.post7、Git提交1d88b8608fd1e44545b5804026c7a57c44240bb4；8份公开源文件已校验Git blob，3份安装/缓存Python文件逐字节相符。该提交的正常CPU RecordFunction路径需要record_shapes&&(with_stack||with_modules)&&profile_memory才写Torch输入细节。record_op_args不在此门内，但另有CANN配置转发，不可泛称无效。见[固定提交的原始实现](https://github.com/Ascend/pytorch/blob/1d88b8608fd1e44545b5804026c7a57c44240bb4/torch_npu/csrc/profiler/npu_profiler.cpp)。声明版本和Python相符仍不等于已安装编译对象、实际callback或绕行覆盖已实测。

内置序列化器提供TensorImpl、storage基址、dtype、元素字节数、shape、strided布局的stride、device类型和序号；不是t.data_ptr。它不提供storage_offset/nbytes、NPU实际格式、分配/参数更新代次、全部读者和tensor内容。Scalar字段是类型标签，ScalarList在此路径不展开实际值。相同已发字段可对应偏移0/2的相邻不交叠view，也可对应偏移0/1的交叠view；因此空白或相同/不同指针标签都不能替代alias证明。参数角色还须结合schema、输入类型/顺序和有效tensor过滤，不能把压缩列表直接当全部原始参数。

后续按共享证据问题推进：复用内置字段，定点补offset/范围/代次/值/原生桥和读者；核对实际9.2 provider ABI后再准备有界验证，不另写一套重复tensor logger。1272个eager与80个captured缺口保持分路，历史9/1352、849PMU、16phase、3136归属及G14 copy/control、完整分支与物理容量/最低MFU均保留。此次新增配对、独立性证明、原生字段回填和整模块闭合均为0。
'''

jobs=[
 {'kind':'review','status':'error','description':'v159localheaderaggregationwrongglobaluniquenessassumptionandinterruptedquadraticJSONdiagnostic;serverfetchsucceeded/allcached;notnativefailures','artifacts':[O/'checkpoint-header-local-corrections-v159.json',N/'read_all_checkpoint_headers_v159a_failed.py']},
 {'kind':'review','status':'pass','description':'v159readonly166-to105identity/142headersandmetadata/exactstopped-imagefiles/declaredGitblobandPythonmatches;no new device acquisition','artifacts':[O/'checkpoint-all-header-index-v159.json',O/'TorchNPU-profiler-field-sources-v159.json',O/'output-weight-checkpoint-coverage-v159.json']},
 {'kind':'design-check','status':'pass','description':'v159independentmetadata/sourcegate/domainchecks;builtinreuseandmissing-fieldambiguity;oldoperatorcasesnotrerun','artifacts':[O/'checkpoint-profiler-preflight-v159.json',O/'TorchNPU-input-field-producer-contract-v159.json',O/'output-weight-initialization-alternatives-v159.json']}]
persist_delta(version=159,previous=158,prefix='checkpoint-profiler',changes=['All142checkpointheadersand143324indexentriesreconciled;6Engramduplicateskeptwithoutloadedwinner/valueclaim','All86WO_A/BentriesBF16;startupglobalcheckpointavailabilitydistinguishedfromcurrentrankresidencyandsteadylogicaldelivery','ExactdeclaredTorchproducerand3Pythonmatchesclarifybuiltinfieldgate/limits;reuseinsteadduplicatedall-tensorlogger','All15full-scopeobligationsandoldnative/physicalgapsretained;no pair/independence/whole promotion'],report=report,action=action,remaining_delta='G159: Currentcheckpoint143324declarations/143330occurrences/86BF16outputweightsanddeclaredbuiltinproducercontractbound;6Engramduplicatedpayloads/loadwinner,historicalloadedobjects/residency,actualcompiledproducer/nativebridges/offsets/extents/epochs/values/readers/physicalcapacityandalloriginalscopegapsremainopen. Independent171ordinarybringupdoesnotbackfillhistoricalprofile.',source_gate=gate,operator_overlay=overlay,queue=queue,jobs=jobs,scripts=[Path(__file__),N/'verify_checkpoint_profiler_v159.py',N/'persist_delta_checkpoint_v159.py',N/'analyze_checkpoint_profiler_v159.py',N/'reconcile_checkpoint_headers_v159.py'],obligation_reconciliations=reconciliations,remote_status='All v159 original166/105 read-only archive/checkpoint/source acquisitions and local checks terminal. Separate171 ordinary bring-up and one generation passed;service may remain running. It is not v159 diagnostic profiler/native-operand evidence;no old captures or whole modules promoted.')
