"""Seal completed 171 evidence once; retain every original coverage obligation."""
from pathlib import Path
from copy import deepcopy
from finalize_origin_coverage_v9 import N,O,read,write
from analyze_evidence_batches_v155 import pin
from persist_delta_checkpoint_v161 import persist_delta
assert not (O/'REVIEW-CHECKPOINT-v161.json').exists()
old=read(O/'REVIEW-CHECKPOINT-v159.json')
action=(
 '继续全模块、全分支和跨算子/层/卡/迭代的原整体目标；复用已证明的共享算子cut，不展开383旧关系行的笛卡尔排查。'
 'v160/v161取证合并封存为一次权威更新：当前6组Engram重复payload逐字节相同，停止为其数值初始内容追查一般loader；实际布局、驻留和更新代次仍须按算子需要取证。'
 '171普通启动及单次回答2已通过；完整C8×32768→1024/前缀90%尚未实测，旧169/166-105身份不回填。'
 '实际CANN9.2映射backing ELF与SDK ABI已核对；一次单独授权的69B eager Pad联合探针完成且输出正确，原服务已恢复health200。授权已消耗，不得重跑或扩展至全模型诊断。'
 '内置字段门与storage基址语义已在本例实测；直接复用它们，定点补offset/extent/epochs/scalar values/native桥和读者。152B HostArgs含[8,24] relocation，word0位于记录的2560B私有分配+1024；不据此定角色、NULL、不交叠或可删除。'
 '下步先把现成profile采集入口与实际版本逐项对齐，冻结可审查的171全Case诊断方案和原生桥缺口，准备工作不触发新设备采集；不再重查已匹配的SDK或重跑旧控制模型。'
 '保留9/1352原生参数、849PMU、16phase、3136归属、G14 copy/control、分支穷尽性、实际数值/读者/物理容量和最低MFU全部缺口；整模块闭合0、整体未完成。')
gate={'version':161,'parent':pin(Path(old['source_gate'])),
 'policy':'Profile/shared operator evidence first; source only for a conclusion-changing fusion/ceiling question; no irrelevant pair sweep.',
 'answered_in_stated_scope':[
  {'question':'Do the six current duplicate disk payload pairs differ?',
   'answer':'No. All six pairs directly bytewise equal and file stats stable. General loader precedence is no longer needed to choose these initial numeric bytes; runtime layout/epochs remain separate.',
   'evidence':'six-duplicate-parameter-values-v160.json'},
  {'question':'Is the actual CANN9.2 SDK/provider identity compatible with the existing adapter ABI?',
   'answer':'Mapped lower-layer ELF backing identity bound for the first own startup; exact four declarations and CPU type/layout checks pass. This is not all runtime selection/error-path coverage.',
   'evidence':'provider-sdk-abi-layout-v160.json'},
  {'question':'Does the builtin metadata gate emit usable fields on this installed runtime?',
   'answer':'Yes for the one own eager Pad pair: profile_memory off has no tensor detail, on emits eight fields. Storage pointer is owner base, not offset-one data pointer; scalar is a type label.',
   'evidence':'joint-profiler-capability-observation-v161.json'},
  {'question':'Does real HostArgs forwarding work in the bounded case?',
   'answer':'One successful MemSet packet,152B,[8,24]placeholder. It is own eager evidence only; task/stream IDs unqueried and runtime bypass/error/whole-model paths not certified.',
   'evidence':'native-hostargs-window-completion-v161.json'}],
 'retired_redundant_work':['No general duplicate loader study just to choose six equal initial payloads.',
  'No repeated matching CANN9.2 header survey or v127 CPU suite.',
  'No new all-tensor logger duplicating runtime-validated builtin fields.',
  'No rerun of unchanged v128 synthetic/scalar suites or irrelevant pair stars.'],
 'next_targeted_questions':[
  {'id':'Q161-FULLCASE-ACQUISITION','question':'Which existing startup/graph-creation/replay profiler entry points can supply one bounded171 fixed-case evidence cohort, and which fields still require narrow supplements?',
   'allowed':'Read saved exact-version entrypoints and existing profile capability indexes; prepare concrete bounded commands. No new native/model execution implied.'},
  {'id':'Q161-TASK-BRIDGE','question':'Can actual native operations be joined by explicit correlation/task/capture IDs without nearest-time or mixed stream namespaces?',
   'allowed':'Inspect only the relevant saved provider/correlation contracts; keep unknown until actual supported bridge evidence exists.'},
  {'id':'Q161-READER-CUT','question':'For a selected optimization cut,what storage generations,values and complete last-reader facts are still absent after builtin reuse?',
   'allowed':'Reuse existing graph/effect indexes; add only shared missing facts,not Cartesian tests.'}],
 'all_missing_fields_remain_UNKNOWN':True,'no_new_independence_or_whole_closure':True,'overall_complete':False}
overlay={'version':161,'parent':pin(Path(old['evidence_overlays'][-1])),
 'changed_shared_facets':['E0','E2','E3'],
 'duplicate_content':pin(O/'six-duplicate-parameter-values-v160.json'),
 'provider_identity':pin(O/'mapped-provider-backing-objects-v160.json'),
 'SDK_ABI':pin(O/'provider-sdk-abi-layout-v160.json'),
 'actual_own_metadata_and_hostargs':pin(O/'joint-profiler-capability-observation-v161.json'),
 'bounded_native_conditional_reuse':pin(O/'joint-native-existing-proof-reuse-v161.json'),
 'authorization_and_service_restore':pin(O/'native-hostargs-window-completion-v161.json'),
 'history_not_replaced_by_current_own_case':True,
 'new_pair_clauses':0,'new_independence_proofs':0,'new_native_operand_or_phase_promotions':0,
 'new_whole_module_closures':0,'whole_modules_closed':0,'overall_complete':False}
queue=deepcopy(read(Path(old['work_queue'])));queue.update(version=161,parent=pin(Path(old['work_queue'])))
for row in queue['work']:
 if row['id']=='Q155-GRAPH-CUTS':
  row.update(state='BUILTIN_RUNTIME_FIELDS_REUSE_AND_SHARED_READER_CUTS',
   action='Reuse existing graph-cut proofs and the actual builtin field route. Six duplicate disk-content questions resolved;do not chase general loader precedence for those initial numeric bytes. Specify narrow offset/extent/epoch/value/role/reader supplements for shared cuts.',
   remaining='Actual loaded parameter layout/residency/epochs and complete graph/effect branch universe remain open;no universal independence or full module closure.')
  row['evidence']+=['six-duplicate-parameter-values-v160.json','joint-profiler-capability-observation-v161.json']
 elif row['id']=='Q155-NATIVE-OPERANDS':
  row.update(state='ONE_REAL_OWN_CASE_SUCCESS_FULLCASE_BRIDGE_OPEN',
   action='Keep the completed probe terminal and service restored. Reuse validated9.2 provider/header/ABI and builtin metadata;prepare one bounded fixed-case acquisition at startup/graph creation where needed. Explicit task/correlation/capture bridges and narrow missing fields are prerequisites;no blind full-model LD_PRELOAD.',
   remaining='Original1272eager and80captured tasks remain unfilled. No all-path error/bypass guarantee, no actual device.o byte attestation,offset/epoch/reader closure or physical time bound.')
  row['evidence']+=['mapped-provider-backing-objects-v160.json','provider-sdk-abi-layout-v160.json','joint-native-existing-proof-reuse-v161.json','native-hostargs-window-completion-v161.json']
queue['new_evidence']=['Six current duplicate disk payload pairs equal','Two CANN mapped backing ELFs and CPU SDK ABI bound',
 'One69B eager Pad pair correct;builtin gate/storage-base fields and one HostArgs packet observed','Own service restored health200']
recs=deepcopy(read(O/'current-obligation-reconciliations-v159.json')['reconciliations'])
for row in recs.values():
 facets=row['current_evidence_facets'];d=[]
 if 'E0' in facets:d.append('Current171 duplicate numeric payload ambiguity resolved for6pairs;actual loaded layout/residency/update and historical values remain unbound. Ordinary startup not full-case measurement.')
 if 'E2' in facets:d.append('Own69B actual builtin fields observed:storagebase not offset view data;offset/extent supplement only this case. Complete generations/values/reader facts remain open.')
 if 'E3' in facets:d.append('Actual CANN backing ELF and SDK ABI bound;one own eager interposition success. No original1352 recovery/task-ID bridge/all-path coverage;captured creation/update separate.')
 if 'E6' in facets:d.append('No hardware counters or device instruction-byte attestation;32B source request only conditional proof reuse,not HBM traffic or kernel time.')
 row.update(v161_evidence_delta=d,whole_domain_closed=False,original_broad_remaining_and_history_preserved=True)
write(O/'current-obligation-reconciliations-v161.json',{'parent':pin(O/'current-obligation-reconciliations-v159.json'),
 'reconciliations':recs,'does_not_delete_or_narrow_original_requirements':True,
 'all_missing_evidence_remains_unknown':True,'overall_complete':False})
report='''# v161：171 联合探针与共享证据增量

本次单独授权的联合最小探针只执行一次，输出正确；临时探针容器已退出，原主服务恢复并返回 health 200，恢复快照中的全部 NPU 进程属于原主容器。单次授权已消耗。171 的全部资料位于 `/data/tiankuan/wio` 下。普通模型启动及单次回答“2”通过，完整 C8×32768→1024、前缀90%负载尚未在171实测。首次启动与恢复后的PID属于两个运行时段，原始169/166-105记录的主机与时间身份保持不变。

此前 v160 的有界 CPU 读证和此次 v161 结果合并做一次权威更新，减少重复检查点工作。6组 Engram 同名参数在基础/extra分片的当前payload逐字节相同，共读取1,258,618,880参数payload字节，相关文件前后stat稳定。这解决了这6组初始数值内容的磁盘副本歧义，后续无需为选数值而研究一般loader顺序；实际加载布局、驻留、更新代次和历史值仍未证明。

首次普通启动时，检查21个自有进程，20个进程各映射两份CANN9.2前端库，共40条记录。容器overlay路径的stat与kernel VMA inode不一致，已沿Docker精确22个层根、仅两个相对路径找到真实backing文件，device/inode及完整SHA相符。它证明该时段映射的ELF backing身份，不证明私有COW代码、运行时调用选择或实际device.o指令字节。4个SDK声明及AArch64类型/布局CPU编译检查通过；可执行文件只依赖libc，没有新NPU上下文。复用已成功的v127c 23例CPU结果，不重跑初次失败构建或整套用例。

联合探针在逻辑设备0创建BOOL[21] owner、offset=1的BOOL[20] view及两个BOOL[24]输出，应用NPU张量共69B，运行时/分配器/workspace开销另计；无模型、无建图、无replay。两段CPU profiler均打开shape/stack：关闭profile_memory的对照没有TLV5/6/7，开启后有6个Tensor详情事件和5个Scalar类型事件。本例8字段的地址等于owner storage基址，实际view数据指针比它多1B；storage_offset、nbytes和实际标量/列表值仍不在该内置记录中。因此复用内置字段，并只补影响alias和融合判断的缺项。

真实拦截记录1条MemSet HostArgs：152B参数包、48个launched blocks、placeholder [8,24]，launch_rc=0；任务/stream getter未调用，rc=-1和task_id=0是logger哨兵。首word落在已记录的2560B NPU分配范围内，偏移1024B；公共输出24B对应的分配器记录为512B。应用字节、storage范围、分配块大小和真实写量不可互换。分配器stream token与原生opaque stream handle不直接混为同一命名空间；CPU profiler原始syscnt与CLOCK_REALTIME也没有按最近时间硬拼接。

inline96B记录与旧v128自有记录完全相同，可在“实际执行device.o匹配已pin对象、[8,24]正确提供第二参数、首word角色未被运行时改变”的前提下复用原标量证明：48个启动块中block0到达一个store位点，source-matched请求32B。本轮不重跑147个合成记录/7056个block trace。该32B不是实测HBM事务或GPU耗时；未把首地址差异当workspace角色、无alias、全读者退役或helper可删除证明。零relocation槽也不是NULL/no-data证明。

本次25份诊断文件共108031B，独立原始TLV/参数包解析和SHA检查通过。期间本地控制模板替换和离线Counter解析错误已记录并修正，没有因此重提交原生探针。主服务的成功恢复是本次授权的组成部分。

覆盖仍保持全部G00–G14、35+7候选、26模块358语义行；清单不是穷尽性证明。新增配对、独立性证明、原始native字段回填和整模块闭合均为0。9/1352原生任务、849 PMU、16 phase、3136归属、G14 copy/control、全部未闭合分支与物理容量/最低MFU仍保留。下一步冻结一次可审查的171固定负载诊断采集方案，优先复用内置profile字段并解决显式原生关联与graph创建/更新缺口。整体任务未完成。
'''
cases=read(O/'case-registrations-v161.json')['cases']
jobs=[
 {'kind':'test','status':'pass','correctness':'pass','case_key':'bringup-one-chat-171',
  'command':'Remote171 bringup_worker.py: guarded own docker start then docker exec main python3 scripts/smoke.py; single generation.',
  'code_ref':'Image ce42d65b; unchanged pinned serve/smoke; bringup_worker.py snapshot in inputs.',
  'description':'171 ordinary bringup health/model-registration and one answer2 pass;not full fixed-case performance/accuracy.',
  'artifacts':[Path('/Users/wio/work/171-vllm-migration-20261102/RUNTIME-READY-171-20261102.json')]},
 {'kind':'review','status':'pass','description':'Completed bounded current6pair payload comparison;actual mapped CANN backing and SDK ABI CPU-only check;no old suites rerun.',
  'artifacts':[O/'six-duplicate-parameter-values-v160.json',O/'mapped-provider-backing-objects-v160.json',O/'provider-sdk-abi-layout-v160.json']},
 {'kind':'profile','status':'pass','correctness':'pass','case_key':'provider-pad20-view-offset1-171',
  'command':'Remote171 controller.py --execute; one guarded stage docker exec timeout45s python3 -I -B probe.py --run --bundle; own service restored.',
  'code_ref':'Authorized probe ab9184273da4;controller7b0e21eb7f1f;adapter4bd42a98eb1a;full source SHA receipts retained.',
  'description':'One authorized69B eager Pad pair correct;builtin raw fields and real152B HostArgs observed;no model/graph/replay in probe;service restored.',
  'artifacts':[O/'native-hostargs-authorization-v161.json',O/'joint-hostargs-probe-file-index-v161.json',O/'joint-profiler-capability-observation-v161.json',O/'native-hostargs-window-completion-v161.json']},
 {'kind':'review','status':'error','description':'Local control-template SyntaxErrors before remote execution and offline Counter TypeError corrected;no native/model failure or resubmission.',
  'artifacts':[O/'local-control-analysis-corrections-v161.json']},
 {'kind':'design-check','status':'pass','description':'Independent raw-byte/scope/restore checks pass;all original obligations retained and no native/whole promotion.',
  'artifacts':[O/'joint-runtime-preflight-v161.json',O/'joint-native-existing-proof-reuse-v161.json']}]
persist_delta(version=161,previous=159,prefix='joint-runtime',
 changes=['Six current duplicate payload pairs bytewise equal;retire irrelevant loader value study.',
 'Two actual mapped CANN9.2 backing ELFs and CPU SDK ABI bound;old CPU suites reused.',
 'One authorized69B Pad pair passed;builtin gate/base-pointer and one real HostArgs packet observed.',
 'Original main service restored health200;one-off authorization consumed;no original1352/native or whole closure.'],
 report=report,action=action,
 remaining_delta='G161: Own171 one-case builtin/HostArgs success and current duplicate content resolved;original1352,eager/captured correlation,offset/epochs/scalars/readers,all branch/physical/capacity/lowest and G14 remain open. Historical cohort not backfilled;full171 fixed case not measured.',
 source_gate=gate,operator_overlay=overlay,queue=queue,jobs=jobs,
 scripts=[Path(__file__),N/'persist_delta_checkpoint_v161.py',N/'verify_joint_runtime_v161.py',N/'assemble_joint_evidence_v161.py'],
 obligation_reconciliations=recs,case_registrations=cases,
 remote_status='Both acquisitions terminal. One authorized native attempt exit0;stage exited;own main service restored running health200. First-startup PID/library snapshots preserved as historical episode;no active diagnostic jobs. Full fixed171 case and diagnostic profile not run.')
