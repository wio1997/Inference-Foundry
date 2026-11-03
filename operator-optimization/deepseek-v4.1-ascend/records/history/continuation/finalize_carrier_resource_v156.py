"""Persist the one-cut activation/weight/resource delta, with full scope open."""
from pathlib import Path
from copy import deepcopy
from finalize_origin_coverage_v9 import N, O, read
from analyze_evidence_batches_v155 import pin
from persist_delta_checkpoint_v156 import persist_delta


def main():
    assert not(O/'REVIEW-CHECKPOINT-v156.json').exists()
    evidence=['rank-device-namespace-bindings-v156.json','projected-carrier-weight-amortization-v156.json','carrier-board-placement-sensitivity-v156.json']
    queue=deepcopy(read(O/'coverage-work-queue-v155.json'));queue.update(version=156,parent=pin(O/'coverage-work-queue-v155.json'),new_evidence=evidence)
    queue['work'][0].update(state='ONE_CUT_JOINT_COST_BOUND_WEIGHT_AVAILABILITY_OPEN',action='Continue the existing TPheadAllToAll/WO_A materialization cut:check a materiallydifferent saved weight/checkpoint/storage artifact or precise WO_A weight-access source slice for all-group availability,layer/version identity and residency. Apply the shared amortization model;do not create candidate pair tasks or rescan unchanged raw streams.',evidence=evidence+['raw-field-availability-and-reuse-v155.json'],remaining='Originalnumeric/pad/head/row/parameter/nativeABI/alias/readers and actual allocation/HBM/wire/capacity remain unbound;cost model is one explicit construction.')
    queue['work'][1].update(state='RANK_TO_EXPORTED_DEVICE_METADATA_BOUND_PHYSICAL_BRIDGE_OPEN',action='Reuse32rank/device/group/log joins;twoTP8groups and16EPglobalranks are recorded. Keep exported-device-to-physical-chip/board bridge and actual math-mode/capacity upper unresolved. Use explicitly conditional occupancy bounds where sufficient;do not choose placement from identical numeric labels.',evidence=['rank-device-namespace-bindings-v156.json','archived-hardware-pool-contract-v155.json','carrier-board-placement-sensitivity-v156.json'])
    action=('继续原完整目标与Loop004，按v155共享证明/问题批次流程推进。v156已把32profile的rank/group/日志/device元数据链条绑定，并对现有headAllToAll前WO_A投影切面统一计入激活与一次权重分发成本。'
            '显式直接逻辑交付模型：每peer旧8192M、新2048M、权重8MiB，净节省7*(6144*sumM-8388608)，同层同权重代次sumM>=1366才为正；当前M800/816/834单次R1新/旧1.478~1.53，R2为0.864~0.89。'
            '这不是实际线流量/HBM/时延或所有实现必要成本，不能据此否定整个候选；常驻/流式/已有复制/加载/多播需各自完整成本图。'
            '下一步优先用另一类已保存权重/存储/checkpoint资料或WO_A权重访问的定点源码，核对所有8组权重是否已可用、代次、驻留及每层56MiB显式内容预算；不能跨层/参数代次累计R。'
            '不重做候选全配对/第四颗星/未变更有限证明/TLV缺失字段搜索。32rank-device是导出元数据桥，不是物理chip/board桥；只有16独立chip双TP8/8双芯片板注入映射条件成立时才用48+s跨板交付分类，未断言实际映射。'
            '26模块358语义行清单未穷尽，9/1352、849PMU、16phase、3136归属、G14和全部broader gaps继续保留，整模块关闭0，物理MFU/最低未成立。'
            '当前checkpoint为增量格式，候选/模块/分支/关系未变更权威继续取checkpoint中的v154实际路径，并叠加v155/v156证据；勿按版本名假设全量复制文件。'
            '所有新SSH只读归档/停止容器inspect和本地核算已终结，无新NPU/模型/context/capture/replay授权。')
    report='''本轮围绕现有“head AllToAll之前执行WO_A投影”这一具体切面，补齐了权重分发与驻留的联合成本，未新增候选配对任务。单看激活载荷4096→1024只能得到四分之一的值载荷，不能据此判断完整方案的收益。

在一个明确的参考构造中，每个原rank持有一组4096×1024 BF16权重（8MiB），新token owner需要全部8组，先按源到目的分别交付一次权重，再在同一层、同一权重版本的R次调用中保留。每个非自身peer的旧激活载荷为8192M字节，新激活为2048M字节。每rank完整代次的逻辑交付净节省为7×(6144×ΣM−8388608)，严格为正要求整数ΣM≥1366；M固定时新/旧为1/4+1024/(M×R)。不能把不同层或参数代次的调用合并为R。

| 当前owner M | 每次分发权重，R=1，新/旧 | 同版本保留两次，R=2，新/旧 |
| --- | --- | --- |
| 800 | 1.5300 | 0.8900 |
| 816 | 1.5049 | 0.8775 |
| 834 | 1.4778 | 0.8639 |

因此，临时逐次分发完整权重的这个构造会让当前尺寸的逻辑交付量增加约48%～53%；同层同权重保留两次时，此项交付量才减少约11%～14%。无限复用的该项比例趋近四分之一。这是指定构造的完整逻辑交付账，既不是实际网络线流量或时延比，也不是所有可行实现都必须承担的权重通信下界。若权重已经复制、从checkpoint各自加载、采用其他流式布局或多播，需替换为对应的完整成本图；负的交付节省不能单独否定算子重组，正的交付节省也不能证明加速。

显式独立BF16复制仍对应每rank、每被复制层56MiB额外权重内容。实际新分配、缓存命中、workspace、scratch、排队读者和驻留寿命尚未绑定，不能把它直接换算为显存收益或逐次HBM流量。已有v153的全部head/row/G/T/Q、pad/bias/nonfinite、权重与数值顺序、原生ABI及资源合同继续有效；本轮没有重跑这些未变更的有限参考证明。

另完成32份profile的显式元数据链条：RANK_DEVICE_MAP/NPU_INFO、parallel_group_info、profiler_info、原始DeviceInfo及历史Worker日志按rank/PID/trace精确对应。记录中有16个worker、两组TP8，local_rank在各组内重复0～7，导出的device ID是0～15。启动脚本也保存了完整可见序列。上述证据关闭的是rank到导出设备元数据的连接，仍没有把导出设备ID认证为物理芯片号。[官方设备ID说明](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta1/API/runtimeapi/aclcppdevg_03_2058.html)区分用户、逻辑和物理设备；该公开页是9.1.0beta1的概念资料，没有被当作所装9.2的实际映射证明。

在明确假设“两组TP8分别使用16个不同物理芯片，完整占用8个双芯片板”的条件下，用s表示各组一芯片的split板数，有s∈{0,2,4,6,8}；每TP组56个非自身有向peer交付中，同板8−s、跨板48+s。只分析5个占用类别及代数约束，没有枚举全部rank排列。这是直接逻辑交付模型；它没有选择实际物理放置，也没有宣称普适的物理通信下界。在均匀载荷与相同交付掩码下，激活/权重联合比例中的放置因子相消；绝对值、网络拓扑、复用与物理带宽仍需独立证据。

下一项关键工作是利用已保存的权重、checkpoint、存储资料或WO_A权重访问的定点源码，判定全组权重是否已可用、是否能跨调用驻留，以及对应的层/参数代次和资源成本。全部26模块、358语义行及跨层/跨卡/跨迭代范围保留；9类1352次操作数、849项PMU、16个phase图、3136项归属、G14和原有广泛缺口没有关闭。整体目标仍进行中，没有新增独立性、原生融合、整模块或物理MFU/最低利用率完成声明。未启动新的设备、模型、容器、context或replay。
'''
    overlay=dict(version=156,base_operator_authority_checkpoint=154,retained_previous_overlay_checkpoint=155,candidate_ids=['F14','F15','F16'],module_ids=['M11','M23'],evidence=evidence,conclusion='Theexistingproject-before-headAllToAllproposalneedsonejointactivation/weight/residencycostgraph. Inoneexplicitweightdistributionconstruction,uniformnew/old=1/4+1024/(M*R);currentcoldsingle-callcanincreasecontentwhiletwo-callreusecanreduceit.',broad_candidate_gaps_preserved=True,not_all_representations_or_physical_costs_closed=True,rank_metadata_bridge_only=True,no_new_pair_or_independence_proofs=True,new_native_or_whole_closures=0,critical_path_gain_us=None,overall_complete=False)
    gate=dict(status='ONE_EXISTING_OPERATOR_CUT_JOINT_COST_AND_RESOURCE_NAMESPACE',questions=['Whatisthecostofprovidingall8recipientWO_Aweightgroupstotokenownersunderoneexplicitreplicated-referenceconstruction?','Canreuseandboardplacementchangeorcancelthenetlogicaldeliverysaving?','Whichrank/group/deviceidentitiesareactuallyrecordedandwhichphysicalbridgeisstillmissing?'],source_scope='Onlysaved32DBmetadata/profiler_info/capturedDeviceInfo/archivedlaunch-visibilityand16workerloglines;existingv153operatorcontracts reused;oneofficial9.1namespaceconceptpage. No newframework/operator source read.',target_operator_source_read=False,no_model_operator_implementation=True,no_remote_Torch_CANN_NPU_model_or_replay=True,no_container_start_exec_or_attach=True,Pad_authorization='GRANTED_AND_CONSUMED',overall_complete=False)
    persist_delta(version=156,previous=155,prefix='carrier-resource',changes=['Bind32rank/device/group/log metadata instances while explicitly retaining physical-chip/board gap.','Price activation and one-time weight deliveries jointly for one existing WO_A/AllToAll cut;derive exact sumM>=1366 strict break-even and currentcold/reused counterexamples.','Prove five conditional board-occupancy classes algebraically;no candidate-pair or full rank-permutation enumeration.','Reuse unchanged operator proof hashes/ledgers and compact delta persistence;zero newpair/native/whole closures.'],report=report,action=action,remaining_delta='G156: ExistingheadAllToAll/WO_Acutnowhasjointweight/activationlogicaldeliveryreferenceand32rankmetadatajoin. Actual all-group weightavailability/residency/epochs/nativeparity/allocations/physicaltraffic/capacity and runtime-device->physicalchip remain open;no wholecandidate/model/module conclusion.',source_gate=gate,operator_overlay=overlay,queue=queue,jobs=[dict(kind='profile',status='pass',description='Read-only archived launch/visibility/stoppedinspect/16Workerloglines and32explicit DB/group/profiler/device metadata joins;no physicalmapping promotion or device workload.',artifacts=[O/'rank-device-archived-provenance-v156.json',O/'rank-device-namespace-bindings-v156.json',O/'carrier-resource-inputs-v156.json']),dict(kind='design-check',status='pass',description='Independent one-cut joint logical activation/weight accounting and strict reuse threshold,5conditional board-occupancy classes;preflightPASS. Prioroperator finitecases not rerun;no newpair/native/wholeproofs.',artifacts=[O/'carrier-resource-preflight-v156.json']+[O/x for x in evidence])],scripts=[N/'read_rank_device_provenance_v156.py',N/'analyze_carrier_resource_v156.py',N/'verify_carrier_resource_v156.py',Path(__file__)])


if __name__=='__main__':main()
