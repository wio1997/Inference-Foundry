# 当前交接 — 2026-10-17（Asia/Shanghai）

## 当前授权和目标

用户已恢复 **172.16.12.32 / 33** 的现场核查、准备、服务启动与测量，并进一步明确两台原有实验作业/预留全部可以停止。仅操作这两台；系统守护进程和访问入口保留。最新要求是 **全部使用 SDMA**，包含设备内、机内、跨机和最终 PD KV 数据路径。控制/初始化通信另记，不能用 TCP/RDMA 数据回退代替 SDMA。

主对话随后转达用户最新要求：**先不考虑 Mooncake，优先核实灵渠 MemCache**。不再新增 Mooncake 部署、修改或测量；已运行探针已结束，保留原始失败与数据校验记录。准确组件现已在镜像中查到：`memcache_hybrid` / `memfabric_hybrid`，不是通用 memcached。优先研究现成接口的 P写池→D读池与直接传输能力，暂不增加 SSD 或做全面后端跑分。用户的性能判断是研究优先级，尚无本环境对照结论。

保持 GLM-5.3 W8A8C8 能力、约定精度和正确性；复用已有正确算子。以 vLLM-Ascend 为正确性/稳定性参考，研究状态、调度与执行机制，最终验证真正 P→KV 传输→D，排除 D 重做 prefill。参数扫描由其他工作负责。本线已有C1临时代表负载的真实PD基线，尚无通过正确性门槛的机制性能收益；业务请求分布、数值SLO和完整正确性验收标准待定。

分支 `codex/glm5-3-vllm-optimization`，本次接手 HEAD `f5571dc0e1c271ae533a347700f24b4a2c105762`，接手时工作区干净。继续 [draft PR #2](https://github.com/wio1997/Inference-Foundry/pull/2)，不自动合并。不改旧副本 `/Users/wio/work/Inference-Foundry` 或原 SGLang 研究分支。

## 协作分工（用户最新校正）

原主对话作为用户沟通、方向/预算与证据复核的主控；本执行对话继续32/33实现、实验、修复及证据交付。主控通过现有HANDOFF/wait_threads读取，无需跨聊天主动发送消息，不增任务/看板，不为普通工程动作增加逐步审批。本次分工校正不打断已加载的per-TP实验。诊断须结合匹配native边界与传输/布局/完成证据，逐token差异不单独证明传输错误；不改模型/量化来迎合结果。

## 当前执行与证据恢复

**最新08:17Z现场**：P11/D10均正常关闭0；诊断请求、真实KV控制已冻结，原件在`node{32,33}-diagnostic-final-v1-backup.json`对应Mac归档并核SHA。P12 `95623b287047e8054863673673a2d13fe05608ae035d0d67ad91499a155c46ac`、D11 `579479baeaf6d4a3194ec7929990d2935799b1b8fbad0b80f2ccc31ad2dc95cf`已启动，正在初始化/加载；root `pd-node32-v12` / `pd-node33-v11`，API与world32 Meta-v4不变。每实例2700s，约08:58Z Mac到期；同6h UID，launch已重查Running/16die/设备空闲。没有payload观察或per-TP保存；P唯一候选为已有FIFO完成fence72ff56，D恢复原worker/backend+对齐scheduler。readiness collector99162有界运行中，Ready后执行`pd_final_put_suite.py`：12条即时P1→D128，再P-native128参考；原D golden/当前D-native单列，不静默换参考。前两条额外比较P/D池local top20。SDK错误、实际3968命中、16个source rank完成事件另审。当前不作机制收益或完整精度验收。主控独立analysis已提交1b78e0d9，本线可继续提交自己的文件。

**08:11Z诊断与下一Loop（优先于下面现场）**：v3无D checksum的4/4同完成边界128输出一致；numeric6/6的8token输出一致但top20最大差0.87248、chosen最大0.17888。追加复用最后同salt的最小对照：P-local、D首次pool、D-pool-local所采top20逐值完全相同；D-native与其重复逐值相同，但与P状态不同。见`numeric-cache-state-analysis-v1.json`，不能继续把原生P/D状态差当SDMA错误。当前正冻结最后source/D控制，之后只为移除启动观察与启用最小交接候选重载P12/D11，有限2700s、无audit/per-TP键。候选`wait_for_final_prompt_put=true`只在最终prompt块等待已有batch.done FIFO事件（worker原3d39b8→72ff56），不固定sleep、不改模型算子精度/SDK/SDMA；事件表示处理结束，不是耐久成功ACK，必须另审SDK错误/实际D命中。验证立即P1→D128时3968命中与原固定参考，同时在D之后用P本地128作为匹配状态参考，不能偷偷换golden或把不同参考合并成“全正确”。若不能保持预定命中/输出则拒绝候选；先完成真实测试，不停在Ready。

**现场08:04Z（以下历史段落不可作实时状态）**：P11 `3d154bff72037bf2e3d11c1ee6a0bb23fd74aa7cf4f92835dbc3ee53e787b6e7`、D10 `c497efe6c455941ff503705714c024914c188b1682385dfe89d7d1c6353883d4`已Ready，API29481/29482，root `pd-node32-v11` / `pd-node33-v10`。原world32 Meta-v4保留27400/01/02。模型有限3600s、约08:31Z Mac到期；08:03Z重查两台6h预留仍Running，UID32 `2cd9108a-eaf0-4164-856b-968bb65b79c6`、33 `ffd5ca30-273d-4646-aa79-7b35478b9416`，约09:10Z节点时钟到期，执行前重查。仅本线服务/资源，不停止其他作业。

恢复原共享MLA键+whole-block对齐，未启用已拒绝的per-TP保存；只加`GLM53_KV_CONTROL_AUDIT=1`观察。worker原3d39b8→b24869、backend原cf2d7a→e0a6f4，scheduler48387a，32/33 staging一致。P PUT前/后在NPU私有stream计算每entry四整数矩、D GET后计算，真实shape/stride/dtype/format记录；只回传控制数组，KV主路径仍device_sdma。控制矩非密码学payload SHA/字节相等证明，观察有同步与时序扰动，不作性能验收。ACL bulk方案仅staging未挂载，不用于真实KV读回。

**当前测试**：`payload-control-v3`四轮4096/128运行中，collector79446。与已完成v2使用相同的真实32key PUT完成guard及同轮bare native-prefix参考；D每进程128key预算已耗尽（文件计数核对），本轮无D checksum计算，P观察仍开。结果决定是否还需要D侧同步诊断。下一步`pd_numeric_boundary_control.py`已准备：等相同真实源key完成后比较6轮4096/8的top20概率，单独标记其不同输出长度/观察条件，尚未执行。不能在Ready/四轮通过处停止，不新增热更新平台，不为只读归因重载。

`payload-control-v2`四轮完成：128源key PUT前/后稳定，1984消费key×rank、每key115条entry的四整数矩全部一致，32rank布局一致，4/4 pool输出等于同轮native。见`real-kv-control-analysis-v2.json`及两台`node*-payload-control-v2-backup.json`，完整源/D控制与请求已SHA回传Mac。v1使用错误默认NONE_HASH种子，0/32key匹配且尚未D GET时失败；改为现场PYTHONHASHSEED=0后复用实例，失败保留。

`payload-observer-budget-control-v1`12轮ABBA完成：native12/12稳定，pool6/12一致。逐请求连接发现立即交接6轮均只加载3072token且输出不同；等待1秒6轮均加载3968且一致。此脚本没有保留v2源完成guard，源审计拖慢发布导致缓存边界混杂，**不能用失败归责D观察停止/传输**。`observer-budget-boundary-analysis-v1.json`保留全部12条；当前v3修正该混杂，不采用固定sleep作为修复。

SDK默认同步完成源码已冻结于`sdk-completion-source-receipt.json`：Python→native BatchGet调用及MemFabric无ASYNC默认等待分支。`KVCacheStoreBatch.done`有FIFO完成事件，wait_for_save目前只enqueue；源码另有释放请求保存fence，不能把异步enqueue直接叫内存安全问题。只有同完成边界的控制结果支持时，才考虑最终prompt保存完成交接候选；当前未改保存完成语义。

模型无关小测：7/31key、512buffer/key、每对象6979584B、异stride、双向one-batch/逐key全部SHA一致、padding未改、close0；只排除该向量小测故障，不证明真实框架stream顺序。1MiB合成ACL async trace明确PCIE_DMA_SQE，真实KV bulk读回已拒绝；NPU int8/BF16四矩小测匹配独立CPU整数参考、trace有AIVEC/MIX无CPU fallback名。上述6份原始小测归档均已SHA回传Mac；一次SSH multiplex会话上限失败的0B partial保留，串行重试通过。世界2小测Meta已正常关闭0，原world32不动。

供主控边界预研的自然产出：`real-kv-layout-capacity-v2.json`记录当前每rank54,528B/token、P1108/D1289个128token块、原例外/scales保留；不是普遍最小状态或服务最大长度。`profile-decode-structural-reference-v1.json`从旧512ctx profile离线提取rank0 B1 decode：78层MLA/attention、5 BF16+16量化Indexer、75组MoE、157all-reduce+1all-gather，少量真实shape/dtype和源SHA。旧profile有扰动，303ms窗口不能替代当前190ms TPOT；次数不是普遍必要资源，SIO/HCCS与SDMA计数有重叠，不相加。无新增NPU采样/重载；主控新analysis目录由主控提交，本线不纳入。

本线继续同PR2，原HEAD7d33725；本轮尚未commit/push。只使用现有HANDOFF与冻结evidence记录，不新增任务/看板。长前缀和C4尚未验收，没有机制性能收益，不改模型/量化/既有算子或SDMA要求。

### 已完成且可接受的基线

`evidence/20261017-pd-baseline/accepted-short-medium-baseline.json`已完成客户端、native命中、SDK总计数和资源覆盖审计。P9/D8完整模型TP16+TP16，固定8K context/4seq/1024批token、eager、`c8_enable_reshape_optim=false`、`mlapo_keep_prefill_weights=true`、loader-fix-v3、whole-block Python对齐补丁；1GiB/rank DRAM池、0HBM、无SSD、纯MemCache。没有启动profiler；正式计时期间没有权重传输、payload全哈希或证据归档。

临时固定C1负载，每组两轮、每轮6条测量，另各轮1条warmup。256输入/64输出：12/12一致，5.1607输出tok/s、TTFTp50/p95=404.34/415.70ms、TPOTp50/p95=190.91/192.79ms；1024/64：12/12一致，5.1606tok/s、TTFT=521.70/542.88ms、TPOT=188.38/190.57ms。所有28条（含warmup）D external_hit=need_load分别128/896，SDK expected=actual=1792get keys、448rank-get，平均rank-get约3.66ms（非全局关键路径）。D无原生错误，16个客户端bmDataOpType均2。5s资源样本各480条，P/D内存约118.3/120.8–120.9GiB，HBM每die约59.9–60.2GiB；资源窗口覆盖基线，但不覆盖较晚的已拒绝首token候选。没有生产SLO/容量或全局最优结论。

原始请求/响应、失败、resources和生命周期console已归档Mac并核SHA：`/Users/wio/work/glm53-execution-evidence-backup/20261017-pd-baseline/node{32,33}-completed-pd-tests-shared-replica-v1.tar.gz`；receipt在本evidence根目录。首次完整模型profile的32-rank SDMA/TP LOCAL/SIO/HCCS证据复用，不拿旧profile耗时当新shape成本。

### 严格正确性边界和实际否定结果

- `correctness-v4`的12项C1单次保护用例通过：127/128/129/255/256/257边界、1024/4096输入、repeat/local命中、stop、256-token部分共享前缀；同缓存边界的native local-prefix与池消费token IDs/text一致、所采logprob差0，冷P/D一致。冷full与缓存路径在部分格式决策不同，独立保留，不宣称完整精度验收。`correctness-c1-gate-v1`追加真实流取消：8token后断开，约0.253s观察running/waiting/delayed-release归零，恢复输出一致。
- C4原始严格输出3/4不一致；随后原生D local-prefix的3轮重复中2轮也有变化，不能归因MemCache，也不能验收C4有效吞吐。
- `baseline-c1-v1`长4096/128重复：6条测量中3条偏离固定native-prefix参考；全部7条（含warmup）external_hit=need_load=3968，无SDK错误。原生D fresh-prime/local复用4次一致（logprobs=5），之后不带logprobs的Pnative→Dpool3组也全一致。长前缀问题仍未闭环，不能推广其计时。
- `handoff-append-first-c1-v1`：P首token立即交客户端并追加到D输入，D生成63个剩余token。warmup即严格输出失败，已拒绝，不计收益。D确实加载1024token，朴素896token，SDK240keys精确吻合，说明机制实际发生而非未执行。详见`rejected-append-first-mechanism.json`。
- `long-handoff-abba-v1`：立即/等待1秒按ABBA各6条，立即6/6一致、等待4/6一致；等待不能修复。固定镜像wait_for_save只enqueue且_previous_save_batch无读者的源码事实不构成根因证明，不采用猜测性同步PUT补丁。config的Indexer topk=2048；4K重复上下文的稀疏选取/量化副本为可调查因素，尚未证明。

### 已拒绝的per-TP保存机制（历史，实例已关闭）

P10/D9只增加`preserve_mla_tp_kv=true`，每TP rank保存原C8 KV并按rank键消费，检验共享副本假设，最多16x逻辑PUT/池存储成本。12条4096/128匹配当前bare native参考的ABBA：native12/12等于原参考，pool8/12（立即4/6、等待1秒4/6）。因此拒绝该机制、不跑完整C1推广门槛、不记收益，见`rejected-per-tp-mechanism.json`。候选代码/计划/失败原件保留，P10/D9正常关闭0、资源采样停止，不是当前服务。

metadata-only探针64个key（4block×16rank）均存在，每对象6979584B，NPU initialized始终false、close0。该证据只证明独立rank键实际建立，不校验payload、不推定逻辑PUT字节均跨机。`conditional-work-accounting.json`与真实布局保存当前图字节计算，非普遍理论下界。之后的小SDK层batch和真实KV控制观察见当前段。

## 前一轮基础部署结果（历史）

纯MemCache P写池→D读池已在完整GLM5.3 W8A8C8/Indexer-int8、TP16+TP16、512 context下通过一个新480-token用例；16-token输出一致，D从池加载384token、本地命中0，外部get keys增加48。不是P→D直传，不是吞吐/精度验收。

上一轮已结束的功能服务：32 P `glm53-memcache-pd-32-fixed-v7-20261017`（29481）；33 D `glm53-memcache-pd-33-fixed-v6-20261017`（29482）；32 metadata `glm53-memcache-meta32-production-v3-20261017`（27400/27401/27402）。只读叠加loader-fix-v3隔离MemCache符号；P关闭官方已知冲突的C8 reshape可选优化。模型、量化、既有算子及SDMA约束保留。D API Ready不是单独的PD证明，上述外部命中与输出对照才是本次功能证据。

服务是本次有限期实验实例：P宿主启动00:57:43、timeout3600秒；D宿主启动00:30:47.691、timeout4500秒；仍受两台7200秒资源预留期限约束。不可把本文当作长期可用保证，继续执行前重查现场。模型profile已完成全32-rank核验；两端大文件备份均已回传Mac并验证归档SHA。最终结果摘要见 [deployment-validation.json](evidence/20261017-memcache-fix/deployment-validation.json)。

## 本线已实测

证据目录：[20261017-inventory](evidence/20261017-inventory/)。后续以每个 run 的 `context.json`、`result.json`、原始 stdout/stderr 和校验值核对；失败也保留。

- 32 宿主机 `B4-33U-Atlas900RCKA3-32`，33 为 `B5-5U-Atlas900RCKA3-33`；K8s 节点名仍为 `a3-5u-atlas900rcka3-{32,33}`。访问 Pod 的 hostPID/hostNetwork 和宿主挂载命名空间已核对。
- 两台均为 8 张物理卡 / 16 die，逻辑 ID=Phy-ID 0..15，每卡两个 die；npu-smi 每 die 65536 MiB。设备内 SIO、不同卡 HCCS_SW；两台全部32个die逐一查询均报告 superPodID=1，server index 分别31/32；各die SDID已留原始证据。同超节点不证明跨机性能。
- CPU：Kunpeng 920 7280Z，4 socket×80 core×2 thread=640 逻辑 CPU，8 NUMA，宿主约 2 TiB RAM。NPU PCI sysfs NUMA=-1，CPU 亲和仍未知。具体网卡/链路/限制见 [HARDWARE_MAP](HARDWARE_MAP.md)。
- 两台实际镜像均为 `quay.io/ascend/vllm-ascend:v0.30.0rc1-a3-arm64`，imageID `sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83`，arm64，RepoDigests 为空。
- 镜像：Python 3.12.13，CANN 9.1.0，torch 2.10.0+cpu + torch-npu 2.10.0.post4，vLLM 0.30.0+empty，vLLM-Ascend 0.30.0rc1；Mooncake 实际发行包为 `mooncake-transfer-engine-npu==0.3.11.post1`，并非未安装。vLLM 源码 SHA `ced6857afa0ea7b2e3f0846a62e1394e90f15607`，Ascend SHA `5f828ad638ee4f9b12db87f9d5481f29076d37f9`，镜像内两仓库工作区干净。宿主驱动 25.5.2，采样固件 7.8.0.7.220。
- 正式权重 `/data/wio/GLM53/GLM-5.3-w8a8c8/` 两台均已存在，incoming 已不存在；196 文件 / 764240832702 bytes（含 `optional/quarot.safetensors`）。当前逐路径 stat 和文件大小匹配，校验清单规范化 SHA `b6843ccf0ffee3cd22357f443dec1006107f958225d13871edc19bebfc88f3be` 与 `WEIGHTS_READY.json` 相符。复用了传输流水线的全量 payload 校验凭据；**本线未重复全量 payload 哈希**，不把 stat 说成独立 payload 校验。
- 量化描述含 `W8A8_DYNAMIC:176830`、`INT8_DYNAMIC:88`、`FLOAT:644`；config 为原 GLM `glm_moe_dsa`、78 层、256 专家/top8、hidden6144。目录名/静态元数据尚不证明运行时 W8A8C8、KV 存储或实际算子路径。
- 33 单 die0 无权重检查通过：容器只见 1 个设备，整数算子输出正确；torch 可见总内存 65787658240 bytes。仅环境功能检查，不是全卡、模型正确性或通信性能证据。该检查发生于旧预留释放前，采样无 NPU 计算进程；不能据此声称当时已取得独占。

## 资源状态及已执行变更

初始 K8s 证据中，32 有 testportal 工作负载请求 2 个 Ascend（逻辑 1、12），33 有 `default/sglang-pd4p2d-reserve-33-20261014` 请求全部 16 个。33 无预留的早期判断是漏读，已纠正；首次新预留被调度器拒绝后已删除，没有绕过进行通信测量。

用户随后明确两台全部可以 kill。已核对 32 的 Deployment `testportal-project/containerworkload-694f3t4n` 只有一个副本且仅位于 32、无对应 HPA，将其副本数 1→0；已删除 33 无控制器的旧预留 Pod。原 Pod 删除完成，系统及访问 Pod 未动。执行凭据：[workload-release.json](evidence/20261017-inventory/workload-release.json)。恢复原业务需另行决定，不能在实验期间自动恢复。

本任务 `default/glm53-execution-reservation-{32,33}-20261017` 首次预留均已 Running/Ready，每台请求16个Ascend、有限7200秒，约07:51 CST复核均为Failed/DeadlineExceeded。此次修复续接已删除本任务过期Pod并重新取得两台16-die预留，见下方“MemCache 崩溃修复与模型部署”的新UID和取证。**每次 NPU 执行前必须重新核对实际 Running/UID/节点/资源归属，不能只看本文。** 集群 admission 拒绝 Pod 直接指定 `preemptionPolicy=Never`；已记录失败，改用集群默认策略，节点 affinity 仅限指定节点，未创建集群级 PriorityClass。

## 前期 SDMA 与单机验证（历史）

- 已核对官方 A3 配置语义：`HCCL_INTER_HCCS_DISABLE=FALSE` 为超节点内 HCCS/SDMA；`HCCL_INTRA_PCIE_ENABLE=1` 配对 `HCCL_INTRA_ROCE_ENABLE=0` 禁用 RoCE；A3 `HCCL_OP_EXPANSION_MODE=AI_CPU` 避免选择 AIV 展开。配置与运行证据分开，具体支持及 communicator 级覆盖仍要核对。
- [hccl_probe.py](tools/hccl_probe.py) 已完成同卡 33[0,1]、异卡 33[0,2] 及跨机 32[2]↔33[2] 的 int8 双向 send/recv 与 bf16 all_reduce；五档字节、预热3+计时12次/项的 payload 校验通过。INFO 诊断日志与 ERROR 级计时分开保存；只有前者用于选路分析。初始缺少 PCIE 配对、CONNECT_TIMEOUT=30低于实际最小120、容器物理/逻辑可见号混用三次失败均保留，未把它们记作性能数据。初始化/控制 TCP 不作为数据通路；不能由 backend 名或环境变量宣称已证实 SDMA。
- **实际 PD connector 的 SDMA 选路和 C8 消费路径尚未运行验证**。HCCL 通过不能代替这项。不要把原来的 AIV/MC² 或 RDMA 配置无审查搬过来；全部 SDMA 约束下，逐个核实实际通信实现。暂停的 Mooncake 探针经过 READ 地址顺序修正后，两台双向64KiB/1MiB/16MiB数据校验完成，但两边析构均出现 `corrupted size vs. prev_size`、退出133；不能记为稳定通过。初次33设备初始化失败发生于模型仍占用die2时，空闲复测已通过初始化，未证明两容器共享设备可用。没有启动 Mooncake PD 服务。
- 已取得本任务资源，基础2-rank跨节点 SDMA 通信功能通过。实际 ACL 映射已核对；容器挂载物理0,2后应使用容器逻辑0,1，不能再用物理0,2筛运行时可见集。跨机测试无 HCCL_LOGIC_SUPERPOD_ID 或外部 rank-table 环境变量覆盖，使用 HCCL root-info/TCPStore 动态建域。继续按真实模型消息补相关组与争用。5 秒背景窗口无明显传输/磁盘读取，只是瞬时诊断；每次测量仍核实相关干扰。
- [TP16 单机容量计划](tools/capacity-tp16-node33.json) 已完成一次完整模型启动和短请求；容器 `62481295791bd1a9e086d5a833cac59a3832a654aa252d79fbbf44e5637bc012` 已主动停止、退出0，保留容器和日志，当前不提供服务。固定 TP16/DP1、无EP、MC2/DSA-CP/FlashComm关闭、int8 KV/indexer、V1、eager、512 context/1 seq。全部16个rank从 `model_runner_v1.py` 加载；每rank日志权重47.21GiB，warmup后torch allocated54.54GiB/reserved54.62GiB（软件口径，非物理HBM），KV预算约7.33–7.35GiB。宿主日志API Ready18:36:16、容器启动18:20:29，加载186shard约477秒；启动耗时仅同宿主时钟范围。短请求HTTP200，12输入/32输出token，以“5”开头回答2+3；单请求功能验证，不是精度或性能验收。原始console183178bytes已拉回Mac校验。实际量化method/缓存tensor布局与C8消费仍需运行对象或trace核验；CLI dtype不是证明。单机容量结果不等于1P1D验收。

## MemCache 初次核实（修复前历史）

- 镜像内 `memcache_hybrid==1.2.0`（VERSION git `4310d68fa8964a3032b71c09296b0ffa80330033`）、`memfabric_hybrid==1.2.0`（`999034e9320abf27bbde7be83a32d3bf4d0f7940`）。SDK API 已加载检查；未安装替换包或编译新kernel。MemFabric的导入provision分支只为950自动编译，本线A3不走该分支。
- 实际 `AscendStoreConnector` 注册的 `MemcacheBackend` 使用 `memcache_hybrid.DistributedObjectStore`，支持显式 `backend=memcache`；SDK配置支持 `device_sdma`。现有PD文档示例是MultiConnector，仍含Mooncake，所以不能用它证明纯MemCache PD。源码中单独Store消费者的 `consumer_is_to_load` **默认false**，纯池消费必须显式true；先按同TP/MLA/Indexer布局核对，不能直接照搬异TP稀疏布局。
- SDK1.2.0提供 `put_from`、`batch_put_from_layers`、`batch_get_into`、`batch_get_into_layers`、key-info/GVA、lease/写完成接口；不等于全部语义已经验证。`COPY_L2G=0`、`COPY_G2L=1`。已核对配置与API签名，后续以真实返回码、payload和运行选路判断。
- 32本任务元数据/配置服务 `glm53-memcache-meta32-20261017` 已主动停止，当前不监听；未占NPU或数据池。运行期间172.16.12.32:27400/27401/27402为metadata/config/metrics，不能叫TCP数据回退。初始12秒就绪观察过短记失败，容器未重启，随后三端口及SDK日志证实MetaService启动（约26秒）。SIGTERM后的服务清理日志已完成，但进程随后同样报heap错误、最终退出134，不能视为干净退出。配置/最终console在 `/data/wio/glm5-3-vllm-execution/20261017-memcache/meta32-v1`，不改全局配置；关闭后的完整归档已备份Mac并核验SHA。
- 空闲32/33物理die2已执行两rank最小合成检查：每rank临时1GiB DRAM池、0HBM池、显式device_sdma、SSD关闭。64KiB/1MiB双向共4对象的P写池/D读池均返回0，接收SHA全部匹配。实际HYBM初始化`bmDataOpType=2`，已与镜像内`HYBM_DOP_TYPE_SDMA=1U<<1`头文件核对；profiler两台各4个`SDMA_SQE`，每个put/get标注区间恰有1个SDMA任务。数据链为源HBM→源侧DRAM池→对端HBM，每对象两次SDK搬运；这不是P→D直传或模型KV消费的证明，也不是建议引入永久缓存层。Trace、日志原件及SHA已备份Mac，仓库留 `memcache-trace-analysis.json`。不拿profile耗时作性能基线。
- 生命周期未通过：仅隐式析构的case退出139；显式`close()`返回0后仍退出133。先释放tensor/清空torch缓存再close也复现。进一步32单客户端只init/close、不建tensor/不注册buffer/不传payload，torch allocated/reserved均0，仍退出133。具体根因未定，不能仅凭此归责到某个库或宣称SDK已稳定集成。第一次LocalConfig拒绝仅MetaConfig支持的`ock.mmc.log_output_target`，已删除该键后固定配置重测，未改协议或回退；所有失败保留。
- 指南要求A3 `device_sdma` 配套灵渠计算网络>=1.5。宿主HDK/CANN版本已取证，**现场灵渠网络版本仍未知**，不从同超节点标签或SDK包版本推定满足；本次正面证据只覆盖已测试的数据路径。纯Store的原GLM C8/Indexer注册、完成通知和D消费仍待模型验证。下一步先对最小init/close复现定位原生退出/动态库生命周期与版本配套，再用单独AscendStoreConnector、`backend=memcache`、`consumer_is_to_load=true`验证P写池→D读池。没有安装替换依赖，没有启动双完整模型，也不回退已暂停的Mooncake路线。

## 官方适配文档核对（最初查询）

用户要求先查 Ascend 官方适配文档。通过 Web Search 与公开 GitCode Git 获取了对应标签/文档，未使用浏览器，未编译、安装、替换依赖或启动服务/NPU作业。来源、固定提交、原文SHA与结论边界见 [official-docs.json](evidence/20261017-memcache-adaptation/official-docs.json)。

- 官方现行[完整底软配套](https://gitcode.com/Ascend/memcache/blob/c3ab6e59caba5f1df5e34222d05df5212da1a6cd/docs/zh/compatibility.md)对A3/HCCS device_sdma要求：灵衢L1>=1.5.0（优选1.5.1）、HDK>=25.5.1、固件>=7.8.0.6.201、OS kernel>=5.10.0；BIOS按内存容量另列要求。现场HDK25.5.2、CANN9.1.0及采样固件7.8.0.7.220达到已核字段的数值下限；L1版本仍未知，未由小测成功推定整套适配通过。v1.2.0 MemFabric README另明确A3 DRAM池化需配套1520L1升级。这里只记录要求，不操作交换机。
- 两个现场VERSION SHA均精确匹配官方v1.2.0标签。MemCache标签的MemFabric子模块固定为`9a9cfbc8a1e026e7c73a558ff27d75372cb2e803`，与独立包`999034e9320abf27bbde7be83a32d3bf4d0f7940`仅两文件的rank-leave/link-down日志及leaveRank_成员改动；ACC/ConfigStore及公开BM头未变化，不能把SHA不同直接判成配套故障。实际二进制构建ABI与加载路径还未核实。
- 官方[issue281](https://gitcode.com/Ascend/memcache/issues/281)/[MR536](https://gitcode.com/Ascend/memcache/merge_requests/536)记录编译期/运行期MemFabric C++ ABI或小版本差异引起C++符号抢占、MetaService coredump，改为纯C ABI与DEEPBIND隔离。源码核对：现场v1.2.0及本次release/1.2 tip1.2.1尚无此隔离，master1.5.0已有；不把新分支修复视为已适配本镜像。上游主要描述启动时string-copy崩溃，现场主要是close后堆错误；目前是相关线索，尚非根因。下一诊断优先实际SO/ELF/ABI与最小case原生栈。
- 已核对v1.2.0 API明确提供close()、device_sdma及1GiB池对齐，所读安装/API文档未找到能解释当前堆错误的特殊关闭顺序。vLLM-Ascend固定v0.30.0rc1指南中A3 fabric环境变量与现场一致；纯Store D消费仍需显式consumer_is_to_load=true。当前没有修复或纯MemCache PD验收结论。

## MemCache 崩溃修复与模型部署（2026-10-17 续接）

用户要求直接解决基础部署问题，已实施隔离修复并继续模型接入。证据：[20261017-memcache-fix](evidence/20261017-memcache-fix/)。

- 真实进程 maps 确认 wheel 原名库与 auditwheel 哈希名库重复加载；原生栈将 close 后 heap 错误定位到 CANN/ATB 全局析构期间的 free。基线 MetaService 可启动且 main 返回0，但进程退出133；仅合并重复库仍退出133。单纯 import 库本身退出0。
- 最终隔离副本 `loader-fix-v3` 保留原1.2.0二进制，将四个 libmf 原名路径链接至 wheel 内 canonical 哈希库，并把 MemCache/UBSIO 的 ctypes 预加载改为 `RTLD_NOW|RTLD_LOCAL|RTLD_DEEPBIND`。通过 PYTHONPATH、LD_LIBRARY_PATH 和只读挂载使用；未替换基础镜像、依赖版本或算子，没有绕过 close/析构。对照证明符号隔离能消除当前复现，尚未确定被抢占的单个 C++ 符号。
- MetaService 连续两次启动/停止/进程退出0。32 原先只 init/close 也失败的 NPU 用例现退出0。32/33 物理die2双向64KiB/1MiB共4对象：put/get返回码、SHA全部通过；两端 close 返回0且进程退出0；每端4个 SDMA_SQE，每个put/get区间恰1个。两个 `memcache-trace-analysis.json` 和归档SHA已保存。这是SDK数据路径及生命周期通过，尚非模型PD验收。
- 两台隔离副本均在 `/data/wio/glm5-3-vllm-execution/20261017-memcache/loader-fix-v3/site-packages`，库校验值一致。小测元数据服务 `glm53-memcache-meta32-fixed-v3-20261017` 已停止且退出0；其ConfigStore会保留首次连接的world_size=2，不能直接供world_size=32模型复用。已重建为 `glm53-memcache-meta32-production-v3-20261017`，仍为27400/27401/27402、有限7200秒，已接受32-rank并挂载P端16个池。首次重建前置bind检查受TIME_WAIT影响，已用SO_REUSEADDR修正；未夺用其他监听端口。预留已续建，32 UID=`a29da17a-3c31-40ca-9f4c-7f72d61c0c27`，33 UID=`7a546c86-8e53-425d-a30e-190454720288`，本次核对Running；仍须执行前重查，有限7200秒。
- 正在部署32 producer、33 consumer：完整GLM5.3 W8A8C8及Indexer-C8，同TP16/DP1，纯AscendStoreConnector/backend=memcache、D端consumer_is_to_load=true、device_sdma、1GiB/rank DRAM、无SSD。v3为补请求级profiler而停止；v4因PYTHONPATH覆盖CANN路径而导入acl失败，已改为保留原值并增加启动前模块定位检查。D的v5因上述world_size旧状态失败，现用 `tools/memcache-pd-node33-fixed-v6.json`，已API Ready且12输入/1输出token请求HTTP200（输出空格，非答案验收）。
- P的v5已API Ready，但首个14-token请求180秒超时。py-spy定位在attention mask构建；进一步GDB原生栈确认Torch队列实际阻塞在前序`aclnnStoreKvBlockMetadata`的`rtsRegisterCpuFunc`，并非已发生KV写池。对应源码`indexer.py:1152`受`c8_reshape_optim_enabled`控制，`ascend_config.py:933`仅对producer启用，公开配置键为`c8_enable_reshape_optim`。D端该分支有效值为false。P已正常停止、退出0，现用 `tools/memcache-pd-node32-fixed-v7.json` 显式关闭该可选reshape优化，继续标准C8缓存写算子；W8A8C8、Indexer-int8、SDMA和模型语义保持。没有编译/引入新kernel。新P已于宿主01:07:00 API Ready，首个真实请求HTTP200（12输入/8输出token，文本以“ 5”开头，19.40秒仅为功能记录）；证据 `node32-memcache-pd-standard-c8-smoke-5f271d0ca7e0/metrics.json`。D保持运行，API端口29481/29482。`memcache_pd_functional.py` 已通过完整模型P写池→D读池功能检查：新480-token提示词、16-token输出，P/D文本完全一致并以“ 5”开头；D外部get key计数0→48。D日志明确vllm_cached=0、kvpool_cached=384，说明加载3个完整128-token块并正常处理余下96token，没有整段重做prefill。两端profiler均正常停止。功能证据 `node32-memcache-pd-functional-v1-600eeb949b16/metrics.json`。模型SDK32个rank均bmDataOpType=2；全32个rank的Ascend trace均有效且含SDMA_SQE，32份节点内TP通信矩阵均只有LOCAL/SIO/HCCS。没有再发NPU请求。当前仅此合成用例功能通过，尚无完整精度/长上下文/并发稳定性或性能验收。
- 官方固定标签[additional_config文档](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0rc1/docs/source/user_guide/configuration/additional_config.md)明确建议池化/AICPU冲突时关闭`c8_enable_reshape_optim`；[issue15896](https://github.com/vllm-project/vllm-ascend/issues/15896)补充HDK26.0以前受影响。现场25.5.2和实际注册栈与此机制吻合，关闭后首请求通过。文档原文字节SHA与摘录见 `official-c8-aicpu-adaptation.json`，不升级驱动、不改变C8精度。

## 完整模型运行证据（本次最终核验）

[deployment-validation.json](evidence/20261017-memcache-fix/deployment-validation.json) 汇总有界用例及所有来源SHA；`node{32,33}-memcache-model-profile-v2-43b402dac851` 保留日志、trace SHA、矩阵摘要及派生计数。

- 32 P 的16个rank共480361个SDMA_SQE，33 D共409744个；每个rank均为正，RDMA命名任务计数均0。每端16份通信矩阵均仅LOCAL/SIO/HCCS；这些是模型TP通信与KV池搬运混合采样，不能把全部SDMA计数归给KV或拿profile耗时作收益。
- 实际量化调用：P的`aclnnQuantMatmulWeightNz` 77520次、`aclnnLightningIndexerQuant` 4864次；D分别65280/4096次。是运行路径证据，不等于完整精度验收，也不能把CPU API与设备kernel计数相加。
- 最初采集器假设已导出trace，实际只有raw profile，保留失败；CPU离线导出第一次因缺少驱动查询设备而失败，且会生成无效/不完整派生trace，未晋升为通过。原生`msprof`单rank对照证明只需`/dev/davinci_manager`查询接口。最终离线容器未挂载davinci0..15计算设备，使用固定镜像和CPU完成解析，全32个rank导出/JSON/SDMA检查通过；无需重跑模型请求。
- 最初原始采样32共3074文件、33共2722文件，已在`profile-raw-preserved`硬链接保全，逐文件与初始manifest的大小/SHA全部匹配，防止解析器data_simplification清理原始采样。原件及派生结果均已备份Mac：32归档1632329001bytes、SHA `ae8e0699a2112bcc1ed34d0b150c610568f7df1687a0bfb397be391ff34d9dc0`；33归档1366352967bytes、SHA `48c928e4053d095361e203efc91497a63285772eb20471906ce8478d8dc62cee`。各自`profile-backup.json`记录Mac路径和完成时间；旧P关闭console亦已单独压缩备份并匹配原始SHA。
- 最终资源快照 `pd-final-resource-check.json` 于Mac01:38:20Z复核两台原UID、16设备预留仍Running，metadata亦Running；`node{32,33}-memcache-pd-final-running-4e69af062ba9` 复核两个模型API与容器。交付前再次通过 `pd-delivery-api-check.json` 核对两个API和metadata仍可用。有限期限沿用上文，后续继续必须重查，未延长期限或创建监控任务。

## 前一轮收尾状态（历史）

Mac 2026-10-16T19:17Z重新核对两台本任务16-die预留仍Running，UID分别 `dfb6ef4d-746f-41ba-a163-20034e9b237f` / `d4d731d9-2d8b-43e1-b986-621b66fa7adf`，7200秒有限期限。仅sleep holder继续保留供续接，**不代表后续时刻仍有效或剩余时间足够**。证据 `final-task-resources.json`。`node{32,33}-final-device-idle-*` 均退出0：两台所有8卡均无NPU进程，本轮模型/Meta容器均已停止，系统和访问入口保留。初版收尾脚本把K8s sleep holder误归为运行实验容器，已按容器名前缀修正后复核，原始失败未删除。

## 操作入口与记录

已有 SSH 复用 socket `/Users/wio/work/sglang-glm53-pd/ssh-control` 已验证，可到用户指定入口，再经既有访问 Pod 进入 32/33 宿主。未找到用户所称 KeyPair-27.pem，当前无需为此停工；未关闭主机身份检查、未使用口令或保存私钥。连接记录不当作长期凭据保证，socket 失效再安全定位。

Mac 固定脚本→上传 `/data/wio/glm5-3-vllm-execution/20261017-inventory/<run>/`→运行/保存原始证据→拉回 Mac 校验。`collect_node.py` 检查 imageID 和本任务资源；结果记录命令、脚本 SHA、git SHA/差异摘要、时间、退出状态。大 trace 原件保留目标节点与 Mac，仓库留摘要和校验值。宿主时钟相对 Mac 有分钟偏差，不调系统时间；跨机不能直接相减 wall clock。

Zcode 入口已确认 0.16.9，常规采集/脚本实现已交其处理并经主控审阅修正；未读取/打印 provider 配置，实际 provider 型号未在本线独立核实。未创建自动化或新 goal。

资料归约已并入 HARDWARE_MAP：CloudMatrix384论文仅取§3.3/4.2.1/4.3.2/4.3.3与CANN9.1正文。论文AIV/独立RDMA平面不当作本线可用能力；全部SDMA约束下优先研究发起/等待、缓冲复用与KV对decode通信的竞争。没有照搬全超节点带宽或EP320结果。
