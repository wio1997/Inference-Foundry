# v169：用同一批 profile 推进算子结论

本轮没有新增 NPU 负载、模型重载或 profile 采集。171 原服务 health 200，StartedAt 仍为 2026-11-02T19:01:54.649699535Z。现有采集已支持更多结论；不因源码或报表字段缺失重复跑完整 Case。

145,280 次 target 小行数 F21 已在声明的当前图顺序合同下关联到同次 GPU 专家计数和原生时间。路线是唯一 MSTX 消息、同线程 aclmdlRIExecuteAsync、官方 connection 关联的 execute/wait envelope、source graph token/capture epoch，以及每次 40 个 source/native F21 在单一串行 stream 上的唯一保序对应。共 3,632 个 replay envelope，全部 cardinality/shape/order 检查无失败。同进程启动域另有 736 个 exact CPU F21 调用，每个均只发一个原生 F21 kernel。145,280 个独立计数 pack 都在本次 writer 之后、下一 target writer 之前，且在 writer stream；mask 类型均 BOOL。这是有明确图顺序前提的关联，不是运行期逐 kernel CPU creator、跨 rank 全局身份或 baseline all-reader 证明。13,648 条 eager 执行仍未关联，保留未知，不做最近时间匹配。

当前固定 expert ownership/protocol 下，2,518 次本卡有效发送行为零却有正 receiver 工作；643 次 receiver 为零却有正发送工作。这直接反驳两个局部零值跳过守卫。另有 122 次本地两者皆零，也不能自动删除协议参与。765 次 receiver=0 的原生 F21 中位 274.675us；全部 145,280 次中位 324.66us，首层中位 440.02us，其他层中位 323.16us。零本地 GEMM 工作仍有原生时间，说明纯本地 MAC 量不能独自解释耗时；远端计算/访存、Vector、网络和协议等待仍是竞争原因，未选择唯一原因。诊断 observer 增加工作和 reader，这些时间不作为生产加速证明。

147,840 条完整 target 计数含 prefill，在声明的跨 rank 共同边界/完整有序协议前提下构成 9,240 组，包含 160 组 prefill 和 9,080 组小行数。在完整专家不可拆分的行工作代价模型中，排序负载 w_k 的最优最大 rank 负载至少为 max(ceil(sum(w)/16), max_k ceil(k/16)*w_k)。因此加总 row-proxy 改善上界从只看平均负载的 2.67555 收紧到 1.58924；小行数子域单独为 1.62549。活跃专家引用数 proxy 上界为 1.52008。它们不是实际时延/吞吐提速上界。

若改为按行拆分、每份保持完整未压缩 W1/W2、静态副本配置且保留原始副本，为达到各次理想平均 row cap，必要额外副本至少 325 对 INT8 专家权重，即合计 10.711669921875GiB，覆盖 16 rank/40 层之总和，不是每卡。该下界不证明布局可行，也不证明内存不足。解析证明已审查，并用独立的 256 个完整专家小模型、256 个可拆分副本小模型核验下界方向；没有把可行排程倒数冒充理论上限。完整 target 在现有 direct-dense 源算法域的 MAC 工作为 314,394,122,649,600；实际处理能力上界、算术模式和物理池仍待匹配，不宣称 MFU。

启动域 source census 已一次性完成：54,336 条原生计算全部有唯一官方 CPU connection 关联，48 个实际 source 符号归并为 215 个 operator/callsite 组、48,368 次 public CPU 调用。52 个实际 profile source 文件、308 帧的 AST 类名/调用点索引已保存。该 census 耗时 1,042 秒，现已终止，不再重跑。它补齐 v168 的离线汇总失败；保留旧失败记录，不改写历史。

39 个多 kernel callsite 组覆盖 4,304 次 public 调用、相对一调用一 kernel 多出 5,968 条执行。该数量只是分析入口，不是冗余量或可删耗时。已按 metadata 整数链、SP padding/rank 限制、draft 选择、embedding-ID、Engram gate 五个共享语义问题组织，复用既有候选。剩余四组为 SparseFlashMlaMetadata 和三个 setup/view-copy 上下文，保留冷启动/更新语义；同名 kernel 不自动判定重复。全 26 模块保留，21 模块有 source-routing 线索；另外五个 host 模块在此启动原生 census 没有入口，不记零工作。路由线索不冒充独占 phase/模块归属，也不把有歧义的祖先重复求和。

BOOL[20]→BOOL[24] 的公开 F.pad 已在 16 个实际 profile 中定位，共 32 次调用、160 条原生执行；每次是 BOOL20→INT8_20、BOOL 标量→INT8 标量、MemSet、INT8 PadV3、INT8_24→BOOL24。旧 catalog 的 exact-20-to24 列表为空，是过窄的提取条件造成：PadV3 输入 shape 是复合的 20;2;，不是单独 20。针对 exact CPU call/source line 的 datatype/shape 复核已纠正，不能把旧空列表当作不存在。

仅补读这一个 20 行 operator 函数后，完整 SP mask 函数可精确表达为 c=ceil(n/p)，out[j]=input[r*c+j]（索引<n）否则 True。公式覆盖 no-pad clone、正 padding、局部有效区和全 padding 区；n=20,p=8 时各 rank 只需返回 3 个布尔值。已给出逐点代数推导和 10,230 个有限布尔样例核验。允许替换整个公开函数、保留 dtype/value/shape/rank ownership/no-input-alias，且没有必要观察者依赖旧 full-pad base/storage_offset 时，该函数是支持的整体组合/表示重组候选。此结论不需要继续追私有 MemSet 参数；单独删除旧 captured native MemSet 仍未证明安全。所列 44B full-pad 或本地 6/5/3B 外部数组必要 I/O 只属于无缓存、字节寻址、fresh materialized output 的声明模型，不是 HBM/wire 流量或真实加速。

91 个远端新派生文件已保存路径、大小和 SHA256；原始 profile 复用 v168 已封存证据。下一步从这些实际调用点推进其他机会族和一致模块利用率/理论比较，只为影响结论的条件补证。完整 35+7 候选、26 模块358语义行、全部条件分支与跨算子/层/卡/迭代融合拆分重组保持；历史 operand/phase 不伪补，也不作为所有公开函数新实现域的共同门槛。all-reader、数值、回滚/状态、copy/control、物理容量、最低 MFU 和未分析范围继续开放。整体 PROGRESS，未完成。
