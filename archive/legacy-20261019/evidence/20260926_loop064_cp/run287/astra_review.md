# Astra High：Run287 ownership census 与双层 Bound 审查

2026-09-26。独立读取 analysis.json、全部40份原始ownership JSONL、Runtime报告、client摘要和当前源码；未使用NPU、未改源码。本文不替Sol裁决KEEP。

## 1. 独立核验事实

- 40 rank-cohort，合计11,968 rank-cycle；每rank五cohort周期294/297/297/315/293（共1,496 cycle）。各JSONL cycle连续，与对应Runtime周期一致。
- 40/40 Runtime报告FULL Graph、handoff_before_model_forward，generated_output_counts均1024；warmup48与bench12共60请求usage1024。这验证此次覆盖，不是数值语义全证明。
- 全部原始记录global_qsl=[0,8,...,96]；本地query总12行，owner为两个请求，每请求完整8行，所以owner_full_update_rows=16。owner集合rank0..7依次为(0,1),(1,2),(3,4),(4,5),(6,7),(7,8),(9,10),(10,11)。
- start_pos=global_seq_lens−8；local_seq_lens等于global长度扣除本rank查询尾部causal offset，非owner为0。独立重算无违例。包括active_mask变化/parking阶段，owner集合稳定。
- 原分析中每个c4 domain的owner compressed-history包络与非owner派生compressed新页均不相交。另独立以原始表检查：owner compressed-history页 vs同domain非owner state新输入页，在indexer/main两类各11,968条记录也都不交。
- 上述cohort编号是worker局部constructor序号；本次与Runtime cohort=seq+1和cycle计数吻合。仍未在probe自身携带request_id，不能把这个序号约定推广为通用跨生命周期身份协议。

置信度：几何/本次页表事实高；删除副本安全性低至中，关键闭包仍未测。

## 2. 页集合“不相交”证明到哪里

证明的是：在已记录blocktable/正确cache域/所用派生公式下，选定两个请求的压缩历史页与其余十请求的派生当前页没有物理page重合。这显著降低“当前owner会经同domain页别名消费非owner新写”的风险。

尚不能推出：
1. 派生compressed index区间 floor(start/4)..floor((start+8)/4) 与实际native CompressorMetadata/scatter完全一致；目前没有actual slot trace。
2. state的历史递归读取也不交。记录的是新输入位置页，native overlap/state读域仍unknown。
3. 多层混合storage alias全部安全。layer2 main/compressed/state与layer0/1/3等共用视图；当前检查没有穷尽这些layer的read/write域。indexer state/key/scale同storage也必须按typed byte区间核对。
4. DSpark、prefix hash/refcount/COW、connector、下一cohort有效frontier不会消费省略写入的旧区间。
5. 当前first-owner只查split请求前4query，可省掉该请求后4更新。后半KV未来cycle仍可见；保留单位是完整请求8行。

compressed history使用global seq的ceil页包络，包含partial/unwritten页，是保守read包络而非实际native读集。page无交若包络确实覆盖真实读/写则有证明力；实际write覆盖尚未核验，不可略去前提。重复0/null页、物理cache域、dtype别名都应明确处理。

## 3. Resource/Hardware Bound 与 Scheduling-aware Bound

Run287推翻了“96行更新是由本rank当前query归属直接强制”的简单假设，但没有把其中80行正式归为可删除工作。

可在模型新增证据行：
Current：每rank按全部12请求×8行更新；
conditional request-owner implementation：每rank16完整行、8rank合计128行；
work/traffic saved：unknown（待native工作与合法生命周期闭包）；
HBM/latency/TPS saved：unknown（权重/tile读量、KV/state写量、shape效率、通信补齐未测）。

16行也不是Hardware Ceiling的必要行数。若对应projection权重/数学state确实在rank间复制且允许一次计算后传送结果，全球唯一输入只有96行；当前owner方案128行保留了split请求复制。要减少到其他布局须把新增数据传输、历史state迁移、负载不均和调度成本计入。不能把当前查询分片架构反过来定义数学compulsory work。

Scheduling Bound不由这次同步probe给出时长；合法删工作会改变资源负载与DAG节点，但实际critical path、rank rendezvous和共享HBM效率仍需因果实验。此Run大量Host同步，不能拿它的wall/cycle校准正式TPS。

## 4. Ownership线下一步：能提候选，尚不能执行删写晋级

合理候选仍是一个c4 layer的indexer update完整请求选择（96→16），不是local12行选择。
最有价值的下一项ownership因果实验应在更短闭包中进行：
- 先取得一个真实layer2 fork/QLI入口fixture，核对native CompressorMetadata实际slot、typed state读写域与key/scale别名；
- 用同一fixture分别执行全12请求更新与compact两owner请求更新，完整保留原QLI请求索引/metadata，比较owner typed cache/state、QLI topk与后续Sparse输出；A/A先证明局部可重放。
- 初始离线/诊断fixture不得作为有效prefix发布；nonowner区域还原后退出，避免在prefix闭包未完成时污染服务。
这只能证明局部等价与成本；完整cohort/后续cohort的省略写入仍待前沿证据。若native/state闭包无法低成本补齐，先不做删写。

## 5. H003是什么，以及是否优先

**H003是“Target decode hidden AllGather 与独立 local Q projection 并行”**，不是H001的Compressor/QLI fork，也不是H002的ownership删写。

具体源码：dsa_cp.py:_forward 当前 overlap_hidden_states_gather仅在has_prefill成立；decode先maybe_all_gather_and_maybe_unpad，再做local wq_a→RMS/dynamic-quant→wq_b→Q RMS/RoPE。local Q仅依赖hidden_states_local与本rank权重，语义上不依赖gathered hidden。首次gather消费者是hidden_states_cache→wkv；wait必须放在该消费者前。distributed/utils.py:all_gather_async已有同TP device_group的dist.all_gather_into_tensor(async_op=True)。910B3的A5全权重o_proj gather应确认禁用；不得意外交换其他collective顺序。

短期真实性能intervention我优先H003，理由是它不删KV/state写集，语义闭包更短，可以较小代价证伪HCCL是否实际进展；不是因为预期收益已证大于ownership。H002保留为潜在更大的资源/架构Gap，继续定向闭包，不能以H003失败结束极限搜索。

### H003首个最小可证伪实验

选一个已知Target layer（建议固定layer2，仅decode96、原c12），三个arm：
A0：当前原始同步helper；
A1：相同async helper、相同buffer/collective，launch后立即wait；
B：相同async helper，先执行独立localQ链，再在wkv之前wait。
A1/B只差wait位置；A0量化helper/stream/event重排开销。mode各rank一致；固定实际shape/dtype/pad规则和相同TP group；不要将async_op=True本身当作overlap证据。

先在固定layer入口输入上比较gathered hidden与local Q输出，A0/A0自重放作基线；因该窗口尚未写KV，可避免Run285五次完整Target上游差异。通过后才做实际FULL Graph A0/A1/B的独立capture/replay，明确graph身份；不可只改env复用旧Graph。
采 all8 ranks：AllGather launch/start/end、localQ各节点及总段、wait/first-wkv start、next collective arrival/end、Target/cycle端点，记录HCCL顺序/消息shape与事件依赖。Graph必须显示通信与Q实际并行；若wait提前阻塞、HCCL不progress或Q膨胀消除join改善，则局部假设被证伪。

预期判据：
- 语义：gather与localQ满足冻结数值门槛，无新增rank/pad/layout差异；继续产品correctness。
- 调度：B相对A1在最迟rank的“gather+Q完成→first-wkv可启动”端点稳定提前，而不是仅sum任务时间减小；收益到下一collective仍保留。
- 实用：B相对A0的Target/cycle收益可分辨，最终由重复未profile正式E2E裁决。单层收益未解析可扩展测量灵敏度，但先过数值/Graphgate；不凭局部×层数宣告KEEP。

H003失败也只说明该合法松边在当前硬件/Graph/资源条件下未转成收益，不证明接近Scheduling或Hardware极限。
