# SGLang 36 机制：用途、优化思路与物理上界实验

这是一个完整任务包：每个机制先解释为什么存在、当前怎样实现，再给出一个可证伪的优化方向、适用条件、最小实验和硬件上界分析。最终目标是在明确负载条件下逼近硬件可达性能，不以完成一次小改动为终点。

**主线固定：Ascend + PD 分离 + 推测解码。** 36 是课程机制数量，不代表 36 个新发明，也不代表 36 个都适用或能加速。历史单引擎教学数据没有作为当前 PD 基线导入。

- [任务合同、授权、实验阶梯与停止条件](CONTRACT.md)
- [执行方式与主控 / ZCode 分工](EXECUTION.md)
- [机器可读的 36 点清单](mechanisms.json)
- [待冻结的初始运行合同](runtime-contract.initial.json)
- [已完成的准备工作及证据边界](preparation-evidence.json)
- [首个实证检查点：安装审计、300组支持性判定与relay CPU反例](evidence/2026-10-19-first-checkpoint/README.md)
- [既有缓存与 PD 调优索引](../README.md)

- [M035 / PD-CHUNK-001：151组CPU等价与局部A/A、A/B](evidence/2026-10-19-descriptor-cpu/README.md)

- [M035后续：真实consumer、交织补测与直接三列CPU候选](evidence/2026-10-19-full-consumer-columns/README.md)

- [M017/M018：Ascend CPU长度所有权、staging与目标镜像CPU复核](evidence/2026-10-19-npu-metadata-staging/README.md)

## 阅读方式

每点按“用途 → 实现 → 成本 → 候选 → 适用/回退 → 最小实验 → 必要工作与上界 → 结果”阅读。可以从 M015 / M016 接着当前 CPU overlap 课程，也可以按原课程顺序浏览。最小实验只是进入优化循环的第一步；如果仍有可解释的剩余空间，继续下一轮单机制实验。

## 全部 36 点

| ID | 机制 | 分析状态 | 类型 |
| --- | --- | --- | --- |
| [M001](points/M001.md) | 聊天模板、分词、reasoning parser | 待路径核验 | config |
| [M002](points/M002.md) | 进程分工、IPC 与流式背压 | 待路径核验 | config |
| [M003](points/M003.md) | CPU affinity / NUMA / 线程数 | 待路径核验 | config |
| [M004](points/M004.md) | Continuous batching 与请求状态机 | 待路径核验 | conditional_candidate |
| [M005](points/M005.md) | 多资源准入、输出预留与抢占重算 | 待路径核验 | config |
| [M006](points/M006.md) | Chunked Prefill 的游标与预算 | 待路径核验 | config |
| [M007](points/M007.md) | Mixed batch 与 ragged / packed 输入 | 条件不适用 | not_applicable |
| [M008](points/M008.md) | Cache-aware scheduling：LPM / DFS-weight | 待路径核验 | config |
| [M009](points/M009.md) | Paged KV、请求映射与预分配池 | 待路径核验 | conditional_candidate |
| [M010](points/M010.md) | Radix 树 vs 前缀关联哈希块 | 待路径核验 | conditional_candidate |
| [M011](points/M011.md) | 引用保护、释放、淘汰与 Copy-on-Write | 待路径核验 | conditional_candidate |
| [M012](points/M012.md) | Unified cache 组件 vs Hybrid KV 分组 | 待路径核验 | conditional_candidate |
| [M013](points/M013.md) | HiCache：HBM / 主机 / 存储分层 | 待路径核验 | config |
| [M014](points/M014.md) | 预取、写回与 layer/page 布局 | 待路径核验 | config |
| [M015](points/M015.md) | CPU / 设备 overlap 调度 | 待路径核验 | existing |
| [M016](points/M016.md) | Future token / placeholder 与依赖解析 | 待路径核验 | conditional_candidate |
| [M017](points/M017.md) | 持久状态、增量更新与 GPU metadata | 安装消费者与CPU反证已核验；NPU待测 | conditional_candidate |
| [M018](points/M018.md) | Pinned buffer、异步 H2D 与写读竞态 | 平台声明与CPU所有权已核验；NPU待测 | conditional_candidate |
| [M019](points/M019.md) | 多 stream、event 与资源竞争 | 待路径核验 | conditional_candidate |
| [M020](points/M020.md) | TBO / SBO vs vLLM DBO | 待路径核验 | config |
| [M021](points/M021.md) | 图捕获、形状桶与 replay | 待路径核验 | config |
| [M022](points/M022.md) | 算子融合、编译与中间张量复用 | 待路径核验 | config |
| [M023](points/M023.md) | 权重 / 激活 / KV 量化与尺度 | 待路径核验 | conditional_candidate |
| [M024](points/M024.md) | Attention：MLA、布局、融合与历史读写 | 待路径核验 | conditional_candidate |
| [M025](points/M025.md) | DSA indexer / Top-K / HiSparse | 待路径核验 | conditional_candidate |
| [M026](points/M026.md) | MoE 路由、重排与 grouped GEMM | 待路径核验 | conditional_candidate |
| [M027](points/M027.md) | 后端选择、形状特化与冷启动 | 待路径核验 | config |
| [M028](points/M028.md) | DeepEP dispatch / combine 的模式选择 | 待路径核验 | config |
| [M029](points/M029.md) | TP / EP / DP / PP / CP 的切分边界 | 待路径核验 | config |
| [M030](points/M030.md) | 专家负载均衡、冗余与最慢 rank | 待路径核验 | config |
| [M031](points/M031.md) | MTP / EAGLE：草稿、树验证与接受 | 待路径核验 | config |
| [M032](points/M032.md) | 采样、grammar mask 与结构化输出 | 待路径核验 | conditional_candidate |
| [M033](points/M033.md) | 增量反分词、SSE 合并与结束回收 | 待路径核验 | config |
| [M034](points/M034.md) | Cache-aware routing 与副本亲和 | 待路径核验 | config |
| [M035](points/M035.md) | PD 分离、KV connector 与异步传输 | 待路径核验 | conditional_candidate |
| [M036](points/M036.md) | 会话 / 混合模型 / LoRA / 多模态扩展 | 待路径核验 | conditional_candidate |

## 当前结果边界

已完成逐点设计、36项安装静态入口审计、300组CPU分派/grammar谓词及324组relay状态差分与636个实际producer反例。详见首个实证检查点。另已完成M017/M018的768组门控、180组所有权用例及目标镜像CPU复核。整体仍E0；本任务PD＋MTP服务器基线、NPU完整正确性和性能收益尚未建立。静态入口不能标成运行路径已确认。

若某能力在当前 Ascend / PD / MTP 组合中不支持，先完成适用性探针并记录证据；不得关闭推测解码或切换到其他模型后冒充主线成功。允许有边界的条件特化，但必须计入检测、切换、缓存预热和回退成本。

## 更新规则

主控每轮冻结一个机制，使用真实命令和原始产物完成验证。将结论分为 ACCEPTED、REJECTED、INCONCLUSIVE、NOT_APPLICABLE 或具体 BLOCKED；保留反证。只有完整验证和剩余空间评估后才更新“最佳已验证实现”，不得宣称有限搜索证明了所有条件的绝对物理最优。
