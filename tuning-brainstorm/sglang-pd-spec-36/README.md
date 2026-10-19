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

- [M021：1633组CPU形状/准入策略与2/3桶反例](evidence/2026-10-19-graph-shape-support/README.md)

- [M019：Q/K副流布局门控与Stream/Event参数合同](evidence/2026-10-19-qk-stream-contract/README.md)

- [M001：完整CPU加载、异步语义与5份精确输入候选](evidence/2026-10-19-tokenizer-inputs/README.md)

- [M001：192次CPU计时、cgroup修复及批处理负收益](evidence/2026-10-19-tokenizer-baseline/README.md)

- [M001：固定两请求cap2局部wall改善，长输入未决](evidence/2026-10-19-tokenizer-cap/README.md)

- [M001：drain候选36组语义与144行CPU计时，含拆批反例](evidence/2026-10-19-tokenizer-drain/README.md)

- [M031：NPU接受合同、123组源码模型与条件反例](evidence/2026-10-19-mtp-verify-contract/README.md)

- [M031：实际metadata CPU组件42组/84次消费者、slot与clone合同](evidence/2026-10-19-pd-mtp-metadata/README.md)

- [M025：NPU seed门控、126项源码模型与六组两轮生命周期](evidence/2026-10-19-npu-dsa-seed/README.md)

- [M023/M024/M025：index cache条件、PD页跨度与108项接口/CPU视图检查](evidence/2026-10-19-npu-index-cache-contract/README.md)

- [M023/M026/M028/M030：MoE生命周期、W8制品描述与114项源码CPU检查](evidence/2026-10-19-npu-moe-contract/README.md)

- [M005/M006：PD prefill预算、提交后停止与57项源码CPU检查](evidence/2026-10-19-pd-prefill-budget/README.md)

- [M009/M010/M011：默认 UnifiedTreeCore、pin/split 与 300 项 CPU 合同](evidence/2026-10-19-unified-cache-pin-budget/README.md)

- [M005/M006/M011：真实 NPU allocator、485 项 CPU 合同与小页 native 门槛](evidence/2026-10-19-npu-allocator-page-contract/README.md)

- [M005/M017：实际页计数调用链、173 项 CPU 合同与同步候选边界](evidence/2026-10-19-extend-page-count-callers/README.md)

- [M005/M017：696 项差分、stale-mirror 反例与计数候选转向](evidence/2026-10-19-cpu-page-count-counterexample/README.md)

- [M005/M017：producer 显式计数候选与 819 项组件 CPU 核验](evidence/2026-10-19-owned-page-count-cpu/README.md)

- [M005/M017/M018：Ascend plan / WAR / 最后读取者与 266 项源策略检查](evidence/2026-10-19-npu-plan-war-source/README.md)

- [M005/M031：PD drain、超时与 Ascend 传输错误路径](evidence/2026-10-19-pd-release-drain/README.md)

- [M004/M016：PD relay最终写入者、完整payload与345项组件CPU核验](evidence/2026-10-19-relay-final-payload/README.md)

- [M016/M031：历史参数勘误、三步hidden必要性与单步前提](evidence/2026-10-19-hidden-eligibility/README.md)

- [M033：真实tokenizer增量反分词、完整发送累积器及间隔取模反例](evidence/2026-10-19-detokenize-finish/README.md)

- [M033：通知/真实HTTP、msgpack状态码反例与条件修复候选](evidence/2026-10-19-frontend-stream-finish/README.md)

- [M032：真实grammar缓存、FSM隔离/回滚与MTP host mask](evidence/2026-10-19-grammar-fsm-mask/README.md)

- [M034：gateway HTTP/gRPC策略差异、字符预算与负载生命周期](evidence/2026-10-19-gateway-pd-source/README.md)

## 阅读方式

每点按“用途 → 实现 → 成本 → 候选 → 适用/回退 → 最小实验 → 必要工作与上界 → 结果”阅读。可以从 M015 / M016 接着当前 CPU overlap 课程，也可以按原课程顺序浏览。最小实验只是进入优化循环的第一步；如果仍有可解释的剩余空间，继续下一轮单机制实验。

## 全部 36 点

| ID | 机制 | 分析状态 | 类型 |
| --- | --- | --- | --- |
| [M001](points/M001.md) | 聊天模板、分词、reasoning parser | CPU候选语义/计时已核验；短/混合局部改善，服务未测 | config |
| [M002](points/M002.md) | 进程分工、IPC 与流式背压 | 待路径核验 | config |
| [M003](points/M003.md) | CPU affinity / NUMA / 线程数 | 待路径核验 | config |
| [M004](points/M004.md) | Continuous batching 与请求状态机 | PD overlap调用顺序已核验；到达/取消实验未运行 | conditional_candidate |
| [M005](points/M005.md) | 多资源准入、输出预留与抢占重算 | PD abort/drain/timeout 源策略已核验；原生释放未决 | config |
| [M006](points/M006.md) | Chunked Prefill 的游标与预算 | 16K chunk 与 200 页分支边界已界定；native 成功路径未验证 | config |
| [M007](points/M007.md) | Mixed batch 与 ragged / packed 输入 | 条件不适用 | not_applicable |
| [M008](points/M008.md) | Cache-aware scheduling：LPM / DFS-weight | 待路径核验 | config |
| [M009](points/M009.md) | Paged KV、请求映射与预分配池 | 默认 unified 路径与索引匹配 CPU 已核验；原生复用未决 | conditional_candidate |
| [M010](points/M010.md) | Radix 树 vs 前缀关联哈希块 | 共享前缀/分裂源码类 CPU 已核验；候选与性能未决 | conditional_candidate |
| [M011](points/M011.md) | 引用保护、释放、淘汰与 Copy-on-Write | 真实 allocator 接入 tree/receipt/free 已核验；物理 KV 生命周期未决 | conditional_candidate |
| [M012](points/M012.md) | Unified cache 组件 vs Hybrid KV 分组 | 待路径核验 | conditional_candidate |
| [M013](points/M013.md) | HiCache：HBM / 主机 / 存储分层 | 待路径核验 | config |
| [M014](points/M014.md) | 预取、写回与 layer/page 布局 | 待路径核验 | config |
| [M015](points/M015.md) | CPU / 设备 overlap 调度 | 待路径核验 | existing |
| [M016](points/M016.md) | Future token / placeholder 与依赖解析 | 历史三步与hidden行合同已核验；原生未测 | conditional_candidate |
| [M017](points/M017.md) | 持久状态、增量更新与 GPU metadata | NPU override/最后读取者策略 CPU 核验；原生收益未测 | conditional_candidate |
| [M018](points/M018.md) | Pinned buffer、异步 H2D 与写读竞态 | plan/WAR 266 项策略 CPU 检查；原生生命周期未决 | conditional_candidate |
| [M019](points/M019.md) | 多 stream、event 与资源竞争 | 布局/包装层已核验；原生接口及性能未决 | conditional_candidate |
| [M020](points/M020.md) | TBO / SBO vs vLLM DBO | 待路径核验 | config |
| [M021](points/M021.md) | 图捕获、形状桶与 replay | 安装与CPU策略已核验；NPU图性能未决 | config |
| [M022](points/M022.md) | 算子融合、编译与中间张量复用 | 待路径核验 | config |
| [M023](points/M023.md) | 权重 / 激活 / KV 量化与尺度 | W8制品/指定加载方法已核验；实际加载与原生未决 | conditional_candidate |
| [M024](points/M024.md) | Attention：MLA、布局、融合与历史读写 | NPU MLA/PD视图源码模型已核验；原生未决 | conditional_candidate |
| [M025](points/M025.md) | DSA indexer / Top-K / HiSparse | 源码门控/状态模型已核验；完整模块与原生性能未决 | conditional_candidate |
| [M026](points/M026.md) | MoE 路由、重排与 grouped GEMM | NPU dispatcher源码/CPU状态已核验；原生MoE未决 | conditional_candidate |
| [M027](points/M027.md) | 后端选择、形状特化与冷启动 | 待路径核验 | config |
| [M028](points/M028.md) | DeepEP dispatch / combine 的模式选择 | Ascend分派门控/状态模型已核验；DeepEP原生未决 | config |
| [M029](points/M029.md) | TP / EP / DP / PP / CP 的切分边界 | 待路径核验 | config |
| [M030](points/M030.md) | 专家负载均衡、冗余与最慢 rank | 路由必要工作边界已建立；实际rank负载未测 | config |
| [M031](points/M031.md) | MTP / EAGLE：草稿、树验证与接受 | NEXTN三步历史参数确认；模型/PD原生待测 | config |
| [M032](points/M032.md) | 采样、grammar mask 与结构化输出 | 真实grammar/hostMTP mask CPU已核验；原生待测 | conditional_candidate |
| [M033](points/M033.md) | 增量反分词、SSE 合并与结束回收 | 前端IPC/SSE组件已实测；原生/回收待测 | config |
| [M034](points/M034.md) | Cache-aware routing 与副本亲和 | 实际Rust路径已核验；策略实测未运行 | config |
| [M035](points/M035.md) | PD 分离、KV connector 与异步传输 | 待路径核验 | conditional_candidate |
| [M036](points/M036.md) | 会话 / 混合模型 / LoRA / 多模态扩展 | 待路径核验 | conditional_candidate |

## 当前结果边界

已完成逐点设计、36项安装静态入口审计、300组CPU分派/grammar谓词及324组relay状态差分与636个实际producer反例。详见首个实证检查点。另已完成M017/M018的768组门控、180组所有权用例及目标镜像CPU复核。整体仍E0；本任务PD＋MTP服务器基线、NPU完整正确性和性能收益尚未建立。静态入口不能标成运行路径已确认。

若某能力在当前 Ascend / PD / MTP 组合中不支持，先完成适用性探针并记录证据；不得关闭推测解码或切换到其他模型后冒充主线成功。允许有边界的条件特化，但必须计入检测、切换、缓存预热和回退成本。

## 更新规则

主控每轮冻结一个机制，使用真实命令和原始产物完成验证。将结论分为 ACCEPTED、REJECTED、INCONCLUSIVE、NOT_APPLICABLE 或具体 BLOCKED；保留反证。只有完整验证和剩余空间评估后才更新“最佳已验证实现”，不得宣称有限搜索证明了所有条件的绝对物理最优。
