# M025：NPU DSA seed 复用与必要工作合同

实际安装代码 `ce0a8fea5f4245839d174b2b8c55692be1810a7f` 的复用门控位于 NPU attention 上层。NextN 设置 `skip_topk=True`，`forward_dsa_prepare_npu` 在没有 previous indices 时选择 indexer，否则复用 previous indices。不能仅凭 `dsa_npu_indexer.py` 没有 seed 参数，就判断 Ascend 忽略 MTP seed。

本检查点完成 126 项源码提取模型检查和六组两轮生命周期检查。它们使用未改动的类/函数 AST、真实 Torch 2.10 CPU tensor，以及真实 ForwardMode、SpecInput、runtime test override；**没有完整导入 IndexTopKShareState 模块，没有执行 NPU indexer 或模型，没有测性能**。整体 serving 仍 E0，已接受的 PD+MTP baseline 仍为空。没有生产代码改动。

## 当前实现与证据

| 位置（本目录保存完整源码） | 已核验的事实 | 边界 |
|---|---|---|
| `models/deepseek_nextn.py` | `from_mtp_carry` → decoder 的 previous indices → update/publish | 模型 forward 未执行 |
| `models/deepseek_v2.py:2029` | NextN 的 skip/next_skip 为 True；普通层由层模式决定 | 不能把 NextN 三步次数套到所有 target 层 |
| `hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py:509` | 选择谓词为 `not m.skip_topk or (m.is_nextn and prev_topk_indices is None)` | 提取该分支与 DCP 谓词，indexer 为 spy；CP/DCP 不活跃 |
| `layers/attention/index_topk_share.py` | carry 读写、capture copy、finally 清理 | 完整模块导入受阻；独立执行未改类体 |
| `models/deepseek_common/attention_backend_handler.py:75` | use_dsa 时 EXTEND、DECODE、VERIFY、DRAFT_EXTEND、IDLE 都进入 DSA_NPU | handler/enum 提取执行，注册仅静态核验 |
| `layers/attention/dsa/dsa_npu_indexer.py:234` | pool scale buffer 决定量化分支；indexer 写 index-K cache 后调用 lightning indexer | 真实 pool dtype、scale buffer、原生内部分支未知 |
| `speculative/eagle_worker_v2.py:1142` | draft-extend 的选行表达式为每请求宽度偏移加 accept_lens-1 | 提取实际赋值 AST，接受长度是夹具 |

固定模型 metadata 的 SHA256 与前次相同：hidden_size=6144、index_share_for_mtp_iteration=True、index_topk=2048、index_kpool=1；实际 helper 得到 seed 宽度 2048。配置及目录名均不能证明权重、activation 或 KV 的运行 dtype。观测 steps/topk/draft=3/1/4 是现有单引擎服务事实，不是本任务的 PD 运行合同。

## 运行记录和反证

| Run | 结果 | 内容 |
|---|---|---|
| R001 / R002 | SOURCE PASS | 八个模型/indexer 文件及实际状态类，固定 HEAD 与内容一致 |
| R003 | INVALID，原样保留 | collector 误把 attention_forward_methods 包当单个 `.py` 文件，FileNotFound；inner exit 1、outer CLI 0 |
| R004 | SOURCE PASS | 新 collector 用 git ls-files 枚举包，共十二个依赖/分派文件；没有重跑 R003 |
| R005 | UNAVAILABLE | 完整状态模块导入经 cp/zigzag → memory_pool → cache_move 依赖缺失的 `sgl_kernel` CPU extension；未运行测试。inner 0 只表示结果已记录 |
| R006 | PASS，源码模型范围 | 32 carry、64 capture、8 门控真值、2 异常清理、20 分派，共 126 项 |
| R007 | PASS，源码模型范围 | batch 2/3、宽度 2048，六组各两轮 draft+draft-extend |

R005 的不可用没有被 R006/R007“修复”。后两次使用完整、未改动的 `IndexTopKShareState` 类体 AST，类体摘要 `c73f5f585a4ebf4422677e8cc9fd26307fb5464199b37b7fc8bcfc8825b9187e`；CP helper 原体保留，CP strategy 为 None 夹具。handler 和 enum 也分别提取。真实 runtime 使用库的 test override；完整模块注册、依赖加载和 NPU 行为仍未验证。

R006 中，三步 draft 的 spy 选择次数分布为 3 次×16 组、1 次×12 组、0 次×4 组。capture 覆盖 EXTEND、DRAFT_EXTEND_V2、DECODE、TARGET_VERIFY；实际 `is_extend(include_draft_extend_v2=True)` 包含 TARGET_VERIFY，不能根据模式名称猜测 eligibility。验证了 carry 引用别名、capture 稳定地址及 copy、选行和异常 finally 清理；未测试 CP gather。

## 一轮零次 draft 选择不等于整轮零成本

| 两轮夹具条件 | 第 1 轮 draft / extend / 合计 | 第 2 轮 draft / extend / 合计 |
|---|---:|---:|
| sharing=True，初始无 seed | 1 / 1 / 2 | 0 / 1 / 1 |
| sharing=True，初始有 seed | 0 / 1 / 1 | 0 / 1 / 1 |
| sharing=False，无 seed | 3 / 1 / 4 | 3 / 1 / 4 |

batch 2/3 各覆盖上述三行。draft context 结束清掉 carry/reuse；随后 draft-extend 使用独立 spec/capture，previous=None，spy 仍被选择一次。实际 select_index 表达式对 accept_lens=[1,4] 或 [1,4,2] 产生 [0,7] 或 [0,7,9]；高级索引生成的下一轮 seed 在原 capture 清零后保持不变。

这些数字只统计**源码谓词选择次数**。draft 每步 N 行、extend 4N 行是本夹具的构造，不证明原生图中的 padding、ragged/accepted rows、kernel 数量或成本。不能把 4→1 称为 4 倍加速，也不能把下一轮 draft 的零次选择称为整轮无索引工作。

## 修正短序列路径假设

M025 原先依据 NPU 文件注释写“已有短序列跳过 Top-K 的路径”，证据强度不足。实际 Ascend handler 对 DSA 不按 512/32768 分流，均选 DSA_NPU；本次所追 NPU indexer 的相应分支会调用 lightning indexer。通用 Indexer 的 CUDA/HIP 短序列分支、以及 NPU MHA 准备函数内的 `return_indices=False`，不能当成本 DSA_NPU 路径已经跳过选择的证据。

这也不证明 NPU 原生 operator 在 L≤K 时一定做完整选择：operator 内部可能有其他策略，当前未知。512/32768 的 route 检查仅证明此 handler 不以长度分派；真实 native 工作量和索引语义需要单独测试。HiSparse shared-index prefetch 在先前安装文档中排除 speculative decoding，其不适用结论维持，不能连带排除一般 DSA。

## 工作量边界、候选与否定条件

当前没有提出或应用新优化。先按实际门控记录 `g_i`、真实 query rows `Q_i`、有效上下文 `L_i`、K、heads/dtype、图 padding、pool/scale buffer、cache 写入和传输事件。索引候选工作可以记为 `sum(g_draft_i * Q_draft_i) + g_extend * Q_extend`，但实际 Q 尚未知，且 query 行数仍不等于 FLOP/字节或时间。物理时间必须另计投影、index-K cache 更新、lightning indexer、capture/gather 和依赖链；重叠部分不能直接相加。

原生最小合同见 [native-next-contract.json](native-next-contract.json)：需要本任务明确分配的 Ascend 资源，先建 PD+MTP 正确性基线，再以相同输入/接受分布测量。重点包含冷/热 seed、共享关闭作为诊断对照、batch 2/3、512/2047/2048/2049/32768、target/draft/extend 分开归因，及 cache 连续性。关闭 MTP、减 K、减少选中 token 均不属于等价优化。

新增 seed capture、选行 copy 和状态维护有成本；短序列、低接受率、图 padding 或 native 已有 fast path 时可能不划算。若同条件 trace 显示复用已生效且剩余开销主要来自必要工作，应停止重复实现该方向；若 cache/索引/输出不一致则拒绝候选。没有同条件原生数据前，不写加速比、accepted length 改善或物理极限。

## 可追溯性与环境

[source-index.json](source-index.json) 保存 43 个规范源码哈希及实际导入文件列表；[source-model-state-result.json](source-model-state-result.json)、[source-model-cycle-result.json](source-model-cycle-result.json) 保存原始结构结果；[full-module-unavailable.json](full-module-unavailable.json) 保留首次完整导入错误。所有 frozen specs、controller receipts 和失败 stderr 均收录。[zcode-audit.json](zcode-audit.json) 为七个精确会话的只读元数据审计：deepseek-flash 是配置别名，实际后端发行版 UNKNOWN；每个 frozen command 恰好一次，无 Write/Edit。outer CLI 0 不替代语义判定。

R005–R007 各使用新的任务 CPU 容器：runc、1 CPU、4 GiB、无设备/网络、只读 root、仅四个 metadata/tokenizer 文件只读挂载，无权重。三个容器均核对 ID/镜像/任务标签后清理，现有服务 ID、启动时间和 restart_count 前后相同。远端与本地墙钟有偏差，时间戳按各自 clock domain 保留，未用于跨主机性能计算。未发送模型生成请求。

Loop 结论 INCONCLUSIVE：源码分支及 tensor 状态证据已增加；完整模块不可用、原生正确性/工作/性能和 PD 生命周期仍待测。保留所有失败，不提升整体 E0。
