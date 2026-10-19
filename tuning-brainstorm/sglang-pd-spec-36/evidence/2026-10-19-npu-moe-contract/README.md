# M023 / M026 / M028 / M030：Ascend MoE 状态、量化描述与必要工作

本轮追到安装代码 `ce0a8fea5f4245839d174b2b8c55692be1810a7f` 的 GLM-DSA MoE 路径，完成 **114 项源码/CPU 检查**，核对模型量化描述，并排除了在当前 W8A8 条件路径上重复“把静态 scale 转换移到加载期”的候选。没有生产改动，没有原生 MoE、通信或性能结果；整体 PD+MTP serving 仍 E0，accepted baseline 为空。

## 路径与模型制品

GLM-DSA 的 target 和 NextN 分别继承 DeepseekV2 与 DeepseekV3 NextN 路径，不能套用同一文件中的普通 Glm4MoeSparseMoeBlock。固定 config 描述 E=256、routed k=8、shared=1、H=6144、I=2048、noaux_tc、sigmoid、n_group=topk_group=1、renormalize=True、routed_scaling_factor=2.5。一个 expert group 仍进入 grouped TopK 配置；scale 应用位置取决于实际 quant method 的融合标志，不能仅由 2.5 推出返回权重之和。

实际源码的条件分派为：

| 条件 | 源码行为 | 未证明的范围 |
|---|---|---|
| NPU 且 moe-a2a-backend=none | 创建 AscendTPDispatcher | 本任务实际配置、分布式通信与 native 执行 |
| ascend_fuseep | 普通 dispatcher 为占位，FusedMoE.forward 直接走 forward_fuseep | 实际 FuseEP 使用、权重布局与性能 |
| DeepEP 等后端 | 另一 dispatcher 路径，部分使用 NPU 通信 group | 包存在不代表本任务使用；native 支持未测 |
| ModelSlim runner=auto | 选择 ASCEND runner | actual loaded scheme、target/draft 的最终 dtype/kernel |

[quant-metadata-summary.json](quant-metadata-summary.json) 来自命名文件 `quant_model_description.json`，只读取 JSON，不读取 tensor 权重。文件 12,975,094 B，SHA256 为 `ddff020caf4158859060a932e90c52339f6cf3274694ba3bb9f7bfc153ae2378`，177,567 个顶层条目；前后哈希一致。筛选出的 MoE 投影 weight 条目均为 W8A8_DYNAMIC：

| 描述文件范围 | routed 投影条目 | shared 投影条目 |
|---|---:|---:|
| target 层索引范围 | 57,600 | 225 |
| 第 78 层（post-target candidate） | 768 | 3 |

target 的数字对应 75×256×3 与 75×3；第 78 层为 256×3 与 3。它们是制品描述计数，不是 GPU expert 调用次数、HBM 字节或最终已加载 dtype。第 78 层按索引标为 NextN 候选，仍需确认实际 draft loader 的映射与配置。

首次 R005 因 8 MiB 冻结上限而未读正文，结果保留为 UNAVAILABLE。R006 在新 Run 中将上限设为 16 MiB，reader 代码不变，只导出哈希、计数与少量示例；未发布完整 13 MB JSON。

## R004：64 项 dispatcher 状态检查

完整 BaseDispatcher 源文件通过 file-spec 原样加载，未启动其父 package；AscendTPDispatcher、相关 NamedTuple、dtype selector、init/finalize wrapper 的 AST 类体保持不变。只在该提取环境中将 `torch.ops.npu` 替换为明确的 spy。真实 Torch 2.10 CPU 执行 cast、别名、reshape 和 hook；没有 native 注册或 NPU permutation/quant/finalize 数值执行。

48 组基本用例覆盖 token rows=2/3、k=8/9、BF16/INT8/MXFP8 输出分支、权重 FP32/BF16、IDs int64/int32，H 取固定 config 的 6144。T 是行数夹具，不是实际请求并发或 prefill/verify 行数。k=9/E=257 是本地 fused-shared 布局诊断夹具，不代表实际 EP placement。

检查确认源码传递 `active_num=T*k`、group_list_type=1，expert counts 转 int64；权重转 hidden dtype、IDs 转 int32。相同 dtype 的 `.to` 保持别名，改变 dtype 会形成新 tensor。MXFP8 的 uint8 scale 是边界夹具，只验证 `[rows,M] → [rows,M/2,2]` view，不能当作 native e8m0 值或格式证明。

| 生命周期构造 | 观测 |
|---|---|
| combine 前没有 dispatch，或成功 combine 后再 combine | RuntimeError |
| 正常 dispatch → combine | finalize 收到该次 metadata；成功后 pending 清空 |
| 同对象连续 dispatch 两次 | 单个 `_dispatch_output` 槽被第二次覆盖 |
| 已有 pending 后注入 init 异常 | 保留原 pending 对象 |
| 注入 finalize 异常 | 保留 pending；夹具中后续成功调用才清空 |

6 组故障/交错、8 组 producer mutation、2 组完整 BaseDispatcher hook 检查均通过。原 buffer 在相同 dtype 下被修改，改变会传到 retained metadata；转换形成的副本不受原 buffer 修改影响。这里有明确的所有权前提，但**没有证明生产会发生该交错、错误恢复或提前修改**，未据此修改状态机或建议重试 native 故障。CPU 方法返回也不能证明异步设备读取已经结束。

详细结果：[dispatcher-source-result.json](dispatcher-source-result.json)。

## R007：50 项 W8A8 静态处理与 apply 检查

从 `NPUW8A8Int8MoEMethod` 提取未改动的 load/apply 方法和三个 base helper。使用小型 CPU 合成 Parameters，固定 FuseEP=False 边界、format_cast identity spy、动态量化与 GMM spy；未执行原构造器、真实权重加载或 native layout/数值。

- 16 组指定加载调用覆盖 w13/w2、local expert 数 2/3、offset 有/无、scale FP32/BF16。scale squeeze 后为 BF16，offset squeeze 保留 storage，权重 transpose/contiguous 在 CPU 上真实执行；仅 w13 设置 dispatcher INT8。
- 32 组 apply 各连续调用三次。静态 scale 保持同一对象与 storage；已有 per-token scale 时动态量化边界调用为零，缺失时每次调用一次。bias 优先 scale_bias，否则 weight_bias。
- 两个缺少 weight 的人工输入在静态修改前被拒绝。它们不是实际模型加载失败。

这 96 次 GMM 边界调用均为 spy。没有测量量化误差、矩阵乘结果、加载次数或时延。源码调用位置与实验共同说明：**该 W8A8 条件路径已经把这里的静态转换放在指定加载方法中，并能复用已有 per-token scale**。当前不提出重复缓存改动。若后续 trace 发现不同分支仍在 forward 重做静态工作，须对那个实际分支另建合同。

详细结果：[w8-source-result.json](w8-source-result.json)。

## M030：路由数量、expert 并集与 rank 负载

设 T≥1 是该 MoE 调用的实际有效 token 行数，每行选 k=8 个不同 routed experts，且没有 dropped routes，则逻辑 routed assignments 为 8T，expert 并集满足 `8 ≤ U ≤ min(256,8T)`。T=2 时 U 可为 8–16，T=3 时为 8–24。T=0 时逻辑 assignments 与并集均为零，图 padding 的设备工作仍须另记。shared expert 另外计数。请求并发 2/3 不足以确定 T；prefill、target verify、draft、图 padding 和 TP/DP scatter 都须分别记录。

[work-accounting.json](work-accounting.json) 给出条件推导。若将 H=6144、I=2048 代入完整未分片的 gate/up/down 稠密投影，则每个 expert 的权重元素为 `3HI=37,748,736`；在 INT8、无压缩/packing padding 的假设下为 36 MiB。每条 routed assignment 的 MAC2 计数为 `6HI=75,497,472`，不含 router、activation、量化、padding、shared 或通信。它们是逻辑账，不是测得的 HBM 读取或 NPU FLOP。

物理成本还依赖：expert/TP 分片、redundancy 与 EPLB 映射、缓存命中、GMM tile padding、shared expert 的本地复制和融合。每 token 发往一个 rank 的 payload 可能被同 rank 多个 experts 复用，因此不能把 8T 直接当作通信包数量。全局平均负载也不能替代最慢 rank；在每条逻辑 assignment 恰好落一个 EP owner 的额外假设下，最慢 rank 至少承担 `ceil(8T/EP)` 条，但真实执行量仍取决于这些映射与 kernel。

## 原生最小实验与否定条件

[native-next-contract.json](native-next-contract.json) 保持 Ascend、PD 与 MTP，输入仍含 2/3 请求和 512/32768 上下文。首先取得本任务资源，确认实际 backend、target/draft quant method 与 scheme、每阶段有效/填充行数、shared fusion、rank mapping；再核验路由索引/权重、permutation/group counts/finalize、状态与 buffer 生命周期、PD 结果和接受分布。

只有这些正确性条件成立，才按相同输入/配置/接受分布采集分阶段 trace、通信字节、每 rank GMM rows 与校准硬件吞吐。已有静态处理无需重复实现；若发现 serialized caller 已满足单槽生命周期，不因人工交错而增加 guard 或复制。不得通过减 k、丢 expert work、关 MTP 或修改量化精度来宣称等价优化。

## 证据与状态

七个冻结命令均在各自 ZCode 精确会话执行一次；[zcode-audit.json](zcode-audit.json) 区分 deepseek-flash 配置别名和 UNKNOWN 后端版本。R005 的 outer/inner 0 只表示不可用结果已记录。R004/R007 两个任务 CPU 容器均无设备/网络、只读 root、1 CPU/4 GiB，只挂载四个 metadata/tokenizer 文件；核对身份后清理，观测服务 ID、启动时间和 restart_count 前后一致。没有模型生成请求。

[source-index.json](source-index.json) 收录 MoE 子集的 36 个规范源码文件；**不是 36 个课程机制均已完成**。重放 CPU 控制路径需要有权访问同一镜像与制品，公开包不含鉴权、真实权重、完整量化描述或模型 I/O。Loop 结论为 INCONCLUSIVE；完整 Ascend 模块、原生 correctness/PD/performance 和硬件物理前沿仍待验证。下一步转向 PD prefill 的准入与 chunk 预算，完善真实 T 与资源账。
