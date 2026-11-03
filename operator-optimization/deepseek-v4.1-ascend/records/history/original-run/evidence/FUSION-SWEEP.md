# 基线源码第一轮融合排查

本轮是融合机会清单，没有实现新融合，也没有证明任何候选能加速。范围为当前 169 镜像中 DeepSeek V4.1 文本 target、DSpark、Engram、DSA CP 和 MoE 调用路径。不是对所有可能算法的穷尽证明。实际分支、shape 和耗时以 trace 为准；A5 packed-cache、视觉路径不计入当前 A3 活跃路径。

源文件均保存在 `sources/`，`STATIC-CALL-CATALOG.json` 是语法调用目录，包含未激活分支，不是运行 DAG。下表代码位置相对 `sources/vllm-workspace/vllm-ascend/vllm_ascend/`。

状态：**已有**＝源码已实现，需 trace 确认分支；**候选**＝值得进一步检验的边界；**条件**＝存在明确依赖、精度、布局或容量条件；**保留**＝不能按简单相邻算子合并处理。没有候选已获性能验证。

| ID | 两两 / 三个 / 拆分重组的边界 | 状态与可能省掉的工作 | 约束、消费者及核对证据 |
|---|---|---|---|
| F01 | mHC pre 内部统计、投影、Sinkhorn、混合 | 已有 `npu_hc_pre_v3` | `models/deepseek_v41/model.py:858`；不能当成大量独立 PyTorch 小算子重新计数，Sinkhorn 迭代依赖保留。 |
| F02 | mHC pre → input RMSNorm | 候选：省一次中间 BF16 张量落地/读取及一次 launch | `model.py:889`；pre 还有 post/comb/pre_mix 输出，不能删除；RMS 的跨 hidden 归约与 BF16 舍入边界须保持。 |
| F03 | attention / MoE 的 mHC post → 下一次 mHC pre → norm | 条件：组合 epilogue/prologue，减少 HC `[T,4,5120]` 中间流量 | `model.py:871,902,905,919`；下一 pre 依赖完整上一 post，且残差要保留；跨层 Engram 和 DSpark aux hidden 消费会阻止单消费者假设。不能用矩阵直接相乘穿过非线性/归约。 |
| F04 | post-attention RMSNorm → FP32 routing 输入 | 已有 `rms_norm_cast`，有回退分支 | `model.py:844`、`device/device_op.py:65`；输出必须同时含 BF16 主干和精确既有 FP32 router 输入，不能用随意重新计算的近似替代。 |
| F05 | RMSNorm → dynamic INT8 quant → Wq_b | 条件：norm+quant 双输出；或 tile 内进入 GEMM | `attention/dsa_v41.py:320,367,478`；qr 也给 indexer，不能只留下 INT8；整行 scale/reduction 和舍入必须等价；多 N tile 可能重复归约，UB/L1 容量与 reuse 要核实。 |
| F06 | Wq_a 与 Wkv 共同输入投影 | 条件：拼接权重或将共同的输入读取/量化提取一次 | `attention/dsa_v41.py:440` 已有 `share_quant` 与 Cube/Vector 交错；CP 下 Q 仅本地 token、KV 全局 token，不能直接假定两者 M 相等。增加大 GEMM 可能损失现有重叠。 |
| F07 | Wq_b epilogue → Q RoPE | 候选：直接写旋转后的 Q | `attention/dsa_v41.py:367,371`；只旋转尾部 head 维，保留 dtype/舍入；输出 64×512，不能因 TP8 就误用 N/8，当前 CP 是 replicated Wq_b。 |
| F08 | Wkv → KV RMSNorm → RoPE → SWA scatter | 条件：优先 norm+RoPE+scatter 三者，投影可分块接入 | `attention/dsa_v41.py:327,297`；cache 是 paged/有页步长，保留无效 slot 行为；当前异步 KV 与 Q 路重叠不可无证据取消。归约需整行，之后 scatter 依赖完整结果。 |
| F09 | Indexer query projection → RoPE → quant | 候选：向后融合 epilogue，直接生成 query INT8 与 scale | `indexer.py:165,168,174`；query 有多 head，scale 形式、量化舍入及 QLI 输入契约必须保持；现有 query quant 在辅助流异步，比较省流量与丢重叠两项。 |
| F10 | Indexer weights projection → FP32 cast → scale → FP16 cast | 候选：epilogue 合并 cast/scale | `indexer.py:171`、`indexer.py:254`；必须保留先 FP32 缩放再转 FP16 的数值效果，不可直接用 FP16 乘法；weights projection 已和 query RoPE 重叠。 |
| F11 | Indexer key norm → RoPE → dynamic quant → key/scale scatter | 候选：归约后同时写 INT8 cache 和 FP16 scale cache | `indexer.py:110`；key 是 indexer cache 的唯一当步生产者，但被后续层复用；不能取消持久写；物理 slot、page stride 和 scale FP16 舍入是接口条件。 |
| F12 | QLI output → index fixup → stable buffer copy / padding | 候选：QLI epilogue 直接写最终索引，或 fixup 同时完成 pad/copy | `indexer.py:274`、`attention/dsa_v41.py:605,245`；后续共享消费者和 ACL Graph 要稳定地址；候选是 block ID，不是 token position；位置过滤、-1 填充、长度输出必须保留。 |
| F13 | sparse attention 的 QK / mask / softmax / PV | 已有 `npu_sparse_flash_mla` | `attention/dsa_v41.py:657`；禁止按物化全 attention matrix 估算强制 HBM 流量。选中 KV、窗口、compressed residual、sink 和 mask 是精度契约。 |
| F14 | sparse attention output → inverse RoPE → wo_a | 条件：attention epilogue/inverse RoPE 与输出投影 prologue 接合 | `attention/dsa_v41.py:714`；当前 CP 中间还有 head restoration 通信，不能把跨 rank 依赖抹掉；attention tiling 与投影 K 分块是否相容未知。 |
| F15 | CP padding → head exchange → wo_a grouped projection | 条件：tile 收到即投影，padding 在打包/消费端处理 | `attention/context_parallel/dsa_v41_cp.py:209`；通信完成只需局部 tile 还是全量是待证接口问题；8 个 group，不是普通大 dense；不能仅相加通信耗时与 GEMM 耗时。 |
| F16 | wo_a → wo_b → output reduction | 条件：重排 tile 流水；保留 BF16 两级投影数值语义 | grouped 投影及行并行后续调用需结合实际 kernel trace；两矩阵合并成预乘权重通常改变舍入和工作量，不能默认等价。reduce-scatter 的本 rank partial sum 必须先完成。 |
| F17 | Compressor input cast → wkv 与 wgate 双投影 | 条件：共享 cast/read、拼接投影或分块供给 pooling | `attention/dsa_v41.py:530`；ratio2 路用 FP32，不能为了融合改 BF16；两个权重不同但输入相同；减少重复读取与增加临时存储要一起算。 |
| F18 | Compressor pooling / norm / ring-state update | 已有 `compressor_from_projected`；内部进一步融合需看 kernel | `models/deepseek_v41/compressor.py`、`ops/triton/compressor/compressor_triton.py`；分块/跨 chunk ring-state 依赖不可删除，当前时间步对状态的写应先于其消费者。 |
| F19 | Compressed latent → indexer key branch 与 attention KV RoPE/scatter | 条件：一份 latent 双 epilogue，分别落对应 cache | `attention/dsa_v41.py:546`；两个消费者，indexer 先线性+norm，不能共用已经 attention-RoPE 改写的 latent；cross-layer cache sharing 禁止消掉最终持久写。 |
| F20 | Router FP32 GEMM → sqrtsoftplus / topk / renorm / route pack | 条件：路由 epilogue 与 packing 接合 | `model.py:283,293`、MoE factory；FP32 logits、top6、相同排序/并列规则、scaling1.5 必须保持。active implementation 要从 trace 定，不把内部已经融合的东西重算。 |
| F21 | Dispatch → expert GMM1 → activation/quant → GMM2 → combine | 已有当前 fused-MC2 自定义实现；内部 tile 流水是后续深入点 | `ops/fused_moe/moe_comm_method.py` FusedMC2 分支；`enable_fused_mc2=1` 不能自动解释成 CANN MegaMoE。只有数据就绪的 expert tile 可启动；统计/路由/远端数据到达与通信资源争用须保留。 |
| F22 | Shared expert gate/up → SwiGLU+clamp → down | gate/up 已合并；activation 的现有融合需核对实际 kernel | `model.py:105,144,199`；clamp10 是模型语义；可考虑 GEMM epilogue 或 tile pipeline，不能丢激活量化舍入。 |
| F23 | Shared expert output + routed output scaling/add | 已有 `muls_add_triton` | `model.py:302`；候选是与下游 HC post 接合，不能把已有一次融合算作两个新机会；join 只等待真实消费者，不消除数据依赖。 |
| F24 | Engram hash / host lookup / row gather / cast | 条件：去重、按页聚合、分块生产与消费，不是一个 NPU kernel 的简单 fusion | `engram/embedding.py`、`model.py:1498,1573`；已提前启动 lookup、延迟 wait；CPU mapped memory、NUMA/共享表、lookback hash 以及跨流 event 生命周期真实存在。未测前不能称它是低利用率瓶颈。 |
| F25 | Engram wkv → gate → residual update | 已有 `fused_engram_gate` 分支；条件候选为 projection epilogue 接入 | `engram/layer.py:35`；rotation 非空时走另一条路径；q/k 权重 FP32 乘积预计算只对权重不可变且舍入一致成立，当前未实现。token mask 和 HC4 residual 必须保留。 |
| F26 | HC collapse → final RMSNorm → LM head | 候选：collapse+norm 优先；LM head 可分块接入 | `model.py:855,1605,1778`；collapse 是 FP32 multiply/reduce 后转 BF16，不能省舍入；词表分片、采样/DSpark logits 其他消费者需核实。 |
| F27 | aux hidden HC mean → concat → DSpark main_proj → main_norm | 条件：折叠 concat 或 tile 供给，norm 可向后接合 | `model.py:1561`、`dspark.py:277`；三个 target 层采样点不同，不能提前算依赖未来层的投影；保留 aux consumers、FP32/BF16契约。 |
| F28 | DSpark embedding → HC repeat / pre_mix 初始化 → first pre | 候选：避免显式 repeat 和重复初始化落地 | `dspark.py:253`；后续层可写 HC residual，零步 stride view 不能直接替代所有可写 tensor；eager draft 的 launch 开销需实测。 |
| F29 | DSpark shared KV projection → norm → RoPE → cat/contiguous → scatter | 候选：直接写最终 cache，消掉 cat/copy | `dspark.py:280,233`；每个 draft 层权重不同，不能跨三层复用相同 KV；slot 不合法行、页结构、KV basis 与位置完全保留。 |
| F30 | DSpark collapse / norm / Markov bias / confidence / sample / target verify | 条件：小算子 epilogue 与 packing；串行 draft block 本身保留 | `dspark.py:143,269,316`；DSpark3 层串行与 block_size5 模型结构不能改成 prompt lookup 或任意64；接受/拒绝和 target logits 的依赖不可跳过。 |
| F31 | Device metadata build → format conversion → indexer/attention consume | 部分已有异步 task 与延迟 wait；候选减少同内容重建、稳定缓冲直接生产 | `attention/dsa_v41.py:844,874,1171,1254`；不同 query/cache 域、不同阶段 group-id 不能随意复用；同步可以后移到首个消费者，不能删除必需的 happens-before。 |

## 边界检查与保留项

1. `view/unsqueeze/squeeze/unflatten` 通常只是元数据；没有 trace 或 stride 证据，不计成可省一次 HBM 搬运。`contiguous/to/copy_` 也可能 no-op 或别名；先查真实 task。
2. 两两和三个相邻节点之外，F06/F17 是共同输入重组，F19 是双消费者分流，F15/F21/F24 是分块生产/消费。只有依赖满足的 tile 可以重叠；资源争用使“无依赖”不等于“免费重叠”。
3. 共享 KV/index 不能按普通单层临时张量删除。Source 层写出的数据要满足所有后续消费者；分块融合应携带 producer/consumer、读写区间和 event。
4. 连续层、autoregressive token、DSpark 串行块的数学依赖不能通过融合抹掉。动态量化必须先取得约定范围内 scale，再计算依赖它的 INT8 乘法；归约允许流水但不能凭空拿到未读数据。
5. 尚未完成：自定义 kernel 内部的完整 instruction/tile DAG、每个候选的容量与数值等价证明、微秒收益或最大性能缺口排名。当前不能宣称“所有能融合的已经融合”。

## 下一步如何接利用率

先按 target attention/projection、indexer、compressor、MoE routed/shared、mHC/norm、Engram、DSpark、metadata/collective 分组。Prefill 与 decode 分开，逐 rank 计算设备任务的时间区间并集、Cube/Vector/MTE 指标和依赖等待；多个流不能直接加总当作端到端时间。MC2 复合 kernel 先单列，不武断拆成纯计算或纯通信。

“最低利用率”要按模块应使用的资源解释：归一化 Cube 为0正常；通信等待不能当成无用 GEMM。排名同时呈现样本覆盖、调用次数、每次时间、占用的时间并集、流水指标、暴露在关键路径上的证据及未知项。只采样到的阶段才能排名，不能把前9个 worker step 当成全程稳态或全部模块。

融合候选先登记，当前仅诊断、不改代码。优先级应结合实际端到端暴露成本和相对理想 model 的差距，而不只挑一个很小但利用率最低的算子。
