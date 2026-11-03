# F20：A3 原生 hash 路由的依赖与边界

## 结论及条件

已读取两份当前同镜像容器中的 A3 原生自定义算子源码。对于 **without-group、norm_type=2、存在 token→expert 表且该行不是 image sentinel** 的源码分支：原生算子已直接走哈希表选择，不执行 `SelectTopKExpertIdx()`；因此先前 Python fallback 中“计算动态 TopK 后被覆盖”的结构不能用于这条原生路径。`ComputeX` 仍先对整行 expert logits 算 `exp → +1 → ln → sqrt`，并建立加 bias 的整行值，哈希分支随后只用表中 ID 从**未加 bias 的 score**中 gather，归一化及缩放。

因此一个更准确的条件候选是：若这条分支确实命中、全量 score 不可观察，则可研究把表索引提前，只为选中项产生 score，并避开该行仅供动态排序使用的 bias 结果。上游 gate `F.linear` 也可研究只生成选中列，但这涉及完整 logits 的其他消费者、重排与数值契约；当前不能证明历史 workload 命中表分支，不能断言其收益或位级等价。视觉行和无表行仍需动态选择，不能套用。

## 证据和来源边界

- stage 与基线两个同镜像容器的 `/vllm-workspace/vllm-ascend/csrc/moe/moe_gating_top_k_hash/op_kernel/moe_gating_top_k_hash_without_group.h` 当前 SHA-256 均为 `f18d63bda201eb407a3861da0f5189d45d439b6630abf52058a6eaa0420d8fdb`。stage 中 build 二进制源拷贝和打包源码拷贝同 SHA。对应的源码快照在 [without-group 源码](frozen-source/csrc/moe/moe_gating_top_k_hash/op_kernel/moe_gating_top_k_hash_without_group.h)。这证明现存源码拷贝一致，不证明历史进程加载了哪个二进制、其构建于这些源码或历史每次调用的 tiling key。
- [host tiling](frozen-source/csrc/moe/moe_gating_top_k_hash/op_host/moe_gating_top_k_hash_tiling.cpp) `:239–246` 要求 softplus 的 group_count 为 1；`:617–646` 在非 950 的普通 tiling 中，group_count=1 指向 without-group 的 key 家族。`:654–671` 仅 ASCEND950 走 arch35 tiling。对应 [kernel 入口](frozen-source/csrc/moe/moe_gating_top_k_hash/op_kernel/moe_gating_top_k_hash.cpp) `:64–93` 关联 without-group 模板。该链是当前源码条件推导，历史硬件识别、key、kernel/binary 的绑定未采集，不能用 trace 名称替代。
- [Python router](frozen-source/vllm_ascend/ops/fused_moe/router/fused_topk_router.py) `:156–176,204–222` 在 sqrtsoftplus 路径设置 norm_type=2、out_flag=False、eps=1e-20，表可为 None，input IDs 还可能经过 gather、padding 或 SP chunk 和 -1 替换。A3 capability 及保存模型字段的条件排除、历史未绑定限制详见 [视觉 fallback 复核](F20-VISION-FALLBACK-MASKED-TEXT-ROUTER.md)。
- `without_group.h:129–216,218–229,272–310,326–347,402–420` 显示每行先完整加载与 score，再选择 hash 或 topk，最后从未加 bias 的 score gather。`outFlag_` 为 true 时 `:410–412` 还会输出完整 score，故不能无条件删除非选中项；该 Python 调用静态传 false，不是历史对象快照。图像判断依赖 bias_vl 和 sentinel 的精确范围，普通 token 与无表时各自的分支都需保留。
- gate 上游使用条件性 `F.linear`，下游还有 EPLB、共享专家、ID 重映射与 dispatch，详见 [router/dispatch 审计](F20-ROUTER-AND-DISPATCH-REVIEW.md)。这个报告未证明历史图的完整 logits 所有消费者、张量身份、缓存或任务依赖，亦未证明将分块结果送 MC2 的接口与通信顺序可行。

## 不能直接将 fallback 的数值公式套到原生算子

原生 `without_group.h:285–302` 对 sigmoid/softplus 从选中项取 `sum + eps` 后除法并缩放；传入的 `renorm=0` 并不表示此分支不对选中权重归一化。Python fallback `fused_topk_router.py:70–76` 则是 `renormalize` 为真时以 `max(sum, finfo.tiny)` 作分母。两种公式不恒等：在仅作 score 层代数示例、单个选中 score 为 `1e-20` 且 eps 为 `1e-20` 时，前者分母 `2e-20`、后者约 `1e-20`。这是公式反例，未证此输入能在硬件上实际产生，未测真实浮点舍入、FTZ 或模型输出，不能据此判定已发生精度故障。源码中的 `Exp/ Ln / Sqrt` 次序也不能凭数学实数等价替代 `torch` 稳定 softplus 的具体数值实现。

新结构必须按实际原生版本保留选中 ID、score 舍入、sum 的归约顺序、`+eps`、缩放、dtype、tie/NaN、无效索引行为、mixed vision 行、SP/EP 排列及下游 consumers。仅基于本源码做了依赖推断，未做 NPU 验证、未修改推理实现，也没有确定框架性能、资源争用或 TPS 上界。

## 第三输出的 schema 与实际写回不能混同（2026-10-30）

同一 EP9 的已选设备记录经 host scope 与 trace flow 关联到 `MoeGatingTopKHash` task 46754；其导出输出形状/类型列有 `[643,6] FLOAT`、`[643,6] INT32`、`[643,384] FLOAT`。下一条 MC2 记录有与前两项形状相容的输入，但尚无 tensor 地址、实际值和 ABI 槽位的历史绑定；详见 [EP9 记录](../diagnosis/EP9-GATING-AND-MODEL-METADATA.md)。这是一条带 profiler 的有界样本，不代表每个请求或每层。

当前保存的三段源码给出一个**条件性的解释边界**：

- 自定义 op schema 的 `op_host/moe_gating_top_k_hash_def.cpp:125–145` 无条件把第三输出 `out` 声明为 `REQUIRED`、FLOAT；`out_flag` 的默认值为 false。schema 声明要求该输出接口存在，并不等于它一定填有有效 score 或一定会被上层消费。
- 同一源码快照的 without-group kernel 的 `Process` 在 `outFlag_` 真时才调用 `CopuOutXNorm`；该 helper 将本行内部 score 写至 `outGm_`。`outFlag_` 假时该具体 helper 调用被跳过，但仍先执行 `ComputeX`，不能因此说这条分支没有计算整行 score。这里描述所查源码控制流，非所有 kernel、compiler lowering 或实际内存访问的证明。
- 保存的 `fused_topk_router.py:203–223` 的对应 native 调用静态传 `out_flag=False`，以 `topk_weights, topk_ids, _` 接收并只返回前两者。源码构成支持“某个调用方不消费第三返回”的条件，但**没有证明 EP9 采样调用就是该对象、flag=false 或选中该 kernel**。

所以记录中第三输出 `[643,384] FLOAT` 的存在，不能单靠 shape 断定设备实际写出了完整 score，更不能把 `643×384×4 = 987,648 B` 作为可以删除的实测写入或理论必需流量。反过来，schema 的 REQUIRED 也不能说明编译器或调用者是否实际分配、初始化、复用或消除该缓冲；若要修改接口，还必须保留其他已注册调用者与 graph/ABI 的合同，不能因 Python 示例丢弃第三项便直接删除。真实 `out_flag`、历史 binary/tiling、输出 storage 和消费者是将 F20 候选绑定到该样本的必要证据；本条不是新增可独立计收益的融合。

只读证据：schema 快照 SHA-256 `33b7a51ed38093f0dbda3544824efed04c04c33b0bc4b25dfaa5044f6e9a50f3`；without-group kernel SHA-256 `f18d63bda201eb407a3861da0f5189d45d439b6630abf52058a6eaa0420d8fdb`；router SHA-256 `06c47fd91e8c36f6f5465cea0b32a184280f842833c2bff27a18fc6d17fcb838`；EP9 设备元数据摘要 `agents/diagnosis/ep9-gating-metadata-binding.json` SHA-256 `aa263709868278cf54dc8eb9734c1598ed6650b25e91a8ccf67cd5b9d5ba6a82`。未运行 NPU、未更改实现。

本机 ZCode 对这三个源码位置独立复核，与本机检查一致（`zcode/f20-outflag-schema.zcode.json`）。补充：当前 without-group `Init` 仍为 `out` 设置 global buffer，并初始化对应队列；跳过 `CopuOutXNorm` 不能推断所有资源准备或输出分配均消失。该核对只验证源码文字，不验证历史 kernel 或物理流量。
