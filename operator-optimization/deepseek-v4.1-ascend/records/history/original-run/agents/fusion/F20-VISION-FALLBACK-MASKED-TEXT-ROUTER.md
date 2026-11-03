# F20：视觉路由回退中，文本行的被覆盖 TopK 与选择性 gate

## 当前基线适用性复核：回退候选不应作为已命中机会

**条件排除：**现存镜像和冻结源码组合下，正常加载 A3 profile 时，`MOE_GATING_TOP_K_HASH_VISION` 为支持，`fused_topk_router.py:180–197` 的 `not fused_vision_hash` 不成立；所以即使其他条件满足，当前源码中的 Python 视觉回退不会被选中，而会走 `:204` 的已有复合 `moe_gating_top_k_hash`。这排除了**这个候选所依赖的源码路径**，不是证明该复合算子内部没有类似机会或推理运行期间绝对不存在别的 Python 版本。之前以下内容是回退函数的条件数学分析，不代表当前 169 基线命中或实际节省 64 倍。

依据：

- 保存的 baseline/stage 镜像包 `_build_info.py` 来源核验见 [INDEXER-RUNTIME-REVIEW.md](INDEXER-RUNTIME-REVIEW.md) 与 `indexer-build-info-provenance.json`：两容器同镜像、目标文件未被挂载覆盖、文件 SHA-256 `65edd0c22127fac498174576a99758e422052ed0180ed332570c19cddab0197a`，文件声明 `__device_type__='A3'`。`device_config.py:26–47` 首选此构建类型；这不是历史 Python 进程的实际已加载模块快照。
- 冻结 `hardware_profile.py:249–278,298–308` 将 capability 放入 `_STANDARD_CAPABILITIES`，而 A3 的 `_A3_CAPABILITIES` 包含整个标准集合；`supports` 使用集合成员判定（`:243–246`）。冻结硬件 profile SHA-256 `efc6d0f3856be400c62b20c6a39229ccc1a2771ec4351a03328ec535824691f5`，router SHA-256 `06c47fd91e8c36f6f5465cea0b32a184280f842833c2bff27a18fc6d17fcb838`。此结论依赖实际历史实例加载相应源码、构建信息和未被补丁替换的 profile；这些历史绑定尚未直接采集。
- 冻结 `evidence/model-config.json` SHA-256 `43d3b2bc55626893d59275b32e04068c0fc916f5c382eb91de73bdbade9c88f7` 的 `vision_config.num_hidden_layers=32`、`text_config.scoring_func=sqrtsoftplus`。同镜像保存的 `DeepseekV41Config` 构造代码将前者映射为 `vision_n_layers`，文本字段平铺；如果按此代码及未经覆盖的输入正常构造，`model.py:210–219` 将创建 `bias_vl`。这仍非历史对象、输入 IDs 或实际路由调用的直接观测。
- 原始 root 和 text 配置均无 `num_hash_layers`；现存源码构造与归一化的完整条件链已见 [module-priority-evidence.md](../diagnosis/module-priority-evidence.md) stage162：如无运行期注入，`normalize_deepseek_v41_config` 补 0，目标层不创建 `tid2eid`。这是独立于 capability 的第二个限制；没有历史对象快照，不能断言历史值为 0，不能用另一配置类的默认 3 顶替。
- [DRAFT-EP7-QUERY-METADATA.md](../diagnosis/DRAFT-EP7-QUERY-METADATA.md) stage196 对三个**草稿侧**窗口命中 `_C_ascend::moe_gating_top_k_hash` 的名称散列；它支持这些有限窗口出现该符号，但不证明目标模型相应层、参数、历史 Python 路径、历史二进制或有无 tid2eid，亦不能推断其他窗口。

因此本次基线中，下文候选降级为“**不同 capability/运行时条件下的参考结构；当前静态镜像路径不适用，历史实例未绑定**”。下一步应把研究转向已有复合算子的语义和上游 gate/下游 dispatch 的依赖，不能将本 Python 回退直接计作 169 的可实施融合或量化收益。没有读取权重 payload、prompt，未启动设备工作。


## 结论和证据范围

对已保存的 `select_deepseek_v4_vision_experts` Python **回退函数**，可以从源码证明一个受条件限制的结构重组：当输入行不是 image sentinel 且 `tid2eid` 表存在时，该行最终专家 ID 取自表，之前算出的动态 TopK 和该行 bias 只用于被覆盖的 `dynamic_ids`；最终权重从**未加 bias 的**对应专家 score 收集，再按原开关归一化、缩放。于是可以研究按该表先选出专家，再只计算这些专家的 gate 和 score，同时完整保留其他消费者和逐运算精度。此处是 F20 既有“table 分支”猜想在**具体 Python 回退路径**的细化，不是新的一项可独立叠加的收益，也不是证明历史调用发生或自定义 hash 算子有同样语义。

冻结源 `agents/fusion/frozen-source/vllm_ascend/ops/fused_moe/router/fused_topk_router.py:32–74, 146–203` SHA-256 `06c47fd91e8c36f6f5465cea0b32a184280f842833c2bff27a18fc6d17fcb838`，与 `f20-router-provenance.json` 中两个同镜像容器的已存源码哈希一致；文件匹配不是历史运行时 Python 对象、加载内容或图的绑定。`model.py:176–179, 210–234, 282–298`（SHA-256 `101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91`）记录外部 router 分支可先执行 `F.linear`，以及仅在 `layer_idx < num_hash_layers` 且非 draft 时构造表。真实层、字段和 runtime 分支未绑定。

具体走该回退还要求实际 `scoring_func == sqrtsoftplus`、`bias_vl is not None`、准备好的 input_ids 非空，以及硬件 capability `MOE_GATING_TOP_K_HASH_VISION` **不支持**。若该 capability 支持，源码改调用现有的复合 `moe_gating_top_k_hash`，它的内部数学、是否已有类似消除和与此回退的等价关系均未知。`bias_vl is None` 或没有 hash 表、draft layer 等其他分支不可套用。模型快照和现存镜像源码可以条件推导 `vision_n_layers` 和 A3 capability（见上方复核），但没有绑定历史运行时对象或实际请求；不能从 kernel 名称判断这些条件。

## 源码级推理

设同一行的 router score `s[e] = sqrt(softplus(logit[e]))`；文本行掩码 `image_mask=false` 且存在表，`I = tid2eid[aligned_input_id]`。函数先对 `s + row_bias` 做完整 `topk` 得 `D`，然后 `torch.where(image_mask, D, I)` 选择 `I`，再取 `s.gather(I)`、可选 `sum/clamp_min/divide`、缩放。对正常完成且未观察中间张量的该行，`D` 和 `row_bias` 不影响返回的 `(weights, IDs)`；其他行并不受这行筛选条件的影响。

这只是该函数的输出依赖证明，还要满足：表项合法、输入对齐和负一替换按原调用完成，`topk`、广播和索引原本成功且没有异常、被观察的副作用或必要的调试/全局消费者；选定 experts 的 logits 和 score 在新计算结构下必须与原指定精度语义匹配，输出 ID dtype、顺序、重复项、`tiny`、归一化、缩放及后续 EPLB/dispatch 规则不变。不能假设按选中列重算和完整 `F.linear` 一定位级一致：kernel 的 shape 改变、FP32 归约顺序、算子 lowering 及 dtype 均可能不同。若 full logits 有额外观察者，必须保留或证明其可重建；本分析没有闭合全部其他消费者。

**图像行仍需动态 TopK。** 对只观察了前两个专家的 score `[5,4]`，其余分别为 `[1,0]` 与 `[10,0]` 时，Top2 从 `(0,1)` 变成 `(2,0)`；因未观察专家可改变结果，不能对一般图像行盲目只算固定两个专家。本例是独立 score 输入域的反例，不断言对应某个具体 gate 权重/历史 token；有额外可证界、预计算或其他算法时不排除剪枝。

混合 batch 的候选结构为：先形成与原路径一致的 aligned input ID 和图像掩码；文本行表驱动选列，图像行保留全专家动态选择；两类输出分别按原顺序写回统一的 row 域并保留全部中间可观察值，最后再进入既有逻辑。历史调用的 token 组成、graph buffer、stream/event、数值契约、shape 及消费关系未知，无法声称该拆分已可实施或一定节省。

## 条件工作量会计，不是速度预测

保存的配置有 hidden 5120、专家 384、每行取 6；若权重为稠密矩阵，每专家一个完整独立点积、无 padding、不复用中间结果，则原每文本行 gate 的常规逻辑工作是 `5120×384 = 1,966,080` MAC，表驱动的 6 个列点积是 `5120×6 = 30,720` MAC，前者是后者的 **64 倍**。这些是该受限结构的逻辑工作量比较，不是物理指令数、硬件工作下界、全模型节省或 TPS 倍数。表行重复、按 batch 重排和矩阵大块效率、索引权重读、归约和准备开销、混合图像行、cache 与通信会改变实际成本；bit 精度和所有消费者约束可能阻止这种重组。

`python3 agents/fusion/f20_vision_hash_mask_witness.py` 在 CPU 上对四专家、二选二、有理 score 和 bias 的 **209,952** 个例子验证选择覆盖及归一化/缩放的算术一致，同时给出图像行反例。它不运行 PyTorch 真实实现，不验证 NPU 位级精度、运行分支或性能；源码的逐行依赖是上述结论依据，有限枚举仅为辅助复算。需要继续的证据是历史 `num_hash_layers`、`vision_n_layers`、capability、router 对象、文本/图像掩码及 logits 消费者和实际算子的二进制对应；当前只读旧数据，未知保持未知。
