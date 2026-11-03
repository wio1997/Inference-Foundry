# 草稿 attention：本地空 query 与跨 rank head 交换

## 对应最低组的范围，而不是过度归因

`evidence/draft-module-diagnosis.json` 和 `agents/diagnosis/module-priority-evidence.md:153–172`：15 个 rank、162 个严格归属窗口合并的三个草稿 decoder block，主流正 AIC 周期加权 `aic_mac_ratio` 为 **2.0151%**，在该限定口径的 41 组中最低。它不是整个 NPU 利用率、全框架最低单个模块、空闲比例或瓶颈原因；草稿 288 个候选中有 126 个因未在 attention bracket 检出 `SparseFlashMla` 未被严格归属，EP15 完全未归属，其他流及部分缺失计数亦不包含。这里选一个潜在重要的草稿 CP attention 边界，仅说明真实代码的数据结构和约束，不能由该代理比例得出这条边界造成低比例。

## 草稿与 target 的重要差异及现有代码

`models/deepseek_v41/dspark.py:101–123` 的草稿 `DeepseekV41DSparkAttention` 创建 V4.1 backend，但角色 `compress_ratio=0`、`is_kv_source=False`、没有 long/index source；因此该 draft attention 的普通路径是 SWA，不能把 target F12/F13 的 QLI / compressed cache 推到草稿上。它按 `get_v41_cp_classes()` 选择 CP 或其他实现；实际历史实例未被直接识别。

在**legacy CP 实例**条件下：

- `attention/context_parallel/dsa_v41_cp.py:105–155,188–192` 给一个 rank 的 query 建局部 token 范围和该 rank 的 valid 行，同时另建全局 cache metadata。源码明确说 padding 参与输出交换而不参与 cache 读取；local query 和 global cache 输入域可以不同。
- `attention/dsa_v41.py:338–355,701–716`：query 数为零时仍可调用 `_update_caches`，然后构造 `[0,n_heads,head_dim]` 的空 attention output；`_project_output` 仍被调用。不能将“无本地 attention query”直接改成“无缓存更新/无后续计算”。
- `dsa_v41_cp.py:209–225` 将本 rank 未满 `per_rank` 的 attention 行补零并调用 `restore_tp_heads`，再按 hidden 行做必要 padding、进入输出投影；`dsa_cp.py:77–91` 的 head all-to-all 与接收映射已由 `F15-CP-PROJECTION-REVIEW.md` 复核。`_forward_o_proj` 的 `npu_transpose_batchmatmul` 已经把转置和第一投影合并；不要把当前通信或该 composite 当成可直接删除的重复计算。

## 一张卡没有 query，仍需收别的卡结果：可手算反例

取简化的 TP/CP 2 rank，每 rank 通信行容量 L=1、全局 head=2、head 宽度=1；一个实际 token 排在 rank0，rank1 的本地 query 数为0。rank0 attention 输出 `[a,b]`，rank1 对空 query 按原路径先补 `[0,0]`。F15 的等分交换式给出：

- rank0 接收按 sender 顺序的第 0 个 head：`[a,0]`；
- rank1 接收按 sender 顺序的第 1 个 head：`[b,0]`。

故 rank1 尽管本地没有生成 query，仍需参与 collective 才能接收 rank0 有效 token 的第 1 个 head，之后按它持有的投影权重处理该 token。如果对应简化两级投影权重均为1且后续按 TP 求和，这个 token 的输出依赖 `a+b`，跳过 rank1 会丢掉 `b`；单边跳过 collective 还可能使其他 rank 永久等待或通信协议不匹配。这里是可复算的源码映射反例，**不证明历史 EP7 那条窗口真的为空 query、也不表示实际数值为 a/b 或发生死锁**。

即使所有 rank 的本地 query 同时为空，也不能只凭本 rank metadata 独立跳过：需证明全组一致、后续输出/缓存/其他模块无需该步，以及所有 rank 的 collective 顺序和图重放约束仍一致。尤其 `_update_caches` 可能仍消耗 global hidden 并写 SWA cache。

## 可研究的重组，不宣称已执行或有收益

- **协议层去除无效行**：在所有 rank 对实际有效 token/head 分布、接收映射和版本一致达成可用元数据的条件下，研究变长分块交换或仅发送有效行，并在接收侧恢复固定布局/零 padding；比单边跳过安全，但涉及确定各目的 rank 的长度和 offset、处理相同数值位置的真实/填充标记、all-to-all split 的跨 rank 一致、空 rank 参与、graph 的固定地址与其他流的 completion event。当前 `restore_tp_heads` 是按固定大小通信，源码未证明可使用逐块 ready 或可变长协议，也未提供减少通信成本的证据。
- **Q 与全局 KV 分离**：允许各 rank 根据真实有效 query 行处理 Q、但按全局 cache metadata 处理必要的 Wkv/norm/RoPE/写；尝试提前处理分块必须保持本层 cache、位置、有效 slot 和未来请求的可见性，以及 SWA 写完成后 SMLA 的依赖。不能按本地 0 query 裁掉 global KV。该条件与 F08 的目标 KV 一般分析有关；本处特殊价值是草稿的 CP 两个行域和零 query 后续交换，收益不得重复计入 F08/F15。
- **数值和资源**：补零的 dtype、值及其图可观察性需保持，真正投影、相应 rank 的输出、TP 归约及顺序不能删；任何以变长重排减少 padding 的方案都必须验证 rank/head/token 映射、每层权重、BF16/量化舍入及空值。即便通信数据量减少，额外元数据、启动/同步、计算密度与链路竞争仍可能抵消收益；现有材料没有实测。

## 已存窗口不能证明的事

`agents/diagnosis/DRAFT-EP7-QUERY-METADATA.md:1–29` 的一个未严格归属 EP7 样本，其 bracket 内未列出 SMLA，记录 5 行 KV-like 操作、`ZerosLike [1,64,512]`；另一个严格样本记录 1 行 Q-like 和 15 行 KV-like。事件名、shape 与空/少 query 条件相容，但 sample 的 backend 实例、本地 query 实际数量、tensor 地址、请求及分支没有绑定，**不能将未导出 SMLA 证明为零 query**。同文 `:135–140` 的 EP0/EP1 六个有 SMLA 的窗口列出各自 all-to-all 和部分 wait，不能混称同一个无 SMLA 窗口或作为上述 P=2 反例的运行证明。通信名称/形状与 `restore_tp_heads` 仅为候选联系，wait 原因也未绑定。

无需重复 profiler 扫描；后续只有找到同一调用的 backend 身份、query/cache metadata 的实际数值、rank 间 collective 映射/事件和 cache 地址，才能把静态候选收紧到历史执行。未修改推理代码、没有启动 NPU。
