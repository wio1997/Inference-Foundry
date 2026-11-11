# 末层 FFN 合并逐位边界与修复判断

## 介绍

沿冻结 910C/CANN 9.2.0、TP8/DP2/EP16/PP1 的第 39 层，核对选行后 DFC、shared Down、BF16 add、HCPost 和 head 输入的完整交付。每个替换原语逐位一致。用旧 C8 实际模型影子和单卡同宽 Down 对照定位失败，再把必须保留的原计算和交付从收益预算中扣掉。

## 到了哪里

| 交付 | 实际路径、条件与结果 |
| --- | --- |
| SP 路由到 routed finalize | `AscendRoutedExperts` 先以原 TP 本地行数切 mask，再按 DP 公共尺寸补 false，随后选择 TopK、执行 W8A8 `FusedMC2CommImpl`。启用版本 1 的 `dispatch_ffn_combine` 时，设备会把 inactive 的 expert ID 原地写成 384，再同步进入路由。`M×TopK≤4096` 走 CombineV2，K6 时分界是 M682/683。SP `finalize` 只取原行数的前缀视图。6 个原 Python 前缀控制验证行身份，一个非 SP 控制验证回到父类。合法非 SP finalize 可能归集后裁片，走原分支。 |
| routed、shared 到最终 Tensor | 冻结 `SEQUENCE_PARALLEL_SEDP` 使用原本地 shard 和复制的 shared 权重。原 shared 路径是 DynamicQuant→GateUp→DequantSwigluQuant→Down；当前 DFC 配置无 shared 辅流重叠。当前两个 routed transform 为 None，runner 缩放为 1，1.5 已进 router，冻结 V1 无动态 EPLB 热度读者和 zero expert。无额外 TP 归约。`torch._foreach_add` 合成 BF16，再交 A3 HCPost DSplit 按原 FP32 Mul/Add 顺序和 BF16 `CAST_RINT` 输出。内部 `(shared,routed)` 与模型外层 Tensor 返回分别是不同合同，合法 tuple、zero expert、非 SEDP 与读 expert counts 的模式继续走原分支。 |
| 原 C8 第一次差异 | 按原 probe commit 重建实际打补丁的 reader 并核对 SHA，复核 4 波×16 rank、116 个 retained 本地行，其中 86 行的 DFC mask 为真。唯一失败位于第 2 波、DP1/TP7，M800→M3，原 local799→compact0。该行的 DFC BF16 routed 输出、HC 系数均相同，合并 BF16 与 HCPost 各有 1 个元素位差。用原合并结果送同一紧凑 HCPost 后，64 份记录的 retained 行均无位差；直接替换为原 HCPost 输出时，最终 normalized readout 也均无位差。缺失的是故障行的独立 shared Down 输出与 add 两个操作数，首次可能差异缩到 shared 支或原 add 的新形状/布局。原最终 readout 相同，仍按原语判旧配方失败。 |
| 路由观测的限制 | 原 reader 在 DFC 返回后才 clone `topk_ids`，compact 对照也从这份调用后拷贝取 ID。冻结设备代码只改 inactive 行，故故障有效行的所测 routed 结果仍可用于排查。全路由和副作用的准入需要分别保存两次调用前的独立输入，并按原域比较调用后的 expert IDs、counts、Unpermute 和发布代次。sender mask 全 false 的 rank 仍能收别卡专家工作。M800→M3 还跨了两个 Combine 实现。 |
| shared Down 的单原语反例 | 两次历史单卡同 seed 实验用相同 K2304、N5120、INT8 NZ29 权重、BF16 权重 scale、FP32 行 scale 和 BF16 输出。固定值与全宽，分别以 `2×M211` 与 `M422` 调用原 `npu_quant_matmul`，整数视图均查到 12 个不同 BF16 元素。两份相同实验支持缩/并 M 可改变原位这一结论。小 M 的有限用例和其中一份保留 M211 的列切片通过，实际模型 M800→M3 的第一处错误仍缺分算子字节。保留 M 的 M3 独立列切片 P50 35.30 µs，原对照 23.88–23.95 µs，通信和重组还未计费。 |

[11 份源码身份、旧 native 证据 SHA、原始位差与模型](last-layer-ffn-repair-review.json) · [此前末层 attention/FFN 分段](末层Attention选行与FFN逐位门槛.md) · [原 DFC 返回、代次和 HCPost 消费](DFC返回与残差消费者.md)。

### 原 C8 定价

保留原 dense FFN 前置、shared、BF16 add 和 HCPost。按原四波共同 EP16 退役时序，readout 指两次 gather、copy、collapse/norm，保留小 logits 投影。原 P90 TTFT 1491.855 ms，历史排队参考 1% 即 14.919 ms。各波共同 join 包络 **1.128、1.893、1.696、2.184 ms**，含 peer 等待。固定排程下，readout 全免 **13.563 ms（0.909%）**，join 全免单独 **6.901 ms（0.463%）**，两段共同全免 **20.463 ms（1.372%）**。全免还未扣紧凑布局、scatter/gather、持有区、事件和新调度，全部新增成本的极乐观上限 **1.386 ms/波**。

每波新增 0.5 ms 时，即使 readout 全免，也要把共同 EP join 真实提前 **3.356 ms，约占该段包络 48.6%**。readout 只兑现 75% 时需要 **97.8%** 的 join 包络。若 readout 仅兑现一半，join 全免且新增费用为零也只有 **13.682 ms**。这些是同负载敏感度。六普通负载缺同边界的请求曝光和原生逐位证据。此前更大的 attention→FFN→head 采用另一个时序边界，保留独立定价。

## 结果

- **已停止 · 原紧凑 FFN、假定 shared Down 换 M 仍逐位一致的通用配方、原 M3 保 M 独立列切片。** 在固定的原 EP join 不变时，旧含 shared/post 的最宽末端包络已低于 1% 排队参考。保留原 shared/post 后，独立末端只剩 0.909% 全免。停止此预算内的小变体。
- **待确认 · 原 shared 形状与舍入下的 routed/readout 联合交付。** 两条可审机制是保留 routed 的原 M、专家分组和原算术 tile，只收缩末端有读者的返回与 readout，或在完整 EP 参与下让保留行在共同退役前交给真实消费者。前者须找出可删搬运并扣除不可删的 GEMM、TopK 归约和 shared/HC 数学，后者须证明原 counts、完成信号、peer bank/逆索引和下一代状态仍安全。新 `DFC` 需要用独立的调用前 clone 比较。最小设备对照按实际 row ID 依次取 routed/原 local finalize、shared Down、BF16 add 两输入和输出、HCPost，再到 compact readout。若仍暴露原 dense 输出，则逐位比较完整输出。选行接口要先闭合所有读者才可按原输出的同一有序投影比较。只有先证实 16 卡上有足够的返回/等待可移出关键路径，才值得抓故障同输入和六普通结果。单独重测 shared M 通用假设已无法改变结论。
- **已确认可实现 · 本轮没有新增替换准入。** 本轮新增了原 finalize 的本地控制和旧证据复算，未运行新 NPU 任务或改动 171 服务。

下一段追 V1 `_preprocess` 的文本直通与 encoder/多模态分支，从输入、缓存与 connector 交付走到 decoder embedding/attention。复用已扫的清退边，核对 copy event、释放和下一轮代次后继续剩余路径。
