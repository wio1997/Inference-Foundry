# F19：compressor latent 分叉的最早可写点与双 cache 发布

只读冻结源码和 stage155 的既有 trace。本次补充先前 F11/F18/F19 的一个具体未闭合点：**index-key 分支何时不再读取原 latent，以及 index/cache、long-KV 各自的首次消费条件**。不改目标实现、不运行 NPU；这是条件性重组，不是已验证优化或收益。F11 的 index 内部融合、F18 的 pool/norm 与 ring 先读后写，均不能与本记录重复计算。

## 现有顺序与可观察对象

`attention/dsa_v41.py:497–565`：ratio2 分支等待 compressor metadata，按各通道池化并归一化产生 latent；ratio1 由相应 compressor 产生 latent。两者都调用 `indexer.update_keys(latent,index_slots,cos,sin)`，返回后才将 **同一个 latent 的 view** 原地做 long-KV RoPE，然后分别写 long-KV cache。index 和 long slot 来自各自 metadata，不能因 shape 相同认定地址或有效行相同。

`models/deepseek_v41/indexer.py:106–125`：当 `owns_k` 且非空，index 路首先调用 `wk(latent)`，接着对 **wk 输出**做独立的 `k_norm → partial RoPE → dynamic_quant`，再把 INT8 key 和经 FP16 cast 的 scale 分别 scatter 到两张 index cache。`wk` 的输入宽 512、输出宽 128，其一般 dense 投影中每个结果依赖整行 512 特征；不能在读到一部分特征时就假定剩余特征不再被读。源码级可见链里 `wk` 是 index 分支对 `latent` 的最后一次显式读取，后面的计算处理 key；**这不证明内部量化 helper、编译生成图、hook 或其他 alias 不保留 latent，也不证明 wk 按行/块提前发出完成信号**。

`dsa_v41.py:568–607,613–680`：QLI 使用量化 query、index K、index scale 和 index metadata 计算/发布 topk；下游 SMLA 消费 `topk` 以及相应 long-KV cache、SWA cache 和 attention metadata。QLI 调用本身参数列表没有 long-KV cache，SMLA 才把它作为 `cmp_kv`。这让两种缓存有不同的最晚首次发布点，但不能省去未来 forward 对持久状态的维护。

## 两种候选和精确的依赖

1. **以 wk 完成读作为分叉屏障，而非等待整条 key 发布完成**：如果实际 wk/quant helper 的所有潜在读者已经确认，并且原 latent 对应行的完整读取确已结束、下游无其他 latent alias 依赖旧值，则 index 路可继续 norm、旋转、量化及 key/scale 写，同时 long 路对该行 latent 做原地 RoPE 并写自己的 cache。不能只在 Python `wk(...)` 返回或 device enqueue 后立刻修改：跨流必须有记录在真正完成读取之后的事件；若只能确定整个 GEMM 完成，就等整个 GEMM，不声称现有接口可逐行释放。新的 long 写与 index 写所在的 cache view 需要证明物理地址无重叠，且 cos/sin 及 slot buffers 到消费结束均不被其他 metadata 工作覆盖。
2. **分离 long 路的 64 维旋转结果**：对于 512 维 latent，long-RoPE 只改后 64 维，前 448 维不变（配置 `head_dim=512, qk_rope_head_dim=64`）。可研究在不原地修改 latent 的情况下，long 分支读原前 448 维、把后 64 维旋转到独立局部结果，并按原 page/slot/layout 合成写 long cache；这样 index 的 `wk` 和 long 分支在逻辑上都只读原 latent，可并发。必须先证明 `wk`、norm 结果及其其他消费者均按原输入语义读取、设备实际 load 边界/alias 不冲突，scatter 保持原目标 dtype、64 维配对及 rounding、页步长、有效/重复 slot 和一致性；原地 latent 内容可能被其他 alias、hook 或编译图观察时，不能直接删除其修改。单行 64×2=128 字节只是**若旋转 tail 为 BF16 的逻辑大小**，不是额外/节省的 HBM 流量或速度结论。

两条候选共同的发布次序必须包括：

- QLI 需要同一轮有效位置对应的 INT8 key **和** FP16 scale 皆完成并可见，还要等对应量化 query、weights、index metadata；只写一张 cache 便放行会把新 key 与旧 scale 混在一起。不同批次有相同 cache slot 或共享 storage 时，须维护原先覆盖次序和 reader 观察的一致版本。
- Long 分支写完后，属于该次 attention 可见域的 long-KV 必须在使用它的 SMLA 前完成，且 QLI 输出 topk 与 attention metadata 对齐；QLI 可以在 long 写未完成时启动的**必要条件**是其真实 kernel 和 metadata 不依赖那次 long 写，并且两张 cache 的访问不冲突。源码调用参数支持该方向，但未证明 kernel 隐含依赖、真实地址或事件，所以不能宣称当前已经可以并行。
- F18 中 C2 ring 是独立持久状态：池化读取旧 ring 并更新 ring 的前后关系必须保持，不能因 index/long 两支可并行就提前覆盖 ring。ratio2 未完成 group 的对齐零行、负 slot、padding、重复 slot 和 CP 全局 cache 行/本地 query 行域必须按原 metadata 处理；空 query 也可能需要更新缓存。
- 源码执行的 all-metadata wait、不同硬件 packed 和 non-packed 分支及现有跨流依赖须保留。两个分支的 Vector/MTE 竞争或调度开销可能抵消重叠；没有测量、无法估计收益。

## 静态跨层消费者的下界

`model.py:519–547,714–745` 与冻结 `evidence/model-config.json`：long-KV source 层为 2、8、14、20，配置层级关联分别为 `2→2..7`、`8→8..13`、`14→14..19`、`20→20..39`。这些是**可能读取同一 source cache 的层角色集合**；SMLA 是否实际读取某行取决于压缩长度、topk、mask、请求等，不能说每层必读每个新写位置。index K source 在 2、8、14、20；按 `is_index_source`，2、8、14 的 QLI 本层可读各自 key，20 的 key 可供 20、24、28、32、36 的 QLI 复用；非 index source 的层可能复用先前 topk，而不一定再次调用 QLI。cache 页之后被重用/释放的准确时刻、同请求跨轮值和真实 buffer 身份未由该静态列表确定。

## 历史窗口仅作相容性线索

既有 stage155 EP8、target 层2 run2、stream89 中，`[24,512]×[128,512]→[24,128]` 的 MatMulV2 位于 261.174–267.854 µs，之后可见 128 维 norm/RoPE/量化、INT8 与 FP16 两次 scatter，512 维 partial RoPE 在 312.133–318.833 µs、BF16 scatter 318.853–331.233 µs，随后 QLI 在 335.853–426.291 µs。完整 dtype、文件 SHA 与模型验证见 `F11-INDEX-KEY-CACHE-REVIEW.md`。这组任务的名称、shape、同流顺序与上述原代码路径相容；**没有 tensor identity、slot 数值、源码调用栈或跨流 event 对应，不证明它们组成同一调用链，不证明可提前释放或可隐藏任何时间**。24 是物理形状，不是已证有效 token 数。

## C2 完成行、cache 目标与 RoPE 位置不是同一个坐标

对冻结 `attention/dsa_v41.py:195–202,229–242,970–1030,1283–1388` 和 `compressor_triton.py:216–320,650–719` 的非 packed 路径定向核对，有一项拆分重组必须保留的跨模块映射：

1. C2 pool 输出保持输入 token 行对齐，先整块初始化为零，再仅把完成组的 pooled 值写在该组**完成 token 行**；已完整组数由每个 request 的起始位置和 used_len 计算。若首组跨 chunk，旧 token 的 kv/score 来自对应 request 的 ring，新 token 来自本次投影。非完整行的零值本身不证明相关 cache 均安全、可丢弃。
2. 非 packed C2 cache builder 不只检查 token 位置：对原始物理 `slot_mapping`，只有非负、`(slot+1)%2==0` 的行先映射到 `slot//2`；接着要求该行在有效 token 范围，且有 positions 时还要求位置为奇数。最后才将物理压缩槽按 cache block 大小变成二维坐标；`skip_ring_state_update` 时该分支全部标无效。`scatter_cache_sk` 调用 `npu_scatter_nd_update_sk`，其注释说明 `[-1,-1]` 被跳过，同时保留目标 cache view 的物理 page stride。index cache 和 long cache 由各自 metadata 的 slot 提供，不能只凭值看起来一样就把两个 tensor 判为同一对象。
3. 普通 C2 metadata builder 对有效奇数位置 p 发布的 `source_positions` 是 **p−1**，未完成或无效行填 0；如有 full source cos/sin，则从这一位置 gather 旋转表。`_write_compressed_source` 把同一 source cos/sin 用于 index-key 和 long-KV 的各自 RoPE；ratio1 则使用传入的当前 token 的 cos/sin。若图外构建 metadata 且设置 group ID，C2 source 路径还先等待 `DeviceMetadataStage.COMPRESSOR`。这些结论分别限定于源码所列分支，packed backend 的 builder、历史实例、slot 数值和 event 未绑定。

一个可手算的条件示例：对齐的单 request 当前 chunk 只含 position 101，且物理 slot 也是 101、没有 skip，pool 对该完成行需要原 position 100 的 ring 值和 101 的当前值；在给定有效页等其它条件下，压缩槽为 `101//2=50`，RoPE lookup 使用 source position `100`。如果把逐块流水中“压缩输出行位置”或“缓存压缩槽 50”直接用作 RoPE 位置，就不能在一般旋转表下保证原始结果；如果把 token 101 当成独立组，则会丢弃历史贡献。这个例子由源码公式条件推导，**不是观测到的 169 请求或 cache slot**。实际序列位置、原物理 slot 和压缩槽是不同坐标，不能普遍假定它们数值相等；页号、请求边界、invalid/padding、重复映射及 cache view stride 都影响能否改成按组 compact 后写入。

已有 EP8 layer2 run2 的 `[24,512] → [24,128]` index 投影及之后 key/scale、long cache 散写等轨迹（上节及 `F11-INDEX-KEY-CACHE-REVIEW.md`）在形状和顺序上与此图相容，但没有同轮位置、block table、slot、ring 与 tensor 身份，因此不能从 24 行推断完成组数或重复率。若计划只对完成行执行 indexer/long-KV 分支，需要额外证明压缩和还原映射、所有被跳过行的外部可观察行为、scale/slot 与 metadata 的就绪和缓存一致性；不可直接把一半物理行当成已证可省计算。离线证据无新增设备工作，历史运行分支及上界仍未知。
