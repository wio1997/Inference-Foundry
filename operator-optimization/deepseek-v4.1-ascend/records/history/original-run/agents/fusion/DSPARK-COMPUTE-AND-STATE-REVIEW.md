# F27–F31：DSpark 数据与状态重组边界

本次补查冻结源码 `models/deepseek_v41/dspark.py`、`spec_decode/dspark_proposer.py`、`spec_decode/dflash_proposer.py`、`spec_decode/llm_base_proposer.py`；只提出有条件重组，未修改模型或运行卡。启动脚本请求 DSpark、K=5 和 eager；具体对象、分支及此次 profile 的请求逐次值并未观测。

## F29：上下文 KV 的两路值、槽位和 persistent cache

`dflash_proposer.py:301-319` 先调用模型的 `precompute_and_store_context_kv`，然后 proposer 执行草稿模型（`llm_base_proposer.py:1480-1495`）。`dspark.py:295-309` 对每个草稿层取该层 attention，把同一 context states 走该层的 wkv → kv_norm → 按 `[nope_head_dim, rope_head_dim]` 拆分 → 对 rope 片段依 positions 旋转 → cat/contiguous，最后将该层结果写入其 SWA cache。不同层对应的 attention 参数和 cache 不能无条件复用，即便输入 context 相同。

非 packed 路径 `_store_standard_swa_kv:223-246` 在 `slot_mapping is None` 或空时不写；对一维物理 slot，先保留 valid=`slot>=0`，非负 slot 按 cache block size 映射为 `[slot//block_size, slot%block_size]` 并转 int32，负 slot 两列填 `-1`；之后 `scatter_cache_sk` 负责目标 cache dtype 转换、contiguous 和跳过 `[-1,-1]`（`attention/dsa_v41.py:227-242`）。`reproduce_dspark_slot_mapping.py` 在 block 128 的 `[0,127,128,-1,-25,4097]` 小例子给出 `[0,0],[0,127],[1,0],[-1,-1],[-1,-1],[32,1]`，不是实测数据。

**组合候选**：研究把 cat 的两段和 slot fixup 直接供给 cache scatter，条件是 preserve wkv、norm、RoPE 及其 dtype/舍入，cache dtype cast、physical page/offset、负 slot skip、所有层私有 cache 和所有其他消费者；它可以改变中间 buffer 的生成，但不能删 persistent cache 写或跨层用错参数。重复物理 slot 的写冲突顺序、页复用、地址 alias、stream 完成与 graph 生命周期需要证明。代码还在调用 `_project_shared_kv` 后才检查空 slot；即使数据写被跳过，提前跳过整个投影也必须先证明 RoPE 缓存、hooks、异常和其他副作用不被观察，不能仅据空 slot 推断无条件可删。

## F30：批量基础计算与逐位置 Markov 采样是不同依赖

`llm_base_proposer.py:1534,1577-1585` 先为选中位置批量计算 base logits，reshape 成 `[batch,K,vocab]`，草稿缓冲第 0 列从 seed 复制。随后 `1586-1602` 对每个 i 依次执行：读 `draft_token_ids[:,i]` → `markov_embed` → `markov_bias` → 把 bias 原地加入该位置 base logits → greedy argmax 或 probabilistic sample → 写 `draft_token_ids[:,i+1]`。因此草稿模型并行产生各位置基础 logits，与其后 Markov 采样 loop 的 token 链条是两个不同层次；不能把整个过程说成逐 token 重跑 decoder，也不能假定这 K 个 Markov token 相互独立。

**组合候选**：对于 greedy 分支研究将当前步 bias 加法和 argmax 的数据扫描组合；可以在第 i 步已知的 token 上开始当前步工作，但下一步仍依赖其结果。必要保留 logits 的数值 dtype/加法舍入、词表分片与跨 rank 选择、argmax 的实际 tie/NaN 规则、原地修改的其他消费者、token 与 seed buffer 生命周期；并行请求不代表共享 RNG/设备资源无冲突。

概率分支 `1590–1606` 需要保留 sampling metadata、RNG 状态及概率张量，后者 stack/reshape 并在 `1637–1641` 保存以供后续验证；不能把 greedy argmax 融合直接套到概率采样。可见代码在存在 vocab mapping 时回退 greedy；V4.1 `map_draft_to_target` 是 identity，但运行时 `draft_id_to_target_id` 的实际对象和采样模式需单独核验。

`dspark_proposer.py:61-71` 仅当配置动态方法为 dspark 才创建 scheduler，`llm_base_proposer.py:1608-1617` 条件调用 `dynamic_spec.update`。`ascend_config.py:1303-1314` 的默认 method 为 None，冻结命令没显式指定 `dynamic_spec_config`；若没有其他覆盖，则动态 confidence 不应视为每次必走。ZCode stage 106 的冻结 `spec_decode/utils.py:380-405,440-479` 显示，若 dynamic DSpark 分支实际开启，`DynamicSpecScheduler.update` 自己对 `draft_token_ids[:,:K]` 批量调用 `model.markov_embed` 并计算 confidence；逐步采样 loop 的每一步也分别对相同 token 列调用 `markov_embed`。这提供一个**条件重复 lookup 候选**：保留前 K 次逐步 embedding 以供 confidence 复用，但须证明模型参数与状态未变、每步结果和批处理形状/collective 的数值语义相同、排列/裁剪/dtype 正确、异步 buffer 未被覆盖且图生命周期允许。被审计路径并非 `dspark_proposer._compute_confidence` 直接调用；实际动态分支默认关闭、运行是否覆盖未知，不能计入本次基线收益。

## F27/F28/F31 的保存条件

- F27 的 `main_proj` 使用 `gather_output=True`，`combine_hidden_states` 是投影后 norm（`dspark.py:192–203,281–282`）。组合不能忽略投影输出的 TP gather、shape 和 norm 对整个 hidden 行的归约；A3 上 main_proj 的配置选择还受 packed-cache/quant config 分支限制，不应仅由注释中的 FP8 checkpoint 判断真实权重/计算。
- F28 的 embedding 先 `unsqueeze`、`repeat` 成 HC 多份，条件 SP shard 后产生 `[1,0,…]` 的 FP32 `pre_mix`，送入 3 个串行 draft layer（`248–276`，具体层数按 config）；不可以把 repeat 的 HC 视为普通共享地址而让后续任意原地写互相覆盖。结合首层 HC 计算是条件重组，但不能消除所有 HC 通道输出和最终 collapse。
- F31 中 `set_inputs_first_pass:267–307` 按 attention cache group 调用既有 `copy_and_expand_dflash_and_dspark_inputs_kernel`，每组都有自己的 block table、kernel block size、context/query slot 输出，但共享 input_ids、positions、context_positions 和 sample indices 输出。对这几类共有输出，提取 group-invariant 部分并避免重复写是检查方向；不能假定不同 group 输出一致，须看 kernel 如何依赖 block size、table、拒绝 token、DCP rank 以及 mask，且维护所有 per-group slot 与 buffers 的生命周期。stage 106 已核对当前两个容器的该 helper SHA；逐项依赖和事件条件见本目录 `DSPARK-PER-GROUP-DEPENDENCIES.md`。原 kernel 已把扩展和多种输出组合在一次调用中，不要重复算作新的已实现融合。

## 草稿 context KV 与首层输入准备的跨分支流水候选

这与 F29 的“各草稿层 context KV 之间流水”以及 F28 的“首层 HC 内部重组”不同：研究的是 **context KV 生产分支**与**草稿首层 attention 之前的输入分支**之间是否能交叠。只从已保存源码推导，不代表本次运行命中、已有 NPU 并行、已实现可行接口或收益。

- 冻结 `spec_decode/llm_base_proposer.py:1473–1495` 的 DSpark 分支先执行 `build_model_inputs_first_pass`，再调用 `self.model(**model_kwargs)`；`dflash_proposer.py:301–319` 把 context hidden、positions、各层 slot 传给模型的 context KV 预计算。`models/deepseek_v41/dspark.py:295–309` 逐层产生并写该层 context cache。其后草稿模型 `:248–271` 做 token embedding、HC repeat、条件 SP 分片和 premix，并调用首个 decoder；`model.py:887–899` 的首层先做 HC pre、输入 RMSNorm 与可选 SP all-gather，再进入 attention。以上说明 Python 源级的先后及数据接口；不单独证明设备流实际同步、依赖或跨流执行时序。
- 输入形成与保护条件：`dspark_proposer.py:240–259,267–310` 中 seed 复制、target hidden 复制及每组输入扩展生成 input IDs、query positions、context positions、各组 slot。不得在相应生产完成前消费，也不得忽略多组对共享输出的覆盖及其值一致性；实际 rank、group、序列并行、采样和 metadata 值未知。已有多组 kernel 本身已融合多项输出，不把它重新算成新融合。
- 条件结构：在上述输入及 context 数据可用后，可研究让原样的 context KV 写入与原样的首层 embedding → repeat/shard/premix → HC pre → RMSNorm（及适用时 SP all-gather）分别执行；在调用首层 attention 前保守等待**本次所需全部 context cache 写完成**和 attention metadata 就绪。若进一步只等待首层 cache 而与其他层写重叠，须额外证明各层 cache 地址、slot 映射、事件及 query 更新不冲突，并确保每层首次读/写前其对应写完成；不得凭层序宣称独立。首层 query KV 与 context KV 若写同一 cache，必须保留原地址冲突、重复 slot 的写先后及最终值。
- 分支若跨流实施，须证明 forward context 与 RoPE/cache 元数据的正确作用域、stream event 的 happens-before、所有读写缓冲区与下一轮重用安全、权重与设备算子线程安全，以及各 rank embedding、SP/TP collective 的同序/可重入。HC pre 的四项输出与 residual、RMSNorm 舍入及后续各消费者不可删除；不可把 F28 条件下 `y=e` 当作数值恒等前提。两个分支可能争抢 Cube、Vector、HBM、通信和 host 下发资源，即便依赖允许也可能不加速；不能由静态串行顺序估算收益。
- 历史窗口尚未绑定对应 context cache、首层算子、请求、事件和张量，故无运行时成本或收益数字。后续仅在已有资料存在时，核对同调用的 producer、consumer、cache slot 与流事件；缺数据保留未知，不启动新 NPU。已核对五份源码 SHA-256：`dspark.py` `fcee256905928917154d8dac94d6611bb83d02da441e8aa154a525ef51e6a601`；`model.py` `101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91`；`dflash_proposer.py` `f3b736f44a59ec0831aa7f2849e2eae6d709951e955992b8016f49e772fce3ff`；`dspark_proposer.py` `70d216de21875e679d510153ebdb0f0ccf130875a0e644f0673d60b67b6b633c`；`llm_base_proposer.py` `458fb50e6839f3a353ddc7181ace4fdae02e35093d8ccd1c4327dc24ae5fff31`。
