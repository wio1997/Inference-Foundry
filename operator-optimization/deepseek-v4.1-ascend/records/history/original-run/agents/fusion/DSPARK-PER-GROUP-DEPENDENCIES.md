# F31：DSpark 多 cache group 的共同输入与特有输出

ZCode `operator-ceiling169-v2-106-dspark-helpers` 退出码 0，验证 model 为 `deepseek-flash`；`dspark-helper-provenance.json` 记录 baseline/stage 容器同镜像、无源码路径覆盖、两文件一致和 SHA，主控侧已复核本地文件 SHA。下面依据 `frozen-source/vllm_ascend/ops/triton/spec_decode/utils.py` 的 kernel 和 `sources/.../spec_decode/dspark_proposer.py:226–357`，是源码条件分析；没有证明实际基线有几个 group 或该调用在每次推理发生。

## 一次 group kernel 已有多项组合

这个单 kernel 同时处理 context position/slot copy 及 query token/position/slot/采样索引展开，不应把其中每对读写当成尚未融合的单独 kernel。它按 `pid` / grid-stride 覆盖 context 和 query 两个域；两个循环采用对应 mask，整个 kernel 返回前还不能假设所有程序块已完成。

在同一次 `for attn_group` 循环内，假设每次输入不被外部并发修改、相同的公共输出地址不和其他输入/输出意外 alias，kernel 的显式数据依赖如下：

| 输出 | 关键输入 | 组间是否依赖 group 专属信息 |
|---|---|---|
| `out_context_positions` | `target_positions`、context 长度 | 不依赖 block table、block size 或组的 slot 映射 |
| `out_input_ids` | next token、固定 noise token、每请求 query 个数 | 不依赖 cache group 专属信息 |
| `out_query_positions` | target positions、query_start_loc、rejected 数与 query index | 不依赖 block table 和 block size；无有效 context 时使用该请求第一位置、再减一处理 |
| `out_token_indices` | K、query 数、anchor 标志 | 不依赖 block table、block size 或 slot；anchor 与非 anchor 的写位置和 mask 不同 |
| `out_context_slot_mapping[gid]` | `context_slot_mapping_ptr[gid]` | **依赖该 group**，必须保留其值和长度 |
| `out_query_slot_mapping[gid]` | effective_seq_len、该 group 的 block table、stride、kernel block size，DCP owner/rank/interleave | **依赖该 group**；非本 DCP rank 所有的 slot 写 -1 |

这并不是声称每一次输入位置与 KV slot 相同：源码 `utils.py:163–170` 明确用 effective sequence length 算 KV 逻辑位置，用 last position 算 RoPE query position；对带重复/非连续位置的输入尤其不能合并它们。`SAMPLE_FROM_ANCHOR=True` 时 `num_query_per_req=K`，否则为 `K+1` 且只把 q_idx>0 写 sample 索引（proposer `39–43` 和 kernel `192–202`）；剥离公共写时必须保留两种域及边界。

因此在上述前提且**确实存在多组**时，源码支持一个可评估的重组：将 group 不变的公共输出算一次，继续为各 group 计算独立 slot。该结论比只看指针名称更强，但不证明 kernel 可以安全拆开、设备会少流量或启动一次就更快。必须验证：

1. 每组共享输入与标量是否实际一致、buffers 无未记录 alias、所有公共元素都完整写入（包括 mask、零请求与拒绝全部 token 情形）。
2. per-group table、kernel block size、slot buffers 的生命周期独立，尤其 `_context_slot_mapping_buffers` 按 layer 指向 group buffer（`dspark_proposer.py:308–311`），后续 `dflash_proposer.py:301–319` 截取用于逐层 cache 写；不可合并为第一个 group 的 slots。
3. `cad.seq_lens` 等在 group 循环**之后**再计算和替换（`313–342`）；更改先后会改变 effective_seq_len。共享输出的所有 producer 完成后才可交给 model、attention metadata、cache 和 sampler 消费，保持 DCP/TP/异步 stream、graph/eager 和事件依赖；该源码循环本身没有展开跨流 ready 证明。
4. `target_hidden_states` 本来经独立复制到 `_dflash_hidden_states`（`258–259`），供逐层 context KV 路径。删除这一步需要单独证明 source 生命周期、别名、异步完成及其他消费者；与 group 无关输出重复写是两个不同问题。

未来如基线只有一个 group，此项重复写优化在该基线不存在；源码未给当前 group 数。即便多个 group，拆分可能引入额外 kernel 和数据竞争，不能在没有时间/资源证据时声称收益或隐藏通信。
