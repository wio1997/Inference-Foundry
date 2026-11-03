# EP8 layer14 的 MC2 本地专家行数：静态接口与采集条件

已知新 profile 中一个 DispatchFFNCombine 的输入形状与 1,023×5,120、top-6、24 个本地专家相容，输出记录含 INT32[24]。**形状和名称不能单独证明该槽位与下面保存源码的变量、实际实例或运行二进制一一对应。**逐专家值、有效 mask、expert placement 与 maxOutputSize 均 unknown。

## 保存源码显示的契约

源码根 `/Users/wio/work/169-vllm-ascend-reproduce-172/evidence/precision-paths-20261028/sources/vllm-workspace/vllm-ascend/`，它们是冻结参考源码，还未证明与本次执行二进制绑定：

- [moe_comm_method.py](/Users/wio/work/169-vllm-ascend-reproduce-172/evidence/precision-paths-20261028/sources/vllm-workspace/vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py)（SHA-256 `7040f691355bd9acb3c668e698de3f7d3854edbc22e3f979f9b03d99800364d7`）:317–320 分配 NPU 上 `int32[num_local_experts]` 的 `self.expert_token_nums`；580–609 的 custom 分支传入 topk IDs、mask、probs 和该输出，然后作为 `expert_tokens` 返回。CANN MegaMoE 是独立分支，不能套用其输出实现。该张量为实例成员，可在后续调用复用覆盖。
- [dispatch_ffn_combine_torch_adpt.h](/Users/wio/work/169-vllm-ascend-reproduce-172/evidence/precision-paths-20261028/sources/vllm-workspace/vllm-ascend/csrc/mc2/dispatch_ffn_combine/dispatch_ffn_combine_torch_adpt.h)（SHA-256 `d9e65033bf74cc633ec9d2472d9957d46b7971265ac48a7cd63ede7221332a40`）:20–56、90 在 INT8 权重分支将 `expert_token_nums` 作为第二个可写输出交给 aclnn；不同 dtype 有 W4A8/BF16 其他入口。host 的 [dispatch_ffn_combine_def.cpp](/Users/wio/work/169-vllm-ascend-reproduce-172/evidence/precision-paths-20261028/sources/vllm-workspace/vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_host/dispatch_ffn_combine_def.cpp):64–74 定义第二输出为 INT32。
- [dispatch_ffn_combine_tiling.cpp](/Users/wio/work/169-vllm-ascend-reproduce-172/evidence/precision-paths-20261028/sources/vllm-workspace/vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_host/dispatch_ffn_combine_tiling.cpp)（SHA-256 `783f76d7869371df949fc5ee80e33dfca8cf81231a65da410fcfea40df89b6f8`）:266–275 令内部 routing 以 expert count 模式（flag 2）输出；[moe_v2_common.h](/Users/wio/work/169-vllm-ascend-reproduce-172/evidence/precision-paths-20261028/sources/vllm-workspace/vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_kernel/moe_init_routing_quant_v2/moe_v2_common.h):43–44 定义 2 为 count、1 为 cumsum。
- [dispatch_ffn_combine_kernel.hpp](/Users/wio/work/169-vllm-ascend-reproduce-172/evidence/precision-paths-20261028/sources/vllm-workspace/vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine_kernel.hpp)（SHA-256 `7dcef4c08c4c47e38bdd01aef0dee997798658ee4d7b871c8e6c70f7ffe596ba`）:792–815 先处理 mask，再路由并跨 rank 汇集计数；402–431 按 EP 来源 rank 维累计**每个本地 expert**的 count；825–829 把最后一个 EP 累计切片写入输出。这里的累计方向是来源 rank，不是按专家依次累加的 group-list cumsum。该值是各来源 rank 路由到本 rank 每个 expert 的总行数，不是本卡输入 1,023 行单独的 top-6 路由量；它也不能还原 peer×expert 细分。
- 上述输出是在 GMM 截断之前的计数。kernel :454–460、517 与 :561–566、621 按专家顺序把 `currentM` 裁到 `maxOutputSize` 的剩余空间。只有同一实例源码、expert 顺序、每专家计数和同次 maxOutputSize 均绑定后，才可按该逻辑计算 GMM 分组的有效 M。不能由这一向量推断通信等待时间。
- [routed_experts.py](/Users/wio/work/169-vllm-ascend-reproduce-172/evidence/precision-paths-20261028/sources/vllm-workspace/vllm-ascend/vllm_ascend/ops/fused_moe/routed_experts.py)（SHA-256 `f9db74836ae210a541432b923118696c7a15be9a10d5776834e1d3b64ce11d77`）:333–355、752–775 显示部分 EPLB 分支能把 expert count 记录进设备侧 load tensor，另一分支也可能按阶段累计。未有证据显示本次开启、记录为单次、或留存对应数值；累计值不能代表某个 layer14 调用。

## 已有数据与最小额外诊断

当前已保存 profile 标记、shape/dtype、scheduler 和 sidecar **没有找到可绑定此调用的 24 个具体元素**；sidecar hook 只保存 tensor metadata。此前 EP0/EP8 文件名清点也没找到明确 dump（见 [LAYER0-MC2-DP-RECOVERY.md](LAYER0-MC2-DP-RECOVERY.md):170–178、208），但不能用文件名否定未检查文件的所有可能。运行时现存预分配输出是易变缓冲，不能作为历史 replay 的值。

如需新诊断，应在独立诊断服务上只选 **EP8、target layer14、一个可识别 graph/replay 与一个 MC2 调用**，确认该实例就是上述 custom INT8 路径与输出槽，保存同调用的 rank、layer、call/graph 标识、输入形状、mask 是否存在、EP size、maxOutputSize、输出 shape/dtype 和 24 个计数；不读取权重、router IDs、prompt 或生成文本。诊断插入点需在该 MC2 输出产生后、共享输出下次覆盖之前；如同流安全地将 24 个 INT32 保存到独立设备槽，再通过已确认的事件/依赖待拷贝完成后传回 host，可避免在热点处直接读取，但仍增加拷贝、同步、带宽和生命周期开销。同步 host copy 更简单，却可能导致流等待，测得时长不可作为原基线性能。

图捕获/重放的关键限制：Python wrapper 通常只在 capture 时执行，不能假设 replay 每次触发 Python hook。一个可行性方案是在图建立时将**该层**的输出快照放入图中、隔离出该次 replay 的独立槽，在该 replay 完成且下一次覆盖前安全取回；必须先验证图捕获支持、目标调用身份、流依赖和缓冲生存期。若无法保证唯一 replay 或避免覆盖，则结果记 unknown，不能静默用后一次值。另一种临时关闭图以观测的诊断会改变运行路径，不能直接作为原 graph 性能或相同调度的证据。本报告只提出方案，未运行设备或更改服务。

## 补充：具体参数来源、有效行计算与最小采集协议

保存源码中，custom 分支在 `moe_comm_method.py:591–607` 将 `get_ascend_config().mega_moe_max_tokens` 作为 `max_output_size`，连同输入 `x_active_mask`、EP `world_size` 和可写输出 `expert_token_nums` 传入；`dispatch_ffn_combine_torch_adpt.h:39–56` 仅在第一组权重 dtype 是 int8 的情况下选择 `aclnnDispatchFFNCombine`。aclnn 的 workspace-size 接口声明将容量作为 host 标量属性，tiling 源将该属性复制进 `dispatchFFNCombineInfo.maxOutputSize`，设备实现读取 tiling 字段。源码配置类的默认 65536 **只是默认值**，不得作为实际捕获图、实际执行二进制或本次样本的容量；本地诊断摘要中的 json、log、txt 搜索也未找到此参数，不能排除尚未搜索的 profile 或二进制 metadata 中含有它。应在图构建时记录传给该调用的 resolved 标量，并绑定该调用、graph 实例、rank、layer、源码与执行二进制身份；如能读取当次 tiling 进一步交叉确认，需先证明解码结构适用该二进制，否则不能声称一致。

对于已证明对应上述内核的单次调用，记依本地专家顺序的输出为非负整数 `c[0..E-1]`，当次容量 `M` 为非负整数。按两个 GMM 分支的 `currentM` 截断与 `preCurrentmSum += currentM`，可重算：

```text
used = 0
for i = 0 .. E-1:
    effective[i] = min(c[i], max(M - used, 0))
    used += effective[i]
R = used = min(sum(c), M)
```

前提为输出确实来自该次调用、专家排列与该 kernel 一致、容量为该次实际值、整数无溢出且计数合法。两个 GMM 分支对相等边界分别使用 `>=` 和 `>`，但相等时结果同为原 `currentM`。这是**逻辑有效行数**，不是硬件执行的 tile 填充行数；单专家分布、ceil 分块、量化和其他计算会影响实际硬件工作量。计算时记录每个 `effective[i]`、`R`、`sum(c)-R`、非零专家数、最大专家行数以及分布；不得从总和单独估算实际 tile 数或排除专家负载倾斜。

最小诊断设计，先验证接口可行性再决定是否执行，并非已有采集结果：

1. 仍采用单独诊断实例与既定负载；先确定 profiled prefill 中异常层 14 与对照层 39 是否真的走图重放、同一个真实 EP 组及是否在该图中各执行一次。记录执行映射、图 ID/形状、replay 序号、rank/层/调用序号、输入行数与 dtype、mask 是否存在、resolved 容量、二进制及 tiling 身份。历史 sidecar digest 不能直接反推出客户端请求 ID；新一次重跑也不能视为历史那次调用。
2. 在已验证的实际执行流上，将每个目标层的 count 输出在对应 MC2 写入完成、同一共享缓冲下一次写入之前，复制到该层**专用、预分配**设备快照槽。若图捕获，快照操作必须进入图并保留真实依赖；若后台图/其他流能写同一个源缓冲，必须先建立经验证的依赖，不能假设 Python 返回即 NPU 完成，也不能假设 host hook 随 replay 重执行。测试工具所增加的复制节点属于诊断插桩，不能说目标计算完全未受影响。
3. 图的固定地址意味着下一次 replay 会覆盖同一快照槽。必须在**确定的目标 replay 完成后、复用该槽之前**，通过经验证的事件与流依赖将快照转移到独立 host 存储；或使用已证明安全且可区分重放次数的隔离槽机制。若需暂停下一次 replay 防止覆盖，则明确记录这会改变调度，不得把该次时延作为原基线。如果无法证明 capture 中拷贝、完成事件、槽复用和 replay 编号的对应关系，结果直接标记 unknown；单次 capture 时调用 Python `cpu()` 或稍后读共享张量都不能保证捕获正确值。不得增加图内跨 rank 等待或单卡重放已含通信的算子，以免死锁或破坏重叠。
4. 尽量同时获取已验证的同一 MC2 通信组全部参与 rank、异常层和对照层的值；实际成员数必须绑定该次调用，不能以 EP8–15 的 8 张观察卡或 TP 组大小代替。此前另一采集的 EP0/EP8 组元数据列出 MC2 组为 16 个成员，但尚不能将其直接绑定到本次调用。单 rank 只能回答本 rank 的逻辑工作量，无法判断组内负载不均。对同一 graph/replay 和层检查每个向量长度、非负性与容量，保存拷贝/事件错误和丢样；预期的计数总和还受 mask、复制或丢弃、expert placement、容量及执行实现约束，未证明前不能要求等于输入行数×top-k。仅记录计数和匿名执行标识，不采集 prompt、生成文本、router ID 或权重；专家计数仍可能泄露工作负载特征，应按受限诊断数据处理。
5. 区分两种假说：若异常层 `R` 显著小、非零专家少或分布更偏斜，说明实际逻辑工作量或分片条件值得继续追踪；若 `R` 与分布相近而异常层时间远长，单靠工作量不足难以解释这一对样本，应继续查内部 stage、同一组 rank 的同步与等待。然而这两种观察**都不能单独证明**通信等待、算子缺陷或完整模型瓶颈；还需同次时间线/内部阶段证据，并报告采集引入的时延、内存和调度扰动。计数采样结果只能解释新诊断运行，不能反填为原 profile 的未知值。

## 本次实例的容量配置：只读证据核查（2026-10-29）

- **启动请求**：stage 保存的该次隔离诊断 `runtime/scripts/serve-profile.sh:31` SHA-256 `ff12be2d10372494fa93bd4358257f1332d60586830b8b38a00818852bb84aca`，与 [launch.json](../../diagnostic-20261029/launch.json) 记录的 diagnostic script hash 一致。解析其 `--additional-config` JSON，里面有 `enable_fused_mc2=1` 等设置，但**没有** `mega_moe_max_tokens` 键，即启动命令没有对此键提供显式请求值；这不等于证明 worker 最终值。
- **APIServer 解析值**：该诊断实例实际保存的 `runtime/logs/service.log:25` 为该 APIServer 的 non-default args，打印的 `additional_config` 同样未包含该键；该日志 SHA-256 `3b211068eaa9522c2d246214f885c65d66d0c49b0cbf58037613f561d757d3a7`。这只是 APIServer 参数记录，不是某 worker 的 `AscendConfig` 实例字段。
- **源码条件推断**：冻结参考源码 `ascend_config.py:600,1794,1893–1919` 显示该字段类默认 65536，而 factory 从 `additional_config` 构造实例并进行后续验证。若运行时所用代码确与该参考源码一致、配置未在其他路径注入或修改，那么 65536 是有依据的候选默认值；目前缺该次 worker 实例值或 tiling 属性记录以及执行代码绑定，故不能把条件推断当作本次实测值。
- **搜索覆盖**：检查该次完整 service.log 的 `mega_moe_max_tokens`、`max_output_size`、`AscendConfig(`，以及 23 个该实例 metadata 文件中前两个字面字段，均无匹配；脚本所 source 的 `config/serve.env` 未含该名字的环境变量。仍未遍历设备 profile 二进制和其他全部可能工件，搜索无匹配不等于证明没有值。具体受控审计数据及各文件哈希见 [mega-moe-max-tokens-config-audit.json](../../diagnostic-20261029/mega-moe-max-tokens-config-audit.json)。

因此目前结论是：**请求端没有显式指定；65536 是参考源码默认值；该次 worker 实际 resolved 值及 kernel tiling 中的值仍 unknown**。在不启动新的 NPU 负载的前提下，应优先查同次残留的可信 worker 配置快照或可与执行 binary 一一绑定的 tiling 记录，而不是从源码默认补造实际值。

## 插桩准备：接口已定位，采样器尚不具备安全执行条件

### 实际绑定点

- **层与主/草稿模型**：保存的 `DeepseekV41DecoderLayer` 构造时从 prefix 写入 `self.layer_idx`，持有 `self.mlp = DeepseekV41MoE(..., is_draft_layer=...)`；`DeepseekV41MoE` 也设置 `self.layer_idx` 并创建 `self.experts`。decoder forward 经 `self.mlp(...)` 调用。既有诊断使用 decoder forward 的 Python 包装提供 layer marker，但 custom op 可能被编译或跨线程运行；它不能自动证明内部 custom MC2 是该层的那次执行。未来适配必须先验证 target main 层实例到 fused-expert 调用的单一映射、相同执行上下文、调用先后关系及 draft 区分；拿不到则拒绝，不能从 layer 数字或 tensor shape 猜。
- **输出缓冲可能跨层共享**：`moe_comm_method.py:50–73` 的 `_MoECommMethods` 按通信类型及专家数量 key 缓存通信实现；`FusedMC2CommImpl:317–320` 为该实现分配一个 `self.expert_token_nums`，custom 调用复用它。是否每层属于同一 key/实例需运行时核实；安全设计必须按共享处理，**层 A 的 MC2 → A 快照 → 任何后续写者**的设备执行依赖必须成立，即使下一个写者属于层 B。不能仅在外层 decoder 的 Python forward 结束后读取这个可变张量，也不能把其地址当层身份。
- **rank 和容量**：`token_dispatcher.py:112–118` 初始化 `get_mc2_group().rank_in_group`、`world_size` 和通信组；应在对应同一具体调用处记录 MC2 组匿名标识、组内 rank、world size，再绑定全局 rank、设备和进程。现有 EP 标签与逻辑 MC2 组关系必须在该次实例确认。容量需在同次 custom 调用实际参数处读取 `max_output_size`，并记录该方法与二进制身份；仅调用前后读取可变单例、推断源码默认值、读取另外一次运行配置均不构成该次参数证据。HCCL 名称可本地匿名化，绝不依靠设备序号猜组关系。

### 设备接口与拒绝条件

只读了 169 自有 stage 镜像中已安装的 Python 接口源码，未 import torch_npu、未初始化设备：`torch_npu/npu/graphs.py:67–72` 的 `is_current_stream_capturing()` 仅报告当前流是否正在 **capture**；返回 false 不排除 graph replay、别的流 capture，也不证明 hook 每次执行。该文件 SHA-256 `f8ba6f797542c3a2ee874f4e3e764c33f7926cb536ec7c9af1160110e6c1aa5d`。`streams.py:28–69,209–235,249–258` 的 event record/wait/query 和 stream wait 说明事件操作及完成语义，`utils.py:263–275` 提供 current_stream，但这些接口存在不证明 custom op 的真实流、跨流写入顺序或图内 event 安全；`current_stream` 还会调用 `_lazy_init()`。相应 SHA-256 为 `00123b33d129387678d110eb75dc1fa3a6e7cca28222c71291bb7376bc45912f` 和 `7c316b573f8e7310630eccb38fcce77bf5f5addaec596ad44aee0d4807789c40`。

准备的 [expert_snapshot_draft.py](../../diagnostic-20261029/expert_snapshot_draft.py) 是仅依赖 Python 标准库的**CPU 契约草稿**：定义必须绑定的字段及逐专家截断计算；其 `request_device_snapshot` 无条件拒绝，没有 NPU 后端、不会采样，也不会因传入一个布尔值或文本声明就绕过证明。目标实现和现有服务均未修改。

在本机用合成计数检查了截断、无截断、非法计数/容量和无条件拒绝采样，均通过；这只验证 CPU 契约逻辑，不验证设备接口、并发或模型输出。

要变成真实可执行插桩，还必须先得到以下证明或实现；任一项缺失则结果记 unknown：

1. 目标 main 模型层与 custom INT8 调用一对一绑定；该实例恰是需要研究的 profile 路径，而非 CANN MegaMoE、草稿或另一层的调用。绑定 rank/设备/MC2 组、当次 op 的 resolved capacity、真实调用及已加载二进制 hash；回放路径另需 graph 实例与 replay 序号。不得从客户端请求 HMAC 反推真实请求 ID。
2. 验证目标算子写 count 的设备流、所有可能的跨流写者、源/目标缓存不别名及存活期，并在该次算子写入完成之后、下一次写入之前，将恰好 24 个 INT32 复制到专用槽。对异步流只能以证明的事件依赖作为顺序依据；不得通过插入跨 rank 等待或强制整机同步制造顺序。
3. graph 情况只能在已验证 graph capture 时加入可重放的复制节点，并证明图未删除/重排此节点、完成/复用关系以及每次 replay 的唯一编号。静态槽被下次 replay 覆盖前须保全并证实；无法唯一对应或者要暂停 replay 才能保全时，应分别拒绝原图归属、标注调度干扰。仅依靠 Python hook 在 replay 重执行、仅调用 `is_current_stream_capturing()`、或在事后读取原缓冲都是不安全方案。
4. 只在原本授权的隔离诊断实例中运行，先核实空闲与所有权；记录仅包括计数、非敏感匿名元数据及必要的有效行统计，禁止读 prompt、权重、router IDs 或输出文本。小量计数仍是受限负载信息。单独标记真实采样额外拷贝、事件、槽位与可能的扰动；仅用同次采集的值解释该次运行。

截至现在，上述 1–3 的关键身份、执行流与 replay 证明均未取得。因此不能安全提供可运行的 NPU 采样器，也尚未启动任何采样。这是对插桩可执行性的限制，不是对整个离线分析任务的阻塞。

## 进一步源码定位：具体接入点与无强制暂停的候选回放取证

再次只读 169 自有 stage 的实际文件，定位到比外层 decoder marker 更直接的对象关联：

1. `AscendRoutedExperts` 调用 `self.quant_method.apply(layer=self, ...)`（`routed_experts.py:739–747`）；W8A8 dynamic 方法在 `w8a8_dynamic.py:203–234` 将同一 `layer` 传到 `build_fused_experts_input`，其构造器 `dataclass/fused_experts.py:105–132` 把它存入 `MoEFusedExpertsInput.layer`。通信方法 `FusedMC2CommImpl.fused_experts` 外围的**隔离 Python 包装器**因此有机会直接读取这个对象，而不是依靠形状猜层。
2. 上游 `vllm/model_executor/layers/fused_moe/layer.py:217,374–379,408–428` 以 prefix 为 `layer_name` 创建 routed-expert 实例，交给 MoERunner；`RoutedExperts.__init__:56–86` 保存 `self.layer_name`，`MoERunner.__init__:250–272` 保存 `routed_experts`。DeepSeek 的模型 `model.py:151–164,236–245,766–815` 提供 layer index 与 `mlp.experts`。候选方案是在主模型加载后建立 `model → decoder.layer_idx → mlp.experts.routed_experts` 对象的只读映射，调用时以 **Python 对象身份**与 `fused_experts_input.layer` 核对，并取其 `layer_name`、当前 MC2 组/worker rank；若包装层需要 unwrap、对象多重归属、未注册、draft/main 混淆或路径不存在则拒绝，不从名字单独推断。对运行时动态替换对象需重新绑定。关联只读取对象元数据，不遍历参数或保存权重、prompt、router IDs。
3. 读取实际 custom op 参数的候选接入点是 wrapper 内局部 `TorchDispatchMode`。stage 中 `csrc/torch_binding.cpp:3226–3231` 的 schema 显示 `max_output_size` 与可写 `expert_token_nums` 是同次 `_C_ascend::dispatch_ffn_combine` 的参数。模式应按实际 schema 字段名解码、只处理精确 op，其他调用直接透传，记录标量 capacity 与是否为预期同一 count buffer 的布尔结果；不记录其他参数数据或设备地址。必须核实此 hook 对当次 PrivateUse1 实际分派和编译/capture 生效、每个目标调用恰好一次且不改变算子语义；`get_ascend_config()` 在外围单独读值仍不是该调用实参的直接证据。stage 上运行了纯 CPU、仅注册临时进程内 custom mutable op 的 [expert_snapshot_cpu_dispatch_probe.py](../../diagnostic-20261029/expert_snapshot_cpu_dispatch_probe.py) 同等代码，观测模式拿到 int 实参 17 且输出保留同一对象；这是**CPU 可行性类比**，不证明 NPU custom op、真实 binary、异步执行或 graph 路径可拦截。首次试验因 stage 的默认临时目录只读而失败，指定本任务已有 `diagnostic-20261029/cpu-analysis-tmp` 后成功；没有执行 NPU 工作。

`model_runner_v1.py:2377–2401,2551–2598` 在单次 `execute_model` 内确定 `cudagraph_mode` 和 batch descriptor、设 forward context 并调用 `_model_forward`；`_model_forward:3192–3251` 最终运行 `self.model`。**execute_model 返回只能标记一次 host 侧执行，不等于设备全部完成**，也不能假设每次执行仅有一个 graph、每个 graph 仅一次 MC2，特别是 piecewise 或 draft 路径。

实际 `ACLGraphWrapper.__call__`（`acl_graph.py:156–174,176–273,284–302`）以 forward context 的 runtime mode 和 batch descriptor 查 `concrete_aclgraph_entries`：entry 首次为空时进行 capture，已有 entry 时调用 `entry.aclgraph.replay()` 再返回 output。一个外围诊断 wrapper 可以候选地观察 mode、entry 的 capture/replay 状态并分配本进程单调序号；首次捕获时 Python 内层 hook 可构造层 → snapshot slot 映射，而 replay 只执行图内已捕获的 copy，**不依赖 Python 层 hook 再执行**。必须查证实际运行是否使用该 wrapper、是否存在嵌套/piecewise 或非目标模型同名 entry，并记录 entry 实例 token 与 capture 序号；任何无法唯一对应的情况不采样。

潜在的无 host 强制暂停设计如下，**仅为设计，尚无可运行的设备后端**：每个目标层在实际 capture 时各用一个预分配设备槽保存 count；每次 replay 提交后，在经验证的 replay 完成依赖所在流上按该 replay 序号把各层槽异步复制到独占 pinned-host ring 槽，并在这次复制之后 record event。后续 replay 如确实在同一依赖流上提交，则排在复制之后，不必等待 CPU；后台只有在 event 完成后读取，消费完毕且生命周期已确认才复用 host 槽。ring 满时跳过并记 dropped/unknown，不得覆盖未消费槽。该方案必须另外证明图跨流任务已汇合到 replay 流、图中拷贝未被删除或改序、D2H pinned/nonblocking 在该 torch_npu 版本受支持、事件有效且各调用只有预期次数；若不成立则拒绝或在明示干扰下用其他诊断方法。不要把 host 返回、event API 存在或一次成功读取当成这些证明。设备/host 拷贝会增加图节点及带宽消耗，新的采样时长不是旧基线时长。

本节定位的主要 source SHA-256：W8A8 dynamic `a67601ee596d9aa4ae72ea3638f67ff7236cd8df884564bcd19f33653a1e0910`；fused-experts input `02c344b5a234da681ef5d8d78c1666750f443c58ba2ca73766d76b350924a9fd`；上游 routed-experts `68c48e5df0ea4a7cc1f04436b7b9153eb53802ecd9bb55d38db8738508daac6d`；图 wrapper `8e1c269e331d005f1361acc5cec24402bff7ae39b75e952ea890da7812804955`；worker `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`。这些是 stage 当下源码文件哈希，尚不能单靠它们证明历史 profile 的每个 worker 已加载同一编译 binary。

### 可审查的 metadata wrapper 雏形

新增 [expert_snapshot_overlay.py](../../diagnostic-20261029/expert_snapshot_overlay.py)，默认关闭，导入时不加载 torch、不会自动挂接、不访问设备。它提供主模型目标层对象绑定、MC2 rank allowlist 和 HMAC 匿名标识；仅在隔离进程显式启用后，通过目标 `FusedMC2CommImpl.fused_experts` 周围的局部 TorchDispatchMode 观察精确 custom op 同次 `max_output_size`，并核对 count 输出是否为该通信对象的缓冲。主模型 root、进程 global rank 和 execution context 必须由独立且可信的调用方提供；其真实性和 thread/capture 适用性未由此草稿证明。未知层、rank 或 context 时跳过，op 未命中、多次命中或输出对象不一致时 capacity 记 unknown。

该草稿的每层 `slot_label` **只是未分配占位符**；不创建或读取设备缓冲，`count_status=unknown_not_sampled`，完成状态 unknown，真正 `request_device_snapshot` 始终拒绝。它不处理 replay：捕获时记录的 Python 信息不能被当成每次 replay 信息。显式启用会给本进程安装运行时 wrapper 和 dispatch mode、增加开销，甚至可能影响编译/capture，不能在基线服务中启用；当前没有安装或启用。

在自有 stage 容器，仅用 CPU synthetic custom op 与伪造对象、独立 Python 进程验证：默认不安装、目标层对象与 capacity 读取、非目标 rank 和 replay 跳过、原函数异常只执行一次、设备采样硬拒绝均通过。该检查不证明目标 PrivateUse1 算子、NPU stream、真实图及输出数值。下一步仍需在有授权的隔离诊断里先核实实际模型树、capture/replay 和 dispatcher 命中，再用专门设计验证 stream/event、预分配槽与图生命周期；在这些条件确认前不得启用任何真实 count 采样。

健壮性补充：任意 dispatch 入口的 schema 读取失败都将当前观测设 unknown，并将原 op 原样调用一次；观测逻辑不会重试已经执行或失败的 op。缺少 torch 时安装返回 false、目标类保持原样；匿名身份不明时仅透传原方法。mode 构造失败、尚未进入时可透传；但已检查该版本 `TorchDispatchMode.__enter__` 会在最终 push 之前修改状态，因此 **进入失败不能安全假定无副作用**：此时拒绝执行该目标调用并报错，隔离进程需退出，不做猜测性回滚。退出失败时目标方法可能已执行，绝不重试，报错并退出该隔离进程；如果目标原本已抛错，保留原异常并附加退出失败说明（运行时支持时）。这属于尚未启用的诊断草稿故障语义，不能当作对生产服务零影响的保证。

[expert_snapshot_overlay_cpu_test.py](../../diagnostic-20261029/expert_snapshot_overlay_cpu_test.py) 的 7 个仅 CPU、模拟接口的异常路径检查通过；另在 stage 的独立进程用实际 PyTorch dispatch mode 与 CPU synthetic custom op 验证正常路径仍返回原对象且获取容量。均未测试真实 NPU 模式或 graph，也未启动 NPU 工作。

## 新 profile 的 EP8、EP15 文件与 schema 定向清点（2026-10-29）

ZCode stage `operator-ceiling169-v2-210-recapture-expert-inventory` 对本次已保存的 EP8、EP15 目录作只读清点，分别找到 184、185 个文件；每侧 3 个 SQLite 数据库的**表名/字段名**，以及 `kernel_details.csv` 的 46 个表头字段，均没有命中限定词 `expert|moe|rout|topk|active.?mask|cumsum|dump|tensor|snapshot`。SQLite 使用只读 immutable 方式检查 schema，没有检查表内容、BLOB、WAL 或数据库以外文件的内容；关键字未命中不代表值一定不存在。EP8/EP15 各自的 `kernel_details.csv` SHA-256 分别为 `20cfb97abe03befda3fbf2c1194e48531aac44ce17d4f5fc130ea8ffb5261730`、`acd8612c67a3703f0dc69209bf25cbb5bab1216b55256d21a0082041b66e43af`，它们可用于绑定已记录的本 rank、第 14/39 层任务字段，不能凭 shape 推断 tensor 值。

**注意新增线索：**每侧均有 5 个 `host/data/aging.additional.tensor_info.slice_*` 和 1 个 `host/data/unaging.additional.tensor_info.slice_0` 原始数据片（另有各自 `.done` 文件），不是“未发现 tensor_info 文件”。本次只检查文件名和大小，未解码内容；尚未找到生产者／本次 exporter 格式与这些切片的对应证明，无法判定是否有可绑定的计数内容。此前引用的 CANN 9.1 beta2 `aclcppdevg_03_1367.html` 直接原文是 `aclprofAicoreMetrics`，并非 `aclprofTensorInfo`；已按官方目录修正为 [aclprofEventAttributes 原文](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/API/runtimeapi/aclcppdevg_03_1371_2.html)。它列出的 `aclprofTensorInfo` 与 `aclprofTensor` 字段包含 op/stream、输入输出类型、format、dtype、shape，未列出 tensor 元素；该结构和这个版本的官方说明不证明 EP8/EP15 切片按此格式生成或不包含其他数据。独立的 EP3 原始切片与当前安装解析器检查见 [tensor_info 契约](MC2-EP3-TENSORINFO-CONTRACT.md)。下一步若有匹配版本的格式/解析器证据，应先解析与目标调用关联的匿名字段类型，必要时只输出计数和其绑定信息，禁止无差别导出张量、请求、权重或文本。

两侧的目录范围和结果见 [recapture-expert-metadata-inventory.json](../../diagnostic-20261029/recapture-expert-metadata-inventory.json)，复现入口 `python3 diagnostic-20261029/audit_recapture_expert_metadata.py`，其通过 SSH 在专用 stage 容器**只读取已保存目录的文件路径、大小和 schema**，结果写本地任务目录；不采新 profile、不访问 NPU。ZCode 退出码 0、`model_verified=true`，模型 `deepseek-flash`。该检查仅涉及 EP8/EP15，不排除其他 14 个 rank 或未检查的其他存储路径。至此，第 14、39 层具体调用可可靠绑定的逐专家 count、实际 capacity 和对应有效行数仍未知。

### EP8 第 14 层单任务：采集接口与条件性数据流

既有 `diagnostic-20261029/target-layer-mc2-16rank.json`（SHA-256 `16d9e17986f59c4caeea1d0b1157f2d839b8344df0fc7c91e1b3105b612448a3`）记录 digest `1b826883...` 的 EP8、第 14 层 scope `737-20` 关联到 `DispatchFFNCombine` task `44217`，CSV 时长 **160365.956 µs**；同 digest、同 EP 第 39 层 scope `737-45` 的 task `45363` 为 **8290.414 µs**。两条导出接口字段相同，均为以下顺序：

| 位置 | 导出 shape | dtype / format | 与冻结 `dispatch_ffn_combine_def.cpp` 接口顺序相容的语义（条件性） |
|---|---|---|---|
| 输入 1 | [1023, 5120] | BF16 / ND | 输入 x |
| 输入 2、3 | [24, 5120, 4608]；[24, 2304, 5120] | INT8 / FRACTAL_NZ | 两组专家权重 w1、w2 |
| 输入 4 | [1023, 6] | INT32 / ND | expertIdx |
| 输入 5、6 | [110592]；[122880] | INT64 / ND | scale1、scale2 的编码槽，不能仅凭 dtype 解释数值编码 |
| 输入 7、8 | [1023, 6]；[1023] | FLOAT、BOOL / ND | probs、可选 active mask |
| 输出 1、2 | [1023, 5120]；[24] | BF16、INT32 / ND | routed 输出、专家计数 |

逐槽名称为**按冻结接口顺序和类型匹配的解释**，导出表本身并未证实每个 tensor 与源码参数或本次加载二进制的绑定；同形不同层也不证明 mask、专家有效行、路由或实际工作量相同。`FRACTAL_NZ` 为导出格式标签，不以显示的逻辑 shape 当作物理连续布局或实际传输字节。

若该样本执行保存的 custom INT8 路径，则 `moe_comm_method.py:580–614` 输入 hidden states、两组权重及 scale、top-k ID、概率、mask 等；AIV 内 `dispatch_ffn_combine_kernel.hpp:792–829` 做 mask、routing/quant、跨 rank 汇集计数；随后按专家和来源 rank 使用 peer memory 取相应行及 per-token scale，并发出 AIC 计算信号（约 :836–884）。AIC 的 `GMM1` / `GMM2`（约 :434–629）和 AIV `DispatchAndCombine`（约 :788–980）通过 cross-core flag 协作；AIV 在 GMM1 信号后进行 SwiGLU 并对 GMM2 发信号，再 combine、同步和按 `expandedRowIdx` / `probs` unpermute 得到输出。源码允许分组/分块交叠，不能将上述阶段耗时简单相加；具体内部阶段时间、真实 rank 组、有效 count 以及通信量均未采到。`routed_experts.py:750–779` 的条件分支可把 expert count 用于 EPLB 负载统计，`routed_out` 则经通信方法 `finalize`；不据此断言此样本启用了 EPLB 或观测到了这些消费。

**条件工作量模型，非严格性能界：**若确认导出矩阵维度确实对应该源码的 `problemShape.k=5120`、`problemShape.n=4608`，且每个有效专家行执行两次常规稠密矩阵乘法，令 `R=sum(effective[i])` 为按同次 `maxOutputSize` 裁剪后的本 rank 专家行总数，则 GMM1 和 GMM2 的逻辑算术量分别为 `2×R×5120×4608` 与 `2×R×2304×5120`，合计 **70,778,880×R** 次标量操作（一次乘加计两次），不含 routing、量化、激活、通信及填充。`R`、实际执行分支、硬件计数、重用和实际实现均未知；该式不是实测 FLOPs、不是证明不可避免的操作数，也不能据此给吞吐硬上界或推断实际可达提速。


## 当前 CANN 安装的 dump 接口：只读检查（2026-10-30）

169 的用户 stage 容器内，`cann-9.2.0/aarch64-linux/ascend_toolkit_install.info` 标记 version 9.2.0；`include/aclnn/acl_meta.h:118` 声明 `aclDumpOpTensors`。`nm -D --defined-only` 对 `libnnopbase.so` 检查发现该符号导出；此前只检查库名包含 acl/opapi 的 33 个对象没有找到，不能据此推断不存在。stage 该库文件 SHA-256 `ab8145a5f6a7c3aa4f1e274263f206dcf4a8737e2266908112bea49d2b3a4534`。在自有服务容器只读查看 `/proc/<pid>/comm` 和 maps，16 个 `VLLM::Worker_DP` 都有 `libnnopbase.so` 映射，映射路径对应文件的 SHA-256 与 stage 文件一致；没有读取 tensor、注入进程、启用 dump 或运行 NPU 诊断。映射和文件内容一致不能证明内存映像没有变化、调用已启用、参数及 graph 的行为，也不能证明对运行服务安全。

因此比“头文件声明但未定位实现”更进一步：隔离诊断可以把这个函数作为候选入口；只导出计数的输入数为零是否支持、图 capture/replay 语义、数据覆盖前的 stream 依赖、输出与 rank/layer/replay 匹配及运行扰动仍须验证。不能借此注入当前服务。官方接口所需 dump 开启及进程内调用的限制见 [aclDumpOpTensors](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta3/API/aolapi/operatorlist_00178.html)。


## 最小 dump 与 graph 验证（设计，尚未执行）

官方 [CANN 9.2 图捕获说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta1/others/acldevg/aclcppdevg_000045.html) 指出捕获期间 stream/event/device/context 的查询或同步在任何捕获模式下均不合法；全局捕获模式下同步内存操作也不合法，放宽模式不意味着它自动进入图或不会影响捕获。不能用捕获阶段同步 host 读取代替 replay 的计数。

在独立进程、空闲隔离 NPU 中，以固定地址的 24 项 INT32 数据为目标，先在 eager 下验证仅输出 tensor 的 dump 返回码和文件内容；再将写入及 dump 纳入一次 graph 捕获，针对 2–3 个不同的输入各自重放，检查每次文件、数值和 replay 标识是否一一对应。捕获报错、不逐次落盘、结果覆盖、值/回放无法绑定，均不能判定该接口满足需要。这个小例即使通过，也不证明真实 MoE 中输出覆盖、跨流同步、组身份和性能扰动已解决。现有服务不注入、不重启。
