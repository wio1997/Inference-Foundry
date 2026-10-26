# M023 / M024 / M025：NPU index cache、PD 页跨度与接口合同

本轮核对了实际安装 SGLang `ce0a8fea5f4245839d174b2b8c55692be1810a7f` 的 NPU pool 选择和量化条件，并完成 **108 项源码提取模型检查**。确认 compact index-K 层布局的门控排除本任务要求的 PD；确认 PD transfer view 可改变每个传输项的页跨度，同时保留 tensor 的地址和总字节数。

检查使用实际镜像的 Torch 2.10 CPU/meta tensor。没有完整导入 NPU pool 或 torch_npu 模块，没有执行 NPU 构造器、量化/写 cache、原生 indexer、通信或模型生成。整体 serving 保持 E0，已接受 PD+MTP baseline 为空；没有生产修改或性能结论。

## 当前路径与证据层次

| 证据 | 已知 | 未知 |
|---|---|---|
| R001：五个规范源码文件 | NPU pool、configurator、KV dtype resolver、通用 index-cache facade、model runner 与安装 Git HEAD 一致 | 本任务实际 pool 对象、dtype 和分配容量 |
| R002：安装文件扫描 | torch_npu 2.10.0.post6、sgl_kernel_npu 2026.9.0；发现 22 个相关源码/头文件候选 | 包 Git、实际加载的注册表/共享库、native 内部分支 |
| R003：108 项检查 | 未改动函数体的 Meta 形状、SGL 返回表达式、compact gate、PD view 行为 | 原生支持、索引值、量化误差、cache 连续性、传输、接受率和性能 |

R002 只通过 distribution RECORD 和两个明确的可选 CANN header 目录读文件：扫描 2439 个文本文件、12,748,279 字节；一个超过 2 MiB 的文件未读，没有达到总预算而跳过的文件。没有导入 torch_npu/sgl_kernel_npu。命中文件和头文件存在不证明某个 kernel 已实际运行。

[source-index.json](source-index.json) 保存七个相关 SGLang 文件的规范哈希；[native-interface-index.json](native-interface-index.json) 保存全部安装候选哈希、扫描限制、schema 和相关 Meta 函数摘要。大型生成文档与无关头文件保留本地哈希，不放进公共证据包。选取的 Meta 函数原文仅用于审阅，完整输入文件可按固定镜像中的路径与 SHA256 恢复。

## M025：PD 排除 compact index-K 层布局

`kv_cache_configurator.py:107` 的 `_should_elide_dsa_index_k` 要求：target worker、未启用 HiSparse/HiCache/external linker，且 `disaggregation_mode == "null"`。上层 `_build_ascend_mla_kv_pool` 还要求 DSA 与 arch35，才会只为不跳 Top-K 的层建立紧凑 indexer_layer_ids。

R003 对 null/prefill/decode 与四个布尔条件做 48 组 getter 夹具检查，仅 null、target、三个 cache/linker 标志均关闭的一组返回 True；32 组 prefill/decode 全部 False。**不能把现有单引擎 compact index-K 层节省直接计入 PD。** 这是当前源码门控的适用边界，没有提出放开门控的优化。更改它必须先证明 PD 的层/条目配对、draft cache 和 transfer groups 完整。

## M023：pool 条件决定量化 indexer，文件名不能决定

Ascend MLA 分配器建立 `NPUMLATokenToKVPool`。packed DSA FP8 storage 的上层条件包含 KV dtype=float8_e4m3fn 与 arch35；pool 内还要求 index_head_dim 非空、requested kv_cache_dim 非空和相应尺寸约束。仅在该 packed 条件成立且 indexer 层数大于零时，分配 FP32 index-K scale buffer，并生成 Hadamard matrix。

`DSANPUIndexerMixin.forward_npu` 根据 `pool.index_k_scale_buffer is not None` 选择 quant/nonquant lightning indexer；此前会计算 index-K 并写 cache。这里不能从 w8a8c8 目录名或配置的 bfloat16 字段推出实际 branch。权重量化、激活量化、KV 编码、index-K 编码和动态 scale 是独立问题；本轮没有验证 M023 原始候选“缓存静态派生 scale”的收益。

固定 config 提供 rank=512、RoPE=64、index_head_dim=128，作为形状来源。条件编码字节账如下，**并非本任务实际运行 dtype 或同语义优化收益**：

| 每个物理 slot / 层的内容 | BF16 DSA 夹具 | packed FP8 DSA 夹具 |
|---|---:|---:|
| latent 与 RoPE | 512×2 + 64×2 = 1152 B | 512 + (512/128)×4 + 64×2 = 656 B |
| index-K | 128×2 = 256 B | 128 B |
| index-K scale | 0 B | 4 B |
| 合计 | 1408 B | 788 B |

该表忽略 allocator/registration 元数据开销，且 DCP 下 target KV 与 global index 的 slot 数不同，不能用单一 token 容量直接相乘，也不能将它换算成加速。

## M024 / M035：kernel view 与 transfer view

R003 提取 NPU pool 的三个未改动方法：`_get_disagg_buffer_view`、`get_contiguous_buf_infos`、`get_dcp_remote_decode_layout`。对象与 tensor 分配是明确夹具，构造器和 `_raise_if_native_kv_cache_disabled` 边界没有运行；后者用 no-op 边界替代。其余方法体未变。

18 组覆盖 target/draft、DCP=1/2/4、MLA BF16/DSA BF16/packed DSA FP8。使用 page=128、容量 512 token 的形状夹具，target 两层、draft 一层。以 config 尺寸建立 tensor：

- target K/V 使用 rank-local slots；draft K/V 与所有 index/scale 使用 allocator-global slots。
- global entry 的 transfer page 有 `128 × dcp_size` 个 slots；local entry 仍为 128。
- view 的 data_ptr、storage pointer 和 nbytes 均保持；item_lens 随 transfer page 变化。
- BF16 DSA 注册 K、V、index-K 三组；packed FP8 DSA 注册 packed K、index-K、scale 三组。
- page 维度不符、global page 数不能整除的两个人工错误夹具被拒绝；不是观测到的运行错误。

示例为 **target、两层** 的真实 CPU tensor 字节数，容量与布局仍是夹具：

| DCP | BF16 DSA 注册总字节 | packed FP8 DSA 注册总字节 | index-K 每个 transfer item：BF16 / FP8 |
|---:|---:|---:|---:|
| 1 | 1,802,240 | 1,008,640 | 32,768 / 16,384 B |
| 2 | 2,129,920 | 1,177,600 | 65,536 / 32,768 B |
| 4 | 2,785,280 | 1,515,520 | 131,072 / 65,536 B |

总字节随 DCP 变化来自所构造 global index pool 的容量与 padding；view 本身没有分配或复制。它也没有证明原生传输会少搬这些字节。完整登记记录见 [source-model-result.json](source-model-result.json)。

## Native 接口与 Meta 合同

安装 schema 中，非量化 `npu_lightning_indexer` 返回 `(Tensor, Tensor)`，量化版本返回单个 Tensor。SGLang 分别以 `[0].squeeze(1)` 和 `.squeeze(1)` 适配。非量化默认 return_value=False 时，Meta 的第二个 tensor 为空；这与部分生成 docstring 的单 Tensor 简写不同，应以实际注册与调用再做原生验证。

40 组检查覆盖 batch 2/3、每请求 query 行 1/4、L=512/2047/2048/2049/32768、quant/nonquant。提取两个原始 Meta 函数，移除注册 decorator，保持函数体 AST 完全一致；执行的是实际 Torch 的 meta shape 操作。SGL 返回适配表达式也直接提取，结果均为 int32 `[T,2048]`。query heads=2 是夹具，不是实际模型分片后的 head 数。

Meta 函数使用 sparse_count 建输出宽度，并未根据 actual sequence lengths 缩短 K。**固定输出宽度不证明原生做完整 Top-K，也不证明存在短路径。** Meta 不计算索引值，不覆盖原生参数验证、mask/tie、padding、workspaces、launch 或时间。GE converter 和 CANN API header 的存在也不证明当前运行使用 GE 或该 API 实现。

## 最小原生合同与否定条件

[native-next-contract.json](native-next-contract.json) 延续 PD+MTP、batch 2/3、固定 K=2048 与真实输入/接受分布。需要任务资源后核对实际 pool 类、target/draft KV dtype、scale buffer、arch、compact ids 和双方登记条目；依次校验 index-K 写入与 seed 复用、PD 内容、最终输出，再做同条件 AA/配对 trace。

当前没有优化 candidate。view 已是别名时，不重复“消除 copy”；compact gate 在 PD 不成立时，不把其潜在容量收益当成可用；改 KV dtype 会改变编码/误差，必须作为独立配置研究，不能冒充等价实现。原生 trace 若证明 native 已覆盖短路径，就停止重复实现该路径。任何索引、scale、输出或接受分布不符合合同都否定候选。

## 可追溯性与结论

三个 frozen commands 均由 ZCode 精确会话执行一次；[zcode-audit.json](zcode-audit.json) 区分配置别名 deepseek-flash 与 UNKNOWN 后端发行版，未发布模型 I/O 或私有设置。R003 CPU 容器无网络/设备、只读 root、1 CPU/4 GiB、仅四个 metadata/tokenizer 文件只读挂载，无权重；已核对身份后清理，观测服务 ID/启动时间/restart_count 不变。没有模型生成请求。

Loop 为 INCONCLUSIVE：源码与 CPU/meta 模型通过，完整模块、实际 native allocation/registry、cache/PD 正确性和物理性能仍未证明。L017 的完整状态模块 UNAVAILABLE 记录维持，不因本轮提取函数通过而消失。下一步先覆盖尚未审计的 MoE 路由/dispatch，再在明确分配的 Ascend 资源上回到原生验证。
