# YOCO 元数据、缓存与注意力对照

日期：2026-10-30。参考实现为 DeepInfer 提交 [`553741557a57df44a46fb3ab72b6aafeaf945337`](https://github.com/deepseek-ai/DeepInfer/tree/553741557a57df44a46fb3ab72b6aafeaf945337)。910C 测试针对 172 上保留的 W8A8 实现；本阶段没有替换线上执行路径。

## 对照范围

官方 [`prepare_tensor_input_v4.cpp`](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/csrc/engine/input_metadata/prepare_tensor_input_v4.cpp) 提供 `mask` 和 `compact` 两种元数据生成方式。对每个查询，可见 SWA 范围受滑动窗口、序列总长与保留长度构成的矩形下界、有效缓存起点共同限制。`compact` 删除矩形外的查询，但在完全位于矩形之外的分块中保留一个占位查询，且不将占位查询写入 ring。

当前 910C 有界重放仅在四个请求、已恢复前缀、最终分块结束位置相同且按 128 对齐等条件下进入；不满足条件时回退原路径。官方 CP 测试参考采用镜像布局，910C 使用连续布局。布局不同，需要在逻辑顺序及多卡执行结果上分别验证。

## 已完成的检查

- 使用官方 C++ 元数据实现和上游 `yoco_equiv.py mask-compact`，保留长度 140、256 各完成 10 个官方测试场景。测试覆盖短序列、早期分块和混合长度；两条路径在测试定义的共享行上通过。
- 在 6 组可进入 910C 候选的合成输入中，对照查询行、序列边界、位置、压缩比例 1 和 2 的逻辑边界，以及 CP 重建后的逻辑顺序。3 组短查询明确回退。910C SWA 可见范围在此项中来自接口约定推导。
- 在 16 组边界输入中调用官方元数据和 910C 实际入口判断。短查询、提前分块、末尾未对齐、不同结束位置、前缀未恢复、零命中及正常 EOS 等情况均记录为 910C 候选回退；不将回退等同于官方 `compact` 已接入。
- 显式设置非零 `swa_valid_start` 后，官方 `mask`/`compact` 在测试共享行保持一致。短查询 64 行的可见范围均受该边界影响；最终分块且查询长度不少于保留长度时，矩形下界不小于有效起点，测试中的 256 和 320 行查询均未改变。短查询目前在 910C 候选中回退，回退路径尚未验证。
- 独立 `mask` 诊断候选保留完整查询，按各请求原始结束位置生成矩形可见范围。CPU 合成测试对 8 个 CP rank 的物理索引逐行核对官方元数据，140 和 256 两组通过；这不代表多卡通信已经执行。
- 对相同保留查询和压缩索引，单设备 `mask` eager 路径与先前 `compact` 算子结果四组最大绝对差均为 0.001953。重新读取设备张量，以官方元数据和 CPU eager 重算，最大差均为 0.000488。两次设备运行分开进行，丢弃行使用合成零查询，不能替代完整模型对照。
- 使用官方生成的写入槽位和合成缓存负载，对比 `mask`、`compact` 与 910C 快照恢复后最终 128 token 的 SWA 内容；140 和 256 两组通过。此项在 CPU 上调用 910C 的实际缓存和快照方法，Engram 历史与 DSpark 状态使用桩对象，只检查复制流程。
- 使用官方真实元数据及 CPU 独立注意力公式，对照单个 910C 设备上的 BF16 注意力算子；140/256 与压缩比 1/2 四组通过。保存张量后重新生成 CPU 参考并独立复核。各组最大绝对误差分别为 0.001085、0.001070、0.001291、0.001192；预设阈值为 0.01。

- 对真实模型和前缀快照增加恢复后检查：16 个 rank 各检查一个槽位，恢复的 40 层 SWA、4 组压缩 KV、索引键和缩放、3 组 DSpark 缓存与对应快照在数值比较中一致；位置和 Engram 历史一致。8 个请求的前缀实际命中均为 29,568 / 30,336 token，生成的各 16 个 token 与先前 mask 参考一致。该检查只比较快照和恢复目的地；浮点 NaN 两侧均为 NaN 时视为相等，不是逐位检查，也不验证生成快照的上游语义。仅一个前缀、每 rank 一个槽位，诊断包含同步，不能用于性能结论。审核文件 `evidence/yoco-prefix-real-audit-20261030.json` SHA256 `59e9b1078b478782ef6e9b5cc91d1e747f9df355c443e465e1a10dc0c20bce74`。

## 短末块的回退可见范围

对四个相同终点 1,024、已恢复前缀、`ignore_eos=true` 的合成请求，使用官方 C++ 生成 `mask` 元数据，并调用保留版本实际的 `eligible` 判断。所有测试查询长度均不大于保留长度，因四个查询均未超过保留长度而回退。对照侧的完整滑窗范围由 910C 调用参数 `ori_mask_mode=4`、`ori_win_left=127` 推导，未在本检查中执行 NPU 算子；缓存有效起点设为 0。

| 保留长度 | 每请求末块查询数 | 可见范围不同的查询行数 | 单行最多多出的完整窗口键数 |
| --- | ---: | ---: | ---: |
| 140 | 13 | 0 / 13 | 0 |
| 140 | 64 | 51 / 64 | 51 |
| 140 | 140 | 127 / 140 | 127 |
| 256 | 129 | 0 / 129 | 0 |
| 256 | 200 | 71 / 200 | 71 |
| 256 | 256 | 127 / 256 | 127 |

对本组条件，完整窗口包含官方 mask 的所有可见 SWA 键，并可能包含更早的键。官方 mask 是有界近似，完整窗口回退不是据此判定的错误；但两者的中间可见范围不能称为相同。本结果不覆盖非零有效起点、不同终点、压缩 KV 索引、设备调用及完整模型输出，也不能外推到主性能测试中约 3,200 token 的末块。候选资格和参考计算证据为 `evidence/yoco-fallback-visibility-review-20261030.json`，SHA256 `e6c7f6e505bb440075217899f9584b4f7c8bd9827a15b841edc24f098f2894b0`。首版脚本误将超过 keep 的查询归入必然回退集合，断言失败并保留；修正后仅分析实际回退的短查询。

### 910C 单设备算子核验

使用两个合成请求、各 200 行查询、保留长度 256，在一张 910C 逻辑设备上分别调用现有 BF16 SparseFlashMla：完整窗口使用 `swa_base=0`，官方 mask 窗口使用页对齐的 `swa_base=768`。两次调用使用相同查询、原始 KV、压缩 KV、压缩索引及 sink；官方 C++ 元数据提供 mask 可见范围。该测试中 400 行有 142 行的窗口不同。

| 压缩比 | 完整窗口与各自 CPU 参考最大绝对误差 | mask 窗口与各自 CPU 参考最大绝对误差 | 两种窗口输出的最大绝对差 |
| --- | ---: | ---: | ---: |
| 1 | 0.000328 | 0.000426 | 0.091064 |
| 2 | 0.000332 | 0.000379 | 0.075806 |

窗口相同的 258 行，设备输出逐位一致。独立进程重新生成官方元数据及 CPU 数学参考，两种窗口均通过最大误差 0.01 的预设阈值。这证明现有绑定在该形状下可以表达相应窗口，且该样本中的窗口差异会影响注意力输出；不能据此推断完整模型准确率或性能，也不能证明其他布局和缓存状态均受支持。实验未执行官方 NVIDIA 注意力内核。

原始结果 `evidence/yoco-short-fallback-npu-20261030/result.json` SHA256 `ab56e8fc4a7e37e5d39e995a8ad3ae5f66b1e4737bf81bce694ca07eea29d113`；独立复核 `evidence/yoco-short-fallback-independent-20261030.json` SHA256 `c45e00c877a9beaf2dbe0fe388f37d9234d7f434eeefa8a489e38928da91c47a`。当前保留版本没有更改。

### 真实模型首个 consumer 层抽查

独立候选对一条重复长输入进行 16 rank 检查，从首个 consumer 层每 rank 抽取 3 行：关闭与开启短末块 mask 的查询、压缩索引和对应 SWA、压缩 KV 样本逐位一致。两侧真实 910C 注意力输出分别与使用官方 C++ 元数据及独立 CPU 公式计算的参考比较，最大绝对误差均为 0.008895，限值 0.01。两种可见范围输出最大绝对差为 1.941406；窗口不同，不能直接视作相同计算的误差。8 条请求各 16 个输出 token 相同。详情及限制见 [短末块候选](../20261030-yoco-short-mask/README.md#真实模型中间张量抽查)。这不是官方 NVIDIA 内核对照，也不证明其他层或完整模型已对齐。

## 主负载 compact 真实张量抽查

使用冻结主负载第一轮的 8 条不同请求和两条原前缀种子：16 个逻辑设备、DP2×TP8/EP16、输入 32,768 token、并发 8、实际缓存命中 235,904 / 262,144。此处生成 16 token，属于带同步插桩的语义诊断，不是 1,024 token 性能测试。

分别使用两个 DP 的实际命中长度调用官方 C++ 元数据：mask 分别生成 13,184 和 13,056 个查询行；compact 各保留 1,024 行。compact 保留行的 SWA 可见范围和压缩边界各为 1,024 / 1,024 与 mask 相同。元数据预检 `evidence/yoco-main-compact-metadata-preflight-20261030.json` SHA256 `9fd3d6b554a3b44507405661b13babc9afb589691388dc0688f327cc6e95317f`。

在保留版本的独立插桩副本中，每个 rank 抽查首个 consumer 层 3 个实际位置，共 48 行。按官方元数据筛选所记录的 SWA 值，以设备真实查询、所选压缩索引及对应 KV 和独立 CPU 注意力公式复算：最大绝对误差 **0.011561**，超过预设限值 0.01；48 行中 1 行超过限值，各行平均绝对误差再取平均为 0.000294。该项尚未通过预设数值验收，不能仅凭平均误差或生成文本一致将其视为通过。8 条请求的前 16 个生成 token 与冻结参考一致；参考来自此前另一服务运行。

对最大误差行补充 FP64 CPU 复算：FP32 与 FP64 参考最大差为 0.00000125；最大误差点的设备输出为 2.296875，FP64 参考为 2.308436，该数值附近一个 BF16 间隔为 0.015625，误差约为 0.74 个间隔。该行 32,768 个元素中 1 个的绝对误差超过 0.01。这个结果提供了数值尺度，但不能确定设备误差根因，也不能替代原先的绝对误差门槛；复算记录 `evidence/yoco-main-compact-precision-review-20261030.json` SHA256 `705ff34a5efb1fbbc8ae442d8813c3e5260feef8a43f859048ae08fd3ee42f95`。

补充按现有 910C 参考测试中使用的 `atol=0.008、rtol=0.008` 统计同一批数据，1,572,864 个元素均在该组合范围内，最大误差与对应范围之比为 0.437。该测试文件是 910C 适配测试，不是官方 YOCO 标准，且测试的不是这里的同一 CANN 算子；这是事后补充的数值背景，不替代预先设定的 0.01 绝对误差验收。证据 `evidence/yoco-main-compact-trace-tolerance-context-20261030.json` SHA256 `bc8d71522c10b8f2beab8730354f4144c0319a48f45c3674497fb9422fc0886e`。

该抽查没有执行官方 NVIDIA 注意力内核，也没有独立验证索引选择本身、全部缓存、后续层或模型质量。实际捕获 3 行 / rank；插桩同步改变时序。独立复算 `evidence/yoco-main-compact-trace-tensor-audit-20261030.json` SHA256 `980d2b36f4b8fa8e3bf3425f2b59ab6b8c04a038e7b54ae08b488e9cb93919eb`；生成输出比较 `diagnostics/yoco-main-compact-trace-test-20261030/model-comparison.json` SHA256 `986ba7d6edec876ee9dbe0a92d2f77efb0dca7629a496cc4939dc99fa70bde71`。原服务已恢复，现有路径未替换。

## 索引器格式和当前接口限制

固定官方提交的 [V4 索引器](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/model_executor/models/v4_modules/index_attn.py)对索引查询和键应用 RoPE 及 E2M1/UE8M0 FP4 量化后计算索引分数。官方调用使用的融合函数名称包含 Hadamard，但这两处调用没有传入可选的 `hadamard_args`，融合函数仅在该参数非空时启用该变换。当前 910C 的 `w8a8_indexer910c.py` 对索引查询和键执行 RoPE 与 INT8 量化；调用 `dsa910c.py` 的量化索引接口，使用 INT8 Q/K、FP16 缩放和 `quant_mode=2`。索引 Q/K 格式不同于模型权重的 W8A8 配置。

固定镜像内的 `quant_lightning_indexer_v2/docs/aclnnQuantLightningIndexerV2.md` 在 A2/A3 部分列出 `quant_mode=2`、INT8 Q/K 和 FP16 缩放，明确不支持该接口的 FLOAT4；同目录算子注册和 tiling 源码也限定对应架构使用 INT8。镜像中注册的 `npu_quant_lightning_indexer_v3` 绑定及包装源码调用该候选索引器接口。镜像源码与实际运行的编译二进制尚未逐项建立对应关系；这是**当前接口**的限制，不能扩展为所有 910C FP4 实现均不可行。模型服务未修改。

因此，目前只能对照同一 910C 算子消费的查询、索引和可见范围，不能声称索引选择已与官方 FP4 路径相同。不同量化可能改变 top-k，影响尚未实测，也不能据此归因已有后续层差异。后续应在独立候选中采集相同查询、索引键和权重，先用可信的官方格式参考比较分数及 top-k，再判断是否有局部实现价值；须保留现有 INT8 路径并验证完整模型质量和端到端性能。

审计 `evidence/yoco-index-fp4-constraint-20261030.json` SHA256 `96d5359dbfc6a13da0524b07ca05b807ebcabf2853a0b89f3434e73f4567b774`，记录上游文件 Git blob、910C 文件及镜像内文档和源码的哈希。该初版审计曾错误地从函数名推断启用 Hadamard；后续核查调用和默认参数后已更正，修正证据 `evidence/yoco-index-hadamard-correction-20261030.json` SHA256 `64b5c78e2d9ed4ececd4ba0ca1699e4b1a8d4199ee081db9ed985a688616f0c7`。

## 环境差异与限制

直接构建官方参考时，GCC 11 缺少 C++20 `<format>`。隔离参考构建仅将日志和错误格式化接口替换为镜像内的 fmt。隐藏全部 NPU 后，原有固定内存分配触发 ACL 初始化错误；为运行 CPU 元数据参考，隔离构建将两处固定内存分配替换为普通 CPU 内存。官方 YOCO 元数据核心源码未修改，SHA256 为 `93a6b1dd533db116d2f7540c78d6bb5419bdb8c1d2063b031ab8b1a8cc138fe2`。这不证明固定内存在有 NPU 可见时无法使用，也不验证异步传输行为。测试构造器未对 Python 暴露 `SequenceMetadata.swa_valid_start`，尽管正式调度会从 Sequence 传递该字段；为注入测试值，隔离参考构建只增加该已有字段的 Python 绑定，未更改元数据计算。

单设备注意力对照使用官方元数据和独立 CPU 数学参考；没有执行官方 NVIDIA 注意力内核。上述检查尚未覆盖其他真实 NPU 前缀及所有槽位、零状态重算、DSpark 回退、多卡通信的完整语义和长输出，也没有测得该对齐工作的端到端收益。现有 256 路径及其性能数据继续保留，不能据此宣称与官方完整对齐。

## 证据

全部原始文件位于 172 的 `/data/tiankuan/wio/910c-deepinfer-172/evidence/`：

| 文件 | SHA256 | 内容 |
| --- | --- | --- |
| `yoco-official-910c-metadata-20261030.json` | `481854bf4597aee6285bc7d4931ec25cd0c0235bc11a8c37b13faffd530228d7` | 合成元数据、CP 逻辑顺序对照 |
| `yoco-official-910c-boundaries-20261030.json` | `aea6395fffeea855e2dae98997e36eff46e6259e4a3dfac8306e91b7e26775e3` | 16 组入口边界 |
| `yoco-official-valid-start-20261030.json` | `73bfb9ee95526d462d75247929758f3b6cfe23914ead3ea713c305cac0f2e4b1` | 非零有效起点与可见范围 |
| `yoco-mask-diagnostic-layout-20261030.json` | `e90cf2dde1e2cb1abd63b7f07c9a8c58b6a9f385107648f7905ee38d9b29c484` | CP 逻辑切分及官方 mask 索引 |
| `yoco-mask-diagnostic-attention-independent-20261030.json` | `621ecee3b3039c48e440eda1f72375a72ac516978bf57e85fbe893b5b02f2c98` | mask 单设备张量的独立复核 |
| `yoco-official-910c-cache-payload-20261030.json` | `d87029679cf58ec7d5dacd8154d3b7f2ef4c66f53f58ed4468cd8d959ae05474` | CPU 合成缓存及恢复 |
| `yoco-official-910c-npu-independent-20261030.json` | `8bd273dfbbcc48415a411bbd2157ab8f201f04494a8aaf4a76af5c1f6439b094` | 单设备结果的独立复核 |

算子输入输出和官方中间元数据保存在 `yoco-official-910c-npu-20261030/`。参考构建和官方测试记录保存在 `yoco-official-metadata-build-20261030/`、`yoco-official-cpu-allocator2-20261030/`、`yoco-official-cpu-l1-v2-20261030/`。
