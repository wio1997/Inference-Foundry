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
