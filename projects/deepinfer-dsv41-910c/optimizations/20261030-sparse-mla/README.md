# SparseFlashMla 局部适配评估

核对时间：2026 年 10 月 30 日。该评估只涉及当前 910C W8A8 部署，不修改现有代码或缓存格式。

## 现有路径

服务器保留版本的目标注意力 `deepinfer/npu_ops/dsa910c.py` 和 DSpark 草稿注意力 `deepinfer/model_executor/models/dspark910c.py` 均调用固定镜像的 `torch.ops._C_ascend.npu_sparse_flash_mla` 及其 metadata 接口。前者使用 BF16 的 SWA 和压缩 KV，128 token SWA 窗口；后者使用显式索引实现非因果草稿注意力。两者的调用参数不同，不能仅因算子名称相同就认为所有执行细节一致。

当前固定镜像的 `vllm_ascend_C` 扩展 SHA256 为 `235bc0205c0dcd24daf6a2649ede9a73d6ca05c9946c0447f5da5ad8f92dcee7`；两个调用文件的 SHA256 分别为 `f849edeaeb14f10428c52b715f5ed8c4a812aac6c9379b416e430f743912cd89` 和 `ddf74a353df214718af3190cb2bd2125e242ab59783161b692e7c7f24321d151`。镜像内未发现 `cann_ops_transformer` Python 模块。核对输出见本机 `sparse-mla-current-inventory-20261030.output.txt` 和 `sparse-mla-current-contract-20261030.output.txt`。

## 相关新实现

- [vLLM Ascend PR 17354](https://github.com/vllm-project/vllm-ascend/pull/17354) 已于 2026-10-15 合入主线；[PR 18808](https://github.com/vllm-project/vllm-ascend/pull/18808) 于 10 月 17 日合入 0.30.0 发布分支。它们针对 FP8 KV 路径接入 `mixed_quant_sparse_flash_mla`，并改变压缩和 SWA 缓存布局。两次合入均晚于当前固定 0.30.0rc2 的发布日期；不能将其视为当前镜像已具备该实现。
- [PR 19256](https://github.com/vllm-project/vllm-ascend/pull/19256) 仍为未合入提案，针对 A3 的 BF16 KV DSpark 草稿路径使用同名编译绑定，并处理参数和图填充边界。其适用范围和测试条件不能直接转为本机性能结论。
- [PR 19989](https://github.com/vllm-project/vllm-ascend/pull/19989) 为未合入的适配器职责重构，描述的是整理草稿参数归一化，并未给出本机可套用的加速结果。

## 价值与边界

当前路径已经调用 BF16 `SparseFlashMla`，因此简单重复接入同一个绑定没有明确增益。新的混合量化实现可能影响 KV 存储和计算数值；权重 W8A8 与 KV 精度是不同配置。现有镜像缺少其 Python 模块，且官方资料尚未证明该组合可直接用于当前 910C，因此不能直接作为现有 BF16 路径的等价替换，也不能仅凭发布说明估算收益。

下一步优先检查当前目标和草稿注意力的图填充、空序列、索引与 metadata 的接口边界，并从现有轨迹确定注意力在本工作负载中的实际开销。若发现可单独修复的接口或执行问题，在独立候选中保留 BF16 缓存和原注意力范围，先比较算子输出与图重放，再验证完整模型正确性和端到端指标。若评估 FP8 KV，必须单独记录缓存语义和数值变化，并以更广泛输入、缓存恢复与模型质量测试评估；不能用少量生成文本一致证明完全等价。

## 已有设备轨迹中的开销

复核既有 rank 0 轨迹：输入 32,768 token、并发 8、输出 32 token；选取 model ID 49 的 14 个 profiler 窗口，每个窗口均有 40 次 SparseFlashMla，事件累计时长中位数 2.032 ms，范围 2.020–2.037 ms；同窗口等待时长约 49.6–65.0 ms。每窗口另有 3 次 SparseFlashMlaMetadata，累计中位数 0.270 ms；8 次 QuantLightningIndexerV2 累计中位数 2.502 ms。索引算子与 YOCO 索引预处理是不同机制。

这些数值来自一次短输出采集，存在 profiler 警告；设备事件可能跨流重叠，窗口身份也不等同于完整端到端生成步骤。因此它们不是关键路径占比或替换算子的预期收益。它们支持先确认调用语义、图填充与缓存接口，再决定是否投入新算子适配，而不能根据官方峰值数字推断本部署收益。审计文件 `evidence/sparse-mla-existing-trace-20261030.json`，SHA256 `cd7e0a4c8fd8c40cba50f09da7974bd307e41eb638b14b6bf6cb46e4830d2c28`；源轨迹 SHA256 `e83474741f52e503e6955c704e0d1102835804cd2566a7ab48064af0eaa0ed17`。

另核对 [DeepSeek FlashMLA 官方仓库](https://github.com/deepseek-ai/FlashMLA)：其公开的 Ascend 算子性能数字和构建支持列明的是 Ascend 950、CANN 9.2 及以上；该结果不能直接移用于本机 910C。当前草稿使用的 A3 编译绑定与 [vLLM Ascend 的 A3 草稿提案](https://github.com/vllm-project/vllm-ascend/pull/19256) 描述的绑定同名，具体接口和版本仍需逐项核验。上述性能与兼容性限制不等于证明 910C 无法另行适配。

## 草稿调用的边界核对

核对截至 2026-10-30 的 [vLLM Ascend PR 19256](https://github.com/vllm-project/vllm-ascend/pull/19256)，该提案仍未合入，head 为 `b50fd0724ec9d7e8bb5a8ae8d10344a348dffe07`。其 A3 路径限定 BF16 KV，显式处理图填充产生的零长度请求和未初始化索引尾部；同时在相关 context parallel 配置下禁用该路径。这些条件属于该提案的实现边界，不能直接推断另一框架的并行布局相同。

当前保留代码的草稿图按实际 slot 数构造，每个 slot 固定 5 个 query；序列长度以 `starts+5` 给出；显式索引宽度 192，超出可见范围的位置写入 -1，并传入有效长度。目标侧和草稿侧均使用 `cmp_ratio=0` 表达不带压缩 KV 的分支，符合当前镜像的实际 ABI。按这段代码静态分析，它没有直接生成该提案所述的零 query 填充 slot，也没有以未初始化的尾部构造显式索引；该判断只覆盖当前固定草稿图，不能推广到动态批处理、异常位置或其他图。

当前服务启用了自有 context parallel 路径，但其组划分与草稿复制方式尚未与该提案逐项对应，不能仅凭环境变量认定兼容或不兼容。现有 NaN 在带额外同步的重复诊断中未复现，也不足以将其归因于这里的填充问题或 SparseFlashMla。现阶段没有证据支持直接替换草稿接口；后续若扩展到动态 slot 或空请求，应先按实际形状核验有效长度、-1 尾部和并行组，再做单因素数值与性能测试。

## 混合量化路径中的同步和布局

[PR 17354](https://github.com/vllm-project/vllm-ascend/pull/17354) 的说明将设备到主机同步消除列为单独的改进：相关最大长度与窗口边界由主机已有信息计算，避免在热路径读取设备张量标量。它同时将 FP8 KV 布局调整为每 token 608 字节，包含 BF16 的位置编码部分、FP8 的其余部分、比例值和填充；这一布局改变不能与单纯减少同步混为一项等价改动。

对照当前保留版固定 SHA 的目标 `sparse_mla` 和草稿 `noncausal_draft_attention`，两处 metadata 所需最大序列长度均由批次的主机侧长度列表计算，已核对的函数内没有以 `.item()` 获取长度；因此该 PR 的这一处优化目前没有明确的直接移植点。这是函数级静态检查，不证明整个推理链不存在其他同步，也没有测量同步耗时。当前不修改代码、不将其记为新收益。

## 910C 混合量化算子的量化模式限制

官方资料随版本变化。CANN 9.2 beta 1 的[接口文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta1/API/aolapi/context/ops-transformer/aclnnMixedQuantSparseFlashMla.md)将 A3 列为不支持；9.2.0 [发布公告](https://www.hiascend.com/productbulletins/detail/830)在 A2/A3 功能增强中列出 CSA 的 TurboQuant TQ4。不能用较早文档否定后续版本的所有 A3 支持。

核对 `hicann/ops-transformer` 主线提交 `6bff82496a8720343fc6144218f059a06a7b9b48`：[算子注册](https://github.com/hicann/ops-transformer/blob/6bff82496a8720343fc6144218f059a06a7b9b48/attention/mixed_quant_sparse_flash_mla/op_host/mixed_quant_sparse_flash_mla_def.cpp)包含 `ascend910_93`；[参数检查](https://github.com/hicann/ops-transformer/blob/6bff82496a8720343fc6144218f059a06a7b9b48/attention/mixed_quant_sparse_flash_mla/op_host/checkers/mixed_quant_variant_checker.cpp)限定 arch22 使用 `quant_mode=3`（TurboQuant），`quant_mode=1/2` 使用 arch35。该主线源码不是当前镜像已安装版本，源码注册也不是本机运行证明。

因此，本次评估的 FP8 KV 接口不能据此直接套用于 910C；TurboQuant 是可独立研究的另一种缓存量化方案，会改变缓存格式和数值。其检查还要求 `cmp_ratio` 为 4 或 128、特定 mask 和对齐条件；当前固定绑定的参数使用 1 或 2，两个接口的压缩比语义需要进一步对照，不能仅按数字映射。当前镜像未发现该模块；没有本机依赖、语义、精度、质量和性能证据前，不替换 BF16 保留路径。
