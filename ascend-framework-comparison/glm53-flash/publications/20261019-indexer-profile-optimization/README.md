# GLM 5.3 Flash 性能排查与优化阶段总结

截至 2026 年 10 月 19 日，最新一轮整模型诊断已经结束，按用户要求暂停后续实验。两处 indexer 修改使相同查询量、相同上下文的整模型 Prefill 采样耗时下降约 6.3 至 7.7 倍。完整 32 前缀预热和 SGLang 正式 C32 吞吐尚未重新测量。

## 实测结果

每个窗口包含同一个 TP0 上连续两次真实模型前向，每次处理 4096 个查询 token。修改前后均为 TP16、DP1、EP1、eager、MTP 关闭；镜像、权重和其余 44 个 overlay 文件不变。设备时间来自原生 NPU `kernel_details.csv`，完整窗口在设备同步后结束。

| 窗口 | 上下文结束位置 | 修改前 | 修改后 | 采样提速 |
| --- | --- | ---: | ---: | ---: |
| 早段 | 8192、12288 | 16.9115 秒 | 2.6786 秒 | 6.31 倍 |
| 中段 | 32768、36864 | 24.1430 秒 | 3.1295 秒 | 7.71 倍 |
| 晚段 | 61440、65536 | 原始轮未采到 | 3.7510 秒 | 不计算 |

早、中段 DSA 评分 kernel 的耗时总和分别从 3.6589、10.7418 秒降到 0.2093、0.6529 秒；每个窗口均为 22 次评分。窗口内无 kernel 执行的间隙从约 10.8 秒降到约 0.05 秒。Kernel 耗时总和与区间并集分别统计，不相加冒充关键路径占比。

本轮两个独立的 74400 token 冷前缀均完成，P、D 两条请求腿共 4 次 POST 全部成功，每条输出 1 token；早、中、晚三段原始导出完整。两个自有容器均以原生 exit 0 退出，两节点均恢复 idle8，先停止 P 写端、证明空闲后再释放 D 目的端。

## 两个主要原因及修改

1. **页表行步长错误地参与编译特化。** 上下文增长时，页表切片的行步长变化，反复生成实际上等价的评分代码。修改 gather 和 score 两个 Triton kernel 的四处声明，将步长作为运行时标量并加入 `do_not_specialize`，保持算术与精度不变。实际六种宽度测试中，两类 kernel 的编译变体均从 6 降到 1；首次编译仍约 18.6 秒，后续旧路径每次约 5 秒，新路径约 3 至 19 毫秒。六组输出逐字节一致。
2. **TP16 重复计算完整 DSA 查询行。** 原 SGLang 私有适配在各 rank 上重复评分全部查询；vLLM 的实际路径启用了 `enable_dsa_cp`，在 attention TP 组内分配查询。修改后，SGLang Prefill 按当前 attention TP 组切分独立查询行，再通过一次原生 all-gather 还原最终 int32 索引。Decode 和小 Prefill 沿用原路径。实际 TP16、五种 case、80 组比较全部逐元素一致；单算子包含真实 HCCL 的中位耗时从 1135.1 降到 77.2 毫秒。

差异见 [性能补丁](patches/sglang-indexer-performance.patch)。归档的 [完整冻结源码](frozen-source/) 保留执行时的配置和控制路径；其中准备阶段的状态文字保持冻结原样，实际结果以本页验收及证据为准。此补丁基于原 TP16 eager 诊断路径，不能用其中较早的 caller 覆盖下一版带 MTP 扩展的 caller。

## vLLM 在哪里及比较口径

vLLM 使用 14P、15D 节点，实际源码位于镜像内的 `/vllm-workspace/vllm` 和 `/vllm-workspace/vllm-ascend`；原生 indexer 路径是 `vllm_ascend/attention/indexer_kpool.py`。SGLang 使用 46P、47D。两者使用本项目同一 GLM 5.3 Flash W8A8 权重入口和原生 SDMA 数据面。

原始 32 前缀预热：vLLM 为 **565.84 秒**，旧 SGLang 为 **6403.19 秒**，差约 11.32 倍。vLLM 已接受的正式结果为 128 个请求、实际输入 80000 token、输出 600 token、C32，输出吞吐 **306.45 token/s**，预热排除。旧 SGLang 正式轮在 33 个终止请求中只有 10 个成功、23 个超时，没有合格的完整正式基线。

配对整模型 profile 已用于定位上述主要成本。两框架原部署存在差异：vLLM P 为 DP2、TP8、EP16，D 为 DP8、TP2、EP16，并使用 MTP5 和 D Graph；本次隔离实验保留旧 SGLang 的 DP1、TP16、EP1、eager、MTP 关闭配置。vLLM 的采样查询量为 3840，SGLang 为 4096，因此跨框架采样不能当作纯框架性能比。

## 验收记录与停止位置

原控制器总状态保留为 **FAIL**：它要求逐笔 `push_complete` 日志，而性能模式写的是每设备 `performance_SDMA_summary`。独立核对原始日志及实际加载源码后，16 个 P rank 合计 **704 次原生 SDMA 写入成功，失败 0 次，28,825,733,504 字节，704 次生产者同步**；两端均记录 `TransDataOpType.SDMA`。据此接受本轮诊断证据，原始失败记录不改写。性能模式未执行逐字节 payload 哈希核验。

BF16 KV、FP32 评分与持久状态等本轮约束保留；本轮只验证两处性能修改相对旧路径的等价性。旧部署中其他数值资格问题、完整并行布局、Graph、MTP 和完整 C32 结果仍需独立验证，不能据此宣称已全面恢复 vLLM 配置或已将完整预热缩短同样倍数。

下一版完整配置源码草稿已保留在本地，**未部署、未运行、未获主控完整验收**，并已按用户要求停止准备进程。本次 GitHub 归档发布已测修改和证据摘要。

恢复工作时应先审查完整配置草稿，保留已验证的步长修改和扩展 query-parallel caller，完成长输入、MTP、Graph、数值、PD 及 SDMA 功能验证；随后再做固定的原生 AISBench C32、80000 输入、600 输出、128 请求正式测试，单独报告完整预热。

## 证据索引

| 文件 | 内容 |
| --- | --- |
| [整模型对照](evidence/whole-model-comparison.json) | 修改前后完全匹配的早、中段及候选晚段数据 |
| [整模型独立验收](evidence/whole-model-validation.json) | 原始失败原因、SDMA 实际汇总、请求及清理证据 |
| [算子独立验收](evidence/component-validation.json) | 六宽度编译与逐字节一致性、TP16 的 80 组比较 |
| [原框架配对诊断](evidence/paired-framework-diagnostic.json) | vLLM 与旧 SGLang 的配对采样及配置差异 |
| [来源与范围](evidence/provenance.json) | 镜像、源码 SHA、原始证据路径、暂停状态 |
| [归档文件清单](manifest.json) | 除清单自身外的归档文件 SHA256 与大小 |

原始 trace、权重、镜像、真实输入输出、provider 和凭证均不进入 Git。冻结的旧一次性 launcher 仅作为审查记录保存，不得重放；重新执行必须创建新的运行身份、完成审核并重新核对现场状态。
