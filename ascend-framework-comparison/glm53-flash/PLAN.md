# Flash 独立适配方案

本文件只服务本模型，不依赖另一模型任务或基线。资源、当前状态、执行动作均以本目录为准。

## 本模型实现步骤

用 vLLM 做参考，按语义适配 SGLang

官方 vLLM-Ascend 当前 Flash 文档给出 A3 `v0.30.0rc1-a3`、W8A8、V1 runner 的路线。RC 不是生产稳定证明，文档 PD 也明确仅供参考；仍要固定镜像 digest、框架提交、SDK 与权重实际身份。本项目只接受RESOURCES固定的Flash权重及实际量化身份。

实施次序：

1. 固定同一份 Flash W8A8 权重、tokenizer、chat template 与量化说明；确定两框架能精确解释同一种量化格式。若 SGLang 需要格式转换，保存逐参数映射、scale/zero-point/shape、数值对照和转换摘要，禁止只改 architecture 或退回 BF16 后仍宣称同精度。
2. 14 上先验证 vLLM 单机真实文本和数值，再 14/15 真实 1P1D。46 上先做 SGLang 单机文本正确性；不要同时调试新模型、PD、MTP、图和 L3。
3. 检查并适配模型层/加载器、KDA 后端选择、卷积状态布局与激活/门控、recurrent state 更新与快照、mHC、DSA/indexer/量化 scale，以及输出头/多模态模块的加载边界。B103 存在 Ascend KDA 实现并不等于 Flash 已正确接线；Kimi 的 SiLU/门控与 Flash 配置也不能直接假定一致。
4. 做固定输入参考 logits/logprobs/greedy 输出及长上下文准确性；量化误差容差事先冻结。小输入通过后逐步长输入、前缀续写与取消/重排，验证状态槽位归属与回收。
5. 上 46/47 真实 1P1D：不仅迁移 DSA 的 KV/indexer/scale，还要迁移对应前缀位置的 KDA recurrent/conv 状态及一致的元信息，验证 P/D token、状态、完成事件和取消回收。先相同 TP，避免把异构 TP 重分片混入首个目标。
6. 逐项恢复兼容图模式、MTP（目标 D5）、连续批处理/混合缓存。MTP 验证接受/拒绝的状态提交、回滚和 draft 所有权；图验证真实 replay，不以 startup capture 代替。短期关闭某项只用于定位，最终性能报告明确哪些优化已启用、为何不能启用。
7. HiCache/MemCache 涉及混合状态检查点，不能只复用原 MLA KV 插件就宣称 Flash 前缀命中成立。先无外部缓存的 PD 功能门控，随后完成本项目合同的自然暖缓存门控再测性能。第一里程碑为文本；图像/视频另建正确性范围，不能用文本通过代表全多模态支持。

从 vLLM 借鉴的是状态语义、算子契约、调度和通信完成条件，而不是照搬 API/分配器。可复用代码须保留许可证和出处；每个移植点记录来源提交、改动与回归，优先小补丁，最终提供可独立上游审查的适配。


## 多模态与测试

先完成文本与本模型1P1D正确性，再逐项验证兼容图/MTP/混合状态缓存。参考本目录[多模态计划](MULTIMODAL_TEST_PLAN.md)、[矩阵](multimodal-matrix.json)和[独立性能合同](BENCHMARK_CONTRACT.md)。所有范围均以实际资格门控为准，不借用其他模型的通过记录。
