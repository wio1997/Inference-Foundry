# 910C 算子与模型接口适配清单

本清单对应本分支已发布的源码快照，说明改了什么、调用什么实现，以及当前验证到哪里。**这里的“适配”包含权重布局、算子接口、张量形状、计算组合和状态管理；调用已有 CANN、torch-npu、FLA 或 vLLM-Ascend 算子，不等于本项目新写了相应底层 kernel。**

源码中保留多种实验后端，不能把“文件存在”当成该后端已进入最终运行路径，也不能把短请求整模型运行当成所有算子形状的独立验收。

## 一、计算算子及组合

下表中的“整模型路径”指已保留的 45 层、四 rank、短上下文实验。便携脚本修改后的硬件验证另行记录，不沿用原现场启动器的通过结论。

| 项目 | 本次适配内容 | 主要源码 | 验证范围与限制 |
|---|---|---|---|
| W8A8 动态量化 Linear | 按量化检查点读取 INT8 权重、通道 scale；处理 `[N,K]` 到计算布局及 NZ 格式；输入动态量化后调用 `npu_quant_matmul`，保留 per-token scale | [w8a8.py](DeepInfer/deepinfer/npu_ops/w8a8.py)、[modelslim910c.py](DeepInfer/deepinfer/hugging_face/modelslim910c.py) | 已接入实测短请求路径；不是把 INT8 权重转成 BF16 作为替代实现 |
| 分组 INT8 GEMM、SwiGLU 专家 FFN | 专家权重按组组织，适配 `npu_grouped_matmul`、中间激活和再量化；提供已有 `grouped_matmul_swiglu_quant_weight_nz` 融合接口的显式选项 | [w8a8.py](DeepInfer/deepinfer/npu_ops/w8a8.py)、[expert_bank.py](DeepInfer/deepinfer/model_executor/models/glm5_next/expert_bank.py) | 专家和整模型路径有实测；融合选项与普通组合路径分别保留，不代表任意配置等价 |
| KDA 投影与投影合并 | 对齐 GLM 检查点中的 Q/K/V、beta、遗忘门及输出投影；满足权重类型条件时，将 Q/K/V/beta/f_a/g_a 合并为一次 FLOAT 投影后再拆分 | [kda.py](DeepInfer/deepinfer/model_executor/models/glm5_next/kda.py) | 当前模型路径已运行；投影合并不代表新写 GEMM kernel |
| KDA 单 chunk 预填入口 | 对已投影、已卷积的一条序列提供 1–64 token 的显式 chunk 调用，输入 FP32 状态不变 | [chunk_prefill.py](DeepInfer/deepinfer/model_executor/models/glm5_next/chunk_prefill.py) | 四 die 单层实测，与安装版算子及分段状态交接对照；新增显式单层 chunk 后端，未接入完整文本 runner、未验收整模型质量或性能 |
| KDA 因果卷积 | 适配 Q/K/V 的短因果卷积、SiLU、历史窗口顺序及分段续算；原生接口调用 `fla_npu.ops.ascendc.causal_conv1d_fn`，保留 Torch 参考路径 | [kda.py](DeepInfer/deepinfer/model_executor/models/glm5_next/kda.py) | 与 KDA 状态共同进入短请求验证；不代表任意卷积宽度、批次或长序列已验收 |
| KDA 递推、门控和 Q/K 归一化 | 对齐 FP32 recurrent 状态、门控参数和布局；保留显式 Q/K 归一化的 AscendC 路径，以及 `vllm_ascend.ops.kda.run_recurrent_kda` 的单 token 解码适配 | [kda.py](DeepInfer/deepinfer/model_executor/models/glm5_next/kda.py) | 便携 worker 默认选择 `ascendc_installed_decode`，可显式选择 `ascendc`；两者均为逐 token 路径，不能据此宣布 chunk-prefill 已解决 |
| mHC 前处理与残差后处理 | 适配多流残差、门控、Sinkhorn 混合及权重精度；接入已有 `npu_hc_pre_v2` / `npu_hc_post`，保留 Torch 参考组合 | [mhc.py](DeepInfer/deepinfer/model_executor/models/glm5_next/mhc.py) | 整模型配置选择 `ascend_installed`；独立参考生成仍有差异，不能把整个数值链宣布完全对齐 |
| MLA 的 Q/KV/输出投影 | 对齐无 RoPE 的 GLM MLA 形状、低秩 Q/KV 投影、归一化、KV 展开及输出投影；提供显式 rank 投影载荷接口 | [projections.py](DeepInfer/deepinfer/model_executor/models/glm5_next/projections.py)、[rank_dsa.py](DeepInfer/deepinfer/model_executor/models/glm5_next/rank_dsa.py) | 当前短请求路径已执行；attention 在实测四 rank 路径中仍有复制，不能称为完整生产 TP |
| DSA indexer、key pool 与 top-k 选择 | 对齐 query/head-weight/key/gate 投影、池化压缩、因果尾部及索引地址；整理 CANN `key_pool` / `pool_key_indexer` 的接口和页表契约，保留实验 ReLU 池评分选择 | [cann_indexer.py](DeepInfer/deepinfer/model_executor/models/glm5_next/cann_indexer.py)、[cache.py](DeepInfer/deepinfer/model_executor/models/glm5_next/cache.py)、[experimental_fp32_expanded_dsa.py](DeepInfer/deepinfer/model_executor/models/glm5_next/experimental_fp32_expanded_dsa.py) | CANN 接口检查代码含 CPU/source-only 契约；当前短请求使用实验选择路径，不能把接口存在写成原生稀疏 indexer 全链路已完成 |
| 原生 Sparse Flash Attention 接口 | 显式加载自定义算子库，适配 `_C_ascend.npu_sparse_flash_attention` 输入、NoPE 几何、索引和缓存载荷 | [native_dsa.py](DeepInfer/deepinfer/model_executor/models/glm5_next/native_dsa.py)、[sfa_inputs.py](DeepInfer/deepinfer/model_executor/models/glm5_next/sfa_inputs.py)、[sfa_weights.py](DeepInfer/deepinfer/model_executor/models/glm5_next/sfa_weights.py) | 独立实验接口仍保留；当前公开 worker 的完整回答采用下述 FP32 实验路径，不应写成已由 SFA 跑通同一结果 |
| 实验性 FP32 展开 DSA | 展开 KV，在 FP32 中执行带显式因果稀疏掩码的矩阵乘、softmax 和归约，输出边界转 BF16；必须显式启用 | [experimental_fp32_expanded_dsa.py](DeepInfer/deepinfer/model_executor/models/glm5_next/experimental_fp32_expanded_dsa.py) | 当前短上下文整模型结果来自该路径；它是有界正确性实验，不是优化后的稀疏注意力 kernel，不代表长上下文或性能通过 |
| MoE 路由、token 分组与结果合并 | 实现 sigmoid/noaux 路由、bias 参与选择、top-k、路由权重归一化/缩放、expert-major 分组、加权合并及共享专家相加 | [moe.py](DeepInfer/deepinfer/model_executor/models/glm5_next/moe.py)、[expert_bank.py](DeepInfer/deepinfer/model_executor/models/glm5_next/expert_bank.py)、[replicated_rank_moe.py](DeepInfer/deepinfer/model_executor/models/glm5_next/replicated_rank_moe.py) | 已接入当前模型；部分核对复用相同底层 NPU 算子，不是完全独立数值 oracle |
| Dense FFN、专家及 Linear 的 rank 分片 | 对 Dense gate/up 和选中专家做实验分片，通过 HCCL collective 汇合局部结果；补齐 rank 路由和输入/布局检查 | [rank_dense.py](DeepInfer/deepinfer/model_executor/models/glm5_next/rank_dense.py)、[rank_linear.py](DeepInfer/deepinfer/model_executor/models/glm5_next/rank_linear.py)、[rank_routes.py](DeepInfer/deepinfer/model_executor/models/glm5_next/rank_routes.py)、[rank_expert_adapter.py](DeepInfer/deepinfer/model_executor/models/glm5_next/rank_expert_adapter.py) | 四 rank 短请求已运行；attention、router、embedding、head 等仍有复制，不等同于全模型生产 TP/EP 或完整 all-to-all 专家并行 |
| Embedding、末端归一化与 LM Head | 有界读取 embedding 行，处理 mHC 多流最终隐藏状态、RMS 归一化及词表投影边界 | [text_boundary.py](DeepInfer/deepinfer/model_executor/models/glm5_next/text_boundary.py) | 随短请求生成运行；词表投影接口不代表实际四 rank 路径已完整分片 |

## 二、支撑算子运行的接口与状态适配

这些是模型集成和运行时能力，**不计作新增计算算子或 kernel**。

| 项目 | 改动及源码 | 当前边界 |
|---|---|---|
| 检查点元数据及量化权重读取 | [glm5_next_config.py](DeepInfer/deepinfer/hugging_face/glm5_next_config.py)、[modelslim910c.py](DeepInfer/deepinfer/hugging_face/modelslim910c.py)、[checkpoint910c.py](DeepInfer/deepinfer/npu_ops/checkpoint910c.py)：显式配置、索引、量化格式、读字节预算与布局检查 | 绑定 README 中记录的模型指纹，不自动代表其他权重兼容 |
| 混合层组装 | [hybrid_decoder.py](DeepInfer/deepinfer/model_executor/models/glm5_next/hybrid_decoder.py)、[native_model.py](DeepInfer/deepinfer/model_executor/models/glm5_next/native_model.py)、[rank_text_install.py](DeepInfer/deepinfer/model_executor/models/glm5_next/rank_text_install.py)：把 KDA、DSA、mHC、Dense/MoE 接入 45 层及 rank 安装流程 | 必须显式提供实验 attention 实现和对应选项；不是通用生产 serving 接入 |
| 请求隔离与跨 rank 提交 | [text_request.py](DeepInfer/deepinfer/model_executor/models/glm5_next/text_request.py)、[request_state.py](DeepInfer/deepinfer/model_executor/models/glm5_next/request_state.py)、[coordinated_bank_lease.py](DeepInfer/deepinfer/model_executor/models/glm5_next/coordinated_bank_lease.py)：私有状态、位置检查、head/状态发布前的一致性及失败处理 | 仅覆盖已定义的同步异常边界，不等于进程崩溃或设备故障容错 |
| 生成中途状态归档与恢复 | [state_archive.py](DeepInfer/deepinfer/model_executor/models/glm5_next/state_archive.py)：archive-v2、owner/layout 核对、KDA/DSA 状态序列化和恢复 | 已保留短请求、45 层、多 rank 恢复结果；长上下文和跨版本恢复未验收 |
| 有界专家缓存及资源生命周期 | [cached_expert_bank.py](DeepInfer/deepinfer/model_executor/models/glm5_next/cached_expert_bank.py)、[cached_expert_session.py](DeepInfer/deepinfer/model_executor/models/glm5_next/cached_expert_session.py)、[rank_cache_scope.py](DeepInfer/deepinfer/model_executor/models/glm5_next/rank_cache_scope.py)：缓存预算、lease、关闭和异常清理 | 实测缓存命中为零；逻辑缓存预算不包含全部瞬态/native/allocator 内存，不能声称已经提速 |
| paged cache、索引 ABI 和 staged commit 原型 | [cann_cache.py](DeepInfer/deepinfer/model_executor/models/glm5_next/cann_cache.py)、[paged_commit.py](DeepInfer/deepinfer/model_executor/models/glm5_next/paged_commit.py)、[private_pages.py](DeepInfer/deepinfer/model_executor/models/glm5_next/private_pages.py) 等 | 代码及局部契约不等于已完成生产 paged-KV、调度器和长上下文集成 |

## 三、当前实际计算路径

当前发布的 [portable/worker.py](portable/worker.py) 在组装时显式选择：

```text
KDA：ascendc_installed_decode（逐 token）
mHC：ascend_installed
DSA：make_experimental_fp32_expanded_dsa_operations（显式实验路径）
专家：selected_per_token；随后安装实验性 rank 分片接口
```

因此，“接了原生 SFA 接口”“保留 chunk/归一化实验后端”“有 paged 状态代码”，都不能替代上面实际执行配置的说明。便携版修改后的硬件状态以最新 README 和新增运行结果为准。

## 四、验证材料及未解决问题

- [算术参考比较](validation/arithmetic-comparison.json)：一个固定短请求的生成 token 对齐。
- [多轮记忆参考比较](validation/recall-comparison.json)：最终答案均为 `42`，但生成序列并未完全一致，首个差异位于零基索引 2。
- [生成中途恢复结果](validation/restore-round357.json)、[启用缓存后的恢复结果](validation/restore-round358.json)：限定场景内的状态和续算验证。
- [源码单文件哈希](validation/source-files.sha256.json)：用于核对被验证的源码身份。

仍未完成：参考差异根因与整模型精度验收、生产级 TP/EP 与 paged-KV、服务/批处理、长上下文质量、端到端性能，以及便携部署入口的真实复现。后续新增或替换后端时，应同步更新本清单及实测配置，不能仅累计“已适配算子数量”。
