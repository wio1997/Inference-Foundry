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
| KDA 单 chunk 预填入口 | 对已投影、已卷积的一条序列提供 1–64 token 的显式 chunk 调用，输入 FP32 状态不变 | [chunk_prefill.py](DeepInfer/deepinfer/model_executor/models/glm5_next/chunk_prefill.py) | 四 die 单层实测，与安装版算子及分段状态交接对照；新增 dense KDA 解码层显式实验分派（最多 8 行）；默认逐 token 整模型短例回归通过，未接入完整文本 runner、未验收整模型质量或性能 |
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

实验性 rank 批量事务提供最多 8 个 token 的逐行状态事务，并非 chunk 内核或生产批处理；四 rank 短例的输入预检、head 异常和状态回滚范围见 [验证记录](validation/rank-batch-commit-round411.json)。

实验性 layer-major 模式仅对显式选择的 dense KDA 层调用 chunk，其余 sparse/DSA 逐 token；层掩码一致性、回滚及 recall17 首 token 未对齐的实测见 [记录](validation/dense-layer-prefill-round411.json)。

显式 singleton KDA 预填开关及跨 rank 掩码校验为实验接口；调用者必须在预填结束时关闭。单层、四 rank 交接和 recall17 仍未对齐的结果见 [记录](validation/single-prefill-round412.json)。

实验性的四分片 KDA 输入投影默认关闭；本地逐次构造权重，开销尚未测量。单层与四 rank 检查通过，但 recall17 与参考仍不一致，见 [记录](validation/reference-projection-round413.json)。

投影实验开关在非 layer-major 分布式路径会协同拒绝；旧单 token rank commit 增加一次 all-reduce，开销未测。[状态验证](validation/reference-projection-guard-round413.json)。

旧单 token rank commit 对本地校验失败先执行协同汇总，避免一方提前退出而其他 rank 进入后续 collective；位置失效的四 rank 检查见 [记录](validation/legacy-rank-preflight-round414.json)。

当前 rank-commit 源码的45层 archive-v2 保存、恢复和续写检查见 [记录](validation/archive-restore-round415.json)；这是受控单请求状态验证，不是设备故障恢复或生产验收。

四 rank archive-v2 固定 probe 已通过发布 runner 的 `--snapshot-probe` 实际调用；资源仍由 operator 配置和已有互斥登记控制。[记录](validation/archive-restore-round415.json)。

当前源码的45层专家缓存对照已接入 runner `--cache-equivalence-probe`；逐步 logits 和关闭清理通过，但四 rank 均为零 hit，不作重用或性能结论。[记录](validation/cache-equivalence-round416.json)。

CLI 请求开启恢复时，反序列化后比较45层状态并跨 rank 汇总结果，验证通过才继续生成；recall17 的四 rank 实测见 [记录](validation/recall-restore-round417.json)。

发布回滚探针覆盖单 rank 过期位置协同拒绝、45层状态保持及重试，worker 上限900秒；固定案例与普通路径回归见 [记录](validation/legacy-rollback-runner-round418.json)。

单 token rank 提交 preflight 现同步校验请求身份指纹、位置和 token 数；[四 rank 结果](validation/rank-request-identity-round419.json) 不覆盖进程或设备故障，collective 开销未测。

单 token rank preflight 包含 token 类型及词表范围检查，embedding 读取故障在进入 stack 前协同汇总；[注入结果](validation/rank-embedding-preflight-round419.json) 不涵盖设备/进程故障。

单 token rank preflight 将 token ID 与身份和位置一并汇总，不一致在 embedding 前拒绝；[四 rank 测试](validation/rank-token-preflight-round420.json) 不涵盖设备/进程故障。

reusable 专家 cache 的适配器预算与请求一致；64 MiB 及零 session 回归范围见 [记录](validation/cache-binding-round420.json)，零 hit 不代表性能收益。

单 die 的真实 recall17 layer0 八 token 边界和40 token尾部控制见 [记录](validation/kda-eight-boundary-round421.json)；未验收整体精度或性能。

实验性四分片输出投影复用同一默认关闭的参考分片开关，在本地以 BF16 平衡树汇合；单 die 的真实 recall17 第0层40 token回放位置0与保留参考逐位一致，8 token 仍有差异。一般 HCCL 等价及性能未测。[记录](validation/kda-eight-boundary-round421.json)。

四 rank 真实 recall17 首 token 对照显示，实验性输出分片会改变全模型 logits；本例 argmax 与参考一致，但完整 logits 不一致且未验收后续生成。[实测范围](validation/recall17-output-tree-round422.json)。

实验性输出分片的四 rank 固定 recall17 未强制生成结果与参考13个 token 一致；首 token 的全部 logits 不逐位相同，见 [结果](validation/recall17-output-tree-round422.json)。

实验性输出分片在四 rank recall42 的48 token生成序列与45 token参考在索引2开始不同；最终答案相同不代表完整精度一致。[结果](validation/recall42-output-tree-round423.json)。

实验性 KDA 分块与本地参考投影的四 rank 非零状态归档恢复及续算检查见 [结果](validation/chunk-archive-round423.json)；仅覆盖受控短输入。

四 rank 便携CLI现可显式开启实验性 dense KDA 分块预填；默认保持原逐 token 路径。固定短算术与默认回归范围见 [结果](validation/layer-major-cli-round424.json)，不代表整体质量验收。

实验分块的四rank缓存关闭/开启完整logits一致性及清理范围见 [结果](validation/layer-major-cache-round425.json)；此独立探针不代表性能收益；实际CLI缓存选项另有固定案例记录。

实验接口支持在显式选中的 sparse MoE KDA 层执行 chunk，MoE 逐 token 调用；四 rank 的第4层短例与受控回滚范围见 [记录](validation/sparse-kda-chunk-round428.json)。默认 CLI 选择与其他 sparse 层执行方式未改变。

实验性便携 CLI 可另行显式选择第4层 sparse KDA chunk；MoE 仍逐 token，实际四 rank 短例见 [记录](validation/sparse-kda-cli-round429.json)。

第4层 sparse KDA CLI 的 recall17 生成序列与参考从首 token 起不同；实验范围及未解决质量差异见 [记录](validation/sparse-kda-recall17-round429.json)。

第4层 sparse KDA CLI 的 recall42 序列亦与参考从首 token 起不同，见 [记录](validation/sparse-kda-recall42-round429.json)。

第4层 sparse KDA 分块选择跨 rank 不一致的预检、回滚与重试范围见 [记录](validation/sparse-kda-mask-round430.json)。

第4层 sparse KDA 的非零状态归档恢复及8加1 token 续算对照见 [记录](validation/sparse-kda-archive-round431.json)。

第4层 sparse KDA singleton 续算中受控 head 失败的协调回滚与重试见 [记录](validation/sparse-kda-singleton-rollback-round431.json)。

实验性 sparse KDA CLI 的 cache 与首 token 后恢复组合见 [记录](validation/sparse-kda-cli-restore-round431.json)。

rank W8A8 输出投影可显式配置最多40个输入 token，以8行分块；[验证范围](validation/rank-linear-tiles-round434.json)仅覆盖算子与受控 rank 故障。

KDA 层和 rank 事务具备显式40 token 实验界限，实际45层40+1输入结果见 [记录](validation/prefill40-round436.json)；默认界限仍为8。

40+1 实验路径的 recall17 完整生成范围与差异见 [记录](validation/recall17-prefill40-round437.json)。

40+1 实验路径的 recall42 完整生成范围与差异见 [记录](validation/recall42-prefill40-round438.json)。

公共 CLI 的40 token 实验性预填充和 singleton 路径见 [验证记录](validation/cli-prefill40-round439.json)；此项不证明全模型精度。

当前默认 tokenwise 路径的固定输入回归范围见 [记录](validation/default-cli-round440.json)。

公共 CLI 40+1 实验路径的 recall42 完整生成范围与差异见 [记录](validation/cli-recall42-prefill40-round441.json)。

40 token分布式事务受控回滚及重试见 [记录](validation/rollback40-round442.json)。

40 token受控事务回滚探针已接入公共CLI，范围见 [记录](validation/cli-rollback40-round443.json)。

40 token预填后的状态序列化、恢复和续算验证见 [记录](validation/archive40-round444.json)。

40 token预填的专家缓存等价验证见 [记录](validation/cache40-round445.json)。
