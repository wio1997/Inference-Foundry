# gen43：lm_head 到 logits 的精度与量化源码契约，阶段完成待 root 验收

本代补齐已保存短诊断的 head 路径约束，没有新模型推理、服务启动、NPU、SDK 借用或性能测试。原导出仍为124/204，未改动 pipeline、原 worker 源码或结果目录。最新地图和未分 R 的 main/index 候选仍为 gen41；本代证据尚未合并进去。

两栈当前挂载 config.json 和 quant_model_description.json 的完整 SHA 均与先前 model-identity-receipt 一致。配置为 Qwen3_5ForConditionalGeneration、text dtype=bfloat16、vocab248320、hidden5120、tie_word_embeddings=False，没有 head_dtype 或 final_logit_softcapping 字段。量化描述的整体标签为 W8A8_DYNAMIC，但 lm_head.weight 单项为 FLOAT。FLOAT 是量化分类，不能解释为 FP32。仅读取 safetensors 索引和文件头，两个挂载均标注该 tensor 为 BF16、形状[248320,5120]；索引完整 SHA 与下载字节一致，文件头另有完整 SHA。未读权重 payload、未算完整 shard SHA，也没有确认历史加载后的 tensor dtype/内容，因此不能据文件头宣称两栈历史投影数值相同。

## 已核验的路径与条件

- vLLM Qwen3.5 conditional 模型通过 Qwen3VL.compute_logits 转到 language_model.compute_logits，再将 lm_head/hidden_states 交给 LogitsProcessor。模型构造只传 vocab_size，基类默认 scale=1、soft_cap=None。Ascend 注册表选择 AscendLogitsProcessor/AscendParallelLMHead；其 _apply_head 显式委托基类，_get_logits 按 normal、PCP 或 fine-grained 路由处理通信，不能忽略 Ascend 覆盖层。当前源代码注册和委托关系不等于历史实际类绑定/插件清单已穷尽。
- vLLM _get_head_dtype 在 generation 且无覆盖时返回 model dtype，pooling 的默认行为不同。head_dtype 与 hidden dtype 相同或为空时，投影走 quant_method.apply；不同 dtype 只接受无量化方法。在所审查 NPU 路径上，不同 dtype 的投影对 hidden/weight 做显式 cast 后 F.linear；CUDA/ROCm 的 mm(out_dtype) 不能套到 NPU。Ascend ModelSlim 将 FLOAT 分类映射为 None，在 embedding 分支选择 UnquantizedEmbeddingMethod。具体运行时 GEMM dispatch/累加精度和动态绑定仍未闭合。
- SG Qwen3.5 conditional 继承 Qwen3VL，并以 text_config 构造模型和独立 ParallelLMHead，LogitsProcessor 未传 logit_scale，默认 None；softcap 取 config.final_logit_softcapping。当前源文件里 should_apply_lm_head_quant_method 有两个顶层定义；Python 使用后一个（第1278行），本代明确执行后定义。UnquantizedEmbeddingMethod、UnquantizedLinearMethod、PackWeightMethod 不走 quant_method.apply；其他有效量化方法可优先于普通 head 的 fp32 flag 分支。
- SG ParallelLMHead 的继承链为 VocabParallelEmbedding，不是 LinearBase。该 ModelSlimConfig.get_quant_method 对这一链返回 None，再由 embedding 构造 fallback 为 UnquantizedEmbeddingMethod；不能误报为其 LinearBase 的 FLOAT→UnquantizedLinearMethod 分支。普通 head 默认按 weight.dtype 转 hidden 后 matmul；fp32 开关另走 FP32 operands，policy 分支另有 BF16 cast。保存的短诊断 server_args 明确 dtype=bfloat16、quantization=modelslim、enable_fp32_lm_head=False、rl_on_policy_target=None；保存的 vLLM engine 日志为 bfloat16/ascend/TP4。这些日志不记录每步实际 head 对象/精度。
- SG 顺序为投影→可选 logit_scale→TP 汇聚/DP scatter→crop/float32 buffer→可选 softcap。buffer 有正确形状时要求 torch.float 并 copy，否则 logits.float()。vLLM 则在投影/汇聚/裁剪后处理可选 softcap/scale，稍后的 sampler 才转 float32（gen40 已归档）。最终转 float32 不能作为投影数值等价的证据；本代没有执行 tensor 数学或观察 logits。
- 不能把 installed SG Qwen3.5 当成实际兼容 overlay。另收取了 exact overlay source、当前 manifest、被 gen67 controller 源 SHA 固定的 sitecustomize，以及 gen67 五份已保存 startup qwen3.5 original/candidate SHA receipts。其 original SHA 对应本次 installed 源码，candidate SHA 对应兼容源码；两个相关模型类的继承及 constructor AST 完全相同。manifest 未替换 qwen3_vl/logits_processor。此窄桥接只覆盖所审查 head 构造，未证明所有下游模块历史 loaded SHA 或完整动态 patch 状态；原兼容修复也不能当作无修改模型。

## 验证与可搬移证据

通过 task-private transport 在原 exact historical container/image 中运行 python -I 标准库读取和 git show，没有 import torch/框架。36个 installed/config SHA 观测覆盖27个不同路径，24个 Python 源码观测与各自 git HEAD 字节相等；重复下载的相同文件保留为不同观测。另8个 overlay/startup/sitecustomize 文件 SHA 逐个复核。以上数量含配置和 JSON，不能统称为源码文件数。远端 Host Architecture=aarch64；本地53个精确 AST/哨兵检查运行于 x86_64，覆盖 dtype/量化拒绝、Ascend 委托、softcap/scale、SG 最后定义、投影与通信/float32 buffer 顺序、ModelSlim head 类型、配置/文件头/日志和兼容构造桥接。23个函数锚点及完整文件 SHA 在 head-source-links.csv。

这些检查只验证所审查源码条件和边界，不是 aarch64/NPU/tensor 精度、模型质量、greedy 等价或性能验证。审查包只含代码、小型配置/索引/收据/日志及 SHA 清单，不含 transport/key、raw profile、权重 payload、tokenizer、SO、unattended 或旧 worker 目录。搬移后重新执行全部53项并比较完整审计 JSON（只排除 verified_ns）和 CSV；包与解包成员逐一复核 SHA。

## 原后台作业与后续

一次状态比较确认原124条完成记录全 JSON 未变，无新增完成 key。reported submitted124/inflight1/pending76，精确集合仍有4个 submitted 未归档；保留两种口径。定向 worker 收据表明124/125/126均为 failed_inspect_live_or_partial，其 lineage 操作 rc137；126于2026-10-26T22:58:21.189898Z结束。本代没有把126的137进一步判为 OOM，gen41同容器 OOM/124exec 链证据也不能代替126精确归因。127的 lineage docker exec PID1845815/start517231112，容器 Python PID1845839/start517231123仍在；命令、boot ID、日志、开始时刻及终止条件均保存。4 masters/4 CPU pool workers 均保留，无 signal/retry/restart/release。原 latched failure 不会自行派发剩余76项；不能仅等待204或把进程存活当完成。

下一代可在新目录把本代 head 契约与gen41地图/main/index候选显式合并，做新内容和搬移 QA；同时先核对127是否 terminal。待原作业终态/所有者协调交接后，继续gen42独立 CPU recovery 候选的长输入验证和全阶段内存边界审查，不修改旧 worker 目录、不绕过原 plan/lineage gates。未完成实际204、最终地图 QA、质量/greedy等价、私有 R 分配/规则刷新/main/index发布与 root 验收。不宣布优化采用，overall_complete=false。

本轮工具失败单独记录：容器未安装 rg，改为标准库窄查找后成功；首次猜测 ops/logits.py 路径不存在，按实际查找的 ops/vocab_parallel_embedding.py 修正后成功。均为审计工具/路径失败，不是技术候选或模型失败。
