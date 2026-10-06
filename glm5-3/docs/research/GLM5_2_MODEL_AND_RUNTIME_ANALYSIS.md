> 历史GLM-5.2资料，来源/模型结构/软件支持事实保留；不是当前GLM-5.3的模型或性能证据。当前入口为[GLM-5.3模型核验](GLM5_3_MODEL_AND_RUNTIME_ANALYSIS.md)与[运行路径](GLM5_3_VLLM_ASCEND_PATHS.md)。

> 研究快照：整理于2026-10-01，随后作为按需资料发布；文中未提交/未运行陈述描述研究采集阶段，不代表后续Git发布状态。实际运行状态以局部HANDOFF和Git为准。

# GLM-5.2 模型特性、MTP与PD调度研究分析

研究日期：2026-10-01（Asia/Shanghai）。本文件为只读外部研究与结构推导，没有连接用户运行机器，没有测量其模型或服务，没有修改 GitHub。用户已补充当前有两台同规格 Ascend 910C 服务器、卡相同；结合此前口径，本计划按每台 8 卡 16 芯理解，这是当前计划解释，不是远端核验结果。DP1/TP16/EP16、W8A8、PD 分离、vLLM-Ascend 0.27来自用户陈述；当前P/D实例数、角色到服务器的放置、各实例实际rank/通信组与进程映射，以及安装版本、启动命令和checkpoint仍待恢复。

本文件负责模型结构与性能含义；vLLM/Ascend当前公开源码的MTP、Graph、量化、Scheduler和connector入口另见同目录`GLM5_2_VLLM_ASCEND_PATHS.md`。按当前问题读取，不要求每次恢复通读全部材料。

## 1. 依据与版本边界

最强结构依据是官方 `zai-org/GLM-5.2` 固定 revision `cf457fa734ab149ffef225f80893eb38c6ff5cdc` 的 config 和权重索引，而不是 GLM-5 的老报告，也不是以模型名猜测实现。官方 Hugging Face API 返回相同 revision。该仓库是 BF16 原始权重；它不能证明用户 W8A8 导出后的量化范围、cache dtype、实际 kernel 或通信实现。

| 来源 | 固定标识 / 地址 | 本次用途与限制 |
|---|---|---|
| 官方 GLM-5.2 配置 | [config.json](https://huggingface.co/zai-org/GLM-5.2/blob/cf457fa734ab149ffef225f80893eb38c6ff5cdc/config.json) | 主模型结构、Indexer 模式、MTP 层数；不是服务配置 |
| 官方 GLM-5.2 权重索引 | [model.safetensors.index.json](https://huggingface.co/zai-org/GLM-5.2/blob/cf457fa734ab149ffef225f80893eb38c6ff5cdc/model.safetensors.index.json) | 检查主层及额外 MTP 权重键，未下载模型权重 |
| 官方 GLM-5.2 模型卡 | [README.md](https://huggingface.co/zai-org/GLM-5.2/blob/cf457fa734ab149ffef225f80893eb38c6ff5cdc/README.md) | 版本专属能力与部署入口；不证明具体 Ascend 路径 |
| 官方 GLM-5.2 博客 | [2026-06-16 发布说明](https://z.ai/blog/glm-5.2) | IndexShare、MTP IndexShare/KVShare、1M 长上下文设计；博客机制不等于当前框架已实现 |
| Transformers v5.12.0 | commit `e0e7504bca2bfd1b85bb0eedb148f7b250226f06`；[modeling_glm_moe_dsa.py](https://github.com/huggingface/transformers/blob/e0e7504bca2bfd1b85bb0eedb148f7b250226f06/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py) | 原生模型的语义参考；不是 vLLM-Ascend 执行实现 |
| GLM-5 技术报告 | [arXiv:2602.15763v2](https://arxiv.org/html/2602.15763v2)，2026-02-24 | MLA/DSA/MTP 设计背景；其中 80 层、744B、40B 活跃、接受长度等不能直接赋给 GLM-5.2 当前部署 |
| IndexCache 论文 | [arXiv:2603.12201v1](https://arxiv.org/html/2603.12201v1)，2026-03-12 | Full/Shared 语义背景；主要实验是 30B 模型，不能把速度数字迁移到 910C/GLM-5.2 |

官方博客初始 HTTP 响应是静态壳，本次直接下载其引用的 `https://z.ai/blog/assets/glm-5.2-iw-RMVcl.js`，只读取内容字符串，未执行 JavaScript，未使用浏览器。静态资产 URL 和 SHA 可供恢复，但网站无公开版本固定承诺。

## 2. GLM-5.2 可直接验证的结构

| 项目 | 固定 revision 的事实 | 调度含义 / 限制 |
|---|---|---|
| 模型类 | `GlmMoeDsaForCausalLM`，`model_type=glm_moe_dsa` | MLA + DSA + MoE 是研究图的基本构成 |
| 主模型层数 | `num_hidden_layers=78` | 主干有 78 层；不能直接使用 GLM-5 报告的 80 层 |
| Hidden / 激活 | hidden 6144，SiLU | 确定 metadata、activation 和通信 tensor 的形状基础，实际分片待查 |
| FFN 分布 | 0/1/2 三层 dense，之后 75 层 sparse；dense intermediate 12288 | 大多数层存在路由与专家阶段，少量 dense 层单独分型 |
| MoE | 256 routed experts，每 token top-8；1 shared expert；expert intermediate 2048 | 不能用平均 FLOPs代表 EP16 的实际 rank 工作量与到达偏差 |
| Router | `moe_router_dtype=float32`；sigmoid；`noaux_tc`；`norm_topk_prob=true`；scaling 2.5；`n_group=topk_group=1` | Router/dispatch/combine/shared 分支的真实依赖与同步需要恢复；参数不证明对应 kernel dtype |
| Attention heads | 64 heads，`num_key_value_heads=64` | 若实现均匀按头 TP16，可对应 4 query heads/rank；不能由此推断 KV-cache 可直接除以 16 |
| MLA query | q low-rank 2048；qk non-RoPE 192 + RoPE 64 = 256 | 主 Q 投影与 indexer 均涉及低秩 query 状态；代码证实 indexer 接收 q_resid |
| MLA KV | latent rank 512 + 独立 RoPE 64；value head dim 256 | 逻辑压缩状态是 576 维；实际 cache 是否压缩、分片、复制、量化、padding 待核验 |
| Indexer | 32 heads × 128 dim；`index_topk=2048` | 稀疏 attention top-k 固定上限，indexer 仍需扫描历史；long-context 服务成本不能当常数 |
| IndexShare | `index_topk_freq=4`、offset=3；显式 78 项 `indexer_types` | 必须用显式模式，不能简单计成 78/4 个 indexer |
| 上下文 | `max_position_embeddings=1048576`；tokenizer 同值；RoPE theta 8000000、default/interleave | 1M 是模型配置能力，不自动成为当前性能合同或当前最大可运行上下文 |
| MTP | `num_nextn_predict_layers=1`；`index_share_for_mtp_iteration=true` | 存在复用单个 MTP 层的设计；不说明用户启用步数或实现路径 |
| 输出头 | vocab 154880；`tie_word_embeddings=false` | 全词表 logits、采样、验收及输出是产品关键路径候选，不能遗漏 |

Indexer 显式模式统计：21 个 `full`、57 个 `shared`。0-based Full 层为 `0,1,2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,66,70,74`。即前两层各自独立，从 layer2 起每个 Full 为随后三个 Shared 提供选择结果。权重索引可见 Full 层的 `self_attn.indexer.*`，Shared 层没有这些权重。

权重索引含 `model.layers.0` 到 `model.layers.78`；其中 0–77 对应配置的 78 主层，额外 layer78 含 `eh_proj`、`enorm`、`hnorm`、`shared_head.norm`、独立 indexer 和 MoE 权重，与一层 MTP 配置一致。不能由键名推断当前服务加载或执行了这层。

官方 HF API 在该 revision 给出总参数统计 753,329,940,480（BF16 753,329,921,024 + F32 19,456），权重索引 `metadata.total_size=1,506,659,919,872 bytes`。这是整个 checkpoint 的统计，不等同于主干参数口径或每步活跃参数。GLM-5 老报告的 744B/40B 不能直接替代该统计；本研究没有自行反推出 GLM-5.2 活跃参数。

## 3. Attention / IndexShare 的性能含义

语义可用图：本层 hidden ready → Q 低秩 / KV 投影 → 本层 sparse MLA；Full 层另外生成 indexer 分数与 top-k，Shared 层消费此前 Full 的 top-k。Attention 输出之后进入本层 FFN/MoE，再产生下一层 hidden。跨层 hidden recurrence 是真依赖；共享 top-k 不使四层主干计算自动独立。

Transformers 源码 `GlmMoeDsaAttention` 明确按配置在 Full 层建立 indexer，Shared 层接收 `prev_topk_indices`，若缺失直接报错。它证实“复用选择集合”，并非复用主 attention 的 KV 或整层输出。Indexer cache 与 MLA cache 是不同状态：Full indexer 投影每 token 128 维 key，主 attention 各层维护自己的状态。

既有模型的 IndexShare 应计入 Current，不可再次当作未来可获得的 75% indexer 节省。官方模型卡的 2.9× FLOPs 是 1M 条件下与原结构的口径，不能当用户吞吐收益或框架调度上限。

对于框架研究，候选是 Q/KV/indexer 真实可并行分支、full→shared indices 生命周期、分支 join、通信提交时刻、Graph 范围与稳定地址等。保留相同选中结果和现有算子服务实现；不要自行扩展共享比率、减少 top-k 或改变 attention 算法。本层的多分支重叠可能争抢资源，只有实测才说明收益。

## 4. MoE 与 TP16 / EP16

75 个主层都具有 routed expert 和 shared expert 分支。语义层面是 input ready → router/top-8 → dispatch → 专家工作 → combine，与 shared expert 从相同输入产生的结果相加。源码将 routed 与 shared 结果相加，说明有分支空间；它并不证明当前 Ascend 已有双流、可无代价重叠或固定通信次数。

若 EP16 均匀放置 256 个专家，则每 rank 16 个专家，这是条件算术，不是当前 placement 事实。Top-8 不保证每 token 均匀覆盖 rank；负载由实测路由分布、token packing、shared placement、容量处理与并行实现决定。TP16 和 EP16 不能简单当两次独立的 16 倍切分；需记录 Attention/MoE 各自通信组、张量分片、每阶段局部 shape 和芯间/卡间物理映射。

调度研究要区分集合通信的输入 ready、enqueue、实际到达、服务与完成。最晚 rank 可能逐层/逐批变化，不能搬 DeepSeek 的 rank7 假说。EP dispatch/combine 与 TP 通信、P→D 传输若共享互联或 device 资源，也不能以各自独立最优时长拼成系统上限。

## 5. Context / KV / PD 边界

官方 GLM-5.2 博客指出，IndexShare 降低计算后 KV 容量与 CPU/缓存管理仍会限制长上下文，并提到 LayerSplit 和缓存传输编排。这是官方服务的研究线索，不能认为用户部署的 vLLM-Ascend 已实现 LayerSplit、跨层 KV 广播或 FP8 KV。

用配置可构造一个仅供 payload 量级理解的条件估算：若主干 cache 确为每层 `(512+64)` 个 BF16 元素，21 个 Full indexer 各存每 token 128 个 BF16 元素，则每个逻辑 token 是 `78×576×2 +21×128×2 =95,232 bytes`。

| 逻辑历史长度 | 主干 latent+RoPE（GiB） | Full indexer keys（GiB） | 合计（GiB） |
|---:|---:|---:|---:|
| 8192 | 0.686 | 0.041 | 0.727 |
| 32768 | 2.742 | 0.164 | 2.906 |
| 131072 | 10.969 | 0.656 | 11.625 |
| 1048576 | 87.750 | 5.250 | 93.000 |

以上是**条件逻辑账本**，没有计 MTP、padding、block 粒度、元数据、allocator 空洞、复制、TP/CP 分布、KV quant scales、临时 buffer 或预留；既非实际单 rank 显存占用，也非实际 PD 线速 payload，不能直接除以 16。当前 W8A8 描述权重/activation，不自动说明 KV cache 是 INT8。

Transformers 参考代码本次观察到在 attention forward 中展开 K/V 并送入其 Cache API；它只适合核对语义，不可把该缓存实现当 vLLM-Ascend 的 MLA 压缩 cache 事实。应由实际 backend、cache spec、传输 tensor 及字节计数确定 PD 数据布局。

GLM PD 交接需要恢复的状态至少包括实际主干 cache、indexer cache、block tables/positions、请求与slot世代。用户已明确GLM采用MTP路线；现场仍需确认其启用参数、seed/hidden/index/KV状态与首个draft的准备边界。每项状态可能需要传输、可由D重建、通过共享cache获取，或在当前实现中不需要跨端提供；不能先把它们全计入实际传输字节。研究要闭合从P最终producer到D首个consumer的依赖，不能只写“传完KV即可”。

## 6. MTP：模型事实和运行实现要分开

官方 GLM-5.2 博客描述跨草稿 step 复用首步 top-k，并用 KVShare 复用来自 target hidden 的历史状态；各 step 参数共享。其 7-step ablation 用 GLM-5.1 backbone/训练数据，最终接受长度提升 20%，属于该实验，不是用户 GLM-5.2 W8A8 的实测接受长度。

该事实为研究提供草稿循环、top-k/KV生命周期与target→draft准备的线索。用户明确使用MTP路线，但实际启用参数、步数、验收机制、index/KV复用是否落地、如何处理拒绝/回退、Graph是否包含完整循环、P→D是否转交MTP所需状态均待核验。一个训练MTP层可以被重复调用，不能由`num_nextn_predict_layers=1`推出只生成一个候选。KVShare也不等于无条件直接共用主干attention KV。本次Transformers主干代码只构造`range(num_hidden_layers)`，未发现完整MTP执行入口，不能靠config的一个flag宣称它实现了完整推理路径。

本轮框架调度极限研究可固定现有 MTP 算法与轨迹作为一个比较类，探索准备、缓存复用、同步、Graph 和交接的实现顺序。若 Agent 另提步数或策略变化，应单独声明比较类和工作量变化；不要把其混成同 work 的调度节省，也不要永久禁止探索新的合法调度策略。

## 7. 对当前开放研究范围的建议

建议目标是：在明确的硬件资源、模型/量化与负载条件下，使用现有正确算子服务实现，最小化完整 PD 产品路径的可实现时间。评估期间可以声明多个并行布局、batch、Graph、通信编排或请求调度比较类；不把第一份配置写成项目的永久限制。

未来P/D实例池数量、每实例并行配置、角色/实例到物理服务器的映射独立可变；用户提出1P3D等后续组织作为探索方向。1P3D表示一个P实例和三个D实例，不自动等于四台物理服务器，也不限定现有两台服务器永远不扩容。每次比较恢复实际实例/设备放置与总资源预算，不预设一角色或一实例占满一台机器。

GLM的外部图应关联请求到达/编码与prefix判定、P排队/分块/主干各层、cache ready、PD发送/接收/状态绑定、D首步、连续执行、输出发布/slot释放，并把TP/EP、CPU、队列、NPU和互联资源加入约束。D内部按实际target/MTP/提交/state的producer-consumer关系展开，不预定固定顺序。候选方向由关键路径和测量不确定性决定，模型结构只提供线索。

可以直接延续 DeepSeek 框架支线的 E_must / E_safe、合法 witness、Current/下界/可执行上界和完整 E2E 裁决。无需先完成严格硬件峰值或所有 compulsory work 的证明才开始框架优化。共享 top-k 和 MTP KV、buffer 世代、P/D 写入/消费完成是真依赖，现有 Python/Graph 边界和同步未必是真依赖。

性能证据需按 P/D、prefill/decode、dense/MoE、Full/Shared、启动/稳态/尾部、MTP 当前配置、实际 batch/context/routing 分型；不能把一个短窗口成本外推所有请求，更不能把并行资源重叠的时段相加当节省。

## 8. 必须保留为 UNKNOWN 的当前环境信息

1. W8A8 checkpoint 路径、config、量化范围、scale/zero与runtime格式；非原始 BF16 仓库。
2. 用户说的 0.27 对应 vLLM/vLLM-Ascend 各自包版本、commit、patch，以及是否 dev/custom build。
3. 当前P/D实例数、角色/实例到两台同规格服务器的放置、各实例实际rank/通信组与进程到卡/逻辑芯映射；DP1/TP16/EP16分别在哪些实例生效及资源是否共用。每台8卡16芯是当前计划解释，不能据此宣布两端已经各有16个运行rank。
4. Attention/MoE/Indexer/MTP 的实际 forward 类、backend、cache dtype/layout、Full/Shared 与 step 复用实现。
5. MTP实际启用参数与步数、PD connector/传输协议、跨rank映射和发送完成/消费完成/安全回收语义；用户已确认研究路线采用MTP。
6. Prompt/output/concurrency/arrival/cache-hit 分布、当前 TPS/TTFT/TPOT、SLO；1M支持不代表应只跑1M。
7. 910C 具体逻辑芯布局、内存容量、芯间/卡间/机间链路、通信组、CPU/NUMA/绑核、共享链路竞争。
8. 真实 trace 中 Host enqueue、device ready、HCCL peer wait、PD 传输与 D 首步的关键路径及 observer 影响。

这些未知定义核验对象，不定义固定研究 SOP。Agent 可以先复用可取得的日志/源码/trace，选择最小能区分机制的实验，再根据结果调整研究顺序。

## 9. 本地证据校验

本次下载/读取文件在 `work/model-reference/GLM-5.2/`：

- `config.json` SHA256 `185f93ee6d12548e16a847e279dc0c3c90b1524c970b0866b42fb545747d859a`
- `model.safetensors.index.json` SHA256 `5fd47a926aefce0f2c917f42523e5e0f3c87e23e389e767c3681536a62f5cf5e`
- `modeling_glm_moe_dsa.py` SHA256 `e62d5eec32e96fd3441db67ef7f595f3d37af6feace5eccc4e26e13fa6ef17dc`
- `blog_glm52.js` SHA256 `a899d32e1106cfe99d22618f6ed27ee18277863e3a1da2ded32311855635941c`

Transformers tag `v5.12.0` 的 annotated tag object 是 `8dd07d885b605406df0ffe2614b106f099aa036e`，其指向的 commit 为 `e0e7504bca2bfd1b85bb0eedb148f7b250226f06`。该模型配置 `transformers_version=5.12.0`，但用户运行的推理实现仍需独立记录。

## 10. 如何转化为开放研究，而不固化成SOP

模型特性是生成问题和解释结果的依据，不能直接决定优化优先级。Agent可以做有价值的新结构原型，再用执行证据修正假设；不要求在动手前证明所有收益，也不要求先填完一张全模型图。

| 模型或系统特性 | 有价值的研究问题 | 不应先作的假定 |
|---|---|---|
| 78层、Full/Shared Indexer与跨层hidden依赖 | 哪些Host/metadata重复、shared indices materialization或buffer边界拖慢实际consumer；现有Graph是否覆盖合适段落 | Shared层可以跳过attention；四层之间自动无依赖；已有IndexShare算未来收益 |
| 主模型与MTP、跨步IndexShare/KVShare | target→MTP准备、循环状态、提交与Host可见计数在哪里串行；是否存在合法复用或更好的Graph/queue组织 | DSpark count mirror可搬；KVShare等于直接共用主干KV；训练层数等于候选长度 |
| Top-8 MoE、shared分支、TP16/EP16 | 每批的局部shape与route skew如何改变最晚rank；通信、shared/routed与其他分支在哪些条件下值得重叠 | EP均衡、固定慢rank、通信时长全是链路服务、独立最短耗时可无代价重叠 |
| 大模型W8A8与长上下文cache | batch/准入、shape桶、workspace与cache预算怎样共同限制连续运行；固定现有算子下如何组织调用 | 全部tensor均INT8、KV为C8、每种shape单位成本相同、TP自动分摊全部cache |
| PD与MTP首步 | P端哪些状态真正要交接，D绑定/重建成本与传输何者限制首步；能否在合法producer ready后提前启动消费者 | 所有列举状态都需传输；发送提交等于D可消费；只计纯传输就闭合PD链路 |
| 大词表、采样、思考与工具协议 | logits/采样/提交/输出如何影响真实tokens与到达；模板、缓存、计数是否增加Host关键路径 | 输出chunk等于token；改变reasoning effort或输出长度就是调度优化 |

源码和trace支持的比较可以包含同合同的执行重排、同硬件预算的新拓扑或Runtime，以及不同合法MTP配置的操作区间。batch、并发、P/D资源分配、部署/并行组织、Graph边界、请求准入与补位均可探索；每次结果明确比较口径。MTP步数或到达条件变化会改变工作量与成本，需要单列结果，不能永久封死这些变量。

W8A8的资源预算需要具体checkpoint。若仅作量级示例，把上述约753.33B checkpoint参数全部假设为8bit，权重约753GB，16芯完全均匀分片约47GB/芯；该算术不是实际部署显存或严格容量下界，未计混合精度、scale、对齐、复制、workspace、cache及P/D各自权重副本。实际加载与cache布局比模型宣传参数更适合决定batch和准入范围。

模型卡面向长上下文、长输出和工具工作流，性能研究仍可从当前实际负载开始，再按问题选择短/长context、短/长output、冷/暖prefix、batch与到达条件。研究可以特化，但应写明适用范围；不把1M、固定c12或旧32K场景设为唯一验收条件。用户不做5.2/5.3精度比较；保持当前计算、请求完成、状态和token计数是有效性能证据的条件。

最终需要说明最佳已验证执行为何接近当前可实现边界、哪些假设已被证伪、剩余哪里未知。无法识别数值界时如实记录，不能由模型结构、理论FLOPs或一次KEEP宣布全局调度极限。
