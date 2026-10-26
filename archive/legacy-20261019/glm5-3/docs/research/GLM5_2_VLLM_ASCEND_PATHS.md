> 研究快照：整理于2026-10-01，随后作为按需资料发布；文中未提交/未运行陈述描述研究采集阶段，不代表后续Git发布状态。实际运行状态以局部HANDOFF和Git为准。

# GLM-5.2 MTP、W8A8 与 vLLM-Ascend PD 调度研究

研究日期：2026-10-01。只做公开源码、官方文档与本地固定模型参考的调查；未访问用户远端服务、未运行性能实验。本文作为可按需读取的源码索引；下面的路径不是用户现有安装的确认，也不是固定优化顺序。

## 1. 来源与适用边界

- 官方 vLLM main 固定 commit：`4c2d277643e217344056e1d2c42115d5f005912f`，commit 时间 2026-10-01T03:54:44Z。
- 官方 vLLM-Ascend main 固定 commit：`a8fcedb03d93e60efceddbfc912406f7fa491d57`。下载文件与路径清单保存在 `work/glm-official-source/`。两仓库独立 main 快照仅用于代码阅读，未验证这一组合可实际运行。
- 官方 Ascend 已有专门的 [GLM-5.2 部署文档](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/docs/source/tutorials/models/GLM5.2.md)，较本地 `work/GLM5-vllm-ascend-pinned.md` 的 GLM-5/5.1 文档更直接。后者保留为历史参考，不能作为 5.2 支持、现有参数或当前软件行为的确认。
- 本地官方 GLM-5.2 BF16 参考 config 与 checkpoint index 固定 HF revision：`cf457fa734ab149ffef225f80893eb38c6ff5cdc`，见 [config](https://huggingface.co/zai-org/GLM-5.2/blob/cf457fa734ab149ffef225f80893eb38c6ff5cdc/config.json) 和 [checkpoint index](https://huggingface.co/zai-org/GLM-5.2/blob/cf457fa734ab149ffef225f80893eb38c6ff5cdc/model.safetensors.index.json)。实际 W8A8 checkpoint 身份和逐层量化描述尚未读取。
- 公开 tag 查询发现 vLLM `v0.27.0`=`4bdc8a788d2e2ce9165d552b3d4d8b72604626bf`、`v0.27.1`=`6e448d0ea9bf3d88d898b65449ca6dc2aec170ac`；vLLM-Ascend tag 列表发现 `v0.27.1rc1`=`3b31886237ed65c435e0965b10e8002ff5769766`。这些 refs 与所读 main 不同，不能把用户说的“0.27”替换成其中任何一个版本。需要分别保存 vLLM、Ascend 插件、镜像与本地 patch 身份。[vLLM tags](https://github.com/vllm-project/vllm/tags)、[Ascend tags](https://github.com/vllm-project/vllm-ascend/tags)

用户已明确：GLM W8A8采用MTP路线、当前有两台同规格910C服务器且卡相同、PD已运行、DP1/TP16/EP16、版本标识0.27。结合此前8卡16芯口径，本计划按每台8卡16芯理解，这是当前计划解释，不是远端核验结果。当前P/D实例数、角色到服务器的放置、各实例实际rank/通信组与进程映射，以及MTP启用参数、每端K、runner与connector、KV dtype、Graph mode仍未核实，不能猜。

## 2. GLM-5.2 的 MTP 与 DSpark 的关系

参考 config 的 architecture 是 `GlmMoeDsaForCausalLM`，model_type=`glm_moe_dsa`；78 个主层、`num_nextn_predict_layers=1`、hidden size 6144、256 routed experts、每 token 选8个 expert、1 shared expert。DSA 配置含 index_topk=2048、index_topk_freq=4、index_skip_topk_offset=3 与 index_share_for_mtp_iteration=true。

**一个训练 MTP 层不等于只能生成一个 draft token。** vLLM 的 MTP predictor 通过 `spec_step_idx % num_mtp_layers` 选层；生成多步 draft 时可以复用这一层。深度 K 来自实际 speculative 配置/调度，而不是 config 中的层数。官方 config 的 ep_size=1 也不覆盖用户启动的 EP16。[MTP 模型实现](https://github.com/vllm-project/vllm/blob/4c2d277643e217344056e1d2c42115d5f005912f/vllm/model_executor/models/deepseek_mtp.py)

当前 upstream config 识别 glm_moe_dsa，将 draft 结构转换为 DeepseekV32MTPModel；配置别名 deepseek_mtp 归一化为 mtp。Ascend 模型注册把该 MTP architecture 接到 `AscendDeepSeekMTP`，GLM target 接到 `AscendGlmMoeDsaForCausalLM`。这表示复用 DeepSeek-style MTP 的实现族，不表示使用 DeepSeek V4 的 DSpark7。[配置转换](https://github.com/vllm-project/vllm/blob/4c2d277643e217344056e1d2c42115d5f005912f/vllm/config/speculative.py#L660)、[Ascend 注册](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/models/__init__.py#L71)

MTP 的输入包括 token embedding、position、上游 hidden state；embedding 与 hidden 各自归一化后拼接并经过 eh_proj，再进入 MTP decoder block。模型返回供 logits 与下一 draft step 使用的状态；应核验实际版本的 tuple/归一化约定，不能把 logits hidden 与下一步 hidden 当成任意可互换 buffer。

参考 checkpoint 的第78层含 eh_proj、enorm、hnorm、MTP MoE 权重和 shared_head.norm；未见 shared_head.head.weight。当前 Ascend 用实际 loaded_weights 判定 MTP 是否持有独立 LM head，缺失时让 proposer 共享 target head。module 已构造不等于 weight 已加载。W8A8 checkpoint 是否含独立 head、rotation 或特殊 mapper 要另验，不能从 BF16 索引推断。[Ascend head 与 rotation 处理](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/models/deepseek_mtp.py#L16)

## 3. 提案、验证、提交的真实接口

### V1 runner 的定位入口

当前源码的 mtp 方法经 `get_spec_decode_method` 进入 AscendEagleProposer，再走共享 Ascend proposer。需要追踪真实构造对象；名字带 Eagle 并不能证明算法是 EAGLE。[方法分发](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/spec_decode/__init__.py#L41)

`worker/model_runner_v1.py` 的关键边界：

| 边界 | 源码定位 | 调度问题 |
|---|---|---|
| 组 target 输入与验证索引 | `_calc_spec_decode_metadata`、`_prepare_inputs` | 哪些列表/索引/CPU→NPU copy 可复用或提前；真实 scheduled token 数和 graph padding 分开 |
| target forward 与 connector | `execute_model` 的 forward context | KV是否已ready；哪些 wait是必要；Graph是否包含当前 batch |
| 验证/采样 | `_sample` → `AscendRejectionSampler` | 接受prefix、recovery/bonus、计数何时在设备上ready；有效tokens≠draft tokens |
| 下轮 MTP 输入 | `propose_draft_token_ids`、prepare_inputs_padded | 从接受结果选token/position/hidden；拒绝后seq_lens/slot不能沿用乐观值 |
| draft 执行 | `_propose` → `_run_merged_draft` | 各 draft step 的Host准备、通信与真正数据依赖 |
| 控制与发布 | `_bookkeeping_sync`、`_copy_valid_sampled_token_count`、`_copy_draft_token_ids_to_cpu` | 真实event/sync位置；哪些CPU消费可延后或异步 |
| KV完成 | `finalize_kv_connector` | 当前路径在draft后finalize；能否提前必须由target/MTP/cache消费者决定 |

这是一张定位表，不要求按表顺序研究。函数名带 sync 也不能单凭名称当作可消除 wall；需核对其当前异步分支和真实消费者。[V1 runner](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/worker/model_runner_v1.py#L1887)

当前验证 sampler 从 target logits 中区分 target 与 bonus 行，按 sampling constraints 运行拒绝采样，输出各请求有效token以及占位值；greedy/random、logprobs、grammar等分支并不一样。不改变当前算法语义的优化可以重组消费与拷贝，但不能假定所有请求均接受K步，也不能用SSE chunk、verify行数或全部draft计数当产品token。[Ascend 验证/采样](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/sample/rejection_sampler.py#L146)

Scheduler 维护 num_computed_tokens、scheduled draft、output placeholders和KV分配；输出反馈修正乐观进度。新draft提交和被拒token后的状态校正，应作为完整闭环观察。源码提供 update_draft_token_ids、远端KV等待/恢复等接口。[Scheduler](https://github.com/vllm-project/vllm/blob/4c2d277643e217344056e1d2c42115d5f005912f/vllm/v1/core/sched/scheduler.py)

### V2 runner 必须另辨认

当前 main 同时有 V2：`AscendMTPSpeculator` 继承共享 `AscendAutoRegressiveSpeculator` 和 upstream MTPSpeculator，使用不同 InputBuffers、metadata与Graph manager。不能把 V1 patch/关键路径直接套在 V2 上。安装是否使用V2须从启动日志、配置和实际class查证。[V2 MTP入口](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/worker/v2/spec_decode/mtp/speculator.py)、[V2共享执行](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py)

## 4. Graph 与 DSA index 重用

当前 **V1** proposer 检测 GLM family 时强制 draft eager，保留 target Graph 设置。因此 target FULL_DECODE_ONLY 与 speculative enforce_eager=true 可以并存；不能说两者矛盾，也不能从 target graph日志证明draft在replay。[V1 Graph gate](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/spec_decode/llm_base_proposer.py#L238)

当前 V2 共享执行实现可组织 draft prefill 与多步decode捕获，使用 AutoRegressiveAclGraphManager；speculative enforce_eager 仍可关掉draft Graph。这说明分支具有不同实现能力，**并不证明GLM-5.2在用户版本/量化/attention backend上可安全开启draft Graph**。若研究突破原有boundary，须重验metadata、seq_lens、slot、buffer、collective与拒绝处理。[V2 draft Graph](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/worker/v2/spec_decode/autoregressive/aclgraph.py)

GLM-5.2 的 checkpoint-level shared indexers 与运行时MTP步内index重用是两种关系。Ascend patch 用 indexer_types 判断真正省略的indexer，MTP首步须计算自己的indices；后续步可按 index_share_for_mtp_iteration 重用并compact有效请求行。V2 upstream MTP另有PCP=1等条件。研究不能把跨主层index共享、target→draft共享、同一MTP的step0→step1+重用混为一谈，也不能用use_index_cache开关独自证明工作量。[Ascend GLM/DSA patch](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/patch/worker/patch_deepseek_v2.py#L41)、[V2 MTP index 生命周期](https://github.com/vllm-project/vllm/blob/4c2d277643e217344056e1d2c42115d5f005912f/vllm/v1/worker/gpu/spec_decode/mtp/speculator.py)

## 5. PD 的 KV/MTP 状态与 handoff

官方 GLM-5.2 文档给出多种PD拓扑、MooncakeConnectorV1，也有MultiConnector叠加AscendStore的示例；部分示例的P与D设置不同MTP步数，D设置target FULL_DECODE_ONLY。它们是部署示例而非对用户现状的推定。**不能据此猜用户P=1、D=3/5、已经使用C8/MLAPO/MC2，或建议直接复制官方多节点资源拓扑。**

用户提出未来可能采用1P3D等组织。P/D实例池数量、每实例并行配置和角色/实例到物理服务器的映射独立可变；1P3D不自动意味着四台物理机，也不把现有两台服务器设为永久不可扩容的上限。部署描述应恢复当前实例与rank放置，后续候选按实际总资源、通信组和connector映射比较，不预设一实例一服务器。

PD接口需要跟踪两条链：

1. Scheduler：get_num_new_matched_tokens → KV block allocation → build_connector_meta → 远端KV等待 → finished_recving → 请求可执行；P端request_finished可延后free，需收到传输完成信号。
2. Worker：register_kv_caches → start_load_kv → 实际传输/错误报告 → wait_for_layer_load与save_kv_layer（若该connector实现）→ wait_for_save/finish_forward → get_finished。基类具有接口不等于所有connector都逐层执行。[KV Connector base](https://github.com/vllm-project/vllm/blob/4c2d277643e217344056e1d2c42115d5f005912f/vllm/distributed/kv_transfer/kv_connector/v1/base.py)

当前 Ascend保留旧monolithic MooncakeConnector与新的pull facade路径。后者按layer name、spec、KV group、TP/PP/PCP/DCP映射做匹配；新push connector只是架构声明，尚未实现。需要从实际connector module/class选择源码分支，不能看到新目录就假定用户已经运行这一实现。[新 facade](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/connector.py)、[pull worker](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/pull_worker.py)

旧实现显式预留MTP metadata plane，避免producer/consumer不同draft配置打乱SFA/index映射；mtp路径计入draft cache layer，prompt KV block传输又排除仅为speculation预留的extra blocks。**draft深度、draft层、物理KV/cache group与传输块不是一一等价。** 重排handoff必须先辨认GLM target/MTP/index缓存、已生成P侧token、有效prompt长度和各端block/stride/dtype。[旧 Mooncake实现](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py#L519)

**KVShare 不等于无条件让 MTP 直接共用主干所有KV。** 公开 V1 proposer 对某些split indexer cache存在物理共享的说明，但仍要求对应draft metadata builder；这不能证明全部attention KV被共享。模型设计中KVShare的来源层、投影、可见长度与state版本，以及安装实现中的cache alias/group、slot/block和PD传输，必须逐项对应。多个MTP步复用一个draft cache层，也不等于该层无须初始化/传输、不发生写入。[draft cache与metadata区分](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/spec_decode/llm_base_proposer.py#L391)

V1 target forward启用speculation时可defer connector finalize，draft后再finalize，以包含draft KV save/put。它是候选关键依赖，不应未经分析移到target结束就宣布KV ready。公开旧代码的某些GQA reformat还保留全NPU synchronize并注明原因未识别，不能把该代码存在当作GLM MLA路径必经开销。

## 6. W8A8 对调度成本的影响

W8A8是格式概括，不足以唯一确定逐层执行。当前 ModelSlim根据逐层描述选择FLOAT、静态/动态量化、linear/MoE/attention与C8 cache等scheme；实际量化metadata和已加载方法必须记录。[ModelSlim dispatch](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/quantization/configs/modelslim_config.py#L601)

公开动态W8A8 linear进行per-token activation quant并调用量化matmul；静态W8A8使用per-tensor activation参数，MoE另有GMM、activation/scale和各backend组织。其shape成本、workspace和scale生命周期不同。W8A8不自动表示KV=int8；官方W8A8C8示例与用户W8A8必须区分。[动态W8A8](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py)、[静态W8A8](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/quantization/methods/w8a8/w8a8_static.py)

本阶段把量化/既有融合/communication backend作为当前primitive实现。可以研究调用顺序、Host issue、queue、现有buffer、shape/batch、Graph边界和通信组织；不开发更快quant/GMM/attention/fusion kernel。改变batch或K使verify、MoE route、index成本变化时，重新测shape-conditioned cost，不能把DeepSeek W4A8/TP8或某一次GLM shape的耗时视为常数。若切换已有backend或融合开关改变primitive，需要另标实验类别，不能全部记成调度收益。

## 7. 延续 DeepSeek 成果的开放候选

| 可迁移机制 | GLM上先查的适用条件 | 不宜照搬的部分 |
|---|---|---|
| 一次bootstrap、稳定buffer、局部接管热路径 | current runner是否已有稳定input/KV/metadata；是否有重复构造 | DeepSeek固定c12/cohort、67cache张量、DSpark7语义 |
| 异步状态镜像、缩小同步消费范围 | accepted count在设备ready到CPU真正消费者；动态结束/拒绝/补位 | 无证明把GLM状态延迟一cycle；DSpark common refresh字段 |
| Target replay与metadata控制 | 当前target bucket/padding/fallback；V1/V2 attention backend | 把旧87.71%内部wall下降预测成GLM E2E收益 |
| Host/rank issue与通信关键路径 | 16rank最晚ready/enqueue和consumer；P/D共享链路竞争 | profiler HCCL duration当removable通信；无资源证据强行overlap |
| 输入编码/hash/准备复用 | 当前frontend/renderer/prefix命中及PD是否重复准备 | 用暖重复prompt收益代表冷或唯一prompt |
| 输出drain与continuous refill | accepted/stateready、client发布、slot释放、下一KV入场 | bulk完成替代流式延迟合同；固定cohort成为永久结构 |

所读GLM MTP实现通过nextN层形成逐步draft依赖；DeepSeek DSpark的三trained-layer closure、DSpark7专属输入、draft/cache、count维护和acceptance更新不能直接搬。旧DeepSeek任务也不能被误写成另有通用Mamba/GDN recurrent state。可以迁移“减少无必要Host控制/状态镜像”和“从真实DAG移除额外边”的方法，不能假定草稿结构、接受率、buffer及可重叠窗口相同。

不预定先优化D、MTP、Graph或PD传输。根据当前关键路径选择最大的可验证Gap或最有信息价值的unknown；可以直接做已有证据支持的结构改动，同时补足因果证据。暂不能数值证明调度ceiling时明确写unknown，不等待完整全模型硬件C⁺证明。

## 8. 下一步最低信息需求

应从用户正在跑的现场恢复以下材料；这些可由简单只读脚本/`zcode --prompt`提取，不需要先运行新benchmark：

1. 当前P/D实例数、P/D/proxy的实际启动命令与effective config、角色/实例到两台同规格服务器的放置、各实例rank/通信组与进程到物理卡/逻辑NPU映射，vLLM/Ascend commits、镜像digest与patch；确认0.27的具体身份与runner V1/V2。每台8卡16芯仍是计划解释，运行rank与设备占用以恢复的现场为准。
2. 实际W8A8模型config、量化描述、weight index身份；MTP是否启用、各端method/K/maxlen/采样配置、是否dynamic speculation；draft/target class、Graph日志与attention backend；不要求评估跨版本精度。
3. connector/proxy的实际class/module与KV配置：KV/index dtype、block size、cache groups/布局、Ascend direct/网络、是否MultiConnector/Store、P生成及handoff token语义。
4. 已有Current与原始benchmark/trace入口：真实输入输出与前缀条件、并发/到达、outputTPS/TTFT/TPOT、accepted tokens/cycle、draft步耗时、排队/KV等待/first-D及16rank时间；先读现有证据，再选择最小新增测量。

拿到这些信息后，才把上面的源码候选映射到用户真实执行路径。保持现有活动实验和未提交改动；不把文档示例升级为用户环境事实。
