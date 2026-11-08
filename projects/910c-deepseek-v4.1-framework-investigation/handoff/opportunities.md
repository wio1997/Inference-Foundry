# 融合优化逐项说明

每项按介绍、进度、结果阅读。展开可查代码、适用条件和证据。[更新方式](README.md#阶段说明与上传流程) · [早期 42 族](evidence/priority-42.md)

## C01

**M00、M11 · 未变block表跳过/增量发布**

- **介绍**　标记真实变更的 block 表行，只上传脏区，保持各组物理表地址和图回放后缀。把减少的 H2D 时间与少提交的关键路径时间分别计价。
- **到了哪里**　待确认。初步判断。待定点复核
- **结果**　部件预算：C8 旧整批 23.66 ms（0.127%），仅计上传数据。

<details><summary>代码与依据</summary>

- **代码定位**　`block_table.py:298` `commit_block_table`、`block_table.py:519` `commit_block_table`、`model_runner_v1.py:1281` `_prepare_inputs`
- **适用负载**　13组、B≤4；dirty与graph后缀状态保留
- **成立条件**　全部13×453可删仍需额外净提交162.24ms，≥27.55µs/copy
- **证据**　[S02 代码与定价](evidence/S02.md)
- **接续**　按实际调用点核对剩余可删的提交时间。关联 M11

</details>

## C02

**M00、M11 · 请求/接受数/ID/位置/长度/spec多输出publisher**

- **介绍**　遍历一次请求映射与接受数，同时写出新 ID、位置、长度和 spec 元数据。扣除仍有读者的 CPU 镜像及各目的地必需的写入。
- **到了哪里**　待确认。初步判断。待定点复核
- **结果**　设备端上界：已定价部件全免为 61.73 ms（旧整批 0.332%）。

<details><summary>代码与依据</summary>

- **代码定位**　`gpu_model_runner.py:1790` `_prepare_input_ids`、`model_runner_v1.py:1771` `_calc_spec_decode_metadata`、`model_runner_v1.py:1876` `_copy_valid_sampled_token_count`、`llm_base_proposer.py:2192` `prepare_next_token_ids_padded`
- **适用负载**　target输入到seed/count及下一轮；前缀/验证各守合同
- **成立条件**　其余净省≥124.17ms/旧整批；raw ID/CPU镜像/分目的count不能丢
- **证据**　[S01 代码与定价](evidence/S01.md) · [S02 代码与定价](evidence/S02.md)
- **接续**　找出必须保留的 CPU 镜像和索引读者。关联 M00

</details>

## C03

**M00、M11 · 共同CP计划与按组slot多输出批写**

- **介绍**　让同轮 slot 写入共用计划和输入，将 26 项工作批量提交，保持每个 cache bank 的地址、无效行和代次。复用历史46/F02 的实验。
- **到了哪里**　已确认可实现。服务实测·限定基线。复用；小批量另查
- **结果**　历史服务实测：B4 两组整批耗时分别下降 3.39% 和 2.22%。

<details><summary>代码与依据</summary>

- **代码定位**　`block_table.py:469` `compute_slot_mapping`、`block_table.py:504` `compute_slot_mapping_draft`、`model_runner_v1.py:3561` `_get_block_table_and_slot_mapping`
- **适用负载**　C8 B4、真实26-job caller；其它job集合另判
- **成立条件**　不是与CP+slot、CPUplan其它基线相加；B1/empty需caller净省
- **证据**　[S03 代码与定价](evidence/S03.md)
- **接续**　复用；小批量见历史46/F03。关联 M00

</details>

## C04

**M00、M10、M11 · target/draft共用CPU DP计数协调packet**

- **介绍**　在目标模型与草稿的 DP 计数可预测时共用一次协调结果。统计真正位于关键路径且可取消的第二次协调，各 DP 保留一致回退。
- **到了哪里**　待确认。初步判断。待定点复核
- **结果**　门槛：旧整批若命中 200 次，每次净省至少 0.930 ms 才达 1%。

<details><summary>代码与依据</summary>

- **代码定位**　`model_runner_v1.py:825` `_sync_metadata_across_dp`、`model_runner_v1.py:3325` `_determine_batch_execution_and_padding`、`llm_base_proposer.py:1013` `_propose`
- **适用负载**　固定K5、padded eager draft、角色/计数可预测
- **成立条件**　所有DP一致fallback，保留EP顺序；CPU barrier等待未必可消失
- **证据**　[S03 代码与定价](evidence/S03.md) · [S10 代码与定价](evidence/S10.md) · [S11 代码与定价](evidence/S11.md)
- **接续**　核对各 DP 是否能一致决策及回退。关联 M10/M11

</details>

## C05

**M06、M08、M10 · 取消没有最终贡献的行：idle/小桶/末层/context分角色**

- **介绍**　按输出读者划分本地假行、末层有效行、已退休 context 行。只跳过无人消费的部分，同时保留远端专家服务、raw 快照和最后一次 cache 写入。
- **到了哪里**　待确认。初步判断。待定点复核
- **结果**　上界：末层 FFN 与读出合计 22.992 ms（长 prefill）；idle 波若有 100 次，旧整批达 1% 需净省 1.859 ms/次。

<details><summary>代码与依据</summary>

- **代码定位**　`prepare_finalize.py:252` `prepare`、`fused_moe.py:485` `_forward_impl`、`model.py:1505` `forward`、`dspark.py:295` `precompute_and_store_context_kv`
- **适用负载**　idle专家服务；B1/2；末层head行；retired context
- **成立条件**　末层先退役raw快照；≥64.89%上界净兑现；context须物理final writer/窗口/代际；不删远端专家
- **证据**　[S03 代码与定价](evidence/S03.md) · [S08 代码与定价](evidence/S08.md) · [S09 代码与定价](evidence/S09.md) · [S10 代码与定价](evidence/S10.md)
- **接续**　数真实 idle 波次；保留远端专家和 raw 读者。关联 M08/M10

</details>

## C06

**M01 · Engram hidden-independent lookup/WKV/统计结果缓存与发布**

- **介绍**　按完整内容键和权重版本缓存与 hidden 无关的 lookup、WKV 和统计结果。按层回放实际命中，再扣失效、未命中协调和容量成本。
- **到了哪里**　待确认。初步判断。待定点复核
- **结果**　门槛：本地周期降 1% 需消除原可缓存耗时的 71.1%–96.3%（中位 75.4%），另扣缓存管理。

<details><summary>代码与依据</summary>

- **代码定位**　`hash_state.py:79` `forward`、`embedding.py:333` `embed_gathered`、`embedding.py:378` `lookup`、`layer.py:34` `forward`
- **适用负载**　层1/14、TP8共享CPU-UVA表；完整tuple/权重代际key
- **成立条件**　还要扣cache/miss协调/容量；position重叠不是命中率
- **证据**　[S04 代码与定价](evidence/S04.md)
- **接续**　按层、权重版本及完整键回放

</details>

## C07

**M02、M05、M07 · mHC norm/coeff主计算、rounded producer→consumer常驻**

- **介绍**　核查 Gram、系数计算及工作区，寻找可共用的归约或可直接交付下一个 Pre 的舍入残差。保留独立的 Post、aux 与 Engram 读者。
- **到了哪里**　待确认。初步判断。待定点复核
- **结果**　预算：HcPre 占同周期约 5.10%；周期降 1% 需净删 HcPre 耗时的 19.6%。

<details><summary>代码与依据</summary>

- **代码定位**　`model.py:858` `hc_pre`、`model.py:871` `hc_post`、`model.py:879` `forward`、`hc_pre_cube_compute.h:398-477`
- **适用负载**　40层80次Pre；初始/Engram/aux/终端例外；draft6次扣重
- **成立条件**　保存FP32 post/comb、HF32与BF16边界；两版row-norm和QA-only原实现已关闭
- **证据**　[S06 代码与定价](evidence/S06.md)
- **接续**　定位可删主体并复核舍入；具体反例见独立 HC 记录。关联 M05/M07

</details>

## C08

**M02、M03 · SP row先投影，再gather窄KV/C1结果**

- **介绍**　在 SP 行所属卡先算 Wkv，将 H5120 的宽 AllGather 换成窄 KV。36 层传 512 维，层 20 传两份 512 维，三处 C2 源保留原路径。
- **到了哪里**　待确认。条件估算。待领原型
- **结果**　情景：长提示的四个 prefill 执行批次毛预算 60.54 ms，兑现 50% 省 30.27 ms（旧 TTFT 2.03%，旧整批 0.163%）。

<details><summary>代码与依据</summary>

- **代码定位**　`model.py:879` `forward`、`dsa_v41.py:327` `_project_kv`、`dsa_v41.py:497` `_write_compressed_source`、`dsa_v41_cp.py:189` `_prepare_inputs_and_caches`
- **适用负载**　长提示prefill；小decode保留旧图
- **成立条件**　净兑现≥24.64%；换M数值、raw-reader、packing/争用；非P90重算。用四个 prefill 执行批次原宽通信与重复投影的节省作毛预算，扣除新 pack、同步与资源争用。
- **证据**　[S07 代码与定价](evidence/S07.md)
- **接续**　验证单层 BF16、行序、cache 和净通信时间。关联 M02

</details>

## C09

**M03 · SMLA direct KV reader＋首partial交付；shared panel为替代**

- **介绍**　按原选择顺序从分页 cache 直读 KV，消除 selected KV 在显存中的暂存和重读；首段 PV 直接交付。把不连续读取导致的 QK/PV 降速计入净值。
- **到了哪里**　待确认。条件估算。待领原型
- **结果**　情景：有效 K=256、QK/PV 等速，TTFT 省 17.78 ms（1.19%），整批省 21.93 ms（0.118%）。

<details><summary>代码与依据</summary>

- **代码定位**　`dsa_v41.py:613` `_forward_attention`、`sparse_flash_mla_csa_block_vector.h:606-653`、`sparse_flash_mla_csa_block_cube.h:416-436`、`sparse_flash_mla_csa_block_cube.h:704-726`
- **适用负载**　长提示G64，真实有效K256/512
- **成立条件**　K256相对QK/PV读速≥约0.805或等速75.9%行；保持原顺序/容量。分别比较原暂存与直读路径的整体耗时，保留相同 mask、softmax 和 partial 顺序。
- **证据**　[S07 代码与定价](evidence/S07.md)
- **接续**　比较完整 kernel 的精度与耗时

</details>

## C10

**M03、M04 · SMLA/RoPE/OA直接产send布局，再交换窄OA结果**

- **介绍**　让逆 RoPE/OA 的生产者直接写发送布局，再交换较窄结果。已有显式 pack 变体关闭；按实际可覆盖形状核算额外权重与 KV 容量。
- **到了哪里**　待确认。初步判断。待定点复核
- **结果**　成本余量：理想 OA 前移后，新传输与融合的额外开销中位仅容许 0.214 ms；新变体净值待定。

<details><summary>代码与依据</summary>

- **代码定位**　`dsa_v41_cp.py:209` `_project_output`、`dsa_cp.py:77` `restore_tp_heads`、`dsa_v1.py:1732` `_wo_a_bmm`、`dsa_v1.py:1759` `_forward_o_proj`
- **适用负载**　需新合法shape覆盖与owner权重容量；旧guard命中0
- **成立条件**　stock+pack新增1.010ms已失败；全owner约2560MiB/rank不得挤KV
- **证据**　[S07 代码与定价](evidence/S07.md)
- **接续**　查实际可覆盖形状及权重额外容量。关联 M03

</details>

## C11

**M03 · QLI逐key压实评分与成员发布**

- **介绍**　压实有效 key 后评分，再按原 TopK 语义恢复顺序、并列和 filler。保留 gather、压实和恢复成本；四层共享成员转换的历史收益另有基线。
- **到了哪里**　待确认。初步判断。待定点复核
- **结果**　工作量：理想少 50.09% 评分 MAC；若暴露 223 波，旧整批达 1% 需每四次 QLI 净省 0.834 ms。

<details><summary>代码与依据</summary>

- **代码定位**　`indexer.py:181` `select_projected`、`dsa_v41.py:567` `_select_sparse_indices`
- **适用负载**　长C1 mode2；保持score/TopK/tie/filler/nonfinite
- **成立条件**　约旧1.981ms的42.1%；原member-only组合未过批量slot服务门槛
- **证据**　[S07 代码与定价](evidence/S07.md)
- **接续**　扣除 gather、压实和次序恢复

</details>

## C12

**M05、M06 · norm/router GEMM/路由的共同producer**

- **介绍**　让 norm、router GEMM 和 TopK 共用输入并直接交付路由结果，消除中间重读和提交。保留动态路由必需的点积与选路数值顺序。
- **到了哪里**　待确认。初步判断。待定点复核
- **结果**　预算：Router 与 RMS 合计占同周期约 1.88%；周期降 1% 需净删该段约 53%。

<details><summary>代码与依据</summary>

- **代码定位**　`model.py:844` `rms_norm_cast`、`fused_moe.py:401` `_apply_router_linear`、`fused_moe.py:412` `_compute_router_logits`、`fused_topk_router.py:140` `_compute_routing`
- **适用负载**　动态sqrtsoftplus384/K6；无固定hash表
- **成立条件**　小ID/TopK-only全免费仍低值；新GEMM/数值/替代成本必须计入
- **证据**　[S08 代码与定价](evidence/S08.md)
- **接续**　分开核算必要 GEMM 和可删交付。关联 M05

</details>

## C13

**M06、M07、M10 · shared/DFC、返回producer与草稿三层联合调度**

- **介绍**　分析共享专家与路由专家各自占用 Cube、Vector 和通信的时间，按能同时执行的部分重排；分别核算目标和草稿。K03 是其中一个具体实现。
- **到了哪里**　待确认。初步判断。待定点复核
- **结果**　情景：F21 独占资源且保留路由时可省 0.594 ms/周期，达到 1% 后新增成本余量中位仅 0.0427 ms。

<details><summary>代码与依据</summary>

- **代码定位**　`fused_moe.py:485` `_forward_impl`、`shared_experts.py:670` `forward`、`dispatch_ffn_combine_kernel.hpp:912-977`
- **适用负载**　target EP16和draft EP16分别计；所有必要expert服务保留
- **成立条件**　不能把wait-inclusive当全free；旧fork/分段/return/deferred/merge原变体停止
- **证据**　[S08 代码与定价](evidence/S08.md)
- **接续**　核对资源窗口；原 fork 不重测。关联 M07/M10

</details>

## C14

**M09、M10 · aux owner→main/norm→三个context KV共同发布**

- **介绍**　从 aux 所在卡组织 main、norm 和三套 context KV 的输入与发布，复用前缀读写。三套 KV 权重、规定的 BF16 边界和各 bank 的发布时点均保留。
- **到了哪里**　待确认。初步判断。待定点复核
- **结果**　预算：原 context 前缀 1.050 ms（同周期 1.897%）；周期降 1% 需净省该段约 52.7%。

<details><summary>代码与依据</summary>

- **代码定位**　`dspark.py:281` `combine_hidden_states`、`dspark.py:284` `_project_shared_kv`、`dspark.py:295` `precompute_and_store_context_kv`、`dspark_proposer.py:226` `set_inputs_first_pass`
- **适用负载**　三份pre-layer aux；保留main norm与三个不同KV权重
- **成立条件**　多付main权重约137.63MB/rank；新M精度、行退休、输入/银行协议；原primitive余量4.61µs
- **证据**　[S09 代码与定价](evidence/S09.md) · [S10 代码与定价](evidence/S10.md)
- **接续**　扣除权重移动、行重排和同步。关联 M09

</details>

## C15

**M09、M11 · 分离sample-copy-ready与draft/cache-write-retired**

- **介绍**　将 token 可交付事件与草稿完成、cache 可回收事件分开。按客户端原响应端点计提前量，后台继续完成必要写入并处理取消和页退役。
- **到了哪里**　待确认。初步判断。待定点复核
- **结果**　上界：C8 即使去掉全部可释放等待也仅省 12.807 ms；普通 code-single 整响应达 1% 需净省 4.666 ms。

<details><summary>代码与依据</summary>

- **代码定位**　`model_runner_v1.py:1876` `_copy_valid_sampled_token_count`、`model_runner_v1.py:2163` `_copy_draft_token_ids_to_cpu`、`model_runner_v1.py:2764` `sample_tokens`、`gpu_model_runner.py:345` `get_output`、`scheduler.py:1874` `update_from_output`
- **适用负载**　async无logprobs/grammar；客户端输出与安全回收分开
- **成立条件**　保持页回收/abort/stale/bank退役；其它请求按各自整响应1%
- **证据**　[S11 代码与定价](evidence/S11.md)
- **接续**　核对普通请求端点、页生命周期及取消路径。关联 M09

</details>

## K00

**M06 · 当前DFC megakernel**

- **介绍**　BF16 expert input + IDs/weights -> dispatch/EP/GMM1/activation/GMM2/return/unpermute。基线已融合routing quant/EP/GMM1/act quant/GMM2/return/unpermute。已有一调用；必要专家结果和通信仍做
- **到了哪里**　已确认可实现。既有实现。复用
- **结果**　基线事实；无第二份融合收益

<details><summary>代码与依据</summary>

- **代码定位**　`moe_comm_method.py:563` `fused_experts`、`dispatch_ffn_combine.h:244-287`、`dispatch_ffn_combine_kernel.hpp:912-977`
- **适用负载**　target384/K6和draft128/K3按各自参数
- **成立条件**　保留CombineV1/V2、权重/scales、mask、peer计数/flags与退休
- **证据**　[实现切口与公式](evidence/megakernel.md) · [逐窗口模型](evidence/megakernel-window-model.json)
- **接续**　作为参照，不重复计收益

</details>

## K01

**M02、M03 · Post/HcPre/RMS到本地KV-ready的前端**

- **介绍**　rounded Post -> HcPre -> RMS -> local KV/C1 projection/norm/RoPE -> KV-ready packet + Y/coeff outputs; Qb sibling outside base launch。局部转发和少spill/调用；owner重排另消重复投影与宽AG。保留rounded残差/迟到系数；窄KV交付是独立通信边界
- **到了哪里**　待确认。设计与条件估算。与C08联合评估
- **结果**　owner机制已有60.54ms毛包络，50%情景30.27ms；不是megakernel新增独占收益

<details><summary>代码与依据</summary>

- **代码定位**　`model.py:871` `hc_post`、`model.py:879` `forward`、`dsa_v41.py:327` `_project_kv`、`dsa_v41.py:350` `_preprocess`
- **适用负载**　长prefill；三C2源保持raw；入口/Engram/aux各守时点
- **成立条件**　净兑现≥24.64%旧TTFT目标；基础launch不塞Qb拖KV-ready；换M精度/tiling；只40个Attention Pre需净删39.05%旧耗时
- **证据**　[实现切口与公式](evidence/megakernel.md) · [逐窗口模型](evidence/megakernel-window-model.json)

</details>

## K02

**M03、M04 · SMLA direct/partial到inverse RoPE与可选OA send producer**

- **介绍**　selected KV -> QK/softmax/PV -> inverse RoPE -> owner OA -> send layout。KV gather/partial交接；更大分支输出payload65536→16384B/query。保留原partial合并/BF16边界和跨8head组的OA依赖；之后窄交换
- **到了哪里**　待确认。设计与条件估算。与C09联合评估
- **结果**　direct+partial K256等速旧TTFT条件约17.78ms；direct+OA联合收益未知

<details><summary>代码与依据</summary>

- **代码定位**　`dsa_v41.py:613` `_forward_attention`、`sparse_flash_mla_csa_block_vector.h:606-653`、`sparse_flash_mla_csa_block_cube.h:416-436`、`sparse_flash_mla_csa_block_cube.h:704-726`、`dsa_v41_cp.py:209` `_project_output`
- **适用负载**　长prefill G64；实际有效K；小shape不外推
- **成立条件**　QK/PV相对速度≥约0.805；新OA transport/fusion余量模型中位0.214ms；额外OA权重2240MiB/rank；原partial/rounding与资源账
- **证据**　[实现切口与公式](evidence/megakernel.md) · [逐窗口模型](evidence/megakernel-window-model.json)

</details>

## K03

**M06 · DFC内部Cube尾部承接shared GateUp**

- **介绍**　原 DFC 的 routed GMM2 完成后，让 Cube 承接 shared GateUp，与 Vector 的 combine、跨卡同步和 unpermute 重叠。两侧完成后再做激活、Down 和合并。每周期净省 = Σ min(原 GateUp 耗时, 可用尾部时长) − Σ 新同步与争用。
- **到了哪里**　待确认。设计与条件估算。优先补尾部时长证据
- **结果**　每层可用尾部 τ=20 µs、新增开销 h=5 µs 时，估省 0.60 ms/本地周期，中位 1.08%（112 个窗口中 96 个达到 1%）。

<details><summary>代码与依据</summary>

- **代码定位**　`dispatch_ffn_combine_kernel.hpp:912-977`、`shared_experts.py:471-482`、`shared_experts.py:534-564`
- **适用负载**　target SP本地M3；prefill/draft另判
- **成立条件**　h5µs时τ门槛18.41–22.26µs，中位18.89；τ尚未测得；量化、AIC仲裁/flags、Vector全核屏障与争用须成立
- **证据**　[实现切口与公式](evidence/megakernel.md) · [逐窗口模型](evidence/megakernel-window-model.json)
- **接续**　先量可用尾部与新增争用。关联 C13 的实现分支

</details>

## K04

**M10 · DSpark三层或whole-draft常驻**

- **介绍**　three independent context banks -> one five-query three-layer forward -> collapse/head/Markov。重复调用/交接/初始化；必要TP/EP、cache及GEMM保留。保留3次EP服务、每层SP/CP/TPcollective和五步ID依赖；需设备TP协议
- **到了哪里**　待确认。设计与条件估算。待协议
- **结果**　三层仅gap免费后仍需删native24.32%；whole draft需11.62%；都是同周期门槛

<details><summary>代码与依据</summary>

- **代码定位**　`dspark.py:248` `forward`、`dspark.py:403` `forward`、`llm_base_proposer.py:1013` `_propose`、`llm_base_proposer.py:1454` `_run_merged_draft`
- **适用负载**　三层一次五query；EP16/128/K3；头五步依赖
- **成立条件**　需接入设备TP/CP协议、metadata/flag epochs、资源和精度；不能把host层循环当成megakernel
- **证据**　[实现切口与公式](evidence/megakernel.md) · [逐窗口模型](evidence/megakernel-window-model.json)

</details>

## K05

**M09、M10 · aux owner到main norm与三套context KV**

- **介绍**　three local aux rows -> main15360-to5120 -> RMS -> three distinct KV projections -> cache store。宽aux交付、重复重组与全局KV投影；主norm和不同WKV保留。main norm不能穿线性；保留三个cache组的slot/代次；需要row-owner权重
- **到了哪里**　待确认。设计与条件估算。待证
- **结果**　原prefix1.050ms；到同周期1%需净删52.71%；尚无新实现预计值

<details><summary>代码与依据</summary>

- **代码定位**　`dspark.py:281` `combine_hidden_states`、`dspark.py:284` `_project_shared_kv`、`dspark.py:295` `precompute_and_store_context_kv`、`dspark_proposer.py:226` `set_inputs_first_pass`
- **适用负载**　DSpark context；实际所需行、三个physical组
- **成立条件**　完整main权重多137.63MB/rank；BF16/main norm不得穿越；旧primitive余量4.61µs，不做原样拼接
- **证据**　[实现切口与公式](evidence/megakernel.md) · [逐窗口模型](evidence/megakernel-window-model.json)

</details>

## K06

**M01 · Engram固定位置plane融合**

- **介绍**　history/hash/CPU-UVA -> WKV -> rotation/gate -> rounded residual。中间plane/局部lookup/gate交付。权重/完整tuple不同不复用；内容缓存收益与单纯fusion分开
- **到了哪里**　已停止。设计与条件估算。停止原范围
- **结果**　所列gate+lookup全免最高约0.743%同周期，局部停止

<details><summary>代码与依据</summary>

- **代码定位**　`hash_state.py:79` `forward`、`embedding.py:333` `embed_gathered`、`embedding.py:378` `lookup`、`layer.py:34` `forward`
- **适用负载**　层1/14 TP8 CPU-UVA/rotated BF16 WKV
- **成立条件**　完整内容缓存或数学/owner重写另判；需旧cacheable work消失71.1%–96.3%再扣新成本
- **证据**　[实现切口与公式](evidence/megakernel.md) · [逐窗口模型](evidence/megakernel-window-model.json)

</details>

## K07

**M00、M11 · 跨组slot与共同输入publisher**

- **介绍**　accepted old state/request map -> ID/position/length/spec/slot publication。重复索引/slot任务、提交与共同plan交付。host镜像与CP CPU plan仍有读者，已有13组linear slot已融合
- **到了哪里**　待确认。B4 批写有历史实测，更大输入发布仍在判断。复用
- **结果**　历史B4整批3.39%/2.22%；共同输入已定价设备全免61.73ms

<details><summary>代码与依据</summary>

- **代码定位**　`block_table.py:469` `compute_slot_mapping`、`block_table.py:504` `compute_slot_mapping_draft`、`model_runner_v1.py:3561` `_get_block_table_and_slot_mapping`
- **适用负载**　B4真实26-job已有历史价值；B1另判
- **成立条件**　更大输入publisher还需其它净省124.17ms/旧整批；host镜像/DP协调保留；已有13组linear slot不重复计
- **证据**　[实现切口与公式](evidence/megakernel.md) · [逐窗口模型](evidence/megakernel-window-model.json)

</details>

## K08

**M09、M10 · 五Markov与局部head融合**

- **介绍**　needed target logits -> greedy/rejection; draft base + five dependent Markov heads -> IDs。局部add/argmax/copy/调用；必要dot保留。必要dot和五步依赖不消失；末层dead行裁剪另外定价
- **到了哪里**　已停止。设计与条件估算。停止原范围
- **结果**　五步55任务即全免最高0.8504%周期；保持dot的target/draft联合旧整批179.925ms<185.908ms

<details><summary>代码与依据</summary>

- **代码定位**　`dspark.py:311` `hc_head`、`dspark.py:320` `markov_embed`、`dspark.py:323` `markov_bias`、`dspark.py:326` `compute_logits`、`llm_base_proposer.py:1400` `_sample_draft_from_logits`
- **适用负载**　五步全vocab greedy；ID依赖保留
- **成立条件**　局部整组停止；更大controller只计新增可删工作
- **证据**　[实现切口与公式](evidence/megakernel.md) · [逐窗口模型](evidence/megakernel-window-model.json)

</details>

## K09

**全模型 · 40层或跨迭代无限常驻**

- **介绍**　complete target or target/draft loop controlled by device descriptors。潜在跨阶段提交/驻留，尚无可信净收益值。现有host scheduler、CPU DP协调、output/default stream完成与cache退休必须重做协议
- **到了哪里**　待确认。概念设计，尚无可报告净值。待协议
- **结果**　当前未具备可接入实现或可报告性能值，不排在局部方案前

<details><summary>代码与依据</summary>

- **代码定位**　`model.py:1505` `forward`、`model_runner_v1.py:2229` `execute_model`、`scheduler.py:1874` `update_from_output`
- **适用负载**　全部TP8/EP16参与与CPU scheduler交互
- **成立条件**　需设备TP、host命令/epoch、output-ready与cache-retired分离；当前default-stream完成/页回收合同不能直接沿用
- **证据**　[实现切口与公式](evidence/megakernel.md) · [逐窗口模型](evidence/megakernel-window-model.json)

</details>

## K10

**M06、M08 · 末层需求驱动selected query/FFN/readout producer**

- **介绍**　layer39 KV all actual rows + selected Q/FFN/head rows -> sampled IDs。无读者的非logit行query/output、FFN/readout；不删远端服务。先合法退役raw bank；保留aux与全量SWA cache、远端真实专家
- **到了哪里**　待确认。设计与条件估算。待读者核对
- **结果**　FFN+readout有利上界22.992ms；需净兑现64.89%；不是预计值

<details><summary>代码与依据</summary>

- **代码定位**　`model.py:1505` `forward`、`model.py:1778` `compute_logits`、`fused_moe.py:485` `_forward_impl`
- **适用负载**　layer39长prefill；SWA全实际行保留
- **成立条件**　先合法退役raw bank/observer/图发布；query metadata/换M精度/EP及全SWA状态保留
- **证据**　[实现切口与公式](evidence/megakernel.md) · [逐窗口模型](evidence/megakernel-window-model.json)

</details>

## history46-F01

**M00、M11 · slot 坐标与 flat 索引双输出**

- **介绍**　block table+position → 坐标与flat slot → cache writers；接受计数/位置 → 26项slot双输出 → 目标图。同一输入除模、索引生成及一次独立提交。
- **到了哪里**　已确认可实现。服务实测·限定基线。已有实测。双输出已有实现，沿用同轮基线。
- **结果**　各自CP基线：15.034s→14.217s，整批-5.43%

<details><summary>代码与依据</summary>

- **代码定位**　`block_table.py:469` `compute_slot_mapping`、`block_table.py:504` `compute_slot_mapping_draft`、`model_runner_v1.py:3561` `_get_block_table_and_slot_mapping`
- **适用负载**　冻结C8；匹配CP条件及slot银行
- **成立条件**　两份输出值、地址、无效行和代次不变；不与F02百分比相加
- **证据**　[module-fusion-estimates](../baselines/module-fusion-estimates.json)
- **接续**　已有实现，按该基线复用。关联 M11

</details>

## history46-F02

**M00、M11 · 26 项 slot writer 批量调用**

- **介绍**　block table+position → 坐标与flat slot → cache writers；接受计数/位置 → 26项slot双输出 → 目标图。26次提交合一次，保留所有写入。
- **到了哪里**　已确认可实现。服务实测·限定基线。已有实测。26项批写已有实现。
- **结果**　13.083/12.926s→12.639s，整批-3.39%/-2.22%

<details><summary>代码与依据</summary>

- **代码定位**　`block_table.py:469` `compute_slot_mapping`、`block_table.py:504` `compute_slot_mapping_draft`、`model_runner_v1.py:3561` `_get_block_table_and_slot_mapping`
- **适用负载**　冻结C8 B4
- **成立条件**　基线已含CP+双输出+CPU计划；适用26项写入
- **证据**　[M11](../modules/M11.md) · [ep16-batched-slot-c8](../baselines/ep16-batched-slot-c8.json)

</details>

## history46-F03

**M00、M11 · 小批量 13/26 项 slot 批写**

- **介绍**　block table+position → 坐标与flat slot → cache writers；13全局writer+活跃本地writer → 后续decode。实际caller提交与可暴露等待。
- **到了哪里**　待确认。已完成初筛。不独立立项。属于F02的小批量扩形状，13项分支尚无足够净省证据。
- **结果**　26项0.961ms/步时，13项需0.880ms/步才两轮整段均≥1%

<details><summary>代码与依据</summary>

- **代码定位**　`block_table.py:469` `compute_slot_mapping`、`block_table.py:504` `compute_slot_mapping_draft`、`model_runner_v1.py:3561` `_get_block_table_and_slot_mapping`
- **适用负载**　普通B1后续对话；B2/B3待时序
- **成立条件**　单请求caller净省尚未实测；短代码独立降级，不把B4外推为B1
- **证据**　[ordinary-input-and-slot](../ordinary-input-and-slot.md) · [slot-batch-by-population](../slot-batch-by-population.md)
- **接续**　查 13 项分支的 caller 净时间。关联 C03 的扩形状

</details>

## history46-F04

**M02、M05、M07 · 两处 Post→mHC premix/RMS 共读残差**

- **介绍**　WKV/gate → BF16四路残差 → 下层mHC；上一层FFN HcPost → Y/premix → RMS → TP8 AllGather；embedding或更新残差/均值读者 → Attention HcPre。残差读取、Y写读和合法多输出交付。
- **到了哪里**　已停止。已完成初筛。暂不立项。若新增扩宽写入，残差共读接近或低于历史请求门槛；跨层服务收益不稳定。
- **结果**　同速字节模型9.12–18.23ms；宽写+25%时14.59ms；原生直连中位10.53/14.02ms

<details><summary>代码与依据</summary>

- **代码定位**　`model.py:858` `hc_pre`、`model.py:871` `hc_post`、`model.py:879` `forward`、`hc_pre_cube_compute.h:398-477`
- **适用负载**　长提示TTFT；34直接Attention入口+40FFN入口
- **成立条件**　保持原舍入、FP32系数和旁路；全量18.23情景要兑现≥81.8%；跨层服务未稳定
- **证据**　[hc-paired-boundaries](../baselines/hc-paired-boundaries.json) · [cross-layer-hc-fusion](../baselines/cross-layer-hc-fusion.json)

</details>

## history46-F05

**M03 · 四 QLI 消费层共享一次候选成员转换**

- **介绍**　第20层candidate → 24/28/32/36层共享成员表示。重复mask/成员表示生成。
- **到了哪里**　已确认可实现。服务实测·限定基线。已有实测。四层共享单改有效，已有批写基线下的增量尚不稳定。
- **结果**　单改整批-1.50%/-1.48%；slot基线上-1.27%/-1.56%，样本重叠

<details><summary>代码与依据</summary>

- **代码定位**　`indexer.py:127` `select`、`indexer.py:181` `select_projected`、`dsa_v41.py:567` `_select_sparse_indices`
- **适用负载**　长上下文C8，24/28/32/36层
- **成立条件**　保留各层query/score/TopK；后者增量未确立
- **证据**　[qli-service-c8](../baselines/qli-service-c8.json) · [qli-batched-slot-c8](../baselines/qli-batched-slot-c8.json)
- **接续**　已有实现；叠加批写的增量未确立

</details>

## history46-F06

**M03 · QLI 四消费层内截短列表归并**

- **介绍**　四消费层token归并 → 稳定index bank。归并比较及列表/分数面写读。
- **到了哪里**　已停止。已完成初筛。暂不立项。归并占旧核时的比例未证实；不能按整个QLI核时计收益。
- **结果**　七非生产层TTFT过线需约20%核时净减；修正版联合整批需消费层净减60.6%–63.5%

<details><summary>代码与依据</summary>

- **代码定位**　`indexer.py:127` `select`、`indexer.py:181` `select_projected`、`dsa_v41.py:567` `_select_sparse_indices`
- **适用负载**　长提示TTFT；修正B4图整批另计
- **成立条件**　若归并提速75%，需原消费QLI核时81%–85%属归并；profile未分出；前三层+源层免费仍不足
- **证据**　[four-mixed-fusion-edges](../four-mixed-fusion-edges.md) · [joint-attention-frontiers](../joint-attention-frontiers.md)

</details>

## history46-F07

**M03 · SMLA KV 直读与首段 PV 直写**

- **介绍**　indices/gather → SWA与compressed QK/softmax/PV；KV源页及共享TopK → 后续层；分块max/sum/PV → 最终attention结果。BF16临时面板写读、FP32 PV部分结果写读。
- **到了哪里**　待确认。已完成初筛。优先原型。沿C09，从真实分页KV写回和QK/PV重读入手，先做局部对照。
- **结果**　256位置、50%差额暴露：17.77–17.81ms(1.19%)；512位置KV单项23.72–23.77ms(1.59%)

<details><summary>代码与依据</summary>

- **代码定位**　`dsa_v41.py:613` `_forward_attention`、`sparse_flash_mla_csa_block_vector.h:606-653`、`sparse_flash_mla_csa_block_cube.h:416-436`、`sparse_flash_mla_csa_block_cube.h:704-726`
- **适用负载**　长提示38压缩层TTFT；短/空cache回退
- **成立条件**　256位置PV+KV需稀疏读速达参考80.4%–80.6%；两次QK/PV读取、sink、归约和发布不变
- **证据**　[smla-partial-handoffs](../smla-partial-handoffs.md) · [smla-panel-occupancy](../baselines/smla-panel-occupancy.json) · [sparse-selection-delivery](../baselines/sparse-selection-delivery.json)

</details>

## history46-F08

**M04 · 大行 OA→OB 同核分块流水**

- **介绍**　OA BF16块 → OB → ReduceScatter输入；a=roundBF16(xWa); y=aWb。舍入后BF16块交付与可兑现投影重叠。
- **到了哪里**　待确认。已完成初筛。备选原型。大prefill有已绑定形状和可删除的投影交接，但依赖搬运/计算重叠；M24排除。
- **结果**　长 prefill 的 TTFT 估省 15.34–17.94 ms（1.03%–1.20%），整批约 0.08%–0.10%。假设两种资源均达参考速率的 90%，短阶段可重叠 50%。

<details><summary>代码与依据</summary>

- **代码定位**　`dsa_v1.py:1732` `_wo_a_bmm`、`dsa_v1.py:1759` `_forward_o_proj`、`dsa_v41_cp.py:209` `_project_output`
- **适用负载**　大prefill TTFT；M24回原图
- **成立条件**　两次乘法和BF16中间舍入保留；资源/权重搬运与同步成本需扣；纯边界5.70–6.51ms不足
- **证据**　[oa-ob-shared-cube](../oa-ob-shared-cube.md) · [oa-ob-fusion](../baselines/oa-ob-fusion.json)
- **接续**　先查片上容量、权重搬运与重叠

</details>

## history46-F09

**M06 · 固定路由专家跨核激活交付**

- **介绍**　TopK/hidden → DispatchFFNCombine → routed out；FP16 gate/up → SwiGLU+Quant → INT8激活+scale → GMM2。GMM1 FP16写读、INT8激活的实际跨核传输。
- **到了哪里**　已停止。已完成初筛。暂不立项。多核仍需同一激活；没有可兑现的新广播或驻留成本依据。
- **结果**　单核最多3次读含前侧11.13ms(0.75%)；全部逻辑传输免费敏感性29.35ms(1.97%)

<details><summary>代码与依据</summary>

- **代码定位**　`dispatch_ffn_combine_kernel.hpp:793-823`、`dispatch_ffn_combine_kernel.hpp:912-977`
- **适用负载**　长提示TTFT；固定EP16、专家/权重分布
- **成立条件**　要过1%，前侧全省后每分派需净省约6.74次INT8单向整行传输；跨核发送与同步要付；20逻辑读≠20次HBM miss
- **证据**　[ffn-merge-premix](../ffn-merge-premix.md) · [routed-activation-readers](../baselines/routed-activation-readers.json)

</details>

## history46-R01

**M00、M01 · 词嵌入/Engram 输入交付**

- **介绍**　token ID → 查表/掩码/SUM → 持久输入 bank；lookup head → SP行所有者/WKV。重复查表、输入重排、SP交付。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。图外bank有其它读者则扣回；不同层权重不能缓存成同一WKV
- **结果**　embedding具名设备+launch整批0.482%，普通≤0.667%；协议常见组分别≤0.083%/0.085%

<details><summary>代码与依据</summary>

- **代码定位**　`model.py:1392` `prepare_engram_inputs`、`model.py:1505` `forward`、`vl_model.py:190` `embed_input_ids`
- **适用负载**　冻结C8及六普通请求
- **成立条件**　图外bank有其它读者则扣回；不同层权重不能缓存成同一WKV
- **证据**　[ordinary-input-and-slot](../ordinary-input-and-slot.md)

</details>

## history46-R02

**M00、M11 · 元数据 tiling 多输出**

- **介绍**　公共tiling → 多份固定地址metadata → 图/eager读者。重复构造与copy/launch。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。原就绪信号早于读者，不计整段异步等待
- **结果**　全部具名API免费：整批0.141%，普通≤0.161%

<details><summary>代码与依据</summary>

- **代码定位**　`dsa_v41.py:879` `_publish_task`、`dsa_v41.py:929` `build`、`device_metadata.py:74` `submit`、`device_metadata.py:130` `wait`
- **适用负载**　当前图与eager
- **成立条件**　原就绪信号早于读者，不计整段异步等待
- **证据**　[profile-boundary-audit](../profile-boundary-audit.md)

</details>

## history46-R03

**M01 · Engram 局部 gate/lookup 与尾部**

- **介绍**　历史token/slot → hash → lookup → WKV；FP32(q_weight*k_weight)、块旋转 → 加权dot；WKV/gate → BF16四路残差 → 下层mHC。统计共读、静态权重乘积、尾部回写。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。这是两种不同范围/分母；不能相加；联合尾部已在F04预算
- **结果**　gate+lookup全免≤0.743%本地周期；尾部TTFT1.30–2.05ms

<details><summary>代码与依据</summary>

- **代码定位**　`hash_state.py:79` `forward`、`embedding.py:333` `embed_gathered`、`embedding.py:378` `lookup`、`layer.py:34` `forward`
- **适用负载**　两个实际Engram入口
- **成立条件**　这是两种不同范围/分母；不能相加；联合尾部已在F04预算
- **证据**　[initial-hc-engram](../initial-hc-engram.md)

</details>

## history46-R04

**M00、M02 · 首层虚拟四路与草稿 padding**

- **介绍**　embedding或更新残差/均值读者 → Attention HcPre；draft embedding → mHC/attention/MoE三层 → collapse/norm。repeat、初始化、重复原行读取。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。HcPre/HcPost都要适配原BF16行，不省后续残差输出
- **结果**　C8整批15.88ms(0.085%)，普通≤0.081%

<details><summary>代码与依据</summary>

- **代码定位**　`model.py:879` `forward`、`model.py:1505` `forward`、`dspark_proposer.py:226` `set_inputs_first_pass`
- **适用负载**　目标/DSpark首层
- **成立条件**　HcPre/HcPost都要适配原BF16行，不省后续残差输出
- **证据**　[initial-hc-engram](../initial-hc-engram.md)

</details>

## history46-R05

**M02、M03 · 独立提前本地 Q 与空侧提早**

- **介绍**　Y/RMS → 本地Q或全局KV → 汇合。仅隐藏Q投影。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。不把排程移动当作删掉计算
- **结果**　TTFT最有利9.011ms<14.919ms；空侧对EP16到达0

<details><summary>代码与依据</summary>

- **代码定位**　`dsa_v41.py:320` `_project_q`、`dsa_v41.py:333` `_project_q_kv`、`dsa_v41_cp.py:189` `_prepare_inputs_and_caches`
- **适用负载**　当前34直连和空侧
- **成立条件**　不把排程移动当作删掉计算
- **证据**　[attention-service-budget](../baselines/attention-service-budget.json)

</details>

## history46-R06

**M03 · Attention 输入多投影/旋转/量化/cache 融合组**

- **介绍**　QA/QR、Qb、index Q/W、KV norm/RoPE → cache。重复输入、qr量化、旋转发布中间面。
- **到了哪里**　已停止。已完成初筛。不立项。其中Qb/index-Q共享动态量化：八站点第二次量化合计22.58µs/图，未扣共享开销；不单开项目。
- **结果**　直接搬运整批7.33–9.27ms；半向量工作+间隙全省0.68%–0.70%

<details><summary>代码与依据</summary>

- **代码定位**　`dsa_v41.py:320` `_project_q`、`dsa_v41.py:327` `_project_kv`、`dsa_v41.py:440` `multistream_preprocess`、`indexer.py:167` `project_query`、`indexer.py:176` `quantize_query`
- **适用负载**　实际主辅流、KV-only与普通形状
- **成立条件**　相同提前排程；不同quant/scale不合并；两流等候不可无条件删
- **证据**　[attention-input-fusions](../baselines/attention-input-fusions.json)

</details>

## history46-R07

**M03 · C2 与索引输出接口联合**

- **介绍**　FP32 WKV/Wgate → pool/ring/RMS → indexK与longKV。pool/ring及索引准备中间面。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。持久索引有后读者；第二流水组可并行须另证
- **结果**　TTFT0.353ms、整批0.461ms；索引准备全免TTFT6.56–6.70ms

<details><summary>代码与依据</summary>

- **代码定位**　`dsa_v41.py:497` `_write_compressed_source`、`compressor.py:49` `prepare_ring_compressor`、`compressor.py:55` `pool_projected`、`indexer.py:106` `update_keys`
- **适用负载**　C1/C2实际源层
- **成立条件**　持久索引有后读者；第二流水组可并行须另证
- **证据**　[c2-selection-pipeline](../c2-selection-pipeline.md)

</details>

## history46-R08

**M04 · Attention 直接写发送布局/接收逆旋转**

- **介绍**　attention output → inverse RoPE/布局 → TP8 AllToAll。pad/transpose/copy或旋转接口。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。两者替代；448/64与正零及接收路径保留
- **结果**　发送整批0.49%–0.51%；接收端新增免费也仅TTFT4.84–5.33ms

<details><summary>代码与依据</summary>

- **代码定位**　`dsa_v41_cp.py:209` `_project_output`、`dsa_cp.py:77` `restore_tp_heads`、`dsa_v1.py:1732` `_wo_a_bmm`、`dsa_v1.py:1759` `_forward_o_proj`
- **适用负载**　有/无AllToAll、空发送
- **成立条件**　两者替代；448/64与正零及接收路径保留
- **证据**　[attention-send](../baselines/attention-send.json)

</details>

## history46-R09

**M02、M07 · 宽残差工作区交付**

- **介绍**　RS结果+原残差 → HcPost → 下个HcPre；BF16残差 → FP32/HF32投影工作区 → Cube。少一份BF16读，或只存FP32。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。后读Y/FFNPost各多读宽面抵消；不支持免费UB→L1假设
- **结果**　双输出TTFT0.36–3.93ms；只存宽平面净省字节0

<details><summary>代码与依据</summary>

- **代码定位**　`model.py:858` `hc_pre`、`model.py:871` `hc_post`、`model.py:879` `forward`、`hc_pre_cube_compute.h:398-477`
- **适用负载**　40FFN/相邻Post
- **成立条件**　后读Y/FFNPost各多读宽面抵消；不支持免费UB→L1假设
- **证据**　[hc-workspace-delivery](../baselines/hc-workspace-delivery.json)

</details>

## history46-R10

**M05、M06 · Norm 同时产三种输入**

- **介绍**　归一化 → router宽输入/共享专家INT8+scale。absmax与匹配量化输入写读。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。router宽输入是规定舍入；六份dispatch行不是本地稠密输入
- **结果**　TTFT慢2.01至快3.09ms；给路由首段50%附加净省仍8.40ms

<details><summary>代码与依据</summary>

- **代码定位**　`model.py:844` `rms_norm_cast`、`fused_moe.py:389` `_prepare_router_input`
- **适用负载**　router FP32/共享INT8
- **成立条件**　router宽输入是规定舍入；六份dispatch行不是本地稠密输入
- **证据**　[activation-quant](../baselines/activation-quant.json)

</details>

## history46-R11

**M06 · Router/TopK/ID/dispatch 入口**

- **介绍**　归一化宽输入 → gate → TopK/ID/权重 → dispatch。分数和ID中间面。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。MM与选中权重保留；归档未传固定表；主机晚提交另记
- **结果**　TTFT1.65–1.85ms、整批0.27%–0.36%

<details><summary>代码与依据</summary>

- **代码定位**　`model.py:844` `rms_norm_cast`、`fused_moe.py:401` `_apply_router_linear`、`fused_moe.py:412` `_compute_router_logits`、`fused_topk_router.py:140` `_compute_routing`
- **适用负载**　动态routing及实际hash调用
- **成立条件**　MM与选中权重保留；归档未传固定表；主机晚提交另记
- **证据**　[router-score-delivery](../baselines/router-score-delivery.json)

</details>

## history46-R12

**M06 · 路由专家单核激活承接**

- **介绍**　TopK/hidden → DispatchFFNCombine → routed out；FP16 gate/up → SwiGLU+Quant → INT8激活+scale → GMM2。FP16写读+最多3次INT8读。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。其余核仍读GM，不能删整行写回；跨核门槛另归F09
- **结果**　TTFT11.13ms(0.75%)，普通整段≤0.148%

<details><summary>代码与依据</summary>

- **代码定位**　`dispatch_ffn_combine_kernel.hpp:793-823`、`dispatch_ffn_combine_kernel.hpp:912-977`
- **适用负载**　当前GMM2列块分配
- **成立条件**　其余核仍读GM，不能删整行写回；跨核门槛另归F09
- **证据**　[ffn-merge-premix](../ffn-merge-premix.md)

</details>

## history46-R13

**M06 · 共享专家两处交接联合**

- **介绍**　Gate/Up INT32 → DequantSwigluQuant → INT8+scale → Down BF16。INT32 gate/up及INT8+scale写读。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。保留激活顺序和点积；W8A8已有DequantSwigluQuant，无独立clamp/cat可再省
- **结果**　有利TTFT8.83ms(0.59%)、整批62.10ms(0.33%)

<details><summary>代码与依据</summary>

- **代码定位**　`shared_experts.py:185` `part1`、`shared_experts.py:192` `apply_activation`、`shared_experts.py:195` `part2`、`shared_experts.py:670` `forward`
- **适用负载**　[M,4608]→[M,2304]→[M,5120]
- **成立条件**　保留激活顺序和点积；W8A8已有DequantSwigluQuant，无独立clamp/cat可再省
- **证据**　[shared-mlp-fusion](../baselines/shared-mlp-fusion.json)

</details>

## history46-R14

**M06、M07 · 共享 Down+加和+HcPost+直接 premix**

- **介绍**　ForeachAddList → BF16 X → HcPost。三处合法中间面。
- **到了哪里**　已停止。服务反例·限定变体。不立项。按原范围停止。先BF16加法舍入；premix读与F04重叠，不相加
- **结果**　三项有利TTFT7.30ms(0.49%)；M3服务慢0.31%/0.19%

<details><summary>代码与依据</summary>

- **代码定位**　`shared_experts.py:195` `part2`、`fused_moe.py:485` `_forward_impl`、`model.py:871` `hc_post`
- **适用负载**　实际q=3/416/419/816/817/818
- **成立条件**　先BF16加法舍入；premix读与F04重叠，不相加
- **证据**　[ffn-merge-premix](../ffn-merge-premix.md)

</details>

## history46-R15

**M08、M10 · 三检查点打包归集与草稿取行拼接**

- **介绍**　四路残差 → 三处mean/TP gather、末层raw buffer；aux Index/Concat → main_proj/TP gather → RMS。两次TP8启动及合法前缀中间面。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。所有发送字节与原均值源时点保留
- **结果**　整批8.02–31.27ms；极有利全额最多0.39%

<details><summary>代码与依据</summary>

- **代码定位**　`model.py:1505` `forward`、`model.py:1798` `get_mtp_target_hidden_states`、`dspark.py:281` `combine_hidden_states`
- **适用负载**　三处aux、同batch代次
- **成立条件**　所有发送字节与原均值源时点保留
- **证据**　[dspark-checkpoint-packet](../baselines/dspark-checkpoint-packet.json)

</details>

## history46-R16

**M08 · 末层 Post 到词表 MM 整段**

- **介绍**　末层Post → collapse(Cast/Mul/Sum/Cast) → RMS → 词表选行。collapse/RMS/选行中间面。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。同段子融合一起排除；仅裁行不算融合，24槽最长分片不变不省通信
- **结果**　整段全部免费TTFT14.41ms(0.966%)、整批78.69ms(0.423%)

<details><summary>代码与依据</summary>

- **代码定位**　`model.py:855` `hc_collapse`、`model.py:1778` `compute_logits`、`logits_processor.py:136` `_apply_head`
- **适用负载**　raw/aux/完整行分支
- **成立条件**　同段子融合一起排除；仅裁行不算融合，24槽最长分片不变不省通信
- **证据**　[terminal-head-delivery](../baselines/terminal-head-delivery.json)

</details>

## history46-R17

**M08、M09 · 目标词表选词与分片摘要**

- **介绍**　TP8 vocab projection → gather/布局/ArgMax。布局扫描、ArgMax，或稠密gather。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。完整raw/logprob读者不能删；tie/NaN/私有处理保持
- **结果**　一次摘要交换整批27.48–47.78ms；完整logits额外ID半扫描4.23ms

<details><summary>代码与依据</summary>

- **代码定位**　`logits_processor.py:122` `_gather_logits`、`logits_processor.py:201` `get_top_tokens`、`sampler.py:83` `greedy_sample`
- **适用负载**　仅greedy ID与完整分数分别处理
- **成立条件**　完整raw/logprob读者不能删；tie/NaN/私有处理保持
- **证据**　[vocab-selection-fusion](../baselines/vocab-selection-fusion.json)

</details>

## history46-R18

**M09、M11 · 目标验证直接生成 seed/count**

- **介绍**　target/bonus ID → 连续接受比较/seed/count。重复检查及count D2H等待。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。seed原位置、完整输出ID及host镜像保留
- **结果**　即使计数回传免费整批10.14ms；普通≤0.065%

<details><summary>代码与依据</summary>

- **代码定位**　`rejection_sampler.py:146` `forward`、`llm_base_proposer.py:2192` `prepare_next_token_ids_padded`、`model_runner_v1.py:1876` `_copy_valid_sampled_token_count`
- **适用负载**　greedy接受与discard
- **成立条件**　seed原位置、完整输出ID及host镜像保留
- **证据**　[draft-submit-waits](../baselines/draft-submit-waits.json)

</details>

## history46-R19

**M09、M10、M11 · 验证+草稿ID+下一轮状态联合**

- **介绍**　target/bonus ID → 连续接受比较/seed/count；已验证ID/计数 → CPU发布 → 草稿/下轮输入；验证计数 → effective length/slot/input pack → 下一迭代。窄ID、有效性、位置及具名等待。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。时间轴去重；不同主机线程空档和换批不能照抄
- **结果**　设备+等待模型63.86ms(0.344%)；满批主机全给足约0.992%

<details><summary>代码与依据</summary>

- **代码定位**　`rejection_sampler.py:146` `forward`、`llm_base_proposer.py:2192` `prepare_next_token_ids_padded`、`model_runner_v1.py:1876` `_copy_valid_sampled_token_count`、`model_runner_v1.py:2163` `_copy_draft_token_ids_to_cpu`
- **适用负载**　当前稳定batch；换批另U10
- **成立条件**　时间轴去重；不同主机线程空档和换批不能照抄
- **证据**　[draft-state-handoffs](../draft-state-handoffs.md)

</details>

## history46-R20

**M09 · top-k/top-p 双阈值融合**

- **介绍**　温度/惩罚 → 概率排序/双阈值 → 随机抽样/拒绝恢复。相同滤前概率、合并mask写回。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。top-p依赖原概率累积；RNG及raw分数不能用argmax等价替代
- **结果**　B4每次18.72us；随机外推整批0.046%/0.134%，普通<0.06%

<details><summary>代码与依据</summary>

- **代码定位**　`rejection_sampler.py:321` `apply_sampling_constraints`、`rejection_sampler.py:378` `rejection_sample`、`sampler.py:92` `forward_native`
- **适用负载**　默认Ascend随机分支，冻结greedy无命中
- **成立条件**　top-p依赖原概率累积；RNG及raw分数不能用argmax等价替代
- **证据**　[random-sampling-fusion](../baselines/random-sampling-fusion.json)

</details>

## history46-R21

**M09 · 有限 token logprobs 与私有mask**

- **介绍**　bonus/target私有分数 → logsoftmax/TopK/rank。扫描归一化、TopK/rank及拼行。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。prompt长行另U09；K=-1或raw读者继续完整发布
- **结果**　整批7.99–15.98ms；最有利附带mask24.53ms(0.132%)

<details><summary>代码与依据</summary>

- **代码定位**　`logits_processor.py:98` `forward`、`model_runner_v1.py:2973` `_sample`
- **适用负载**　冻结几何假设全启用
- **成立条件**　prompt长行另U09；K=-1或raw读者继续完整发布
- **证据**　[score-readers](../score-readers.md)

</details>

## history46-R22

**M00、M10 · 输入bank与DSpark主投影转置**

- **介绍**　aux Index/Concat → main_proj/TP gather → RMS。一处输入copy、rank-major转置。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。RMS仍归约全5120并读gamma；图copy命中未知用有利代理
- **结果**　合计有利TTFT0.623ms、整批8.05ms(0.043%)

<details><summary>代码与依据</summary>

- **代码定位**　`model_runner_v1.py:1281` `_prepare_inputs`、`dspark.py:281` `combine_hidden_states`
- **适用负载**　冻结7 eager+446图TP8调用
- **成立条件**　RMS仍归约全5120并读gamma；图copy命中未知用有利代理
- **证据**　[optional-paths-and-raw-head](../optional-paths-and-raw-head.md)

</details>

## history46-R23

**M10 · 三层context RMS/RoPE/cache与提交**

- **介绍**　每层WKV → RMS → RoPE → scatter cache；draft embedding → mHC/attention/MoE三层 → collapse/norm。拼接、位置查表、地址与中间写读。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。三层WKV及各自norm不共用；计数等待与submit依赖保留
- **结果**　半设备预算27.94–30.82ms；含主机整批0.21%–0.24%

<details><summary>代码与依据</summary>

- **代码定位**　`dspark.py:281` `combine_hidden_states`、`dspark.py:284` `_project_shared_kv`、`dspark.py:295` `precompute_and_store_context_kv`、`dspark_proposer.py:226` `set_inputs_first_pass`
- **适用负载**　453次TP8调用
- **成立条件**　三层WKV及各自norm不共用；计数等待与submit依赖保留
- **证据**　[draft-context-fusion](../baselines/draft-context-fusion.json)

</details>

## history46-R24

**M10 · 五步 Markov 校正到 ID 及W1/W2复用**

- **介绍**　W1查表 → W2 bias → Add/round → ArgMax/ID → 下一步。Add/扫描/ID写回；命中时替代查表投影。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。五步串行自回归、BF16舍入和原词表映射保留
- **结果**　具名任务含末轮全免整批67.65ms(0.364%)；普通≤0.170%；W1/W2免费0.213%

<details><summary>代码与依据</summary>

- **代码定位**　`dspark.py:311` `hc_head`、`dspark.py:320` `markov_embed`、`dspark.py:323` `markov_bias`、`dspark.py:326` `compute_logits`、`llm_base_proposer.py:1400` `_sample_draft_from_logits`
- **适用负载**　五个前驱ID序列
- **成立条件**　五步串行自回归、BF16舍入和原词表映射保留
- **证据**　[draft-state-handoffs](../draft-state-handoffs.md)

</details>

## history46-R25

**M10 · DSpark 可选共享 TopK hook**

- **介绍**　llm_base_proposer hasattr → compact_topk_indices。本路径无对应调用。
- **到了哪里**　已停止。已完成初筛。不立项。按原范围停止。hasattr为假；其它MTP模型的功能不能算到DSpark
- **结果**　发生次数0，可省工作0

<details><summary>代码与依据</summary>

- **代码定位**　`dspark.py:414` `compute_draft_logits`、`llm_base_proposer.py:1400` `_sample_draft_from_logits`
- **适用负载**　固定DeepseekV41DSparkModel
- **成立条件**　hasattr为假；其它MTP模型的功能不能算到DSpark
- **证据**　[optional-paths-and-raw-head](../optional-paths-and-raw-head.md)

</details>

## history46-U01

**M00 · 视觉条件入口**

- **介绍**　视觉编码 → 空间重组/投影 → 语言输入。patch/Transformer/空间投影与语言输入布局。
- **到了哪里**　已停止。已完成初筛。范围外。冻结文本请求无视觉输入。
- **结果**　当前收益0；图像负载未知

<details><summary>代码与依据</summary>

- **代码定位**　`vl_model.py:190` `embed_input_ids`、`vl_model.py:261` `forward`
- **适用负载**　当前14文本请求无命中
- **成立条件**　仅图像请求命中且N·净省关键时间≥0.01·T时重估
- **证据**　[M00](../modules/M00.md)

</details>

## history46-U02

**M01 · Engram 自然跨层/跨步缓存**

- **介绍**　lookup head → SP行所有者/WKV。相同lookup/常量投影。
- **到了哪里**　已停止。已完成初筛。暂不立项。按C06，完整内容命中且收益覆盖协调及容量才重开。
- **结果**　原本地周期1%需消除约75.4%可缓存部分（常量免费假设）

<details><summary>代码与依据</summary>

- **代码定位**　`hash_state.py:79` `forward`、`embedding.py:333` `embed_gathered`、`embedding.py:378` `lookup`、`layer.py:34` `forward`
- **适用负载**　相同token、位置、历史、层权重版本
- **成立条件**　自然命中率×每次实际可省量扣缓存维护、通信后过线；不同层WKV不直接共用
- **证据**　[M01](../modules/M01.md)

</details>

## history46-U03

**M02、M05 · Gram 对角化、范数归约与精度约束**

- **介绍**　XW与diag(XXᵀ)、Sinkhorn → pre/post/comb/Y；分块max/sum/PV → 最终attention结果；BF16残差 → FP32/HF32投影工作区 → Cube。不需要的Gram非对角、行归约合并。
- **到了哪里**　已停止。已完成初筛。暂不立项。有位差先例；尚缺满足精度的具体替代实现。
- **结果**　M3例Gram 5242880 MAC vs 61440平方；mix投影7864320 MAC仍保留；无通过精度的性能值

<details><summary>代码与依据</summary>

- **代码定位**　`model.py:858` `hc_pre`、`model.py:871` `hc_post`、`model.py:879` `forward`、`hc_pre_cube_compute.h:398-477`
- **适用负载**　所有80处mHC，历史严格四输出合同
- **成立条件**　旧改版FP32 post/comb有位差，停该实现；保留HF32/K归约或先有明确模型误差预算，不能放宽后宣称等价
- **证据**　[M02](../modules/M02.md)

</details>

## history46-U04

**M02、M05 · Sinkhorn 真正定点提前停止**

- **介绍**　XW与diag(XXᵀ)、Sinkhorn → pre/post/comb/Y；归一化统计及Sinkhorn → 系数/Y。已逐位不变后的剩余迭代。
- **到了哪里**　已停止。已完成初筛。暂不立项。提前退出需满足真实定点及判定成本，尚无过线的实现证据。
- **结果**　两处需兑现中位差额77.6%；单侧最有利TTFT≤9.96ms

<details><summary>代码与依据</summary>

- **代码定位**　`hc_pre_m_k_split_core.h:330-435`、`model.py:858` `hc_pre`
- **适用负载**　Attention+FFN两处合计
- **成立条件**　完整FP32 tile在一整轮后逐位稳定才可精确停；稳定深度与检测/同步成本未知
- **证据**　[mhc-iteration-cost](../baselines/mhc-iteration-cost.json)

</details>

## history46-U05

**M03 · 同代 KV 面板驻留**

- **介绍**　KV源页及共享TopK → 后续层。重复gather面板，不减少模型KV逻辑状态。
- **到了哪里**　已停止。已完成初筛。不独立立项。C09的一种替代实现，共用KV交付预算。
- **结果**　连续copy参考TTFT4.02–5.77ms；有效稀疏速率<约465GB/s可能改变优先级

<details><summary>代码与依据</summary>

- **代码定位**　`dsa_v41.py:613` `_forward_attention`、`sparse_flash_mla_csa_block_vector.h:606-653`、`sparse_flash_mla_csa_block_cube.h:416-436`、`sparse_flash_mla_csa_block_cube.h:704-726`
- **适用负载**　同源后续层；足够容量与自然命中
- **成立条件**　192MiB L2报告不证明可用容量/共享；命中×省流量扣占用竞争需过线
- **证据**　[sparse-kv-retention](../baselines/sparse-kv-retention.json)

</details>

## history46-U06

**M06 · 其它合法 MoE 后端/selector**

- **介绍**　prepare → dispatch → apply_moe_mlp → combine/finalize。仍真实存在的prepare/dispatch/MLP/combine中间面。
- **到了哪里**　已停止。已完成初筛。范围外。备选后端没有冻结负载命中证据。
- **结果**　该固定profile无匹配后端预算，Nbranch及净开销未知

<details><summary>代码与依据</summary>

- **代码定位**　`moe_comm_method.py:563` `fused_experts`、`routed_experts.py:693` `forward_impl`
- **适用负载**　AllGather/AllToAll/分离MC2/CANN MegaMoe条件
- **成立条件**　仅该selector实际可达且Nbranch·Δcritical>0.01T才深查；保持mask/capacity/HCCL状态
- **证据**　[M06](../modules/M06.md)

</details>

## history46-U07

**M06 · 固定布局的专家返回有序直写**

- **介绍**　GMM2 → return/reducer → 本地routed output。return暂存或重复归约/同步。
- **到了哪里**　已停止。已完成初筛。暂不立项。必要的远端服务与有序归并仍保留，可删的净关键时间未证实。
- **结果**　不把专家等待或冷热放置当成可删；新增净省量未证实

<details><summary>代码与依据</summary>

- **代码定位**　`dispatch_ffn_combine_kernel.hpp:912-977`、`prepare_finalize.py:316` `finalize`
- **适用负载**　保持专家选择、权重及EP16所有权
- **成立条件**　保持原加权累加顺序；直写多读者与每轮返回代次明确，净减≥1%才重开
- **证据**　[M06](../modules/M06.md)

</details>

## history46-U08

**全模型 · 全层/跨层 megakernel 额外空间**

- **介绍**　Post → mHC → attention → FFN → 下层；draft embedding → mHC/attention/MoE三层 → collapse/norm。仅能指认的新读写/launch/同步。
- **到了哪里**　已停止。已完成初筛。不独立立项。需指明已有局部候选以外可新增消除的工作。
- **结果**　局部已归F04/F07/F08/F09；没有额外通用加速百分比

<details><summary>代码与依据</summary>

- **代码定位**　`model.py:1505` `forward`、`model_runner_v1.py:2229` `execute_model`、`scheduler.py:1874` `update_from_output`
- **适用负载**　全40目标+3草稿层
- **成立条件**　非线性、cache与TP/EP barrier不能凭常驻消失；若声称额外空间须量出Δsubmit/Δtraffic并扣资源退化
- **证据**　[M07](../modules/M07.md)

</details>

## history46-U09

**M08、M09 · 有限 prompt 评分共同扫描/TP摘要**

- **介绍**　vocab FP32扫描 → LSE/TopK/指定词rank；bonus/target私有分数 → logsoftmax/TopK/rank。LSE、TopK、指定词rank扫描及稠密分数交付。
- **到了哪里**　已停止。已完成初筛。范围外。原负载未命中有限提示评分。
- **结果**　本地0.79–2.20ms；含跨卡2.42–4.82ms；固定请求无命中

<details><summary>代码与依据</summary>

- **代码定位**　`logits_processor.py:122` `_gather_logits`、`logits_processor.py:180` `_get_logits`、`model.py:1778` `compute_logits`
- **适用负载**　prompt logprobs开启；例1464行K5
- **成立条件**　三次小通信可能抵消；Nscore与K满足Nscore·每行净省−通信>0.01T；全量K保持全输出
- **证据**　[target-readout-and-prompt-scores](../target-readout-and-prompt-scores.md)

</details>

## history46-U10

**M09、M10、M11 · 随机/概率草稿/换批/动态长度的状态交付**

- **介绍**　温度/惩罚 → 概率排序/双阈值 → 随机抽样/拒绝恢复；已验证ID/计数 → CPU发布 → 草稿/下轮输入；draft probs或ID → target映射/动态长度 → verifier。该分支重复扫描、状态copy与实际submit等待。
- **到了哪里**　已停止。已完成初筛。暂不立项。不同选择和换批条件混合，缺匹配负载的独立收益。
- **结果**　当前无该分支端点估值；不外推greedy/caller大间隙

<details><summary>代码与依据</summary>

- **代码定位**　`rejection_sampler.py:146` `forward`、`rejection_sampler.py:378` `rejection_sample`、`llm_base_proposer.py:2389` `prepare_inputs_padded`
- **适用负载**　未测合法配置；保持RNG、vocab映射与代次
- **成立条件**　Nbranch·每次净暴露量超过目标预算才取证；概率映射不兼容时源码回退greedy；分数与confidence读者保留
- **证据**　[draft-state-handoffs](../draft-state-handoffs.md)

</details>

## history46-H01

**M00、M11 · CP 局部长度与 CPU 计划公共项**

- **介绍**　批计划/局部长度 → 26组metadata builder。重复长度、后缀处理与CPU计划。
- **到了哪里**　已停止。已完成初筛。不独立立项。已有调度与CPU计划基线。
- **结果**　CP单改整批6.10%/4.89%；CPU复用10.36%/9.56%

<details><summary>代码与依据</summary>

- **代码定位**　`dsa_v41_cp.py:87` `_build_global_metadata`、`dsa_v41_cp.py:112` `build`、`model_runner_v1.py:3449` `_build_attention_metadata`
- **适用负载**　原历史CP及CPU计划各自基线
- **成立条件**　属于公共工作/主机共享，不能全写成新增设备融合收益
- **证据**　[module-fusion-estimates](../baselines/module-fusion-estimates.json)

</details>

## history46-H02

**M06 · 前缀换 DP、冷热专家与权重放置**

- **介绍**　现有KV前缀 → 新请求任务划分。工作量或负载均衡改变。
- **到了哪里**　已停止。已完成初筛。范围外。部署放置调整。
- **结果**　C8额外命中上界0 token；专家布局收益不纳入融合

<details><summary>代码与依据</summary>

- **代码定位**　`scheduler.py:561` `schedule`、`routed_experts.py:693` `forward_impl`
- **适用负载**　固定C8当前前缀与专家分布
- **成立条件**　固定专家与权重放置，只回收能指认的物化或重复计算
- **证据**　[c8-prefix-ceiling](../baselines/c8-prefix-ceiling.json) · [fusion-attribution](../fusion-attribution.md)

</details>

## 冻结源码索引

在相同快照按以下路径定位，可用 SHA-256 核对版本。表内未公开源码的文件，使用工作台环境中已有的对应快照。

| 文件路径 | SHA-256 |
|---|---|
| `vllm_ascend/worker/block_table.py` | `215305c699a9df9c118f47c6f7b71d8457ef54e98b5b8571536fd373de7b21ed` |
| `vllm_ascend/models/deepseek_v41/compressor.py` | `31516e5ba234e34fa01e73e864fbc88bd6473d4a5274a1d43da8a79bab3cf494` |
| `vllm_ascend/worker/device_metadata.py` | `efe13f0e801b70d7fba7498ac44062a8eccfd78e5b6303cf658d33dd22217580` |
| `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/dispatch_ffn_combine/dispatch_ffn_combine.h` | `d2b5712ad97f7505233f340e18660627424e5e1fe4c33be68f8e34b51a479ab1` |
| `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/dispatch_ffn_combine/dispatch_ffn_combine_kernel.hpp` | `7dcef4c08c4c47e38bdd01aef0dee997798658ee4d7b871c8e6c70f7ffe596ba` |
| `vllm_ascend/attention/context_parallel/dsa_cp.py` | `7e8dd93eeda988866138342523ecfa7cc9c42ce67669d4b1507ee8c5f3452305` |
| `vllm_ascend/attention/dsa_v1.py` | `698765a3662332ad108fd9ef128eb87fd313544cd887f5154f924a5915ee123a` |
| `vllm_ascend/attention/dsa_v41.py` | `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd` |
| `vllm_ascend/attention/context_parallel/dsa_v41_cp.py` | `039ec9d18e0a8a4aa39f9d5b4bea15204cea3f397a3fce97a5097b266bc9822e` |
| `vllm_ascend/models/deepseek_v41/dspark.py` | `fcee256905928917154d8dac94d6611bb83d02da441e8aa154a525ef51e6a601` |
| `vllm_ascend/spec_decode/dspark_proposer.py` | `70d216de21875e679d510153ebdb0f0ccf130875a0e644f0673d60b67b6b633c` |
| `vllm_ascend/models/deepseek_v41/engram/embedding.py` | `a7741ed16522be4fe607e062e1c988c4e2435af7d258fe7db99f672d4508c4c9` |
| `vllm_ascend/ops/fused_moe/fused_moe.py` | `2f42ce56a7636bf618204dbc2c1aeb2a2c95fed69fe9e5bcfac995354a00beb4` |
| `vllm_ascend/ops/fused_moe/router/fused_topk_router.py` | `06c47fd91e8c36f6f5465cea0b32a184280f842833c2bff27a18fc6d17fcb838` |
| `vllm/v1/worker/gpu_model_runner.py` | `87c29d08c0bbf66993d8b984811e7325e35e6436242a773feb55ec88f16b2c51` |
| `vllm_ascend/models/deepseek_v41/engram/hash_state.py` | `62dd8271db66f4a735de7e9f764b3a1a4b7e66b56c352352f27093f3ef40eb38` |
| `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/hc_pre/hc_pre_cube_compute.h` | `9056b512f88569025bd3cec598a66d5461b5fd4c14bf6a7eb6c963ff6537ed69` |
| `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/hc_pre/hc_pre_m_k_split_core.h` | `4d697b15e541216722369284cbd8c9bd5f4154cd0d2251cdb06a6cd46f42fc14` |
| `vllm_ascend/models/deepseek_v41/indexer.py` | `d5d6a438bbced4e63d729dac94c08869df30dab4a71ceacecdea211386b3db33` |
| `vllm_ascend/models/deepseek_v41/engram/layer.py` | `dc9457c88aa3c90133115bc608447e037b0693f8db71bb8da37f1ead44021f80` |
| `vllm_ascend/spec_decode/llm_base_proposer.py` | `458fb50e6839f3a353ddc7181ace4fdae02e35093d8ccd1c4327dc24ae5fff31` |
| `vllm/model_executor/layers/logits_processor.py` | `6b0603d67b0c756253c2fdc882a3896d2e873a16e9aa2ef877aabca8d36bdb5f` |
| `vllm_ascend/models/deepseek_v41/model.py` | `101f96b38e9fc47e7be147ddfe56976aee896867c73a33a7d95c3e3c833a6e91` |
| `vllm_ascend/worker/model_runner_v1.py` | `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c` |
| `vllm_ascend/ops/fused_moe/moe_comm_method.py` | `7040f691355bd9acb3c668e698de3f7d3854edbc22e3f979f9b03d99800364d7` |
| `vllm_ascend/ops/fused_moe/prepare_finalize.py` | `dff4fac292200fc06f60aef48505f1ee38314ea5adb53560b325901db308eb24` |
| `vllm_ascend/sample/rejection_sampler.py` | `bb1d39de851540c16c2dd2b16df60c60a1fc4957cedb88bdce9417efc0d55eda` |
| `vllm_ascend/ops/fused_moe/routed_experts.py` | `f9db74836ae210a541432b923118696c7a15be9a10d5776834e1d3b64ce11d77` |
| `vllm_ascend/sample/sampler.py` | `867f1770614bb594baf5ecfb858aa4a20dc413cbfbe4f94e06b3786900b581ee` |
| `vllm/v1/core/sched/scheduler.py` | `c1db45f3bbad3a875dd8638331b8870c4380ac1ab4734265ba5c863833a82884` |
| `vllm_ascend/ops/fused_moe/shared_experts.py` | `471db26e9e9e4ba70c2ccd43502ad39bff6699bbd4ca191319a9b66fc20458f9` |
| `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/sparse_flash_mla/arch22/sparse_flash_mla_csa_block_cube.h` | `55210741e16d8cc455215849c73e2c06db10f23efeb73ca87b4a16847339ccc3` |
| `vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/custom_transformer_impl/ascendc/sparse_flash_mla/arch22/sparse_flash_mla_csa_block_vector.h` | `685190e19519b9a44da7fa95832c06e4b41849577488f4efd0d2df80e1158757` |
| `vllm_ascend/models/deepseek_v41/vl_model.py` | `bb98a2e2a0e24c18aafe17229eb593937aaed2e37c1fd7cc2003c45ce872d282` |
