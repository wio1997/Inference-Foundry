# DSpark 首步 metadata 与注意力消费者

## 介绍

沿 runner 交付的 common metadata、物理 block/slot 与校正后的请求长度，追到 DSpark 三层 context cache 写入及 eager 注意力。核对冻结路径上的 staging 成本，再记录合法后端与图分支。

## 到了哪里

- 冻结路径每轮仅有一个 drafter attention group，三个 layer prefix 都在 gid12。initializer 按 KV group 内的 backend class 和 spec 分组。首个 gid 供输入扩展选择物理 slot。首个可执行组则经当前 config 下的 `backend.get_impl_cls()` 选取。V4.1 的 cache backend 名称沿用 Cache，实际由 `get_v41_cp_classes()` 返回 DSACP、PCP 或 base implementation。DSACP 与 PCP 同开会报错。冻结配置启用 DSACP。
- DSpark 使用并行草稿和 eager。首步 `build_draft_attn_metadata` 按组切原始 block table，改绑为每组 query slot，调用 V4.1 `build_for_drafting(..., draft_index=1)` 并以每个 layer prefix 登记同组的 metadata 对象。V4.1 方法委派到实际 builder 的 `build`。DSACP 保有 global cache 与 local query 两份视图，分别处理有效行、QSL、device seq lengths 和 operator packet。
- 上下文建立后，`_run_merged_draft` 先调用 DFlash adapter，按实际 context 数切 slot 并在三个独立层 cache 预写 K，再执行 draft forward。V4.1 的显式 side-effect op 由层名找到实例，按 SWA prefix 取 metadata。A3 SMLA 读取原 KV、block table、local QSL、device used lengths 和当前 packet。提交的任务以 packet 身份等待 ATTENTION stage。按组提交、首读与 release 的代次需同原流程一致。
- 通用串行路径仅在 `not parallel_drafting` 且 DCP 条件满足时推进下一步。MTP 的 cache-only 分支只用一个 executable group 推进长度、位置和 slot，再为 cache-only 组 build。通用 helper 保留越长 mask、PAD、CPU/device mirror、DCP slot 索引、原 full table 的地址域和逐步 staging。FULL 路径另有 runtime mode、updatable backend、ENPU 更新前置或非 ENPU 更新后置的条件。冻结 DSpark 不经过这些后续更新，也没有多组重复扩展。

## staging、DP padding 与退役

- 每组 query slot 由扩展 kernel 直接写入，builder 的 shallow copy 随后把通用 `slot_mapping_group[0]` 改绑成按 gid 的原视图。通用 slot copy 和 PAD fill 的目标在这条 eager 路径没有设备读者。通用 seq/QSL 缓冲确实被 CP builder 使用，仍需保存原长度、DCP 资格与 padding。DP 取全组最大 query 行数后，只在存在额外行时填 input IDs、positions 与每组 query PAD。context 消费者切到 `_dflash_num_context`。空跑的 `_clear_dummy_slot_mappings` 独立清理每组两张 slot 表。
- 首步 build 返回后，通用代码 gather 样本位置并 clone 外层 common 的 block table。首步 CP/global metadata 已绑定 raw per-group view，后置 clone 无法改变这份已交付的对象。gather/clone 输出进入串行下一步的分支在 DSpark 关闭。runner 的 ephemeral execute state 在采样前已摘取并清空，返回后只传草稿 IDs/概率，未找到这两个后置结果的冻结路径消费者。仍保留 `token_indices_to_sample` 的真实 copy 与 padding。
- DP2 的 MoE draft 走 CPU 组规约。每个 rank 写自身 token 数与 mode，规约后为 draft 生成一致的最大输入长度；eager 依然需要这次同步。仅把启动移到已知 `B × Q` 后、与当前 first pass 重叠，须保留同一 collective 序列、真实与 dummy 参与方、错误前置条件和等待点。DP1、dense skip 与带 KV consumer 的合法控制另走原分支。
- 两个 CP builder owner 的字段分别服务 global KV 发布和 local Q。global builder 的 `build_query_metadata=False`，local builder 在 A3 非压缩 SWA 发布 SMLA packet。任务若异步，closure 捕获本轮 QSL、设备 lengths、topk 长度和输出 packet；先记录 inputs-ready，首次实际 SMLA 读按 packet 身份等待，后续同身份省重复 wait。release 给下轮 producer 留复用 fence。冻结 eager 的 release 在 runnable 返回之后，异常若越过该处仍需按原 reset/恢复边界审查。不能只凭 wrapper 返回就允许重写 packet。

## 可共享的整数数据

global builder 已为每个 query 行生成逻辑 SWA 索引和可见长度。CP 取实际 local 行切片后只转交索引，local builder 再算 `(indices >= 0).sum(..., dtype=int32)`。冻结窗口 128、K5、最大序列 34816，索引宽 256。对同一有效行，`S` 为设备长度，`Q` 为该请求 query 长度，合法非负位置对应 `L = S - max(S - Q - 128, 0)`，其余列为 -1。只要 `0 <= L <= index_width`，沿同一 actual slice 传 global 的 INT32 长度，值与 local 计数相同。0 query 与非法行按原跳过及 sentinel 分支。Torch fallback 的长度 rank、物理索引入口和 FULL 固定地址分支保留各自合同。 已对 34812 个合法长度做整数对照。配平细诊断每卡 11 次 global 索引 kernel 合计 0.062391 ms，约占窗口 0.0061%。global kernel 改写单独继续停止，可见长度共享仍随完整交付评估。

两个 builder 也从同一个 raw query slot 视图做坐标转换，local 版本先裁 actual 行。冻结 DSACP 的 cache 写读 global slot，SMLA 读 local QSL、used lengths、indices、block table 和 packet。可进一步判断 local slot 副本是否需要实体化，或随 global 切片交付。noncausal 的 CP suffix 已归零，候选可以省零 H2D 与整数减零，仍需保留 clamp、零 query lane 和原 bytes。三项共享放进完整首步交付一起判断。需要覆盖实际末读、bank 首写、异步异常、mask/stride/dtype 与支付后的曝光。target 阶段 gid12 的表地址可参与共同交付，target 与 draft 的 QSL、位置、已接受长度和 mask 各有代次。

## 结果

| 16 卡均值，每卡 11 轮 | 应用墙钟 ms | 有时钟余量的设备 Free 重合 ms |
| --- | ---: | ---: |
| 整个 first pass 到 builder | 60.797375 | 1.399312 |
| DSpark first pass | 5.403406 | 0.115547 |
| DP 同步 | 14.713797 | 0.339500 |
| DP 同步之后到 builder | 3.262781 | 0.064297 |
| 完整 drafter builder | 37.102453 | 0.871172 |
| 后置 gather 至 clone | 0.837734 | 0.016969 |

匹配配平 C8 32K/1K 细诊断 16 卡、176 轮，窗口均值 1025.502734 ms。两处原 FT/body 未匹配保留。flow 链在 176 轮分别绑定 176 个 IndexCheck、176 个 IndexAiCore、176 个 clone memcpy。同步后到 builder 另绑定每轮三次 copy 和三次 fill，含仍须保留的长度/QSL 工作。设备任务分账，host 事件和实际设备执行间隔约 24–28 ms。

| 状态 | 范围 | 判断 |
| --- | --- | --- |
| 已停止 | 冻结 eager 独立 staging 剪枝 | 放宽到同步后整段和后置 gather/clone，应用墙钟 4.100516 ms/卡，加已关联全部原生任务 0.265516 ms/卡，直接时间筛选值 4.366031 ms/卡（0.4257%）。这还包括必要操作。 |
| 已停止 | 单独前移 DP 同步，仅重叠当前 first pass | 每轮以两段较短者计，未扣任何新成本时合计 5.236172 ms/卡（0.5106%）。推进到更早一轮或覆盖更多工作另按完整交付判断。 |
| 已停止 | 删除冻结路径里的重复组扩展、串行推进或 FULL 更新 | 当前一组、eager 并行，三项额外工作均未发生。 |
| 待确认 | 完整 DSpark 首步 metadata、target gid12 与 context 联合交付 | 首步总墙钟含已有 overlap、DP 与必要 builder。候选需合并实删项、保留原操作数与事件，之后才能认领净收益。原 target gid12 单段定价保留。 |
| 已确认可实现 | 新替换原语 | 本段无新增准入。 |

同提示轻量参照约 15.968 s、229/231 frame。假定全员命中、全部直接时间兑现且新增成本为零，独立 staging 和现有 first pass 的 DP 重叠分别约占批时 0.572% 和 0.686%。这里只给匹配参照的乐观筛选，原 C8 与六普通负载的请求实际净省仍未绑定。保持原 target gid12、较大共同 frame、Markov/context 父段独立分账。

下一段沿 runner._prepare_inputs 的原分支和输入行代次追实际 target 读者，核对 num_computed / accepted、位置与 QSL、输入 ID、Engram history 和 GPU staging 的首写末读。用原有 frame 与请求证据给完整交付定价。 每个替换原语逐位一致。本段只读冻结源码和已有 trace，无新 profile 或远端写入。

证据见 [冻结启动字段](draft-output-config.json)、[冻结资源组](input-resource-source-plan.json)、[实际 context/cache 消费者](三份aux到DSpark_context_cache的字段派发与共享判断.md)、[匹配 trace](dspark-staging-price-raw.json)、[定价和源码范围](dspark-staging-review.json)、[SWA Triton kernel 身份](dspark-staging-kernel-identity.json)、[目标阶段 gid12 定价](target-draft-metadata-price.json)。

| 源码文件 | 已核对范围 | SHA-256 |
| --- | --- | --- |
| `vllm-ascend/vllm_ascend/spec_decode/dspark_proposer.py` | 39–75, 127–225, 240–355, 380–429 | `70d216de21875e679d510153ebdb0f0ccf130875a0e644f0673d60b67b6b633c` |
| `vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py` | 300–337, 638–651, 659–687, 736–775, 1004–1011, 1051–1179, 1210–1398, 1454–1508, 1636–1646, 1963–2190, 2491–2512, 2560–2651 | `458fb50e6839f3a353ddc7181ace4fdae02e35093d8ccd1c4327dc24ae5fff31` |
| `vllm-ascend/vllm_ascend/spec_decode/dflash_proposer.py` | 301–319 | `f3b736f44a59ec0831aa7f2849e2eae6d709951e955992b8016f49e772fce3ff` |
| `vllm-ascend/vllm_ascend/models/deepseek_v41/dspark.py` | 78–104, 278–309, 403–412, 443–459 | `fcee256905928917154d8dac94d6611bb83d02da441e8aa154a525ef51e6a601` |
| `vllm-ascend/vllm_ascend/attention/dsa_v41.py` | 56–64, 274–288, 633–675, 839–842, 1427–1444 | `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd` |
| `vllm-ascend/vllm_ascend/attention/context_parallel/dsa_v41_cp.py` | 34–45, 72–95, 112–167 | `039ec9d18e0a8a4aa39f9d5b4bea15204cea3f397a3fce97a5097b266bc9822e` |
| `vllm-ascend/vllm_ascend/worker/model_runner_v1.py` | 825–860, 2045–2124 | `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c` |
| `vllm-ascend/vllm_ascend/worker/device_metadata.py` | 74–150 | `efe13f0e801b70d7fba7498ac44062a8eccfd78e5b6303cf658d33dd22217580` |
| `vllm/vllm/config/speculative.py` | 1450–1465 | `9c642a87264339c0dbc1d0b57330572149df0f9d5e16c6893b9895e50a619e19` |
| `vllm/vllm/v1/spec_decode/llm_base_proposer.py` | 110–131 | `5bd11c41f0a8cf1272b5880d9878043f876bee6c7e04fb781e613f92930b15ab` |
