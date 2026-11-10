# 有效Token与Padding：统计口径与已有结果

**存在padding，但必须按执行阶段统计。** 已还原的原C8草稿body行域中，额外行占17.29%；最新配平批次的完整框架padding比例尚未记录，保留未知。两组数据不能互换。

## 1. 三种“有效”分开记

- 业务产出：最终交付给用户的token。
- 算法所需位置：当前阶段要处理的真实query/校验/状态位置；被拒绝草稿在被校验时仍属于真实工作。
- 执行行域：tensor或SP行域实际传给算子的行数，可能含DP统一、对齐、图档位和占位。硬件tile与buffer容量再另列。

`行域利用率 = Σ当前阶段有效行 / Σ该阶段执行行域`；`草稿接受率 = 接受草稿 / 提出草稿`；`输出吞吐 = 实际交付token / 请求批次时间`。三者不是同一个指标。

## 2. 新批次：实际输入、输出与草稿计数

20261110T090101Z / balanced-reference-1；同轮细函数采集的轻量参考批次

|项目|token数|口径|
|---|---:|---|
|请求输入|262,144|8×32768，含命中的prefix|
|实际输出|8,192|8×1024；TP8副本不重复计|
|prefix命中|236,032.0|查询262,144.0；查询减命中26,112.0不直接冒充GPU计算token数|
|提出草稿|8,805|全局计数器DP0+DP1|
|接受草稿|6,450|接受率73.25%|
|未接受草稿|2,355|属于投机算法成本；不是全部shape padding|

请求usage与全局prompt/generation计数严格核对。当前完整scheduled tokens、逐轮actual/padded以及每阶段validity未记录，不能由这些计数推出全模型padding占比。

## 3. 原C8草稿body：由已有profile还原的行域

原C8 / 20261102T1901 profile，8条原请求；与镜像配平新批次分开

227个活跃波×2个DP=454组，关联3632个rank窗。以每层行域计：真实query为`5×ownB`，DP统一为`5×maxB`，TP8载体为`8×ceil(5×maxB/8)`。三层相同scope可都乘3，比例不变；这不是8840个最终输出。

|分类|行数/每层|占物理行域|
|---|---:|---:|
|真实草稿query|8,840|82.7096%|
|TP8对齐补齐|1,808|16.9162%|
|DP统一额外行|35|0.3275%|
|无请求侧synthetic行|5|0.0468%|
|合计物理行域|10,688|100%|
|额外行合计|1,848|17.2904%|

**17.29%是该草稿body行域的额外行比例，不是全框架无效计算比例、FLOPs比例或可兑现时间收益。** CP已经裁剪部分query；MoE已经按mask过滤部分sender；dummy rank仍可能为远端真实token处理专家。各算子执行哪些额外行，需要逐消费者核对。

典型例：满批草稿20→24行，4行补齐，占16.67%；B1草稿5→8，占37.5%；非对称DP中ownB1/maxB2为5→16，占68.75%。这些分别包含不同的对齐与DP成本。

|真实own B|DP max B|物理行域|DP波组数|本组额外行比例|
|---:|---:|---:|---:|---:|
|4|4|24|430|16.67%|
|2|2|16|8|37.50%|
|3|3|16|5|6.25%|
|1|1|8|3|37.50%|
|2|3|16|3|37.50%|
|1|2|16|2|68.75%|
|3|4|24|2|37.50%|
|0|1|8|1|100.00%|

逐波原始关联及对账：[draft-dp-waves.csv](draft-dp-waves.csv)；Task ID保存在JSON。

## 4. 每一段要查什么

|阶段|有效域 / 执行域|当前结果|
|---|---|---|
|请求与输出|prompt / 实际交付output / 请求usage与全局计数器|新批次已统计：8请求，输入262,144、输出8,192；prefix命中另列。|
|Target输入与图档位|scheduler.total_num_scheduled_tokens（含有效投机校验位置） / SP补齐→graph bucket→DP统一后num_tokens_padded|源码已核，逐轮人口缺失：TP8：6/12/18/24行至少变为8/16/24/24行；最终图档位还要按实际dispatcher记。|
|DSpark body / dense行域|5×本DP真实请求数 / 8×ceil(5×DPmaxB/8)|原C8已还原454个DP波：有效8,840/物理10,688行；额外行17.29%。|
|Attention / KV读写|CP实际query、seq_lens、有效slot和稀疏索引 / Q tensor / 分页cache池 / graph padding容量|有效域保护已核；实际读流量未统计：CP裁剪actual token、无效slot=-1、长度/索引限制读取；cache池shape不等于每次读全部。|
|MoE专家|mask后sender、实际recv及每expert行数 / SP carrier / top-k路由扩展 / GEMM tile|原生计数历史证据可复用，当前全批尚未合账：本卡query=0仍可能接收远端expert任务；top-k复制不是padding；空expert GMM已有guard。|
|拒绝草稿与context占位|当前阶段仍被真实消费者需要的行 / prepare_inputs_padded保留旧QSL并用selector过滤|新批次接受率已统计，占位传播量未测：拒绝草稿在校验时是真实工作；之后可作为padding保留，必须按阶段追踪，不能一律把num_actual_tokens当业务有效数。|
|预分配与硬件tile|有效字节/逻辑M,N,K / 池容量、persistent buffer、tile对齐extent|应分存储与执行单独统计：大buffer不表示全量计算；tile尾块占比也不等于耗时占比。|

额外说明：`prepare_inputs_padded`显式保留被拒绝位置作为后续context占位，QSL不压缩，用selector选择有效输出。因此字段名`num_actual_tokens`也可能描述“保留位置数”，业务有效域须沿读者追。不能把本阶段拒绝率和另一阶段padding率相加。

## 5. 固定配置下的设计量级，不冒充逐轮人口

以下先只计TP8对齐。target uniform宽度6；draft宽度5且两DP均衡。Target仍可能受实际graph档位/DP重派发影响。

|路径|请求B|逻辑行|TP8行域|补齐占行域|
|---|---:|---:|---:|---:|
|target_uniform|1|6|8|25.00%|
|draft_balanced|1|5|8|37.50%|
|target_uniform|2|12|16|25.00%|
|draft_balanced|2|10|16|37.50%|
|target_uniform|3|18|24|25.00%|
|draft_balanced|3|15|16|6.25%|
|target_uniform|4|24|24|0.00%|
|draft_balanced|4|20|24|16.67%|

**FULL桶须另算。** 冻结runner对FULL decode执行`LCM(query_len=6,TP=8)=24`捕获尺寸对齐。若本轮实际走24行图桶，B1/B2/B3/B4的scheduled行是6/12/18/24，图桶padding分别为75%/50%/25%/0%。这是有条件的行数计算，尚无本批次逐轮桶命中人口，不认作实测比例。


## 6. 后续统一采集契约

每轮按target/draft、prefill/decode、real/dummy、DP和阶段记录：真实请求数、scheduled数量、DP统一数量、graph档位、stage-specific有效数量、tensor行域、slot/mask有效量、MoE真实收发量。有效域的定义必须同时记录，避免误用actual字段。

在计数已位于CPU或原有日志时直接记录；设备mask用设备端归约累积、批末异步读一次，避免每轮.item()/synchronize()。预填/解码、target/draft、dummy分别合账。

先按DP组/交付点去重；逐阶段sum(live)/sum(extent)，不平均单轮百分比，不把不同阶段或不同采样相加。

优先优化“确实执行了无效行、且占关键时间”的阶段。仅缩buffer shape但硬件tile不变，或算子已经按valid mask跳过，未必改善性能；新增compact/索引/通信成本必须扣除。

## 7. 源码与范围

- `/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/dspark.py:249–276`；SHA256 `fcee256905928917154d8dac94d6611bb83d02da441e8aa154a525ef51e6a601`。
- `/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py:2389–2468`；SHA256 `458fb50e6839f3a353ddc7181ace4fdae02e35093d8ccd1c4327dc24ae5fff31`。
- `/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py:2653–2669`；SHA256 `458fb50e6839f3a353ddc7181ace4fdae02e35093d8ccd1c4327dc24ae5fff31`。
- `/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:3272–3281`；SHA256 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`。
- `/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:3380–3426`；SHA256 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`。
- `/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:1060–1108`；SHA256 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`。
- `/vllm-workspace/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_v41_cp.py:112–154`；SHA256 `039ec9d18e0a8a4aa39f9d5b4bea15204cea3f397a3fce97a5097b266bc9822e`。
- `audit/idle-attribution-171/source/model_runner_v1.py:6348–6403`；SHA256 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`。
- `audit/framework-timing-171/source/gpu_model_runner.py:894–906`；SHA256 `87c29d08c0bbf66993d8b984811e7325e35e6436242a773feb55ec88f16b2c51`。

源片段完整保存在[source-evidence.json](source-evidence.json)。[机器账本](view-data.json) · [统计校验](validation.json)。本轮完全本机复用既有请求、原生记录和冻结源码；新增NPU/profile/模型请求/服务操作均为0，未重核live环境。全模型逐阶段有效行账尚未完成，不用历史17.29%填补当前缺口。
