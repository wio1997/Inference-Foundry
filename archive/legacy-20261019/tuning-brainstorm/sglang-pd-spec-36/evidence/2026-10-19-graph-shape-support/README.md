# M021：NPU 图形状桶、MTP 宽度与 replay 准入

固定安装源码 `ce0a8fea5f4245839d174b2b8c55692be1810a7f`。**本轮证实了 CPU 侧策略边界，没有创建原生图、运行 NPU replay 或测性能。M021 的运行/性能结论 INCONCLUSIVE，整个 PD＋MTP 任务仍为 E0。**

原候选是“如果 2/3 请求被补到过大桶，就增加高频桶”。实际源码表明，请求桶还要通过 MTP 宽度、attention/TBO 对齐和资源上限过滤；直接加入 2/3 不一定改变捕获结果。

## 当前实现及单位

本轮新采集的 18 个源文件全部匹配 installed Git HEAD，见 [source-index.json](source-index.json) 与带行号的 [source-excerpts.json](source-excerpts.json)。源文件存在、函数可提取和分派可计算，都不能替代实际 runner 实例和图表证据。

| 位置 | 当前行为 | 对候选的约束 |
| --- | --- | --- |
| NPUGraphRunner | 继承 DecodeCudaGraphRunner；父类使用 BaseCudaGraphRunner 的选桶逻辑 | 不能只查看 NPU 子类后就判断缺桶 |
| `get_batch_sizes_to_capture` | 从 resolved decode.bs 取候选；按 attention/TBO 对齐筛选，按对齐后的 request-pool 上限裁剪、去重排序 | 配置桶 ≠ 过滤后 capture_bs；还不等于原生图成功捕获 |
| width | target verify 走算法 hook；EAGLE target 通常采用 draft_tokens，draft decode 走 topk，draft extend 走 draft_tokens；adaptive override 需另查 | steps、draft_tokens、topk 不是同一个宽度；本轮测试明确注入 width=1/4，不推断实际实例 |
| padding | 选择最小的 `bucket >= raw`，超最大桶断言拒绝；禁用 padding 时要求完整 ShapeKey 存在 | raw=3 可准入但补到 4；准入 True 不等于 replay 成功 |
| DP gather | decode-family 的 original_global_num_tokens_cpu 在此代表每 rank 原始请求数，以 max 选统一桶；还可受 DP vote 保护 | 不能各 rank 独立按本地请求数选不同 graph/eager 路径 |
| ShapeKey | size、stream_idx、LoRA/prefix variant、attention variant 共同标识 | 同一个 batch size 不能无条件共享图；NPU backend 的 can_run 是键集合检查 |
| uniform-width guard | 正值 actual per-request width 必须等于 capture width；未设置或 0 绕过这一 guard | 仍需其他准入条件；该 guard 不是完整输入合法性校验 |
| config schema | decode.bs 是请求桶；prefill.bs 是 aggregate-token 桶 | 同名 bs 的两种单位不能合并比较；prefill 配置列表不证明 prefill 图已启用 |

固定源码的 alignment 先取 1，TBO 时乘 2，需要 gathered buffer 时乘 attn_tp_size，再满足 attn_cp 的整除要求。非 TBO 以 `bs * captured_req_width` 做对齐；TBO 则以每请求宽度 1 做对齐。`require_gathered_buffer` 在本轮是明确注入的依赖条件，实际进程取值未知，不能仅由 TP=16 推断它为 True。

## 实际 CPU 探针与反例

[完整结果](cpu-policy-result.json)包含每个配置、键、准入和更新用例。执行的是固定源码函数或选定方法，注入配置、后端字典、CPU tensor 与 graph.update spy；没有把模拟对象当成原生图。

| 检查 | 实际用例 | 结果与覆盖限制 |
| --- | ---: | --- |
| 配置/对齐/池上限/compile 子集 | 864 | 通过；逐 raw 尺寸核对最小合法桶和超限断言，配置为夹具 |
| 完整 ShapeKey membership | 81 | 通过；同 size 的其他 stream/variant 不命中 |
| 宽度、padding、DP max、LoRA 准入 | 640 | 通过；保留 raw=0、正值宽度不匹配及超最大桶 |
| embedding override、DP vote、encoder、TBO、NGRAM 保护 | 32 | 通过；附缺失 DP request-counts 的实际 RuntimeError |
| NPU 更新键选择 | 16 | 通过；依据 captured mode 选 V2 键，runtime IDLE fixture 不改变选择 |
| Python update payload | 额外一项 | Tensor 类型分支生成 CPU int32 参数；native `.update` 为 spy，未验证异步读取/所有权 |

下面三行都固定 attn_tp=16、capture width=4、pool=32，唯一对照是在候选配置 `[1,4,8,16]` 中加入 2/3：

| 显式依赖条件 | 原过滤结果 | 加入 2/3 后 | raw 2/3 选桶变化 |
| --- | --- | --- | --- |
| gathered buffer=True，TBO=False | 4/8/16 | 4/8/16 | 4/4 → 4/4，无变化 |
| gathered buffer=False，TBO=False | 1/4/8/16 | 1/2/3/4/8/16 | 4/4 → 2/3 |
| gathered buffer=False，TBO=True | 4/8/16 | 2/4/8/16 | 4/4 → 2/4 |

这是反对“加入 2/3 必然有效”的实际函数反例。第二行只证明可以减少这个夹具的 padding token 数；没有证明 NPU 更快。对齐要求成立时可继续研究合法的最小桶，例如 width4/alignment16 条件下的请求桶4，但必须先确认真实条件、内存余量和负载频率。

首次 R002 在 `ceil_align → ceil_div` 抽取依赖缺失处报 NameError，inner exit1；外层 CLI exit0 不能覆盖它。[首次错误](first-failure.log)及冻结脚本、哈希完整保留。主控只在新脚本/新 R003 中补齐原样 ceil_div 依赖，结果通过；未修改 SGLang 源码，也未让 ZCode 自行修复重跑。任务工具因原 invalid Run 仍在而拒绝 accepted；本轮保留 INCONCLUSIVE，具体解释见 [verdict-boundary.json](verdict-boundary.json)。

## NPU 更新路径不能一概而论

`_uses_v2_seq_len_update` 使用捕获图的模式选择 V2 更新键，以覆盖运行时 IDLE rank。实际长度是否更新、何时读完以及 key 是否被底层算子接受，本轮都未执行。

NPU target 的 ragged/DSA/DSV4 分支直接 replay，其他架构才走通用 CPU seq_lens 更新。draft 的 DSA/DSV4 也跳过通用逐 step CPU 更新；attention-DP 的 seed 准备还可能触发 MIN all_reduce，以统一 graph/eager 决定。不能把这些带架构和并行门控的源码描述成“GLM 此时已执行通用 update”。M017 的 CPU 镜像保留依据仍包含已定位的 eager metadata 消费者；本轮没有证明 DSA 整条链可以删除镜像。

## 只读观察配置

[HTTP200 限定字段](observed-resolved-config.json)显示观察进程 decode.bs=`[16]`、decode.max_bs=16、padding 启用、compile/TBO=False、EAGLE draft_tokens=4/steps=3。prefill.bs 是另一种单位的列表。该进程 `disaggregation_mode="null"`，不是本任务 PD 资源和基线。

这些是 resolved 配置；实际过滤后的 capture_bs、target/draft 各自宽度、adaptive override、graph key 表、实际 replay/fallback 数、capture HBM 与时间都 UNKNOWN。没有读取/修改进程内存，也没有向服务提交模型请求。因此不能据 `[16]` 宣布 2/3 请求实际补到16，更不能直接计算服务收益。

## 必要工作、上界与下一步

若同一阶段 raw 请求数 B、固定宽度 w、选中桶 P，真实 token 行数为 Bw，形状 padding 行数为 `(P-B)w`。这是工作量账本；不同桶的有效算力、attention 上下文长度、MoE 专家并集和通信竞争可能不同，不能把 token 比例直接当时延比例。

需要分别测量真实形状 eager/最小合法图、replay 启动、输入刷新、padding kernel 开销、捕获时间及常驻 HBM。启动与刷新有依赖顺序，只有实证可重叠的项才能取 max；剩余空间也要计入新增桶挤占 KV、额外启动预热和选择成本。本轮可达带宽/算力和所有设备时间均 UNKNOWN。

冻结前先采集实际 P/D 运行合同、target/draft 的 w 与合法桶、raw/padded 形状频率及 fallback 原因。只增加一个真实高频且合法的缺失桶，再做 A/A 与 A/B、接受长度和每有效输出 token 总成本对照。没有资源归属确认前，继续 M019 事件/stream 所有权审计；不以模拟图替代 Ascend PD＋MTP。

## 复现与出处

成功脚本 `scripts/probe_graph_support_cpu_v2.py` 的默认布局是本地 Task；发布的 6 个 `source-inputs` 是不改字节的 CPU 依赖快照，可按脚本路径映射到 `sources/installed/python/sglang/srt`。需独立 CPU PyTorch/NumPy 环境；默认输出为独占 Run 目录，不能覆盖既有结果。它不是原生图测试脚本。

[zcode-audit.json](zcode-audit.json)核验每轮精确 session：配置别名 deepseek-flash，后台实际 release UNKNOWN；执行冻结命令各一次，输入哈希未变。原始 model I/O、认证设置未发布。[artifact-index.json](artifact-index.json)记录发布文件及本地 raw source 入口哈希。源码归属 SGLang Team，快照保留原文件头。
