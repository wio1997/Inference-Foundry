# MoE prefix 表示与多图捕获：2026-10-19

原生 AllGather routing 输出专家 count，W8A8 GMM1 随后转换为 prefix。已有 routing 接口支持直接输出 prefix；本原型只改变既有算子的属性和传给后续 GMM 的 list type。模型、W8A8C8、int8 KV/Indexer、原生通信与全 SDMA 路径保持，未新增算子。

已修复首版多图原型的异常续写，但 target 与 MTP 同时改为 prefix 的服务结果仍混合，保持默认关闭。此前 RoPE 连续表在其独立配对实验中获得的 C4 +14.71% 不依赖此原型；本轮两种模式均启用相同 RoPE 改动，不能把两轮比例相乘当作整体实测收益。

## 图构建缺陷与修复

首版功能测试 native 8/8、candidate 6/8，候选后续输出异常。第二版加入仅重复原生计算的 native-replica 图，其两题完整 64-token 也异常；MTP-only 两题完整序列相同，16 rank 的 32 组真实 MTP core 成对输出逐位相同。问题并非必须改变 prefix 算术才会触发。

原型在同一个 forward context/metadata 中连续手工捕获 native、replica 和 prefix。镜像中 SFA 的 `_int64_kv_slots` 会把 int32→int64 的结果缓存在 metadata 的动态属性上，后捕获的图可能引用先前图负责更新的 Cast 产物。精确函数 AST 的 CPU 所有权模型重现该依赖，现场 block table 和实际 metadata 均为 int32。未直接观测 D2 内部动态缓存，不能声称已唯一隔离全部故障因素。

第三版对每套 family 分别调用原框架 `_warmup_and_capture`，为对应属性 variant 预热至少两次，保留原同步，并用新的 dummy forward context/metadata 捕获。helper SHA 为 `173c720dd79dd656ec9cd88b51eeb373752ca71567254f88bcb1d367fcfac90c`，build `ed1750d5…`；CPU 17 项通过，原生 NPU AST 与其他四个 overlay 未改。12 条 native/replica/MTP-only/target-only/both-prefix 前后对照完整序列全部一致；16 rank×3 family 共 48 组可枚举输入地址、shape、stride、dtype 与运行时一致。这不覆盖内部缓存或全部图数据流。

## 同时修改 target 与 MTP 的有限服务结果

同一 P/D、真正 P→KV→D、官方原样 proxy、四个闭环槽位，输入 256/257/1536/1537，输出 64/128。C1 24 条正式请求；C4 case 轮换 ABBA×3，384 条正式请求和 16 条 warmup。observer、profiler、export 与相关传输在正式计时前结束，native counters 只在请求或块边界计时之外采集。

| 指标 | 原生 | both-prefix | 变化 |
|---|---:|---:|---:|
| C4 吞吐 | 157.046 tok/s | 171.590 tok/s | +9.26% |
| C1 TTFT 等 case 比 | 1 | 0.981957 | −1.80% |
| C1 完成延迟等 case 比 | 1 | 0.959358 | −4.06% |
| C1 TPOT 等 case 比 | 1 | 0.953707 | −4.63% |
| C4 p95 TTFT | 1466.351 ms | 1391.403 ms | −5.11% |
| C4 p95 完成延迟 | 3262.082 ms | 3212.778 ms | −1.51% |
| C4 p95 每请求平均 TPOT | 20.817 ms | 21.512 ms | **+3.33%** |

三轮吞吐分别 +4.71%、+19.01%、+3.86%。第二轮 native block 4 中一条 N1536 请求 TTFT 为 7522.55ms，整块吞吐降到 125.29 tok/s；原始块和请求全部保留，未删除异常值。其具体原因未知，不能把汇总 +9.26% 当作稳定改进幅度，也不能将这一请求归因于 prefix。

两 mode 各生成 18432 tokens；native draft5106/accepted13778，candidate draft5089/accepted13821。generated/draft 变化约 +0.33%，draft/s 变化约 +8.90%；这只是计数恒等式，不能完整分离接受率、设备工作和 host 等待。

普通功能 16 条、C4 功能 8 条、all16 实际 native/prefix target 与 MTP 路径、cached 16 条通过。26 题两模式数值答案全过，完整 64-token **25/26 相同**。quality-0 首次 native 续写选 gamma，candidate 选 beta；补充 native A、native-replica、prefix、native B 四条完整输出均为 beta，且与原 candidate 一致，实际观察到 native 变化。所有差异保留，不将其称为已证明的无害舍入或 CANN 故障，也不作一般质量接受。

10 次 client close 与 16 次完整 peer/reuse 通过，队列和 KV 回零；明确 backend abort 日志/计数均为 0，未证明内部 abort 完成链。

由于尾延迟回退和一次 native 慢块，本轮结论为 `MIXED_SERVICE_RESULT_PREFIX_DEFAULT_OFF`，不是无条件联合 SLO/容量收益。三套图和 private pool 是研究对照成本，不代表生产内存或最大容量。

## 实际执行与下一隔离

正式测量和取消检查结束后，仅新增一条 unprofiled warm 和一条 true PD8 profile。全16 rank 停止，CPU 只导出 rank0，耗时 103.13s。相比此前仅启用 RoPE 的 trace，Cumsum 从 **400 次降至 0**（此前 target 图375次、MTP25次），百万行 RoPE Slice 仍为0。

25 个 MTP core 的 CPU 提交中位 2069.61μs，首个物理 compute 比 CPU 提交结束晚 12.88–40.63ms。profiler 条件下 MTP 依旧明显排队领先，不能据此声称未采样服务已确定关键路径。下一轮固定 MTP 原生，只改变 target 图的 histogram 表示，以缩小变动范围；这是新条件的配对实验，不能与本轮跨窗口数值直接相减归因。

主要证据位于 `evidence/20261018-p-first-seed-device-v2/recovery-v2/`：`moe-prefix-D3-reviewed-verdict-v1.json`、`moe-prefix-D3-quality-decision-v1.json`、`moe-prefix-D3-triage-decision-v1.json`、`moe-prefix-D3-supply-analysis-v1.json`。全部 raw profile 为575877934B，Mac备份86493803B，SHA `9f0bfa2f1dab31480cc5c80b37b19dded322fed216b2684ea0ca6515aac36847`。模型/官方proxy/源码/功能记录另有逐文件 SHA 快照；大 collector JSON 不入 Git，保留本机压缩原件和收据。

## 仅修改 target 的服务结果

固定 MTP 原生、仅 target 使用 routing 直接 prefix 后，完整 C1/C4 和取消检查已完成。C4 为 **161.917→168.664 tok/s（+4.17%）**，三轮 +3.18%/+1.41%/+7.98%；C1 TTFT/完成延迟/TPOT 比为 **0.988229/0.957243/0.934899**。C4 p95 TTFT/完成延迟改善0.26%/6.13%，但 **p95 每请求平均 TPOT 恶化5.57%**，因此仍默认关闭。不能将两次不同窗口的数值相减当作唯一因果分解；MTP 改为 prefix 并非出现该回退的必要条件。

26题两模式数值全过，25/26完整64相同；quality-0与此前已保留的两种输出完全相同。功能16、C4功能8、cached16通过，16rank实际target prefix与两arm原生MTP均有记录。10次client close/16次完整peer及reuse后队列与KV回零，明确backend abort仍为0。两mode各生成18432 tokens，native draft5059/accepted13850、candidate draft5138/accepted13763。证据为`moe-target-reviewed-verdict-v1.json`，原件不删除异常值。

## 保留原生路由和 GMM2 的 int32 Cumsum

为缩小表示变化，下一原型保留原生 AllGather routing 的count1；仅将非负histogram转为int32执行既有Cumsum，再转回int64供原fused GMM1。GMM2收到原始count对象和list type1，MTP仍原生。N≤24、topK8、TP16下histogram总量上界3072，整数转换精确，不改变模型/激活/KV精度。helper为`moe_cumsum_i32.py`，SHA `b4681af7b27f7abc9138d0f3df17684357acb33c224f055155c037131c588c7a`；复用原先17项检查的图family owner，新增15项表示和consumer恢复检查。

同一原生编译模式和nondefault stream的routing+prefix片段，N6为 **58.148→46.577μs**、N24为 **60.592→49.009μs**；原生四项routing输出逐位一致，固定地址输入及expert ids变化后replay仍正确。原尺寸NZ权重、scale、dispose/event/unpermute的完整MoE链8种shape/分布eager和96次交错private-pool replay全部逐位一致，原count未被改写。片段计时不预测服务收益。最小profile中实际出现带`aclnnCumsum_CumsumAiCore`的任务名和Cast，但Accelerator Core字段为N/A，任务名不单独作为物理引擎归属证明。主文档只用于核对接口能力：[CANN aclnnCumsumV2](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-math/aclnnCumsumV2.md)；最新文档不证明本镜像选用了V2。

新D Ready后，16rank每个N6/12/18/24捕获均实际记录75次prefix、75次fused GMM1与75次恢复原生GMM2；功能16、C4功能8、cached16、quality52数值全部通过，完整64仅19/26相同。按有限gate暂停正式性能，保留全部差异；7个case的native-A/原生副图/target-i32/native-B共28条对照数值全过，**初次和复测的每种candidate完整序列均在原生或原生副图中实际出现**，4个case原生A/B本身不同。因此允许继续有限服务计时，不声称通用质量接受或已证明无害舍入。

旧P剩余有限时长不足完整验收，SHA备份并空闲停止后只替换P，保留已加载D。正式结果尚待新P Ready及新真实PD功能检查；精确CID、有限期限与唯一运行标签见HANDOFF。证据为`cumsum-i32-fragment/`、`cumsum-i32-full-fragment/`、`cumsum-i32-paths-v2-summary.json`、`cumsum-i32-quality-control-summary.json`、`cumsum-i32-quality-reviewed.json`。
