# 实施方案

## 1. 先把问题拆对

Flash 线是新模型可运行性与兼容性任务（E0）；原模型线已有功能与历史专项数据，但最新自然暖 C64 不合格（新合同下 E1）。第一阶段不能靠扫参数或叠加旧收益来替代正确性和可重复基线。

优化目标是同输入、同正确性与资源条件下的 TTFT/TPOT/有效吞吐，而不是保证永远选某个框架结构。保留 vLLM 直接面向目标 HBM 的异步加载思路和 SGLang 的前缀/预取/生命周期管理；是否绕过 L2、何时预取、放在哪里均由实际依赖和资源竞争决定，不预设一个路径适合所有场景。

## 2. 资源与通信核验

建议角色见 RESOURCES.json。先记录每台型号、die 数/HBM、驱动/CANN、CPU/NUMA/DRAM、本地盘权重、正在运行的作业与服务；检查 14↔15、46↔47 和每组 4P↔2D 是否在所需 HCCS/SDMA 可达域，不能仅由 IP 相邻推断。

将三类流量分别确认：模型内 TP/EP 的 HCCL 集合通信、P→D 的 KV/混合状态交接、MemCache DRAM→设备的缓存读取。`ascend`、`memfabric` 或 `device_sdma` 只是入口/配置线索，还须记录实际库、协议选择、连接日志、可用物理计数器和无 RDMA 回退证据。不存在的物理计数器标 N/A，不用逻辑字节冒充链路字节。

两框架初始不同时压测。共享 HCCS/SDMA/源节点内存控制器会产生竞争，节点分开不等于隔离。测试客户端从旧 43 AISBench 源码/工具迁移到独立内网 CPU 资源（166 是候选，访问与能力待确认），保持版本、流式 usage、连接池和 CPU 采集一致，避免在被测 D43 上制造客户端瓶颈。Mac 只负责离线依赖和控制，不通过公网慢链路做主负载发生器。

## 3. Flash：用 vLLM 做参考，按语义适配 SGLang

官方 vLLM-Ascend 当前 Flash 文档给出 A3 `v0.30.0rc1-a3`、W8A8、V1 runner 的路线。RC 不是生产稳定证明，文档 PD 也明确仅供参考；仍要固定镜像 digest、框架提交、SDK 与权重实际身份。当前原 GLM-5.3 的 764 GB 权重不能用来加载 Flash。

实施次序：

1. 固定同一份 Flash W8A8 权重、tokenizer、chat template 与量化说明；确定两框架能精确解释同一种量化格式。若 SGLang 需要格式转换，保存逐参数映射、scale/zero-point/shape、数值对照和转换摘要，禁止只改 architecture 或退回 BF16 后仍宣称同精度。
2. 14 上先验证 vLLM 单机真实文本和数值，再 14/15 真实 1P1D。46 上先做 SGLang 单机文本正确性；不要同时调试新模型、PD、MTP、图和 L3。
3. 检查并适配模型层/加载器、KDA 后端选择、卷积状态布局与激活/门控、recurrent state 更新与快照、mHC、DSA/indexer/量化 scale，以及输出头/多模态模块的加载边界。B103 存在 Ascend KDA 实现并不等于 Flash 已正确接线；Kimi 的 SiLU/门控与 Flash 配置也不能直接假定一致。
4. 做固定输入参考 logits/logprobs/greedy 输出及长上下文准确性；量化误差容差事先冻结。小输入通过后逐步长输入、前缀续写与取消/重排，验证状态槽位归属与回收。
5. 上 46/47 真实 1P1D：不仅迁移 DSA 的 KV/indexer/scale，还要迁移对应前缀位置的 KDA recurrent/conv 状态及一致的元信息，验证 P/D token、状态、完成事件和取消回收。先相同 TP，避免把异构 TP 重分片混入首个目标。
6. 逐项恢复兼容图模式、MTP（目标 D5）、连续批处理/混合缓存。MTP 验证接受/拒绝的状态提交、回滚和 draft 所有权；图验证真实 replay，不以 startup capture 代替。短期关闭某项只用于定位，最终性能报告明确哪些优化已启用、为何不能启用。
7. HiCache/MemCache 涉及混合状态检查点，不能只复用原 MLA KV 插件就宣称 Flash 前缀命中成立。先无外部缓存的 PD 功能门控，随后完成同合同自然暖缓存门控再测性能。第一里程碑为文本；图像/视频另建正确性范围，不能用文本通过代表全多模态支持。

从 vLLM 借鉴的是状态语义、算子契约、调度和通信完成条件，而不是照搬 API/分配器。可复用代码须保留许可证和出处；每个移植点记录来源提交、改动与回归，优先小补丁，最终提供可独立上游审查的适配。

## 4. 原 GLM-5.3：先建立新的 SDMA 基线

1. 保留现有 SGLang 六节点。使用 30/31/32/38 做 vLLM P，42/43 做 D；以历史可运行角色布局为起点：P attn DP4 × TP16 / EP64，D attn DP16 × TP2 / EP32。框架全局 TP 的命名不一定相同，必须核对实际 rank 分组，不能仅对齐 CLI 数字。
2. 两框架固定同权重/分片、量化及 KV/indexer 精度、NPU 数量、实际 token 容量、内存余量和输出约束；允许各自兼容的图/融合/异步机制，逐项记录差异。D MTP5 是目标，记录实际接受长度。内存比例无需盲目相同，HBM 实际预算与资源占用必须可解释。新 vLLM 版本在部署前锁定其原模型/SDMA 支持，不能用“支持 Flash”推断所有组合已验证。
3. 先确认真实 SDMA P→D 和原生 MemCache 的已知键/字节及模型正确性，满足独立前缀缓存合同后再测 C16→C32→C64，每档四批。第一批完成不等于通过稳态或整体 SLA。
4. 同输入/同客户端合同运行两框架；正式缓存保留自然驻留，不在预热后强制清 L1/L2。外部池容量相当、缓存命名空间隔离。每请求记录不同层命中、P/D 实际执行、传输量与排队，不能用历史统一共享前缀数据填新表。
5. 对失败先做最小活性复现和有界修复，再用新 run 重测。不得将旧 CPU bootstrap 预取候选直接覆盖 B103；先判断原版是否仍有同机制，必要时重基并做多 rank、撤回/取消/超时回归。
6. 需要纯框架归因时匹配两边相同 SDK 等可兼容条件；否则先标为“端到端部署栈比较”。出现显著差距后交换两组机器重复关键 C64，区分机器/拓扑差异与框架差异。

## 5. 性能诊断与优化候选

统一请求 ID，记录同机 monotonic 时间与跨进程因果事件，不要求修改系统时钟。TTFT 拆为客户端/网关等待、D 准入/握手、缓存查询/读入、P 剩余 prefill、PD 交接、D 首次 decode/图/MTP验证、响应发送；有重叠的阶段用时间区间分析，不把各阶段时长直接相加。

| 优先级 / 候选 | 要区分的机制 | 判断证据与否定条件 |
|---|---|---|
| P0 准入与预取资源闭环 | P 提前占 L2 等 D，D 已分配槽位等其他 P；rank 共识或回收不一致 | 全请求生命周期与槽位所有者；同负载恢复活性且无状态泄漏才通过，CPU fixture 不够 |
| P1 D 驻留与差量 PD | 同前缀回访是否重复发送完整 KV；调度是否打散缓存亲和 | 实际命中 DP、发送索引/物理字节与 TTFT，不能用本地 running=0 表示未计算 |
| P1 KV 表示与复制数 | target/draft/indexer/scale 是否多分配或重复搬运；TP 内重复 SDMA 读 | 消费者需要的真实层数/页数与生产分配/读量，校验所有消费者、拓扑和状态语义；不拿合成布局收益替代模型结果 |
| P2 放置与路由联合 | 源 DRAM 物理归属、NUMA、HCCS 距离、热键、P/D 选择 | 按源/目标/rank 归并实际读量、亲和性收益与负载偏斜；远端/本地不预设独立带宽可相加 |
| P2 异步流水与可选直达 HBM | L2 等待、批量描述符/注册开销、SDMA 与计算冲突、预取价值 | 完成事件、所有权、传输/计算重叠及端到端反事实；不以串行路径微测证明生产预取无用 |
| P3 chunk/batch/融合调度 | 剩余 prefill、集合通信与 decode 搬运竞争 | 固定其他变量，观察 TTFT/TPOT/排队/成功率；旧增大 chunk8192 已在限定用例退化，不盲目开大 |

每个优化只冻结一个机制和变量集：正确性、预测变化、对照、原始产物、否定条件、回退。必须在同业务合同下改善完整服务并守住 TPOT/失败率，才接受为服务优化。局部拷贝带宽、D 差量发送、网关兼容修复各自有价值，但不能累加成整机百分比。

## 6. 交付与协作

两个对话各维护自己 track 的 STATE、run 目录、源码和部署锁；公共输入 manifest 和合同只冻结一份版本。当前主对话负责公共合同与性能判断，用户可按接手提示开两个任务，无需额外定时 agent。

每轮保存 image/engine/kernel/SDK/权重 SHA、实际 argv/源码映射、角色与传输协议、数据 manifest、预热记录、逐请求结果、排队/running、原始错误、CPU/NPU/链路采样及回退。只有完整审计后的 run 进入 accepted；缺证据标 INCONCLUSIVE，不能让优化掩盖失败。

上游来源（2026-10-16 读取）：

- [vLLM-Ascend Flash A2/A3 文档](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/tutorials/models/GLM5.3-Flash.md)。
- [B103 精确源码的 KDA 接线](https://github.com/Ascend/sglang/blob/ce0a8fea5f4245839d174b2b8c55692be1810a7f/python/sglang/srt/layers/attention/attention_registry.py#L507-L519)。
- [Ascend KDA 实现](https://github.com/Ascend/sglang/blob/ce0a8fea5f4245839d174b2b8c55692be1810a7f/python/sglang/srt/hardware_backend/npu/attention/ascend_kda_backend.py)。
- [SGLang Flash cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/GLM/GLM-5.3-Flash.mdx)：GPU 范围不能当 A3 资格证明。
- [vLLM-Ascend MemCache/SDMA 池](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md) 与 [SGLang PD](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/pd_disaggregation.mdx)：版本锁定后仍须核对实际传输路径。
