# M034：gateway 缓存亲和的真实源码路径与支持边界

本轮获取固定提交 `ce0a8fea5f4245839d174b2b8c55692be1810a7f` 的 gateway 全部 **408 个文件**，逐项验证 Git blob SHA1 与 SHA256，附录保存27个相关文件及18条源码观察。**没有 Rust 编译、CPU 策略实测、Ascend 推理或性能结果。**工具链获取失败，不能将源码检查算作测试通过。总体维持 E0，accepted baseline 为 null。

## 当前做法与这次查明的差异

现有 cache-aware 策略根据文本字符前缀估计副本亲和性，负载失衡时转向最小负载。它是已有能力，本轮没有调整阈值或提交运行代码候选。

| 合同 | 固定源码中的行为 | 实验含义 |
| --- | --- | --- |
| HTTP PD 策略 | 分别取 prefill/decode policy；按角色筛候选，再排除不健康或熔断的 worker | 可以研究独立 P 策略，但实际安装版本、参数与拓扑未知 |
| gRPC PD 策略 | factory 保存两套策略；实际 selection stage 取一次 model/default policy，连续为 P、D 选择 | 配置了 P/D 策略不等于该路径读取它们；这是源码支持性差异，尚无运行见证 |
| 缓存身份 | tree key 为 `pool::normalize(worker.model_id)` | 隔离 P/D/regular；没有独立 adapter 键，仍需核实部署及引擎身份合同 |
| 亲和估计 | Unicode 字符匹配比例严格大于 cache_threshold 才走命中分支；选中时先写树，失败回调仅记日志 | 失败或取消请求可能留下估计，不能当成实际 KV 命中 |
| 低匹配路径 | 选择最小请求负载，平局随机 | 顶部注释“选择最小树大小”与当前实现不同 |
| 淘汰预算 | `tenant_char_count` 与 max_size 比较，按字符数扣减 | max_tree_size 在此是每 tenant 的文本字符预算，不能当作节点数、KV 页数或 HBM 字节数 |
| P 负载 | 非流式 guard 覆盖整个 dual dispatch；流式两个 guard 附着在 response body | P.load 可包含已进入 Decode 的请求；必须另测真实 P 队列 |

HTTP 分支的模型过滤取决于 `enable_igw`；gRPC 分支还过滤连接模式，再按 model、角色与可用性选择。策略以首个健康 worker 推导树键，依赖候选池同模型/同角色的前提。未证实部署满足该前提。HTTP bootstrap host/port 来自选定 P，room 参与 P/D 交接；这不是实际传输、DP rank 或资源回收的运行证据。

源码入口：[HTTP PD](https://github.com/sgl-project/sglang/blob/ce0a8fea5f4245839d174b2b8c55692be1810a7f/sgl-model-gateway/src/routers/http/pd_router.rs#L979)、[gRPC selection](https://github.com/sgl-project/sglang/blob/ce0a8fea5f4245839d174b2b8c55692be1810a7f/sgl-model-gateway/src/routers/grpc/common/stages/worker_selection.rs#L231)、[factory](https://github.com/sgl-project/sglang/blob/ce0a8fea5f4245839d174b2b8c55692be1810a7f/sgl-model-gateway/src/routers/factory.rs#L101)、[cache policy](https://github.com/sgl-project/sglang/blob/ce0a8fea5f4245839d174b2b8c55692be1810a7f/sgl-model-gateway/src/policies/cache_aware.rs#L385)、[字符预算](https://github.com/sgl-project/sglang/blob/ce0a8fea5f4245839d174b2b8c55692be1810a7f/sgl-model-gateway/src/policies/tree.rs#L718)。详细锚点与哈希见 [source-review.json](source-review.json)。

## 生命周期与测试边界

注册 workflow 同时初始化 model policy 与独立 P/D cache-aware policy；移除 workflow 同时清理两类策略，P/D 按角色分派。原始 Rust 测试包含独立/共用 policy 的 P/D 隔离、移除、registry 分派及初始化门槛，HTTP PD 测试还包含健康过滤、DP request 准备和流式 guard 释放。**这些测试已定位，执行数为零。**也没有用 Python 翻译或桩策略替代原 crate。

源码 Cargo 版本为0.3.2、工具链声明1.90；完整固定树没有 gateway Cargo.lock。本机 PATH 无 rustc/cargo/protoc/cmake，存在只读 OpenSSL3 与 clang。将来构建须记录当前解析出的锁文件、工具链和依赖哈希，它们不能证明历史部署二进制相同。

| Run | 实际结果 |
| --- | --- |
| R001 | PASS：11个固定源码文件取得并核验；正确性 N/A |
| R002 | ERROR：Contents API 在 tree.rs TLS 握手超时，前6个文件保留 |
| R003 | ERROR：完整归档下载340秒超时，未进入校验或 Rust 安装 |
| R004 | ERROR：稀疏 Git fetch 在75秒后无法连接 GitHub:443 |
| R005 | ERROR：**源码子门槛 PASS，408/408 文件核验完毕**；随后 Rust bootstrap 下载220秒超时 |
| R006 | ERROR：官方组件 Range 恢复遇 TLS 连接超时，清单校验/组件安装均未完成 |

六次冻结命令均经 ZCode 简单执行器各执行一次，输入未变；外层 CLI 都返回0，其中五次内层失败按原样保留。[执行器审计](zcode-audit.json)仅含元数据，不公开配置或模型 I/O。大下载、二进制与原始日志仅保留本地哈希。[回执](controller-review.json)区分源码成功与环境失败。

## 接下来怎样验证，何时否定

先在网络/工具链可用时运行原 crate 的 cache-aware、HTTP PD 与 guard 测试；再用明确区分的真实 crate 测试，观察 gRPC selection 是否调用配置的 P/D policy。该差异尚未修补，也未归因到当前服务。

原生实验遵守 [native-next-contract.json](native-next-contract.json)：确认任务持有至少两个完整、可选 P 副本及固定 D，核实 gateway transport/版本/参数、原模型 config 身份、Ascend PD+MTP 支持和正确性，然后只改一个已有阈值。历史 NEXTN 参数为 steps3/topk1/draft4；当前运行参数未知，不能改写成单步参考。

先保留2/3并发共享前缀、长上下文和冷请求，再覆盖跨淘汰周期的长文本及失败/取消。源默认 abs_threshold=32 时，在仅有至多3个计数请求且无其他负载的条件下，该绝对失衡门槛不可触发；部署默认值未知。需要同时测 gateway 耗时、字符估计与真实 token/page hit、P 引擎队列、交接重叠、TTFT/ITL/SLO goodput、D 队列和 MTP 接受长度。新增成本包括字符扫描、树锁/heap/淘汰及可选 mesh 同步；热点聚集、负载信息滞后或丢失亲和都可能回退。

硬件上界使用实测依赖关键路径和真实副本缓存状态建立可行调度乐观界；不可用的未来信息只能构成界，不能当成可实现策略。错误模型/adapter/角色、移除后接流、bootstrap错配、重复输出、资源泄漏、冷请求饥饿，或排队收益被重复 prefill 和维护成本抵消，均否定候选。只有一个可选 P 时没有 P 路由选择空间，但当前拓扑未知，不能据此宣称本点不适用。

**结论：源码路径获得限定支持，运行正确性和性能均未建立；L033 INCONCLUSIVE，E0，无候选接受或生产变更。**44/48任务资源分配仍待确认，它也不是多 P 容量证明。
