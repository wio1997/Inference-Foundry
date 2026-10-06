> checkpoint158之后按[AGENTS](../../AGENTS.md)执行：下列契约是correctness要求与设计知识，不要求增加同名模块或保留通用Runtime层。Sol性能架构师可删层、改变状态/调度/执行顺序、大幅重构；高成本大范围改造触发独立Astra Challenger Review。代码假说→Patch→Correctness→matched A/B→真实完整E2E，无匹配对照不算性能成果，无完整E2E不提升Current。连续无代码KEEP、模块增长而Current不升或compatibility长期占主线须Goal Review。标准PD含真实KV Transfer；各副本完整Prefill+Decode是Full Replica / Complete-request Placement，配置/部署/代码收益分账。历史Run/raw/裁决不修改。

> 研究快照：整理于2026-10-01，随后作为按需资料发布；文中未提交/未运行陈述描述研究采集阶段，不代表后续Git发布状态。实际运行状态以局部HANDOFF和Git为准。

# GLM 专用动态 Runtime：内部契约设计建议

日期：2026-10-01（Asia/Shanghai）。本文件是架构建议，不声称当前部署已经实现；没有访问运行机器或执行启动/性能测试。设计目标：开发GLM专用推理框架，V1/V2按代码Gap选择，允许大范围代码重构；输入、输出长度、请求集合和PD/并行配置以后都能变化。当前阶段使用现有正确算子，优化控制、调度、存储与通信编排。

用户明确功能都要，能力清单不能成为白名单。默认保留GLM对应的完整推理服务能力与语义，由Agent从接口、参数、协议、状态分支、现有测试和真实行为恢复并持续扩展。流式、动态进出、取消/超时、prefix cache、MTP、tool/JSON/grammar、logprobs、抢占/恢复等是必须保留的基本能力示例，不是全部边界。尚未验证的部分明确待验证，不能默默关闭或降级换取跑分。

专用与兼容执行路径保持同一请求/状态/提交/输出契约；复杂功能可先保留可运行适配，再依据关键路径专用化。兼容路径的覆盖、切换和成本进入E2E，不把fallback当永久不优化的豁免。本文接口是实现建议，可以重构，功能完整性不能由早期表格缩减。

## 1. 专用的边界应落在模型语义，不落在某一次 benchmark

GLM 专用可以固定模型实现族、权重/量化身份、Full/Shared IndexShare 规则、MLA/MoE/MTP 语义和对应算子接口。请求参数、上下文增长、终止时机、并发、批次成员、草稿深度配置、P/D 角色与并行拓扑属于运行配置和动态状态。

V2 可以用作启动、权重/算子/通信组装载、输入 arena、Graph 管理和 MTP 执行的基础，但专用 Runtime 应能拥有请求/步骤的控制权；不是只能给旧 scheduler 加参数。最终保留哪些 V2 组件、重写哪些 coordinator，由依赖和收益决定。

固定 buffer 地址服务于已捕获图；固定 shape 是一个容量/执行分型；两者都不意味着固定 request IDs、固定 cohort、固定输入/输出长度或固定 PD 分配。专用执行层需要把“动态逻辑状态”和“稳定物理执行地址”明确隔开。

## 2. 建议保留的内部对象

以下字段是概念契约，实际代码可以合并对象或用结构化张量表达，不要求逐个增加 Python 对象。

| 对象 | 必须表达的内容 | 作用 |
|---|---|---|
| ModelContract | 模型/权重/量化 revision、主层模式、算子与cache ABI、可选MTP语义 | 避免在一次bootstrap后默默换模型或cache格式 |
| TopologyEpoch | P/D/combined角色、TP/EP/PP/CP组、rank映射、connector能力、通信顺序、cache分布 | 一个Graph/transfer plan所属的执行世界；多拓扑通过新epoch建立 |
| RequestState | request incarnation、输入tokens与prefix、采样/终止配置、已提交输出、逻辑位置、状态版本 | 请求生命不等同batch row或slot |
| SlotLease | arena id、slot id、slot generation、当前request owner、仍在途的读写/Graph/transfer引用 | slot可循环复用，但陈旧结果不能写入新请求 |
| KVLease | cache role/layer/group、分配世代、有效内容版本/范围、block映射、writer completion与reader约束 | 分配身份和内容版本分开，地址相同不代表状态相同 |
| BatchView | 一次执行的batch epoch、active req→row映射、每row token span、logical positions、valid/padding、局部/全局shape | 让compaction、混合prefill/decode与Graph padding不改变请求语义 |
| IndexView | 所属Full层/草稿首步、query身份/顺序/位置、KV历史范围/版本、valid rows、生产完成 | Shared层/后续MTP step消费的typed选择结果 |
| SpecAttempt | 每请求target commit version、K/step、draft ancestry、verify结果、recovery/bonus及提交范围 | speculative tentative state 与已提交请求状态分开 |
| GraphLease | capture key、稳定arena/workspace、metadata generation、拓扑epoch、in-flight读写范围 | 重放前绑定动态值，重放后才允许对应物理资源重用 |
| TransferTicket | request/slot/KV世代、源目标cache spec、实际传输有效范围、完成/失败/取消状态 | PD接收完成及安全回收证据，不能只携带裸地址 |
| OutputCommit | request identity、递增commit序号、有效token范围、终止标记、交付进度 | 对外输出有序、只发布已提交tokens，并与内存回收解耦 |

不要求热路径逐次hash大型KV，也不要求所有世代上送Host。世代和valid range可以用固定大小的设备状态；完整快照/hash用于抽样诊断或验证。热路径只保留正确性需要的最小身份与依赖。

## 3. Full/Shared IndexShare 契约

GLM-5.2 参考配置为78主层，Full层21个，Shared层57个；Full索引是 `0,1,2,6,10,...,74`。该规则是模型身份的一部分，不是动态scheduler自由改写的索引器跳过策略。模型版本更换时，由新 ModelContract 给出模式。

Full层生成的 indices 必须绑定到当前query的逻辑位置与行映射、Full锚点、实际可见历史和对应内容版本。Shared层复用同一逻辑query的选择结果，不是“碰巧shape一样就复用”。

跨层共享buffer可以固定地址，具体实现可用一个工作区、多个ping-pong工作区或明确consumer fence；要求是在最后一个旧消费者完成前不被下一组/下一batch覆写。是否需要两份buffer由实际stream和Graph并发决定，不预设数量。

Batch compaction或重排时，IndexView必须按request/position重新映射或生成正确的gather view；batch row不是缓存身份。若新query、KV前缀版本、行顺序或Full锚点不匹配，应重新计算或明确变换，不能把“reuse”当默认安全。

主干跨层IndexShare与MTP首step→后续step共享是不同contract；不得由target最后Full层的indices未经证明直接替代MTP自己的首步选择。

## 4. MTP / 验证 / 提交契约

一层MTP权重可以支持多步draft，不能把`num_nextn_predict_layers=1`当深度固定1。Runtime需要允许当前选定MTP配置、不同端的配置以及未来策略变化，但性能比较应说明是否改变K、算法工作量或有效接受轨迹。

每个SpecAttempt属于一个确定target commit version。Draft step的token、position、上游hidden、indices和KV可见范围必须来自允许的先前生产者；不得使用尚未产生的接受结果、未来token、路由或离线保存的轨迹。

MTP的hidden输出有不同消费者（logits、下一draft输入等），应保存真实归一化/投影约定；不能仅因维度相同而互换。多步IndexShare/KVShare按照实际实现的可见历史和投影语义定义，不能简化成无条件alias主干全部KV。

验证结果输出每请求的有效accepted prefix、必要recovery/bonus、逻辑序列推进与可发布输出。Commit是唯一改变已提交请求进度的事务。Tentative KV可保留在物理页内，但拒绝后的无效后缀不能成为下一步合法读取的历史；恢复valid ranges/block映射不一定要求物理清零。

当请求在accepted prefix内遇到EOS、用户stop或最大输出长度时，提交/发布需要截到其语义终点。不能等全部cohort满长度，也不能把Graph的K+1验证行或填充值当输出。

Device state可以成为步骤状态的权威副本；Host镜像允许延迟，但有Host真实消费者的控制决策必须读取已完成版本。请求终止、slot归属和新请求绑定不能建立在尚未回传的乐观计数上。无需为全部请求加入统一Host fence。

## 5. KV与PD状态契约

Cache按照角色区分主干MLA、Full indexer、实际需要的MTP状态和其他backend缓存。TransferPlan描述当前端实际提供/消费哪些状态、哪些共享、哪些在D端重建，以及字节布局、dtype、block size、layer/group映射和partition转换。

不把潜在状态全部强制上网，也不宣称target forward结束就完成整个handoff。某个consumer可以运行的条件是其真正所需状态到达且对应writer完成；其他层/组可继续传输。是否逐层可用、是否允许早启动、是否需要所有rank rendezvous由实际connector和计算依赖决定。

P/D可以使用不同草稿配置或parallel layout，前提是transfer descriptor能表达兼容映射或合法重建。跨布局重排如改变现有primitive服务路径，应记录其成本类别，不能把全部结果算作固定primitive下的调度收益。

用户当前有两台同规格、卡配置相同的910C服务器，结合前文按每台8卡16芯作计划解释，实际rank/实例映射未远端核验；未来明确可能1P3D等配置。Deployment应独立表达P/D池实例数量、每实例TP/EP等并行与容量、server/rank placement和传输关系，不硬编码一对一或把一个实例等同一台机器。共享GLM执行核心，池级路由/准入/背压与局部执行规划通过当前部署实例化。拓扑扩容与当前机器资源切片都可研究，实际模型/cache容量决定可行配置。

多D时，目标D选择与KV位置、prefix命中、剩余cache、预期decode工作和传输等待有关；P端准入需要反映D池可消化能力。选择时机、重路由、复制或迁移由Agent按依赖与实际成本研究，不预设把完整KV广播到所有D或把请求固定轮询。动态增减实例若实现，建立资源有效性与在途请求规则；可配置多种部署不等于已经具备无成本在线弹性。

KV有效范围以逻辑token与block/version表达；prefix hit、增量prefill、部分最后block、草稿extra reservation要分清。固定output cap或完整prompt传输不能作为唯一布局。

Abort不是完成写入的证明。发起取消后，已发出的P计算、远端写入、D当前读者或Graph可能仍在途；只有必要读写完成、取消确认或隔离后的安全条件成立，旧KVLease才可回收。异步完成通知携带request/slot/allocation generation，陈旧完成不能把新请求标成ready或finished。

TopologyEpoch变化应建立新的通信组、cache/transfer descriptor和capture；在变更边界drain或迁移实际仍活跃的状态。支持多个拓扑不默认意味着在途请求能无损热迁移，在线迁移若研究应有单独契约和成本。

## 6. 动态请求、slot与终止

建议分开三个事实：

1. **逻辑完成**：最后一个有效输出已提交，或取消已阻止进一步逻辑提交。
2. **输出完成**：客户端需要的有序结果与终止事件已交付；慢客户端和输出queue可以有自己的背压。
3. **内存可回收**：旧请求引用的Graph、采样、KV读写、PD写入和必要回传均已不再访问相应地址。

三个事实无需全都变成一个同步barrier。只要确切resource安全，空出的slot可以绑定新请求；旧请求输出记录可在另一小型队列继续交付。若资源暂不可回收，scheduler应看到该resource的pending credit，而不是强制所有请求等cohort结束。

新请求进入既有arena，获取新SlotLease generation和独立KVLease。长输入可以分块prefill；不同context/output长度的请求可以同时存在；完成与补位以单请求状态驱动。Capacity不足时采用已声明的排队/扩容/迁移/拒绝策略，不能默默截断输入或把输入限制成最初benchmark长度。

参与集合通信的plan必须在同组rank上保持一致。请求退出或取消可让该row失效，但不能让某个rank擅自跳过仍被其他rank调用的collective。语义必要的是协同的通信序列，不是每个Python步骤都加全组barrier。

## 7. 固定地址与shape特化的正确用法

Capture key首先由实际backend要求决定，通常涉及角色/拓扑epoch、阶段、token/request容量、spec宽度、cache/quant ABI、相关metadata与通信布局。不要随意把每个request的输入/output长度都塞进key，也不要假定仅batch size足以唯一决定图。

一个容量为B的Graph可承载当前有效n≤B个请求，前提是有效row、position、slot/block映射、mask和padding正确，且padding不会制造非法KV写入、额外提交或不同rank通信序列。Padding工作量和资源开销进入性能成本。

例如capacity16的decode graph可先执行12个请求，之后3个完成、5个新请求就绪时执行14个；row可以重排，request identity与slot/Index/KV映射跟着BatchView更新。逻辑output长度可分别为80、1000或动态EOS，与capacity16无关。

长上下文KV通过动态page/block pool增长；Graph引用稳定metadata arena和当前block映射，不要求每请求预留1M tokens。如果某backend把部分长度或workspace固定入capture，它应成为明确shape class；超出class时选择另一个已就绪capture、兼容路径或扩容。

不预设所有布局都有eager fallback。部分V2细粒度TP路径要求每步使用capture，公开代码也有该检查；这种配置需要合法的Graph分派、按需capture、确实可运行的兼容路径，或能够完成的等待/资源重建。拒绝条件只沿用完整功能合同允许的非法输入或资源限制，不能仅因尚未capture就永久排除原本合法、已支持的请求。Fallback应实际能执行当前topology与算子，而不是纸面保底。

稳定buffer可用arena/ring/pool lease管理。重新record Event、覆写metadata、复用临时输出和切换Graph均需对应generation及consumer完成条件；“固定地址”不能被解释成“只有单batch才安全”。

## 8. 模型/算法真依赖与当前框架顺序

| 边界 | 应保留的真实关系 | 可以研究消除/替代的实现顺序 |
|---|---|---|
| 主干层 | 本层FFN输出/残差后才能生成下一层hidden | 中间Python调用、metadata重建与额外Host往返 |
| Q/KV/Indexer分支 | 各消费者需要对应生产值，Full indices先于其sparse attention消费者 | 同输入的独立投影/准备被单stream强制串行 |
| Shared层 | 对应Full锚点indices和正确KV版本 | 每层重复构造相同metadata或拷贝indices |
| MoE | 路由决定dispatch；专家结果与shared结果在combine处join | shared与routed分支人为串行；过早/过宽通信wait |
| MTP steps | 下一草稿输入依赖真实先前token/hidden；配置的共享状态有效 | 每step都要Host解释、全量计数回传或不必要Graph边界 |
| Verify/Commit | 有效接受prefix与恢复结果先于已提交进度更新 | CPU每次拿完整draft/verify输出才能启动所有后继 |
| KV写/读 | 正确writer generation完成后对应reader可读 | 无条件全模型sync或等全部不相关层传输 |
| PD handoff | D所需cache/seed/metadata就绪；跨layout转换完成 | target结束后才统一打包、register、发起所有传输 |
| request完成 | 有效结果有序提交/交付；资源引用结束后才能重用 | 固定cohort统一park、bulk-only输出与全cohort释放 |
| collective | 同communicator上匹配的组成员/次序及输入就绪 | 无数据依赖的Host全rank barrier、把peer等待当传输service |

这些是研究候选而非宣称存在收益。移除顺序边时还要检查buffer WAR/WAW、地址别名、Graph ABI、资源争用与输出语义；添加private scratch、ring arena或commit阶段可能使提前准备合法，但额外copy/内存也有代价。

## 9. 针对大改动与慢启动的交付粒度

用户允许大范围重构，因此可把相互依赖的状态/执行设计作为一个完整Runtime版本交付，不必强制每轮一行patch或一个微机制。仍应让各逻辑模块可诊断：请求/slot状态、BatchView、Index/MTP状态、Graph lease、PD ticket与OutputCommit有明确边界。

推荐把会影响启动/capture的内容编成一次构建与bootstrap，常见调度策略用已声明的runtime开关选择；一个启动session可复用已加载模型和有效capture进行多种负载、顺序控制与对照，前提是每轮请求/缓存/observer状态恢复可证明。改模型/quant/backend/topology/capture ABI时不能假装同session自动等价。

重构的离线验证重点可以放在高价值契约：动态row映射、slot陈旧结果、KV/transfer世代、接受prefix/EOS截断、拒绝回退、取消后回收、capture选择与rank一致性。它们能在不加载完整模型的状态机/模拟adapter中先查出控制错误；不能替代NPU/Graph/多rank的真实验证。

首次真实启动应验证实现版本、权重/算子加载、cache/Graph/connector实际绑定，然后覆盖动态补位、不同长度、MTP回退、终止、PD与目标拓扑关键边界。选择的回归范围跟改动风险相匹配，不把整个benchmark矩阵每次重复。性能最终看完整服务输出和当前工作负载范围；固定小场景可以定位，不能升级成框架永久产品规格。

## 10. 来源与当前支持状态

模型结构依据见 `work/glm52_architecture_research.md`，官方GLM-5.2 config revision为 `cf457fa734ab149ffef225f80893eb38c6ff5cdc`。

公开V2/PD定位来源来自此前固定公开源码：vLLM `4c2d277643e217344056e1d2c42115d5f005912f`，vLLM-Ascend `a8fcedb03d93e60efceddbfc912406f7fa491d57`。二者独立main快照只是定位，尚未证明与用户0.27安装兼容。

- [Ascend V2 runner](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/worker/v2/model_runner.py)：AscendInputBatch/InputBuffers、输入padding、MTP/sampler与Graph路径，以及部分细粒度TP的Graph要求。
- [V2 MTP speculator](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/worker/v2/spec_decode/mtp/speculator.py)、[autoregressive speculator](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py)、[draft Graph manager](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/worker/v2/spec_decode/autoregressive/aclgraph.py)：可作为组件入口，不代表GLM在用户backend已验证。
- [KV connector base](https://github.com/vllm-project/vllm/blob/4c2d277643e217344056e1d2c42115d5f005912f/vllm/distributed/kv_transfer/kv_connector/v1/base.py)、[Ascend pull facade](https://github.com/vllm-project/vllm-ascend/blob/a8fcedb03d93e60efceddbfc912406f7fa491d57/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/connector.py)：Scheduler/worker状态与传输能力接口，实际connector实现需核验。

本文件提出的通用内部契约不是当前源码已经具备的能力清单，也不是固定SOP。实现Agent应将其映射到实际安装并保留可证伪的动态行为与完整E2E证据。

## 11. 用V2资源层演进，而不维护整仓库大fork

可检验的一种资源/控制边界为：复用可运行V2的model loader、量化/attention/MoE等现有算子、cache分配、HCCL组、MTP/sampler、Graph及connector必要能力，通过少量明确适配接口提供给GLM执行层。GLM拥有动态batch/step控制、请求与slot状态、补位、PD协调和发布；这些控制代码允许大幅重写。保留什么、替换什么由实际依赖和收益决定，接口名与文件拆分不是固定要求。

当前公开V2的`NPUModelRunner`将req_states、input_buffers、speculator与Graph manager分开，并分别提供execute_model/sample_tokens入口，便于辨认边界；它仍继承upstream GPUModelRunner并依赖forward context、global patch、cache metadata与通信状态，不能直接当独立小库。应抽取当前GLM使用的最小闭环，隔离private API与全局状态，而不是无目标地改整个多模型框架。

当前main含0.28/0.29兼容与0.30+路径注释，用户0.27是否可直接使用这一V2仍需核验。V2无预设研究优先权，结构清晰也不是更快的实测结论；是否继续由最大可信代码Gap、matched A/B及完整E2E决定。若本机V2缺少必要GLM/PD能力，按最小兼容代价补齐适配或移植所需执行组织，不盲目升级所有依赖。迁移时可暂保留现有输出/服务适配，尽早形成真实请求闭环；最终边界由性能与维护成本决定。

公开V2存在设备修正num_computed_tokens→D2H→下一步Host seq_lens消费者的链；这给出可定位的研究问题，不证明它是现场主瓶颈。重构须检查消费者、state版本和有效请求，不能简单删除event.synchronize。不同topology下Graph要求也可能不同，实际能力描述应参与scheduler决策。

## 12. 同驻留复用与必要重建

| 变化 | 可研究的复用 | 何时不能沿用旧执行资源 |
|---|---|---|
| 请求长度、结束、动态batch与准入 | 驻留模型、通信组、容量arena和合法capture | 超出capacity/backend约束时选确实可用的执行类或扩容；不保证任意shape自动可用 |
| 预置调度/Host/等待策略 | 在drain或一致的rank切换epoch内选执行变体 | 改变state协议、Graph已捕获逻辑或仍在途资源时需对应重建/恢复 |
| Graph shape、metadata/地址、MTP深度 | 复用相容的模型和已有capture，按需新增capture | capture key、消费者、buffer/workspace、通信布局或代码不相容时不能replay旧图 |
| TP/EP/PP与P/D布局 | 通用代码、模型语义、编译产物中的相容部分 | 通常需重分片模型、重建communicator/cache映射与Graph；多拓扑支持不等于在途热迁移 |
| checkpoint、quant或primitive backend | 未变化的控制代码与有效构建缓存 | 权重、cache/scale ABI、编译签名和Graph依赖改变时重新绑定或重建 |
| 服务进程重启 | 正确有版本的源码/构建/编译缓存 | live Graph依赖进程内地址，不能直接当可跨进程复用的序列化产物 |

同驻留切换、局部P/D重启与初始化复用是要核验或实现的能力，不是已经存在的现场事实。一个启动session可以运行多组有辨别力的对照和动态负载，前提是记录每组代码/策略身份，正确复位请求、cache和observer状态，所有rank使用一致执行方案。不得为了省启动时间把不同工作量或暖缓存条件混成同合同结果。
