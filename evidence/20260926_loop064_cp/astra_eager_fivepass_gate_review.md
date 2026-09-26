# Astra High：Loop064 同状态五次 Target 数值 gate 设计审查

2026-09-26。独立只读源码审查；未运行 NPU、未修改运行源码。承接 run284/astra_matched_scope_review.md。

## 本轮结论

Sol 提出的单服务 eager 顺序 off / overlap / off / immediate / off 是目前最高信息价值的最小数值实验。先不扩 all21。它能隔离相同输入/缓存上的重排与并行语义，不负责证明 FULL Graph 存储安全或性能提升。

标准路径不应悄悄重放旧 Target Graph：
- borrowed worker/model_runner_v1.py:2223–2254 在 EXTREME_RUNTIME_TARGET_GRAPH=0 时设置 aclgraph_runtime_mode=NONE、skip_compiled=True、graph_update=None。
- ascend_forward_context.py:128–130 传入对应 forward context。
- compilation/acl_graph.py:138–145 在 NONE 时直接 runnable，绕过具体 graph entry；vllm compilation/decorators.py:509–513 在 skip_compiled 时直接 forward，绕过 AOT/compiled 模型。
因此已见标准嵌套 wrapper 均受 context 控制；没有发现必须另关的嵌套 Target Graph。启动阶段或 DSpark 的图仍可存在，不能仅凭服务有图就判该 Target 非 eager。自定义/native 内部机制未穷尽，须用实际执行标识验证。

## 必须保留的最小证据

每次试次记录 rank、cohort、cycle、试次0..4、mode、scope、Target输入地址/内容指纹、NONE/skip_compiled状态、层2实际命中分支、捕获计数前后。环境变量只记录在宿主进程不够；须是执行 worker 内实际读取值。
一次性 _extreme_cp_fork_logged 不能证明后续 immediate 也命中。可用 opt-in诊断分支计数/事件证明；不把print移入正式热路径。

原 TargetPageAudit.execute_with_self_replay 连续调同一个target，只比较top2和acceptance，本质A/A/A；本轮需要在每次execute前设置mode，finally恢复原mode。若改回Graph，env切换无效，因为 ACLGraphWrapper entry key只有BatchDescriptor，不含mode/scope；不能把该eager方案直接用于FULL。

## Snapshot/restore 与输出生命周期

1. 在完成prepare+derived metadata之后、第一次Target之前建立唯一原始S0，所有五次都由S0启动。Target输入、draft、positions、counts不得跨试次推进；每次state advance/DSpark只在五次结束后执行一次。
2. RuntimeAssets.snapshot_pages 已在snapshot和restore前后synchronize；保持这些诊断屏障，避免aux旧写越过restore。它覆盖全部传入cache视图及mutable_tensors（runner中含topk_indices_buffer、mtp_hidden_buffer）。不能为了只改layer2而缩为只restore layer2：五次Target都会写其余层cache。
3. 记录实际写页覆盖严格通过、operator_missing_actual_pages=0、每次verify_restored exact。restore验证证明已捕获页被恢复，不证明“没有漏写页”；现有源码/operator slot coverage仍须同时保留。压缩state前沿/跨页尾部不能仅由flat96 slot mapping代替。
4. attn_metadata递归快照须涵盖所有受更新张量；forward context的compressor metadata memo每次应为新substep，不能让另一mode复用旧输出/stream provenance。DirectTargetHandoff每次新建set_ascend_forward_context是正面证据，运行时可记memo generation。
5. 立即clone每次full logits、hidden/aux hidden、acceptance integer和layer2 post-cache/state，然后再restore/下次forward。Target输出可能指向_mtp_hidden_buffer或Graph/内部workspace；只保留tensor引用会被下一次覆盖，形成假parity。克隆完成需在restore之前有正确stream依赖。
6. FixedGreedyAcceptance.execute 写state.accepted_tokens；greedy_accept开头fill(-1)再完整覆盖，当前Target不读该输出，因此重复acceptance可比较；仍应clone每次结果。不要把先前AcceptanceOutput引用当独立数据。
7. layer2 post-state比较覆盖main compressor state+compressedKV、indexer state/key/scale及SWA实际触及页，按正确视图/物理页比较，不仅main compressedKV。main/indexer存储alias audit应开启。其余层全量post-cache可成本过高，至少保留restore的全覆盖与输出误差分布；任何layer2输出差异需追因而非接受长度1024即PASS。
8. 五次结束后以最后off试次的Target/acceptance及对应cache继续服务是一条干净的A0轨迹。明确报告诊断服务后续产品轨迹为off，不伪装成候选overlap E2E。

## 数值门槛：使用既有标准，不发明 atol

AGENTS.md:226明确允许self-noise、margin、acceptance/count parity，不能机械要求不同run token完全一致。

- 先比较三次off(A1,A2,A3)。它们是同state自重放噪声基线；若restore/shape/地址/选层不成立，整个试验INVALID。
- full logits/hidden/aux/cache分别记录dtype、finite/nonfinite、exact比例、max_abs、相对L2或归一误差、误差分位数；logits另存argmax/top2 margin及每个分歧位置。padding/无效输出与有效语义区域按已有契约区分。
- acceptance counts和sampled IDs记录所有12请求与96候选位置的差异。若A自重放整数稳定而B出现高margin改变或state新差异，不可晋级；先检查race/遗漏恢复。若A自己就在低margin处翻转，B相同现象不能自动归因优化，也不能据3次A建立统计确定的“允许误差阈值”；标数值INCONCLUSIVE并用具体差异/margin决定最小补证。
- 不以单个全张量max_abs作为唯一阈值，也不在看到B误差后扩大容忍。三次A的经验noise envelope是诊断证据，不是硬件概率上界。
- 理想通过：结构/缓存恢复全过，A内部稳定，B/Immediate满足冻结数值门槛，整数语义一致，layer2post-state没有无法解释的额外变化。若既有门槛未给具体float阈值，报告完整误差和自噪声供Sol按已冻结Correctness裁决，不自创数字。

## Ownership 前沿的范围

本实验只重排现有工作，未删任何请求KV。无需先证明整个“非owner永不消费”理论才能执行该gate；那属于另一条删除工作候选的更强前置条件。本实验需要的是所有被五次执行触及的状态可恢复，以及同state结果/后状态等价。prefix warm-cache生命周期在未更改写集且后状态等价时没有新增省略写入风险。

现有 ownership_manifest_reaudit.json 给出 Run75 all8 ranks 的Target/mtp conservative hull不重叠，是有用历史证据，但不替代本次layer2实际指针audit；也不证明后来cohort的prefix有效区间。

## Run284 与后续

Run284仍是matched profiler局部正信号：16个按rank/cycle序号配对样本到next AllToAll皆更早，配对中位-39.875µs；不是16独立同state重复，不是正式E2E。Rotary+58.7µs说明联合资源成本必须进入DAG，未消灭全部局部收益。

五次eager gate通过后，下一步最有价值是one-layer FULL Graph数值/事件闭包验证，而非直接formal：
- 独立A/B图，显式graph identity与mode/capture日志；不能改env后重放同entry。
- capture也写cache，必须恢复；outputs立即clone。Graph pool与workspace不得在A/B切换时错误复用。
- DSA的update_graph_params在dsa_v1.py:1632为no-op，未发现该DSA路径必然有task-handle切换风险；其他全局graph workspace的具体注册仍应记录，而不虚构blocker。
- 通过才扩大all21同state筛查和matched profile；正式晋级必须未profile重复A0对照Product E2E。

本轮数值gate不产出TPS；它解决“正的局部调度信号是否建立在语义等价上”这一关键不确定性。
