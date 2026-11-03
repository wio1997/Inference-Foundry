本轮用已有 profile 和直接源码证据，对 S13 剩余36个具体静态 guard 逐项核对。累计63个 guard 的本地数据流、分支域和邻接读写协议均有记录；这一局部范围不再作为未分析项。未把它扩大成全部运行时字段、实际 alias、capture bank 或整模块已证明。

复用 v16/v27 的 stream/C1/C2/cache 证据、v35 的实际 CP 输入输出及空 query 调用栈、v40 的 QLI reader。新补12个小方法、3行多流尾部和有限 bank/topology 字段，未重新读整份 DSA、整个模型或 SDK。可核实的关键合同包括：global metadata 只解析本 layer 的4个 cache planes；CP 复用索引按本地 qr 行；index query 与 weights 输入不同，量化需要5项条件且 consumer 等实际事件；C2 元数据等待、source RoPE fallback、旧 latent 读先于 long 原地 RoPE；packed mixed/prefill cache write 在 main join 后，其余选中路径在 aux 写后 join。

还排除了几项容易虚算的“优化”：

- select_projected 已给 prepare_indexer_indices 传 indices_output、lengths_output；直接发布是已有行为，不能当成尚未实现的收益。
- 普通同配置初始化将 shared TopK bank 分配为 INT32 [max_tokens,config.index_topk]，topology 使用同一 index_topk。若没有后续替换，同宽前导行切片已连续，pad 不走，unsqueeze 是 view，contiguous/int 可以保持 storage。不能把源码三步当成必发的三个 kernel；通用 K>topk 分支也不截断。
- shared state reset 本身返回 None，不清零。跨层复用依赖 source layer 当轮覆盖有效行，不支持跨迭代仅按地址缓存旧值。
- 空 cache 路径处理 indices=-1、topk lengths=0 和 candidate-source lengths=0；长度与候选 block ID 都是独立输出，不能只保留 token IDs。

保留2个条件细化、14项约束：F06 的部分行 quant reuse，仅适用于相同行量化函数/scale/参数/通信和非 trivial quant 路径，且必须保留局部 Q 就绪与当前 overlap；F12 的 stable-bank copy 省略，须证明实际 producer 返回同一 range、已完整写入且没有 version/debug/capture observer。当前正例 Q_A/KV 是 unquantized F.linear，没有计入任何实际量化收益；精确 CVLinear.quantize 与 prepare_indexer_indices helper 不在本地冻结资料，未假装已读。5个行函数例与一个 batch-scale 反例、36个格式例只证明数学/布局条件，不证明 native 性能或精度。

一次补查 CVLinear.quantize 由于本地缺文件失败，原错误保留；随后只核对 archive member 也不存在，没有重试堡垒机。登记4项：定点 source、条件/guard 分析和正式审计3 pass，缺冻结文件1 error。本轮没有远端作业、模型/torch/NPU/服务/健康操作。

整体仍未完成。M21 保持18个稳定语义范围和268个导航 site，原8项 mandatory gate 仍 partial/incomplete；真实剩余为精确 dispatch/PCP helper、实际 main/nested graph/QLI body 与 parameter banks、所有读者和跨代复用、同分母资源上界、完整 EngineCore/transport，以及其余模块全范围审计。本地 saved-graph receipt 只有当前80图摘要，full result 仍在166；旧169的11图样本不替代当前 bank 证据。前轮166 SSH鉴权问题尚未恢复，保留缺口但继续可用本地分析。

35模型候选、7框架候选、0整模块关闭，26理论比较 INCOMPLETE、3136资源行、849缺PMU、16 phase-unbound 图均保持。Goal active；条件优化未实现本身不被当成未分析缺口。
