本轮推进 M22 graph/capture/replay 生命周期，采用已保存 profile → 明确未知边界 → 有限源码补证，未重复 M21 的63个邻接 guard。

两类命名 metadata 的执行位置已获得明确推导。v37 全32份 TASK census 会保留所有 metadata 类型，非 native fallback 数为0，SMLA321、QLI208。v20 的同一快照完整身份分类覆盖154940个 DIRECT、132000个 GRAPH、10个 QUEUE，并逐身份断言 DIRECT 不在 GRAPH 集合；对应历史脚本 SHA 已核验。因此这529个命名 metadata TASK 均为 DIRECT、这两类在 GRAPH 集合中为0。该推导没有重新查询远端，也不证明 opaque/fused内部没有 metadata 工作，或plan bank已绑定；529个metadata PMU缺口仍保留。

定点合同说明了几项会改变优化合法性的边界：

- wrapper 按 runtime mode 与 batch descriptor 分派，首次 capture 消费当时参数，之后 replay 使用已捕获图并返回已存输出，新 args/kwargs 不会重新执行 Python callable。
- DEBUG 地址列表仅含顶层位置参数 Tensor。runner 普通路径通过 partial(self.model,**model_inputs) 调用，位置参数可为空，因此地址 assert 可以在 kwargs Tensor 地址变化时仍通过；它不是全部输入固定地址的证明。两个纯数据反例只说明检查范围，不是运行时 Tensor 实验。
- enable_enpu 来自 C 环境 ENPU_ENABLE，不能与已保存 enable_npugraph_ex 配置等同。其实际值未恢复。wrapper 和 runner 的同步/更新条件不同；updatable backend、主模型 ContextSource、draft SharedSource、ENPU update→replay 与非ENPU replay→update 都保留。不能仅凭 Python顺序推断 native 参数代际。
- generic update 只在 FULL、非capturing、非sparse、非compress 时进入。内核叫 SparseFlashMla 不足以证明 live Python flag。model forward finally 会释放在途 metadata submission，异常路径也必须保留。
- capture 的 offloader sync/join、global graph pool、弱输出/工作区、首次返回与 replay 返回的持有关系都约束 bank 生命周期。弱引用操作不被算成待消除的设备拷贝。Engram 的每轮真实路由与 capture 输入 bank 沿用既有历史/DP合同。
- SMLA metadata producer 除 GPU长度 tensor 外还有 CPU最大长度、请求数、head/layout 等标量。移入 graph 需要正确更新这些标量或证明固定捕获上界安全；只刷新 GPU tensor 内容不构成完整证明。

本轮记录3个 F31 条件细化、17条约束：producer 在图内外的放置；以完整代际协议细化全流同步；独立 bank/graph 的有限代际流水。GPU stream wait 不会阻止 CPU 改写 native descriptor，必要的 host完成或独立native参数对象必须保留。还需真实反馈输入就绪、全部rank一致、捕获 bank地址及最后读者、descriptor×bank内存/捕获成本。没有删除同步、修改实现或宣称实测收益，三项也不相加。

同时对照原始M22范围：2文件142units、1101静态导航site（graph文件55、worker1046）全部保留。它们是 source association，不等于1101个graph语义分支。旧矩阵32条历史进展完整归档，改为16个稳定语义范围；41处核心guard有逐项数据流说明。graph文件另28个registry/weak-workspace/proxy等helper site和worker邻接责任范围仍待有限核对，原8项mandatory gates仍partial/incomplete。没有把导航分配或报告成稿当成整体完成。

一次初始 reader 把 imported use_updatable_graph 当成本文件函数而 StopIteration；发生在source artifact写出前，错误已保留，按明确import修正后成功读取小函数。本轮5项登记4 pass、1 error；无远端登录、模型/torch/NPU/服务/健康操作。166前轮鉴权问题仍待恢复，full graph结果尚在166，16次phase仍未绑定。

35模型候选、7框架候选、0整模块关闭；26理论比较INCOMPLETE、3136资源行、849缺PMU、16phase-unbound保持。下一步是28个具体graph helper、实际graph/native任务/bank/所有读者/代际，以及同分母资源、完整进程transport和其余模块范围。Goal active；未实现条件优化本身不被当成分析未完成的理由。
