# 共同frame的真实覆写与双端退役判断

Goal ACTIVE；loop040为限定源码/顺序模型/成本判断，逐原语仍须逐位一致。全链路未扫全，本轮原语准入0。

## 介绍

继续同代metadata的大交付，从共同CPU事实、device qsl/length/positions到资源组装、metadata producer和compressor/QLI/SMLA读者，再追到下一代第一次覆写。不是单独替换一个where。

## 到了哪里

[15源及已读行段](common-frame-lifetime-frontier.json)，含实际CommonAttentionMetadata定义、Ascend unpadded、V1 executor与target/draft caller；359个字面字段访问只供定位，不代表动态alias或全部读者穷尽。先前本地快照缺attention/utils.py的读取错误已记录，随后从同owned容器只读取得；没有框架进口或NPU实验。

确认了三种不同寿命：

- **Python字段与scalar**：V1 submit在返回前同步调用各task.run并提交原生操作。DSpark之后重绑cad字段不会重新执行已提交闭包；延后callback或改变capture则须另绑定。replace是dataclasses浅替换，unpadded保留大量张量view；新对象并不等于新存储。
- **device输入/输出存储**：runner1435–1462先写global qsl/CPUlength，1591–1606再写positions/device lengths，发生在CP builder之前。只在builder入口等旧producer不足以保护global alias。release和nextproducer的buffer_reusable等待不保证旧未消费QLI完成在下一次主流输入写之前；输出复用还须包含aux reader末读。
- **CPU pinned源**：device stream等待只排设备队列，不阻止CPU写源buffer。必须用新私有快照保活至DMA完成，或确认完成后才回收；有界pool满时的host等待须付费。不能把“有device wait”当CPU自动快照。

[有限顺序模型](metadata-frame-order-cpu.json)覆盖12个device场景、8个host快照场景，共1033个拓扑序；给出release-only、只等首QLI、devicewait-only pinned复用的反例和所需边。它不是运行时race观测、服务bug证明、硬件同步或逐位验收。

正常V1可在原metadata stream最后记录producer_done，在下一代第一次持久输入写前安排依赖，同时保留consumer/aux完成；所有原COMPRESSOR/ATTENTION/INDEXER frontier与固定输出地址保留。若把common device计算异步挪走，原eager batchcopy、C2长度及slot计算也要随依赖迁移，否则会先读旧值。submit设置in-flight后task.run异常、进入模型finally之前异常的owner不能靠该finally保证退役；要建立terminal完成或隔离，不能清host状态后直接复用。

## 结果

**待确认，值得开发判断**：原首producer计算一次的同代frame、各资源组装与完整发布/退役协议。CPU快照、device acquire位置及producer/consumer端已经具体化；仍缺native事件代次/FULL捕获与错误退役实现、全部alias等价、逐原语位对拍和真实净价。较大metadata空间保留，旧CPU-plan/25H2D/已融合slot不重复算。

**已确认可实现**：本轮0。

**已停止**：release-only输入复用、devicewait-only pinned复用，以及只挪common producer却保留eager依赖读者的原样方案。不停止带完整owner/fence的新协议。

对贯穿Attention的head方案补了一个价格分支：[全局query packet](global-query-packet-price.json)。cp_global已有原global qsl、权威device长度/positions与C2长度，缺的是head8 query packet，完整压缩causal语义仍待核。

两个代表rank各12次迭代、每次3个目标SMLA packet加2个QLI metadata；旧三目标packet参考合计615.940/619.526us每迭代，不按40层重复计。原最后metadata至首QA最小余量2.808/2.608ms，至首SMLA最小2.902/2.703ms；只是旧时序余量，不保证新工作被隐藏。

若沿用Qb耗时1/4、旧AG参考、SMLA不退化，并另付每交付5us pack/sync：新三packet与旧成本相同且真正替换时，旧条件净1.228%/1.299%窗口保持；若保留旧packet后新增、且新成本全暴露，则降为0.442%/0.500%。若同时SWA减半，后一条件为1.859%/1.923%。共144个敏感性场景含两种packet模式、代价/曝光和pack成本；不是测得新kernel或请求收益。原生packet模式/graph fallback必须先讲清，不能给额外metadata默认零费或把旧余量当已实现收益。

这些明确缺口不阻断其余路径。下一段继续较大的O投影→SP交付→mHC/FFN读者，核真正OA/OB/HC原生几何与舍入；不为已经停止的小逆RoPE打包或旧M3 OA配方重复补测。

本轮远端只读源码/旧profile，本机stdlib顺序/成本分析；NPU、模型加载、请求、新profile、显式远端写入、服务动作均0。
