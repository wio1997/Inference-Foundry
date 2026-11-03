# Profile 分析进展与待证项

本轮遵循用户要求：充分使用现有 profile；相关 profile 无法定案时，才就影响判断的具体问题定点查代码。本轮没有新增目标源码审阅、远程连接或设备工作。

## 当前覆盖状态

32 个既有融合/拆分重组候选均有 profile 入口、待证问题和下一步。26 个暂定模块均有候选或框架/状态证据入口。登记完整不等于分析完成；候选全集与分支闭包仍开放。

|候选状态|数量|可据此得出的结论|
|---|---:|---|
|观察到既有复合边界|3|F01/F04/F13不能重复计作尚未发生的复合调用；内部及延伸边界仍开放|
|已复核选定局部切片|4|F07/F08/F12/F20可落下述局部结论，不是完整候选闭合|
|已复核选定量化汇总|2|F05/F09有重复相容签名，输入快照及实际参数等价未证|
|已绑定主图段间记录|2|F30/F31有具名工作记录，全部角色、消费者和分支未证|
|只有family入口，scope/stack待查|21|继续查现有profile，尚不以通读源码替代关联分析|

## 可以落地的局部判断

**F07：保留已存在的交叠。** 1920 个选定 target 窗口中，Q 投影相容记录与另一流 cache scatter 相容记录全部相交。逐窗的scatter尾段减RoPE时长为正，但这只是区间长度充足。九个局部窗口的27条事件记录没有 `id/id2`，其connection相同；不能据相邻时间配对event或认定某段时间可释放。排除把两段task时长直接相加作为融合前耗时的做法，不排除合法重组候选。

**F08：同签名不是等工作量证明。** 1280 个选定norm/RoPE/scatter三任务scope覆盖物理行数4760、5144、8180。T4760的320条scatter中，279条较长记录与选中的量化矩阵记录有交集，41条较短记录没有同样的选定交集。这个相关性不证明争用、真实slot数量相同或可消除时间；继续保留实际有效槽、tensor版本和持久消费者缺口。

**F12：调用位置已缩小。** 两条CONNECTION_IDS关联分别定位到Indexer和prepare的提交栈。它们足以定位下一步的接口边界，不证明两者实际tensor边，不是Dequeue线程栈，也不是请求关键路径。

**F20：准备链值得调查，但不能把pad时长等同收益。** 320组pad/gate与MC2按rank/layer/scope唯一匹配。320条pad及where/gate栈的完整覆盖是已存review中的报告；本轮本地输入没有其逐条栈结果，不能将这次键连接称为独立重建了320条调用栈。已有整数映射的局部合同可以复用；历史ID内容、全部消费者和binary绑定不由重复读源码补成实测。

**F05/F09：双量化的适用范围明确了一层。** 16rank×3轮×40层的已存筛选汇总中，八个指定层的384窗都是D/Q/D/I，其余1536窗是D/Q。相同shape/dtype及先后次序支持复用候选，不能证明两次输入值、版本或解析后的量化函数相同。后续只复用既有局部量化合同和已存host-flow关联，保留真实输入及参数缺口。

## 官方批次与框架证据的新增关联

官方两窗32份样本的 `kernel_details.csv` SHA-256均在图执行汇总中唯一匹配，并与记录的window一致，形成16rank×2window的明确文件级绑定。它不证明scheduler/request/tensor身份，也不授权跨rank直接相减时钟。

第二窗每rank有12个40-MoE形状相容主图段候选，其间11段，共176段。每段均出现具名的 `rejection_greedy_sample_triton`、输入扩展、DSpark索引构造和SparseFlashMlaMetadata记录。greedy kernel的导出区间中位2.740微秒；这是局部task尺度，不是F30完整代价或可节省时间，也不排除未覆盖处的概率分支。

这些段间区间中位跨度35.819692毫秒、全部已导出记录并集中位10.926192毫秒；两项中位数不能相减推导空闲或提速。导出覆盖不全、内部等待、host工作和时钟契约都影响解释。

rank0/8的worker prepare各12个host scope已通过同一source_dir关联到第二窗口，且记录的root exclusive与descendant union恰好覆盖scope。exclusive表示未被嵌套记录覆盖的时间，包含Python/native、等待与profiler开销，不是CPU活跃或可删除时间。另一轮worker采样的两个py-spy退出码均为-15，没有栈文件；该轮proc计数独立保留，不能当函数栈证据。

## 后续最小查询

先检查F06/F10/F16/F32相关的既有文件：`f16-ep3-projection-provenance.json`、`f32-ep3-projection-stacks.json`、`f32-ep0-7-projection-rows.json`，沿已存host-flow和通信group核对投影/行域。其他待查项保留在v2候选台账中。

本轮结论以 `candidate-profile-ledger-v2.json`、`module-profile-ledger-v2.json`、`cohort-and-gap-bindings.json` 为准，v1文件保留。Loop004仍在进行，0个整模块关闭，整体目标未完成。
