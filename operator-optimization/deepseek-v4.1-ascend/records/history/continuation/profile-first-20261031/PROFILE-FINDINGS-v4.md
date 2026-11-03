# Compressor 与长间隔：Profile 分析 v4

本轮遍历全部1280个已存target scope，复用局部状态/精度合同，并在166迁移归档中只读核对旧EP3原始数据库。没有新增目标源码正文审阅、推理或设备采集。

## F17/F18/F19 的 profile 范围

96个C2-source scope（层2/8/14）均有两条相容FP32投影、一个pool、一个cache-update、一个post-pool norm和一个index-key投影。32个C1-source scope（层20）均有两条512输出投影和一个key投影，没有本轮筛选的pool/update记录；这与两种配置路径相容，不能把C2图套给C1，也不把导出缺失解释为运行成本为零。

双FP32投影的导出间隔中位21ns，最大2.37008ms；pool→update间隔20–41ns。已存旧decode样本另行保留，未与本批重profile时长拼接。

复用局部合同后，可以明确这些候选边界：

- F17：拼权重后生成连续`[T,1024]`，再将两个列切片直接交给原连续`[T,512]` helper，在T>1时行stride不满足接口。该直接变体在冻结helper域中不适用；两份连续输出、显式stride或拆包仍是不同的条件候选。保留独立FP32投影舍入及ring对这两份值的消费。
- F18：pool按通道的score加权，后续RMS归约512个通道；不能仅从各输入token的平方和推出pooled RMS。pool的BF16边界、零/未完成行及ring状态都保留。ring既有旧值读后再写约束，也有尾部新值写写版本约束；已存反例为条件构造，不能据此断言当前观测发生冲突或全部kernel必须整阶段串行。
- F19：index key与scale在QLI前发布一致版本，long-KV在其attention读者前完成。Indexer投影先读未作long-RoPE修改的latent；跨流提前修改需要真正的读完成或不修改原值的表示，Python返回/已下发不足以证明。两份持久cache及后续层消费者不能消掉。
- 实际完成的压缩行、物理压缩slot、RoPE source position是不同坐标。本轮没有测到实际slot/validity；不能由ratio=2或物理M直接声称一半后处理可删除。compact变体继续保留有效映射、零行可观察行为、重复slot、scale、生命周期和graph地址条件。

## F32 的 C2-source 联合接口账

若前移投影并替换hidden gather，C2源层还要保留SWA的BF16投影，以及compressor的两份FP32投影及其ring/状态消费者。按每行5120维BF16 hidden计算，原逻辑接口为10240字节；SWA的512维BF16加两份512维FP32共5120字节，条件特征payload比为2。1023行的每rank示例分别为10,475,520和5,237,760字节。

这只是在权重、每行数值、行所有权、padding、metadata、layout和通信接口均满足条件时的接口账。它不是实际链路/HBM字节或2倍提速；多个collective/打包开销、changed-M舍入和资源交叠仍未证明。C1源层使用自己的图，不套这个账，也不与F06/F17/F19节省相加。

## 原始DB核对了长间隔，但尚未定案原因

历史DB已在迁移归档的完整agent-kit/runs层级找到，大小586,985,472字节，SHA-256与原证据一致：`084b8718888e33271435ec79d7573c95ca1615f3e7484c807ea24590fad234c4`。连接使用只读immutable/query-only，查询前后文件stat未变。每个端点按device/stream/task ID/起止时间精确匹配，再查询其间全部TASK记录。

|选定区间|设备导出间隔（µs）|同一host线程两API记录间隔（µs）|TASK记录并集（µs）|WAIT类记录数|
|---|---:|---:|---:|---:|
|F22 EP3/L4 长例|1765.200|1020.460|1.080|0|
|F22 EP3/L38 短例|0.860|2382.910|0.860|5，均在其他流|
|F17 EP3/L14 长例|2370.080|378.170|0|0|

host间隔与device间隔各自在自己的记录时钟内计算，没有跨时钟相减；三例来自不同层/digest，不是等工作量对照。长例没有对应WAIT类TASK记录，短例反而覆盖其他流的WAIT/NOTIFY记录。因此不能从WAIT标签或host API间隔直接解释这些长间隔。未记录部分也不能当物理空闲、通信不存在或可删除时间：导出完整性、其他进程/未导出工作、异步队列、调度与时钟契约仍开放。

完整查询留在服务器：`.../whole-scope-continuation-20261030/profile-readonly-20261031/ep3-gap-task-evidence.json`；本地回执为`archived-ep3-gap-evidence166.json`，本轮对原DB的核对不等于新设备测量。

## 当前状态

v4台账纳入F17/F18/F19及F22长间隔证据，仍有13项需要在本轮继续纳入细粒度profile/栈材料。已有研究先复用，不把这些数量当完成比例。全部候选的完整可行性、分支/消费者/跨迭代及资源闭包仍开放，0个整模块关闭，整体目标未完成。

下一步核对F11 key/scale缓存链和F23 routed/shared汇合，分别连接本轮的latent fork与共享专家证据，再继续HC、Engram、DSpark及其它未覆盖范围。
