本轮补入一条此前未登记的跨模块关系：F24 Engram设备行缓存与F34 C2状态重组竞争同一个NPU设备的存储预算，并扩展已有HG85联合预算。没有把空白关系一概认定为独立，也没有重新认领v62行缓存、v79 ring压缩或v85预算公式本身。

关键限制先明确：在v79的当前unpacked/no-override缓存池模型中，C2偶数行逻辑视图压缩不改变common-ID的540928B预算除数，也不改变allocated block数。三个2.268GB ring视图不是三份独立分配，不能把它们相加当可释放显存。新的私有状态库或paired-owner协议可能释放common-ID预约，但仍要计算新bank、metadata、迁移和其它KV容量；global ID也不是一块可直接交给普通Tensor allocator的空闲显存。

行缓存按精确保留encoded codes及配套scale分析。v62给出的每行payload为BF16 512B、INT8 group32 288B、MXFP8 group32 264B。仅在明确选定“32B对齐payload+64B slot metadata”的示例布局中，slot为576/352/352B；64B不是已实现ABI或必要最小值，真实index/状态/pinning/epoch/event/allocator等开销还须计入D0。键必须区分table/layer/head、完整rowID、table/scale generation、格式/布局及owner配置；C2 request/page owner ID与embedding row ID不是同一种语义。

令B=N*C+r、0≤r<C，L为真正释放的旧common-ID预约，D为私有C2、K个row slots、J个WKV memo项及其它新增开销的合计。一个共同预算下，净空闲ID变化为 L+floor((B+DeltaB-D)/C)-floor(B/C)。本轮示例DeltaB均为0；v85的335MB假定raw credit属于F26尾部快照方案，须有它自己的读者/退役证明，不能归给ring视图压缩。若要求不减少原空闲ID，选定布局须满足 K*q+J*w+D_ring+D0≤L*C+r。K、J不能分别领取同一份最大容量。

144组预算参考和36组容量边界给出一个具体反例：S=U=4、采用新私有even-ring库时，三个层的ring需786432B；4096个INT8示例slot再需1441792B，合计2228224B。在D0暂设0、未知同一个r的条件下，净空闲ID变化为-1或0，不能承诺增加KV容量。示例D0不代表真实metadata免费。若再放256项既有示例WKV结果+ID-key memo，总额变为15360000B，净空闲ID变化为-25或-24。它可以是明确付出KV容量的取舍，不是同一容量上的多个免费收益。预算只属于同一设备/allocator；其它芯片显存不能相加，要求共同block数的参与组还应按rank瓶颈约束。

容量也不等于命中率。两条均为6次访问、3个不同key、容量2的冷LRU示例，按交错或连续排列会分别miss6次和3次。对选定full-row读取路径，host逻辑payload是p*M；写入缓存增加p*I设备写，miss直接转发或填充后再读取又有不同设备读代价。它们不是物理UVA交易/HBM字节；硬件cache、合并访问、实际owned比例及格式仍未知。slot容量模型不能给出数值加速比或硬件带宽上限。

与F25 WKV memo的组合必须先做全消费者需求并集：只有memo命中且其它读者与TP/ready协议都允许时，那份lookup才能消失；行缓存随后只针对仍需求的row accesses计价。同一份已被memo删除的host读取不能再算row-cache节省。v118的false gate通常仍需求value特征，v114的hash/history/DEAD/commit rollback和全局bank/collective参与也不能自动删除。

跨迭代复用还需要明确退役：代码与scale若在逐出期间来自不同generation，2×3=6和4×5=20的两种合法内容可能被混成10或12；必须先完成内容复制再发布VALID，并保留读者直到dequant/扇出/异步消费者结束。C2旧reader结束也不够，queued writer、初始化/zero、物理padding store、capture地址及request-ID重用都要退出旧存储；稳态容量不能提前用于尚未退役前的lookup。迁移期间old+new bank共存和workspace须纳入同一峰值预算。

前置603项核对、1800个独立小规模联合容量集合通过。新增REL121-01，HG85加入F24，中央条件关系134→135、待核对461→460，联合组仍26。356条语义记录/26模块、9类/1352次原生关键字段、849缺PMU、16未绑定phase图及3136含糊归属均不变，整模块关闭0。没有新设备工作或实现；v120 Pad授权已完成消耗。实际预算/命中率、最低MFU及整体理论数值上限仍未建立，整体目标继续。
