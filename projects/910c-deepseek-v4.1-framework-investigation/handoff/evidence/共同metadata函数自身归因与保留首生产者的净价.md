# 共同metadata函数自身归因与保留首生产者的净价

介绍：把原同代公共 frame/metadata 的源码、读者和双完成栅栏判断，接到刚完成的函数归因；按最终交付判断重复计算与组装，不为单个 where/cast 单开小方案。每个替换原语逐位一致。

到了哪里：5 当前磁盘模块 SHA 与新诊断 adapter 全部相符；4 个限定入口 body，executor 此次只核身份并复用原 lifetime review。新样本 16 rank 各 11 闭合循环，平均窗口 1025.503ms。只读取已有诊断，不重采。它与旧 839ms、940/930ms 窗及原八条不同 prompt C8 都分列。

结果：`CPMetadataBuilder.build` 自身与输入准备 Free 的交集 138.642ms/卡，local-token helper 自身 53.917ms，合计 192.559ms；base builder 自身另有 118.491ms。原 analyzer 以最内层函数去重，并把 90.347ms 新标记边界单列扣除，三者的自身时间可以相加，inclusive 不能相加，标记也不能再扣一次。自身仍含库调用/等待和诊断扰动，不等同纯 Python 或可删时间。

CP 先调用独立 global builder，再调用 local base builder。base 的 batch/slot/C2/rope/native task 已有共享；CP 包装层在 local base cache 命中前仍重复 helper、QSL/pin/copy、suffix 和设备权威长度处理。runner 每轮新建 batch 字典，每 KV group 有物理资源字典，不能整对象跨组复用。GDN override、DSpark per-group table/slot 和 spec common 选择分支保留。

诊断内 CP/helper/base 调用数约为每循环 27/27/54，但不能据此认定有 27 个完全相同参数的独立 group。源代码可证明双 base owner，窗口计数仍受边缘裁切和 2 个 FT/body 未匹配范围影响；原差值保留，不强行修成两倍。

全部 144 个付成本情景均保留整段 base builder 的 118.491ms，不重复认领现有缓存。只为 CP 自身和 helper 指定“可共享部分”，保留首 producer，再付 host snapshot/pin/H2D、每资源组装、发布、栅栏和争用成本。首 producer 采用同函数平均调用成本是明确 proxy，并非实际首条时长。

若仅 helper 的上述自身区间可共享，按平均首调用保留、只有一半 Free 交集能变成关键路径净省，每循环新增 0.5ms，11 轮条件净省平均 20.460ms，即本诊断窗口 1.995%；再假设 CP 自身 25% 可共享，则为 37.148ms、3.622%。16 rank 各自成本与全部负收益情景保留。这不是实测替换、完整请求预测或生产加速。不能与其它窗口的 DFC/Attention 百分比相加。

待确认，值得继续：同代兼容 key 与实际重复 caller；仅共享原首 producer 的不可变 CPU facts，并保留各 builder 固定地址 device QSL、物理 slot/table、C2 ring 和消费者闭包 attrs；按原设备长度计算，不用 CPU 上界替代；沿原 acquire-before-mainwrite、CPU 源退役、producer_done 与 consumer_done 继续证明异常/图事件和每原语位。实际共享比例、首调用成本、新增开销及原 C8/六普通的关键路径曝光缺口具名保留。

已停止：按 shape 跨代缓存、整份 metadata 跨物理组复用、把 base 已共享工作或标记耗时重新计为收益。已确认可实现：本轮新原语 0。全链路未扫全，当前只是入口/价格证据，不能把具名缺口当扫描完成。

下一条待扫：共同 frame 的最小字段 key、两个 base owner 与全部消费闭包；共享 CPU helper 的原返回值在哪些调用点会变异，哪些 device 输出必须每资源发布。新诊断只用来排序和付成本判断，不操作服务。证据：common-frame-measured-{entry,entry-review,source-excerpts,price}.json；原 common-frame-lifetime-frontier.json 的拒绝配方和 lifetime 门槛继续有效。
