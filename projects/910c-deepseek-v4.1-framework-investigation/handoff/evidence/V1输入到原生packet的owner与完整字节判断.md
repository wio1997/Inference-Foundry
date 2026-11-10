# V1 输入到原生 packet：owner、真实读者与完整字节

介绍：沿冻结 V1 worker 的实际调用继续上段 107 处未闭合位。把 CPU/设备输入、batch 值、物理 cache 映射、异步 packet producer 与 model graph 的读者分别追清，再判断较大共同 frame 能否少算、少搬。每个替换原语逐位一致，不能以浮点容差或忽略 packet padding 放行。

到了哪里：统一入口核对 171 hostname 成功，同 owned 容器/image；16 份 Python 依赖为 11 旧 SHA 匹配、5 新 body，11 份 native 依赖为 8 匹配、3 新 body。另复用同 SHA 的 `SpeculativeConfig.use_ngram_gpu` 两行，并未称为本轮重新读取。原 156 位保留原条件、SHA、位置、历史交集和空的运行观测；本段新增 21 项受限源码约束，累计 70，仍有 86 位未闭合。位数不是算子数，也不是完整递归调用图。27 份身份并非 27 份全部 body 已审；限定行段与 identity-only 项分开公开。

结果：新增绑定来自实际 caller 的 capture/dummy 默认参数、冻结 PP1/async+K5、V1 Engram 字典与事件传播。执行原三个 Engram 准备方法的 16 个显式 stub 控制：normal/capture、FULL/NONE、overlap 开关、有无 Engram。所有返回都提供非 None 的 lookup 字典，`graph_events=True` 只随同一 FULL context 交付；SP 入口清掉 mask event 后，层内不会再次等待该 mask。V1 文本 kwargs 不交付 V2 的 `engram_query_start_loc`。这些是控制流/字典证明，未实际 hash、查表、运行事件或加载 Transformers 对象。

|交付与读者|本段结果及融合边界|
|---|---|
|全局 QSL 初次 H2D|worker1435–1437 发布 prefix；1622–1626 的 slot mapping、1645 的 logits index 在 FIA padding1105 的第二次发布之前已读取。停止“直接删初次复制”。较大改写发布顺序须重新追完整准备 DAG，不能从重复函数名推死值。|
|全局 request/device 长度|async 修正、CPU 镜像和权威 device `seq_lens` 分开；metadata3611 的公开 CPU 字段置 None，内部 `_seq_lens_cpu` 仍保留原镜像。不能用 CPU 上界代替设备长度，或把全局/局部镜像合并成一个字段。|
|CP 局部 QSL|每 builder 私有 `_cp_query_start_loc`；base 的输出坐标可能早已 alias 第一个 batch QSL。但 SMLA1180、QLI1260 的闭包直接捕获 incoming `common.query_start_loc`，并不会自动改读 canonical 字段。|
|packet 首 owner|`smla:c0/c1/c2` 与 `qli:c1/c2` 各保留首个发布者的 packet、task/group ID。它们与物理组顺序有关，不能把第一个 batch owner 当成所有 packet owner。|
|C2 物理地址映射|1013 的 `valid_end` 也读 incoming QSL；共享 key 只在某个物理 cache 组/ratio/storage-block 内有效。新的物理组可有 QSL 读者，即使该 ratio 的 packet key 已命中。全局 compressor ring 另读全局 common，不在局部 CP 控制中删除。|
|V1 producer 与 graph|producer graph-external，model graph 消费 canonical 坐标与原持久 packet 地址。显式重绑 incoming QSL 可以避免一部分复制，但会延长 canonical 输入末读；不能只保留原来某一个 builder 的事件寿命。|
|跨迭代退役|GPU parent 的 input-prep event 是 CPU 传输复用约束；metadata `release` 并不是所有未消费 QLI 的 join。需同时保留 CPU 不可变租约、producer_done、consumer_done，以及最早输入写之前的 acquire；不能把 producer stream 中较晚的 reuse wait 当成已保护主流上的早写。|
|embedding 到模型入口|继承的 `_preprocess` 在 supports-MM 分支中，即使纯文本也先 embed，再 copy 到持久 `inputs_embeds`；模型 forward 因而可能不再内部 embed。raw IDs 仍供 Engram/vision router。配置选择、实际 embedding 实现及更大首层交付留作下一段，不从“纯文本”名称跳过。|

原 base `build/_publish_task/_build_batch_metadata` 另做 32 个符号控制：12 个合成 builder、8 个物理组、4 种顺序、空/非空 query、FULL/NONE，对照 incoming QSL 原私有来源和显式 canonical 重绑。保留符号 non-QSL 参数、五类 packet/task owner、任务数及 canonical 输出 alias。三个 poison 反例分别在新 SMLA key、新 QLI key、C2 物理组映射读到了未写 incoming QSL。空 query 仍保留五个 task，SMLA 清包，两个 QLI 仍执行。**这不是实际 26 builder 人口/顺序 dump，也不证明值、原生 bytes、指针、事件或请求收益。**

针对会影响 canonical 重绑取舍的 packet 地址问题，继续追了真实 adapter→L2/L0→AICPU serializer。两个结构体都是整数数组；原 `GenMetadata` 从 split 结果写调度字段，没有输入地址字段。原函数体与原 packet header 的 12 个 host C++ 哨兵控制证明：SMLA struct900 个32位字、QLI struct864 个32位字被重写；wrapper 各分配1024字，余下124/160字，即496/640B，不在这些函数的写范围内。测试只替换 Tensor/SplitResult 外壳，未改原 serializer body。最初 harness 缺少原 cpp 的 `using namespace optiling` 而编译失败，修正外壳后通过，失败日志保留。

不能由此断言 CANN runtime 一定留下未初始化尾部，也没有声称服务发生精度问题；较外层 runtime/loaded binary/分配器和完整 packet 字节尚未核实。停止“结构体字段相同就已通过全部4096B逐位验收”的说法。保留原 packet，不能默许 mask 掉尾部；完整 primitive 字节门槛仍待确认。SMLA L2 首次猜测路径漏了 `op_host`，目录 tar 清单定位后重新读到正确源，不是环境漂移或 live 任务等待。

先用已有匹配数据定价：保存 serve SHA 与当前冻结 SHA c297… 一致；16 rank/11 周期同窗中 CP/base/local-helper 的互斥 **self-Free** 合计311.050016ms/卡，独立窗1025.502734ms，占30.3315%。已单列的90.346922ms marker 边界不计入这三个函数价。这个 gross 含必要首 producer、物理映射和原生 packet，不是全部可删，更不能与旧 helper/device-only/CPU 微基准重复相加。

删除比例与关键路径转移未知，默认均0。仅作条件价：若整段实删10%、诊断到轻量请求转移50%、新增退役/保留等成本0.1ms/frame，条件净约1.313864ms/frame；用同镜像配平 C8 的229/231个实际 metadata frame 和15.967999s轻量参考，整批条件省302.182ms、1.892425%。这不是实测融合收益，也不迁移到原8条不同 prompt C8 的 TTFT 或六普通负载。较大 parent 仍有深入价值；旧 device-only2.098217%只是其子范围，不是上界。不为小 QSL copy 单独补采第二套 profile。

待确认、值得深入：显式 canonical incoming QSL 加上原 CPU/helper 与设备长度 frame，共同保留原物理映射/packet owner；先闭合输入 epoch/租约、全部读者、双端退役/最早写、完整 bytes，再付费定价。实际组顺序、所有资源与 graph 指针集合、空/尾/错误、原始 C8/普通请求收益仍是具体缺口。已确认可实现：仅此前受限 CPU 原型1按原资格保留，本段新增原语/框架准入0。已停止：删除已有中间消费者的首全局 QSL H2D；无重绑而删所有 private QSL；只看 packet struct 就声称全部原语位通过。这些停止不扩大到更大共同交付或其它融合。

下一段 loop047 Run133903Z 仍 PLANNED：沿实际继承的 V1 `_preprocess`、supports-MM/配置 convertor、embedding 真正实现追到持久输入和首 HC；同时追 global QSL/physical block-slot 经 Engram、DSpark per-group 视图到全部末读/最早覆写。继续86位及递归 helper。完整搜索尚未完成。

证据：`v1-input-dependencies.json`、`v1-input-source-review.json`、`v1-qsl-owner-controls.json`、`v1-packet-dependencies.json`、`v1-packet-write-extent.json`、`v1-input-parent-price.json`、`v1-input-source-excerpts.json`。本段无 Torch/Transformers/NPU/模型导入、新请求、profile、服务动作或远端写入；容器 running 不是 health/loaded-object 证明。
