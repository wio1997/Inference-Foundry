# Engram 成对 UVA 寿命与整段 lookup 价值判断

介绍：从 checkpoint 的 INT8 code/E8M0 scale，沿实际成对加载、共享主机注册 owner、device pointer bank、lookup/dequant、TP 交付与末读/close，判断紧凑表示及可共享计算的合法范围。每个替换原语要求逐位一致，保留原 FP32 scale、FP32 乘法和 BF16 输出；此前 CPU 位证明不升级为 NPU 准入。

到了哪里：12 份当前实际 source/header 身份，11 份有非空审阅 ranges，见[审阅表](engram-storage-source-review.json)。当前两个 checkpoint header 重读，权重均为 I8、scale 均为 `[rows,8]` 的 F8_E8M0；只读 header，tensor payload 读取 0。当前两个源码根下 3236 个 Python 文件的字面 caller 搜索，只找到 `close_host_offload` 的定义，没有直接调用；这不排除动态、native 或用户调用。[原目录与守卫](engram-storage-catalog.json)保留搜索范围。

结果：**停止同一旧 C8、1% 固定延迟预算下的 standalone compact scale / group broadcast / lookup-only 开发**。在有 7248 次原生 lookup、906 组 TP8 的同负载历史绑定中，即使整段 lookup 全部免费，固定原顺序端点模型的 TTFT 最有利也只有 **0.118045%**、整批只有 **0.046259%**。因此不再为这个小范围补 NPU 位验收或性能实验。较大的 input/owner/WKV/iteration 联合仍单列待确认；共享主机 payload 差也单列，不能兑换成暖请求或 NPU KV 容量收益。新增已确认可实现原语 0，全链路未扫全。

## 实际成对发布路径

模型在表分配前 preflight checkpoint，并按每张表的 header 选择 storage dtype。当前 910C 是 STANDARD adaptor，I8 code 分配 INT8，原 scale 分配 FP32。A5 的 code-type、E8M0 scale、registration flag 是另一条受守卫路径，不能拿来直接认领本机资格。

`bind_checkpoint` 把 weight 与 scale 参数的 loader 都指向同一回调，但回调**只在 `param is self.weight` 时加载**。加载器从 weight 选中的 index 同时解析 code 与 scale，按 65536 行 chunk 写入自己的 vocab slice；scale 回调不写，防止 iterator 顺序覆盖已配对的 scale。模型还把 `.engram.embed.scale` 名称映射为 `.weight_scale_inv`。DefaultModelLoader 的 `model.load_weights` 是本轮审阅到的上游委托参考，未把该源方法当作当前 live loader class 的独立证明。

当前 I8 分支把 checkpoint scale 数值 copy 到 FP32，不断言它一定是 E8M0；通用 BF16、MXFP8→INT8 重量化、其它浮点重量化和 dummy 路径仍是合法分支。紧凑候选必须在分配前独立检查 source scale dtype/shape：只在相配的 I8+E8M0 条件下保留原 code，并以 `view(uint8)` 复制 scale 位；其它分支保留原路径。仅改 `scale_dtype=uint8` 不改 I8 loader 会数值转换：E8M0 编码127代表1.0，转数值 UINT8 得1，不能替代原 byte127。

lookup 也需要独立的 scale 选择。当前 `MXFP8` flag 同时选择 code 指针的 float8 类型和 UINT8 scale 解码；INT8 code + E8M0 scale 不能直接把它置 true。已有 MXFP8 decoder 在编码255产生 `0x7fc00000`，原加载 FP32 是 `0x7f800001`。此前修正255后的 256 个 scale 编码和65536组 signed-INT8×scale CPU 位证明保留，仍不证明 NPU 的次正规数、NaN、mul/narrow 或全部原语输出。

## 注册 owner、chunk 与末读

共享模式每个 TP shard 的 DP peers 映射同一物理 host segment，但各 rank 有自己的 `aclrtHostRegisterV2`、device address 和 INT64 pointer bank。初始化将 errors 做 CPU all_gather 和 barrier 后，leader 才 unlink 名称；unlink 不等于其它已附着映射被释放。加载只有共享组 leader 写表，最终 CPU all_gather 传播错误并同步加载完成。

当前 A3 的 `0x10000002` 是 MAPPED|PINNED，与安装 CANN 9.2.0 的 SDK 定义和 uint64 size/uint32 flags ABI 一致；A5 的 `0x2` 只在其 adaptor 中选择。SDK 声明不是 CANN 实现、SIO cache 一致性或新类型的设备行为证明。

`CHUNK_ROWS=2^22`：D256 INT8 code 每 chunk 1GiB，8 个 FP32 scale 是128MiB，8个 E8M0 byte 是32MiB。紧凑表示不减少 pointer bank 的元素数，却改变 row_bytes 和 chunk base offset。kernel 已把 ID widen 为 INT64、先算 chunk/local，再读 per-chunk pointer。新内存、pointer bank、kernel scale-type 特化与 capture 必须同代创建；不能把旧 FP32 bank 的地址或已录制 graph 直接重解释为 UINT8。

正常 Host/Shared close 在 ptr bank 位于 NPU 时先 synchronize，再 unregister；正常注销失败会保留 owner 供重试。`close_host_offload` 逐个 buffer close 成功后才替换对应参数 alias。**这不证明所有失败路径原子**：Host constructor 的异常 cleanup 会无条件尝试 unregister/free；两份 code/scale 的分配也不是一个自动回滚事务。新包装器不能仅因正常 close 有同步，就忽略半初始化/成对分配失败。

同步已提交的设备工作也不会使未来 graph replay 失效。`load_checkpoint` 的 CPU 集合操作本身不是已有 lookup 的设备 fence。冷分配、原地址稳定的 serving 可以保留原顺序；若未来方案支持热改类型/reload，必须明确 drain、旧 graph 失效/重录、code+scale 代次及失败回滚。当前字面搜索未找到 close caller，不能推成“自动 reload/shutdown 已安全”。V2 `finish_execution` 有 retire 调用与 finally；当前 V1 失败 forward 的直接 join 仍未在本轮闭合，不能用 V2 代替。

## 先价整段，停止同预算小方案

[价格原件](engram-storage-price.json)从同一旧 C8 原始记录重建 lookup → TP8 group → model wave → request deadline，使用既有固定顺序端点函数。7248 个 TASK ROW、906 个 TP8 group 与旧汇总计数吻合，分母重现 TTFT 1491.854839ms、整批18590.769232ms。另保留跨全部 group 的各 TP 最大 lookup 时长串行和14.779885ms作核对，**不把各 rank 耗时乘八**。

| 条件 | TTFT 端点条件省时 | TTFT 条件比例 | 整批端点条件省时 | 整批条件比例 |
| --- | ---: | ---: | ---: | ---: |
| 整段 lookup 免费，替代成本0 | 0.281477–1.761056ms | 0.018868%–0.118045% | 5.090113–8.599814ms | 0.027380%–0.046259% |
| 假设时长按288→264字节比例减少，替代成本0 | 0.023456–0.146755ms | 0.001572%–0.009837% | 0.424176–0.716651ms | 0.002282%–0.003855% |

第二行只是 `1/12` 的 byte-proportional sensitivity，不是物理带宽或 latency 上界。假定独立冷 row、分离 code/scale bank，在32/64/128B line模型中，原32B scale和新8B scale均占一条line，单行读的 charged bytes可以完全不变。相邻行、cache line复用、广播降低指令数与预取会改变机制；即便 lookup 全免费，这个固定调度小范围的空间仍不足。

价格同时付每个 lookup 新增1/5µs，负 wave credit 不截掉。原 collective readiness 等待保留，不把长 allgather span 当成可删 lookup 流量。这些是旧原顺序模型下的条件端点敏感性，不是测得收益或新调度的通用上界。当前正在另行授权采集的配平 C8 没有在此混用，六普通负载也未套用这些数值。

两表共享 host scale 请求 payload 可从24,576,731,200降到6,144,182,800 B，差 **18,432,548,400 B =17.166648GiB**。这是 dtype/shape 决定的共享主机申请/注册尺寸差，DP共享只算一次；没有新分配或实测释放 resident RAM，更不是 NPU HBM/KV pool 容量。初始加载、注册减少也没有暖请求信用。只有具体 host 容量/placement 需求或更大交付机制会改变去留，才继续其原生资格化。

## 地图处置

| 分类 | 范围 | 结果与具体缺口 |
| --- | --- | --- |
| 已停止 | 旧固定 C8 同1%延迟预算的 compact scale、group broadcast、lookup-only kernel 开发 | 连整个 lookup 全免也不足；结束该小范围的位/性能补证，不迭代小配方。 |
| 已停止 | INT8 直接开 MXFP8 flag；只改 scale dtype；复用热 graph 的旧 FP32 bank | 分别改变 code 类型、丢 scale 位型、破坏字节步长/地址/graph 代次。 |
| 已停止 | 主机 payload 差乘16或当作 NPU KV 容量 | 原本已共享，且是 host scale bank。 |
| 待确认 | 紧凑 scale 的独立主机容量价值 | 缺会改变开发选择的实际容量/placement目标与真实 accounting；如继续，仍需每原语逐位、typed loader、注册/半初始化、graph代次验收。 |
| 待确认 | input/TP-head→SP-owner/WKV/iteration 联合 | 保留原6144 BF16完整feature顺序与 GEMM几何；缺 all-participant transport/reorder/ready deadline、图与空/异常回退、六普通和完整C8净价。不能与本轮比例相加。 |
| 已确认可实现 | 本轮新增0 | source、header、历史CPU scale proof与价格均不能替原生新原语准入。 |

下一段进入此前双缓存大交付尚未扫到的 **index-K/long-KV RMS → in-place partial RoPE → DynamicQuant → code/half-scale scatter** 原生读者。既有更大 input-ready/query/SWA parent 全免费条件2.1327%、half/h20µs1.0449%只用于说明值得继续；本轮没有将小 epilogue 单独报价成该收益。先闭合 RMS 真实 OOT 注册/选中分支，再追当前 TorchNPU/CANN 或自定义库的实际 tile、原舍入和 slot/cache 最后读者。

环境：171 hostname 核对成功，同 owned 容器/镜像；仅远端只读 source/SDK/header 与本机旧数据复算。无本轮 Torch/模型/NPU导入、推理、profile、显式远端写入或服务动作。并行配平控制器及其未提交产物保持原样。见[前沿](engram-storage-frontier.json)、[探索记录](engram-storage-exploration-notes.json)。
