# EP1 三个最大主流导出缺口：恢复记录

## 已证范围

stage145 对 EP1 的三个固定主流窗口（opId 402、405、408）计算正时长 TASK 区间并集，最大缺口分别为 420.352、590.368、425.492 µs。缺口的定义仅为不被该主流正时长 TASK 覆盖，405 内有零时长 EVENT RECORD；不能称设备空闲。详见 [DRAFT-EP7-QUERY-METADATA.md](DRAFT-EP7-QUERY-METADATA.md) 的 stage145。

stage146 选取三个缺口各自唯一的前后边界，以及 405 内的零时长事件，共七个目标。七条各有一条同 connection CANN_API；后边界各经连接集合匹配到一条 PyTorch matmul 或 quant/matmul 名称类候选。七个 `connectionId << 32` 生成的 trace flow 候选各有一对 `HostToDevice` 的 s/f，两端均未截断；s 时间精确等于关联 CANN_API 的 startNs（转换单位后）。f 对其中 5 个目标精确等于 TASK 起点；其余两个 MIX 外层任务的 f 分别对应同 taskId 的内层记录，较外层开始晚 0.940/1.220 µs。五条 X 事件能用开始时间、stream 和 taskId 精确匹配；两个 MIX 外层记录不能用该键匹配。此处仅验证记录相符；flow ID 生成契约、connection 复用及事件对象/张量关系仍未证明，不能认定同步因果或完整下发链。

三个前边界 EVENT WAIT 各自的 CANN API 结束到其后边界 kernel launch CANN API 开始，按**同一 rank 的同一 CANN_API 时间域**分别为 **393.850、591.310、426.700 µs**。这只是两个选定 API 的时间间隔：其间可能存在其他线程或其他调用，也可能有 profiler 采集盲区。不能称这些时间为 CPU 空闲、框架调度耗时、必然同步等待，或某对端拖慢；不得用它与设备时间差直接计算下发延迟。七条 HostToDevice flow 的 s 端均报告相同 pid/tid（762111/6840），作为追查同一导出线程的候选线索，不自动证明调用没有其他执行线程。

## 下一步与数据

最小可证伪入口：先读取 CANN_API/PYTORCH_API 的线程字段及其时间域定义，再在以上三个固定本地 CANN 时间区间内查询已捕获的相关线程 API 名称类、时间并集以及遗漏/截断情况；若无法绑定线程，标记 UNKNOWN 而不将进程级活动冒充线程活动。通过这一检查只能区分“存在已导出调用”与“未见已导出调用”，不能独自判定 CPU 空闲或证明设备因果。无需新 NPU 负载。

[ep1-gap-host-links.json](ep1-gap-host-links.json) SHA-256 `a422ed4011b0388515cd98e0d56582fbcd9433fc29fb6583234c1903be3f1f72`；EP1 trace SHA `f3b2558385750fe1471a64baa340d0cde4a2b604f551a5ff3ef95730f9dceafc` 是本次首次计算，先前仅校验文件路径与大小。ZCode stage `operator-ceiling169-v2-146-ep1-host-links` exit-code 为 0、`model_verified=true`、模型 `deepseek-flash`。

## stage148：已保存 CANN API 覆盖

六个边界 CANN_API 的 `globalTid` 均为 `3195455675064`。按该字段同值限定，每个 CANN 时间域区间内部的记录分别为 **17、6、17 条**，区间裁剪后时间并集分别为 **49.090、20.170、49.640 µs**；独立复算一致。与区间长度之差分别为 **344.760、571.140、377.060 µs**，仅是本表所选记录的并集补集，不等于 CPU 空闲、Python 计算或具体调度耗时。API 名称以散列保存，按关键字分类的计数不是互斥分组，嵌套的 API 区间也不能直接相加。

在 402 和 408 中，各有三条 launch 类 CANN 记录的 connectionId 与 stage145 的 stream35 `KERNEL_AIVEC` 对应记录分别相同；405 中，另外两条 event 类 API 的 connectionId 与 stream35 的 `EVENT WAIT/RECORD` 对应。说明所选 CANN 记录并非空白，为同设备其他流任务的候选关联；connection 语义与复用仍待证，不能由此证明这些操作造成主流等待。`globalTid` 同值是数据库内分组证据，其编码和与 trace 显示 PID/TID 的关系尚未确认。PyTorch 表虽也有 `globalTid` 字段，但它的时间列为 TEXT，此次只检查 schema，没有证实两表的时间域和线程编码可以对应，也未查询 PyTorch 区间。

[ep1-host-api-intervals.json](ep1-host-api-intervals.json) SHA-256 `1fc63fdefb9f0809e31737ebd02c8081d2b7e9e733e8dd594a483d8e78017c84`，包含 3 个窗口的全部所选 CANN 记录和字段。stage `operator-ceiling169-v2-148-ep1-host-api` exit-code 0，`model_verified=true`，模型 `deepseek-flash`；无新 NPU 负载。

## stage149：跨表线程与关联语义

三个对应 PyTorch 行的 `startNs/endNs` 均为纯十进制整数 TEXT；CANN 对应列为 INTEGER。三个 PyTorch `globalTid` 均为 `3195455668968`，三个 CANN 均为 `3195455675064`，**字段值不同**。字段值与时间数量级相似不证明二者线程或时钟域相同；目前不将 CANN 窗口直接作为 PyTorch 时间窗口，也不跨表做减法。

已冻结的 [torch_npu FwkApiDbParser](../../sources/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/profiler/analysis/prof_view/prof_db_parse/_fwk_api_db_parser.py) SHA-256 `8a8c558bee4995efde7c4f91fc4afc7fc03697b8d0d33e970e88b487fd8df39f`，与 stage149 安装镜像文件哈希完全一致。其 `get_torch_op_connection_ids_with_task_queue` 用相同队列 correlation 标识关联 enqueue/dequeue，要求 CANN launch 的 globalTid 等于 dequeue 的 globalTid、launch 时间落在 dequeue 时间内，再寻找同 enqueue 的 globalTid 且包含 enqueue 时间的 Torch op；无 queue 分支要求 Torch op 与 CANN launch 同 globalTid 且时间嵌套。构造出的连接集合解释了不同线程仍可能有关联，但源码分支选择、具体 queue 行、真实版本 provenance 和时间转换是否按预期生效尚待验证；不能仅凭字段差异断言本次必经 queue，也不能把源码尝试的时间比较当成测量误差或时钟同步保证。

stage149 还在 profiler 三个指定源码子目录按关键词检查 36 个 Python 文件；命中配置及 localtime 转换入口，未能证明这次导出每条 CANN/PyTorch 记录的共同时间基准。下一步应沿三个固定连接检验队列记录是否存在、类型与相关标识和导出覆盖，并核实来源时间转换；缺证据时保留 UNKNOWN。已存在本地同哈希源码，不需要另一次读取上述源码的远端 Run。

[ep1-pytorch-clock-provenance.json](ep1-pytorch-clock-provenance.json) SHA-256 `c9f9ac9173e73dd67cb5a3b6cc8446690d67e1e22bb55f772e4491654454085e`；stage `operator-ceiling169-v2-149-pytorch-clock` exit-code 0、`model_verified=true`、模型 `deepseek-flash`，未启用 NPU。

## stage152：三个队列记录候选的关联链

对 opId 402、405、408 的三个固定 PyTorch 操作，各自在相同 `globalTid` 且操作时间内部的记录中，唯一找到一条 `type=50002`、名称类别包含 enqueue 的记录；导出 enum 和同镜像常量将 `50002` 标为 queue。每条 enqueue 经本地 `CONNECTION_IDS`，与另一 `globalTid` 的一条 queue/dequeue 候选共享本地 correlation ID，依次为 `1342579697`、`1342579761`、`1342579852`；三条 dequeue 的 `globalTid` 分别等于对应 CANN launch 的 `globalTid`。上述“唯一”仅限本次固定 parent、选定字段和查询范围，不能替代 exporter 的全局唯一性契约。

每组导出数值均满足已保存同镜像 `_fwk_api_db_parser.py` 的严格比较形式：PyTorch 操作开始 < enqueue 开始 < enqueue 结束 < PyTorch 操作结束；dequeue 开始 < CANN launch 开始 < CANN launch 结束 < dequeue 结束。结合已检查的 connection-set 与 flow 候选，导出数据中的 **PyTorch op → enqueue → dequeue → CANN launch** 关联得到明显加强。源码是特定版本的匹配算法；实际导出生成链、线程编码与跨表时间变换、测量误差仍没有完整核验。尤其，数字满足嵌套条件不构成时钟精度证明，**不能据此做跨表时间差、给出队列耗时，或证明造成此前 300–590 µs 的主流缺口**。

stage151 曾因脚本入口 `arg.ArgumentParser()` 拼写出错而退出 1，失败证据保留；修复后本机 `--help` 返回 0，stage152 使用新 stage 重试。其结果 [ep1-fixed-queue-candidates.json](ep1-fixed-queue-candidates.json) SHA-256 `761064a054d3146e30bc634162e89bf7312b54858bbe3d3e8cb3812cb2d51360`；stage `operator-ceiling169-v2-152-queue-retry` exit-code 0、`model_verified=true`、模型 `deepseek-flash`，没有新 NPU 负载。

最小后续调查：先在已冻结源码中确定 `PYTORCH_API` queue 记录时间的生成/时钟转换、以及 `globalTid` 生成语义；若来源存在，针对固定的三个候选仅检验同表 producer/worker queue 数据生成链与同表时间关系，不把 CANN 表时间差当排队时间。若该构造位于未保存的模块，再做单文件有界只读检查；若仅见二进制或无法核验，则停止该路径并保留误差和耗时为 UNKNOWN。

## stage153：OP_MARK 解析入口

同镜像 `prof_parse/_fwk_file_parser.py` 的 `get_task_queue_data()` 读取 `FileTag.OP_MARK` 对应原始标记，按 bean 的 `time_ns` 排序，分别以 `(tid, origin_name)` 匹配 enqueue/dequeue 的起止标记，再用两个 `time_ns` 之差填充各条持续时间；存在未匹配时会记录警告。这解释了 queue 记录的一部分构造逻辑，但此函数本身没有给出 `time_ns` 的硬时钟来源、转换公式、误差或 `globalTid` 编码，也未验证本次捕获的各标记是否丢失。

该文件导入 `prof_config/_fwk_file_parser_config.py` 的 `FILE_BEAN_MAP` 来决定 OP_MARK 对应 bean，又导入 `_constant.py` 的 `contact_2num`。下一入口固定为 OP_MARK 配置、映射的 bean、该 helper，以及当前 parser 余下的入库代码，检查 `time_ns` 和 `globalTid` 构造；不能先假定转换结果。已准备只读查询，交主控决定执行，不自行操作远端。

[fwk-queue-source.json](fwk-queue-source.json) SHA-256 `be68a06e8f7ec2c3daa6a5a3e92d998984db1fc3eafff4d6a6d1cfac2468bfad`；文件内部源码哈希 `5368902817cf6344706f5faad6511d4c06aa297054779bf409a361333b198430` 已与 stage149 采集的同镜像哈希一致。stage `operator-ceiling169-v2-153-queue-source` exit-code 0、`model_verified=true`、模型 `deepseek-flash`；无 NPU 负载。

## stage154：queue 时间和线程的条件性解释

同镜像 `FwkFileParserConfig.FILE_BEAN_MAP` 将 OP_MARK 映射到 `OpMarkBean`；该 bean 对 enqueue/dequeue 起止标记均通过同一个 `ProfilerConfig` 的 `get_local_time(get_timestamp_from_syscnt(raw))` 得到 `time_ns`，后续 parser 配对并写出队列行。已有冻结 config 源码显示，syscnt 转换启用时使用记录的频率、起始计数和偏移，涉及浮点乘法与整数取整；关闭时直接返回输入值。这表明**如果本次数据确由该源码版本、同一配置实例生成，三组同表队列端点可作同域比较**，但实际捕获生成链、配置值和误差上界尚未完整核验，不能把数值精确度当硬保证。

stage152 三对候选在导出表内的 `dequeue.startNs - enqueue.endNs` 数字分别为 **2.610、2.580、2.710 µs**。在上述条件下，它们是从已配对 enqueue 结束标记到 dequeue 开始标记的间隔；不能据此称为全部工作排队耗时、整个主流空档的原因或跨 CANN/PyTorch 的延迟。标记的语义、捕获缺失、采样扰动、相关 ID 复用及时间转换误差仍限制解释。

parser 的 `get_fwk_api` 对 PyTorch 操作及队列使用 `contact_2num(pid,tid)`，helper 为 `(pid << 32) | tid`。在该源码实际生成该表且 pid/tid 宽度满足预期的条件下，PyTorch 操作线程字段拆解为 `744/744`、队列 worker 字段为 `744/6840`；这只说明该版本的 PyTorch 表构造规则，**没有独立证明 CANN 表字段使用相同编码**。

[fwk-queue-bean.json](fwk-queue-bean.json) SHA-256 `82190a66d12959e0fd4029e3ef9335d76056b2658fee2d1285d4ce99405ee35b`，config 和 bean 内容哈希是 stage154 首次记录；parser 和 helper 按此前同镜像哈希核验。stage `operator-ceiling169-v2-154-queue-bean` exit-code 0、`model_verified=true`、模型 `deepseek-flash`，无 NPU 负载。

## stage156：六个队列锚点邻域

对 stage152 三组 enqueue.end、dequeue.start 各取前后 100 µs，共六个 `PYTORCH_API` 本表窗口；按各自锚点 `globalTid` 查询所有类型，裁剪并汇总各类型及整体并集。两个线程按当前数值文本检查均无无法解释时间的行，六个窗口的并集已独立复算；各类型和窗口可能重叠，不能相加。此处时间解释受 stage154 的同配置、版本和精度条件限制。

三个 producer 窗口的全部 API 并集均覆盖 **200 µs**，但每窗都有一条同名称哈希、总时长约 **2.24–2.41 ms** 的 type=50001 外层记录跨越整个窗口；它的范围覆盖不证明线程一直在执行计算或 CPU 利用率为 100%。若仅为观察其余记录而排除这一指定外层行，其余记录的并集分别为 **117.63、142.05、114.55 µs**，这也不是去除嵌套后的实际计算时间。三个 producer 窗口中 queue 类型并集分别 **12.37、15.90、11.70 µs**。

三个 worker 窗口仅选到 queue 类型，分别 3、4、3 条，区间并集 **39.54、36.64、37.16 µs**；各自与 producer 窗口中对应数量的队列记录存在相同本地 correlation ID，为额外已导出队列链的候选。worker 未覆盖时间不是 CPU 空闲：这里没有捕获全线程指令执行，其他 API、运行时和采集遗漏仍可能存在。

这六个 200 µs 窗口由 PyTorch 表锚点构造，不是前述 CANN 窗口，也不能覆盖或解释此前完整的 300–590 µs 主流缺口。后续如能通过已保存的匿名名称哈希与调用栈，确认长外层 op 在源码中的角色，可进一步分析其子结构；若没有相关栈或可靠关联则停止此入口，不能把外层持续时间当耗时根因。

[ep1-queue-neighborhoods.json](ep1-queue-neighborhoods.json) SHA-256 `30810bf8927325678ad5a5ffd9ad909bb7519ff72bc1b5468096b0ed7ca47b87`；stage `operator-ceiling169-v2-156-queue-neighborhood` exit-code 0、`model_verified=true`、模型 `deepseek-flash`，无 NPU 负载。

## stage158：外层记录的关联止点

按同线程、起止、type 与名称哈希，在 EP1 DB 唯一匹配三个 stage156 的长外层记录。三条均 `callchainId=null`、`connectionId=null`，查询未取得调用栈或具体代码路径；允许类别仅命中名称中的 `vllm`、`forward` 子串，不能确认具体函数、模型调用或其对时间的贡献。`PRAGMA table_info(PYTORCH_CALLCHAINS)` 返回空，仅表明此表没有可供查询的 schema；其他采集来源是否有栈不在此查询范围。外层记录持续时间不能当 CPU 有效计算时间。

按预定条件停止该邻域继续检索；不凭泛化名称解释缺口或收益，改看其他已归属异常。[ep1-outer-ops.json](ep1-outer-ops.json) SHA-256 `f0748c021d0631e74c3dd326809cc32ab3ea29a91bb90b3eef354ca16a4321a8`；stage `operator-ceiling169-v2-158-outer-ops` exit-code 0、`model_verified=true`、模型 `deepseek-flash`，无 NPU 负载。
