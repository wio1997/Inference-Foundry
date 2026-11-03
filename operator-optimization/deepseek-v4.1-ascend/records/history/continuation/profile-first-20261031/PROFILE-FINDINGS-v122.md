本轮围绕 MoE dispatch 的一个具体重组机会补齐了原始数据：同一输入 token 的多个专家若在同一 EP 目标上，能否只传一份激活，再按原来的专家行映射展开。先复查已有 profile 和观测索引，再只读取原服务器保存的 tensor/metadata 归档；没有加载 Torch、模型或 NPU，没有 replay，没有运行新的设备探针。先前批准的 Pad 探针已在 v120 完成，本轮没有重复使用该授权。

来源是原 observer 的 87 个正式 EP16 cohort、1392 份 rank/layer 快照及配套 metadata，共核对 2784 个文件的 SHA，读取 35,632,142B。快照含 post-call TopK IDs、实际 active mask、计数及映射。用标准库受限解码器分析，逐份核对了 epoch、专家 histogram、receiver counts 和既有 sender/receiver 矩阵。1392 份记录均声明 linear placement、无冗余专家、EP 成员 0..15；记录的 capacity 参数均为 65536，最大 receiver assignment count 为 14852，计数模型中没有截断。保留的实际样例 owner map 也符合连续分区。历史加载对象、全部实际指针/placement 与网络拓扑仍不由这些元数据单独证明。

target 的 80 个 cohort/1280 份快照中，远端 assignment 引用 A=2,565,204；按“源记录行、目标 EP”分组 D=2,204,823，结构性重复引用 360,381。draft 的 7 个 cohort/112 份快照为 A=731、D=727，仅重复 4 次。这里累计的是跨层、跨 rank 的记录行，不是去重请求数、全部 C8 token 数或执行次数。capture/replay 可覆盖归档文件，observer ordinal 不能直接作请求编号。这批 MoE 观测没有 Engram lookup IDs，不能据此估计设备行缓存命中率。

当前 dispatch 已经传 INT8 code，逻辑 record 是 H+32，其中 H=5120；不能以 BF16 激活作为基线夸大压缩空间。候选主布局明确计入 H bytes code、32B header（保留 FP32 scale raw bits、source row、epoch、fanout count 等）及 64B 的最多六个 8B 描述符区域，每条描述符为 local expert 与原 expanded-row ordinal。固定 stride 为 5216B，这是一种声明布局，不是最小 ABI，也没有宣称现有 padding 可以直接复用。

还保留原来的每专家 assignment counts A，并单独发布 unique-packet counts D。选定计数方案为每 rank 16 个 uint32、每 cohort 全矩阵 1024B；若全部其他 15 个参与者各接收一份，每 cohort 的逻辑远端内容为 15360B。这个量不是物理链路字节，也不强制新增独立 collective；若复用现有协议空间，必须证明新 ABI、所有读者、地址/offset 与发布完成条件。

在所有拟合并组的原量化 code 与 scale 均可按要求共用、上述布局/计数协议合法的条件下，target 样本的 dispatch 逻辑内容从 13,215,931,008B 变为 11,501,585,568B，减少 1,714,345,440B，即 12.9718%。BF16 返回仍按 A 发生，参考量 26,267,688,960B 保持；dispatch 加该返回参考量的减少比例为 4.3419%。这两个比例都不是 MC2 或请求耗时收益，也不是全模型可达加速比。

同一主布局对 draft 反而使逻辑内容从 3,766,112B 增至 3,899,552B，增加 133,440B，即 3.5432%。因此不能直接把 target 判断套给 draft。也不能因此否定所有 draft 布局：本轮另列 K 专用布局和 4B packed descriptor、外部 frame epoch 的紧凑布局，连同两种计数计划共 12 个角色/布局/协议模型。紧凑 draft stride 可为 5152B；仅在新增计数内容为零且新协议成立时，它有 20,608B 的微小正节省，加入选定 replicated count 后则变为负值。整数位宽、旧行 ordinal 范围、外部 epoch 和同调用边界都是显式条件。

分组 D 是结构性复用，不等于量化字节一致性已证。归档没有保存 code/scale，不能从“同 token、同目的地”单独推出原 assignment 内容相同。冻结源码中的 full-load CopyOutXQuant1H 已在分区内对一行量化后复制到多个 expanded positions；因此新方案不能再领一份 K 倍量化计算减少。其它 key、分区边界、加载实现及 scale 位模式仍需证明。候选可以采用 producer 合法性证明，也可采用 byte-certified grouping、比较失败逐条回退；后者的有效 packet 数可介于 D 与 A，比较、读取、计数、packing 和分组本身都不是免费。

保持语义的局部候选图已登记 13 个节点：实际 postmask IDs、原量化与 A counts、token/destination grouping、精确原行描述符、字节证明或检查、新 packet stage、D counts、peer pull、原行/scale 展开及原 GMM1/SwiGLU、GMM2/return、weighted unpermute，另保留概率输入边。A counts 与原 inverse/probs 直接连到相应原消费者。描述符必须恢复原专家/源 rank/行顺序和 capacity 前缀，不能由 packet 到达顺序替代；router score TopK 和 MC2 专家分组也不是可直接消除的同一个 sort。

接收端可以选择恢复现有 dense rows，或另行选择间接 GMM iterator/L1 gather 的候选；后者不等于当前 CATLASS 已自动支持，不能同时领两种路径的收益。专家计算和返回是权重相关的 assignment 结果，不能把 v119 的行工作、GMM FLOPs、权重读取或返回量中的 A 一律替换为 D。所有旧 round、scale/probs raw bits、alias、副作用、全 rank 参与、capture bank、remote pull 完成与旧读者/queued writer 退役都保留。新压缩 stage 与旧量化 source bank 可能重叠，峰值存储可能增加；只能按实际 packet/所需 tile readiness 分析，不能假设整层串行屏障。

理论对照独立标为 SYMBOLIC_CEILING：只有证明选定域的实际必要物理工作 W_r 与容量上界 C_r 后，才可用 T>=max_r(W_r/C_r)。本轮逻辑 record bytes 没有直接代入物理网络/HBM 必要量，新增分组/比较/packing/计数/展开也尚无同调用内部时延；数值时间下界、吞吐上界、实测/上界比均为 null。没有从这些载荷比例推导硬件利用率或最低利用率模块。

6400 个独立小规模 descriptor/旧序/capacity 截断参考用例通过，前置 7071 项核对通过。现有 REL73-07 与 HG73-MOE 补入证据，没有新增根候选、中央 pair、联合组或 observed 图节点。台账保持 35 项模型候选、7 项历史框架记录、356 条语义记录/26 模块、135 条条件关系/26 联合组/460 待核对 pair；2073 类/285598 次条件审阅、9 类/1352 次待补原生字段、849 缺 PMU、16 phase 未绑定图和 3136 含糊归属不变，整模块关闭仍为 0。源服务器只读任务完成、SSH 已关闭；整体目标继续，不能因本报告成稿而标记全部优化机会已经分析完。
