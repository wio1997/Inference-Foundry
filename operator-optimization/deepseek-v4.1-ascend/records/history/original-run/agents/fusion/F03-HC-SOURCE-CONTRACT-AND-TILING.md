# F03：冻结 HC 源码、归约域和分块重组

本报告更新 `F03-POST-PRE-TILE-READINESS.md` 中“未取得内核实现”的状态。ZCode stage119 获取的 `f03-hc-inventory.json` 包含两只固定 169 容器一致的 25 个文件、missing 为 0，库存 SHA-256 `34db9a40381a140c50e2864776011a0841a29f415bc43a8200b3097bad7c9259`。已逐文件验证 hash 并解码到 `agents/fusion/frozen-source`。以下行号相对该快照；这只是**容器内源码与接口声明**，未证明历史工作所加载的算子二进制由这些源码构建，也未取得某次调用的 tiling 数据及真实 tensor 身份。

## 数学依赖、接口和现有物理同步

1. `tests/e2e/nightly/single_node/ops/singlecard_ops/test_npu_hc_pre.py:53–84,186–215` 提供 CPU 参考：每个 token 将 `x[...,HC,H]` 的末两轴展开，对完整 `HC*H` 求平方均值并计算归一化系数；经线性投影得到 `pre[HC]`, `post[HC]`, `comb[HC,HC]`，`comb` 先对其最后 HC 维 softmax、再跨另一 HC 维归一化，随后按迭代次数交替归一化两维。该参考每 token 独立，未对 token 轴做数学归约。这个测试有浮点容差，不是设备实现逐位等价证明；参考也明确将 matmul 输入转成 HF32 表示（`46–65`），不能把它当普通 FP32 GEMM 的逐位契约。
2. `hc_pre/op_host/hc_pre_def.cpp:24–67` 声明输入 x 和输出 y 为 BF16；`hc_fn`, scale, base, 可选 `pre_mix` 以及 post、comb、可选 pre 为 FP32。参考测试 `191–210` 明确 v3 有外部 pre_mix 时输出 `[T,H]`, `[T,HC]`, `[T,HC,HC]`, `[T,HC]`。源码 `hc_pre_tiling.cpp:97–118` 支持形状 `[T,HC,H]` 或 `[B,S,HC,H]` 并将 B、S 相乘作为 bs。
3. 该 A3 分支 `hc_pre/op_kernel/hc_pre.cpp:90–105` 执行 Part1 再执行 Part2；Part1 在 `hc_pre_m_k_split_core.h:82–175` 对 token M 和 K 轴分核、存每个 K 子块的投影与平方和，并在最后 `SyncAll<false>()` 后才进入 Part2。Part2 在 `:348–400,457–506` 按当前行收集/归约 K 分块及计算 HC 矩阵。**这是一设备内的现有全核同步，不是跨设备通信，也不证明数学必须作全 token 同步。** 即使数学逐 token 独立，当前代码也不等于已按行对外提供 ready 事件。
4. A3 `hc_post.cpp:50–55` 使用 `hc_post_d_split.h`；其 `:299–385` 对每个 token 的 feature 片段，读取同一 token 的 x、各 HC 路 residual，以及该 token 的 post 和 comb；FP32 乘加，再将双字节结果 `CAST_RINT`。同一 feature 片段需要各 HC 路残差及系数；源码显示按 token 与 D 循环，不是跨 token 数值归约。源代码逻辑支持研究按 token/feature 局部生成 R，但不能凭此认定从该 kernel 中途可直接对另一 kernel 发布可见片段。

## 新的可条件拆分：先计算 y，再算本次系数

参考 `test_npu_hc_pre.py:72–84` 和 A3 `hc_pre_m_k_split_core.h:410–431`、`hc_pre_base.h:504–513` 都显示：**仅当外部 pre_mix 存在时**，`y[t,d] = cast_BF16(sum_h(pre_mix[t,h] * R[t,h,d]))`；该值不依赖本次的 `hc_fn / hc_scale / hc_base` 计算结果，而本次的 `post/comb/pre` 仍依赖完整 R 归约与投影。

`model.py:902–919` 的 FFN pre 显式传入 earlier `attn_pre` 作为 pre_mix。因此在保持数值和生命周期的条件下，可研究从 post 的 R 和已就绪 `attn_pre` 提前生成 y，将 y 送到 FFN norm、router、MoE；同时完成本次 post/comb/pre 系数，然后在 FFN post 使用前等待系数和完整 MoE 结果。原源码仍需分别提供 residual R、`ffn_post/comb`、`ffn_pre` 给其已知后续消费者，且 must 保留原 FP32 各 HC 路乘加、归约顺序、BF16 舍入、input IDs、跨 rank 贡献、事件、图内 buffer 地址及生命周期。前述公式是依照参考写出的数学形式，不意味着更改归约顺序后逐位一致。

现有 A3 实现并非已经完成该拆分：它的 y 仍在 Part1 `SyncAll` 及 Part2 生成 pre 之后执行，即使 hasPreMix 为真也如此；因此此处只是源码支持的依赖重排候选，未实现、未验证正确性或收益。无外部 pre_mix 时 y 使用当前 pre，本次完整投影和归约就是 y 的依赖，不能使用该提前路径；信号发布粒度、混合 kernel 间资源争用以及当前实际输入是否有 pre_mix 仍需确认。

## 上游 epsilon 问题与源码 tiling 条件

冻结 `hc_pre_m_k_split_core.h:484–506` 在首个列求和和循环中的列求和后，两次以 `curRowFactor * hcMult` 对 `reduceLocal` 加 epsilon；`hc_pre_tiling.cpp:160` 计算 `hcMultAlign = RoundUp(hcMult, 32/sizeof(float))`，而 `hc_pre_m_k_split_core.h:247,264–271` 以该对齐宽度分配行缓冲。这与 [issue #19357](https://github.com/vllm-project/vllm-ascend/issues/19357)、[PR #19401](https://github.com/vllm-project/vllm-ascend/pull/19401) 所述模式同构；并非仅因为镜像标签相同而猜测。循环中相邻的 row 归一化同样写 packed 长度，但 PR 对其注明 packed 存储是正确的，不应把它一概改成 aligned。

按 issue 对 column sums 与消费者对齐步长的说明，若 `hcMult=4, hcMultAlign=8, curRowFactor=2`，加法处理槽 `0..7`：第 0 行有效列 `0..3` 和 padding `4..7` 获加 epsilon，第 1 行有效列 `8..11` 不在范围。该机制需要单次处理多个 row 和实际浮点影响；epsilon 数值太小可能被舍入，是否改变 comb 取决于数据。不等同于发现了不同 token 间数学归约，也不等同于本次历史任务已被证实出错。

源码 host tiling `hc_pre_tiling.cpp:150–157,188–253` 先算 `rowOfFormerBlock=ceil(bs/aivCoreNum)`，`rowFactor` 在 `dFactor == d` 且 UB 空间允许时增长，否则保持初值 1。**如果**实际使用该源码的 A3 路径、`bs=3` 且 `aivCoreNum>=3`，则每块至多一行并由该算法得 `curRowFactor=1`；前述“多行时尾行遗漏 epsilon”机制不触发。反过来，`bs>aivCoreNum` 只是多行块的必要条件之一，并不保证 rowFactor>1：还要看 UB 与 d 切分，且实际是否出现差异依赖输入。stage122 已取得归属到 HcPre 的 profiler 物理形状，见下节；实际 tiling 读取的 AIV 数及二进制来源仍未绑定，因此只能给出条件推导，不能说 169 生产结果安全或受影响。

尚缺：已加载 HC op binary 与源码/版本的可核验映射、目标 HC 的 physical shape 和 tiling key/`curRowFactor`、实际 graph tensor ID 与流事件、重排后的数值与服务影响。继续只读关联已有证据；当前不安排额外 NPU 测试。

## stage122：已采样 HC pre 的物理形状

对 `agents/fusion/f03-hc-shape-correlations.json` 本地独立核对：结果 SHA-256 为 `5bc0b00f56a167a3abbda8ec498ba7f4242b8045c1c4b96504d830fdfd5f0657`；内部查询脚本、冻结窗口与 CSV 清单的 SHA 均与本地文件一致。记录包含 16 rank × 3 个内部 replay 采样轮 × 40 层 = **1920 个完整窗口**，无歧义或缺失窗口。两条按位置归属的 HcPre 各 1920 个，均记录输入 `3,4,5120;24,20480;3;24;3,4`，dtype `BF16;FP32;FP32;FP32;FP32`；输出 `3,5120;3,4;3,4,4;3,4`，dtype `BF16;FP32;FP32;FP32`。该 5 输入记录与存在外部 pre_mix 的接口及其 y 依赖重组机会相容；它不证明第五个输入在每次执行时的 tensor 身份、值与别名。

角色归属是相同 rank、model、stream、非重叠层窗口中，依据此前严格检查的 `HcPre → HcPost → HcPre → HcPost` 次序及 attention/MoE 等结构不变量，将第二个 HcPre 视为 FFN pre。查询在每个窗口内用 marker ID 核验唯一性和顺序；graph 可能复用 ID，未用 ID 独立跨 replay 连接。该映射比“同一层里有 HcPre”更强，仍不是直接 tensor 读写或 runtime 算子绑定证明。样本是已捕获的 target decode 内部轮次，不覆盖全部请求、prefill、eager draft 或动态 shape。

**条件计算可收紧至具体采样 shape**：若这条被捕获记录实际采用冻结源码的 A3 host tiling，且编译/运行的 `aivCoreNum >= 3`，输入的 physical `bs=3` 给出 `rowOfFormerBlock=ceil(3/aivCoreNum)=1`；初始 `rowFactor=1`，即使 D 全载且进入增行循环，由于每块最多 1 行，最终也为 1；尾块同理，故所述逐块多行列 epsilon 遗漏机制不出现。已有 `evidence/actual-sku-and-cp.json` 的 `Ascend910_9382.ini` 文本列 `vector_core_cnt=48`，但它不是这次具体 HC tiling 上下文返回值，也不自动证明当前加载了对应平台配置、未受逻辑/虚拟化限制，不能将 48 写成已捕获的 `aivCoreNum`。

未取得该 replay 的 tiling key、`stage2RowFactor`、实际 `aivCoreNum`、运行 binary 与源码构建关联或真实数值结果；因此上面的排除只在这些条件下成立，而非对本次全部服务、全部形状或所有数值问题的判断。stage122 为 ZCode 只读既有 profiler 查询（exit 0、模型 `deepseek-flash` 验证），无新增 NPU 工作。

## stage124：EP8 layer 0 的六个 HC pre profiler 条目

对 `f03-hc-task-metadata.json` 本机复核，结果 SHA-256 `ac8a58ff487a988ec46a23ca7c48fe2c91bee70acdb5c56961cef2d4d22862e4`，其脚本和窗口 SHA 与本地文件一致，DB SHA 与 EP8 已保存证据一致。六个条目分别是三轮窗口的 attention/FFN pre：每个按模型、流、时间窗及 marker ID 找到 1 条 TASK，每个 globalTaskId 对应 1 条 COMPUTE_TASK_INFO，但该 globalTaskId 在 TASK 中各出现 7 次。因此 compute 记录是相应 ID 的关联元数据，不能仅靠 ID 将它当成某次重放独立捕获的 tiling 或把重复行数算成独立样本。

六条解码 `name/opType=HcPre`，shape 与 stage122 一致，`attrInfo` 均为 `attr_0:4|attr_1:20|attr_2:0.000001|attr_3:0.000000`。原始记录的 `blockNum=24`、`mixBlockNum=48` 是 profiler 字段原值；未取得字段语义将其绑定到 `aivCoreNum`、`stage2RowFactor` 或实际可用核数，不能把 48 当作该调用实际 tiling 核数。属性的打印精度未知，`attr_3:0.000000` 也不能推断输入参数恰好为零。

该已检查的 profiler 属性没有展示可解释的 tiling key、rowFactor、算子二进制路径/SHA 或构建关联；这不证明其他产物不含这些信息。任务类型、kernel 名和参数也不能证明 source-level tensor 身份。结合 stage122，仍只有对冻结源码/假设 AIV 条件下的排除，不能升级为实际运行的数值结论。stage124 为 ZCode 只读查询（exit 0、模型 `deepseek-flash` 验证），没有新增设备任务。

## HC comb 的现有串行依赖：条件性 39 步

针对上文的 A3 冻结源码，而不是对所有可能实现作判断：`hc_pre_m_k_split_core.h:484–506` 先计算 softmax，然后依次做一次列归一化，以及 `iterTimes - 1` 次循环；每次循环先行归一化，再列归一化。当前步骤要读取前一步的 `combFragLocal`，其归约、加 epsilon 和除法之间保留对应的向量管线次序。因此当且仅当这一实现实际采用 `iterTimes=20`，comb 路径中存在 **20 次列归一化 + 19 次行归一化，共 39 个相互衔接的归一化步骤**，另外还有 softmax；每一步内部的多个元素和不同 token 可并行，这不是 39 次 kernel 调用或 39 次全核同步。源码每步加 epsilon 后、除法前使用 `PipeBarrier<PIPE_V>()`，不要将其等同设备级或跨卡 barrier。

`hc_pre_def.cpp:70–73` 声明 `hc_sinkhorn_iters` 默认 20；`hc_pre_tiling.cpp:51,78–87` 将属性索引 1 读入 `iterTimes_`。stage124 六条 profiler 元数据中 `attr_1:20` 与此相容，但尚未证明 profiler 属性序列和本源码 schema 的绑定，也未证明历史二进制由该源码构建或历史真实 tiling 值为 20。所列 39 步是**该固定源码、给定值下的结构约束**，不是任意数值等价算法的通用下界或硬件耗时下界。任何替代数值算法、变更轮数、epsilon 或归约顺序都需单独证明其语义及精度满足实际契约。39 步的各自耗时未记录，不能据此断言它们是 HcPre 历史中位约 34.819 微秒的主因，也不能据此计算提速空间；此前 F02/F03 的外部 pre_mix 提前交付 y 候选已经覆盖并行系数支路，不重复登记为新机会。

核验文件 SHA-256：`hc_pre_m_k_split_core.h` 为 `4d697b15e541216722369284cbd8c9bd5f4154cd0d2251cdb06a6cd46f42fc14`，`hc_pre_def.cpp` 为 `051e9b4f722a3b36f1bb447cafd7700e1e37cdd14648d417b66b3f7db03af109`，`f03-hc-task-metadata.json` 为 `ac8a58ff487a988ec46a23ca7c48fe2c91bee70acdb5c56961cef2d4d22862e4`。本次仅分析已保存材料，未运行 NPU 或修改目标实现。

## HC 各输出和 comb 迭代之间的具体依赖与共享缓冲

进一步区分固定源码中的执行顺序和输出的数值依赖：A3 `hc_pre_m_k_split_core.h:362–400` 先归约平方和、计算本行 `rsqrtLocal`，再归约投影的前两片并计算 pre；`:401–439` 可选写出 pre，读取外部 pre_mix，并按 D 片计算、写出 y；`:442–455` 计算、写出 post；`:457–510` 独立读取、归约第三片投影，再经 softmax 和 comb 归一化生成 comb。这里是单 row 循环体的源码调用/提交顺序；MTE 和 Vector 的底层时间重叠需要具体硬件时序，不能只从源码顺序推断。条件 `iterTimes=20` 时，39 次相互衔接的归一化只属于 **comb 数值分支**，并不是 pre、post 或外部 pre_mix 分支的 y 的数值输入，也不能据此推算 39 步中每一步的硬件耗时。

可研究的更细粒度依赖图是：同一 token 的投影与平方和完成后，pre、post 与 comb 分别使用各自的投影片段、共享归一化因子、各自的 scale/base；各自数值不要求先获得另两个输出。已提供外部 pre_mix 时，y 依赖本 token 的 x 和外部 pre_mix，连共享 `rsqrtLocal` 和本次投影也不需要；未提供时 y 必须使用本次 pre。每个输出的消费者和最后等待点仍按 F02/F03 所述，尤其 attention 的 HC post 使用 post 和 comb，FFN pre 要使用 attention pre，不能将 comb 未完成等同整层已经完成。该结论限定冻结实现的数值依赖和所载模型路径，不证明任意替代算法逐位相同。

**不能将上述数值无依赖直接当成当前源码可安全并发。** `:398–400,429–430,444–446,476–485` 中多个过程复用 `rowBrcbLocal0` 和/或 `hcBrcbLocal1` 临时缓冲；`hc_pre_base.h:432–439,516–529` 的 ProcessPre、ProcessPost 对传入 mix 局部张量原地操作，ProcessPre 还令输出与首片 mix 同址。若重排为不同执行单元，必须证明各阶段 scratch 与原地输入互不覆盖，保障 pre/out、post、comb 各自产生后的存储、事件和生命周期，正确处理前面 K 分块归约和现有 `SyncAll<false>()` 的可见性；执行单元还可能争用 Vector、存储及带宽。现有源码把各输出 CopyOut 到 GM，不等于有可供外部 attention 或其他 kernel 使用的逐块 ready 事件：当前调用契约没有证明任何输出可在 `HcPre` 完成前对外消费。不同 token 之间的数学独立也不自动排除现有实现的全核同步。该结构分析未证明历史二进制、实际事件时序、并行收益或精度，未启动 NPU，也未改实现。

**2026-10-30 重复 Hc 任务的同窗时长对照：**从已有 `diagnostic-20261029/target-layer-all-kernels-16rank.jsonl` 逐层筛选 HcPre 与 HcPost 各恰好两条记录，对每对核对名称、类型、输入输出的 shape/dtype/format、core、block、stream 均相同，按设备开始时间排序；四个摘要每组 320 对、两算子总 2560 对，未发现导出时间上对内重叠，也没有因签名或条数排除的组。对两组 `[1,1023,5120;1,1023,4,5120;...]` HcPost，第二次/第一次时长比分别中位 0.863 和 0.856（配对比值范围分别 0.799–1.006、0.796–0.933）；各任务独立时长中位分别 141.737→121.219 µs、143.687→122.779 µs。另两组分别为 595/643 行，HcPost 比值中位 0.974/0.998；同四组 HcPre 两次比值中位 0.990–1.029。中位数比与两个中位数之比不可混同。两条同层任务的作用点、输入内容、有效工作量、与其他流争用不由相同签名证明；不同摘要/形状也不是受控实验。该结果是被选记录的条件差异，不是在线收益或根因。复算脚本 `agents/diagnosis/compare_hc_repeated_scope.py`；逐窗匿名比值、原始任务时长与来源哈希见 `agents/diagnosis/hc-repeated-scope-anonymous-pairs.json`，已独立按比值排序，不能跨算子对照设备身份。

独立复算进一步固定输入 SHA 并核验 2560 个 `(EP,摘要,层,算子名)` scope 唯一；比值使用 CSV 中的 `Duration(us)`，对内不重叠检查使用 trace 起点及持续时间，两种口径不应混为独立硬件计时。脚本 SHA-256 `679748c736f3735d769f17bdb8015525fde4566c28c11880826cf4b405d6d7ae`，聚合 SHA-256 `6169276fe118ef6803d4f7d91ee6e9c0c245865eb215510699fe77a802c40804`。

**2026-10-30 有界 host 角色核查：**尝试只读取既有 EP0、`DeepseekV41DecoderLayer.forward:736-51` 的 profiler SQLite，把该 scope 主机 API 经 `CONNECTION_IDS`、`TASK` 与 `COMPUTE_TASK_INFO` 连接到 HcPre/HcPost，进一步查询调用栈；在第一条 scope 查询即得到 SQLite `disk I/O error`，尚未查到 HC host 记录。复查唯一匹配的主库：文件大小 662482944 字节，文件头为 SQLite 格式，未见对应 `-wal`/`-shm`；`mode=ro&immutable=1` 重试同样在首条查询报 `disk I/O error`。这些信息不足以诊断文件系统或判定数据库损坏；本次仅覆盖该库和这个拟查询 scope，未执行到关联阶段，也不能断言其他库缺失调用栈。既有 stage122 的 marker 次序条件归属仍保留，不能升级为两次调用的直接 host 栈、张量或事件身份。未复制或修改数据库，未启动设备任务。

**2026-10-30 EP8 已存小形状样本补查：**另选先前已用于 `f03-hc-task-metadata.json` 的历史 profiler 数据库，仅看 EP8、第 0 层、第 1 个内部窗口里标记位 1 和 3；按 stream、model、task ID 和时间窗口筛选，两处均各匹配 1 条名为 `HcPost` 的设备记录，二者 `connectionId` 均为 184330。此前同窗两处 HcPre 也记录同一个 connectionId，说明至少这四条设备任务共享该标识，不能仅靠它一一识别调用点或 tensor。对这两个 HcPost 的 connectionId，在该库 `CONNECTION_IDS.connectionId` 查询未返回行；所列核心表中也没有 `PYTORCH_CALLCHAINS`。这次没有得到直接 host 调用栈关联，空查询仅覆盖该库、该字段和这两个标记，不能断言其他关联方式、其他数据或历史栈不存在。该样本与 1023 行的慢速对照并非同一采样，不应用它直接解释后者；标记角色依然只是基于窗口内顺序的条件归属。SQL 只读（`mode=ro`、`query_only`），库读取前后大小、修改时间、inode 未变，未复制或修改数据库，也未启动设备任务。

**2026-10-30 HcPost 同 scope 已选任务的区间核查：**复用上面两组 1023 行、各 320 对的数据，`db852bff…` 为 EP0–7 × 40 层，`1b826883…` 为 EP8–15 × 40 层，共 640 个 layer scope 和 1280 条 HcPost。逐 scope 核对两条的导出物理签名一致、flow ID 唯一、顺序不交叠，并将第一/第二条的 CSV 时长成对核对到先前匿名配对结果。之前的第一/第二中位数分别是 **141.737/121.219 µs** 与 **143.687/122.7785 µs**；对应逐对比值中位 **0.8625 / 0.8559**。两组是同一次 profiler 采集，物理 shape 一致不保证真实输入、有效行、权重或请求等价。

对每条 HcPost，只检查**同 device track PID、同一个已选 layer scope 内、由 host flow 关联且有 kernel CSV 的其他任务**，按本卡 Decimal 导出时间作正区间相交；选中记录内同 track 其他任务与其他 track 任务交叠数均为 **0/1280**。因此不能用“这些已选其他任务在导出时间区间与 HcPost 本身直接交叠”解释该范围内的前后时长差；不能据此排除 scope 外、其他线程、没有 CSV 的同步事件、未导出活动、前序活动残留、kernel 内部等待或争用，也不证明整卡空闲或时长差的具体根因。

冻结 [model.py](../../sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py) 的条件路径是 attention 结果→第一次 `hc_post`→FFN pre/MLP→第二次 `hc_post`；HcPost 导出输入为 BF16/BF16/FP32/FP32、输出为 BF16。该调用顺序与 profiler 的两处 HcPost 结构位置相容，但本次没有绑定真实 tensor/storage 与运行时源码版本，不能把顺序单独升级为该 1280 条的直接 producer/consumer 证明。复算 `python3 agents/diagnosis/check_hcpost_scope_overlap.py`，输入同 scope 关联 JSONL 和旧逐对结果，均核验 SHA；脚本 SHA-256 `4d30b967e8d71090063d8091f40e0f26cac1ea3000dbaa4e0cea5c1d30cd3f79`，结果 [hcpost-scope-overlap.json](../diagnosis/hcpost-scope-overlap.json) SHA-256 `8cd5e0f6d99185d618185e1753f3fcd9cb609ee149f549586be82033c865f34e`。只读旧数据，未启动 NPU、未修改目标实现。
## 已选 L0 两条 HcPre 的时长与计数代理（2026-10-30）

对历史 prefill 导出的 L0 scope，按同设备 track、同流及开始时间配对两条同物理签名 HcPre。32 个 scope 分为物理行数 M=595（8）、643（8）、1023（16）。第一条/第二条时长之比的范围和中位数分别是 0.9815–1.0545（1.0441）、0.9875–1.0440（0.9982）、0.9891–1.0980（1.0441）。同一对的 `aic_total_cycles × aic_mac_ratio` 比值全范围 0.9973–1.0029，`aiv_total_cycles × aiv_vec_ratio` 比值全范围 0.9984–1.0015。

这说明在所选记录中，两种导出代理的配对值接近，仍可出现约 9.8% 的单对任务时长差异；不能用这两个代理的相似就推断时长完全相等，也不能仅凭单个 ratio 给出原因。两个 HcPre 的输入内容、参数、有效工作量、实现身份及时钟/采集干扰未知，未证明第一条与特定模型源码调用的 tensor 绑定。计数字段的定义、单位、采样和之间的耦合也未独立证实，代理不是实际执行量、硬件利用率或时间分解；它们的接近不能排除访存、同步、调度或其他竞争。

复算 `python3 agents/fusion/analyze_hc_l0_pair_profiles.py`，脚本 SHA-256 `9cce290f87a2981a001322c18ab466491cea709c408481949767775be58ee487`，核验原 JSONL SHA-256 `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`，检查同轨、不同 flow、行数组和代理按导出字段的算术；它不验证 trace 采集完整性或操作语义。

## 同 scope RoPE 重复任务的时长负对照

沿用同一次重 profiler 采样，针对两个 1023 行摘要中的 `digest × EP × layer` scope，对每个 scope 分别选取一对 HcPost 和一对 InplacePartialRotaryMul。同一对按该设备轨道上的开始时间排序，要求两条事件同 track、同 stream、时间区间不相交，名称、类型、输入输出 shape、dtype、format、core 和 block 字段一致。RoPE 输入为 `[1023,1,64,512]` 与两份 `[1023,1,1,64]`，dtype 为 BF16/FLOAT/FLOAT，输出 BF16、格式 ND，1120 条记录均为 AI_VECTOR_CORE、block 47、mix block 0。这里仅检查有对应导出记录的 scope，不以形状推断张量内容、模型调用位置或有效 token。

每对使用 trace duration 计算第二次/第一次比值，然后在该摘要内取中位数；与上文基于 CSV duration 的统计是不同口径的同源记录，不当作独立测量。

| 摘要 | 已选 scope | RoPE 比值中位数 | HcPost 比值中位数 | RoPE 比值大于 HcPost 的 scope |
|---|---:|---:|---:|---:|
| `db852bff…` | 280 | 1.003896 | 0.863495 | 275 |
| `1b826883…` | 280 | 1.004708 | 0.855338 | 278 |

另外 80 个 scope 没有纳入该同形 RoPE 对照：EP7 和 EP15 的对应两个 RoPE 记录每条为 **1019 行**，而不是 1023 行；每个 EP 各 40 层，不是任务缺失。这个对照排除了“在所选 scope 中，所有重复算子的第二次耗时都按同一个约 0.86 倍数变化”的简单描述。不同算子执行单元、调用点、数据、参数、有效工作和未导出活动均可能不同；它不能证明缓存预热是否存在，或解释 HcPost 差异的原因。采样具有 profiler 扰动，同一轮的层和 rank 也不是独立请求，不能据此估计线上收益。

复算 `python3 agents/diagnosis/compare_hc_rope_scope.py`；脚本 SHA-256 `1bfa4cc99e0b584793eeb5e86124c16dc37c2ff7100889dd3489076df6d94c30`，读取并核验 `diagnostic-20261029/target-layer-all-kernels-16rank.jsonl` SHA-256 `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`。脚本核查 640 scope 唯一、每对导出签名和流、先后与不重叠，并确认两个 1019 行排除组；不访问设备或修改目标实现。
