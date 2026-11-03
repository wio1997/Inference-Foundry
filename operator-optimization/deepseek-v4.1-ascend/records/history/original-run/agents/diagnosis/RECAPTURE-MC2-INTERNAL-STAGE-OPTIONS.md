# 单个 DispatchFFNCombine 内部阶段：采集能力与边界

## 判断

当前 169 的整模型 profiler 数据能够把本次 EP8 target 第 14/39 层的 `DispatchFFNCombine` 关联到各自的提交层和设备任务；**不能从已经保存的单任务时间、外部 HCCL/EVENT WAIT 或聚合 PMU，分解出此内核内 dispatch、跨卡收发、GMM1、GMM2、combine 和等待各自的时间**。这不是说 910C 上绝对没有硬件采集方式，而是现有记录没有建立内部阶段到本次内核/replay 的关联。

CANN 有更细的算子调优能力，**可能**在不更改源码的情况下从原二进制的指令/流水采样观察部分行为；需要实际版本、二进制符号、数据采集和 PC→阶段映射都成立，尚不能承诺能区分全部阶段，更不能仅凭同名安装对象视为历史任务使用的二进制。若需要明确的阶段起止，一种更有把握的研究路线是在隔离诊断副本里对相同内核源码加少量设备侧时间戳、重新编译并采集。它是新二进制、会扰动执行，**不能把测得时间直接当作未插桩基线，也不能预先保证算子输出或并发行为不变**。

## 本机镜像与资料能确认什么

- 169 已保存的 profiler 信息记录 `torch_npu 2.10.0.post7`、`CANN 9.2.0`。只读检查同镜像中固定工具路径：`/usr/local/Ascend/cann-9.2.0/tools/profiler/bin/msprof` 与 `/usr/local/Ascend/cann-9.2.0/tools/msopprof/bin/msopprof` 存在，`/usr/local/Ascend/cann-9.2.0/include/mstx/ms_tools_ext.h` 存在。仅凭文件存在不能确认当前工具能采到指定内核或支持特定运行组合。
- 实际 CANN 9.2 头文件 `/usr/local/Ascend/cann-9.2.0/aarch64-linux/ascendc/include/basic_api/impl/kernel_operator_dump_tensor_intf_impl.h:49–54` 的 `PrintTimeStamp` 在 `ASCENDC_TIME_STAMP_ON` 宏开启时才调用 `DumpTimeStampImpl`；文件 SHA-256 `62be2472cbc203cd7623f29004b54b5da3678dbe09a6de3334432e27126955cc`。这个开关是该源码实现中的**编译条件**，不能假设只改运行时 profiler 开关就给现有内核添加点位。
- 同镜像 `.../ascendc/include/basic_api/impl/kernel_prof_trace_intf_impl.h:26–28` 有 `MetricsProfStart/Stop`，文件 SHA-256 `16a6cf0a5afd94092b71ccc25d344149283b5c230ccd742f6d4e658a3ede8780`；`:30–42` 的 `MarkStamp` 实现受 `__NPU_ARCH__ == 3510` 条件约束，不能将其当作已确认的 910C 可用标记接口。接口存在还需核对诊断 kernel 的实际编译架构和工具版本。
- 华为 CANN 9.1 [PrintTimeStamp 接口](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/debug_interface/onboard_print/PrintTimeStamp.md)列出 Atlas A2/A3 产品支持，并描述按 descId 记录设备侧系统 cycle。它支持的是内核中**已经添加**的点位，不是对任意运行中二进制自动插桩。实际 169 910C 产品归类、工具对当前编译产物的支持和 clock 换算仍须与 CANN 9.2 实例核验；不能把此 9.1 产品表当作已执行的 9.2 验证。
- 华为 CANN 9.1 [msOpProf 用户指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/devaids/optool/docs/zh/user_guide/msopprof_user_guide.md)描述 MC2/LCCL 通算流水、AIC/AIV block、HCCL 及设备侧点位，并提供 `msprof op` 采集案例；案例启用 `-DASCENDC_TIME_STAMP_ON -g` 后重编译。指南也描述 `instrTimeLine` 指令级流水：需要相应指标采集，记录有上限和可能丢失，`-g` 有助于关联源码。不同工具/版本、符号、API 及 kernel 场景的可用性须逐项确认；未观测到的轨道或点位不能视为没有执行。
- 同一指南对重放明确有约束：MC2/LCCL 不支持 range 重放；多卡多算子不支持 application 重放，kernel/range 还可能清除 L2。**不能未经核查拿单卡或默认重放替代多 rank、带 peer 交互的真实运行**，尤其不能重复修改通信状态、假定 replay 和原顺序语义不变。`--warm-up` 对 MC2 不生效的文档约束也应核对实际 9.2。
- 华为 CANN 9.1 [HCCL profiling 说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/API/hcclug/docs/zh/user_guide/perf_analysis/profiling_op_behavior.md)说明可采独立集合通信及其 notify、传输事件，Level1 可显示更多编排。当前源码使用 `HcclShmem` 和 peer 内存访问，而不是已经证明这些访问会按标准 HCCL 算子呈现；独立通信事件有其各自 identity，不能仅用同窗或同 rank 指认为 MC2 内部阶段。
- `torch_npu` 的 `msprof_tx`/MSTX 适合标记 host/设备外层代码区域及提交链，不能凭一个外层范围把复合内核内部自动拆开。整体 task 的 AIC/AIV/MTE 比率、周期及 CSV `Wait Time` 也没有被证实为这些内部段的分子、分母或 flag 等待。

## 能定位的源码位置（不是历史执行证明）

冻结源码 `sources/vllm-workspace/vllm-ascend/csrc/mc2/dispatch_ffn_combine/op_kernel/dispatch_ffn_combine_kernel.hpp` SHA-256 `7dcef4c08c4c47e38bdd01aef0dee997798658ee4d7b871c8e6c70f7ffe596ba`：

- `:218–230` 显示 AIC 分支的 `GMM1`、`CrossCoreWaitFlag`、`GMM2` 和 AIV 分支 `DispatchAndCombine`。这几个分支并发协作，不能把其区间简单相加。
- `:799–811` 有路由量化、既有 `SyncAll` 和跨 rank 同步及 expert token 汇总；`:846–866` 包含 peer buffer、`CopyGMToGMPerToken`、既有同步和向 GMM1 发信号。它们是源码位置；当前 profile 没证明实际执行了同一版本的每个分支。
- `:451`、`:590` 的 AIC 跨核等待，`:920–954` 的 AIV 等待/激活与后续信号，`:958–971` 的 combine、清理与 cross-rank sync 是候选标记边界；`:771` 还有轮询等待。GMM、复制和同步可能在不同核、group 和 pipeline 上重叠，某个核的区间不自动等于全核或整层墙钟。
- 现有 `mc2-stage-field-audit.json` 列出了普通 kernel CSV 的字段和源码线索，但无已确认逐内部阶段的字段。安装对象的 SHA/tilingKey 曾被记录，然而历史任务没有绑定到该对象；因此上述源码与采集事件间的二进制身份仍是待补前提。

## 最小隔离诊断设计（尚未实施）

1. **先评估无源码修改手段。** 在已有容器的隔离诊断副本中，确定实际加载的原 kernel 二进制 SHA、架构、编译调试信息、tilingKey 和单次调用的 rank/step/stream/task 身份；核验 CANN 9.2 `msprof op` 对这个多 rank SHMEM kernel 的支持、采样范围、重放行为和工具输入。若原二进制能提供可映射的 PC/指令流水，可先只报告能闭合映射的片段；没有证据时标 UNKNOWN，不将其宣传为完整阶段计时。该步骤也是未来可选实验，不在本次执行。
2. **需要明确阶段时，复制源码和构建产物到专用诊断目录，编译插桩副本。** 原应用和其他容器的安装路径保持原样；仅在诊断容器通过独立库/算子加载路径选择新对象。保存源码、编译命令/选项、诊断二进制和原二进制哈希、软件版本、真实已加载路径与 tilingKey。启用本镜像实际存在的 `ASCENDC_TIME_STAMP_ON` / `PrintTimeStamp`，并先核实该 9.2 构建链和多卡 MC2 采集兼容性。不要假定 `MarkStamp` 在本架构可用。
3. **分角色、分块、成对记录稀疏标记。** 为 AIC 的 GMM1、显式 cross-core wait、GMM2，以及 AIV 的 routing/跨 rank 汇总、peer dispatch、既有 wait、SwiGLU、combine 和终末 cross-rank sync 各设计开始/结束 descId。优先少量选择有代表性的 core/rank/group，再扩大覆盖；每条结果必须保留 rank、core 类型/ID、descId、group/epoch 和对应调用标识。标记应插在已有明确语义边界，严格区分“发出异步拷贝/计算”与“已有同步保证完成”；**不要为了计时新增同步或拆成几个 kernel**，否则会改变通算重叠，所得是另一条执行路径。若现有 API 不携带 group/replay，需单独定义无敏感数据且不改变同步的关联办法；做不到则标记不完整。
4. **输出与行为验证。** 固定等价的输入、量化 scale、路由结果、通信组和参与 rank，在不暴露用户 prompt/权重内容的诊断场景比较原版与插桩版输出、expert count、异常与通信完成情况；先确认所需数值契约再选择逐位或容差比较。记录插桩点数、采样丢失、构建差异和是否改变资源/调度；比较插桩开关、标记密度的影响。没有验证通过前不能称输出一致或把诊断耗时外推。固定张量值/路由的读取和保存本身另需受控设计，不在本次执行。
5. **时间解释。** 优先每 rank、每 core 的本地时钟差和各区间并集；跨 rank 直接相减必须有验证过的时钟同步与误差界。区间重叠可反映通算流水，既有 wait 的前后点只能衡量指定执行线程在该边界观察到的区间，不能等价认定对端通信耗时或设备完全空闲；若同步语义和标记位置不能证明，给出不确定性。

本次仅查询文档、固定镜像文件及已冻结源码，未修改目标实现、启动 profiler 或运行 NPU。

## 169 CANN 9.2 安装版本 help 的直接核验（只读，未采样）

本机 ZCode 通过 169 的个人 stage 容器执行 `msprof op --help`，返回码 0，完整原文 `diagnostic-20261029/msprof-op-help-9.2.txt`，SHA-256 `a0b7ef0b45194a791d4a2c637ceaedb6ec5c79997898cd042f914807c81a9760`。它显示实际工具的 `--replay-mode` 可选 application、kernel、range，**默认 kernel**；也列出 `TimelineDetail`、`Source`、`Roofline`、`--kernel-name` 等选项。这只证明命令行选项存在，不证明 910C、多 rank SHMEM、被测二进制的适配、源代码映射和采样结果有效。它的日志曾尝试打开只读 `/root/ascend/log/...` 并报告失败；不能据此声称完全无写入尝试。没有执行任何采集或 NPU 负载。由于默认 replay 对跨 rank 通信的安全性和等价性未知，不能直接套用默认命令去分析正在运行的服务。

另一个采集盲区：已保存 `msprof-application-help-9.2.txt` SHA-256 `8730c0e494f25711cc4d94b61bb55f7fa9e2bd2b0948d72f4f20a8e20cbeb098` 显示 sample-based `--aic-freq` 范围 1–100 Hz。以最高 100 Hz 理想化为每 10 ms 的瞬时周期采样，EP3 的已选两个摘要各 40 层 MC2 记录分别为 1.536–4.183 ms 和 2.237–9.574 ms：80 条都短于一个名义周期，均存在采样相位使单次任务不被命中。实际采样可能为窗口或频率抖动，工具能否应用于该多 rank 自定义 kernel 也未知；这只是提示不可把 sample-based 未命中当作无活动，不说明旧 task-based PMU 不可用，也不能据此判定任何内部阶段的时间。

## 追加：现有安装对象与具体任务的身份缺口

只读核对当前自有 169 stage 容器的固定目录 `/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910_93/dispatch_ffn_combine`，限定名称 `DispatchFFNCombine_<32位hex>.o/.json`，共三个对象及对应 JSON。对象 SHA 与各自 JSON 的 `sha256` 字段一致：

| 对象名称中间标识 | 对象 SHA-256 | JSON SHA-256 | JSON 中 tilingKey / kernelType | 静态输入签名摘要 |
|---|---|---|---|---|
| `17df546b46e1ea56007584fac7e6f429` | `4cf1474cb1009b96fc743f5a396a9c07a6980146c41818b60cdbf0d3453a43a9` | `860503a22b09e59fc05a99829ebc7577e4bfa2499ea44ca0761a4c7a2211f2a7` | `1000010 / MIX_AIC` | BF16 输入/输出，INT8 权重声明 `FRACTAL_NZ` |
| `de54a9f3b8398c214d5562aa327d0e29` | `d88acd30e3b56893952a9e7094dc4c2c45ceed08942383f9e9d4c9dd3e2260b5` | `66bf444f2e150f95097a996b58bdc8ac5254ebff77700986b172da1032fa4a1c` | `1000010 / MIX_AIC` | BF16 输入/输出，INT8 权重声明 `ND` |
| `fc6ca5e719791250fe17f077893fe681` | `a0aec99d7145b29bb226231ff24d751d0798ff5ead7a2be84cc5db0632797979` | `68d82c1302ecb6ec4e5094a0f297e034aba6e326fea563a2a03131e1b0445d42` | `1000010 / MIX_AIC` | FP16 输入/输出，INT8 权重声明 `ND` |

这只证明当前目录中存在三套不同文件及其自述 metadata；**三个对象拥有相同 tilingKey，因此只捕获 `1000010` 仍不能唯一确定对象**。上述静态 `format` 声明也不是已证的运行时选择契约，不能仅以签名排除其他加载来源或将某个对象强绑定任务。此前 `binary-path-audit.json` 中的 `17df...` 安装对象与当前同名文件 SHA 一致，但这个一致性不证明历史进程实际加载该对象。

本次新 EP8 的第 14 层 task 44217 和第 39 层 task 45363，根据已冻结的 `kernel_details.csv` SHA `20cfb97abe03befda3fbf2c1194e48531aac44ce17d4f5fc130ea8ffb5261730`，两条均报告 `DT_BF16;INT8;INT8;INT32;INT64;INT64;FLOAT;BOOL`，输入格式 `ND;FRACTAL_NZ;FRACTAL_NZ;ND;ND;ND;ND;ND`、输出 BF16/INT32。它们与 `17df...` 的安装声明相容，但没有证明其就是被加载的文件；新 trace 的已检查设备 X 记录提供 `Model Id`、`Task Id`、`Physic Stream Id`、`Task Type` 等，已关联到各自 CSV，仍不提供编译对象 SHA 或已绑定的完整 kernel symbol/tiling 数据。

最小可区分证据是：在同一诊断进程的真实加载/注册阶段记录实际对象的**内容 SHA 或可唯一映射的 binary handle 与内容**、完整函数符号与 tilingKey；再通过经验证的图节点/launch 关联到同 rank 的 stream/model/task、目标调度摘要和每次 replay。句柄的地址单独不是永久身份，须考虑注册/复用与 graph replay。若通过文件加载，应核验加载时内容与哈希；若从内存加载则要记录受控哈希与对应句柄，不能把现在路径存在的文件当历史加载证明。厂商 [自定义算子加载说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/others/acldevg/runtime_doc_dev_0019.html)描述了从文件或数据加载、按函数名取得 handle 和异步 launch 的一种 API 路径；**尚未证明当前 vLLM-Ascend 实例走该公开 API 路径**，也不能假设图模式每次 replay 都再次调用 host launch。应先检查该版本是否已有可关联的日志/符号/导出；必要的新取证只在受控诊断副本内设计，避免触及用户数据和改变计算结果。本次只读目录及两条已保存 CSV，未做新设备工作。
