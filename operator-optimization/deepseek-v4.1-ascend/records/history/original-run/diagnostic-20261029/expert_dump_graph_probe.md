# 独立验证：24 个 INT32 值的 dump 与图回放

**这是未运行的诊断草稿，不是已验证的采集工具。** 当前 C++ 版本（SHA-256 `97ef3d6c4b777a1f9944d5e5804bbcd2a378df025106e45806e8b98044f7d065`）已在 stage 容器用 CANN 9.2 头文件、`libascendcl` 和 `libnnopbase` 编译链接通过；未执行 NPU。不访问真实服务、模型、提示词、路由 ID 或权重，不修改推理代码。它用于判断目标 CANN 是否能只 dump 一个输出，以及图内 dump 是否在 replay 时逐次反映当前数据。成功也不能证明真实 MoE 的 tensor 生命周期、跨流依赖或性能影响。

调用方须在 169 自有隔离 stage 容器内，先核实设备编号映射和全部设备进程归属；本程序**不会检查设备占用或建立容器隔离**。若与本人服务共享芯片，须确认没有其他人的进程、本人服务运行中和等待中请求为零、计算活动处于低负载并检查显存余量。瞬时空闲不能保证后续无请求；独立进程仍可能带来短时资源扰动。不得注入服务或影响其他用户的任务。

程序默认拒绝运行。仅待环境和设备核实后，准备位于本任务 `diagnostic-20261029` 内的**已存在且为空的独立子目录**，再从已核实的 stage 容器中执行：

```sh
python3 expert_dump_graph_probe.py --mode eager --execute-isolated \
  --checked-idle-device-and-isolated-container \
  --device-id <经核实的逻辑设备号> --output-dir <已存在的空目录>
```

若经核实设备仅由用户本人项目服务占用、服务暂时无请求且用户允许共享，可将 `--checked-idle-device-and-isolated-container` 换为 `--checked-own-idle-service-and-no-foreign-processes`。这两个标记均为调用方声明，程序不独立核验；应保存执行前后服务指标并在出现其他用户进程或异常时停止诊断，不停止服务。

程序以清空后的环境启动独立子进程，工作目录设为隔离目录，并将 `TMPDIR`、`ASCEND_DUMP_PATH`、`ASCEND_WORK_PATH`、`ASCEND_PROCESS_LOG_PATH`、`ASCEND_CACHE_PATH` 和 `NPU_COLLECT_PATH` 指向该目录，同时开启日志打屏；不继承 `ASCEND_DUMP_SCENE` 和 `LD_PRELOAD`。这个限制针对已知的应用输出路径，并不证明驱动或系统级日志不会写入其他位置。不会删除文件或重置设备。程序使用 `ASCEND_RT_VISIBLE_DEVICES` 限定一个逻辑 ID，再以进程内序号 0 调用 ACL；[CANN 9.2 文档](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta1/API/runtimeapi/aclcppdevg_03_2286.html)描述了这种重映射，但容器内逻辑 ID 对应的物理芯片及空闲状态仍须在调用前核实。

程序在隔离目录编译 C++，通过 `aclmdlInitDump` / `aclmdlSetDump` 开启输出 dump，随后构造固定地址的 24 项 INT32。`--mode eager` 只执行一次普通调用（1000–1023），结束后必须解析输出并人工确认全部 24 项，再决定是否进行下一步。它不会自动开始 graph。

确认 eager 数值后，重新检查设备空闲状态，并使用**另一处已存在且为空的隔离子目录**，用相同参数将 `--mode eager` 改为 `--mode graph --eager-values-verified`。图模式执行一次图捕获及三次 replay（2000–2023、2100–2123、2200–2223）。`--eager-values-verified` 是调用方的确认，启动器不会自行验证 eager 的数值或验收状态；调用方必须先完成该验证。`inputTensorNum=0`、图捕获与逐次输出是否受支持均为**未知**；任何 API 返回失败都只说明该尝试未成功。捕获期间不做 stream/event/device/context 同步，不自动重试。

日志中的 `EXPECTED` 和 `API_CALLS_COMPLETE` **不是通过**。stage 的 CANN 9.2.0 已安装 `tools/operator_cmp/compare/msaccucmp.py`；其 `convert --help` 提供 `-d`、`-out`、`-v` 和 `-t npy`。实际产生的 dump 格式和转换结果尚未验证；如果证实是版本 2，可以针对隔离目录中的**单个 dump 文件**调用：

```sh
ASCEND_WORK_PATH="$probe_out" ASCEND_PROCESS_LOG_PATH="$probe_out" \
ASCEND_SLOG_PRINT_TO_STDOUT=1 PYTHONDONTWRITEBYTECODE=1 \
python3 /usr/local/Ascend/cann-9.2.0/tools/operator_cmp/compare/msaccucmp.py \
  convert -d "$dump_file" -out "$probe_out" -v 2 -t npy
```

`probe_out`、`dump_file` 必须由调用方指定为本次隔离目录及其中的合成数据文件，不要对其他目录做批量转换。再用 NumPy `np.load(path, allow_pickle=False)` 核对数据类型为 INT32、形状为 `(24,)`、每项数值正确：eager 为 1000–1023；三次 replay 分别为 2000–2023、2100–2123、2200–2223。还必须核对文件与各次 replay 一一对应，无覆盖、漏写、重复或无法归属。文件存在、文件数量或返回码为零均不足以判定成功。若捕获失败、输出无法解析或回放无法绑定，记录失败/未知并停止。

代码参考 [aclDumpOpTensors](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta3/API/aolapi/operatorlist_00178.html)、[Dump 配置](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_0315.html)及 [图捕获限制](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/920beta1/others/acldevg/aclcppdevg_000045.html)。必须先在目标容器核对头文件、链接和参数，不将文档示例当成本机实测。
