# Trace / Profiling：有限采集、离线汇总、局部读取

本流程补充 PROTOCOL.md 的 P4 数据准备，并为 P5 性能对比、P6 原因定位提供证据。先复用已有采集；缺证据时准备最小诊断采集计划。当前包没有绑定真实硬件，也没有执行采集。采集属于诊断取证，不是算子优化。

## 1. 先确定需要回答的问题

使用 [采集计划模板](templates/profiling-plan.json) 记录目标模块/Case、实际源码与环境版本、输入、设备/rank、阶段、需要的指标、采集范围和停止条件。采集命令必须匹配目标环境安装版本，不直接套用最新版网页参数。

- 基线：同 Case 的无 profiler 耗时/吞吐、原始样本、计时边界与同步规则。
- 归属：模块→框架算子→kernel 的关联，包含嵌套层级、完整模块路径及同类模块的不同实例。
- 利用率：所选主指标的活动/核心时间分子和对应分母；只有 kernel 耗时不能自动得到利用率。
- 原因：区分等待、访存、通信、负载不均等所需的具体事件或硬件指标。

沿用会话已有授权，不重复确认已授权动作。新的设备采集仍需已明确的主机/容器、运行入口和可用设备范围；仅有本地工作包不代表获得了服务器执行权限。诊断标记只放在独立运行入口或临时副本中，不改算子语义、不部署。没有权限或数据时继续离线分析并记录缺口。

## 2. 分层采集，按证据升级

| 层次 | 采什么 | 何时结束/进入下一层 |
|---|---|---|
| A：无 profiler 基线 | 同 Case 正式计时与稳定性样本 | 得到可比基线；缺基线时只能做带条件的性能对比 |
| B：短窗口概览 | CPU/NPU 时间线、step、kernel/API 时间和已存在的模块关联 | 覆盖所需阶段/分支且足够判断下一项缺失证据 |
| C：模块归属或定向指标 | 缺失的模块范围标记；必要的引擎、通信、shape/FLOPs 等 | 只为当前待证原因采所需类别，证据足够即停止 |
| D：离线汇总与局部提取 | 全范围聚合表、小摘要、指定模块/窗口明细 | 支持定位或明确保留未知；不自动进入解决方案 |

采样设计的默认起点是：完成模型加载、编译、缓存建立等应用预热后，采 3 个稳定迭代，repeat=1；这是本包的初始诊断预算，不是统计充分性保证。短程序可改用明确的 start/stop 窗口。长服务的一个 step 若包含很多请求，必须同时限制请求数/窗口时长，不能认为 active=3 必然很小。

prefill/decode、不同 shape、动态分支和不同阶段分别记 Case。只采一个代表 rank 的结果只能支持该 rank；通信/负载不均需要覆盖相关参与者，记录跨设备时钟对齐依据。缩小采集范围不等于缩小最终结论所需的覆盖，缺少覆盖就明确降级结论。

先定有限 active/repeat、时长预算、输出容量与磁盘余量；按采集器实际支持能力落实停止机制。空间预警时请求 profiler 正常停止并保留完整部分及日志，不随意杀死用户作业。若工具只能按 step 结束而不能硬限制文件大小，要明确记录该限制。

## 3. Ascend PyTorch 的短窗口配置示例

官方 API 支持 schedule 与 prof.step() 配合限制采集轮次，并可关闭额外的 shape、内存和调用栈收集。参数及默认值应查对应安装版本。[Ascend PyTorch Profiler 文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0033.html)

下面是嵌入已授权诊断入口的示意，run_one_step 和 profile_output_dir 由实际任务提供；它不是本机可直接执行的采集程序，也不自动建立模块归属。

```python
# 先完成应用自身预热，并选定当前进程应使用的设备。
import torch_npu

plan = torch_npu.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
config = torch_npu.profiler._ExperimentalConfig(
    profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
    data_simplification=False,
    export_type=[torch_npu.profiler.ExportType.Db],
)
with torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU],
    schedule=plan,
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_output_dir),
    experimental_config=config,
    record_shapes=False, profile_memory=False,
    with_stack=False, with_modules=False, with_flops=False,
) as capture:
    for _ in range(5):  # 1 wait + 1 profiler warmup + 3 active
        run_one_step()
        capture.step()
```

profiler warmup 不替代模型的应用预热。明确 repeat=1，避免使用 repeat=0 导致随长作业持续循环。不要为了缩短文件而禁掉问题所需的 CPU 活动、通信参与者或依赖事件。上述配置仅作概览；Level0 不保证包含模块归属、Cube/Vector 利用率、FLOPs 或完整通信诊断指标，缺什么就记录并按 C 层补证。

正式基线与诊断采集分别保存。记录 profiler 开销；不要把 profiled 时间直接替换无 profiler 基线，也不要把全局开销比例机械应用到每个模块。只有 profiled 模块计时时，对比注明扰动未知或已评估。

## 4. 模块归属与利用率需要不同证据

先检查已有模块范围、关联 ID、step/rank、时间戳及算子/kernel 映射是否完整。只凭 kernel 名称或 host 时间窗不能可靠分配异步 NPU 工作；范围裁剪须保留跨窗口发起的依赖和相应关联记录。

支持的 msprof-analyze 版本可使用 module_statistic。官方准备条件包含模型级 MSTX 标记、对应采集开关及 DB 输出；with_modules=True 单独不能保证归属完整。按必要模块层级选择标记，避免给所有叶子调用逐个打点。输出有模块层级、算子/kernel 关联和耗时统计。[module_statistic 官方说明](https://github.com/Ascend/msprof-analyze/blob/master/docs/en/advanced_features/module_statistic_instruct.md)

在确认本地版本支持且数据满足前提后，可对分析工作副本运行：

```sh
msprof-analyze --version
msprof-analyze --help
msprof-analyze -m module_statistic -d ./profile_work -o ./profile_analysis --export_type db
```

不同版本 CLI 和 MSTX 参数存在差别；对照安装版本的帮助及文档选择，不猜参数，不默认安装/升级宿主软件。缺模块标记时先记 UNKNOWN；有采集授权再用临时诊断入口补齐。

module_statistic 的 kernel 耗时和不是模块窗口长度，不能直接除以任意窗口当利用率。新版 operator_mfu 是另一个分析功能，依赖支持算子的 FLOPs、dtype/shape、峰值和所需采集信息；模块 MFU 还需要模块范围。它不能替代设备忙碌率，也不自动证明本包要求的严格算法域上界。按实际版本核对所需开关和支持算子，不因某算子没有 MFU 值就填 0。[operator_mfu 官方说明](https://github.com/Ascend/msprof-analyze/blob/master/docs/en/advanced_features/operator_mfu_instruct.md)

## 5. 原始 profile 很大时怎样处理

原始 trace 保留在数据所在机器，存路径、版本和哈希；不把 GB 级 JSON 或完整 CSV 直接输出给模型。读取顺序固定为：文件清单/元数据 → 全范围聚合表 → 有界摘要 → 指定模块的局部证据。

1. **盘点而不展开。** 只列已授权采集目录的文件类型、大小、rank/step、工具版本及解析状态。原始数据、解析产物、日志、基线分目录；解析放工作副本，避免覆盖已有导出。
2. **优先官方离线解析和 DB。** 按版本使用 torch_npu 的离线 analyse 或现有解析结果。常见文本导出包括 kernel_details.csv、operator_details.csv、step_trace_time.csv 和 trace_view.json，是否生成取决于配置及场景；字段名和单位先读 schema/表头核对。[官方输出目录说明](https://www.hiascend.com/doc_center/source/en/canncommercial/800/devaids/profiling/atlasprofiling_16_0035.html)
3. **先聚合全部有效数据。** 按 Case、rank、step、模块和算子保存全量统计，保留未归属项。跨 stream 的耗时和不能当经过时间；忙碌时间用适用的区间并集，核心活动按核心时间定义。不能只保留最慢 Top-N 后声称已找到利用率最低模块。
4. **生成有界摘要。** 首读摘要默认最多 32 KiB；分别展示利用率最低组、耗时最高组、长等待/大空洞和未覆盖项。显示总行数、完整索引和下一页位置；截断只影响展示，不丢聚合数据，不以摘要上限限制扫描覆盖。
5. **带条件取局部证据。** 指定 Case/rank/step/module 和时间区间，提取相关 kernel、提交/完成、依赖/通信事件及必要前后文。单次展示默认不超过 200 行或 32 KiB，超出就分页。保留原 event/关联 ID 和查询条件，裁剪片段不能当完整因果图。
6. **大文件处理保持有界。** CSV 用逐行读取；DB 用只读查询、过滤列和分页，先查实际表结构再写 SQL。JSON 使用环境已有的流式解析器/官方解析器，不能无界 json.load；缺合适工具就记录解析缺口，不擅自安装或用头尾几行代替完整统计。gzip 只减磁盘体积，不代替流式处理。

支持的 msprof-analyze 版本还提供计算/API 汇总、空闲区间和 DB 分析能力，可按当前证据需求选用；先核对所需表、采集级别与 CLI，避免全开功能。[官方分析能力与参数](https://github.com/Ascend/msprof-analyze/blob/master/docs/en/advanced_features/README.md)

## 6. 固定交付与恢复入口

以下是需要生成的任务产物，文件名不是声称厂商工具会自动输出这些文件。本包提供流程与模板，实际转换/查询需依据收到的真实 schema 执行。

| 产物 | 内容与用途 |
|---|---|
| profiling-plan.json | 采集目的、版本、有限范围、预算、授权及基线对照 |
| profile-index.json | 原始/解析文件路径、大小、哈希、schema、覆盖、处理命令及日志 |
| profile-summary.json | 模板规定的小摘要、最低利用率/高耗时/等待组、缺口及完整表入口 |
| module_metrics.csv 或 DB 表 | 全部模块统计，包含口径、覆盖和未归属项；不只留 Top-N |
| operator_metrics.csv 或 DB 表 | 算子/kernel 统计及到模块的映射 |
| evidence/ 中的局部片段 | 按问题提取的事件、来源 ID、过滤条件及裁剪边界 |

报告和 STATE 只链接摘要与索引；恢复时先读这两项，再读被点名的局部证据。每个结论都能回到原始数据和精确查询条件。采集失败、丢事件、缺 rank/step、解析错误和采样不足单列；缺失不填零。

数据足以定位或明确无法补证后停止。当前阶段交付原因、证据及未知，不执行工具自动建议中的修复或调优操作。
