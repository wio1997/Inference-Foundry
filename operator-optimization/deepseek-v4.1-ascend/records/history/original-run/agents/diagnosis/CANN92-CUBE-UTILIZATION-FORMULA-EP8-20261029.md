# CANN 9.2 源码和 EP8 历史采集：cube_utilization 的口径

只读分析已保存的 EP8 采集及 169 stage 容器已安装文件，未运行 NPU 负载。此结果避免把 `cube_utilization(%)` 误作 Cube MAC 算力利用率；不推断原事件的硬件根因或新的理论上限。

## 实证

历史采集 `diagnostic-20261029/runtime/diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_737_20261029023525648_ascend_pt` 的 `PROF_000001_20261029023525659_00000737LBADKQBH/mindstudio_profiler_output/op_summary_20261029024048.csv`（SHA-256 `3088d48149dc0e97b5876f5822776ba9a69a968ddd326fb2d776c9a829bb2f1b`）和 `ASCEND_PROFILER_OUTPUT/kernel_details.csv`（SHA-256 `20cfb97abe03befda3fbf2c1194e48531aac44ce17d4f5fc130ea8ffb5261730`）中，stream 47、task 44217 与 45363 各唯一行，两个文件中的时长、周期、比例及此字段相同：

| scope | task | duration μs | aic_total_cycles | aic_mac_ratio | cube_utilization(%) |
|---|---:|---:|---:|---:|---:|
| 14 层 | 44217 | 160365.956 | 6788541718 | 0.003 | 97.990 |
| 39 层 | 45363 | 8290.414 | 335281950 | 0.164 | 93.616 |

层归属来自已存同线程 scope → flow → device task 关联，不是从 task ID 或 shape 猜测（`diagnostic-20261029/target-layer-mc2-16rank.json`）。两任务均报告 `MIX_AIC`，24 个 Block / 48 个 Mix Block。

同批 `PROF_.../device_8/info.json.8` SHA-256 `32bb376ca72cbd6505af361d3e9f4d0e87a17a0985ee52f5e76c472a445fc829`：`cannVersion=9.2.0`、`DeviceInfo[0].aic_frequency=1800`、`ai_core_num=24`。代入下方安装源码公式，以 CSV 已舍入 μs 时长计算，分别为 `round(100*6788541718/(1800*24*160365.956),3)=97.990` 和 `round(100*335281950/(1800*24*8290.414),3)=93.616`。这是对这两行的数值一致性核查，不是证明历史执行器一定加载了该安装副本；显示时长已经舍入，匹配不足以证明所有小数及边界规则。

## 本机同版源码

路径根目录 `/usr/local/Ascend/cann-9.2.0/tools/profiler/profiler_tool/analysis`：

- `viewer/runtime_report.py:265–282`，SHA-256 `2a979d6674486f733d187396805670e828dd58754b7453b0a034edb85d6a7fcb`：`cube_usage` 仅用 `mac_ratio` 做 NA/零条件判断，并不把它乘入结果；非零分支为 `round(100 * total_cycles / (aic_frequency * ai_core_num * task_duration_ns) * 1000, 3)`。频率注释为 MHz、任务时长 ns。
- `common_func/data_manager.py:53–86`，SHA-256 `f02b37be762a8c9cbe31e3764ad6f7ffb7e497bea317825598b620c917c55221`：从 `InfoConfReader` 读取频率和核数，把函数结果以 `cube_utilization(%)` 为表头加入数据。`common_func/ms_constant/number_constant.py` 的 `NS_TO_US=1000.0`，SHA-256 `2fae295da0237456578cb966235fb2546f7ac4cec57c046e1f4e684354604490`。
- `viewer/ai_core_op_report.py:248–261,399–427,462–475`，SHA-256 `74d7c67a29fdc74802b46f9bfad3bdc1ba078bfe6ba966c8c1d17f47425802e2`：先生成该字段，再格式化原本 ns 的 task duration 为 μs。`msinterface/msprof_export_data.py:_get_op_summary_data` 指向该报告实现。已存 `ratio-export-source-excerpt.json` 保存 torch_npu 的 `_kernel_view_parser.py` 源码（SHA-256 `d8c4c75c6ce95726726b20399cf05cd752be22d730ae49117ca9f2f4db0261fc`），其从 CANN op_summary 生成 kernel_details；但相同采集字段与源码算术一致不等同于已绑定历史代码执行及每一步数据来源。
- `mscalculate/stars/ffts_pmu_calculator.py:398–423,464–555`，SHA-256 `72b65a386260f79b34ab1894e6ba4787616a62584c988272195d4a9a6909f66d`：对于该实现的 MIX 解析路径，主核 total_cycle/PMU 来自 context，另一核侧来自 block PMU 分组求和；block 模式还按从核类型过滤。`MIX_AIC` 标签与该路径相容，但尚未独立证明这两条历史数据经此路径、主核 context 的底层聚合机制或数据未丢失。

因此这两条 97.990%、93.616% 与 MAC ratio 的 0.3%、16.4% 不矛盾：前者在此实现和观测样本下符合按设备信息频率、核数和任务时长归一化的周期字段，不能把它当作有效矩阵乘运算的百分比；后者仍需结合其统计分母、精度、任务覆盖及相应硬件指标解释。不能用两个数字相减或把它们直接推到整片芯片、无 profiler 的生产负载及具体根因。

## 复核命令和限制

以下命令在已有 stage 容器执行，只读文件，不调用设备：

```sh
BASE=/usr/local/Ascend/cann-9.2.0/tools/profiler/profiler_tool/analysis
sed -n '265,282p' "$BASE/viewer/runtime_report.py"
sed -n '53,86p' "$BASE/common_func/data_manager.py"
sed -n '248,261p;399,427p;462,475p' "$BASE/viewer/ai_core_op_report.py"
sed -n '398,423p;464,555p' "$BASE/mscalculate/stars/ffts_pmu_calculator.py"
sha256sum "$BASE/viewer/runtime_report.py" "$BASE/common_func/data_manager.py" "$BASE/viewer/ai_core_op_report.py" "$BASE/mscalculate/stars/ffts_pmu_calculator.py"
```

文件中 `cannVersion`、安装路径、相同源码 hash 与数值一致支持口径判断，不证明历史解析器二进制与源码关联、频率是实际瞬时频率或硬件标称峰值。两行上可重复的周期归一化不能回答内部等待、数值计算量、通信重叠或采样扰动大小。硬容量与全局吞吐硬上界仍缺少版本匹配的权威单位、时钟域、资源粒度与物理约束，维持 UNKNOWN。
