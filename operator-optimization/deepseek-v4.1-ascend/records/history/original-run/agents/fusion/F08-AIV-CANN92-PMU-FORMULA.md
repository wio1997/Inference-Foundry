# F08 AIV：当前 CANN 9.2 源码的周期到时间计算，及其适用边界

本核查只读已安装 Python 源码和先前 EP3 聚合记录，未运行 NPU，也没有修改目标实现。当前 stage 容器的源码同属冻结镜像，但未证明原采样时实际加载了相同文件或目标 DB 走了该具体处理路径。

## 找到的源码公式

当前容器 `/usr/local/Ascend/cann-9.2.0/tools/profiler/profiler_tool/analysis/`：

* `common_func/utils.py:175-181` 的 `Utils.cal_total_time`：`round(total_cycle * 1000 * NS_TO_US / freq / block_num * ceil(block_num / core_num), 2)`；任一 `freq`、`block_num`、`core_num` 为零时返回 0。`common_func/ms_constant/number_constant.py:70` 中 `NS_TO_US = 1000.0`。若频率以 Hz 输入，则这个函数的时间换算系数为 `10^6`，计算值按微秒理解。
* `common_func/info_conf_reader.py:218-232` 从 info 文件读取 AIC 指定频率，并乘 `FREQ_TO_MHz`，源码 `pmu_calculator.py:98-111` 另说明动态采样频率从 MHz 转 Hz 并能覆盖默认值；注意这里默认读取的是 AIC 频率，不能只依据样本的 AIV 字段宣称实际输入。
* `mscalculate/stars/ffts_pmu_calculator.py:380-389` 选择对应 core 数与 block 数，使用默认或动态频率后调用此函数；`:315-371` 的非 MIX 逻辑明确使用 context PMU，并写出 total_cycle 和 total_time；`:391-426` 的 MIX 逻辑区分主核 context 与从核 block PMU。`mscalculate/aic/aic_calculator.py:105-109` 也调用同一函数。这只证明这些代码路径各自的行为，不证明 EP3 原始 DB 使用其中哪条。
* 同版 `mscalculate/stars/chip6_pmu_calculator.py:438-472` 对非 MIX 和 MIX 主核采用 `duration_from_syscnt(end-start)`，MIX 从核另走周期公式。因此不能将上述算式用于所有任务和硬件路径。

源码 SHA256：`utils.py` `b5bf3354e36ed1e099628857881b506e8f5b4965507bf5cbe4e89c87c847d99e`；`ffts_pmu_calculator.py` `72b65a386260f79b34ab1894e6ba4787616a62584c988272195d4a9a6909f66d`；`pmu_calculator.py` `2b39936005b5ee3905ce6b675002d5893d0c1f53955393d34032b76fd12d7229`；`aic_calculator.py` `50f3fca9dc72aea390488418ca84cf3112a6b8c948e7203417d0112e2cf6be17`；`chip6_pmu_calculator.py` `e8e830708a7d2f572b5ef5e12932b84f8a1b8326faf61596b0ca8e7ce28c35cc`。

## 有条件复算和未证实之处

已有 `f08-counter-binding-audit.json` 显示 EP3 样本的 blockNum=48，设备 metadata aiv_num=48、aiv_frequency=1800。**假设**上述 FFTS 周期公式实际使用 block=48、core=48、频率=1.8 GHz，1,146,886 和 1,268,528 周期分别得到 13.27 和 14.68 μs（保留两位）；它们恰与 DB 的 13,274.1435185 和 14,682.0370370 数值除以 1000 再保留两位吻合。源码公式有两位小数，原 DB 值有更高精度，无法直接作为同一个生成路径的证据。

尚未绑定原 `TASK_PMU_INFO.aiv_total_time` 的生产/导出路径、默认 AIC 与 AIV 时钟是否相同、动态频率采样是否参与，以及原始字段单位、block 计数与硬件真实活跃覆盖。不能将 DB 值认定为纳秒、不能将该代码的条件计算视为两个任务实际执行或等待时长，更不能用时间差归因。即使可证明该式已使用，其 `ceil(block/core)` 是导出器的数学换算，不直接证明这些核心同时活跃。

参考：官方 [CANN 9.0 msprof op 文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/devaids/optool/docs/zh/quick_start/installation_guide.md) 定义逐 block `aiv_time(us)` 与 `aiv_total_cycles`，但与本次 9.2 TASK_PMU_INFO 汇总字段的版本及数据形式不同，不能作为该字段单位或聚合公式的直接证明。
