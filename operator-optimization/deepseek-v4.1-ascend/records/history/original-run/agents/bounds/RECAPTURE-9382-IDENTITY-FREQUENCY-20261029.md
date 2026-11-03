# 169 芯片身份与频率硬界：反例和可复现条件

## 结论

现有证据支持 **某次 169 板卡查询对象**的 `Chip Name=Ascend910`、`NPU Name=9382`，以及历史 16 条 profiler 元数据各自记录的 `ai_core_num=24`、`aiv_num=48`、`aic_frequency=1800`。厂商 A3 的型号命名规则下，前者与 `Ascend910_9382` 匹配；这不是 16 条历史 profile 每一行各自的硬件 runtime SKU 证明。CANN 安装配置可提供相容的计算模型输入，但**本轮未找到把 1800 MHz 证明为该板卡所有时刻、所有状态的不可超越频率上限的资料**，也未证明相关资源域封闭，故不能得到 169 数值硬算力或 TPS 上界。

相关已保存原始证据和作用范围见 `agents/bounds/vendor_sources_9382.md`、`agents/bounds/int8_ceiling_certificate.md` 和 `evidence/profile-device-capacity-info.json`；尤其不可把单个 board0 的型号查询泛化为其他 15 条逻辑设备的 SKU。

## 真实安装配置的反例

只读个人 stage 容器当前 CANN 9.2 的 `/usr/local/Ascend/cann-9.2.0/aarch64-linux/data/platform_config/Ascend910_93*.ini`，运行 `python3 agents/bounds/cann93_platform_counterexamples_readonly.py` 得到 `agents/bounds/cann93-platform-counterexamples.json`，逐文件记录路径、SHA-256 与字段。无设备计算、无修改镜像。

| 配置版本 | Short SoC | NpuArch | Cube / Vector 配置数 | cube_freq 配置字段 |
| --- | --- | --- | --- | --- |
| 9362、9363 | Ascend910_93 | 2201 | 20 / 40 | 1500 |
| 9372 | Ascend910_93 | 2201 | 20 / 40 | 1800 |
| 9381、9382 | Ascend910_93 | 2201 | 24 / 48 | 1800 |
| 9391、9392 | Ascend910_93 | 2201 | 24 / 48 | 1850 |

四组同样列出 `DT_INT8=16,32,16`。因此从同一 short SoC、NpuArch 或 tile 形状**不能逻辑推出**24 核或 1800；同为 24/48 与 1800、64 GiB 的 9381/9382 也不能据这些字段区分。这是当前安装的**软件配置反例**，不是声称这些机型的真实物理峰值已测量，也不推翻 board0 已保存 9382 查询。

单纯作为同一估算公式的敏感性示例，假设其他条件不变，24×1850 相对 24×1800 是 `37/36≈1.02778`；20×1500 相对 24×1800 是 `25/36≈0.69444`。这展示错误绑定可以改变模型输出，不是 169 的实测或有效物理上界。

## 实际头文件与厂商频率定义

169 stage 容器挂载的 `/usr/local/Ascend/driver/include/dcmi_interface_api.h`，SHA-256 `d32dbcbde82ca0bbdd1c2876075c30600ae27abdd057d304b57380ba6b0ff51a`，行 924–930 为 `enum dcmi_freq_type`：7 名为 `DCMI_FREQ_AICORE_CURRENT_`，9 名为 `DCMI_FREQ_AICORE_MAX`；行 1776–1777 声明 `dcmi_get_device_frequency`。它们是名称和 ABI 输入值，头文件这些位置没有说明编号 9 是否覆盖所有瞬时超频/短窗状态。

厂商 [A3 DCMI 26.10 API 文档](https://www.hiascend.com/doc_center/source/zh/HDK/2610/A3/A3DCMI/dcmia3_049.html) 将同 API 的输入 7 描述为 AI Core 当前频率、9 描述为 AI Core **额定频率**，并明确后者是在 TDP 功耗场景可持续运行的频率。该版本文档与本机 driver/header 的版本映射尚未验证，且未证明 profiler 的 `aic_frequency` 字段就是类型 9 的返回值。枚举中的 MAX、额定、实测当前和无条件物理瞬时最大值不能混为一谈；文档的定义**不能证明可以超频**，但也没有提供所需的“任意状态不能更高”的保证。另见 [MindCluster 对额定频率和实时频率的区分](https://www.hiascend.com/document/detail/zh/mindcluster/500/faultdiag/faultdiagug/mindxdlFDUG017.html)，它也不是历史 profiler 字段来源证明。

`evidence/actual-sku-and-cp.json` 保存的 `common` 查询返回退出码 215，因此它没有提供对应的实际额定频率；不能从其他字段补造该结果。不为填补该缺口运行新的设备工作。即使以后读到额定或功率上限，仅凭额定/TDP 数值也不能推出绝对频率上限：必须另有绑定具体芯片、固件/频率状态及工作温度/供电条件的一手上限定义；单凭平均功率亦不能推出逐时钟计算速度，除非得到严格的能量/工作量或功率—频率约束和缓冲初态。

## 能写的条件式

[昇腾 2201 Cube 指令模型](https://asc.gitcode.com/api/appendix/cube_instruction_theoretical_perf_summary.html) 对普通 INT8×INT8 `Mmad` 列出 `16×16×32` 并行参数；[官方 msprof-analyze 固定源码](https://github.com/Ascend/msprof-analyze/blob/6b8f431d4724043f44a39eef000fd5c6600784d8/msprof_analyze/cluster_analyse/cluster_kernels_analysis/operator_mfu/chip_peak_flops.py) 用 `16×32×16×2` ops/cycle 估算，1 MAC 按 2 ops 计。这两个来源提供**条件模型**，但不单独保证历史实际内核使用此对象、不重叠指令的整体发射上限或计算不能分配到其他资源。

若能证明在给定执行域内所有有效直接稠密 INT8 MAC 都只能在 `C` 个该类 Cube 上进行、每核速率上限为 `u` MAC/周期、整个计时域各核的真实频率不超过 `f_max`、不存在计时窗口外预先执行的必要工作或未计入的可并行路径，且不需要额外的管线初始信用，则容量式为：

`T ≥ W_min / (C × u × f_max)`。

其中 `W_min` 必须是该允许实现域和精度契约下**证明不可避免**的工作量；如允许改变算法、预计算、缓存、稀疏、合并工作或其他设备，直接稠密乘法次数不自动是下界。对于有预先在途任务的任意子窗口，须另证明其没有携带入窗口的有效工作，或以已证的管线/突发容量 `B` 放宽为 `W_min ≤ C × u × f_max × T + B`；没有 `B` 的证据时不能把持续速率直接当作任意短窗口完成量的严格界。

可计算但不可直接验收为硬界的参考：若额外**假设** `C=24`、`u=8192 MAC/cycle`、`f_max=1.8 GHz`，才得 `353.8944 TMAC/s`；精度或资源变更不可混用。并行资源、同一窗口内的起点和分片须分别证明；通信、访存与计算可重叠时，独立资源所产生的下界取最大值，只有必然先后关系才可相加。

## 后续可证途径

1. 查历史 worker 的 runtime 全 SoC 名及 16 rank 与实际板卡映射，按版本审计 profiler 生成 `aic_frequency` 的来源；不能从现有两个相容配置中反向唯一识别。
2. 询取制造商针对 169 BOM/driver/firmware 的 AIC 绝对时钟和可用资源上限、核间发行规则、可并行计算路径及任意短窗突发约束；收到后核验它是否适用于运行状态，而不是营销额定值。
3. 在算子语义和精度条件不变的前提下，单独证明不可跳过的工作量、缓存初态及算法范围，按完整任务窗口计算。任何欠缺继续保留未知，既有条件数值不升格为实际硬界。
