# 已选索引整理 scalar 比率：同镜像 CANN 候选公式

只读冻结镜像 stage 容器中当前安装的 CANN 9.2 profiler Python、已有 32 条选中任务聚合；未启动 NPU、未修改推理实现。安装文件与历史数据库之间尚无运行时版本和生产路径的完整绑定。

## 源码可证明的范围

在 `/usr/local/Ascend/cann-9.2.0/tools/profiler/profiler_tool/analysis/`，`common_func/constant.py:111-112,44-54,375` 实现 `scalar_ratio = pmu / task_cyc`，零分母返回 0；`mscalculate/calculate_ai_core_data.py:298-310,348-368` 把对应 PMU 值传入注册的公式，但有公式排除列表和零周期分支；`mscalculate/stars/ffts_pmu_calculator.py:319-344` 的非 MIX 路径使用 context PMU 和 `data.total_cycle` 调用此计算；`mscalculate/aic/aic_utils.py:28-53` 从配置将事件映射到名称。该源码证明对应分支的处理，不证明这 32 条历史 DB 条目由它产生。

`msconfig/ai_core_config.py:24-35,183-200,536-540` 的默认配置把事件 `0x9` 标为 `scalar_busy_cycles`，把它映射为 scalar_ratio，单条公式写为 `r9/task_cyc`；`:363-374` 的默认聚合公式写为 `SUM(r9)/SUM(task_cyc)`。`:803-805,895-900,1117-1128` 同时表明 V6 配置使用 `0x1` 和 `SUM(r1)/SUM(task_cyc)`，且按 chip id 选择配置。因此不能只凭默认命名就认定实际芯片事件编码、采样范围或聚合方法，更不能不分分支将 SUM 作为本次历史 DB 的计算方式。

源码 SHA-256：`constant.py` b5f0bc0f31f4eef523830be6416e01a6cb89d7fb27e4a9624a1048b22753ec51；`calculate_ai_core_data.py` f804d3fb97b9957a1a4dc9c60c6074cbc89b9030522cdd9916b6524b660d2a9c；`ai_core_config.py` b8cb0d2a768b2bf15291d9c31eb764cf6bae7d4a39ecdf6820cd4f85f28f7685；`aic_utils.py` 4f5669c2beb8e641db36f465f58cb8d39ac0b6e23f80a944d4c39b4851b6906c；`ffts_pmu_calculator.py` 72b65a386260f79b34ab1894e6ba4787616a62584c988272195d4a9a6909f66d。

## 仅是算术一致性

复现 `python3 agents/diagnosis/check_prepare_scalar_formula.py`，输入 `prepare-pmu-32-selected.json` SHA-256 为 `77ebe89f02b38a9c5dda0e749086b2a4e51f20ecb1a282e4adf8b2fd0ea5a4af`。32/32 条中 `aiv_scalar_ratio * aiv_total_cycles` 都得到整数浮点结果；例如 EP3 layer 2 是 `0.8692811418474331 * 9726133 = 8454744.0`。vec、MTE2、MTE3 同样与整数相符，最大浮点偏差分别不超过 1.17e-10、2.92e-11、7.28e-12。推导整数是比率乘周期的反算值，并非独立读取到的事件寄存器，因此只能说与上面的候选除法一致，不能证明分子即该事件的真实采样计数或认证历史导出链路。输出 `prepare-scalar-formula-check.json` SHA-256 `4d3dd014236747808f9480c879fcbca7cc0754a1b2da98e977ce285657cd5e81`；脚本 SHA-256 `da639227e23547a49460725e3e20bfd992e499e40ea5aee84e6cef7e747c5316`。

由此仍不能将 87% 解释为墙钟时间百分比、有效算术指令比例、其他流水空闲、等待或瓶颈根因。真实芯片的事件定义、采集模式/溢出、核间聚合及历史数据库写入和版本绑定需要额外证据。历史数据两个导出视图并非独立实验。
