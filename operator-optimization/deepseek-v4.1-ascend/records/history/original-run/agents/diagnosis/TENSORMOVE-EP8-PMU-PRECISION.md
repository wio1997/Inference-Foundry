# TensorMove 的 0.0%：原始数据库定点核对

在已有采集的 digest `1b826883…`、第 0 层、EP8 上，固定一条 `AI_VECTOR_CORE` 的 `aclnnInplaceCopy_TensorMoveAiCore_TensorMove`；其输入输出均为 BF16 `[8184,5120]`。历史 CSV 原行 3532 报告 duration `37.100 μs`、`aiv_vec_ratio=0.0`，不表示数据库中此比例为零。

此前选中 scope→flow→CSV 的结果含完整 task 和开始时间。这里直接核对同一次保存采集的原 CSV（SHA-256 `20cfb97abe03befda3fbf2c1194e48531aac44ce17d4f5fc130ea8ffb5261730` 与既有抽取记录一致），按 model `4294967295`、stream `47`、task `43441` 和十进制开始时间精确乘 1000 后的纳秒时间在 EP8 SQLite `TASK` 中唯一匹配 `globalTaskId=1865`；以 globalTaskId 连接 `TASK_PMU_INFO` 和 `STRING_IDS`。28 条 PMU 行具有相同 timestamp，位于 TASK end 前 121 ns。只有这条记录被高精度核验，不扩展到同组其他七卡或其他 digest。

| 字段 | CSV 显示比例 | 同任务 DB 原始比例 | DB 的配套 time 数值 |
|---|---:|---:|---:|
| aiv_vec | 0.0 | 0.00023145867877048598（0.0231459%） | 7.777777777777778 |
| aiv_scalar | 0.009 | 0.008900137290817497 | 299.0740740740741 |
| aiv_mte2 | 0.42 | 0.41998590581973916 | 14112.916666666668 |
| aiv_mte3 | 0.466 | 0.46567832892344846 | 15648.333333333334 |

同一次 DB 的 `aiv_total_time` 是 `33603.31018518519`，`aiv_total_cycles` 是 `2903326`。对表格四项分别计算 `time / aiv_total_time`，与 DB 对应 ratio 一致至浮点精度；这说明在**这一任务、这一组数据库值**中存在相同数值分母关系，不能单独证明采集器内部实际使用该公式、字段时间单位、归约方式或涉及哪些硬件核。四个 CSV ratio 恰符合将 DB ratio 保留三位小数的算术结果；仅凭一条任务不能证明历史导出器实现或普遍舍入规则。CSV 中配套 time 的显示也与 DB 值除以 1000 后保留三位的数值相符；不能由这一数值符合认定其原始单位。

因此本例的 vector 显示 0.0 只对应一个非零的小向量字段；同任务的 MTE2、MTE3 字段也非零。它并不证明设备或算子空闲，亦不能证明真实搬运字节、带宽、墙钟时间或性能瓶颈；不同流水可能同时活动，禁止将两个比例相加作为任务占用率。源码与算子的实际目标张量身份和运行时行为未由这些计数器证明，不能因名字推断拷贝有效字节或数据通路。

当前容器 CANN 9.2 已安装 profiler 源码及其适用边界见 [F08 AIV 公式核查](../fusion/F08-AIV-CANN92-PMU-FORMULA.md)：同一镜像并不证明历史采集实际使用对应源码或特定导出路径，尤其不能把另一版本的逐 block 文档当成本次 DB 归约定义。尚未验证本次采集 DB ratio 的原始生产或 kernel_details 导出公式。

复算：`python3 agents/diagnosis/audit_tensormove_pmu.py` 只通过自有 stage 容器只读原 CSV 和 SQLite，保存完整行、身份及 PMU，不运行 NPU、不改目标实现。脚本 SHA-256 `3c5283f5ddcec3e04cf60b03daeda987b4a54e108d121daa28de280242d061aa`；结果 [tensormove-ep8-l0-pmu.json](tensormove-ep8-l0-pmu.json) SHA-256 `5c1d8b7017968b096cb8d859d065109c47869fedb16b351ada504148e06cd643`。重复读取依赖原采集文件仍存在；CSV 哈希校验及精确身份唯一性失败则脚本报错。
