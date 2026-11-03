# 169 新 profile：一个低报告 MAC 比例的 MC2 任务

> **导出后重核：**离线解析已结束，记录报告 16/16 `kernel_details.csv` 存在；EP8 的文件完成后重读时，原候选的文件大小、SHA-256、行号、身份和文内周期字段均与首次观察一致。它是一个已导出任务的单条记录；其 request、层和 replay 关联尚未证明。解析成功或哈希一致不证明无采样扰动、全数据完整或可信的时间/字段硬件契约。

## 观察及范围

对新诊断 EP8 的 `ASCEND_PROFILER_OUTPUT/kernel_details.csv` 做只读筛选，选取**文件顺序第一条**同时满足：`Type=DispatchFFNCombine`、导出物理输入首项 `[3,5120]`、AIC 周期为正、`0 < aic_mac_ratio ≤ 0.020`、导出时长在 `(0,10000)` 微秒的行。此筛选旨在找一个可追溯样本，**不是全局最低、同批平均或特定层的排名**。

| 字段 | 本行导出值 |
|---|---:|
| CSV 行号（含表头） | 1450 |
| 算子名 / 类型 | `DispatchFFNCombine` / `DispatchFFNCombine` |
| model / stream / task | 49 / 89 / 832 |
| 本卡起点（原始字段） | `1793241326722760.255` μs |
| 任务持续时间 | `266.075` μs |
| Accelerator Core / Block / Mix Block | `MIX_AIC` / 24 / 48 |
| 输入形状 | `[3,5120];[24,5120,4608];[24,2304,5120];[3,6];[110592];[122880];[3,6];[3]` |
| 输入 dtype | `DT_BF16;INT8;INT8;INT32;INT64;INT64;FLOAT;BOOL` |
| AIC total cycles / MAC ratio | `6,725,373` / `0.019` |
| AIV total cycles / vector ratio | `22,576,074` / `0.003` |

读取范围是 EP8 这一份导出 CSV；该文件 SHA-256 `20cfb97abe03befda3fbf2c1194e48531aac44ce17d4f5fc130ea8ffb5261730`、大小 `12669029` 字节。读取前后文件描述符的 device/inode/size/mtime 未变化，源文件当前路径位于本人 staging 容器 `/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_737_20261029023525648_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv`。本哈希首次记录于导出期间；离线解析结束后以完整读取重新计算并确认一致，记录的是完成后本次读取的文件内容，不是事前冻结哈希或对采集语义的证明。

物理形状和旧 target MC2 记录相容；目前尚未把该行绑定到新一轮 request、scheduler/runner call ID、具体 decoder 层或同次跨 rank collective，不能称为第 0 层或特定请求。`taskId=832` 本身也不保证唯一 replay；新记录可以用来设计关联，不能以相似编号推断身份。排除掉超过 10 ms 的任务是本次样本选择，不代表它们是 warmup，也不代表所选行是稳态。

## 该行可计算的字段代理

精确按 CSV 中显示的十进制值计算：

- `AIC MAC 代理 = 6,725,373 × 0.019 = 127,782.087`；AIC 分母为 `6,725,373` 导出 cycles，显示比例为 `1.9%`。
- `AIV Vector 代理 = 22,576,074 × 0.003 = 67,728.222`；AIV 分母为 `22,576,074` 导出 cycles，显示比例为 `0.3%`。

这是把导出 cycle 字段乘显示 ratio 的**算术代理**，不能称为 MAC 运算数、真正忙碌时长或芯片利用率。AIC/AIV 分母不同，MIX 任务内管线与其他流可能并行，不能相加成墙钟。数字 `0.019/0.003` 有限显示精度；本次未核实 CANN 9.2 exporter 的舍入、分母聚合与样本覆盖。

仅为量化显示精度的影响，**如果** ratio 恰按 0.001 最近舍入、AIC cycle 显示无误且字段确实绑定相同记录，则 AIC 乘积的保守数值范围是 `[124,419.4005, 131,144.7735]`。这不是已证明的物理周期误差范围。源字段即使显示 0 也不能判定设备无工作；ratio 小也不能指定通信等待、scalar、访存或其他根因。

## 工作账与理论界所需条件

这些形状允许设置一个**待验证的固定实现账**：设 `A1`、`A2` 分别是此 rank 这次两次 GMM 经路由与容量裁剪后实际参与的专家行数；若逻辑矩阵维度确为 `5120×4608`、`2304×5120`，且两个阶段对各有效行执行完整传统稠密点积，则

`W_logical = 23,592,960×A1 + 11,796,480×A2 MAC`。

本次 CSV 没有给出 `A1`、`A2`、有效 mask、接收/裁剪、真实 binary 与 tiling，也未证明输出中的 24 是实际专家行数。因此 `W_logical` **不能赋数值**；本 rank 输入 3 行不能代替多个 EP rank 向专家发送后的 `A1/A2`。即便将来拿到这些行数，此式首先是固定实现的条件逻辑账，不等于物理指令数，也不自动是允许融合、预计算、稀疏跳过后的必要工作量。

若另外独立证明：某个 `W_min` 是此算子等价输出在允许实现域内、同一个窗口必须完成的工作，且覆盖所有可用计算路径的资源对窗口任意前缀满足 `Q(t)≤β+P_max·t`（`P_max>0`，`β≥0` 包含在途或突发信用），则 `T ≥ max(0,W_min−β)/P_max`。只有额外证明 `β=0`，才可写为 `T ≥ W_min/P_max`；先前未明确这一短窗条件，现补正。该行 `T_obs=266.075 μs` 可与这个 `T_lb` 比较，仅在绑定了同一任务、精度、有效输入和窗口且计时误差可用时计算 `T_obs/T_lb`。目前 `W_min` 和 `P_max` 均未知，**不能计算理论差距**；`0.019` 和观测时长也不能反向充当不可超越的硬件上限。通信必要量、方向性硬容量和与计算的重叠同样未绑定，不能把 `T_obs` 分成通信/计算/等待。

下一条直接补证：关联此行至新记录的匿名 request/runner call ID、层、replay 及 group；再找同次有效专家行计数、kernel/tiling 和版本匹配的采集语义。证实准确身份后，再查厂商对该机路径的硬容量与输出契约；缺项保持 UNKNOWN。

## 只读复算

在允许访问上述文件的 staging 容器运行以下代码，校验完成后确认的文件哈希、位置与字段，并用 `Decimal` 复算。它不打开 NPU、不读取用户 prompt/生成文本/权重数据；如文件变化会失败：

```python
import csv, hashlib, io, json
from decimal import Decimal
from pathlib import Path
p = Path('/work/vllm-ascend-reproduce-172-on-169-20261028/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41/diagnostic-20261029/runtime/diagnostic-traces/dp1_pp0_tp0_dcp0_ep8_rank0_737_20261029023525648_ascend_pt/ASCEND_PROFILER_OUTPUT/kernel_details.csv')
raw = p.read_bytes()
assert len(raw) == 12669029
assert hashlib.sha256(raw).hexdigest() == '20cfb97abe03befda3fbf2c1194e48531aac44ce17d4f5fc130ea8ffb5261730'
reader = csv.DictReader(io.StringIO(raw.decode('utf-8')))
row = next(r for i, r in enumerate(reader, 2) if i == 1450)
assert (row['Name'], row['Type'], row['Model ID'], row['Stream ID'], row['Task ID']) == ('DispatchFFNCombine','DispatchFFNCombine','49','89','832')
assert row['Duration(us)'] == '266.075'
assert row['Input Shapes'].strip('"').startswith('3,5120;24,5120,4608;24,2304,5120;3,6;')
aic, mac = Decimal(row['aic_total_cycles']), Decimal(row['aic_mac_ratio'])
aiv, vec = Decimal(row['aiv_total_cycles']), Decimal(row['aiv_vec_ratio'])
assert (aic, mac, aiv, vec) == (Decimal('6725373'), Decimal('0.019'), Decimal('22576074'), Decimal('0.003'))
print(json.dumps({'duration_us': row['Duration(us)'], 'aic_cycles':str(aic), 'aic_mac_ratio':str(mac), 'aic_proxy':str(aic*mac), 'aiv_cycles':str(aiv), 'aiv_vec_ratio':str(vec), 'aiv_proxy':str(aiv*vec)}, indent=2))
```

本次在离线解析完成后，通过 SSH wrapper 在本人 staging 容器重新读取并复核同一 CSV；没有启动新 NPU 工作或修改服务。
