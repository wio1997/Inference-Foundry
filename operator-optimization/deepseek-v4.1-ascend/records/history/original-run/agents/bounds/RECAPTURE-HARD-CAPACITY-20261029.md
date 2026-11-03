# 169 硬容量公开资料补查（2026-10-29）

**结论：本轮没有找到能把 169 的 INT8 Cube 总吞吐、最高频率和 HBM 方向性带宽，直接收紧为不可超越数值硬上限的公开一手资料。** 新找到的是昇腾开源软件对 `Ascend910_9382` 的核数配置，与已存 profiler 和 SDK 快照一致；它是独立的软件口径，不是芯片微架构规格书，也不是物理最大并行能力或本次所有 rank 的 SKU 证明。该结论不表示其他文献一定不存在。

## 新的一手资料：RecSDK 固定源码版本

- 华为昇腾组织的 [RecSDK：固定提交 `8251e16519db9132e33d33685d9d7a50607e1567` 的 `lcal_comm.cpp`](https://gitee.com/ascend/RecSDK/blob/8251e16519db9132e33d33685d9d7a50607e1567/src/lccl/src/lcal_comm.cpp)；[原始文件直链](https://gitee.com/ascend/RecSDK/raw/8251e16519db9132e33d33685d9d7a50607e1567/src/lccl/src/lcal_comm.cpp)。2026-10-29 HTTP 取得并保存 `sources/primary/rec-sdk-lcal-8251e165.cpp`，SHA-256 `343a7164c3e3c4bff4f2d5d580a3a8cf107eec71f7285922793a68f1829ce26f`。
- 具体代码行 39、68、85、102–112、325：该代码把 `rtGetSocVersion` 返回的字符串映射到芯片枚举，并使 `Ascend910_9382` 对应的 `GetCoreNum` 返回常量 `24`，赋给通信组件的 `physicalInfo_.coreNum`。这是该源码版本**用于自身逻辑的核数**；并未给出 INT8 乘加每周期最高发射量、频率上限或是否存在其他并行计算路径，也不能证明历史容器实际使用了该 RecSDK 版本。
- 已存 `evidence/actual-sku-and-cp.json` 的 `Ascend910_9382.ini` 同时记载 `NpuArch=2201`、`ai_core_cnt=24`、`cube_core_cnt=24`、`vector_core_cnt=48`、`cube_freq=1800`，该 ini 的记录 SHA-256 为 `d2d84c0ded236393ef40284673cb62bc234e503ed4f0967f31a637f965aadd29`。已有 `evidence/profile-device-capacity-info.json` 中各 device 元数据记录 24 个 AI core、48 个 AIV 和频率字段 1800。它们彼此相符，但历史板卡查询的 `NPU Name=9382` 并不单凭名称证明每个 rank 的 runtime `rtGetSocVersion`、本源码的应用情况或该配置为物理硬界。

复核命令（只读已存文件，不访问设备）：

```sh
shasum -a 256 sources/primary/rec-sdk-lcal-8251e165.cpp
rg -n -C 6 'CHIP_910_9382|GetCoreNum|AI_CORE_NUM_24|rtGetSocVersion' sources/primary/rec-sdk-lcal-8251e165.cpp
```

## 为什么仍不能填入数值硬界

1. [昇腾 Cube 指令理论性能表](https://asc.gitcode.com/api/appendix/cube_instruction_theoretical_perf_summary.html) 的 2201 INT8 Mmad `16×16×32` 是对应单指令的**理论模型并行参数**。配合先前 `msprof-analyze` 模型及 24、1800 可以复算一个条件标称模型，不能由此单独证明该芯片所有允许指令及资源的最高 issue rate、持续/突发硬容量和精确历史版本，不能认作不可超越的设备上界。此前细节见 `hardware_sources_audit.md`。
2. `cube_freq=1800` 与 profiler `aic_frequency=1800` 是已保存的配置及元数据字段；没有证明其为覆盖所有工作状态的物理最高频。测得最高速度或压力测试峰值也只是观测值，不能替代不可超越上限。
3. 同一系列 OEM 页面中的 FP16 整机算力与 3.2 TB/s 内存带宽是产品标称，缺少本次物理 chip/卡的绑定与读写方向、共享路径和短窗口突发说明；不能推导本机 INT8、单 chip HBM 或任意 cut 的硬上限。存储相关配置 `ddr_rate` 的单位、粒度和共享口径亦未证实，不能当字节/秒的物理硬界。
4. 精确到 SKU 的芯片架构/ISA 与最高时钟规格、并行路径共享关系、指定存储 cut 和任意时间窗口的方向性速率及初始突发包络，仍须原厂/供货方对应型号技术材料证明；即使取得，也还需证明目标算子允许融合/重排时的必要工作量。

本轮先核对 `hardware_sources_audit.md` 与 `sources/primary` 已有资料，之后只用 Web 搜索和 HTTPS 读取公开资料；未访问 169、启动设备负载或更改服务。除了上述固定提交源码，没有发现足以收紧目标数值硬界的新一手证据。
