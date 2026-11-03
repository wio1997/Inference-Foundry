# 已保存的 169 设备身份：只读复核

## 已有记录能证明什么

- 先前保存的 `analysis/operator-ceiling-20261028/actual-sku-and-cp.json` 的 `mapping.stdout` 包含 8 个 NPU 编号，每个含 2 个 Ascend910 逻辑芯片，合计逻辑芯片 0–15。该映射的名称是通用 `Ascend910`，不足以对每个芯片单独确认 9382 或完整产品型号。
- `analysis/operator-ceiling-20261028/model-source-inventory.json` 的 `hardware_queries.board0` 仅查询 NPU 0、chip 0：NPU Name 为 9382，Chip Name 为 Ascend910，Board ID 为 0xb1，Chip Version 为 V1；同对象的内存查询输出 HBM 容量 65536 MB 和时钟字段 1600 MHz。这些是历史查询字段，不代表已验证整机服务器型号、频率硬上限或内存带宽。
- `evidence/profile-device-capacity-info.json` 保存 16 条 profiler 元数据，其核数和频率字段一致。结合型号规则只为对应设备提供旁证，不替代每颗芯片的精确板卡型号和资源拓扑说明。

本地检索的任务证据中未找到服务器 DMI 型号、主板产品名或 `npu-smi -t product` 的直接记录；因此 169 是否属于华为公开网页上的 Atlas 800I A3 / 800T A3 均为 **UNKNOWN**。已有两种产品网页的不同标称算力不能直接套入。

## 最小补证操作

交主控安排现有 stage 容器执行 `stage_dmi_readonly.py`：它仅读 `/sys/class/dmi/id/` 五个不涉及序列号、UUID 的产品字段，任何不存在或权限受限字段标记 UNKNOWN。返回的 DMI 仅代表**容器可见值**：若容器/主机被虚拟化，需单独确认它是否代表物理宿主机或泛化的虚拟机型号。

此阶段没有执行上述远端脚本或任何 `npu-smi` 调用。之后即使得到服务器名称，也需厂商规格与实际型号、板卡数量、精度和设备资源口径逐项一致，才能进一步作为条件模型证据；单靠服务器名字仍不能自动证明 HBM 单方向或给定通信 cut 的物理硬上限。

`host_identity_local_audit.py` 可复算以上不含序列号的信息，生成带来源 SHA 的 `host_identity_local_audit.json`；脚本仅读已保存本地文件。
