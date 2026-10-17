# Flash 官方路径与现场差异（2026-10-18）

官方教程固定至 [22730f46](https://github.com/vllm-project/vllm-ascend/blob/22730f46da2d821dc3955f47d56d16284eb2de81/docs/source/tutorials/models/GLM5.3-Flash.md)，明确基于 v0.30.0RC1。现场 image090、Ascend commit5f828ad、MF/MMC1.2、CANN9.1/HDK25.5.2 不升级。以下是源码核对和下一次有界诊断的决策，不代表正式性能配置已经生效。

| 项目 | 当前官方与实际源码 | 当前候选/实际 R164 | 下一步及必要性 |
|---|---|---|---|
| P/D 拓扑 | 新教程 P2TP8/D8TP2；旧 RC1 tag D16TP1 | P2TP8；D16TP1 仅 source candidate | P 沿官方；D8TP2 成为官方参考候选，原 D16TP1 保存为旧版参考，不混称相同配置；下一次仅 P，不提前资格化 D |
| KDA | 实际原生 ops/kda.py 与固定 main **逐字节相同** d6027b2… | 三个 FP32 bulk overlays | 下一次原生整模控制移除三个数学 overlays。R026 原生 chunk 合成 F32 state gate FAIL 保留；R127/R159 候选组件 PASS 不等于必须覆盖官方。任何原生运行数值失败仍 FAIL，无门限降低 |
| DSA indexer | main 相比镜像只把 INDEX32 page-table mask 改成 valid | 自写 per-head ReLU/32 tile | 下一次使用完整固定官方 main 文件 9b74018…，撤下自写公式；仅官方可见页 mask 修复，不全量升级框架；R080 诊断参考保留 |
| P 优化 | 官方 FlashComm/DSACP/fusedMC2/shared-expert DP，overlap false，4096/64 | FlashComm、overlap true、8192/32 | 下一次官方 P 优化 flags；max-model-len133120 满足 80K 合同；P enforce-eager 明确是有界正确性诊断差异，非正式收益 |
| D graph/MTP | 官方 FULL_DECODE_ONLY、draft graph MTP5、FlashComm | D draft eager/FlashComm 未显式 | D 下次服务计划对齐，实际源码能力与 replay/MTP counters 尚待验证；P draft eager 与官方一致 |
| KV dtype/量化 | 官方 W8A8 + int8 KV 示例，公开权重 URL 不提供本地 checkpoint SHA | 冻结本地 W8A8 identities；KV dtype 未指定 | 保持当前 KV dtype，记录实际生效值。int8 KV 是另一个需重新数值/状态验证的 profile，不能借官方示例直接更换；本地与公开权重完整等价未证明 |
| MemCache barrier/device | 镜像已含 PR16287 设备绑定/metadata-only scheduler；barrier 默认 True | child 未显式配置，但并非 False；D metadata-only 是额外候选 | 明确 child barrier true，不重复移植 PR16287。P 使用原生 pool_worker，撤下仅对 D 有效 overlay；D 可按官方每 worker 20GB 原生 BM，避免零容量假设 |
| 池容量与部署 | 官方示例20GB/worker、hccn.conf RO、HCCS资源声明；指南要求检查 A3 可分配内存 | R164 56GiB/worker，缺 hccn mount/HCCS声明；native HalMemCreate ret6 | 下一次20GB/worker、hccn RO、显式 HCCS。320GiB 只是申请量，不是 C64 容量证明；不得系统 drop_caches/compact/改 hugepage。错误 ret6 不猜根因 |
| SDMA/SDK | 官方 MooncakeV2 示例不构成本项目禁止 RDMA fallback 的运行实证；native MMC 使用 device_sdma | strict native MF SDMA shim；direct 原始3扩展+同 inode SMEM alias；optional sparse lazy import | 暂保留：R157 真 cache+SDMA/退出0证据；公共包路径 R148/R153 的真实 -6 证据；SDK二进制不变。R164 guard 是私有入口保护，不是官方不支持；后续若原生满足相同资格应撤私有入口。无新生命周期补丁 |
| 观察器 | 官方原生 API/多连接器 | 实际 SDK put/readback 与 shutdown queues 仅 qualification | 仅有界诊断保留，formal 必须 flag0/无 CPU hashes。未真实 put/get 不报命中；PR18956 控制面清理修复不等同 SDK 析构修复，不盲移植 |

官方精度结果采用 GPQA/MMMU-PRO 的数据集评分，不声明逐 token 与自写 serial oracle 一致；本项目冻结 LP/state/repeat/PD gates 仍保留，原生整模控制和原生精度评估分开记录。R165 如成功启动也不自动成为已接受的性能基线。旧 R150/R154/R164 verdict 不改写。
