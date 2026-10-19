# 32 die 配置与内存初审（尚未冻结服务启动）

本项目普通 W8A8，现场核验 2026-10-17。每节点 16 die × 64 GiB HBM；每 P/D 角色两节点，合计 32 die / 2 TiB HBM。镜像分别固定 rc1 `090eeee6…` 与 B103 `c5cf2deb…`，完整 ID 见 RESOURCES。现场 npu-smi 25.5.2，各节点快照无运行 NPU 进程；这是瞬时占用记录，实际启动前重新核验。

读取 30 上所有 safetensors 的 21,729,720 字节张量头，未读取权重 payload。张量 payload 合计 764,015,533,568 字节，目录 193 文件合计 764,240,829,290 字节；差值含头和其他元数据，不能混作 HBM 占用。78 个 target 层加 1 个 NextN 层，256 routed experts，64 attention heads，kv_lora_rank=512，rope=64，index head dim=128，indexer 权重出现于 22 个层（含 NextN）。量化描述统计 W8A8_DYNAMIC=176830、FLOAT=644，不是 C8。

| 张量类别 | 磁盘 tensor bytes |
|---|---:|
| target routed | 726348595200 |
| target dense | 20956302336 |
| target shared | 2837299200 |
| draft routed | 9684647936 |
| draft dense | 4150858240 |
| draft shared | 37830656 |

EP32 的 routed 分片约 21.4214 GiB/die。把全部 dense/shared（包括 draft）按 attention TP1 完整保留时，加 26.0605 GiB，合计约 **47.4819 GiB/die**。这只是保存张量字节的初审，不是实际框架 weight allocation：lm_head F32 转换、embedding/head 复用、权重重排、fused shared expert、量化辅助张量/算子缓存可能改变结果。不能以“764GB/32”估算全部运行权重。

初始候选而非执行冻结：P attention TP8 × attention DP4，EP32；D attention TP2 × attention DP16，EP32，PP1/CP1。64 heads、256 experts 均整除相应组。vLLM 以 TP8/DP4 和 TP2/DP16 表示；SGLang 的 launch TP32 与 attention DP4/16、enable_dp_attention、EP32 需要通过其实际 group 构造验证，不能把两个框架相同字段名当相同分组。SGLang DP attention 下 dense FFN 是否继续 TP32，以及 EP shared 融合/LM head 的分组，仍需核源码并记录实际 rank。暂不启动未验证配置。

D TP1/DP32 作为容量比较候选保留；TP2/DP16 可降低部分 attention/dense 副本，但 MLA KV 仍可能在 TP 内复制，不能把 KV 容量按 TP 直接相除。TP4/DP8 后续仅在实际 HBM/运行数或性能证据支持时比较。

普通 BF16 MLA 的理论 target KV 主体为 `78 * (512+64) * 2 = 89856 bytes/token`，80K 主体约 6.695 GiB/请求；另加实际 indexer/scale、NextN、页对齐、块表和 speculative lookahead。Indexer 共享层和 NextN 共享路径必须用真实 cache spec/分配证明，禁止把 78 层 indexer 都算一份或套旧容量。80K + 600 输出 + MTP5 的 block rounding 与运行/排队容量尚未实测。一个 DP 能运行多少条由实际 free HBM、KV bytes/block、图、workspace 和 scheduler 共同决定。

P 暂以 eager 为功能起点；D 目标 MTP5，vLLM 固定镜像源码明确把 glm_moe_dsa 映射为 DeepseekV32MTPModel。图与融合仅在真实正确性通过后进入性能冻结；不把 eager 保护用例当最终性能配置。MemCache 原生 SDMA 与 P→D SDMA 独立核验，尚未通过。SDK1.2/1.3 的差异使初次比较属于端到端部署栈比较。

原始证据在 `runs/20261017T0210-capability-headers/`，同机原始数据与 SHA 索引保留。CPU 内存、物理 NUMA/HCCS 距离与各级 cache pool 容量须继续补齐；不从这份算术推导宣称最优配置或服务增益。
