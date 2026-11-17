# 普通 HCCL 与 MoE 通信预算分离：八卡组件观测

固定 SGLang A2 daily/CANN9.1/Ascend910B1，使用未修改的原生 get_torch_distributed_pg_options：非 moe 命名组从普通 HCCL 环境取预算，moe 及默认组使用 DEEPEP_HCCL_BUFFSIZE。两个全新八 rank 进程顺序运行，E384、hidden5120、topk6，完整卡组。

| 方案 | HCCL_BUFFSIZE | DEEPEP_HCCL_BUFFSIZE | 数值和图检查 |
|---|---:|---:|---|
| 基线 | 1152 | 1152 | 八卡全部通过 |
| 候选 | 200 | 1152 | 八卡全部通过 |

默认、world、tp、moe_ep 组均实际执行 collective。另验证8192×5120 BF16 TP all-reduce、三次改变输入的图重放，以及 INT8 DeepEP dispatch/identity-expert/combine 的两轮预热和三次图重放，路由计数精确匹配，LL最大绝对误差为0。该 identity-expert fixture 不执行真实 MoE GEMM。

在组、TP图及LL缓冲均初始化后，候选每卡空闲显存比基线增加 **3.7179–3.7198 GiB**。原始八卡差值（bytes）：[3993382912, 3994087424, 3993489408, 3993759744, 3992055808, 3993260032, 3993452544, 3992866816]。这是组件显存观测，尚未证明完整模型 KV 可用容量增加或端到端性能改善。

G54 首次探针因 torchrun standalone 公布的容器主机名无法解析而停在 rendezvous，未得到数值结果。G54b 只将 rendezvous 改为显式 localhost/static，并以 flock 和 bind 检查端口；算子及两套候选参数不变。当前控制器已正常结束，容器非运行、OOMKilled=false、八卡进程清空，所有自有卡锁释放。

下一步在完整模型部署时验证相同分离预算，检查模型加载、EP、PD、目标/草稿图和全形状预热，再测可分配 KV 与端到端指标。不能仅按节省字节换算最终吞吐。

[逐 rank 结果与显存差值](../evidence/components/G54b_HCCL_split/evidence/results.json) · [控制器终态](../evidence/components/G54b_HCCL_split/state.json) · [原生 PG options](../evidence/components/G54b_HCCL_split/probe/native_pg_options.py)
