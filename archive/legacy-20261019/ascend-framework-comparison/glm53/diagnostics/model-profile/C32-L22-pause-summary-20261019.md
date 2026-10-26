# GLM-5.3 C32：L22 完成，按用户要求暂停

本次仅普通 W8A8、固定两套 2P2D、P→D SDMA、C32、真实 80K 输入/600 输出。最新两边请求、profile 与采集均已结束；不再自动开始候选、回滚、模型或 profile。保留当前服务与缓存。原优化目标尚未达成。

## 结果与证据边界

| L22 诊断 | vLLM | SGLang |
| --- | ---: | ---: |
| 逻辑请求 / 后端实际尝试 | 32 / 64 | 32 / 64 |
| 计数与终态通过 / 输出 token | 64 / 19232 | 64 / 19232 |
| 前缀缓存通过 | 32/32 | 32/32 |
| profile 原始控制结果 | 两节点 start/stop 均 200 | start 200、stop 500，控制失败原样保留 |
| 原生文件证据 | 32 rank 原始采集完整；仅解析两代表 rank 副本 | 32 rank 的六类 CSV 完整稳定；两代表 trace 已采集 |

vLLM 的 daemon worker 未自动解析原始数据；另用已安装 torch_npu 官方离线 analyse 接口在副本上解析两个代表 rank，未重跑请求或 profile。SGLang 的 stop 500 原因仍未知，原生文件完成性另行验证，不把 500 改写为成功。

这两次是带 profile 的诊断，不替代正式性能基线；没有本轮新增优化收益。D 的采样门槛为原生 DP4、DP10 活跃且总 running≥8，实际触发 12；不是全 D32 或全 16DP 同时活跃。实际 batch/context 尚未逐 replay 完全对齐。

## 差距归纳

两边代表 rank 均为每 target replay 75 次 MoE Dispatch、每周期 5 次 eager draft Dispatch。用同一硬件进程的核心任务和重复首任务切分，按 Decimal 在每台主机内计算时间差：

| 原生 DP | 首周期差 SG−VL | target Graph 核心跨度差 | Graph 之间间隔差 |
| --- | ---: | ---: | ---: |
| 4 | 78.552540 ms | 8.912508 ms | 69.640032 ms |
| 10 | 78.901106 ms | 9.401834 ms | 69.499272 ms |

SGLang 两个代表的 Graph 间隔约 90 ms，vLLM 约 20.5 ms。SGLang eager MoE Dispatch 的设备跨度中位约 5.7–6.0 ms，vLLM 约 50–55 μs；但 target Graph 中同名算子的中位均约 43–47 μs。长跨度可能包含通信等待或到达不齐，不能宣称计算效率相差百倍。

实际 trace 中，vLLM 有异步任务队列流，workspace 查询在主线程、执行 API 在后端线程；SGLang 的对应 API 同在主线程。vLLM target 使用 MlaPrologV3，SGLang 为拆分路径。这些是恢复后的候选方向，未隔离因果、未部署 TASK_QUEUE_ENABLE=1、未验证收益。SG 每代表只有一个完整周期，VL 有七个，profile 扰动与请求上下文仍限制比较。

## 正式性能结论保持不变

| 历史正式 C32（128/128） | tokens/s | TPOT P50 | TTFT P50 |
| --- | ---: | ---: | ---: |
| vLLM 参考 | 530.8125 | 19.1527 ms | 21.1620 s |
| SGLang L12D 历史参考 | 389.8741 | 36.6586 ms | 17.7572 s |
| 当前 SGLang L17 NoCopy | 411.9606 | 37.9931 ms | 16.6567 s |

L17 对 L12D 的 TPS 增加 5.665%，TPOT 中位却增加 3.640%；冻结的解码改善目标未达到，NoCopy 解码优化未验收，固定 SLA 未通过。当前 D36/41 仍为 L17 NoCopy；L12D 是历史参考。没有本轮部署或回滚。

## MemCache、记账和恢复入口

按 Ascend/框架对应部署文档与实际 SDK 源码核对。L20 结束后元数据检查发现既有 32×625=20000 个 key 全部存在；这不倒推解释 L20 请求时的 miss。L21、L22 使用真实 tokenizer 生成新后缀后，32/32 前缀命中通过。未通过清缓存或重复预热掩盖 miss。

SG 实际累计后端尝试为 6793；完整资格成功 6492 保持不变。L17 旧 64 个 pending 不变；L22 新 64 个计数/终态/缓存通过，因未重做完整逐请求 SDMA 传输资格，另列 native pending。vLLM L20/L21/L22 仅本段合计 192 后端 /57696 输出，独立记账，不混入 SG。

- 终态独立验收：[C32-L22-terminal-primary-acceptance-v1.json](../../results/C32-L22-terminal-primary-acceptance-v1.json)
- 可复算 trace 分析：[v2 结果](../../results/C32-L22-paired-trace-primary-analysis-v2.json)、[分析脚本](../../scripts/analyze_C32_L22_paired_traces_local_v2.py)。v1 API 过滤漏掉 AscendCL@ 前缀，v2 已修正；图周期结果不变。
- 证据按路径、大小、SHA256 索引；原始 trace、业务请求/输出文本、凭证、权重、镜像及 SDK 二进制不上传。
- 恢复后先核对现场身份/负载、batch/context、全 rank eager Dispatch 分布、队列与 NoCopy/SDMA 生命周期兼容性，再冻结单变量候选、当前运行时回退方案，做最小正确性和无 profile C32 验证。有可信收益后再考虑正式 128 请求。

官方部署/解析参考：[vLLM Ascend KV Pool](https://github.com/vllm-project/vllm-ascend/blob/main/docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md)、[原生 profile 离线解析](https://docs.vllm.ai/projects/ascend/zh-cn/latest/developer_guide/performance_and_debug/service_profiling_guide.html)、[TASK_QUEUE_ENABLE](https://github.com/Ascend/pytorch/blob/master/docs/zh/api/environment_variable/op_execution/TASK_QUEUE_ENABLE.md)。实际运行结论以本项目原生证据为准。
