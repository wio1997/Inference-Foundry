# 官方 PD / MemCache 文档对齐（2026-10-17）

本次为文档及匹配源码只读核查，不是新设备实验。vLLM-Ascend 固定到现场 `v0.30.0rc1` 的提交 `5f828ad638ee4f9b12db87f9d5481f29076d37f9`；不把 latest 的示例直接当作现场支持承诺。

## 结论与执行影响

1. **已有官方 MemCache PD 场景，但当前纯池传递不是所读官方示例的完整复现。** [GLM5.3 §5.3](https://github.com/vllm-project/vllm-ascend/blob/5f828ad638ee4f9b12db87f9d5481f29076d37f9/docs/source/tutorials/models/GLM5.3.md#53-memcache-kv-cache-pool-deployment) 明确由 MooncakeConnectorV2 负责 P→D 直传，AscendStoreConnector 提供 MemCache 池；[通用池指南](https://github.com/vllm-project/vllm-ascend/blob/5f828ad638ee4f9b12db87f9d5481f29076d37f9/docs/source/user_guide/feature_guide/kv_cache_pool_ascend_store.md#scenario-2-memcache-backend) 的 PD 示例同样组合 MooncakeConnectorV1 和 Store。现场纯 Store 的 P写池→D读池需单独验收。未找到匹配示例不等于证明不可用，也不能据此违反用户“先不考虑 Mooncake”的约束。
2. **官方 GLM 参考布局与现场不同。** 其 PD 是四节点：P 为 TP16/PP2，D 为 DP16/TP2；现场两节点各 TP16、DP1。文档保留 Experimental 标记，并区分 v0.23.0 混部示例与 0.30.0RC PD 示例。因此不能宣称现场组合已被官方端到端验证。
3. **P只生成一个 token、D收到原提示词，并不单独构成错误。** 匹配版本的 [官方代理](https://github.com/vllm-project/vllm-ascend/blob/5f828ad638ee4f9b12db87f9d5481f29076d37f9/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py) 也采用这一请求结构，但转发 `kv_transfer_params`。它依赖直传连接器的远程 KV 等待、块保留与完成通知合同。不能从代理结构推导纯 Store 的 P HTTP 返回就代表所有池数据已可读。
4. **不能仅凭 wait_for_save 入队断定缺少同步。** [旧池设计文档](https://github.com/vllm-project/vllm-ascend/blob/5f828ad638ee4f9b12db87f9d5481f29076d37f9/docs/source/developer_guide/Design_Documents/KV_Cache_Pool_Guide.md) 的函数描述与当前实现不能混用。当前源码另有 `wait_for_requests_saved(released_req_ids)` 释放/复用栅栏。已有 SDK 核查也说明默认 BatchGet 等待 SDMA 完成。两者都不等于完整证明模型计算流、写池与代理发布之间的正确顺序；本次不继续扩大 HTTP 调用链诊断。
5. **图执行属于应先补齐的官方基线。** [GLM5.3 §5.2](https://github.com/vllm-project/vllm-ascend/blob/5f828ad638ee4f9b12db87f9d5481f29076d37f9/docs/source/tutorials/models/GLM5.3.md#52-prefill-decode-disaggregation) 的 D 使用 FULL_DECODE_ONLY，P 使用 eager。原 5.16 token/s 来自 eager 诊断路径，不代表官方充分优化性能。当前原生 eager/graph 对照继续，不被本次阅读打断；捕获成功仍须核对实际 replay、正确性及计时。

## 已核对的前置条件

| 官方要求/说明 | 现场记录与结论 |
| --- | --- |
| A3 HCCS 使用 `device_sdma`，灵衢版本至少 1.5；`ACL_OP_INIT_MODE=1`、`ASCEND_ENABLE_USE_FABRIC_MEM=1` | 协议、环境变量及 SDMA trace 已有证据；灵衢 L1 版本尚未确认，不能宣布全套兼容性已核验。 |
| 所有节点同 `PYTHONHASHSEED`，连接同一个池；D从池加载需显式 `consumer_is_to_load=true`（默认 false） | 部署配置已设置，不是本次新发现的漏项。 |
| `use_layerwise` 仅支持 P 端 | 当前为 false；不能盲目在两端都开启。 |
| `c8_enable_reshape_optim` 与旧 HDK 上的池化/AICPU 操作可能冲突 | [官方配置说明](https://github.com/vllm-project/vllm-ascend/blob/5f828ad638ee4f9b12db87f9d5481f29076d37f9/docs/source/user_guide/configuration/additional_config.md#L88) 已给关闭办法；现场已关闭，保持 C8 精度。 |
| API成功或本地前缀命中不足以证明共享池生效 | [GLM指南验收段落](https://github.com/vllm-project/vllm-ascend/blob/5f828ad638ee4f9b12db87f9d5481f29076d37f9/docs/source/tutorials/models/GLM5.3.md#534-start-the-proxy-and-verify-pooling) 要求本地无前缀时验证池加载。现有短请求命中证据不能替代长输入与并发验收。 |

官方性能示例还启用了 AIV、EP、MC2 等；它们不能未经数据路径核对就照搬到本线严格全 SDMA 条件。文档没有为本次“有效缓存区域读回全零”提供可直接套用的根因结论。

## 来源与复用

上述原文于本次通过直接 HTTPS 获取。GLM 文档内仍链接旧 `kv_pool.md`；同一提交中实际指南为 `kv_cache_pool_ascend_store.md`，使用后者。官方 GitCode FAQ 本次网页读取仅返回页面壳，没有将其计入已读正文。

| 文件 | 原文字节 SHA256 |
| --- | --- |
| GLM5.3.md | `ba7797e7711f1d181d980ae326c399aa50c0ec204dfca026ca7c98a536dff5f1` |
| kv_cache_pool_ascend_store.md | `7e214d6722eedd7d6301a011382da837fd94bc94c08485de4c0442ab2c3b9430` |
| disaggregated_prefill.md | `f78e41ee95efbd47aa165bca688be0c169e6e9a790e16f7b65cfb9c9e6cfc3e1` |
| additional_config.md | `ad8ae987345f1407314b5c377108299c6cbea0f8e0ce20605c93a0d16891d888` |

MemCache/MemFabric 1.2.0 的配套与源码已有 [官方文档核查记录](../../evidence/20261017-memcache-adaptation/official-docs.json)、[C8适配记录](../../evidence/20261017-memcache-fix/official-c8-aicpu-adaptation.json) 和 [SDK完成语义记录](../../evidence/20261017-pd-baseline/sdk-completion-source-receipt.json)，避免重新取证。此审计保留当前唯一设备优先项：先完成原生图执行对照；PD修复候选暂不扩大测试，也不宣布已修复。
