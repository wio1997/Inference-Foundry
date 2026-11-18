# 原生0.32：四档选定长上下文语义与缓存检查

固定同一W4A8权重、模型tokenizer和原生epoch630。使用官方chat模板渲染的有效UTF8文本，分别精确32768/65536/131072/262144输入，文档约12.5%、50%、87.5%处记录三个合成值，严格返回三个值及分隔符。请求T0/seed0、max128、非流式，逐档首次与立即重放；这些C1短输出案例与C16/1024性能矩阵独立。

[32K/128K/256K父级核验](../evidence/native_long_quality_G718/G718_parent_remaining_long_quality_assessment.json)和[64K父级核验](../evidence/native_long_quality_G718/G713_parent_native64_semantic_cache_assessment.json)证明八次响应的三个值均严格正确，同档请求字节一致、输入长度准确、finish=stop；父级独立验证原始响应SHA并重算实际P/D计数。

| 输入tokens | P重放命中tokens | D重放命中tokens |
|---:|---:|---:|
| 32768 | 32640 | 32640 |
| 65536 | 65408 | 65408 |
| 131072 | 130944 | 130944 |
| 262144 | 262016 | 262016 |

[冻结资产manifest](../evidence/native_long_quality_G718/frozen_assets_manifest.json)记录请求SHA、合成值oracle、实际token IDs SHA、tokenizer SHA及与448个预留benchmark前缀无碰撞。实际服务chat-tokenize → detokenize → completion-tokenize与挂载tokenizer编码一致。完整请求文本保留受控证据包，供SGLang原字节复用；[客户端源码](../code/native_readiness_G659/long_context_quality_g703.py)和原始响应/metrics已归档。API cached_tokens仍填0，命中结论来自两端真实prefix counter。

这只证明四档选定检索和立即缓存重放，不是统计性的长上下文质量认证，也不证明C16大工作集热命中。完整GSM8K1270/1319属于独立短上下文质量结果；raw-ID UTF8边界及129-ID greedy差异仍保留未解决状态。SGLang需复用同一冻结输入和标准。

另已在P实际镜像中读取CANN toolkit9.2.1（V100R001C12SPC002B201、aarch64）；[实际元数据](../evidence/native_long_quality_G718/G701_tokenize_schema_2203_parsed.json)保留来源。同一镜像的D未在本次单独读取该文件。不能将此与SGLang旧A2的CANN9.1视为相同依赖环境。
