# M036：真实调用方到 Core 的身份核验与迁移反例

固定 SGLang 提交 `ce0a8fea5f4245839d174b2b8c55692be1810a7f` 的53个源码文件已经逐个核对 Git blob 与 SHA256。有效参考 R007 完成16组264项 CPU 检查，**248通过、16项身份隔离失败**。离线 JSON namespace 候选在同一组264项检查中全部通过，但新增的旧缓存混存检查仍失败：返回384个旧 CPU 索引。因此候选 **NOT_ACCEPTED_UNFENCED_LEGACY_MIGRATION**，没有应用到服务。整体 E0、accepted baseline=null；没有原生推理、性能或当前部署故障结论。

## 当前做法与候选

原始请求归一化接受非空字符串 extra_key/cache_salt，空字符串归为 None。真实 LoRA registry 把名称解析为 ID，并更新已缓存的批内子对象；tokenizer carrier 和 Scheduler 将这些字段交给 Req。Req 将 `(extra_key or "") + lora_id` 用作缓存 extra_key。base 请求可令 caller key 等于 adapter 的真实 deterministic ID，和 caller key 为空的 adapter 请求得到相同内部 key。[归一化](https://github.com/sgl-project/sglang/blob/ce0a8fea5f4245839d174b2b8c55692be1810a7f/python/sglang/srt/managers/io_struct.py#L823)、[registry 解析](https://github.com/sgl-project/sglang/blob/ce0a8fea5f4245839d174b2b8c55692be1810a7f/python/sglang/srt/managers/tokenizer_manager.py#L3502)、[Req 构造](https://github.com/sgl-project/sglang/blob/ce0a8fea5f4245839d174b2b8c55692be1810a7f/python/sglang/srt/managers/schedule_batch.py#L1100)。

本轮把上轮局部键反例推进到实际 normalizer → registry resolver → tokenized carrier → Scheduler.handle_generate_request → Req.init_next_round_input → 原 UnifiedTreeCore 的组合，插入并核对真实 Torch CPU 索引来源。这里仍绕过重型模块初始化与完整服务边界；它证明指定原函数组合确实返回错误 namespace 的 metadata，不证明当前 GLM-5.3 服务启用了该业务或产生了错误模型输出。

离线候选将所有非默认 base/adapter namespace 编为紧凑 JSON：`["sglang.cache-namespace.v1", extra_key, lora_id]`；默认 `(None,None)` 保持 None。它区分了新编码中的两个字段，增加序列化、键长度和维护成本，并改变所有非默认内部/存储 namespace。候选只覆盖 Req 构造函数，未应用到规范源码或生产；其余原方法继承，完整服务模块没有导入。[候选补丁](candidates/L035-versioned-namespace/namespace.patch)。

## 有效检查与反例

复用已冻结的 M001-v1 两个 CPU 输入文件：512输入/128输出预算与32768输入/256输出预算。批量分别2/3个请求，遍历 page1/128、is_eagle=False/True、PREFILL/DECODE 枚举，共16组。批内请求和 Core 操作为顺序执行，**不是2/3并发服务压力测试**。P/D 和 ASCEND 是 metadata/config 夹具，未创建引擎、传输或 MTP worker。page1 仅作 CPU 对照，不能推出 NPU 支持。

base caller key 为真实 ID `2fe67d87a2225f97aea47d7902dc978c`，adapter A 的 caller key 为空；两者 token 相同。参考实现16组均跨 namespace 命中。512输入时，page1 返回511/510项（is_eagle=False/True），page128返回384项；32768输入时分别返回32767/32766项或32640项。这些是受原查询最大前缀和页对齐约束的索引数量，**不是物理 KV 值或设备内存读写**。同身份仍命中、返回索引来源、不同 adapter 未命中、registry holder 释放等对照保持通过。[R007 原始结果](caller-cache-reference-result.json)。

R008 在相同16组264项中的失败数由16变为0，全部变化均为隔离合同项，其他结果未变。但这只适用于全新 Core。额外构造保留的旧版 base 条目，其任意 caller 字符串恰等于新 adapter 的 JSON key；随后使用候选查询，仍返回旧条目的384个索引，来源核验通过。总计17组266项，265通过、1项合同失败。[候选结果](candidate-cache-result.json)、[同条件正确性比较](correctness-comparison.json)。这不是性能基线或性能 A/B。

迁移失败带明确前提：旧 namespace 条目仍存在且没有独立世代隔离。当前服务是否热升级、是否清空缓存、是否持久化或跨层恢复，均 UNKNOWN；不能把反例写成所有部署必现。字符串版本标签位于 caller 可控制的字符串空间内，不能独立证明缓存世代。P/D、各副本、host/storage tier 的实际世代和旧状态退役必须另行闭合。上轮 streaming slot 的3项身份稳定前提反例也未被这个 raw-tree 编码补丁修复。

## 真实执行范围与外部边界

执行完整原 GenerateReqInput、SamplingParams 模块、registry/RWLock、RadixKey、UnifiedTreeCore、FULL component、actions/interfaces 与 event recorder；执行列明的原 tokenizer/Scheduler/Req 方法体。源码加载记录保留于结果中的 source_components。类加载不代表其每个方法都被调用。

CoreView 只适配 capabilities 与 match_prefix 调用，实际查找/插入由原 Core 执行。时间、metrics/rank、配置 getter、充足容量、无 grammar/DLLM callback 和捕获队列均为外部夹具。没有 HTTP/Pydantic endpoint dispatch、进程 IPC、TP/DP consensus、完整 UnifiedRadixCache 构造、实际 admission、权重、LoRA pool、NPU forward、采样或输出正确性。复用的软件为本机 Python3.12.14、Torch2.8.0、msgspec0.19.0、numpy2.3.3、transformers5.12.1 与独立 Pillow12.0.0 overlay；不代表目标镜像。

另以原 ModelRunner.maybe_init_lora_manager 检查 dispatch：enabled target 调用初始化边界一次、enabled draft 零次、disabled target 零次。源码 draft worker 明确传入 is_draft_worker=True。这加强了 target-only gate 的源码与 CPU dispatch 证据，**没有构造 manager/worker、加载 adapter，也没有确认原生 LoRA+Ascend PD+MTP 支持**。[runner gate](https://github.com/sgl-project/sglang/blob/ce0a8fea5f4245839d174b2b8c55692be1810a7f/python/sglang/srt/model_executor/model_runner.py#L759)、[draft 构造参数](https://github.com/sgl-project/sglang/blob/ce0a8fea5f4245839d174b2b8c55692be1810a7f/python/sglang/srt/speculative/draft_worker_common.py#L90)。

session radix 文档说明顶层 session_id 作为引用标签、关闭移除引用但不立即释放 KV，不能据此推出 streaming adapter 身份稳定。源码 facade 先尝试 streaming match；本轮没有执行完整 composer/facade，因此该入口仍需独立验证。[facade](https://github.com/sgl-project/sglang/blob/ce0a8fea5f4245839d174b2b8c55692be1810a7f/python/sglang/srt/mem_cache/unified_radix_cache.py#L549)。

## 失败保留与计数口径

| Run | 结果与用途 |
| --- | --- |
| R001/R002 | 固定源码收集 PASS |
| R003 | 执行前发现缺失 numpy.float64 绑定；INVALID，命令未执行、会话未创建 |
| R004 | 加载完整 TreeComponent 时缺 IntFlag；INVALID，无语义结果 |
| R005 | 完成16组248项，raw 40失败；其中24项来自错误 bootstrap_room oracle，整体 INVALID，不接受部分计数 |
| R006 | 修正批量 room 为101+i及传递检查，16组264项，248通过16失败 |
| R007 | 将正向命中/来源断言与跨命中数量解耦，作为候选比较参考；16组264项，248通过16失败 |
| R008 | 共同264项全部通过；新增2项迁移检查1失败，整体候选不接受 |

R006/R007 是同一覆盖的参考整理，不能累加成32组或528项独立覆盖。R003 没有虚构执行收据。其余7条冻结命令由 ZCode 简单执行器各执行一次，模型别名 deepseek-flash、后端 release UNKNOWN；外层 CLI0 与内层退出/语义结果分开审阅，输入哈希未变。[审计](zcode-audit.json)、[控制器审阅](controller-review.json)。所有失败脚本/收据保留，修复只发生在新文件/新 Run；无失败重跑覆盖。

## 下一步与接受门槛

优先验证 caller 不可控制的缓存世代、混版本/host/storage 生命周期，以及真实 streaming composer 的身份校验；先证明原路径边界，再冻结一个修补方向。否定条件包括任何跨 namespace 旧索引复用、同身份缓存失效、旧状态/holder 泄漏、取消重试后错误恢复，或兼容性维护成本无法接受。不能仅靠更换字符串前缀接受候选。

原生门槛仍是任务44/48资源归属、确切模型/config/量化/KV dtype、安装版本、真实扩展业务、P/D 拓扑与 MTP 参数。历史 NEXTN steps3/topk1/draft4 只作参考，当前运行值 UNKNOWN；模型 config 恢复哈希不匹配仍保留。四类扩展业务与原 GLM-5.3 的 Ascend PD+MTP 组合仍 UNKNOWN。[原生合同](native-next-contract.json)。

本轮没有时间测量、吞吐、TTFT/ITL 或硬件下界校准。JSON 成本、失效损失和必要 adapter 计算均需在正确性通过后的同条件原生证据中评价。Loop **INCONCLUSIVE**；JSON 候选拒绝无世代隔离迁移，任务保持 ACTIVE/E0，整体目标尚未完成。
