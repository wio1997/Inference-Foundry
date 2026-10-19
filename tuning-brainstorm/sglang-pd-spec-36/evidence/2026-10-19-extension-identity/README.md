# M036：扩展身份、局部反例与组合支持门槛

本轮核验固定提交 `ce0a8fea5f4245839d174b2b8c55692be1810a7f` 的31个相关源码文件。原始 CPU 组件完成8组54项检查：**50通过、4失败**；另一组9项 LoRA 加载元数据门槛检查通过。此前4次加载器 INVALID 全部保留、部分检查不计入结果。**原生推理未运行，完整身份合同未通过，无性能结论、候选接受或服务变更。**整体 E0，accepted baseline 为 null。

## 当前做法与本轮方向

会话维护历史及复用状态，LoRA 维护 adapter 身份和在途引用，媒体预处理根据内容与处理设置建立身份，混合状态模型还需要独立的递归状态生命周期。这些均是已有能力。本轮先检查身份和门槛，不启用不存在的业务扩展，也没有执行 pin 策略性能 A/B。

| 分支 | 固定源码能力与门槛 | 原 GLM-5.3 业务存在性 / Ascend PD+MTP 组合 |
| --- | --- | --- |
| 会话 | session_params 重建上下文；session_id 可作 radix 标签；流式 slot 保存 KV metadata，单次在途追加，拒绝 replace/offset/drop_previous_output | UNKNOWN / UNKNOWN |
| LoRA | registry 引用计数、稳定启动 ID、加载与 pin 留槽门槛；存在 Ascend 后端。兼容性 hook 声明 target-only、draft 未适配，实际 target/draft 配置传播未执行或完整追踪 | UNKNOWN / UNKNOWN |
| 多模态 | 真实媒体 snapshot、内容 digest、artifact key 与 processor fingerprint helper | UNKNOWN / UNKNOWN；CPU artifact key 不等于引擎 KV 身份 |
| 混合状态模型 | 配置注册、text-config 解包、predicate 与 backend 路由 | UNKNOWN / UNKNOWN；无递归状态运行实验 |

UNKNOWN 不改成“未启用”或“不支持”。历史原模型 config 的恢复 SHA 不匹配仍保留，tokenizer 匹配不能替代模型身份。任务44/48资源仍待确认，安装参数、真实请求与拓扑不可用。本轮 CPU 的2/3个异步 holder、小 token 数组与2×2图像，均不是 c2-short / c3-long 原生服务负载。

## 两类条件身份反例

**1. base 与 LoRA 的 caller key 可在当前构造中重合。**原 `Req.__init__` 将 caller extra_key 与 lora_id 直接拼接。使用真实 `LoRARef.deterministic_id` 得到32位 ID，构造同 token 的两请求：base 的 caller extra_key 等于该 ID、lora_id=None；adapter 请求的 caller extra_key=None、lora_id 等于该 ID。原构造函数得到相同 extra_key；原 RadixKey 得到相同 child key，并匹配了完整2-token前缀。普通 base/adapter、两个不同 adapter、不同 salt 的对照检查通过。该 ID 来源于 name+NUL+path，并非权重内容 digest。[构造路径](https://github.com/sgl-project/sglang/blob/ce0a8fea5f4245839d174b2b8c55692be1810a7f/python/sglang/srt/managers/schedule_batch.py#L1100)、[RadixKey](https://github.com/sgl-project/sglang/blob/ce0a8fea5f4245839d174b2b8c55692be1810a7f/python/sglang/srt/mem_cache/radix_cache.py#L166)。

**2. 流式 slot 的局部入口依赖 namespace 稳定前提。**原 SessionSlot 保存并恢复实际 ReqKvInfo 对象；原 find_active_slot 按 session ID、持有状态和 pre-abort 查找。分别改变 incoming adapter、caller extra_key、cache_salt，局部方法仍返回旧 slot，随后恢复同一 metadata 记录。因此“身份改变应拒绝旧 slot”的3项合同检查失败。同身份对照通过。源码 facade 先尝试 streaming match，再走 raw tree；该 facade、实际池和设备 forward 尚未运行。[slot 入口](https://github.com/sgl-project/sglang/blob/ce0a8fea5f4245839d174b2b8c55692be1810a7f/python/sglang/srt/session/streaming_session.py#L171)、[facade](https://github.com/sgl-project/sglang/blob/ce0a8fea5f4245839d174b2b8c55692be1810a7f/python/sglang/srt/mem_cache/unified_radix_cache.py#L549)。

两类都是**原始局部 CPU 路径见证**。测试中的 metadata row=3 是字段值，没有分配物理 KV。上游 API、完整 composer、实际开启组合和业务约束仍需闭合；不能将这些结果称为当前服务错误输出、泄漏或已证实的 Ascend 故障。尚未提交修补。

## 获得限定支持的行为

[完整 CPU 结果](extension-cpu-result.json)按组保留每项布尔结果及边界：

- registry 的2/3个 holder、批量 None/重复项、无部分计数增加、unregister 后等待最后 holder、等待取消/重试及同名新 ID 的隔离；实际 RWLock 与 ConcurrentCounter 工作，没有 scheduler 权重加载。
- 原 Session.create_req / finish_req / abort_req 的追加、单在途拒绝与中止后 token 数组修复；输出 token、sampling 属性是夹具，没有真实 MTP 接受或模型输出。
- 真实 Pillow/numpy/Torch CPU snapshot 在原对象修改后保持独立，dtype/mode/内容变化与 artifact 处理参数分离，mapping 次序稳定。processor fingerprint 由夹具提供；没有执行编码器、文件/URL读取、实际处理器配置或 token/pad/KV 复用链路。
- 原 LoRA speculative compatibility 函数的8项配置谓词，包含 EAGLE 与相应拒绝条件。getter 提供已解析配置夹具，未运行真实 ServerArgs 归一化；原始 NEXTN 在该函数被拒绝符合它要求先归一为 EAGLE 的前提，不是 NEXTN 服务故障。

[另9项检查](lora-gates-result.json)运行完整 LoRAConfig.from_dict 与原 validate_new_adapter：拒绝新增词表、DoRA、重复名称和占满最终槽的 pin；已有词表 token 被正确过滤。Manager 构造和 memory_pool 未实例化，所以允许通过不代表 rank/shape、权重、量化或原生内核支持。Ascend 后端源码使用 NPU 算子并拒绝额外 embedding，但本机未导入或运行该后端。[加载门槛](https://github.com/sgl-project/sglang/blob/ce0a8fea5f4245839d174b2b8c55692be1810a7f/python/sglang/srt/lora/lora_manager.py#L296)。

## 运行与身份记录

R001/R002 固定源码收集 PASS；R003 官方 Pillow12.0.0 独立 overlay PASS，原环境未改。R004 缺 functools.total_ordering、R005 缺 enum 绑定、R006 carrier introspection 的未解析类型、R007 ImageData 导出缺失，均 INVALID；修复只在新版本/新 Run，原件未改。R008 完整执行并 FAIL（4项合同失败）；R009 仅元数据门槛 PASS，不覆盖 R008。

本机 Torch2.8、msgspec0.19、numpy2.3.3、transformers5.12.1 与 Pillow overlay 分别记录，不充当目标运行镜像身份。九条冻结命令均由 ZCode 简单执行器各执行一次，外层 CLI0 与内层结果分开审计；详见 [zcode-audit.json](zcode-audit.json)。Req 只提取完整构造函数和指定 helper，重型包初始化绕过；时间、metrics、rank、sampling 与 resolved-config 为明确外部夹具。加载进 namespace 的类并不代表其所有方法已执行。

## 下一步、成本与否定条件

先追踪 raw GenerateReqInput / tokenizer carrier / scheduler / UnifiedTreeCore 的身份端到端传递，以及 target/draft LoRA 配置和流式会话调用方稳定性。只有证明业务可达、合同确需禁止后才形成具体修补；新增身份编码、会话 guard 或失效必须计缓存兼容性、旧状态释放和额外维护成本，不能仅改测试使其通过。

原生进入条件见 [native-next-contract.json](native-next-contract.json)：任务持有资源、原模型身份、真实开启分支、安装版本和 Ascend PD+MTP 全组合先确认。保持历史 NEXTN steps3/topk1/draft4 参考，不推为当前参数。确有热点 adapter 且加载受限时，才对照一个已有 pin 策略，保护冷请求和 base 请求，并测 HBM、加载字节/次数、TTFT/ITL、接受长度与 SLO goodput。

必要的编码、首次加载、adapter 计算和递归状态工作不会因缓存估计消失。按实际重叠建立关键路径；当前硬件速率、必要字节与收益均 UNKNOWN。错误共享、重复输出、重试/取消后状态遗留、pin 挤压、接受率下降或维护成本抵消收益均否定候选。完整身份正确性未通过时不接受任何性能结果。**L034 INCONCLUSIVE，整体 E0，目标仍未完成。**
