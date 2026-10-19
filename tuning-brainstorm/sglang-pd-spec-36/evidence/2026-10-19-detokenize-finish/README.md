# M033：真实 tokenizer 的增量反分词、发送间隔与 finish 组件证据

固定公开 SGLang `ce0a8fea5f4245839d174b2b8c55692be1810a7f` 的 13 个文件已核验 Git blob。历史 GLM-5.3 tokenizer 两个文件从公开仓库恢复，SHA256 与历史镜像记录完全一致；独立 CPU 环境使用历史相同的 transformers 5.12.1 / tokenizers 0.22.2。24 组反分词场景与 28 组完整发送累积器场景均通过，确认了现有组件能力和一个配置组合的可见输出风险。本轮没有服务补丁、原生推理或性能结论，整体仍为 **E0_SPEC_ONLY / 无接受基线**。

## 当前路径与已有能力

请求经 `Req.init_incremental_detokenize` 保留最多 5 个输入 token 的 surrogate context，并通过 `send_decode_id_offset` 只发送新 IDs。`_GenerationStreamAccumulator.accept` 每次接收一个请求的当前接受结果，生成一行 payload；默认 `stream_interval=1` 已经按一次接受块发送，不是逐个接受 token 调一次 detokenizer。分组批解码、空片段过滤、分块文本暂存都是现有实现，本轮不将其作为新增优化。

`DetokenizerManager` 维护每个 RID 的 token / committed text / sent offset。遇到 UTF-8 不完整片段时，可以先发 printable 前缀，等更多 token 后重试；后续通过 pending 偏移避免重复发送。接到 finished reason 后，它拼接最后文本、裁剪 stop，并删除该 RID 的解码状态。`TokenizerManager` 另有请求文本累积、通知与请求表删除；OpenAI 路径最终封装 SSE，使用 `[DONE]` 和后台 abort task。后两段本轮只有源码追踪，未作为运行成功计分。

固定源码默认值为全局 stream interval 1、请求 override None、batch notify size 16、incremental streaming output False、sampling 的 spaces_between_special_tokens True。它们是**源码默认值**，不是本任务当前运行配置；R004 显式覆盖 skip/space 组合，R005 的 space=False 是夹具设置，不冒充完整默认配置。[源码索引](source-index.json)保存固定 commit、文件哈希和精确行链接。

## 实际执行与比较范围

| Run | 执行内容 | 结果与边界 |
| --- | --- | --- |
| R001 | 10 个公开固定源码文件 | Git blob SHA1 + SHA256 核验；未重读远程安装树 |
| R002 | 公开 GLM-5.3 tokenizer 恢复 | HF revision `ba32f8c19c8738e2c57c71798c601d477a7bf844`；两个历史 SHA256 完全匹配；未下载权重 |
| R003 | 创建独立 `.venv-detok` | Python 3.12.14 / transformers 5.12.1 / tokenizers 0.22.2；旧 Torch CPU 环境未改 |
| R004 | 原样选取完整 Req / detokenizer 方法与两个完整状态类，实际 HF backend | 24 组，12,291 次断言；不是 12,291 个独立用例 |
| R005 | 原样完整 `_GenerationStreamAccumulator` 类 → 进程内 carrier → 原 detokenizer 方法 | 28 组，17,224 次断言；carrier / 序列化 / weight version 边界明确 |
| R006 | 3 个发送默认值、hook 与 sampling 参数源码 | 公开固定 blob 核验；实际 runtime 未确认 |

[反分词结果](detokenizer-cpu-result.json)和[累积器结果](accumulator-cpu-result.json)保留逐组配置与观测。大体量重复断言清单留在本地原始结果，公开视图保存其哈希、数量与逐组汇总；不是丢弃失败。全部六次冻结命令各执行一次，外层 CLI 与内层退出码均为 0，输入哈希保持不变；[ZCode 审计](zcode-audit.json)只公开模型 alias、调用与退出元数据。实际后端模型 release UNKNOWN。

R004 用 2 / 3 个交错 RID、128 / 256 个输出 token 的**合成组件序列**，接受块为 1、2、4 与混合 1/4/2/3，比较分组与逐行解码的每一条文本增量，并对照同一 tokenizer 对相同 IDs 的全量 decode。这不是原生 512 / 32768 输入 token 的 P/D 请求实验。分组路径累计观察到 402 次不完整 UTF-8 前缀、198 次已发送但尚未 commit 的 pending 状态；最终文本与 IDs 无重复或丢失。

另覆盖字符串 stop / no_stop_trim、stop token、同一次接受块中 stop 后 IDs 排除、收到 abort-shaped finished reason、RID 复用、空 batch、上下文 placeholder clamp、混合 skip/space 选项、空片段过滤、文本块 materialize。slow 路径只是同一 HF backend 的 slow-protocol facade，不是另一种真实慢 tokenizer。

## 配置反例：间隔与接受步长不一定组合成平滑发送

原发送条件在 interval > 1 时是 `len(output_ids) % interval == 1`，并非“自上次发送已累积 interval 个 token”。R004 先执行其原 AST 表达式；R005 再执行完整累积器确认路径。

| 接受步长 | stream interval | 输出预算 128 时每个请求的非结束事件 | 首次发送位置 |
| --- | --- | --- | --- |
| 4 | 1 | 31 | 4 tokens |
| 4 | 2 | 0 | 128 tokens，结束行 |
| 4 | 4 | 0 | 128 tokens，结束行 |

2/3 个交错请求、输出预算 128/256、步长 1/2/4 与间隔 1/2/3/4 的 24 个组合均完成最终输出；上述等差序列会一直错过余数 1，所以只到 finish 分支才强制发送。请求级 interval=1 能覆盖全局 interval=4。stop 前缀抑制与 finish bypass 在完整累积器场景中也已执行。

这是**受控恒定接受步长的反例**，不是已观测的原生接受分布，也不是当前服务事故。真实 MTP 接受长度通常随步变化，需要采集其分布与时间线。不能直接通过提高现有 interval 假定获得有界可见延迟，也不能将更晚发送视为模型吞吐提高。源默认 interval=1 下的“每次接受块一次请求输出”已有实现，不制作重复功能补丁。

## 反例和未覆盖边界

- capacity=2 同时收到 3 个 RID 时，原 LimitedCapacityDict 会驱逐最早状态，原 detokenizer 随后抛出带 `SGLANG_DETOKENIZER_MAX_STATES` 建议的 RuntimeError。这是受控容量边界，未声称当前服务溢出或泄漏。
- stop 的最终 token 位置由夹具给出；执行了原 through-stop / 增量 IDs / 前缀检查 / 文本裁剪，未执行完整 scheduler 的 stop 匹配、采样或模型。
- R005 的 BatchTokenIDOutput 是 SimpleNamespace 进程内载体；wrap_as_pickle 为显式 identity 边界，空 weight-version events 的计算为 None 边界。未验证 msgspec / ZMQ / pickle 编码、跨进程通知、权重版本计算。
- 未导入完整服务模块。原样方法抽取绕过重型构造 / 环境依赖；beam、可选 logprob / hidden / routed / MM / custom-info 分支不在 R005 范围内。
- 接到 abort-shaped finish 后删除 detokenizer 状态，不证明真实 client disconnect → scheduler abort → KV / request-table 释放，也不证明慢客户端背压、SSE usage / finish 序列或重试正确性。
- HF 相同版本和文件身份不代表本地 macOS CPU 等同目标镜像或 Ascend。没有设备工作、HTTP 客户端可见延迟、P50/P99、同条件性能对照。

## 下一步与裁决

Loop **INCONCLUSIVE**：组件假设获得限定支持，原生 PD+MTP、完整协议与取消回收尚未验证。没有候选被接受，整体 E0。top-k=1 复制方向因缺少原生暴露成本 / 别名与图执行证据暂留假设，不做盲目性能补丁。

[原生后续合同](native-next-contract.json)固定原模型、Ascend、PD 与推测解码，先测未改动的 Python 输出路径，关联 token-ready → accept/payload → detokenizer → HTTP → client 的时间线。使用 2×512→128 与 3×32768→256 用例前须通过容量与实际 tokenizer / runtime 冻结。可见延迟预算 UNKNOWN 时不做合并参数优化；先查是否确有输出端暴露成本。44/48 资源分配仍未落实，未使用未归本任务的历史服务。

历史参考参数仍为 CLI NEXTN / steps3 / topk1 / draft4；当前规范化 runtime UNKNOWN。本轮没有将其改成单步，未迁移模型、关闭 PD / 推测或缩短原生负载来制造结果。
