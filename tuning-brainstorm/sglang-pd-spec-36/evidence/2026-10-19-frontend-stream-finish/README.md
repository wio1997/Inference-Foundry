# M033：默认通知到 HTTP/SSE、msgpack 状态码反例与条件修复

固定 SGLang `ce0a8fea5f4245839d174b2b8c55692be1810a7f` 的前端路径已进入真实消息结构、ZeroMQ、异步通知和本机 HTTP/SSE 组件实验。普通累计/增量输出保持了文本、IDs 和 usage；可选 msgpack 路径存在一个已复现的错误表示差异：已知 HTTP 状态码解码成整数后，原 SSE 分支将其当普通 abort，遗漏 error 帧并继续发送 token-ID 扩展。候选仅在该分支前归一化已知整数状态码，同一 36 组、744 项检查由 36 项失败变为全部通过；另有 9 组、52 项实际 HTTP 检查通过。

**这是一份本机组件正确性报告。默认源码 IPC 是 pickle=True，当前目标服务的实际 IPC/依赖版本 UNKNOWN。没有部署到运行服务，没有原生 Ascend PD+MTP 结果、KV 释放证明或性能加速结论；整体 E0 / 无接受基线。**

## 当前实现与本轮变化

默认 `incremental_streaming_output=False` 时，TokenizerManager 中间输出保存 `text=None` 与累积 IDs 的引用，在 waiter 取出最新结果时再拼文字，避免每个上游片段立即重建整个前缀。可选增量模式则把排队的文本、IDs 和对应 metadata 合并。finished 状态从请求表删除后，已创建的 waiter 仍持有 ReqState，继续收到最后输出。OpenAI 层分别处理累计文本切片和直接增量，最后发送 finish、可选 sglext/usage 与 `[DONE]`。这些都是既有能力。

本轮没有改变通知批量、SSE 合并粒度、模型或采样。新增的是[条件正确性候选补丁](candidate/status-code.patch)：在 `_generate_chat_stream` 中把可识别的整数转成 `HTTPStatus`，保持原错误分支；不修改 finish_reason 字典、unknown 整数和字符串的原有回退行为，也不修改其他方法。完整候选文件与原方法 AST 审查可追溯，生产未应用。

## 冻结环境与来源

[源码索引](source-index.json)包含 18 个固定 blob、SHA256 与符号行链接：15 个在本轮读取，3 个复用 L030 中未变化的同 commit 文件并再次冻结哈希。没有把远程安装树视为重新验证过。

独立 `.venv-front` 使用 Python 3.12.14、FastAPI 0.146.0、Uvicorn 0.54.0、httpx 0.28.1、Pydantic 2.14.1、OpenAI SDK 3.35.1、orjson 3.13.0、pyzmq 27.2.0、numpy 2.5.4；具体选取和完整锁见[环境记录](environment/result.json)。这些是本机协议环境，**不是目标镜像版本等价性证明**。旧 CPU/tokenizer 环境未修改。

HF backend 仍为 transformers 5.12.1 / tokenizers 0.22.2，使用 L030 恢复的历史 GLM-5.3 tokenizer 两个精确 SHA256 文件；本轮没有换模型或下载权重。生成的是合成接受 IDs，不运行任何模型。

## 执行清单与失败保留

| Run | 内容 | 结果与范围 |
| --- | --- | --- |
| R001 | 10 个固定公开前端/协议源码 | PASS，blob 核验 |
| R002 | 建立独立前端环境 | PASS，本机依赖锁；不代表镜像 |
| R003 | 5 个 wire/schema/默认开关依赖源码 | PASS，source default pickle=True |
| R004 | 原桥接矩阵，msgspec 0.19.0 | **INVALID**：完成 8 个 pickle 场景后，在首个 msgpack token-ID 数组解码处停止；部分计数不算全矩阵通过 |
| R005 | 单独数组与完整 BatchStrOutput 诊断 | 7 条观测完成：4 个直接 ext 往返相等，3 个嵌入消息解码报错；不等于前端成功 |
| R006 | 独立 msgspec 0.20.0 overlay 与官方修复来源 | PASS；不修改旧环境，不是升级部署 |
| R007 | 相同源码/矩阵，0.20.0 overlay | **FAIL**：36 组 / 744 项完整执行，6 个 msgpack coded-abort 组合中的 36 项失败 |
| R008 | 相同矩阵与环境，仅应用状态码候选 | PASS：36 组 / 744 项，无失败 |
| R009 | 真 TCP loopback HTTP/SSE 与客户端断开 | PASS：9 组 / 52 项；其中包含原版错误行为的复现，见下文 |

所有冻结命令各由 ZCode 执行一次，输入哈希均保持不变。外层 CLI 都为 0，但 R004 内层为 2、R007 内层为 1；不能用 CLI 0 抹去失败。[审计](zcode-audit.json)只保留 alias、工具与退出元数据，后端模型 release UNKNOWN。失败/invalid 的结果、回执、原始日志哈希全部保留。

## 先排除依赖前提：0.19 的 memoryview 边界

R004 尚未走到状态码分支，就在原数组 ext_hook 出现 `bytes length not a multiple of item size`。R005 发现直接 ext 长度正常，而完整消息中 callback 看到的 shape/长度与实际 ext 不符。官方 [msgspec PR #836](https://github.com/msgspec/msgspec/pull/836) 已修复导出 memoryview 的 shape 更新；[官方 changelog](https://github.com/msgspec/msgspec/blob/main/docs/changelog.md)将该修复列入 0.20.0。

仅叠加隔离的 0.20.0，保留原源码、其余依赖与用例，R007 完整读过所有数组并走完矩阵。这个对照支持**本机 0.19 与所用数组扩展路径的兼容性前提**，不构成目标镜像必须升级的结论。真实 target msgspec 版本与所选 IPC 均未确认。官方原始 API 结果留在本地，公开[来源摘要](dependency-upstream-summary.json)含 URL、merge commit 与哈希。

## 语义反例与候选对照

`BatchStrOutput.finished_reasons` 的固定类型是 `Dict[str, Optional[Union[str,int,List[int]]]]`。相同 `HTTPStatus.SERVICE_UNAVAILABLE` 在 pickle 后仍是 HTTPStatus，在 msgpack 的精确结构解码后变成 int；原 `_generate_chat_stream` 的 `isinstance(status_code, HTTPStatus)` 因而漏掉错误分支。

| 同一输入 | 原版 pickle | 原版 msgpack 0.20 | 候选 msgpack 0.20 |
| --- | --- | --- | --- |
| 普通输出 / graceful abort | 通过 | 通过 | 通过 |
| coded abort 400 / 500 / 503 | SSE error | 普通 abort finish，并发送 sglext IDs | 对应 SSE error，省略 finish / IDs |
| 599 或字符串 `"503"` | 源码为回退，未单独 HTTP 实测 | 源码为回退，未单独 HTTP 实测 | HTTP 检查确认回退行为 |

[R007 原始反例](bridge-baseline-result.json)与[R008 候选结果](bridge-candidate-result.json)完整保存所有组合。两者逐组配置、检查函数和主矩阵 AST 相同，见[同条件正确性比较](correctness-comparison.json)。不是性能对照，也不是 744 个独立请求用例。

矩阵覆盖 2/3 个交错请求、128/256 个输出 token、默认累计/可选增量模式、正常与有意排队的消费者、通知 size16 与两个 size1 交错场景、graceful 与 coded abort、派发幂等和失败后重试。合成接受步长为 4，前端上游仍用原增量反分词方法，消息使用**完整原 BatchStrOutput / AbortReq 类型**与原数组 hook；真实 ZeroMQ inproc 往返。全量 IPC registry 缩小为所测 carrier，不证明其余消息类型、跨进程生命周期或 P/D 通信。

protocol、sse_utils、usage_processor 三个文件完整导入，原 TokenizerManager / Chat 的有关方法完整抽取执行。重型包初始化、服务构造、GenerateReqInput 归一化、generate_request 前处理、模型和调度器均有明确夹具边界；没有把它们称为完整 serving import。

## 本机实际 HTTP 与客户端断开

[R009 结果](http-result.json)使用任务自有 Uvicorn/FastAPI/Starlette，监听 `127.0.0.1` 临时端口；httpx 真实收取并解析 SSE，随后关闭服务器。没有打开浏览器，也未连接未归本任务的服务。

- 默认 pickle / 累计模式与候选 msgpack / 增量模式的文本、token-ID 扩展、usage 和 DONE 均保持一致；命名 `sglext_ids` 事件也已接收。
- 原版 msgpack 的 503 场景在 HTTP 上复现为普通 abort finish，**该行为没有被接受**。候选则发送 code503 error 与 DONE，省略 finish/IDs。R009 的 PASS 表示“已复现并区分原版与修复”，不是原版 bug 通过正确性。
- 预先注入的首帧前 ValueError 得到 HTTP400 JSON；首帧后 ValueError 得到 HTTP200 流中的 SSE400 与 DONE。这里验证封装语义，错误来源是夹具，不冒充真实模型校验/OOM。
- 真实客户端在可见文字后关闭连接，原 `create_abort_task` 保持其 2 秒等待，随后派发一次精确 AbortReq（又经过原 wire 路径）。`abort_sent=True`，RID 仍等待真实 scheduler 的 terminal 响应。**因此验证止于 abort 派发，不证明 scheduler 已终止或 KV 已释放**；finally 的夹具清理不算回收证据。

## 新增成本、风险与适用边界

候选仅在 finish_reason 路径做本地整数判断与已知 HTTPStatus 转换；未知整数可能走一次 ValueError 回退。没有测量这些操作的性能，也不把它作为加速措施。源默认 pickle 下原来正确的枚举保持不变；实际服务如果没有启用 msgpack，这个已知编码差异未必会触发。

真实镜像的 msgspec 与框架版本、完整服务导入、P/D 端实际 IPC 一致性、GLM reasoning/tool parser、可选 logprob/beam/MM/custom-info、权重更新和真实调度器取消链仍需验证。本轮 time/logger/metrics/exporter 为夹具，时间值不得用于性能结论。不同 case 的帧数受异步调度影响，也不是原生模型事件分布。

## 裁决与下一步

Loop **INCONCLUSIVE**：局部协议反例已成立，候选在冻结 CPU/HTTP 范围得到支持；原生目标、完整服务回归与物理 KV 生命周期仍未完成。没有 native 接受候选、运行服务修改或性能提升结论。

[后续原生合同](native-next-contract.json)先冻结真实 image / runtime / IPC 与依赖，保持原 GLM-5.3、Ascend、PD、推测解码和 2×512→128 / 3×32768→256 负载，再验证取消至真实 KV/slot 释放、可选解析器与输出协议。资源44/48尚未分配。下一轮可推进原36机制中尚缺独立证据的 M032 默认采样/grammar 路径，避免反复累计已覆盖的前端断言。

历史参考仍为 CLI NEXTN / steps3 / topk1 / draft4，当前规范化运行态 UNKNOWN；本轮没有把它改成单步或切换模型。

[复现边界与路径说明](replay-notes.md)记录本机冻结命令、依赖二进制、tokenizer恢复和新Run要求，避免将研究快照当作可直接部署的服务补丁。
