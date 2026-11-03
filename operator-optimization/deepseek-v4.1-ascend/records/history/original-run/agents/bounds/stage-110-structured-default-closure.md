# Stage 110：结构化输出默认源码链闭合

## 哈希核验

ZCode `operator-ceiling169-v2-110-generate-protocol.exit-code` 为 0，verified 回执记录 `deepseek-flash`。`generate-protocol-snapshot-status.json` 显示 baseline/stage 两个运行中容器镜像及目标源码 SHA 一致、目标无覆盖挂载；本地重算快照 SHA 一致：

`generate-protocol-snapshot/generate_base_protocol.py`：`cbf29ae008b7b51234d74e602f1395b5cbef660fa79e65eedab8228e9c49c9ee`。

## 受条件约束的完整正常路径

1. `stage-101-local-benchmark-client-review.md` 核对了历史 client-source manifest 与本机客户端源码；正常客户端路径和冻结配置构造 `/v1/completions` 请求，不主动加入 `response_format` 与 `structured_outputs`。该结论依赖这些源码、配置与历史请求实际使用且未被改写。
2. `completion-path-snapshot/completion_router.py:35-55` 的 POST `/v1/completions` 使用 `CompletionRequest` 并调用 completion handler。`completion_protocol.py:122-132` 对这两个字段设置默认 None；相关字段的 before validators 遇省略值直接返回。若经过通常解析后协议对象两个字段确实都是 None，且不是 beam-search 分支，`completion_serving.py:168-176` 调用 `to_sampling_params`。
3. `completion_protocol.py:309-313,398` 将两个字段传到 `structured_outputs_from_response_format`，并将返回值传给 `SamplingParams.from_optional` 的 `structured_outputs`。`generate_base_protocol.py:136-142` 明确在 `response_format is None` 时返回传入的 `structured_outputs`；两值皆 None 因而返回 None。常规模型采样默认只作用于代码列举的参数，不能代替检查本函数返回值。
4. `structured-request-snapshot/sampling_params.py:410,477` 保存该 None；`structured-object-snapshot/structured_output_request.py:33-42` 对 `structured_outputs=None` 返回 None；`structured-request-snapshot/request.py:85-90,283-285` 由此使 `use_structured_output=False`；`draft-constraints-snapshot/structured_output.py:294-297` 的 validator 直接返回 draft 列表，不扩张。以上假设 sampling params 从 API 到 Request 的传输期间、以及从 Request 构造到使用前没有注入或修改相应字段、方法。

**结论**：对于两个字段均为 None 的普通 completions 请求，在所快照的源码正常执行且对象与函数未被额外修改的条件下，结构化输出不启用，该 validator 不增减 draft。这是有明确假设的源码逻辑链；不能作为历史每条请求 runtime 对象的直接观测。

## 未完成的推论

- 服务端 parse、其他输入路径、运行时 patch 与请求对象未被修改没有逐条历史证据；不能说已证明历史每次实际请求状态。
- 即使这条链的条件成立，也不能单靠它排除所有 draft 注入、替换 proposer 或服务端事件边界变化；全局每次最多 5 draft / 6 sampler 槽、客户端 token 与事件精确映射都需各自闭环。
- 计算 TPS 理论上界需要可信的必要工作/容量/依赖或事件最短间隔，客户端 token 计数、测得带宽及 profiler 采样不能充当硬件物理上限。

本轮无设备操作或目标服务修改。
