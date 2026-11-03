# Stage 109：completions 路由至 structured sampling 参数

## 哈希与来源

ZCode `operator-ceiling169-v2-109-completion-path.exit-code` 为 0，verified 回执记录 `deepseek-flash`。`completion-path-snapshot-status.json` 记录两个运行中容器镜像相同、三个源码目标无覆盖挂载且哈希相同；本地对三个快照重算 SHA 均一致：

- router `b93b00298a74a62a3f2d8da1c0a9380a2be5823f5ab7ab99a2c911906280d769`
- serving `75d05ddddf303734a078a3e42fcb173fef58ea56da09bc0b5c7525cb9bbbf527`
- protocol `5f7930635fd3924826015211032fb298206b27c7f04cd6cd2224bb128e4a3a50`

## 源码调用与默认条件

1. `completion-path-snapshot/completion_router.py:10-14,35-55` 注册 POST `/v1/completions`，参数类型为 `CompletionRequest`，调用 `handler.create_completion`。保存的基线日志确认该路由与 HTTP 200，但本快照并非历史每次函数调用记录。
2. `completion_serving.py:168-176,193-211` 中，若 `request.use_beam_search` 为 false，调用 `request.to_sampling_params(...)`，之后将 SamplingParams 传给 engine_client.generate；beam search 是另一分支。基线 AISBench 的匹配哈希构造源码和冻结生成配置未向 payload 显式加入 beam search，但这里不声称排除未检查的运行时改写。
3. `completion_protocol.py:122-132` 的 `response_format` 与 `structured_outputs` 均默认 `None`；`:425-432,458-464` 的字段相关 before validators 在各字段省略时直接返回输入。`:309-313,398` 没有直接传默认值，而是调用 `structured_outputs_from_response_format(self.structured_outputs,self.response_format)`，再把结果交给 `SamplingParams.from_optional`。因此两个字段均为 None 时 helper 的返回值目前**不能仅根据名称推断**；其被导入源于 `vllm.entrypoints.generate.base.protocol`，该文件还未快照。
4. 基线客户端 payload 构造、参数及日志证据见 `stage-101-local-benchmark-client-review.md`；服务器收到两个字段均省略并最终得到 None，还要满足该调用链无其他注入/validator 合成，以及真正经过这些匹配文件的历史条件。链的后半段（采样参数结构化值为 None 到 Request 标志 false、validator 原样返回）已由 `stage-104-structured-object-review.md` 证明。

## 最小剩余动作

已准备 `compare_generate_protocol_readonly.py`，只核对协议源码明确 import 的 `/vllm-workspace/vllm/vllm/entrypoints/generate/base/protocol.py`；两容器路径/哈希不一致或不可读时停止。只有核验 helper 对 `(None,None)` 的返回及现有验证器、无改写条件，才可以将源码链收紧。即便完成，该条件链不证明完整服务实际所有事件的 draft 上限、客户端 token 的逐事件映射或数字 TPS 硬上界。

本轮未操作设备，未修改服务。
