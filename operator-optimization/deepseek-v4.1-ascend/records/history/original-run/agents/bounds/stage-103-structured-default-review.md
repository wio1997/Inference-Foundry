# Stage 103：采样参数和请求对象的结构化输出默认链

## 核验

ZCode `operator-ceiling169-v2-103-structured-default.exit-code` 为 0，verified 回执记录 `deepseek-flash`。`structured-request-snapshot-status.json` 记录两容器均运行、镜像 ID 相同、目标无覆盖挂载、对应文件哈希一致；本地对两文件 SHA 重算与状态记录一致：

- `structured-request-snapshot/request.py`：`da610d5203e72d2d524adc5fb39df11502a23520f2f31d9a44cf13d343f9c25d`
- `structured-request-snapshot/sampling_params.py`：`f21efc98bf8ce8aab88eb67adb4207285e509a167d976e14d3e2e8282e024bea`

仅证明源码在采集时在两容器一致，不直接证明历史所有请求的 runtime 对象。

## 已确定路径

1. `sampling_params.py:339` 的 `SamplingParams.structured_outputs` 类字段默认 `None`；`:410,477` 的 `from_optional` 参数默认 `None`，并将其传入新参数对象。`:1074-1081` 中 structured 验证在字段为 `None` 时直接返回。明确提供其他值或上游构造时重新指定该字段，不适用该默认结论。
2. `request.py:238-265` 将 `EngineCoreRequest.sampling_params` 传给 `Request` 构造；`:85-90` 再调用 `StructuredOutputRequest.from_sampling_params(sampling_params)`；`:283-285` 的 `use_structured_output` 恰为产生的 `structured_output_request is not None`。`:181` 新请求初始化 `spec_token_ids=[]`，这是后续归纳草稿长度的初始条件，不足以排除其他模块的后续注入。
3. `structured_output.py:294-297` 若 `request.use_structured_output` 为 false，draft validator 返回原列表，不扩张。详细上下游分析见 `stage-101-local-benchmark-client-review.md` 与 `baseline-structured-and-prefill-scope.md`。

## 剩余缺口

- `request.py:24` 从明确模块 `vllm.v1.structured_output.request` 导入 `StructuredOutputRequest`，但本轮快照不包括其 `from_sampling_params` 实现。不能仅由名字断定 `sampling_params.structured_outputs is None` 时一定返回 None。
- API completions 请求省略结构化字段后，是否在构造 SamplingParams 时传入 None，还是根据其他字段/服务默认值合成参数，尚未由服务端协议及转换源码证实。客户端源码及配置支持无字段，但无法替代此映射。
- 全链还依赖这些文件在历史运行时被使用、相关对象不被其他代码改写；当前不是对象的动态检查。缺口未闭合前，不能把全部请求的 validator 都视为无结构化分支，更不能据此给出全局 TPS 硬上界。

最小进一步取证：首先双容器哈希核对 `/vllm-workspace/vllm/vllm/v1/structured_output/request.py`，沿 `from_sampling_params` 看 None 条件；之后仅根据实际服务端 API 入口找到 completions protocol 及其到 SamplingParams 的构造。若缺失、路径/哈希不一致则保持 UNKNOWN。本轮没有 NPU 操作及目标服务修改。
