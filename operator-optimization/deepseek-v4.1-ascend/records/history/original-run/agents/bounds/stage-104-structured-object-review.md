# Stage 104：从采样参数到结构化输出标志

## 核验

ZCode `operator-ceiling169-v2-104-structured-object.exit-code` 为 0，verified 回执记录 `deepseek-flash`。`structured-object-snapshot-status.json` 记录 baseline 和 stage 两容器运行、镜像 ID 一致、目标无覆盖挂载、文件 SHA 一致；本地对快照重算 SHA 也一致：

`structured-object-snapshot/structured_output_request.py`：`6a06212f8f130555eb406ba03c92bf9e26fbfa1d747d83b8f68766ebf967490e`。

## 证明的条件链

- `structured_output_request.py:33-42`：`sampling_params is None` 或其 `structured_outputs` 为假值时，直接返回 `None`；非假值但 `all_constraints_none()` 为真也返回 `None`。
- `structured-request-snapshot/sampling_params.py:129-143` 的 `all_constraints_none()` 检查 json、regex、choice、grammar、json_object、structural_tag 六项均为 None。`:339,410,477` 显示在未显式传入时采样参数 structured_outputs 默认为 None，`from_optional` 同样如此。
- `structured-request-snapshot/request.py:85-90,283-285` 将 `from_sampling_params` 结果存为 `structured_output_request`，`use_structured_output` 判定它是否非 None；`draft-constraints-snapshot/structured_output.py:294-297` 在标志为 false 时将 draft 原列表返回。

因此，**如果传入 Request 的采样参数为 None，或者 structured_outputs 为 None/假值，或者其六项约束全 None**，且这些已核验函数及对象没有被其他代码修改，则 `use_structured_output=False`，draft validator 不扩张。这是条件性源码证明；与纯测量平均值不同。

## 不足以证明的部分

- 虽然客户端冻结配置和匹配哈希的 payload 构造源码支持未发送相关字段，尚未验证服务端 completions API 的默认、转换及服务默认/其他字段是否会创建非空 StructuredOutputsParams。因此不能仅凭 API 省略字段声称所有历史请求实际标志已为 false。
- 文件内容一致不是历史进程实际对象及每个调用的快照。即便闭合 structured-output 条件，也还须检查其他 draft 路径、sampler 实际调用及客户端 usage 到引擎事件和计时窗对应，才可以给固定实现附加 5+1 的计数条件；不构成跨实现的数字 TPS 物理上界。

本轮没有 NPU 操作，也未改变服务环境。下一步仅在能依据服务实际入口定位 completions protocol 和 SamplingParams 转换时核对该局部源码，否则保留 UNKNOWN。
