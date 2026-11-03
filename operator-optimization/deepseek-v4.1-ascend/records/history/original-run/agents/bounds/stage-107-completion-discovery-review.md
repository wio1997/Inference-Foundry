# Stage 107：completions API 路径候选

`operator-ceiling169-v2-107-completion-discovery.exit-code` 为 0，verified 回执记录 `deepseek-flash`；本地重算 `completion_discovery_inside.py` 与 `completion-discovery.json` 中的 query SHA 一致。stage 固定 `/vllm-workspace/vllm/vllm/entrypoints/openai` 子树扫描 26 个 Python 文件，无读取错误/大文件跳过/候选截断；输出仅文件名、匹配标记与行号，没有请求内容。此扫描并未证明 runtime 已加载任意候选，也未核对 baseline 文件身份。

四个候选中，三个在 `completion/`：`api_router.py` 同时匹配路由和 CompletionRequest、`serving.py` 匹配 CompletionRequest 和 to_sampling_params、`protocol.py` 匹配 CompletionRequest、to_sampling_params 和 structured_outputs。另一个 `chat_completion/protocol.py` 命中部分标记；因为扫描只返回行号、不包含引用上下文，这些共同命中不证明调用关系，也不能单凭匹配排除 chat 文件的其他用途。基线实际请求地址记录为 `/v1/completions`，因此最小下一步是对上述三个 `completion/` 文件作双容器哈希快照，再检查路由、import 和调用，以源码确定或否定链条；不把 chat 标记误当成该路径的证明。

已准备 `compare_completion_path_readonly.py` 并完成 AST 语法检查，沿用目标镜像、运行状态、绝对路径、可读性、双容器哈希和传输完整性的失败即停止护栏，不调用 NPU。即便源码最终闭合，历史调用仍附带加载匹配源码及无运行时替换的条件。
