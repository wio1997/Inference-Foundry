# 验证证据

以下文件是已完成验证的可携带摘录。发布过程只做文件/哈希/提交核对，新增模型加载和模型请求均为 0。

- `acceptance.json`：范围、原版逻辑比对、组件复用与历史执行计数。
- `joint-result.json`：最终普通服务 v185 的 8 组自由采样比较、16 个采样请求和 1 个 EOS 请求。
- `frozen-audit.json`：独立封存输出审计 v186。
- `graph-dispatch-audit.json`：v188 的 16 rank 图就绪、共同派发路径及 driver 重放。原版 follower 留在 RPC 循环，任务计数仅由 driver 0/8 输出；不宣称逐 follower 的独立计数。
- `components/`：原始组件结果的 `result` 摘录及原始文件 SHA-256。v132、v153 的夹具不完整状态保留；通过部分的复用范围见 `acceptance.json`。
- `delivery-audit.json`：24 文件补丁应用与最终源码哈希核验。
- `runtime.json`：本次验收实际环境、开关及内存/模型长度参数。

原始远端日志、张量字节快照、过程失败和未派发记录保留在本地证据档案；完整出处以摘录中的原文件 SHA-256 关联。本目录不提供未发生的性能计分，历史功能通过也不代表所有模型及开关组合已穷举。
