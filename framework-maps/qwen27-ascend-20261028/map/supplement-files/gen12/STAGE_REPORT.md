# gen12 阶段记录

完成 Q06 只读 observer 候选、四份 SHA 固定源码副本与九点 decorator 补丁；移除新 decorator 后原 AST 完全一致。候选不自动安装，不改原文件或 live 调度。冻结 pending 的 RID/slot/generation/pool epoch/result weak identity/local launch，planner 仅只读观察并始终 fallback；原 result/release 保留所有权。

384 个 CPU 原循环/finish AST stub 场景，原调用次数、事件顺序、输出/结束原因/队列一致。另有七个原 pool AST 操作、四个源码漂移拒绝、clear 后同 generation 不同 epoch 拒绝、返回与异常对象身份、observer 错误隔离、device generation 零 scalar 读取。首次 safe helper 漏接 kwargs 的 TypeError 已修复，失败日志保留。

aarch64 原容器只读源码与 msgspec 字段投影核验成功；这是 CPU descriptor 检查，没有 torch/sglang import，不是服务 hook 或 NPU 验证。本机 CPU stubs 为 x86_64。实际 torch CPU generation、import-loader、四 rank runtime hook、真实 GDN/cache/lifetime/TP/输出等价仍未验证。

原主 pipeline/assembler/sharder/autostage 身份均保持。观察 rank84 与 pipeline submitted82/inflight4/pending118；不把计数或存活当最终204验收。所有八卡逻辑占用与后台作业保留；本轮无 workflow lock claim、SDK borrow、NPU、service start 或 termination。

入口 Q06_SHADOW_HOOK.md；代码 shadow_hook.py；源码绑定 source-binding.json；CPU证据 cpu-shadow-results.json；容器布局证据 record-layout-runtime-receipt.json。源码副本与 explicit allowlist 包不含 transport/key/native profile。

下一轮先核对完整204 gates；若仍等待，准备模块 loader/原兼容 overlay 的 CPU 导入 fixture，随后再在原锁与精确资源核对下收集独立短 canary 的 shadow 数据。本轮候选没有启用 Q06，不代表优化已采用。最终地图需显式合并 gen8–12 并对实际204重新打包/QA；root验收/私有归档仍待办。
