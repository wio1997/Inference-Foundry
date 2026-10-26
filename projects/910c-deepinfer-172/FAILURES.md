# 保留的失败和适用边界

| 阶段 | 实际结论 | 后续处理 |
|---|---|---|
| 原理想FP32 expert/layer、部分DSA诊断 | 曾失败；不能当作量化模型与理想FP32逐点相等 | 保留失败；另用实际源精度CPU oracle验证真实量化/算子边界，见DSA-PRECISION |
| packed generate v1–v5 | Sparse MLA metadata报507018；公开文档的ratio1与固定镜像无compressed-KV ABI不同 | 匹配实际镜像ratio0；短前缀原生eager，长上下文实际CANN |
| native serving v3 | MC2窗口要求2,145,386,496 bytes，普通200MiB组不足 | 独立MC2组2560MiB；全40层真实执行 |
| native v4 | 前缀子项挤出seed anchor，正式命中计数异常 | **INVALID**，不计性能；只停止自己进程，改为仅冷请求建立anchor |
| native v5 | 图捕获中的共享专家group_count产生CPU→NPU复制，107030 | 预分配持久device常量 |
| native v6首次 | capture返回未真正launch的输出，C8预热全文前缀不匹配 | **INVALID**，无正式合格结果；总是先replay后返回 |
| native v6-replay | 全16条正确，59.40token/s，但低于目标 | 作为原生基线保留；复用了健康进程中已捕获图，驻留来源单独记录 |
| native v7 | 全16条正确，138.83token/s，仍低于目标 | 保留基线；CPU查表和mmap优化的单项贡献未隔离 |
| native v8 | 全16条完整输出相等，668.35token/s | 固定吞吐目标通过；首token更慢、重复语料敏感性仍明确报告 |

早期raw completion未加聊天/BOS格式，立即EOS；含“2025”的文本曾被弱contains5判据误认为算术成功。
这些短smoke不计入正式验收，也不作为两DP算术正确性的证明。最终验收是同一冻结输入的16条完整1024-token输出。

完整原始历史仍在服务器项目目录；复现归档附带历史status、v4 INVALID标记、v5/v6失败progress及代表日志。
最终正式AISBench详情、逐请求事件/输出、源码SHA和真实controller均在final-native-v8.tar.gz中。
旧记录保持原始失败状态，没有用新源码或新结果覆盖历史证据。
