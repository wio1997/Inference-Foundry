# 当前交接 — 2026-10-17 19:24Z

本轮全 SDMA 官方原生 PD 已完成部署和性能基线；现场实验已结束。P11、D12均退出0，官方proxy v2已关闭，两台各8card/16die的npu-smi均无计算进程。原预留UID32=3b3ad135-298d-4a5d-96d1-b5070a34dc90、33=6049992b-4112-4bd2-9578-0a2bbde40314已正常删除，19:23Z复核均不存在；没有续租。不要再把旧CID/API/holder当live事实。只操作了本任务实例/holder，系统和访问入口保留。

MTP、已有共享专家重叠和原生graph的收益归为官方基线完善：P11/D10两轮formal240/240、同arrival925/925、持续2115/2115及保护/复用通过。两种合成负载持续吞吐256/1024为208.67/165.03tok/s，对无MTP84.45/77.61为几何均值2.29224×；不作为原创机制收益，不等同生产容量。全32rank旧硬件地图复用，P11/D10新增实际rank0窗口SDMA原始数据与trace均已Mac逐文件SHA备份。

自定义worker v1/v2均无无条件收益，未采用；v3/v4没有自动继续。最新p-first-publication v1改变发布依赖：原生聚合KV完成后发布P已有last_token_id，保留D全部原生工作/history/async占位/资源回收，核对并只抑制一个重复token。限制greedy文本ignore_eos/min0且无stop/logprobs/penalties等，其他fallback；不宣称P/D首token普遍可互换。冻结source SHA4f42984e953b1d126bc2f9e4d35dc5c47c6d5623136a5b97df99a9f96b72812e。

该原型smoke60/60、两轮formal240/240、48/48 eligible数值oracle（D10同48/48）、24 C4 native/PD各24/24、16短输出/取消复用pairs和8个HTTP1↔4转换均通过；不证明某个物理KV block的回收。原12算术仍11/12，既有2+3=4错误保留，未完成全模型精度验收。最终实际提前发布/核对次数及18次明确取消见p-first-publication-verdict-v1.json，mismatch0、推理error0。

两轮formal对P11/D10：TTFT p50等权几何均值0.80080，但第二token1.00424、最后token1.00491基本没快；首二gap2.46130、平均TPOT1.12897、吞吐0.99201。实际draft轮数3183→3240，无减少工作证据。相同到达两轮925/925通过、发送max0.30905ms：256 TTFT p50 442.99→328.83ms，1024 629.86→508.71ms；平均TPOT10.99→12.81、11.89→13.49ms。原生后台phase accounting在16/16 formal窗口产生负prefill-duration增量，需随发布时序适配；性能结论只用独立client monotonic时序。

结论：限定范围内提前首token有效，但没有减少D重复工作、后续等待扩大且后台phase统计有回归，**不作为默认优化采用**。新鲜同文件gate0 D13控制触发原900s预算门槛，未启动；离线D13计划lifetime1且原预算已结束，不能直接launch。对照仍有不同D实例/续写混合的限制。

下一最小实验应研究真实状态交接：确认full prompt KV已就绪及各C8/Indexer组布局，再使P首token成为D首query输入，替代D重复的最后prompt query；先用原生Request/NewRequestData检查history、max_tokens/usage/stop与async归属，再验证真实前两个D步骤及第二token时延。还需修正phase事件顺序。不要从旧trace直接推改变批次/工作后的成本，不为普通动作增加审批/平台，也不因已有版本而自动继续。

当前结果入口：analysis/official-pd-optimization-20261017.md；evidence/20261017-official-pd-optimization/p-first-publication-verdict-v1.json、p-first-v12-full-token-timing-v1.json、p-first-v12-same-arrival-vs-native-v11-v1.json、p-first-server-phase-metrics-audit-v1.json。全部正式raw、模型退出日志和profile均有Mac SHA核验归档。早期环境、MemCache SDK崩溃隔离修复及历史Run见analysis/handoff-history-through-p11-20261017.md。继续draft PR2，不自动合并，不主动向其他聊天发消息；主控直接读本文件。
