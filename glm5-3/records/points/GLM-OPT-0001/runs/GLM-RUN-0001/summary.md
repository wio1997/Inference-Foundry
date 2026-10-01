# GLM-RUN-0001 — 冷缓存流式链路诊断

真实单请求81932输入/256输出；usage、length结束和DONE、内层退出0通过。冷缓存cached_tokens=0，TTFT38.417s，首段后墙钟TPOT14.625ms，整体42.149s。原始SSE与源代码hash留166。

裁决INCONCLUSIVE（性能）：诊断链路有效，不替代61440输出SLA，不是Current KEEP或稳定容量证书。此前现场synthetic phase/controller fixture用于执行合同，非模型性能。

2026-10-01测量修订：早期parser未包含native delta.reasoning，原first_content是后续可见正文端点，不能称完整TTFT。离线读取原SSE、未新发请求；正确首输出与TPOT见first_output_reduction.json：

- GLM-RUN-0001/0: first native output 34.509184s，TPOT 29.949935ms；旧content端点 38.416966s。
