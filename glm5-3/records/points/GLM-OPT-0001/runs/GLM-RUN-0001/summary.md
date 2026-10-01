# GLM-RUN-0001 — 冷缓存流式链路诊断

真实单请求81932输入/256输出；usage、length结束和DONE、内层退出0通过。冷缓存cached_tokens=0，TTFT38.417s，首段后墙钟TPOT14.625ms，整体42.149s。原始SSE与源代码hash留166。

裁决INCONCLUSIVE（性能）：诊断链路有效，不替代61440输出SLA，不是Current KEEP或稳定容量证书。此前现场synthetic phase/controller fixture用于执行合同，非模型性能。
