# GLM-RUN-0003 — 并发2完整E2E

4/4请求有效，81932实际输入→61440输出，temperature0/seed20260930，90%公共前缀；prefix warmup1/1、full4/4的真实CLI退出均0。原生P/D重启后预热；引擎/计算未改。

TTFT P50 6.0588s（<4s FAIL）、P75 7.2498s、P90 9.3837s；TPOT P50 22.9ms（<18ms FAIL）、P90 24.6ms。四样本P99不作稳定尾部证书。有效输出245760token，完整full CLI阶段 3044.601s，80.71994TPS（含初始化/归约开销，非server瞬时TPS或每请求平均TPS）。

裁决INCONCLUSIVE；此1P1D/c2工作包未达SLO，不提升Current。旧c3有取消attempt且并发不同，不能算代码收益。下一问题：利用当前P闲置的完整模型，验证完整请求部署与可变到达路由；合法并行/部署空间仍未测，不宣布硬件极限。

精确artifact路径/大小/hash与逐请求有效性见manifest/reduction；raw在服务器。
