# GLM-OPT-0001 — 恢复现役合同并区分服务容量损失

- 类型：characterization；状态：ACTIVE。GPT研究、代码与裁决；现场Zcode直接--prompt执行、监控、归约；controller串行共享资源。
- 现场：166 P:9081、167 D:9900、166 proxy:8000；两端glm52-single。当前源码身份与安装沿用2026-10-01 preflight，恢复Job RECOVER-20261001T0900Z通过真实CLI/Result/bridge；P/D/proxy健康、request_num=0，恢复窗口无匹配controller/benchmark/锁持有。源基线c1361b81；任务代码与raw日志均在166。
- 合同：GLM-5.2 W8A8、TP16/DCP16/EP16；P eager、MTP1、batch tokens16384；D FULL_DECODE_ONLY、MTP3、batch tokens128；max_model_len144384。专用控制层不得牺牲完整接口、有效输出、KV/代际和MTP提交；不开发新计算kernel。
- 历史复用：保留preflight与旧c3六条有效请求的已有身份、长度与SLA结果，不重跑恢复模拟或c3长输出。c3 TTFT P50/P75/P90失败，TPOT P50=24.4ms失败；吞吐原始计数包含9 attempts，不能将result.csv 115.58 TPS当作独立六条有效吞吐。D MTP稳态接受长度约3.4–3.56、接受率约80–85%，不足以归因低接受率；P仅执行prefill输出1，P/D MTP不对称本身不是缺陷。
- 机制先验：PK-080/120/121/122/123/127只复用资源竞争、提交/跨rank和观察器/可比性方法；旧DSpark和7月GLM收益不继承。
- 当前问题：同80k输入下并发如何改变真实decode TPOT、供给/KV等待与有效吞吐；长输出/混合输出和动态到达如何移动容量边界。先用少量短真实E2E获得区分信息，短诊断不替代61440输出的正式SLA。
- 修改范围：runtime/controller.py、phase_runner.py、stream_probe.py及诊断observer；task-local AISBench adapter保存真实内层退出，失败阻断解析。复用现场未测candidate，不声称旧candidate已验收。现役引擎、proxy和计算实现暂未更改。
- 执行：GLM-RUN-0001计划真实81920原始prompt→256，concurrency1，显式resident_history_unknown。controller/source/command/返回usage/finish/DONE与原始SSE留服务器。Run执行状态以该目录state/phase和结果为准，未返回前不填性能。
- 当前最佳：无正式可重复KEEP；数值界与稳定容量unknown。新Run可形成诊断结论，不能因短输出样本宣布达标或到顶。
- 已知边界：task-global flock保护遵循锁的controller和legacy suite；直接legacy lifecycle入口仍可绕过，不声称全入口防护。未知中断stage不得自动重放；未测试现役长任务强制中断。不要求全部复现缺口补齐后才研究。

Run0003正式c2四请求245760有效token、61440/request均完成，TTFT(first SSE) P50 6.0588s及TPOT P50 22.9ms未达SLO；详见Run3 summary/reduction_v2，Current不提升。早期Run1/2 first_content遗漏reasoning，已离线修订真实首输出/TPOT，原raw不改；不能再引用14.625ms作为Run1真实first-token TPOT。
