# checkpoint165 — saved raw to a native host-cache candidate

本阶段没有新增代码级性能 KEEP。H4/H5 的原 correctness、限域 A/B/E2E 和正式 INCONCLUSIVE/PARKED 裁决保留，Current=None/active stack 空。继续研究；这不是归因或产品任务的完成声明。

Run249 的真实 critical path、必要计算/TP-DCP-EP-MTP 通信及 peer/event/queue 关系见 [checkpoint164](CHECKPOINT164.md) 和 [主归因](CRITICAL_PATH.md)。最大已定位区域仍为模型内部 eager host supply，主要在 MoE；不能把 ON 的747ms pre-enqueue gap拼成OFF的265ms/token预算。

一次必要最小 [Run254](../../runs/GLM-RUN-0254/summary.md) 已完成：stock P249/D253，一个2334-prompt/8-token请求，原ID/chunks和2334 external KV hits；D16 mains的3223 user CPU-clock samples、zero lost、2.241s window。没有新NPU profile、参数扫描、kernel变更或大规模Run。大raw留现场，controller completed/phase exit0/final_status与历史manifest PLANNED分开保存。

[CPU归因](CPU_INSTRUCTION_DECISION.md) 的 offline 分组已独立复算。2780个inclusive PythonKernelHolder samples包含handler/inner native工作，不能声称直调wrapper删除86.255%。248个GetOpApiFuncAddr samples全部来自实际两MC2 V4 predicates：dispatch134/combine114，actual binary BL/literal与固定源码均核对。这是具体重复控制逻辑，不能当必要EP通信。

[生产最小patch](mc2_capability.patch) 缓存两处函数局部bool，保留分配、动态路由、workspace/executor、V3/V2 fallback及kernel/通信参数。并发positive/negative native CPU检查通过，原查询约51us/次；380+380机械参考38.593ms不是模型预算/E2E收益。H6是唯一活动候选，正在构建隔离native产物，未安装、未重载D、未提交模型A/B。独立frontend两object可编译但内部queue/stream/workspace符号不导出，因此没有用不完整插件替换现场。

CPU build C缺完整ACL headers、D缺super_kernel.h，失败日志保留；E只补现场缺失头文件、复用隔离对象。真实Zcode handoff VALID，controller1669498/start305637950；最终状态以现场result/build_state为准，不按此记录重放。原库新进程CPU import/schema/four-V4-symbol reference已通过，NPU_initialized=False。candidate须编译→独立CPU import/ABI→模型correctness→common-binary matched A/B/A/B→相同自然EOS E2E，不越过门槛认定收益。

[实际eager/replay边界](EAGER_REPLAY_SOURCE.md) 已亲读并经Challenger核对：非FULL breakable整个MLA仍eager；FULL已有target replay机制，不能把关闭enforce_eager当代码patch。现场Mooncake逐层wait/save均pass；GLM drafter明确强制eager，target graph不等于全部target+MTP。动态buffer、混合请求正确性与收益未验证，不翻graphflag、不新增graph Run。

CPU原库审计前后protected P249/D253 owners/source/idle guards一致；H4/H5不叠加，native candidate未激活。精确Scheduler/Core/Executor/ModelRunner四分wall仍缺entry/exit标记，unknown保持unknown。继续已有source/raw下钻并完成H6真实验证，不因无formalKEEP或缺精确wall数结束研究。
