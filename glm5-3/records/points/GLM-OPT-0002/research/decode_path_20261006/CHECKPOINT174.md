# checkpoint174 — correct real observer API; one bounded graph diagnostic

本阶段没有新增代码级性能 KEEP。研究stack[H6,H5]保留，目标Decode FULL已开；GLM MTP原源码强制eager，greedy K1支持patch仍待设备正确性和matched收益。

Run269/270均原FAILED保留，各一次H6/H5/FULL/eager恢复已完成exact2334/8、health/idle/all16。Run269诊断witness目录缺失；Run270观察器读取不存在的extra context字段，0candidate请求/capture。实际16rank静态能力positive不是capture证据。[Run270归因](MTP_DIAGNOSTIC_CONTEXT_FAILURE_270.md)与[三失败GoalReview](GOAL_REVIEW_MTP_DIAGNOSTIC_175.md)记录Root失误、真实API反例和继续判断。

Production v2未变963f02e5…；observer v4按真实dummy/runtime kwargs区分调用，NONE不加contract要求，拒绝/地址变化无tensor D2H；actual ForwardContext/proxy/production-run AST、16生命周期/9fallback/4dummy/real FS+mmap本地及实际容器CPU通过。[独立最终review](V4_FINAL_REVIEW.md)无剩余具体observer blocker，建议只做最小capture correctness。

唯一Run271于08:47:04Z启动controller1232551/start310843208，speca7c3af23…/46pins。四个固定小PD请求、正常一次owned D重载、至多一次恢复；P249保护。实测按Run271 state/active_epoch/guards，不能继承旧D PID。目标真实capture→all16动态地址/metadata replay→exact标准PD；性能对照未开始。通过后再单独冻结same-worker matched A/B/A/B完整PD。H11未入stack/Current。

AISBench/GitHub每轮cache salt及P/D分端点计数已同步，正式80K/600/93%条件另验，不把声明93%或D external100%当本地cache实测。研究继续，formal API/SLA/稳定性/完整stack接受不作为停止理由。
