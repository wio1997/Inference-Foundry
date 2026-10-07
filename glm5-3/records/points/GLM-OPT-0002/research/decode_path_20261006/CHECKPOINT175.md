# checkpoint175 — true nested capture and single-graph patch

本阶段没有新增代码级性能 KEEP。H6/H5 验证研究栈保留，H11 未入 stack，target Decode FULL 已开。

Run271 第一次进入真实 MTP capture，外层 ACL 调用 runner 后置 Breakable 包装的 draft，内层 capture empty_cache 同步被禁止；0 candidate 请求/replay/性能。原 FAILED/raw/spec 保留，一次 H6/H5/FULL/eager 恢复09:01:11Z完成 exact2334/8 与两端 health/idle/all16。Root 亲读[原始归因](MTP_NESTED_CAPTURE_FAILURE_271.md)，继续修正真实代码缺陷，不做 flag 扫描。

最终最小组合：proposer19d31822…保留捕获地址/owned output 合同且要求精确 NoopOffloader；runner4c90054a…仅跳过已建立外层 GLM K1 图的后置 draft 内层包装。target/其他 drafter 不变；不同 wrapper 的 offloader replay fence 不等价，不支持 prefetch scope。实际 Noop Base类AST负例、实际 load order/Breakable cleanup 反例、1024runtime/19pointer/18static/24wrapper/16observer生命周期本地与真实容器 CPU 通过。[独立 final review](MTP_GRAPH_V4_REVIEW.md)无新增具体阻塞，设备证据仍待补。

唯一 Run272 于09:12:10Z启动 controller1949288/start310993881，specdf00a08d…/54pins。四固定小 PD 请求，正常一次 owned D 重载，最多一次双文件恢复及 gold warm；P249 保护。实测按 Run272 state/active_epoch/guards，不能继承 D271 PID。性能对照尚未开始；真实 all16 capture/replay/exact完整 PD 通过后再冻结 same-worker A/B/A/B。

AISBench 及 GitHub cache round 隔离、分端点 delta 已完成。正式80K/600/93%条件与实际命中分账；Current/API/SLA/稳定性/whole-stack接受仍另验，不撤回 H6/H5 或停止研究。
