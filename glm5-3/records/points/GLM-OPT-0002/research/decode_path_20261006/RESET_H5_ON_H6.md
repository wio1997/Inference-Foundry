# H5 on retained H6 — Performance Research Reset

2026-10-07，用户active research stack裁决持续有效。H6已实际重新启用并保留：Run259初次初始化因空witness目录同步遗漏失败，唯一冻结恢复完成H6 golden8、16 cache/native/source/idle见证；Run260随后同workers执行20条 H6 vs H6+H4。H4字节/输出正确、短TPOT中位数改善6.06/9.10%；完整B1 draft12/accepted11不同于A11/11，B2虽同11/11但D wall慢0.265656s，故frozen INCONCLUSIVE，H4关闭，H6 mode1保持。原始raw/失败/裁决不重写。

当前产品GLM-5.3 W8A8标准PD；Current=None，完整API/SLA/稳定性/full-stack尚未裁决。模型、资源和代码baseline固定H6，MC2 mode1始终启用。P249保护不变；D仅一次正常owned重载，重载时恢复H4 original finalize，安装H5同worker诊断选择器。重载后的A/B共用相同H6库/H5源码、workers、DP1/TP16/EP16/DCP16/MTP K1/eager/配置/观察方式。没有profile/扫描/新kernel。

Hypothesis（唯一active）：shared-overlap=false同一compute stream上的MoE事件没有跨流消费者；H5删除这些record及None-safe waits可继续减少层间host submission。实际 routed finalize然后shared量化/MLP；保留必要shared TP sum，必要EP/DCP通信与CANN内部event不变。五文件moe_event.patch，false overlap返回None、true overlap仍记录事件；CPU原证及Run253的176native检查REUSED；H6只缓存native capability、不改事件或数学。新真实模型gold、KV、MTP counters、全16 helper None/cache/mapping见证仍必需。

Distinguishing evidence：Run261 normal20条=4 correctness（两mode各short+自然EOS）+4warm+8 timed short+4 timed complete。2334/8与58/23两个既有fixture，IDs/content/finish/externalKV保持。现有Scheduler cumulative counter加原SSE arrivals；SSE分组不视为有序engine step或正确性门。A1/B1/A2/B2，MC2固定1，只切event0/1；warm/all16 witness在timing之外。正常一次D重载；异常最多一次H6+H5-off owned baseline恢复，保留失败。空witness目录由H5诊断helper自行mkdir，不能重现Run259同步遗漏。

Decision table（与Run260同口径冻结）：4完整 cumulative signature一致、8短signature一致，且所有correctness/16helper/cache/same-workers gates通过；每对complete D wall saving > max(|A2-A1|,|B2-B1|) D wall drift，两个complete TPOT与PD wall均改善，两个short中位数改善 → H5进入active research stack [H6,H5]，mode1保留。否则有效结果 → H5 INCONCLUSIVE/REJECT，event0，仅保留H6。功能/输出/ownership失败 → INVALID，安全撤回H5，H6不清空；外部owner不盲杀。P drift与D saving分账，累计H6+H5对stock未测，不相加独立百分比。研究KEEP不替代产品promotion。

后续继续离线核清已有FULL target wrapper的实际coverage与动态metadata/PD/MTP边界；FULL中eager_break decorator直接进入capture，不能推断MLA必然eager。Graph配置诊断不是代码收益。较小packet cache仅是源码候选，不能把12.342ms native暴露当可删或最大gap。actual make_backend纯CPU约0.974us，常用量化/Norm availability已缓存，boxed/TLS重复路线已关闭；不因整体SLA未完成而停研究。
