# gen10 阶段记录

Q06 规划前尾结果 drain 的源码设计/CPU顺序模型阶段完成，待 root 验收；整体未完成、未采用优化、未发布。

已核验6份冻结源码、19个范围与原 manifest SHA，确认planner期间decode分配已推进KV和decode计数，因此不能以kv_committed_len代替output_ids，也不能在run_batch之前跳过已准备batch。最窄候选是在planner之前先消费确定最后一个在途结果，原finish/release仍保持唯一所有权；状态契约未知时回退。

CPU使用实际冻结event_loop_overlap和update_finish_state AST，其他worker/stop/stream/state均为显式stub。384个组合、22个拒绝gate、52个上界预算反例和4个finish优先级例子核验通过。合成cap32少1launch/1prepare只是模型计数，无真实模型收益/greedy结论。首次测试捕获already-finished遗漏并修正；manifest初始命名空间读取错误已修正，原失败日志保留。所有失败均非设备方案结论。

按RID/step重读四rank原observer文件，SHA一致，commit40/41/42输出计数31/32/32、增量1/1/0；原始证据未更改。记录没有规划前slot generation/queue/state，候选实际适用性仍unknown。GDN/cache和FutureMap复用未证，未写生产补丁或执行新NPU/service/SDK。

本次只观察一次外部作业：原PID/start/command/boot匹配；rank状态76 verified_archived，pipeline submitted75/inflight3/pending125，差异保留。assembler/sharder/autostage仍等；gen82 terminal/container stopped/slot resumed复核。没有终止任何后台作业或释放任何卡组。精确host/PID/start/command/log/终止条件见external-jobs-receipt.json。

产物入口Q06_DESIGN.md、q06-design.json、source-contracts-with-excerpts.json和cpu-model-results.json。gen9地图仍为当前预览；本代没有重建地图，设计补充需显式叠加到最终204并重包/QA。私有R/main/index归档及root验收仍待办。

下一步先沿target hybrid cache/GDN finalize、slot generation/FutureMap和长度契约做源码闭合，再准备fail-closed只读shadow gate字段。没有这些事实就不改live调度。native204完成后优先实际档案闭合、overlay和最终QA。不重跑正式基线或已有消融。
