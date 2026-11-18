# 完整图工作区显存不足与D端预算候选

G437的8份worker安装凭证覆盖此前遗漏的频率索引，完整target verify前向继续到原生FP4索引器。日志显示bs16的首次捕获循环迭代完成，继续bs12时，8个rank均在 `fp4_index_logits_decode` 的Triton工作区申请538MiB处报NPU OutOfMemory。完整图集及draft图尚未验收。D控制器退出并清空8卡，保留完整错误、清理状态和原始归档SHA；容器OOMKilled=false不能否定NPU分配器的明确显存不足记录。

实际KV pool的full token容量为3405696；target图前可用约5.60GB，首个图后日志约0.39GB。G450仅修改D端静态显存比例0.92→0.82，为完整图、warmup临时分配及native workspace增加空间，保留现有P437及其0.92预算。逐项argv比较只有该值变化；固定镜像、权重、量化、EP8、DSPARK5、完整target/draft图和max_bs16保持不变。这是待验证的运行预算调整，不是性能提升结论。

新D端已由任务控制器启动，实际池容量至少263296、全部要求的图捕获、PD状态、语义和性能仍须独立验收。静态比例减少后的有效池容量不能用旧容量代替，最终vLLM对照也须明确对应角色的预算与实际容量。

- [完整错误日志](../evidence/full_model_G437/2204/server.log)
- [D退出与8卡清空](../evidence/full_model_G437/2204/state.json)
- [原始SHA和候选范围](../evidence/full_model_G437/2204/graph_budget_assessment.json)
- [新D端实际启动凭证](../evidence/full_model_G450/D_launch_receipt.json)
- [D端候选配置](../code/full_model_candidates/sg_pd_g450/candidate_d.json)
- [候选边界与保留的P端](../code/full_model_candidates/sg_pd_g450/candidate_status.json)
