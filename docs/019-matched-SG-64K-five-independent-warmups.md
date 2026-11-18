# 同机 SGLang 64K 的五轮独立预热

G181 在同一对模型主机、相同容器和配置上完成五轮独立65536→1024、C16、每轮16个请求，共80个实际usage核验通过的请求。首轮后暂停并保存 P 端371个编译缓存文件基线，再 resume 其余四轮，已完成首轮没有重发。最后两轮 D 没有新增编译文件，duration变化0.096542%、TTFT变化0.370868%、TPOT变化0.507614%，均低于5%；五轮后 P 缓存文件大小、mtime 与首轮后完全相同。

- [D 末两轮编译与耗时稳定门槛](../evidence/matched_SG_G181/64K/warmups/warmup_stability_65536.json)
- [P 独立缓存观察](../evidence/matched_SG_G181/64K/warmups/G272_64K_final_warm_gate_2201_parsed.json)
- [正式请求前门槛状态](../evidence/matched_SG_G181/64K/warmups/service_gate_sglang_g255_before_formal_g273.json)
- [全部 Run，含每轮五项原始指标、逐请求无损JSONL和SQLite](../records/runs.json)

这些预热不计分，也不用于框架收益结论。首轮 cold32/hot32 在门槛核验后单独提交，继续使用原7200秒整体截止时间；最终正式验收仍要求原始指标、实际usage、P/D编译和真实prefix计数证据。本页不将正在执行的正式阶段当成完成结果。
