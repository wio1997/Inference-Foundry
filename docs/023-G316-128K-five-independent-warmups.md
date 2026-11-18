# G316 的 128K 五轮独立预热

在同一 P/D 主机对上的 G316 服务 epoch 完成五轮独立 131072 输入、1024 输出、C16、每轮16个请求，共80个请求。父级重新解析原始 JSONL、汇总 JSON 和 CSV 并核对原始 SHA；预热不计分。

最后两轮 D 编译缓存没有变化，duration 相对变化0.161175%、TTFT 0.183843%、TPOT 2.369668%。D 终态后独立读取 P，461个编译缓存文件的大小、mtime 与首轮预热后基线完全一致，未新增、删除或改变文件。预注册门槛为 duration/TTFT ≤5%且 D 直接编译变化为零，并要求独立 P 证明；TPOT同时作为 guardrail 报告。

原始 G181 128K 超时失败仍保留。G316 只将原生 PD 等待超时300→3600秒，其他计算参数保持；不能将该可用性修复解释为性能收益。五轮后首次正式 cold32/hot32 已提交，保留原始10800秒整体截止时间，已完成预热没有重发。正式结果尚待原始usage、指标、P/D编译和prefix计数验收。本页没有把预热指标作为框架性能结论。

- [最后两轮 D 直接编译与稳定性](../evidence/matched_SG_G316/128K/warmups/warmup_stability_131072.json)
- [P 独立编译缓存证明](../evidence/matched_SG_G316/128K/warmups/G382_P128_final_cache_2201_parsed.json)
- [D 实际终态与五轮审计](../evidence/matched_SG_G316/128K/warmups/G381_GLM_128K_terminal_2202_parsed.json)
- [首次正式提交句柄](../evidence/matched_SG_G316/128K/warmups/G383_GLM_first128_formal_start_2202_parsed.json)
- [每轮原始请求、SQLite及五项指标索引](../records/runs.json)
- [实际运行客户端源码](../code/SG_G316_128K_client/run_aisbench_cold_hot.py)
