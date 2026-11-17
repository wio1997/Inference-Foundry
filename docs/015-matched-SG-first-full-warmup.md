# 同机 SGLang 首遍完整32K预热

同一对GLM主机上的SGLang完成第一遍32768→1024的AISBench标准预热，16请求、C16，实际服务usage全部通过，保存标准TTFT、TPOT、输入/输出/总吞吐原始结果、逐请求详情和SQLite时间点。此次仍包含首次完整形状预热，不进入正式计分或框架收益比较。

- [本遍完整Run、五项指标与raw指针](../records/points/GLM-OPT-0001/runs/GLM-RUN-0066/manifest.json)
- [P端在第一遍后、其余预热前的编译缓存基线](../evidence/matched_SG_G181/P_cache_after_warm1.json)
- [首遍后自动暂停的状态](../evidence/matched_SG_G181/first_warm/run/state.json)

第一遍后暂停，采集实际P缓存基线，再用显式resume跳过已完成首遍，继续其余四遍独立输入；保留原始服务epoch、输入SHA和7200秒生命周期，不重发已完成首遍。warmup_only保持true，首轮cold32/hot32尚未发送。后续仍需最后两遍D编译和时延稳定、P编译证据以及全部四档正式矩阵与profile。
