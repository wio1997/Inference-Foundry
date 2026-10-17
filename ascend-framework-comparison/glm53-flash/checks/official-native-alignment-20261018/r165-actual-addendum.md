# R165 实际验证补充

R165 原冻结验收仍 FAIL：缺16个 shutdown queue观察；strict数值仅9/16通过且3组重复ID不一致。以下是原始receipt独立审计事实，不替换该判定。

- 原生 KDA3文件和 pool_worker 镜像SHA实际验证通过；官方 indexer 仅 valid-mask 变更；两个 P API ready，16POST全部HTTP200。原生 API0/0、模型容器0、CPU Meta.main0/容器0、idle8；日志无 worker SIGKILL/宽限期后强制退出行。
- 20GB/worker 配置下原生16个MMC初始化及跨DP屏障均完成。1GB页分配ret6后2MB页fallback成功，不应再记作启动失败；这不证明未来C64缓存容量。
- 27个实际模型对象 put/known-key/full-object readback 全部通过，共181043200B逐字节SHA一致。实际只观测DP0八个writer，groups0/2/3/4；group1没有写入记录。不得按“每rank每group必须put”捏造额外对象。
- native累计计数器由0开始：每DP prefix queries4918、HBM-prefix hits1920；external queries2998，DP0 external hits0、DP1 external hits1920。是引擎聚合token计数，非单请求hit/tier证明，不叠加重叠层级。APIusage prompt_tokens_details为null，保留null。
- P实际MTP计数：DP0 drafts10/drafttokens50/accepted20；DP1 11/55/20。只资格化P发生过MTP，不证明D图或D MTP。
- 原生服务与typedcache数据链路取得进展，但首例参考selected LP误差0.4103584>.25，且DP0 cases0/1、DP1 case1同路径repeatIDs不同。数学答案42/56/0.8/6197不能替代冻结LP/IDs/repeat门。

下一步 R169：原生KDA/官方indexer，在参考TP16DP1/确定性通信/无MTP/无prefix/无async配置下的一个8POST有界控制。用于区分原生计算路径与优化配置，未改数值门限。B R168另在46/47进行一次有界整模storage候选验证，A不以B成功为前提；正式测试与重B互斥。
