# 当前G316的优化候选与编译采集修正

AllGather优化候选已从旧G181重建为当前G316的P配置，保留原生PD等待3600秒；D候选与当前G316配置逐字节相同。旧G309候选标记为已被替代，不能恢复其300秒超时。复用原生Config、EP8/288专家构造器和草稿resolver的CPU检查，只证明准备接口，不证明NPU、完整模型、PD或端到端收益。完成当前256K首次正式冷/热及原始证据归档后才进入优化阶段，激活仍须独立资源、数值、图和实际profile/A-B门槛。

D编译采集器已另建G488目录并实际部署，原G199源码和所有历史快照保留不变。新实现绑定G316实际容器ID、环境、可写挂载及六组缓存路径；正式检查核对新增、删除和元数据变化。历史warm4+5的错误目录快照不改写，正式恢复必须另提供严格早于完整warm4的当前D基线、五轮后的独立末态及P证明，并继续核对原始manifest、数据、客户端、服务身份、截止时间与首次提交ledger。当前只有部署凭证，没有启动修正采集器或接受正式结果的结论。

G496两个末态脚本仅为准备内容，需在实际第五轮结束后按依赖顺序执行和父级复核；不能把文件存在当成已完成采集。128K历史D编译验收仍未验证，见此前显式更正。

- [G464准备状态、原生source pins和未验证范围](../code/native_allgather_candidate_G464/preflight_contract.json)
- [当前P优化候选](../code/native_allgather_candidate_G464/candidate_p.json)
- [当前D相同配置](../code/native_allgather_candidate_G464/candidate_d_unchanged.json)
- [单独部署的修正采集器](../code/SG_current_D_collector_G488/run_aisbench_cold_hot.py)
- [实际部署且未启动凭证](../evidence/current_D_collector_G488/deployment_receipt.json)
- [当前D五轮末态采集准备](../code/SG_current_D_collector_G488/collect_current_D_256K_allwarm_g496.sh)
- [独立P末态采集准备](../code/SG_current_D_collector_G488/snapshot_G316_P_256K_allwarm_g496.sh)
- [128K验收更正](025-G316-D-compile-root-correction.md)
