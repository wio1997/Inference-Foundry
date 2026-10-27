# gen11 阶段记录

Q06 混合状态/relay 源码与只读 gate 设计阶段完成，待 root 验收。整体未完成，没有部署优化、NPU工作或新性能结论。

核查13个源码输入、27个方法范围。原SG容器只读取回7文件，4个与现有冻结快照SHA相等；新增FutureMap源码证实普通token relay按slot索引，未在该路径校验pool generation。generation在pool.clear重置，所以shadow身份需pool epoch。NPU兼容GDN kernel的原manifest和四份历史worker load receipt闭合；源码会对共享recurrent state原位读写，尚未证明所有stream/state安全。

执行7个实际保存源码AST：1814个边界算术组合、8个finished-handler stub组合、49个gate拒绝及5个离线rank集合。构造缺少new-stash时读旧token的反例与正常restash控制，说明不能仅依赖generation；未声称现场race。纯离线shadow gate遇到unknown回退，全满足也仅供canary审查，不可改变live schedule。

保留原finish/cache/free所有权及实际consumed KV长度。运行时cache类型/extra-buffer/lazy/跨流引用、GDN内容及新槽位立即复用仍unknown；质量、greedy等价与step→output-token缺口仍在。本轮未安装hook、未消费live队列或改服务。

只观察一次原后台链：PID/start/command/boot均相符；rank状态78 archived，pipeline submitted77/inflight4/pending123差异保留；后级仍等待。gen82 exact container stopped/slot resumed已复核。所有原作业、资源所有权保留，未因本轮结束终止或释放。

入口Q06_STATE_AND_RELAY.md与q06-state-contract.json。source-contracts-with-excerpts.json及cpu-contract-results.json给出原文范围、SHA与CPU边界。stage包使用显式allowlist并逐成员readback重哈希，不含transport/key/native原始profile。

下一步准备SHA固定的只读shadow hook并先CPU验证不改变方法返回/调用数/queue，再决定是否在原performance lock下做独立短正确性canary。不能在未知字段下启用Q06。完整204可用后优先对实际档案做closure，显式叠加gen8/9/10/11并重新打包和浏览器QA；当前gen9预览不等于最终204。私有R/main/index和最终root验收仍待办。
