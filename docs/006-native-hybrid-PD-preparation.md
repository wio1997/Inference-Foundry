# 固定原生压缩KV路径与双角色配置准备

从保留的已停止固定0.30镜像容器读取上游GPU model_runner及tokenizers，15个文件、242651 bytes逐项SHA核对；本分支保存选中的runner、V4.1 tokenizer及此前取回的DFlash speculator原字节。没有启动容器、包导入或NPU操作。

generic MooncakeHybridConnector在启用混合cache组时调用 `_transfer_kv_cache_all_groups`，压缩布局从descriptor和runtime views重建共享物理页，按组传输。源码另一个非混合路径的MTP末层加1逻辑，不应直接用来断言压缩KV的草稿支持情况。

上游 `sample_tokens` 在 `speculator.propose` 后调用connector的post_forward；DSpark沿用DFlash，在propose里为当前target hidden states调用 `precompute_and_store_context_kv`，非dummy路径按draft组提供context slots。generic connector的request_finished_all_groups返回全部计算组，并裁剪滑窗尾部。这些方法和调用顺序支持后续选择原生generic路径检查；不证明真实P侧草稿页面已写完、跨机消费顺序、实际注册布局、prefix一致性或模型质量。

G458仅准备配置：同一P3/D4，TP8/EP8/DP1、角色预算0.92/0.82、chunk8192/1024、C16、263296上下文、DSPARK5、P eager、D完整target+draft图、两端原生HBM前缀缓存及异步调度。KV dtype auto仍须实测BF16。使用generic hybrid PD，无外部KV池服务；这对应本任务原生前缀缓存能力对照，不能用PR12的external hit=0推导两种实现已经等价。PR12的0.32/910B3/C4/draft eager没有移作当前配置或结果。

配置未部署，CLI与运行时未验证。实际cache容量、所有target/draft权重、D完整图、P prompt-context初始化、跨机状态、语义/奇数长度及prefix命中均列为门槛。只有SG当前作业结束并核实同机8卡归属后才能启动；准备配置不会占用或打断当前SG作业。

- [原始源码与候选SHA范围](../evidence/native_PD_preparation_G458.json)
- [P配置](../code/native_PD_candidate_G458/candidate_p.json)
- [D配置](../code/native_PD_candidate_G458/candidate_d.json)
- [运行验收与资源条件](../code/native_PD_candidate_G458/readiness_requirements.json)
- [上游sample_tokens顺序](../code/fixed_native_PD_G456/vllm/v1/worker/gpu/model_runner.py)
- [原生DFlash context写入](../code/fixed_native_PD_G456/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py)
- [V4.1原生tokenizer](../code/fixed_native_PD_G456/vllm/tokenizers/deepseek_v41.py)
- [generic按组布局与传输](../code/fixed_native_PD_G456/vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py)
