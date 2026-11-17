# 固定镜像的 DSPARK 图路径与参数口径

权重定义 dspark_block_size=5、三个MTP层。层数与每轮草稿 token 数是不同配置。固定 vLLM 的 V2 DSparkSpeculator 以 anchor-as-first 方式为每轮五个草稿位置计算预测；num_speculative_tokens=5。固定 SGLang 的 gamma 取模型 block size，speculative_num_draft_tokens 包含 anchor，故对应值为6。两者计划匹配五个草稿预测，不沿用 GLM 的 MTP3 参数，也不修改权重配置。

固定 vLLM 旧 DSparkProposer 将 use_cuda_graph 设为False，不满足本任务D侧草稿图要求。同一镜像内已有 AscendDSparkSpeculator/V2 路径，复用上游DSpark→DFlash图管理；patch_dflash_speculator 将其 manager 替换为原生 DFlashAclGraphManager，捕获和重放使用Ascend更新参数。enforce_eager=false 是必要配置，仍须现场捕获与真实模型重放证据。

SGLang 的 _decode_graph_allowed 由 decode backend 与是否PD prefill决定；附近的 not _is_npu 条件属于另一项并行布局限制，不是关闭DSPARK图。它会将 capture_decode_cuda_graph 传给原生draft worker。该源码链说明应优先保留现成图路径，不能据此声称完整模型图已跑通。

所有依据均是两套固定镜像中原样提取、SHA固定的源码；没有升级镜像、驱动或CANN。当前结论限于静态契约。后续必须验证完整权重、PD目标/草稿上下文、量化、目标/草稿图实际捕获及重放，之后才能提交正式性能。

[参数与源码SHA权威记录](../evidence/native_graph_contract.json)
