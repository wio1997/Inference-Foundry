# D918完整图失败：补回已有complex64频率gather

固定daily A2镜像/910B1/aarch64/CANN9.1.0/torch_npu2.10post6不变。八卡96个FP4分数/slot用例和96次变输入图重放仍仅证明其覆盖的score/select/mapper函数，未覆盖完整模型q投影和频率表接入；不能把它升级为完整模型通过。

## 实际失败与资源状态

[原始八rank装载检查](../evidence/frequency_wrapper_G943/G932_P832_D918_readiness_2204_parsed.json)有八个target各1384参数、八个draft各73参数的native严格装载记录，missing均为空。随后D918在target full graph捕获中失败。[实际完整栈/进程/容器/owner证据](../evidence/frequency_wrapper_G943/G937_D918_FAILURE_REVIEW_JSON_parsed.json)定位npu_paged_indexer.py中的inputs.freqs_cis[pos]：aclnnIndex错误161002，DT_COMPLEX64不在支持列表。没有证据把它归为HBM不足，也没有因此减小并行、graph batch、上下文或量化标准。

controller于18:56:41UTC实际failed_released，旧容器停止且保留，NPU八卡清空/owner released，原始server/controller日志SHA保留。等待就绪的功能进程随后[精确结束](../evidence/frequency_wrapper_G943/G938_FUNCTIONAL_WAIT_STOPPED_JSON_parsed.json)，请求文件为零，没有提交模型生成、短profile、完整质量或正式AISBench。P实例、路由及DS workload guard保留；GLM设备没有操作。

## 根因与最小修复

原832的[频率适配](../code/full_model_D918/freqs_install.py)在native scoring.decode_scores等八处索引位置注入freqs_compat.gather。分页候选改写FullTopKIndexer._torch_decode时绕开了已改写的decode_scores，却重新写了complex64直接索引，丢失该既有NPU边界处理。这是候选接入错误，非组件分数数值不一致，也不证明分页机制失效。

[修复证明](../evidence/frequency_wrapper_G943/sg_D_paged_preparation_g939_manifest.json)将该表达式替换为原832同一份[gather helper](../code/full_model_D918/freqs_compat.py)：将complex64看作FP32的实虚部，按同一pos gather，contiguous后view回complex64，保留原位模式。helper原始SHA14bcb9ef3165ed382a5dbef8f6b479d3dd862d527a6bd8f649cb7eb12e0a422c未变。原native queries、RoPE、FP4编码、head weights、分数和选择数学保持。

score_paged、select_slots、_map_sorted_selected_slots三个函数AST与已通过924门的版本相同；这只限定已有96+96证据继续适用，未声称新wrapper或完整模型已验证。新的[源码来源索引](../metadata/D941_source_origin_index.json)复用D918权威文件，只另存实际变化delta及完整部署SHA，不重复维护数学实现。

## D941实际启动，后续验收

确认D918 controller实际退出/8卡释放后，[D941实际启动](../evidence/frequency_wrapper_G943/G940_D941_FREQUENCY_FIX_ACTUAL_START_JSON_parsed.json)，复用本任务210111946字节compiler cache副本，原件保留。相同native argv和环境仅更新任务路径，数学变更仅为补回已有频率gather；没有换镜像、降低batch或关闭图。

[实际进程检查](../evidence/frequency_wrapper_G943/G942_D941_ACTUAL_MODEL_REVIEW_JSON_parsed.json)与[当前状态](../metadata/current_pair_P832_D941.json)绑定新controller/容器/八卡owner。仍须实际八rank target/draft完整图捕获、真实模型候选路径和PD生成；新pair guard身份验证后才可提交功能请求。完整1319题同提示/T0/seed0/thinking=false/max2048/C16/严格####grader、SG四档五轮独立预热及首次cold/replay、最终匹配profile与端到端收益均未完成。

后续[037](037-D941-PD-SSE-real-kernels-and-full-quality.md)补充D941实际八rank图、PD/SSE、真实分页kernel与仍存的C1大位置表，完整1319题已启动但结果待验。
