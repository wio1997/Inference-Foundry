# 当前版本切换：vLLM Ascend v0.32.0rc1

用户将目标镜像更新为`quay.io/ascend/vllm-ascend:v0.32.0rc1`。当前要求见[机器可读状态](../metadata/current_version_requirement.json)，旧0.30的源码/组件/准备配置保留历史身份，不能直接作为0.32完整模型、PD、图或质量验收。

[版本化模型指南](https://github.com/vllm-project/vllm-ascend/blob/v0.32.0rc1/docs/source/tutorials/models/DeepSeek-V4.1-Flash.md)列出1017 W4A8、A2、Model Runner V2、DSpark与PD。受控参考审计指出选定113文件与官方版本一致；本次按其上游固定commit24487f4取得25个直接相关文件，全部逐项校验长度/SHA。它们是官方源码参考，不是本节点实际镜像已验收的证明，也不是要批量覆盖旧包的补丁。当前两机B1与参考B3、拓扑/负载不同，历史性能不能移作本任务结果。

注册表已确认所选tag的manifest list digest为`sha256:ff6d1ea57df08dd9382c51d202d0dd2d9acc35d53a65bcb0dfd906886e674912`；这是分发清单digest，实际arm64 config/image ID待目标服务器pull完成后独立核对。两机初次pull因旧Docker不支持显式--platform而在下载前失败；未改Docker配置，后续只移除该选项，宿主机原生选择架构，完成后仍须校验arm64/digest。09:29UTC两机均实际pulling，没有起新NPU模型。

旧0.30的CPU-only预检仅为历史诊断：G608缺少driver共享库；G609只读映射库后可导入部分配置，但general plugin查询实际设备架构失败，未完成CLI解析。没有映射NPU或在这些容器创建engine，CPU资源已释放；不以它们断言新版本不能使用。

- [请求tag的注册表元数据](../evidence/image_refresh_G612/requested_vllm032_registry_tag.json)
- [25份固定源码校验结果](../evidence/image_refresh_G612/pinned_upstream_source_fetch.json)
- [原生模型](../code/fixed_vllm032_reference/vllm_ascend/models/deepseek_v41/model.py)
- [原生cache dtype/布局选择](../code/fixed_vllm032_reference/vllm_ascend/models/deepseek_v41/cache_config.py)
- [P首次pull明确失败](../evidence/image_refresh_G612/2203/first_pull_platform_option_rejected.json)
- [D新pull实际存活](../evidence/image_refresh_G612/2204/immutable_native_arch_pull_started.json)
- [旧0.30无设备CLI门槛未完成](../evidence/superseded_030_CPU_preflight_G609/parse_result.json)


后续实际取证：两台已于09:35UTC完成同一manifest的arm64拉取，实际config/image ID是`sha256:d5492ecfdb3071ff2d4bae3f849245014c54b63035d9eefb35243869461ddae1`，与manifest list digest不同，分别保留。09:38UTC在无NPU设备CPU容器逐项读取两机实际镜像：各113文件全部SHA/bytes与固定上游匹配，vLLM HEAD5df63a7、Ascend HEAD24487f4，已跟踪/未忽略状态为空。元数据为vllm0.32.0+empty、Ascend0.32.0rc1、torch_npu2.10post7、Mooncake0.3.13。这是图像文件/元数据审计，不是依赖运行或模型验收。

SG当前诊断已终止并确认两角色队列/传输为空后，通过原owned controller于09:41–09:42UTC释放八卡。G630原生0.32 P/D随后实际启动，没有覆盖旧0.30源码或安装适配补丁；保留公共TP8/EP8/DP1/PCP1、chunk8192/1024、P0.88/D0.80、C16、上下文263296、DSPARK5及P eager/D完整target+draft图的配置意图。图与缓存/PD/质量需实际执行证明，启动凭证不能代替。

- [P实际113文件与metadata](../evidence/image_refresh_G628/2203/actual_source_and_metadata.json)
- [D实际113文件与metadata](../evidence/image_refresh_G628/2204/actual_source_and_metadata.json)
- [P原生0.32实际启动](../evidence/native_full_model_G630/2203/actual_start.json)
- [D原生0.32实际启动](../evidence/native_full_model_G630/2204/actual_start.json)
- [完整原生参数执行入口](../code/native_PD_candidate_G630/launch_pd.py)


G634实际09:48UTC：两端native CLI已接受完整参数并开始8个worker的HCCL初始化；/health仍connection refused，没有服务ready或权重/图/PD通过结论。原生日志确认D的`enable_fused_mc2=1`与`multistream_overlap_shared_expert=true`冲突，后者实际被置false；后续报告以实际生效参数为准，不能按CLI意图声称两个优化同时开启。Mamba causal_conv警告来自全局patch导入，尚未证明它是本模型实际执行路径的错误。

- [P实际原生CLI/worker初始化](../evidence/native_full_model_G630/2203/actual_startup_G634.json)
- [D实际配置冲突与worker初始化](../evidence/native_full_model_G630/2204/actual_startup_G634.json)
