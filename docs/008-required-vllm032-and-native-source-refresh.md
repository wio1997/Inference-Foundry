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
