> 最新完整运行版本为[G42累计补丁](../patches/sglang-a3-1017-g42.patch)；候选[G43累计补丁](../patches/sglang-a3-1017-g43.patch)仅完成32K初测。二者相对同一固定上游，配套辅助模块与SDK覆盖须按各版本Docker命令使用。见[003条目](003-g42-g43/README.md)。

# 补丁与源码

## 固定基线

[完整源码包](../sources/sglang-d0e354bb7a0a75f2bd72f3724431198abcd9304b.tar.gz) 对应实际镜像内 commit `d0e354bb7a0a75f2bd72f3724431198abcd9304b`。镜像自带 Python 包中 5,191 个文件与该提交逐字相同，另有构建生成的 `_version.py`。本分支不依赖后续 upstream main 的变化。

- [SGLang 补丁](../patches/sglang-a3-1017.patch)：模型、ModelSlim、权重加载、Engram、mHC dispatch、缓存配置和 NPU 后端接入，另包含镜像生成的版本文件。
- [sgl_kernel_npu 补丁](../patches/sgl-kernel-npu.patch)：对镜像原始三个 SDK wrapper 的变更。
- [补丁基线/产物哈希](../patches/manifest.json)：可核查每个目标文件的来源及适配 SHA。
- [最终运行覆盖目录](../compat/runtime/g30)：包含完整修改后文件和框架外桥接模块，是实际部署代码的入口。

补丁与覆盖目录是同一变更的两种取用方式。Docker 复现脚本使用 bind mount 覆盖，不会改写镜像；如果选择集成源码补丁，仍需提供桥接 Python 模块、vendor 和三类扩展，不能只套用 diff 就认为整套适配完整。

## 改动范围

| 部分 | 原问题与处理 | 源码 |
|---|---|---|
| ModelSlim / checkpoint | 接入该模型的权重命名、W8A8 专家和原生加载路径；Q/KV 的 80 个目标投影实际为 FLOAT | [modelslim.py](../compat/runtime/g30/modelslim.py)、[weight_utils.py](../compat/runtime/g30/weight_utils.py)、[模型](../compat/runtime/g30/native/sglang/srt/models/deepseek_v4.py) |
| A3 cache/index | 使用 BF16 KV、INT8 index K、FP16 scale、1024B metadata、压缩比 1/2，复用框架原生请求、pair/history 和缓存所有权 | [pool](../compat/runtime/g30/a3_v41_pool_candidate.py)、[cache binding](../compat/runtime/g30/a3_v41_cache_binding.py)、[backend](../compat/runtime/g30/a3_backend_candidate.py) |
| 旧算子与当前 CANN 共存 | 保留旧 A3 QLI/Sparse MLA device/AICPU payload，私有 opapi 解析；用当前 CANN 重建严格 host 注册子集 | [filter](../compat/cache/vendor_filter.py)、[host 配方](../compat/host_subset_recipe_g5)、[assembly](../compat/host_subset_assembly_g4) |
| workspace 生命周期 | 解决每次调用保留约 4.20GiB workspace 的增长；按 thread/device/ACL stream 复用容量池，保留旧 owner 供已捕获图引用 | [G8 扩展](../compat/engram/attention/candidate_extension_g8) |
| DeepEP 初始化 | 在首次私有 metadata 调用前完成一次实际 DeepEP eager 通信；解决已观察到的启动顺序失败，底层初始化根因仍未确定 | [startup](../compat/runtime/g30/a3_deepep_startup.py) |
| DRAM Engram | 原 I8/E8M0 group32 表保持原数据，按节点共享 CPU 内存，通过 CANN 映射访问；门控使用 block32 逆旋转 | [engram](../compat/runtime/g30/engram.py)、[UVA](../compat/runtime/g30/engram_npu_uva.py) |
| mHC / RoPE / router | NPU dispatch 到镜像已有 Triton mHC；DSpark 无 headnorm Q 用现有 NPU RoPE 的实数展开；router 接入原 A3 norm2 接口 | [mHC](../compat/runtime/g30/native/sglang/kernels/ops/layernorm/mhc.py)、[RoPE](../compat/engram/attention/rope/g3_core5/candidate_extension)、[router](../compat/cache/router/candidate_extension_g2) |

这些改动包含存储格式和浮点执行路径变化，不能统称“数学完全不变”。DSpark RoPE 相对原 complex 公式有一个 BF16 末位差异，CPU 实数展开对照也出现同一差异；这解释了该局部观察，不证明完整模型精度。

算子依赖原始 inventory 和 host TU 输入哈希包含于 [donor source](../compat/reference/donor_candidate) 与 [输入清单](../compat/host_subset_recipe_g5/inputs_manifest.json)。20 个最终 payload 中 17 个保留原字节，另 3 个路径对应当前 CANN host 注册库；原实测哈希见 [vendor manifest](../environment/validated_vendor_manifest.json)。原 manifest 的“未上板”字段是生成时状态，后续验证以 [最终部署审计](../evidence/deployment/g30/audit_summary.json) 和 [最终验收](../evidence/original_acceptance/final_acceptance.json) 为准。

第三方代码保持其源文件许可证声明；[SGLang 许可证](../licenses/SGLang-Apache-2.0.txt) 与 [vLLM Ascend 许可证](../licenses/vLLM-Ascend-Apache-2.0.txt) 随附。厂商二进制由调用者按原镜像分发条款取得，本分支只提供清单、源码和重建方法。
