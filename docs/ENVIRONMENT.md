# 环境与版本

以下版本对应本分支收录的实测，不把归档时最新版本替换进既有结果。

| 项目 | 实际版本 / 配置 |
|---|---|
| 硬件 | 单机 8 张物理 Ascend 910C / 16 个逻辑 64GiB die，aarch64 |
| 驱动 | 25.5.0 |
| 主机 RAM | 约 2.16TB |
| 模型盘 | 8 块非旋转 NVMe 构成的 55.9TiB XFS LVM |
| 模型 | Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8 |
| SGLang 镜像 | `quay.io/ascend/sglang:cann9.2.0-a3-20261111` |
| 多架构 index | `sha256:744c3e35988945134e3706dbcc7f3885f6639cbe3f2b69c51687c41429f8d935` |
| ARM64 manifest | `sha256:8b52b8f5819cf48262f4a2212aa9218c5d4c063dbe71a78e641bb7d0b6d57ebf` |
| 实际 ImageID | `sha256:08367522c4f9803fb876808c2cd4afe459e7e2b0adaab5a0daf36061edf8e517` |
| 实际 SGLang commit | `d0e354bb7a0a75f2bd72f3724431198abcd9304b` |
| 实际源码来源 | `lilDss/sglang`，`refactor/dsv4-cann-transformer-kernels` |
| SGLang 包版本 | `0.0.0.dev22278+gd0e354bb7.d20261111` |
| CANN | 9.2.0，inner `V100R001C12SPC001B237` |
| Python | 3.12.15 |
| torch / torch_npu | `2.13.0+cpu` / `2.13.0` |
| sgl_kernel_npu | `2026.9.0` |
| triton / triton_ascend | `3.6.0` / `3.6.0+dev20261105160244` |
| DeepEP | `1.0.0+d5107fb6.cann.9.2.0.b237`；源码固定提交 `d5107fb699fe1f0ce3fd900365264211000a1297` |
| MemFabric Hybrid | `1.3.0` |
| 算子依赖镜像 | `quay.io/ascend/vllm-ascend:v0.30.0rc2-a3` |
| 算子依赖 ImageID | `sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b` |
| AISBench 镜像 | `ghcr.io/aisbench/aisbench_benchmark:v3.1-20261013-master-openeuler22.03-py310-aarch64` |
| AISBench ImageID | `sha256:05358d58d23f2679c71e48d93797d5d2a28d72eedc7fb4900479c89955465d76` |
| AISBench 实际包版本 | `ais_bench_benchmark==3.1.20261013` |

镜像标签中的 OCI revision 不是实际 SGLang 源码 commit。这里以镜像内 Git HEAD 和源文件内容为准。完整上游源码压缩包与镜像包内 5,191 个源码文件逐个相同，另有镜像构建生成的 `_version.py`。见 [源码/补丁清单](../patches/manifest.json) 与 [镜像包源码哈希](../sources/image-python-manifest.json)。

[完整依赖 JSON](../environment/installed_distributions.json) / [依赖版本列表](../environment/requirements-observed.txt) 是对该 ImageID 的包元数据观察。CANN、厂商算子、架构和二进制 ABI 不能由一次 `pip install -r` 替代；使用固定镜像和配方。

模型校验包含 142 个索引引用文件的尺寸、143 个 safetensors 的 header/tensor 映射以及 6 个控制文件 SHA256。未重算约 781GB 权重的全部 payload 哈希。未复制进仓的权重从 Eco-Tech 对应模型交付获取，并须保留 `optional/quarot.safetensors`。见 [权重检查证据](../evidence/support/audit_result.json)。

Engram 使用两个 CPU 共享 memfd，Rss 合计 **198005896KiB，约 188.83GiB**。CANN 注册采用 MAPPED/PINNED 路径；Linux smaps 的 Locked 为 0，不能解释成 Linux `mlock`。见 [DRAM 映射](../evidence/deployment/g30/engram_smaps.json)。
