# GLM-5.3-Flash W8A8 的昇腾 910C 适配

本分支提供 **DeepInfer 的实验性适配源码、运行环境记录和验证材料**。这是独立创建的分支，不继承本仓库 main 的内容或历史。

源码基于 [DeepInfer 上游](https://github.com/deepseek-ai/DeepInfer) 的 `553741557a57df44a46fb3ab72b6aafeaf945337`，叠加本次适配。上游 Apache-2.0 许可证保留在 [DeepInfer/LICENSE](DeepInfer/LICENSE)。本分支未包含模型权重、第三方子模块实体及六个上游图片资源。

**当前状态：已冻结源码有下述硬件验证；便携运行脚本仍是草稿，尚不能按本页一键完成环境复现。整体精度、性能及生产部署均未验收。**

## 目录导航

| 目录或文件 | 内容 |
|---|---|
| [DeepInfer/](DeepInfer/) | 包含适配修改的源码快照 |
| [portable/](portable/) | 参数化后的运行、结果对比脚本和示例消息；修改后尚未在 NPU 验证 |
| [validation/](validation/) | 算术、多轮记忆及状态恢复验证材料，含源码文件哈希清单 |
| [OPERATORS.md](OPERATORS.md) | 算子与模型接口适配清单：改动、源码位置、实际后端和验证限制 |
| [SUBMODULES.md](SUBMODULES.md) | 上游第三方子模块的固定版本 |

## 已验证的开发环境与镜像

| 项目 | 实测记录 |
|---|---|
| 主机架构 | aarch64 |
| NPU | 昇腾 910C；运行时设备标识 `Ascend910_9382` |
| 驱动版本 | `25.5.0` |
| 基础镜像标签 | `quay.io/ascend/vllm-ascend:v0.30.0rc2-a3` |
| PyTorch | `2.10.0+cpu` |
| torch-npu | `2.10.0.post7` |
| flash-linear-attention-npu-a3 | `26.9.1+deva4a7958` |
| CANN、自定义 OPP | 使用上述镜像中的组件；精确 CANN 版本仍待补充核验 |

已记录的**本地镜像 ID**为：

```text
sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b
```

该值不是已核验的镜像仓库 manifest digest，不能直接当作 `docker pull ...@sha256:...` 的地址。镜像标签可能变化；以上版本仅描述实测环境，尚未验证镜像重建流程，也不代表换镜像或依赖组合后仍然通过。

## 模型权重与源码身份

权重以只读方式使用，不随分支分发。复现前应核对自己的模型文件与以下记录是否一致：

| 对象 | SHA-256 |
|---|---|
| 量化权重索引 | `dfaf0918f4c088c98b011def38475d9b1cb395f35ed5e279abc0c57610e23be4` |
| 模型配置 | `33e63ec7fe607658be712bd6dd3c16c6549960d8e7f0483d34b939881b55f943` |
| 对话模板 | `34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891` |
| 实测模型源码清单 | `7b6f73ccab6987daec9d29fdc46804a7a67a25c684222d5310914fb36c8b1857` |

单文件哈希见 [validation/source-files.sha256.json](validation/source-files.sha256.json)。`portable/` 是在实测快照之后修改的版本，其通过语法检查不等于完成硬件验证。

## 已验证的范围及结果

验证使用四个协同 rank，执行完整 45 层模型，实验上下文上限为 128 token。

- **算术案例**：DeepInfer 与参考路径的生成 token 一致。
- **多轮记忆案例**：最终答案均为 `42`，但参考路径与 DeepInfer 的推理序列存在差异，原因尚未查明，不能据此宣布整模型精度通过。
- **生成中途状态恢复**：四个 rank 的 archive-v2 恢复能够复现 44 个生成 token；关闭缓存和启用 128 MiB 缓存两种配置均有验证材料。
- **缓存**：观测到的缓存命中次数为零，没有证明复用缓存带来提速。
- **停止条件**：生成停止标记 `154827` 与 tokenizer 的 EOS `154820` 是两个不同的 token；自然命中配置的停止标记不应写成命中 EOS。

上述结果仅覆盖具体的短请求和配置，不代表独立整模型精度、性能或生产可用性已验收。生产级 TP/EP、批处理、服务生命周期及长上下文质量仍未验证。

## 便携运行脚本：当前仍需补齐的复现步骤

[portable/worker.py](portable/worker.py) 保留了状态恢复和有界生成实验，并将模型、源码、自定义算子库等路径改为参数化配置。原现场控制器和绑定特定主机的启动器未公开；**修改后的便携脚本尚未在 NPU 执行**。

运行环境目前需要由操作者提供启动器，完成资源互斥、物理 die 与进程映射核验、请求 JSON 构造、各 rank 的 `go` 屏障文件，以及四个 HCCL rank 的启动。示例消息不是可直接执行的完整请求 JSON。

各进程涉及以下环境变量，具体路径和设备范围应按操作者自己的环境设置：

| 变量 | 用途 |
|---|---|
| `GLM_MODEL_DIR` | 模型权重目录 |
| `GLM_SOURCE_DIR` | DeepInfer 源码目录 |
| `GLM_ASCEND_LIBRARY` | 自定义昇腾算子动态库 |
| `GLM_EXPERIMENT_RUN_DIR` | 本次运行的独立产物目录 |
| `ASCEND_RT_VISIBLE_DEVICES` | 当前进程获授权使用的设备 |

[portable/compare_reference.py](portable/compare_reference.py) 接收两份兼容的四 rank 结果 JSON；参数化后的 rank 映射尚未执行验证。当前没有已验收的完整容器部署命令，不应把这份脚本草稿当成开箱即用部署包。

后续优先补齐参数化启动入口，并在真实 NPU 上执行公开文档中的命令，记录结果后再更新复现状态。

## 第三方依赖

固定版本见 [SUBMODULES.md](SUBMODULES.md)，上游地址见 [DeepInfer/.gitmodules](DeepInfer/.gitmodules)。本分支的根提交未保留子模块 gitlink，需按清单从各上游仓库显式获取对应提交，不能假定运行 `git submodule update` 就能恢复全部依赖。上述实测镜像之外的构建和依赖兼容性仍未验证。
