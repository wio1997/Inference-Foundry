# GLM-5.3-Flash W8A8 的昇腾 910C 适配

本分支提供 **DeepInfer 的实验性适配源码、运行环境记录和验证材料**。这是独立创建的分支，不继承本仓库 main 的内容或历史。

源码基于 [DeepInfer 上游](https://github.com/deepseek-ai/DeepInfer) 的 `553741557a57df44a46fb3ab72b6aafeaf945337`，叠加本次适配。上游 Apache-2.0 许可证保留在 [DeepInfer/LICENSE](DeepInfer/LICENSE)。本分支未包含模型权重、第三方子模块实体及六个上游图片资源。

**当前状态：已冻结源码有下述硬件验证；便携请求与 rank 入口已做限定硬件验证，仍不能按本页一键完成环境复现。整体精度、性能及生产部署均未验收。**

## 目录导航

| 目录或文件 | 内容 |
|---|---|
| [DeepInfer/](DeepInfer/) | 包含适配修改的源码快照 |
| [portable/](portable/) | 参数化请求构造、rank 入口、运行和结果对比脚本；验证范围见下文 |
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

单文件哈希见 [validation/source-files.sha256.json](validation/source-files.sha256.json)。`portable/` 在实测源码快照之后修改；请求构造器和 rank 入口已按下述范围进行硬件验证。

## 已验证的范围及结果

验证使用四个协同 rank，执行完整 45 层模型，实验上下文上限为 128 token。

- **算术案例**：DeepInfer 与参考路径的生成 token 一致。
- **多轮记忆案例**：最终答案均为 `42`，但参考路径与 DeepInfer 的推理序列存在差异，原因尚未查明，不能据此宣布整模型精度通过。
- **生成中途状态恢复**：四个 rank 的 archive-v2 恢复能够复现 44 个生成 token；关闭缓存和启用 128 MiB 缓存两种配置均有验证材料。
- **缓存**：观测到的缓存命中次数为零，没有证明复用缓存带来提速。
- **停止条件**：生成停止标记 `154827` 与 tokenizer 的 EOS `154820` 是两个不同的 token；自然命中配置的停止标记不应写成命中 EOS。

上述结果仅覆盖具体的短请求和配置，不代表独立整模型精度、性能或生产可用性已验收。生产级 TP/EP、批处理、服务生命周期及长上下文质量仍未验证。

## 便携运行入口与复现边界

[portable/launcher.py](portable/launcher.py) 的 `prepare` 读取交替对话 JSON、按模板分词并核对长度；`rank` 启动 [worker.py](portable/worker.py)。在现场分配器的锁、登记和映射门禁下，这两个入口已在四 rank 执行：每 rank 的 prompt argmax 和 44 个生成 token 均匹配 round 357，archive-v2 在 42 token 后恢复，停止 token 为 154827，进程组均已销毁。脱敏结果见 [restore-round360.json](validation/restore-round360.json)。

**仍不能一键独立部署。** 测试时由未公开的现场分配器持有共享资源锁、登记进程、核验宿主 PID 到物理 die 的映射并释放各 rank 的 `go` 屏障。公开入口不负责这些操作；操作者必须提供等效的资源分配、互斥、登记、核验及超时机制。以下为经测试入口的参数化示例，须先完成授权资源分配并设置镜像内实际路径：

```sh
export GLM_MODEL_DIR=/path/to/GLM-5.3-Flash-w8a8
export GLM_SOURCE_DIR=/path/to/DeepInfer
export GLM_ASCEND_LIBRARY=/path/to/vllm_ascend_C.so
export GLM_EXPERIMENT_RUN_DIR=/path/to/unique-run
python3 portable/launcher.py prepare --run-id example --context-tokens 128 \
  --max-new-tokens 64 --cache-mib 0 --expected-final 42 \
  --restore-after-first-token < /path/to/messages.json > "$GLM_EXPERIMENT_RUN_DIR/request.json"
# 由资源分配器给每个 rank 设置经核验的 ASCEND_RT_VISIBLE_DEVICES：
python3 portable/launcher.py rank --rank "$RANK" --die "$DIE" --port "$PORT"
```

分配器还需提供四个不同 rank/die、共同端口、HCCL 和镜像环境，并在核验全部映射后才建立每个 `rankN/go`。仓库示例消息是算术请求，应使用对应预期答案和生成长度。参考结果对比脚本的参数化 rank 映射尚未验证。精确 CANN 版本与镜像重建仍未验证。

## 第三方依赖

固定版本见 [SUBMODULES.md](SUBMODULES.md)，上游地址见 [DeepInfer/.gitmodules](DeepInfer/.gitmodules)。本分支的根提交未保留子模块 gitlink，需按清单从各上游仓库显式获取对应提交，不能假定运行 `git submodule update` 就能恢复全部依赖。上述实测镜像之外的构建和依赖兼容性仍未验证。
