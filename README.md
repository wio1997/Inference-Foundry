# GLM-5.3-Flash W8A8 的昇腾 910C 适配

本分支提供 **DeepInfer 的实验性适配源码、运行环境记录和验证材料**。这是独立创建的分支，不继承本仓库 main 的内容或历史。

源码基于 [DeepInfer 上游](https://github.com/deepseek-ai/DeepInfer) 的 `553741557a57df44a46fb3ab72b6aafeaf945337`，叠加本次适配。上游 Apache-2.0 许可证保留在 [DeepInfer/LICENSE](DeepInfer/LICENSE)。本分支未包含模型权重、第三方子模块实体及六个上游图片资源。

**当前状态：已冻结源码有下述硬件验证；便携请求、rank 入口和参数化运行器已做限定硬件验证；复现仍依赖操作者提供兼容的资源登记和预算策略。整体精度、性能及生产部署均未验收。**

## 目录导航

| 目录或文件 | 内容 |
|---|---|
| [DeepInfer/](DeepInfer/) | 包含适配修改的源码快照 |
| [portable/](portable/) | 参数化请求构造、四 rank 运行器和结果对比脚本；验证范围见下文 |
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

[portable/runner.py](portable/runner.py) 是操作者持有的参数化四 rank 运行器：它使用既有共享锁和登记文件，核对分配、卡占用、镜像及源码/权重哈希，调用操作者的有界预算 helper，启动并登记进程，确认宿主 PID 到 card/chip 的映射后释放 `go` 屏障。它不停止其他工作负载。运行器配合 [launcher.py](portable/launcher.py) 与 [worker.py](portable/worker.py) 在四 rank 上执行一次短多轮恢复：每个 rank 的 prompt argmax 与 44 个生成 token 均匹配 round 360，archive-v2 在 42 token 后恢复，停止 token 为 154827，进程组均已销毁。脱敏结果见 [restore-round361.json](validation/restore-round361.json)。

可显式传入 `--kda-backend ascendc` 选择显式归一化的实验递推路径；默认仍为 `ascendc_installed_decode`，可通过同一选项明确回退。实际便携命令在四 rank 短多轮案例中验证了 `ascendc` 选项传播、首 token 后 archive-v2 恢复和自然停止，生成 45 个 token 与 round 373 一致；见 [结果](validation/portable-recurrent-round376.json)。这不是全模型质量或生产验收。

先将 [operator-config.example.json](portable/operator-config.example.json) 复制到任务目录并填写自己的**真实**路径、锁、已运行隔离容器 ID、镜像本地 ID、授权 die/card/chip 对应关系、赋权和登记 JSON、源码清单及 SHA-256。示例中的零值、路径和设备号只是占位符，不表示授权或已验证配置。`pinned_files` 必须包含可信源码清单、权重索引、模型配置与模板哈希。容器内外的源码、运行目录和 launcher 路径必须分别映射到同一内容；模型权重应只读挂载。

运行器要求操作者提供可信的预算 helper 和策略文件；helper 必须实现 `reserve(root, lock, goal, source_sha256, attempt_id, workers, timeout_seconds)`，并在传入的同一共享锁下原子核对授权和持久化预算。公共仓不包含现场策略或现场控制器。分配与登记 schema、设备映射、容器环境、NPU 可用性和 helper 的正确性由操作者按现场资源协议核验；没有兼容策略时不能运行。公开代码不提供服务、部署或自动抢占。

在已授权且配置完整的现场，参数化调用方式为：

```sh
python3 portable/runner.py --config /path/to/operator-config.json \
  --messages /path/to/task-root/messages.json --run-id example \
  --goal reproducible-release --context-tokens 128 --max-new-tokens 64 \
  --cache-mib 0 --expected-final 42 --restore-after-first-token \
  --experimental-fp32-dsa
```

消息文件必须位于任务目录内，且与 `--expected-final` 一致。结果写入唯一运行目录；若 worker 状态不确定，登记保留，必须先核实后接管。示例命令之外的部署、镜像重建、生产 TP/EP、服务生命周期、长上下文及独立整模型精度尚未验收。参考对比脚本的参数化 rank 映射仍未验证。

## 第三方依赖

固定版本见 [SUBMODULES.md](SUBMODULES.md)，上游地址见 [DeepInfer/.gitmodules](DeepInfer/.gitmodules)。本分支的根提交未保留子模块 gitlink，需按清单从各上游仓库显式获取对应提交，不能假定运行 `git submodule update` 就能恢复全部依赖。上述实测镜像之外的构建和依赖兼容性仍未验证。

## 四 rank 状态回滚探针

[portable/rollback_probe.py](portable/rollback_probe.py) 的四 rank 硬件结果见 [验证记录](validation/portable-rollback-round388.json)。它是实验探针，要求由操作者提供已持锁且已登记的四 rank 分配、隔离容器、超时、独立运行目录和 host PID 到 die/card/chip 的核验；探针不会停止或接管其他作业。各 rank 运行前将 `GLM_MODEL_DIR`、`GLM_SOURCE_DIR`、`GLM_ASCEND_LIBRARY`、`GLM_EXPERIMENT_RUN_DIR` 和只含该 die 的 `ASCEND_RT_VISIBLE_DEVICES` 设置为现场真实路径和设备，并以 `python3 rollback_probe.py <rank> <die> <共同端口> four` 启动。各 rank 创建 `rankN/ready` 后，操作者须验证宿主 PID 与授权设备映射，再分别创建对应的 `rankN/go`；启动、登记、错误处理和回收由操作者负责。权重索引、配置、运行时依赖和实际 probe 哈希必须与验证记录核对；该示例不替代现场的资源策略。
