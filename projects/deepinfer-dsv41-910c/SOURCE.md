# 源码、依赖与归档

## 1. 来源与版本

上游仓库为 [deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer)，适配基于提交 `553741557a57df44a46fb3ab72b6aafeaf945337`。此前原生适配检查点为 `55c7b50cebbee87d43f352bbda874111b990d101`；本次部署还包含经过验证的 Engram 共享读取和 MC2 容量调整。

部署源码以 `artifacts/source.tar.gz` 和 [source-SHA256.json](config/source-SHA256.json) 为准，版本标识为 `allopt-mc2-cap4096-v1-20261028`，共 1436 个文件。该快照直接从已测服务器版本逐文件核验后导出，包含所需源码与 vendor 文件，不含 `.git`、权重或编译缓存。

`code/` 为归档中适配相关文件的可读副本，不是第二套独立实现。`scripts/start-service.py`、`recorded/benchmark.py` 及两个 AISBench 配置保留实际运行版本。交付辅助脚本单独存放，不改写模型源码。

## 2. 归档内容

| 文件 | 内容 |
|---|---|
| `artifacts/source.tar.gz` | 完整已验证源码快照 |
| `artifacts/evidence.tar.gz` | 当前两轮原始结果、冻结输入与参考输出、启动命令、执行路径、源码清单和内存审计 |
| `artifacts/support.tar.gz` | AISBench 适配器、CPU 绑定记录、共享读取父版本结果与共享页审计 |
| `SHA256SUMS` | 本交付目录内除该清单自身外的文件校验值 |
| `config/archives.json` | 三个归档的字节数与 SHA256 |
| `config/environment.json` | 固定镜像中读取的实际包版本 |

归档 SHA256 同时用于传输后验证。源码逐文件 SHA256 对应服务器实际运行目录，不使用单一 Git 提交号替代未提交但已部署文件的版本信息。

## 3. 依赖边界

NPU worker 使用固定镜像中的 PyTorch、torch-npu、Triton、CANN 扩展及 custom OPP。该入口直接设置源码路径运行，无需执行上游全量 `pip install -e .` 或面向 NVIDIA 的构建流程。

CPU HTTP 接口复用固定镜像的 `vllm.tokenizers`，目标模型推理不调用 vLLM Python 引擎。低层算子 ABI 与该镜像绑定，使用其他镜像不能仅凭标签名称判断兼容性。

AISBench 源固定为 `7bcb6440da11bef020943b2af9ed7c1ac8362306`，来源为 [aisbench/benchmark](https://gitee.com/aisbench/benchmark)。部署端为不含 `.git` 的导出目录，版本引用沿用冻结记录。

## 4. 许可与发布内容

DeepInfer 使用 Apache-2.0，许可文本见 [LICENSE-DeepInfer](LICENSE-DeepInfer)。完整源码归档保留上游及 vendor 原有许可、来源与声明；适配文件中的算子参考说明继续保留。

仓库不包含模型权重、镜像、SSH 凭证或钥匙串数据。使用者需自行取得对应权重和固定运行环境。本次发布不修改上游 DeepInfer 仓库。
