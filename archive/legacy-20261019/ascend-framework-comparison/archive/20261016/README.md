# 现有 SGLang 工作归档（2026-10-16）

这是一份源码、配置、权重身份、优化证据和恢复位置的 Git 归档；不是带全部权重/镜像/SDK 二进制的可搬运 Docker 镜像。没有在本次归档中重启模型、读取全量权重或发起模型请求。

| 路径 | 内容与边界 |
|---|---|
| current/OPERATIONS.md | 20:16 前实际日更运行栈与恢复依据，覆盖旧上午交接说明中的过时网关信息 |
| current/records/node-N | 六节点实际关键源码、启动/缓存配置、容器与包版本、96 scheduler 身份和实际 SDK 库 SHA；历史快照而非实时探针 |
| support-deployment/ | 旧交接中补充的建容器模板、共同环境、provider/Meta 源码与配置；其中旧 router/旧 P Unified 不能覆盖 current/ 的新快照 |
| weights/ | 193 文件源清单、历史 payload SHA、34/35/36 的已有结构验证；今天未重哈希 764 GB，也未证明所有其他新节点有同一份权重 |
| sdk-python-runtime/ | 私有 MF LOCAL loader 与 SDK Python 模块的实际文本源码，含来源包/成员 SHA；不含 native 二进制 |
| b103/ | 发行候选来源、ARM64 digest/依赖/完整层校验 receipt、Flash 接线调查；已下载到 Mac，未导入/部署 |
| optimizations/ | 状态不同的已上线修复、已接受限定实验、未上线 CPU 候选；看 OPTIMIZATION_LEDGER.md |
| results/ | 历史强制 L3、自然暖 C16、chunk A/B 等摘要/审计；raw 大包只存外部位置和 SHA |
| historical-vllm/ | 用户提供的旧 2026-10-08 六机 RDMA + 单共享前缀资料，保留原文与 SHA，**不进入新比较性能表** |
| ASSETS.json | 不入 Git 的权重、镜像、SDK/客户端离线包、原始测试压缩包的位置和校验摘要 |

current/records 的实际源码是 python/sglang/srt 下的相对路径。恢复前先核对每节点镜像、路径、源 SHA 和 SDK；使用保留服务器基础镜像+当前角色的实际源码/配置+私有 SDK，不能只 docker run 基础镜像就宣称环境相同。不要把本归档中的脚本视为当前现场的一键重放入口。

incident/STATUS.md 等原始事故快照保留当时观察（如246条已完成）；最终结论以本归档 LEDGER 和公共 HANDOFF 的252成功/4超时为准，不能用较早快照覆盖最终结果。

公开仓库不包含私钥、密码、ZCode provider 配置、模型 tensor、原始业务请求文本。完整独立请求 raw 及编译 SDK 二进制仍在本机/现场原任务目录，ASSETS 记录可查。源码随原项目许可证，SGLang/MF/MC/vLLM 的来源提交与补丁引用保留。

运行 `python3 ../../tools/verify_archive.py` 可只在本地校验归档文件，脚本不连接现场、不加载模型。
