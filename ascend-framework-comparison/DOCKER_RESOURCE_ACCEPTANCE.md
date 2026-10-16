# Docker资源准备完成；开始模型功能验证前阅读

本次15台资源准备已完成并由主控独立复核，精确时点/边界见[最终验收JSON](inventory/20261017/final-docker-resource-acceptance.json)。本任务未创建/启动推理容器，未发新模型请求；资源就绪不等于模型适配、SDMA/PD、缓存、图/MTP或性能通过。

| 工作 | 节点 | 权重与镜像 |
|---|---|---|
| Flash vLLM1P1D | 14P、15D | 本地Flash97文件逐文件SHA完成；实际vLLM image090eeee6 ARM64已导入 |
| Flash SGLang1P1D | 46P、47D | 同一精确Flash快照；实际B103 imagec5cf2deb ARM64已导入 |
| 原GLM53 vLLM2P2D | P30/31、D43/45 | 本地原GLM53；实际vLLM image090eeee6 ARM64已导入 |
| 原GLM53 SGLang2P2D | P34/35、D36/41 | 本地原GLM53；实际B103 imagec5cf2deb ARM64已导入 |
| 备用 | 32/33/37 | Docker/统一目录/原GLM53本地入口与源码快照已具备，未装多余候选推理镜像/启动模型 |

38和42排除，不能额外借用备用节点的计算/缓存。所有15台Docker27.1.2、ARM64及Ascend默认runtime已核；Docker实际层目录仍为`/docker_data/data`，没有迁移。

统一路径：原权重`/data/wio/models/GLM-5.3-w8a8`，Flash权重`/data/wio/models/GLM-5.3-Flash-w8a8`（四台Flash），离线镜像`/data/wio/images/`，配置/源码/日志/实验`/data/wio/ascend-framework-comparison/{config,source,logs,runs}`。权重真实落本地NVMe，原模型入口为指向原有本地副本的符号链接，未复制另一份764GB。

SGLang本地便捷别名：`sglang-ascend:cann9.1.0-a3-B103-verified`；父独立核实它在六台均指向`sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263`。启动配置应固定该完整imageID；别名不是新的框架版本或稳定认证。vLLM固定`sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83`。原来源tag、完整归档SHA和权重manifest见[资源布局](DOCKER_ASSET_LAYOUT.md)。

## 本轮导入错误及修复

B10317,106,358,784bytes在Mac上传、源1及六台复制均SHA通过，六台首次导入却在我们脚本的`for tag in repo_tags`退出：该包按imageID导出，`RepoTags=null`。这是本次资源脚本没有处理无标签镜像的错误，发生在docker load前，不能归因于SGLang模型运行或镜像损坏。[原始错误](inventory/20261017/b103-null-tag-import-diagnosis.json)完整保留。

新修复把null规范为空列表，仍拒绝无效类型并保留已有标签冲突检查；每台重核现有17GB归档完整SHA、config imageID/ARM64/21层关系，之后docker load、核实际imageID并设置不覆盖未知标签的本地别名。六台均成功；没有重传镜像、重复制权重、改系统SDK/驱动/BIOS、启动模型或恢复定时。旧FAILED状态保存到每台本任务`import-repair-20261017/PREVIOUS_STATUS.json`，新状态位于任务根STATUS和修复子目录PROGRESS。历史一次性脚本见[修复记录](operations/assets-unify-20261016/import-repair-20261017/README.md)，禁止重放。

## 两个对话如何继续

两个GPT6.1 Sol High对话分别读[Flash提示](tracks/flash-adaptation/START_PROMPT.md)和[原GLM53提示](tracks/glm53-2p2d/START_PROMPT.md)。GPT主控做方案、具体代码/命令审查和独立验收；所有实际构建、部署、服务启动、NPU/模型请求、压测/profile必须由用户Mac本机ZCode DeepSeek执行。详见[执行规则](EXECUTION_CONTRACT.md)。

先检查本组设备独占、现有进程/锁、K8s预留、端口和共享服务，冻结本轮并行/量化/缓存/图/MTP配置。原模型2P2D每角色32die不能沿用旧EP64/TP64；Flash仍须适配混合状态/KDA与PD。通过已知答案/真实SDMA/PD和缓存正确性才跑同合同基线，再分析profile做受控优化。两轨道满载测试错峰。独立前缀、四轮新后缀、预热排除及失败全保留的合同不变。

Mac上冗余的17,106,358,784byte B103 tar已按用户此前要求，经本机ZCode DeepSeek完成最终SHA复核后删除；主控独立确认路径不存在。源1及六台离线tar、Docker已导入镜像和本地校验/源码记录保留。逻辑文件大小不代表已测量的APFS物理释放量。见[清理记录](inventory/20261017/mac-b103-cleanup.json)。后续从源1/data/wio/images取归档，不能再运行依赖已删除Mac tar的旧上传器。
