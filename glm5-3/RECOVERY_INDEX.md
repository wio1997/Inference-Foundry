# 新会话恢复导航

这是唯一恢复导航地图，只保存路径、权威引用和发现方法；稳定字段与核验命令按节查[ENVIRONMENT_RECOVERY](ENVIRONMENT_RECOVERY.yaml)；执行顺序和最小读取范围见[START_NEW_CHAT](docs/START_NEW_CHAT.md#恢复顺序)，本表只查本次需要的行。不要通读runtime或全部Run。路径失效先沿权威引用重新发现，未知明写，不重启、不重放旧controller。

服务器路径缩写：`R=/data/tiankuan/wio/Inference-Foundry`；`D=/data/tiankuan/wio/glm52-pd/deploy`；`G=R/glm5-3`。以下容器内源码路径需在对应服务器的`glm52-single`中读取，不能当成本机checkout。

| 要找什么 | 路径 / 权威入口 | 发现方法 |
|---|---|---|
| branch / HEAD / rule | Git；[启动身份核验](docs/START_NEW_CHAT.md#0-先核验branch--head--rule-version)；[AGENTS](AGENTS.md) | fetch研究分支`glm5-3-autonomous-20261001`；查HEAD、入口及其权威细则的最新rule commit、AGENTS内version；不固定旧SHA |
| 稳定环境清单 | [ENVIRONMENT_RECOVERY](ENVIRONMENT_RECOVERY.yaml) | 只读本次需要的values / source；缺项或漂移时用对应verify / discovery，不整套重复核验、不填动态值 |
| 仓库 / 部署 / 容器 | `R` / `D` / `glm52-single`；现场`D/HANDOFF.md` | `git -C R status --short --branch`、`docker inspect glm52-single`核验checkout及挂载 |
| 当前规则 / 模型分工 | [AGENTS](AGENTS.md) / [模型策略](docs/research/AGENT_MODEL_STRATEGY.md) | 规则权威只在AGENTS；旧next_action不约束路线 |
| 研究现场与产品入口 | [HANDOFF](HANDOFF.md) / [CURRENT_PRODUCT_MAP](CURRENT_PRODUCT_MAP.md) | HANDOFF选当前checkpoint/Run；产品地图定位源码；manifest→public_config / native_fresh_audit→现场文件及process |
| active optimization point | [points.jsonl](records/points.jsonl) / HANDOFF | 仅提取id/status/path/latest_run/current，与HANDOFF确定本次当前点；多个ACTIVE时不按最大ID猜，不展开runs数组 |
| PERF_KEEP / 当前stack | [CODE_PERFORMANCE_LEDGER](CODE_PERFORMANCE_LEDGER.md) / [CURRENT_PERFORMANCE_STACK](CURRENT_PERFORMANCE_STACK.md) | 只查对应候选/patch与证据引用；不复制指标或从最快Run推断Current |
| Run manifest / summary / raw | [RECORDING](RECORDING.md)；`G/records/points/<OPT>/runs/<RUN>/` | Git读manifest.json / summary.md；按manifest.artifact_index、source/ref、Job Result找服务器raw；若采用TaskCtl布局，以记录中的实际locator为准 |
| 大artifact / 冻结源码 | 同一服务器Run下`sources/`、`runtime_bundle/`、`restored/`及raw；`D/logs/`、`D/results/`、`D/plugins/` | manifest→artifact_index的path/hash；Git缺大文件时SSH读取该locator，不另建目录账本、不全盘扫描 |
| 模型权重 | 当前目标：两机`/data/tiankuan/wio/GLM-5.3-w8a8`，新服务名`glm-53`；用户2026-10-06已确认完成；fit/功能证据沿最新HANDOFF | 上传完成后核验config/index/tokenizer/全部分片，再核验启动argv；目录存在不等于完整或已加载，不hash权重payload；旧resident及历史Run不改标为5.3 |
| vLLM | 容器`/vllm-workspace/vllm`；[GLM-5.3源码入口](docs/research/GLM5_3_VLLM_ASCEND_PATHS.md) | 用实际服务Python与PYTHONPATH做顶层`importlib.util.find_spec('vllm').origin`，再查安装源Git HEAD/dirty；资料中的公开源码不是现场版本，旧模型可运行不证明5.3支持 |
| vLLM-Ascend | 容器`/vllm-workspace/vllm-ascend`；同上 | 同法`find_spec('vllm_ascend').origin`；按native argv/runner和plugin import选代码，勿导入模型或初始化NPU |
| benchmark / AISBench | [adapters/aisbench](adapters/aisbench/README.md)、[prefix_bench.py](adapters/aisbench/prefix_bench.py)；容器`/opt/aisbench-benchmark`、`/opt/aisbench-venv` | 当前Run spec/stage argv→实际脚本/环境；以该源码及安装metadata为准，不直接执行旧benchmark |
| profiler / trace | [profiling入口](docs/research/VLLM_ASCEND_PROFILING.md)；Run artifact_index | profile gate / profiler配置→命令与输出路径；service_config.environment→router/token memo trace；旧profile只按身份匹配复用，不自动采集 |
| controller / state / lock | [controller.py](runtime/controller.py)；`/data/tiankuan/wio/glm52-pd/controller-owner.json`、`.controller.lock`、`.formal-test.lock` | owner→对应Run/state.json与controller spec；manifest.controller_state辅助定位；lock存在不等于controller仍活跃 |
| Zcode CLI / Job→Result→bridge | 166宿主机已知`/usr/local/bin/zcode`；[ZCODE_PROTOCOL](docs/ZCODE_PROTOCOL.md)、[zcode_bridge.py](scripts/zcode_bridge.py) | `command -v zcode`；manifest.Zcode_result→Job目录（point/jobs或run/jobs）→job.json / result.json / bridge.json及原日志；后端型号/环境按已有任务配置核验 |
| 现场操作资料 | `D/HANDOFF.md`、`D/SSH_AND_OPERATIONS.md`、`D/PD_START_AND_TEST_GUIDE.md` | SSH按路径读所需章节；操作指南只是入口，启动/停止须先核验实际所有权 |

SSH：使用本机既有配置`ssh 910c-166` / `ssh 910c-167`；先`ssh -G 910c-166`核验配置。已有ControlMaster可用`ssh -S /tmp/glm52-166.sock 910c-166`（167对应`/tmp/glm52-167.sock`），socket失效不能当服务器失联。166现场已有的内网通道为`ssh -o BatchMode=yes root@172.16.10.167`；166/167内网地址分别为`172.16.10.166` / `172.16.10.167`。本机没有别名时，已知跳板入口为`ssh -p 2224 -l 'cuizihua@root@172.16.10.166' bastion.aiops.baai.ac.cn`，167将登录目标改为`cuizihua@root@172.16.10.167`；认证仍使用已有本机配置，详情按`D/SSH_AND_OPERATIONS.md`恢复；不把密码、token、私钥复制到Git或聊天。

每次现场只读重验：PID、boot_id、start_ticks、native epoch、controller身份/阶段、锁与资源owner、实际argv/PYTHONPATH/config、监听端口、NPU ownership、服务health及驻留observer。用state/process/既有health端点核验，不发生成请求，不因历史PID/端口启动或终止服务；连接失败记unknown。不要运行会隔离epoch或写状态的observer来代替只读检查。

权威边界：稳定环境字段→ENVIRONMENT_RECOVERY；Git事实→Git；规则→AGENTS；当前现场→HANDOFF + 服务器state/process；Run事实→manifest/raw；PERF_KEEP→CODE_PERFORMANCE_LEDGER；当前性能stack→CURRENT_PERFORMANCE_STACK。两份导航只更新路径/引用，不保存动态状态，也不替代Performance Research Reset或产品裁决。
