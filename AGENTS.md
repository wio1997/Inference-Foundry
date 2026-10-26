# Current branch scope

- 当前172运行与适配记录为 `projects/910c-deepinfer-172`。先读其 README、STATUS.json 和 runtime 配置。
- `projects/910c-deepseek-v4.1-flash` 是已冻结 vLLM 基线；其原 STATUS 描述冻结时现场，当前容器 STOPPED 且保留，不覆盖原基线数据。
- 服务器由用户逐次指定；本项目明确为 172，不能把它当其他任务的默认目标。
- 本项目的模型、配置、运行日志、缓存和证据均放在 `<WIO_ROOT>`。
- 用户最新指定逻辑设备 0–15，共16个，固定 DP2×TP8、EP16；仅 Engram CPU offload，禁止主干 CPU prefetch。旧 DP1×TP8 OOM 属于历史，不是当前配置。
- 仅在用户授权范围内操作；不要停止其他容器/调优进程。设备空闲检查不能代替已知任务的协调。
- 凭证从当前 Mac 钥匙串直接送入认证；禁止打印、落盘或提交凭证。
- 不把容器运行、模型加载中或接口就绪等同于功能验证成功；每个状态保留实际证据。
- Git 仅存脚本、配置和小型证据；不要提交权重、离线镜像、大日志或编译缓存。
- `archive/legacy-20261019` 是完整历史快照；其旧 Current、KEEP、PID、任务及 AGENTS 只用于相应历史线。
- 公共网站用 API、HTTP 或 Web Search；浏览器访问需先取得用户许可。

- 简单机械工作（环境检查、服务启停、运行现成脚本、基准测试和日志提取）交给本机 ZCode 中配置的 DeepSeek。主 Agent 负责方案、复杂修改和独立验收，并从实际 rollout 核对模型身份；ZCode 不需要自行查询凭证或模型配置来证明身份。
- 当前native固定32768输入/1024输出/C8/89.99%命中两轮验收已完成：668.35token/s，16/16完整输出与冻结vLLM一致。TTFT仍比vLLM慢；提示查找收益依赖重复上下文。仅Engram CPU卸载，主干CPU prefetch关闭。
- 用户要求先看文档、减少无用测试；固定目标已通过，不为交付重跑或重启。新任务不得覆盖冻结证据。
