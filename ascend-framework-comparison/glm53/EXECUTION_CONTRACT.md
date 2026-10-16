# 原 GLM-5.3：本机ZCode DeepSeek执行规则

本文件只约束project_id=`glm53`；本项目自己的GPT6.1 Sol High对话独立决策和验收，没有另一个模型的前置任务、统一排队主控或共同基线合同。

主控可做只读核查、源码/文档研究、本地代码草拟和审查。实际资源准备、离线构建、镜像导入、容器/服务创建启停、部署、SDK/框架overlay、NPU/模型请求、性能/profile采集均交用户Mac本机已有DeepSeek配置的ZCode执行。保留配置，不读取/输出provider或凭证。

入口`/Users/wio/work/sglang-glm53-33-41-deploy/zcode.sh`。每次显式指定本项目独立执行目录`--cwd`，提示包含project_id、唯一目标和本次已冻结SHA的精确launcher；不能省略cwd而把另一项目记忆或默认公共任务目录当当前工作。

1. 读取本项目真实状态/receipt、进程身份和锁；无重复任务才继续。先确认模型身份/路径与本项目资源白名单。
2. 审查具体脚本/参数/目标/边界/回退，冻结本次SHA。服务变更前准备可恢复配置；历史一次性launcher不可重放。
3. 本机ZCode只执行精确launcher一次；后台正常长任务保留，按有意义检查点或用户问进度时读取，不连续盯、不建定时。
4. 主控独立核对实际imageID、库/源码、模型/缓存/PD正确性以及原始失败；正确性通过后才出本项目性能基线。局部微测不冒充端到端收益。
5. 只更新本项目STATE/HANDOFF/run与结果；独立合同版本不依赖另一模型。实际物理争用如实记录、选择合适窗口，不操作对方服务，不发送跨对话消息。

运行根计划为`/data/wio/inference-foundry/glm53`，配置/脚本/数据/日志/locks/Meta/provider分别使用本项目名称；当前只完成文档拆分，未创建该新宿主机运行根或启动模型。Docker层仍在`/docker_data/data`，权重从本项目RESOURCES的本地目录只读挂入。
