# 两个对话与本机ZCode执行规则

用户2026-10-17再次确认：准备自己开两个长期对话，都选择**GPT6.1 Sol、High**；实际启动服务、profile等使用其Mac本机的**ZCode，模型DeepSeek**。本次没有创建对话、替用户修改模型设置或恢复定时任务。

| 对话 | 主控范围 | 唯一使用的推理节点 | 交接入口 |
|---|---|---|---|
| A：Flash适配 | vLLM正确性参考、SGLang文本/混合状态/PD适配，再多模态 | vLLM14P/15D；SGLang46P/47D | [Flash接手提示](tracks/flash-adaptation/START_PROMPT.md) |
| B：原GLM53比较优化 | 新SDMA 2P2D vLLM基线、SGLang合格基线、profile和受控优化 | vLLM P30/31 D43/45；SGLang P34/35 D36/41 | [2P2D接手提示](tracks/glm53-2p2d/START_PROMPT.md) |

备用32/33/37需另行明确用途，不默认贡献算力/缓存；38/42排除。每条线只操作自身节点、容器、端口、Meta/provider、源码overlay、日志和锁。两条线先核实SDMA/HCCS与CPU/内存/客户端是否共享，满载性能/profile错峰，不能以不同节点等同于无链路干扰。

## 主控与执行器

各对话的GPT6.1 Sol High负责读官方文档/源码、方案判断、代码与具体命令审查、冻结实验变量、核查结果与交付。公开网站用HTTP/API/Web Search；不使用Browser/Chrome/Computer Use。只读检查及本地草拟/审查可以由主控完成。

实际资源准备、离线编译、镜像导入、容器创建/启动/停止、SDK或框架适配部署、NPU探针、模型请求、性能测试和profiling采集，**均通过用户本机已配置DeepSeek的ZCode执行**。主控不能自行换成直连shell启动这些任务。ZCode不是另一个任意改方案的主控，不得自行变更已冻结的模型/拓扑/参数/数据/统计口径。

本机入口：`/Users/wio/work/sglang-glm53-33-41-deploy/zcode.sh`。使用现有配置，不读取、输出或提交provider配置/密钥，不替用户切换DeepSeek。新对话先验证其能使用该现有入口；无法执行时如实报告，不绕过约定。

每次实际操作顺序：

1. 检查当前receipt、状态、实际进程身份和锁，确认没有同一任务仍运行；历史FAILED不等于当前进程已退出。
2. 主控审查具体脚本、目标白名单、路径、命令、界限和回退，固定本次SHA。读取当前基线，不重放历史一次性launcher。
3. 通过本机ZCode只执行本次已审查的精确launcher一次；确认真实receipt，后台长任务不重复启动。每轨道独立任务目录/锁，调用ZCode入口时避免相互改配置。
4. 主控独立检查真实文件、Docker imageID、服务健康、缓存和模型正确性。性能/profile原始证据包含版本、配置、开始结束及失败；模拟/微测/功能探针不冒充服务性能。
5. 当前阶段交接给人后再推进，不创建定时跟进。正常长传输不连续盯，用户问进度或到有意义检查点再读取。

## 并行工作与版本

两个对话分别从`codex/ascend-sglang-vllm-comparison`的最新提交接手，各自独立checkout/worktree和轨道工作分支，避免同时修改同一工作目录或公共状态。正式比较固定共同BENCHMARK_CONTRACT版本；公共资源/合同变更先记录协调，合回比较分支时保留两侧证据，不强推覆盖另一对话提交。

正式性能采用每档C条独立74400-token前缀、各自预热、四轮新后缀，4C条不同完整80000/600请求，预热单列并排除吞吐，保留自然缓存。测量实际缓存命中/层来源、P/D运行排队、图/MTP与SDMA路径；不能暗改为共享长前缀或仅统计成功幸存样本。历史4P2D/RDMA不能代替新2P2D/SDMA基线。

资源到齐是允许开始功能工作，不是稳定/适配/性能通过。两个对话均应先读[当前HANDOFF](HANDOFF.md)、[资源路径与门控](DOCKER_ASSET_LAYOUT.md)，不能把旧归档Current/PID当现状。
