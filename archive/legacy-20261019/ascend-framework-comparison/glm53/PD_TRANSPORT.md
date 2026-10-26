# 本项目P→D数据面：SDMA

project_id=`glm53`。用户指定本项目两个框架的所有P→D推理状态传输均使用**SDMA**，包括target/draft KV、indexer和scale等模型推理状态。原GLM-5.3与Flash各自验证，不相互借用通过记录。

这是部署约束；文档更新没有证明现场已启用SDMA。禁止在SDMA不支持或失败时悄悄切到RDMA并继续给出本项目基线。框架内部的backend/provider名称与实际数据通路分别记录，不凭“Ascend”、一个环境变量或机器同属超节点就判定SDMA。

TCP/HTTP/ZeroMQ等服务发现、bootstrap、调度控制与小型握手消息可保留；必须把控制面和推理状态payload分开计数。TP/EP等模型内集合通信也不与本条P→D要求混为一项。MemCache的DRAM读写通路单独验证，其SDMA成功不能替代P→D状态传输验证。

执行前读取当前镜像对应源码和SDK能力，确认真正支持SDMA的配置/API和fallback行为，冻结具体参数、库/源码SHA、rank/设备映射及超节点连通性，不发明flag。先做有界数据正确性和真实PD请求，记录源/目标设备、实际调用的provider/SDK通路、传输字节/状态种类、完成事件、输出正确性与错误；可用后端诊断、trace或计数器相互印证，保留证据能力限制。

只有实际加载路径与真实请求数据通路可证为SDMA、无RDMA payload fallback，才将对应运行标为PD_SDMA_VERIFIED。若当前镜像不支持，先有界修复/适配并保留回退，不把仅启动成功或无RDMA流量当充分证明。性能包含实际PD等待和传输；不与旧RDMA基线直接归因。
