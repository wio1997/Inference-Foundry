# DeepEP完整启动后续：两种原生分发模式均失败

**R044尚未成为可用PD＋MTP服务。** 上一检查点按R045快照记录“加载中”；R046/R047随后证实Prefill normal和Decode low_latency都在CANN tiling阶段失败。原模型、TP/EP16、PD、MTP3/1/4、图配置均未缩减，整体继续E0，accepted_baseline=null。

| 角色 / 实际路径 | 原始错误 | 状态 |
| --- | --- | --- |
| P44 / normal | aclnnCamMoeDispatchNormal，EZ1008，Failed to execute tiling function | 子进程失败；已确认致命错误后停止残留父进程，最终137、OOMKilled=false |
| D46 / low_latency | aclnnMoeLowLatencyDispatchV2，EZ1008，Failed to execute tiling function；graph warmup包装报错 | 容器自身Exit0，但启动语义失败、HTTP未就绪 |

[同一模型句柄的后续观察](runs/L039-R046-full-model-state/stdout.json)、[完整错误上下文及自己服务日志的保存位置/SHA](runs/L039-R047-moe-failure/stdout.json)。原始完整Docker日志保存在两端任务目录`L039-R047-diagnostics-{44,46}/model-docker.log`，两个失败容器仍保留；只停止了已证明失败的P父进程，没有触碰其他任务服务、模型字节或长期CPU store。

这次失败在dispatch阶段早于grouped matmul。没有再次出现旧16/256错误，**不能据此证明旧错误已经修复**；新路径尚未执行到所需验证点。P的normal亦失败，因此仅把D切成normal不能作为已经验证的解决办法。当前证据只定位到tiling阶段，尚不能区分输入参数约束、算子注册、CANN/DeepEP构建匹配或其他原因。[主控审查](notes/L039-deepep-tiling-review.json)。

下一步读取这两个任务目录中的CANN plog与实际算子参数、DeepEP/CANN/OPP构建及注册信息，冻结具体可证伪的支持性检查，再决定新配置或修复。不能从通用graph内存提示直接调参，也不重放已经失败的完整加载。PD分离和推测解码主线保留。

R046/R047两个命令各执行一次，冻结输入不变；它们的采集/归档命令执行成功与模型启动失败是不同层次。[仅会话元数据的执行器审计](notes/L039-tiling-zcode-audit.json)。无服务输出、吞吐或硬件上界结论。

前序[原生SDMA三组检查、原模型封存及首次16/256反例](../2026-10-19-910c-native-transfer-bootstrap/README.md)保持原样。M026/M028追加本次失败，其他34项机制数据不变。[当前精确句柄及终止状态](notes/L039-model-handles.json)。新环境需先取得资源、核实版本与模型并登记新Run；脚本供审查，不应重放旧容器命令。
