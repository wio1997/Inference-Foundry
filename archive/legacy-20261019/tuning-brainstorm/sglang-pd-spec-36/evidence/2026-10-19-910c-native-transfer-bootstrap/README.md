# 910C：原模型双端封存、真实SDMA与完整启动反例

**原W8A8C8模型已在44/46完整校验并封存；44→46真实SDMA三组字节检查通过。完整32NPU模型启动随后暴露了MoE专家维度错误。只切换现有DeepEP分发的新启动仍在加载，尚未通过服务就绪与PD＋MTP工作负载验证。整体E0，accepted_baseline=null。**

所有测试在910C执行；Mac用于控制、静态审查与发布。没有缩短原输入、关闭PD/MTP或修改生产源码来获得通过。

## 真实两节点传输的有限结论

R031在44/46各用一个NPU0，镜像配置ID`sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263`、安装源码ce0、MemFabric1.3.0。两端原AscendTransferEngine及继承的注册/搬运接口使用新的任务store `tcp://172.16.12.46:39431`，与其他任务隔离。[冻结计划](specs/L039-paired-transfer-plan-v1.json)、[原始事件/隔离/清理](runs/L039-R031-paired-transfer/remote-result.json)。

| 检查 | 实际搬运 | 接收端核验 | 结果 |
| --- | --- | --- | --- |
| contiguous-g1 | 单段4194304字节 | 全4MiB独立CPU golden | 0差异，ret0 |
| scatter-g2 | 三段共458752字节 | 全4MiB，含3735552未写字节 | 0差异，未写区域不变，ret0 |
| scatter-reuse-g3 | 同缓冲换代，三段458752字节 | 新内容、新sentinel，全4MiB | 0差异，ret0 |

共实际发送5111808字节、接收端核对12582912字节；两端register/deregister通过，两个探针容器及临时store已回收。完整source文件匹配固定HEAD，没有应用候选描述符改动。

这是合成uint8缓冲的三组支持性结果。源模式周期为256字节，不能单独排除所有256对齐的源偏移错误；真实页映射、KV/辅助状态、batch_register返回处理、16rank并发、失败/取消和MTP生命周期仍需验证。没有测带宽或时延；导入、store、建链和跨主机时钟不用于性能推断。[主控范围审查](notes/L039-native-controller-review.json)。

## 原模型身份与传输收尾

R034确认两端复制进程均Exit0、VERIFIED。R035重新对照源端与两端全部193文件/186分片，764089835336字节/副本，所有文件名、大小和SHA一致；源文件stat未变，目的stat封存，元数据重新算SHA。[封存/等价/清理回执](runs/L039-R035-transfer-finalize/remote-result.json)。P44先重新读盘核验旧160文件635926286240字节，再补传128163549096字节；没有用普通W8A8或Flash替换。

仅删除了本任务精确ID的源/目的四个复制容器和37/44/46三份临时TLS目录。原模型、克隆、旧失败进度与归档保留，源37其他模型服务未触碰。源端落盘active_requests可能在finally递减前保存，不能用这个旧计数判定仍有复制；收尾以三个已知客户端的终止状态和全量合同为依据。[已结束的传输句柄](notes/L039-transfer-handles.json)。不要重放或继续轮询已经删除的容器。

R033另建长期CPU配置store：46私网39440，容器`if-mech36-l039-r033-store-46`，common init/create_config_store返回0，无Torch/NPU导入或设备节点，CPU2/内存1GiB。它继续供本任务模型使用，和已经删除的文件传输TLS材料不同。[实际store/容量](runs/L039-R033-shared-store/remote-result.json)。

## 完整模型启动与真实失败

R036在主机派发前因44旧预留R027已Failed被拒绝，零模型容器创建；R037查明UnexpectedAdmissionError/no healthy devices，同时当前物理NPU显示OK且空闲。不能将物理健康等同有效资源租约。R038重新检查占用后取得新非抢占44P预留，UID`5c1d29fb-bcf1-4ce0-bd9a-41bc83e2586a`；46原预留保留。两个Failed旧对象均保留，故障根因尚未证明。[资源复核](runs/L039-R037-bootstrap-gate/stdout.json)、[新预留](runs/L039-R038-reservations/stdout.json)。

R039随后真正启动两端各16rank：TP16/EP16，原模型只读，BF16及BF16 KV意图，PD/SDMA，NEXTN实际规范化EAGLE、3steps/topk1/draft4，context33792/chunk4096/maxrequests3。每端CPU96、内存上限1536GiB、私有IPC/shm256GiB；源代码启动前3948个Python文件与HEAD一致，两端manifest SHA相同，仅有已知打包toml差异。[完整冻结计划](specs/L039-full-bootstrap-plan-v2.json)、[真实启动](runs/L039-R039-full-bootstrap/remote-result.json)、[源码身份](source-evidence.json)。

R040/R041观察到并行组初始化、所有rank进入实际权重加载，每卡约49GB HBM，但HTTP未就绪。R042追查同一进程后发现Decode已退出，Prefill只剩失败后的父进程；进程“Running”和容器Exit0都不能当启动成功。

两端真正的首要错误相同：`aclnnGroupedMatmulWeightNz`报161002，**scale batch size[16] 与 groupList length[256]不匹配**。D在graph warmup报错，P亦在实际前向触发；通用的降内存/关graph提示不是这个参数错误的根因证据。[完整错误上下文及原始日志位置/哈希](runs/L039-R043-moe-failure/stdout.json)。D自身Exit0却语义失败；P父进程在确认同一致命错误后停止，最终137且OOMKilled=false。旧容器、原始日志和失败记录保留，不覆盖。

## 仅改变分发路径的新验证

固定ce0源码显示：NPU的`none`后端选择AscendTPDispatcher，它按全局num_experts产生256组计数；EP16权重/scale按每rank16专家分片。原grouped matmul把这个256组向量直接交给16专家scale。现有DeepEP分发显式接收num_local_experts，普通模式的Ascend pre-permute从num_recv_tokens_per_expert建立本地计数。这是与错误一致的源码解释，需要新实际执行证伪/确认。

镜像已有DeepEP `1.0.0+00db6cd6.cann.9.1.0.b243`，包含normal/low_latency/fused接口；静态存在不等于功能通过。R044只增加`--moe-a2a-backend deepep`，继续使用同一原模型、TP/EP16、PD、MTP3/1/4、graph、长度与资源限制。[单变量假设及冻结计划](specs/L039-full-bootstrap-plan-v3.json)、[新启动回执](runs/L039-R044-full-bootstrap/remote-result.json)。R045确认新两个容器仍Running并进入权重分配/加载，资源UID匹配且Running/Ready；接口尚未监听，**未证明16/256错误消失或DeepEP完整服务正确**。[最新观察](runs/L039-R045-full-model-state/stdout.json)、[当前精确句柄](notes/L039-model-handles.json)。

## 对课程点位的影响

M035新增“原始传输后端、单NPU对、有限字节模式通过”的组件证据；真实KVManager/页池/端到端路径仍待闭合。M026记录完整原模型下的MoE分发/量化接口失败；M028记录已有DeepEP模式路径的单变量启动验证，当前未决。其余33条机制数据不变。没有接受任何性能候选或晋升完整服务基线。

R030–R045共16条冻结命令各执行一次，输入SHA未变；R036内层失败及R039的后续语义失败均单独保留，不因外层CLI0改判。[执行器元数据审计](notes/L039-native-zcode-audit.json)。原始ZCode模型I/O、权重、镜像二进制和私钥不公开；仅含任务脚本、结构化结果、必要源码及自己服务的启动日志窗口。

下一步观察R044同一组句柄，确认实际target/draft、KV、graph和API就绪；无进展时用本任务worker计数/新错误定位，不能从等待超时重启。通过后再冻结输出/采样/到达过程、任务PD路由和取消/回收合同，建立A/A基线。TTFT、ITL、有效token/s、接受长度和可达硬件上界仍UNKNOWN，整体继续E0。
