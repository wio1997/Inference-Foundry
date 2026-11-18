# 双端 no-SP 仍复现语义错误

**DSPARK 不是现有故障出现的必要条件。** 两端实际关闭 DSPARK 后，32 个诊断请求均完成 HTTP/SSE，算术错误、代码末尾遗漏及 GSM8K 重复生成仍出现。这个结果没有完成完整质量验收，也没有证明 PD 是唯一根因。

## 实际配置和启动证据

固定 SGLang daily A2 镜像仍为 7da963869ee54984dfeea2445befdf181710ef846448466db2361247759904ec；910B1/aarch64，CANN9.1、torch2.10+cpu/torch_npu2.10post6，原 W4A8 权重、BF16 KV、FP4 index 和适配内核保持。TP8/EP8/PP1，context263296、max-running16、overlap/前缀缓存开启；P chunk8192/eager/内存0.88，D chunk1024/full graph/内存0.80。该配置只用于定位，最终仍须恢复 DSPARK/PD 对齐。

原 P832、guard1026 及其容器内路由在两次空闲采样、旧客户端实际结束和 D1020 已释放后受控退休；旧容器、缓存和日志保留。[退休记录](../evidence/paired_NoSP_G1056/G1040_P832_guard_router_retirement_2203_parsed.json)确认控制器、guard、路由退出，八卡 owner 释放，路由端口及锁可用，随后才启动双端1038。

[16 rank 实际门槛](../evidence/paired_NoSP_G1056/G1045_parent_pair1038_allrank_gate.json)记录两端 target 各八份1384参数严格装载、missing=[]、无draft账本；D八卡query1 full graph bs=[8,16]完成。P有17个request slots，D有49个；C2均为request-scoped ring8，每rank三层，实际分配与原生计价分别为1794048/4939776字节，wire item均为32768字节。容量不同没有伪装成相同；实际API两端speculative_algorithm=None、prefix开启，P bootstrap的dsv41_spec_layout=None。原生完整PD校验没有修改。

## 路由观察超时的实际解释

新原生路由的第一次45秒观察窗口到期，但进程未退出。[同一PID复核](../evidence/paired_NoSP_G1056/G1049_pair1038_router_same_handle_2203_parsed.json)确认其约51秒后完成注册，两worker健康、load0，没有重启或重复启动。固定镜像的初始detect_connection_mode仍硬编码探测/health，常驻健康检查才使用/ready；这解释了此次注册耗时，相关探测不算正式预热。

[提交前门槛](../evidence/paired_NoSP_G1056/G1050_pair1038_router_guard_layout_2203_parsed.json)再次确认新guard实际持有canonical flock、绑定新容器、两次P/D队列为0、bootstrap布局None。之后才在原CPU客户端运行有界C16诊断，无并行NPU负载、profiler或请求重试。

## 原始请求和语义结果

[父级核查](../evidence/paired_NoSP_G1056/G1056_parent_NoSP_semantic_assessment_public.json)逐项重建32条SSE内容、usage、DONE与SHA，并确认32份请求JSON与D974/DSPARK诊断逐字节相同；HTTP task AST相同，仅诊断输出目录和说明改变。107个原始文件逐项验SHA。worker在DONE行后、最终空行前停止采集的边界与原轮次一致，原字节未补写。

- 16条短请求均传输成功；严格格式及答案同时匹配7/16：算术3/8、代码复制4/8。D974/DSPARK旧诊断为8/16，这不是统计质量或性能对比。
- 例如“13乘5再加7”输出65，预期72；KD32190DFD复制为KD32190DF。部分结果只是格式不符，但这些具体算术及字符遗漏是真实内容错误。
- 首16条GSM8K仍仅作max256的生成前缀诊断：3条stop、13条length；两条stop答案匹配，另一条stop内容错误且没有有效####答案。不得据此计算全1319题、max2048预算的最终质量分数。
- 客户端实际结束、仅剩sleep，P/D ready及队列0已核对；两端模型、路由和guard继续保留。没有提交新的完整GSM8K或AISBench测试。

No-SP改变了执行宽度、装载epoch及批次打包；这里不声称只有一个性能变量改变。故障在实际无DSPARK的配置中仍存在，足以否定“只有DSPARK才会导致错误”，但不能排除额外SP问题，也不能将故障直接归于PD。

## 缩小下一步范围

[固定镜像源码与实际平台标志](../evidence/paired_NoSP_G1056/G1056_fixed_source_platform_descriptor.json)显示is_cuda=False、is_cuda_alike=False、is_npu=True。按已核对源码推断，CUDA folded verify epilogue的构造条件不成立；这只是源码/平台推断，没有把它写成旧SP实例的实际调用计数。后续若回查SP，应先查实际NPU greedy acceptance、target query6及状态提交，而非直接修改CUDA fused accept tail。

下一步使用单节点D直接HTTP、no-SP、prefix与full graph的有界诊断，去掉P/router/PD分离执行，区分共享target计算和分离路径。ring8分配及原生计价保留；角色、prefill执行与容量变化仍需披露。扩展helper的CPU契约已通过、未初始化NPU；D1060仅准备，尚未部署或执行。若单机仍错误，不能继续只围绕PD或SP修改；若单机正常，也仍需区分P侧执行、传输、缓存和调度，不能直接宣布PD根因。

[当前索引](../metadata/current_pair_P1038_D1038.json)列出剩余目标：修复语义、恢复DSPARK/PD完整配置、全量1319题质量、SGLang四档AISBench首轮冷/热结果、最终匹配P8/D8 profile与差异地图及端到端收益。本轮没有改变已完成的vLLM分支或基线。
