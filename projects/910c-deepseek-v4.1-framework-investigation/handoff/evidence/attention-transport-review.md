# Attention宽输入交付：owner协议与AG+MM流水

2026-11-09；冻结171/910C/CANN9.2.0、TP8/DP2/EP16与C8 eager first-token。路径来自实际Decoder → DSACP → SWA/C1/C2读者；历史价格仅在形成方案后匹配。全链路搜索仍未完成。

**用户最新验收要求：每个替换原语逐位一致。** 下方保留形成方案时的条件模型；实际诊断已将stock AIV AGMM直接替换归入已停止，不能按模型级容差放行。loop-004原INCONCLUSIVE是用户明确该要求前的时点结论，未回写成当时已知。

## D-A1-R：保持full-M的单owner生产

- **介绍**：先把BF16 H5120行收至owner，保持原全局M/N/K和权重生产KV，再发布窄结果；本地Q提前，与收集及其它rank计算重叠。完整producer还须覆盖C1第二投影和C2两FP32投影，保留pool/ring读序。只改所有权不自动保证不同rank、布局或kernel的逐位等价。
- **到了哪里**：待确认。读取安装包`torch_npu/distributed/distributed_c10d.py:158–270`，确认默认`gather`通过object broadcast+AllGather模拟，兼容模式且支持机型时才走group gather/send-recv；不能把API名称当真实root-only通信。基类vLLM gather还在root `torch.cat`，可改成预分配view但须实际支持相应接收。当前服务兼容模式未核实，未把缺省源码分支当现网分支。原服务配置归档`weight_nz_mode=1`，匹配的`_should_trans_nz`对BF16仅mode2转NZ，故配置路径支持ND；加载后的实际storage format仍未直接观测。
- **结果**：停止“直接调用缺省gather即可消除全宽广播”的实现假设；它保留全宽AG并多出协议及窄发布，不能领取旧窄AG和local-M节省。更大owner机制保留为具体缺口：真实root-only/P2P或兼容gather的拓扑、收集+必要物化+窄广播时延，与提前Q后的资源/就绪联合模型。现存通信校准没有Gather/Broadcast数据，不用AllGather字节比代替。单owner仍保留full-M计算，不能领取7/8计算删除信用。此分支不值得先写完整服务原型；可以先转向有现成融合接口的流水方案。

## D-A1-P：AllGather与SWA投影流水（stock直接替换已停止）

- **介绍**：从SP本地归一化行启动`npu_all_gather_base_mm`，让宽输入到达与BF16 Wkv投影流水；本地Q可先启动。AIV方案保留宽gather结果供其它必要读者，不声称消去全宽通信或7/8投影。原C2三个源层保留旧路线；C1第二BF16投影暂保留原实现。
- **到了哪里**：现成接口的定点对照完成，stock直接替换**已停止**。冻结源码manifest中的CVLinear及其它已归档文件没有调用此接口；实际安装TorchNPU2.10.0.post7的codegen YAML已登记带`comm_mode/gather_output`的schema。官方文档支持A3、TP8、BF16，FP32输入不在支持集合；推理AIV要求`gather_output=True`，不能套AI CPU训练模式的无宽输出路径。实测已证该容器接口可执行，但不满足用户逐位一致要求。框架`multistream_preprocess`原已用Q Vector与KV Cube重叠；保留原数值实现的其它流水仍须计入AIV争用，不能继承这个接口的性能。
- **结果**：独立从原profile绑定40层×16rank×4body的2560个AG/投影窗口，含cache-only及空侧。M3328/6528可保持stock乘法M；M3346/6534/6540被SP补至3352/6536/6544，stock融合先乘后切片会改变M，不能以full-M名义绕过数值核验。先给37层每TP组最大的`min(原AG,原SWA MM)`全部免费、其它工作及原EP到达区间固定，8请求条件TTFT信用最高1.87456%，最晚carrier累计27.3943ms；兑现75%最高1.40592%，50%最高0.93728%。该范围尚有量级，不能按低值停止；需要最小TP8 native对照决定现成接口是否继续。以上是乐观条件信用，不是预计/实测收益，不与旧owner30.27ms或C2预算相加。

## 数值与测量合同

用户已明确要求每个替换原语逐位一致。ND Wkv切M的302个位差和本次stock AGMM的位差因此足以停止这些具体替换；不再补测它们的原Q争用或服务收益。它们不否定自定义保持原归约/舍入点的更大流水或所有权重排。

实际诊断使用真实layer0 Wkv和固定合成输入，TP8 ranks0–7，Mlocal416及818；宽gather在8rank全部逐位相同，原GEMM对照分别有526/1,703,936和1005/3,348,480个位差，max abs均0.00048828125。所有rank输出finite。即使M3328没有padding也有位差，不能只归因padding。孤立7次交替event样本中，组内最大时长的中位差分别−96.06/+60.56µs，波动较大；无Q竞争、模型请求和服务收益验收。没有用这次微小正差外推请求收益。

前三个启动失败分别是记录write-only配置项、CONNECT_TIMEOUT低于本版本下限、默认设备socket16666已占用；均未执行AG/MM。已完整保留错误，不把它们判为接口不支持。最后仅用诊断进程级合法超时和`HCCL_NPU_SOCKET_PORT_RANGE=auto`完成；按[官方HCCL端口说明](https://www.hiascend.com/doc_center/source/zh/canncommercial/900/API/hcclug/hcclenvref_07_0021.html)隔离socket，不改服务、全局环境或挂载。健康检查前后均200。

原件：[完成诊断](native-ag-swa-evidence.json)、[汇总](AGMM-native-summary.json)、[原样脚本](probe_ag_swa_pipeline.py)。运行目录仅在用户RW映射下新建`fusion-ag-swa-pipeline-20261109-a1…a4`，没有向其它环境写入。

## 可复算依据

- [安装包只读提取](installed-communication-source.json)：gather实现与AGMM schema；只用stdlib读文件，无Torch import/NPU初始化。
- [官方参考pin](AGMM-official-reference.json)：链接与SHA，明确文档适用限制；[官方原文](https://github.com/Ascend/op-plugin/blob/master/docs/zh/custom_APIs/torch_npu/torch_npu-npu_all_gather_base_mm.md)。
- 逐调用窗口（2.6MiB原件保留本机）、[便携模型输入](attention-transport-inputs.json)、[条件价格](attention-transport-value.json)。原件hash和调用ID保留；不是全链路覆盖率证明。

```sh
python3 analyze_attention_transport_routes.py --recompute --portable-input attention-transport-inputs.json --output /tmp/attention-transport-recomputed.json
```

下一条独立交付是`SMLA output → restore_tp_heads → OA/OB → Attention HcPost → FFN HcPre`，从真实输出读者反推跨投影、通信、残差/归一化的范围。D-A1的精度/协议缺口保留，不能令其它前沿停住。
