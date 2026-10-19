# Indexer K 前处理与主 Prolog 的并行编排

当前完整服务已用融合 MlaPrologV3。精确源码 SFA9b94f67f…在调用前保留原 hidden_states，Prolog 不修改该对象并原样返回；Indexer 的 forward_k 只读原 hidden_states、独立 cos/sin 以及自身权重。K/weights 的 BF16 线性、FP32 LayerNorm（eps1e-6，再回BF16）、Triton RoPE，以及原本启用C8层的Hadamard／int8量化，均不依赖主Prolog的Q结果。原生已有K与Q通信重叠，本研究只检查更早的主Prolog阶段。

旧rank0 trace的182次精确BF16 WK形状 `[N,6144]×[160,6144]`，N=6/12/18/24，均紧跟同一物理流的MlaPrologV3，二者之间无其他kernel，间隔中位约1.05–1.12微秒。WK中位14.87–18.25微秒，Prolog中位38.91–42.03微秒；这些是旧诊断样本，不能相加当可恢复wall时间，也不能用层数乘出服务收益。

Indexer缓存精度按原模型逐层决定；同一N24片段观察到20次BF16缓存写、32次int8缓存写及对应FP16 scale写。不能把W8A8C8简化成所有Indexer层同精度。新片段必须覆盖原BF16及C8两种分支，保持Hadamard、原算子和原缓存写入次序。

安装镜像的shared_experts已使用torch.npu stream switch和事件／wait_stream，utils5f88fd83…中npu_stream_switch直接返回torch.npu.stream。外部多流案例仅用于启发；可用性以实际镜像源码为准。

冻结合同 `indexer-prolog-overlap-fragment-contract-v1.json`：单卡N6/N24，原AST提取的Prolog／forward_k／quant方法，synthetic同维权重，独立原生／候选ACL图pool，3代变化输入。候选K在独立流执行，主流保持原Prolog，join后才原生缓存写；逐位比较Q/query norm、主KV、K/scale/weights及完整Indexer缓存。通过后才测完整producer→join→缓存写片段的ABBA回放时间；不包括TP／CP／PD、完整top-k和attention，不能替代整模型质量或服务验收。

现场仅停快照后的自有D，P保留。33原holder按UID-tested deadline收束，创建新的10800s独占holder。v1在串行native首Prolog阶段报实际ND权重与要求NZ不符：独立进程缺少allow_internal_format开关，未进入两种编排比较。失败原件SHAe258362e…保留。v2只设置与此前工作片段相同的NZ开关，并在主算子之前断言三个权重真实format=29；不改变native方法、形状、精度或编排合同。

v2通过NZ和原生Prolog，随后在尚未初始化Triton设备属性时停止；v3补镜像原生init_device_properties_triton。v3又在原生Triton JIT的libkernel.so映射处停止，CPU-only同容器配置确认/tmp挂载为noexec。v4仅给片段容器的临时tmpfs增加exec，计算payload仍v3；没有更改主机权限或开发新kernel。所有失败在首个native K路径初始化时发生，尚无parallel比较或有效计时，不把这些初始化失败归为并行机制失败。

v4结果：4组eager与48次变化输入的独立ACL图回放均逐位相同，涵盖Q/query norm、主KV、Indexer K/scale/weights及完整缓存。完整片段图回放中位：N6 BF16 82.282→59.898µs，N6 C8 104.064→68.797µs，N24 BF16 101.693→68.885µs，N24 C8 123.461→86.223µs。全部12块/组保留，峰值418800640B。该片段不含完整top-k/attention、TP/CP/PD，不是服务收益。

下一问题是当前MTP eager的CPU提交与allocator生命周期代价：原target FULL graph已冻结执行编排，不能靠运行时marker改变图内流。先做一次独立eager诊断，保留原算子和缓存join，为辅助流结果显式record_stream(main)，及时释放结果时仍保护main消费者；只测eager完整片段，不重测图回放。若值得接入，初版仅在真实MTP eager路径启用，原target FULL/capture保持原生，避免混用两个编排的图缓存。

eager直接切流诊断：4组均逐位一致，但host中位候选／原生分别为1.2550／1.2095／1.2504／1.2119，完整event也同向回退。拒绝直接eager整模型接入。本次转向只捕获K生产者：3个owned输入copy在aux，K-only私有图回放；main保持原Prolog与原native缓存写。每次aux复用前等待当前main，join后才消费K，graph owner保留输出，避免释放后复用问题。该新片段的copies／外层Python提交／join均计入eager总时长，不重复旧完整图benchmark，新增copy的SDMA尚未证明。
