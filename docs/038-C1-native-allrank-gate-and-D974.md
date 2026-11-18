# C1原生发布/消费：八卡门槛通过，D974进入完整模型验证

本轮完成了[八卡C1组件验收](../evidence/C1_allrank_D974_G982/G977_C1_PARENT_ACCEPTANCE_JSON_parsed.json)：144个用例、144次变更输入后的NPU图重放通过。D941在失败质量运行退出、队列排空、原始证据冻结后受控退休；组件控制器实际退出并释放全部NPU/CPU占用后，才[启动D974](../evidence/C1_allrank_D974_G982/G977_D974_ACTUAL_START_JSON_parsed.json)。完整模型质量和端到端收益尚未验证，[当前状态](../metadata/current_pair_P832_D974.json)保留全部未完成项。

## 干预及不变的选择语义

[D941的短设备窗口](037-D941-PD-SSE-real-kernels-and-full-quality.md)已证明C2进入分页FP4路径，但C1仍直接调用共享decode_scores/write_decode，绕过FullTopK覆盖。该路径仍产生大容量int64 slot表，不能把此前的局部机制通过表述为全部消除。

[C1适配](../code/C1_native_gate_G969/npu_c1_common.py)复用已验证的原生分页FP4打分及selected-K物理映射。保留同宽FP32 scores、原生DenseBlocks候选块发布、原生块内top-k和排序；用DecodeScores的私有子类保存请求映射，slots为None，由write_decode映射最终选出的512个位置。它不会进入Hopper的PagedDecodeScores分派，也没有增加候选mask剪枝。原生非paged NPU分支忽略candidate_mask的行为保持不变。

decode_scores/write_decode修改函数代码并保留函数对象身份，因此原生导入别名仍有效。安装顺序明确位于既有freqs_install之后，继续使用同一复数频率gather；D941的C2实现、权重、量化、TP8/EP8、DSPARK5+root、图配置、上下文、缓存与分块参数不变。[D974集成差异](../code/D974_C1_delta/integration.py)及[完整部署源码SHA](../code/D974_C1_delta/deployment_source_sha256.json)可核对。部署包其余文件继承D941，公开差异目录本身不是可直接启动的完整包。

## 实际正确性范围

[固定镜像CPU门槛](../evidence/C1_allrank_D974_G982/G970_C1_gate_CPU_2204_parsed.json)执行了实际八站点频率安装器、C1安装器、frozen msgspec子类/slots=None及别名检查，确认未初始化NPU。基准decode_scores、write_decode、select_decode及DenseBlocks publish/consume的函数和consumer globals在修改之前独立冻结，避免对照误调用候选代码。

[NPU用例](../code/C1_native_gate_G969/probe.py)使用实际平台探针，没有覆盖SM能力。每rank18个用例、18次图重放，覆盖ratio1/2、空批次/零容量、零可见长度、奇数和ragged长度、underfilled候选、实际2048块×8的位置规则、额外2/3块排除路径，以及实际容量263435/131717。batch覆盖1/6/16/24/80/96；最大容量图覆盖batch1/24/96。变更查询、权重、位置、复数频率、请求行、请求映射及packed K后，将每次候选重放与新鲜原生eager对照，严格核对FP32分数位模式、块ID、发布/消费的物理slot和padding。

144用例中32个为空批次/零容量返回None检查，112个为非空路径。查询使用预量化FP4加2的整数次幂频率乘数的fixture，packed K为原生68B布局、随机payload、scale字节127；它不覆盖完整投影/RoPE，也不代表所有量化数据。完整模型装载、图捕获和实际设备路径仍须独立验收，不能把144组件用例当模型质量分数。

## D974与质量问题的后续

[D974实际现场](../evidence/C1_allrank_D974_G982/G978_D974_observe_2204_parsed.json)确认控制器、固定镜像容器及权重加载，尚未就绪。新建[guard980](../evidence/C1_allrank_D974_G982/G980_GUARD_ACTUAL_TRANSFER_JSON_parsed.json)绑定实际P832/D974容器，确认旧guard实际退出、新guard持有canonical flock；没有改写旧身份记录。[限定功能进程981](../evidence/C1_allrank_D974_G982/G981_FUNCTIONAL_ACTUAL_START_JSON_parsed.json)有3600秒硬边界，等待该实例后检查两项固定答案、1024-output SSE和两步D设备诊断。其结果目前未验收。

20:20UTC的[实际D装载账本](../evidence/C1_allrank_D974_G982/G983_functional981_D974_2204_parsed.json)确认8个target各1384参数严格加载通过、missing为空，draft仍在加载，尚未就绪。[P侧实际进程与FD](../evidence/C1_allrank_D974_G982/G983_functional981_D974_2203_parsed.json)确认functional981仍等待同一D实例、guard980真实持锁；不是仅依据state字段报告进程存活。

失败的质量958保留原始22行，其中14个600秒超时；八个实际响应仅三个通过原严格grader。[冻结响应复核](../evidence/C1_allrank_D974_G982/G979_failed958_response_audit.json)列出原始SHA、expected/extracted及完成原因。部分响应存在格式异常或缺少####，所以消除超时不能替代质量判断；不修改提示、grader、超时预算来制造通过。

下一步先核对D974八rank target/draft严格加载、完整图、PD/SSE及真实C1设备kernel，再执行相同1319题质量门槛。SG四档AISBench、最终匹配P8/D8 profile、逻辑工作量归一化差异地图及E2E收益仍未完成。vLLM原生0.32的既有完整质量和四档原始结果继续留在独立分支；本组件结果不构成框架性能比较。

## 后续实际验证：完整图、PD/SSE、八卡设备路径通过

[D974父级就绪核对](../evidence/D974_readiness_G990/G984_parent_D974_readiness.json)确认八个target各1384参数、八个draft各73参数严格加载通过，missing为空；target宽6、bs4/8/12/16和draft宽5、bs8/16的full graph均有八rank实际begin/end。20:22:42UTC的/ready为200。启动前freeze_gc连接拒绝栈原样保留，随后真实API已就绪，不据早期警告判定服务终止，也不因警告重启。

[限定功能原始记录](../evidence/D974_readiness_G990/functional/result.json)于20:25:46UTC完成，进程随后实际退出。两个固定答案分别为4和Paris；SSE有精确18输入、1024输出及DONE，[父级逐响应SHA和计数核对](../evidence/D974_readiness_G990/G987_parent_D974_functional_mechanism.json)已完成。路由健康恢复期间保留原服务，未重启或重试模型请求。

[短D设备窗口](../evidence/D974_readiness_G990/G986_functional981_profile_2204_parsed.json)的八rank各有10889条compute，device ID逐rank匹配；每rank实际出现16次分页FP4打分、16次selected-K物理映射、零次旧非分页FP4打分。所观察窗口的Index/FloorDiv不再含263435/131717容量形状；其他标量/候选块相关FloorDiv仍存在。Python构图日志同时覆盖C1层20/24/28/32/36及query rows24/48/72/96，不能单独作为图重放证据；这里以真实device kernel补足限定机制验证。

SDK CPU解析已结束，1014个文件、465034381字节按SHA冻结，大型原始trace保留在受控存储。这个18-input/B1/two-step诊断不是最终16个32K请求的匹配profile，不能直接用其duration或10889/11019 compute数量推算端到端收益。P端完整匹配窗口、逻辑工作量归一化和框架差异地图仍未完成。

完整质量988于20:28:05UTC[实际启动](../evidence/D974_readiness_G990/G988_MATCHED_SG974_FULL_QUALITY_START_JSON_parsed.json)，采用原1319题、同一提示/官方严格grader、T0/seed0/thinking=false/max2048/C16、600秒请求预算和零重试；worker task AST与旧958相同。启动前重新核对两端就绪、队列为零、限定功能已退出、SDK已结束、冻结文件未改变以及CPU客户端剩余租期。[实际进程观察](../evidence/D974_readiness_G990/G989_quality988_actual_2204_parsed.json)仅是进行中的部分计数，不是完整分数或通过结论；历史832/941质量失败继续保留，不能用本次限定功能覆盖。

[后续完整质量失败及C16诊断](039-D974-quality-timeouts-and-C16-diagnostic.md)：质量988已实际停止，16个600秒超时，历史限定通过不代替完整质量。
