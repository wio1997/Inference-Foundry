# MC2 frontend residual and capability-cache candidate

Latest: [H6 completed](H6_DECISION.md) with Run256 scopedPOSITIVE/formalPARKED and restored stock. Build-running/no-model wording below is historical. Residual descriptor/executor/queue claims remain unknown and require their actual lifetime/ordering contracts, not a cached raw pointer.

Follow-up after Run254: [CPU_INSTRUCTION_DECISION](CPU_INSTRUCTION_DECISION.md) identifies all248 sampled queries as these two MC2 predicates and records native host-cache correctness. H6 is now the sole candidate under isolated build, not installed/model-tested. The earlier PARKED/offline-only decision below is preserved as the pre-Run254 decision; new evidence advances source/build validation without declaring a global saving budget or PERF_KEEP.

本阶段没有新增代码级性能 KEEP。H5限域23-token完整请求的两对D收益已保留，Current=None。H6缓存候选不在现场激活，不追加NPU A/B，也不替代最大的未归因host区域。单独的H7只补一次profiler-OFF CPU函数证据，后续归约同一raw。

## Existing Run249 decomposition

[离线归约](mc2_frontend_residual.json)在同一CPU线程的不重叠380次MC2 envelope内计算子scope union，避免嵌套加总。

| D15两操作合计、profiler ON | 时间 |
|---|---:|
|dispatch+combine outer scope|391.290280ms|
|输出tensor allocation union|12.337990ms|
|所有导出child scope union|18.707190ms|
|outer未被导出child覆盖|372.583090ms|

单独原CANN producer workspace query共19.920690ms，位于外层内；这里不能再直接加进上表。残差含descriptor/executor准备、调用/提交和profiler/调度开销，不是372.6ms可删CPU。七个dispatch输出分别被expert MLP和异步combine消费；当前没有buffer、descriptor或executor复用的生命周期/动态路由安全证明。

固定op-plugin源码中TASK_QUEUE_ENABLE=1走EXEC_NPU_CMD_V1，在调用线程转换参数、准备workspace/executor后再入队；2走V2。当前未切换flag或扫描参数。[原始固定版本](https://raw.githubusercontent.com/Ascend/op-plugin/8b9c8534fa41eff367a41c155843daa530ab3a08/op_plugin/utils/op_api_common.h)与[来源hash](mc2_source_refs.json)保存边界。

## Installed binary, rather than an assumed wheel source

实际libtorch_npu.so SHA256为83fb9a0eb249aef6bca7f8463047fcb4d3062cc4849f17ddf31a5e050c838842。内部symbol table被strip；不能使用objdump显示的邻近长模板名来命名函数。

[紧凑实际反汇编](mc2_compiled_capability_path.json)显示dispatch V4 literal0xe7e538后的0x31edf40和combine V4 literal0xe74220后的0x31ba4e8均调用0x490a9d0。它在0x490aa88与0x490aa98调用GetOpApiFuncAddr0x490a010两次。能力check发生在后续EXEC_NPU_CMD已有静态函数指针guard之前；后者已经缓存，并不等于前者也缓存。该局部编译结构与源check_aclnn_kernel_available一致。具体op注册函数仍无内部名字，但两个V4分支调用及其两查询callee是实际编译事实。

源loader的custom/default路径与handlers分别在初始化时形成const vectors；查询仍会realpath/dlopen/dlsym/owner检查。ConvertType中多数function pointer本身static，不能声称所有descriptor转换都重做符号搜索。没有现有证据支持把INFO/DEBUG logging当当前主因。

## CPU-only discriminator and source-only patch

[原始CPU Job](../../jobs/MC2-LOOKUP-CPU-20261007/result.json)实际inner0/VALID。独立同安装进程warm20后11×100次两符号查询，NPU_initialized=False、model_request=False；before/after32owners和source不变。dispatch wall中位53.03213us/CPU52.773us，combine wall52.55540us/CPU52.4544us。760调用机械外推40.1232614ms只是量级参考，含ctypes及独立进程查找状态，**不是模型预算、上界或可测收益**。落入冻结RESET的25–100us档，决定是继续offline，不进入NPU A/B。

[最小patch](mc2_capability.patch)只在两个V4分支前加入函数局部static const bool；V3/V2、分配、所有CANN调用参数、数学/通信/输出不变。[逐文件反向替换校验](mc2_capability_identity.json)与[独立审查](ASTRA_MC2_CAPABILITY_REVIEW.md)支持固定、已就绪库生命周期下源码无blocker。库路径、文件、权限及加载状态必须在worker生命周期固定；负缓存不支持热安装新V4库，换库须restart。native wheel编译/import、模型correctness及matched A/B/E2E尚未执行，不能将source机制检查写为完整correctness通过。

H6作为最大Gap假说PARKED。descriptor/executor缓存仍没有安全证明。剩余首要证据由Run254的一次OFF CPU IP/callchain诊断提供；其结果与源码继续归约，不按缺精确wall预算结束研究。
