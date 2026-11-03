# DispatchFFNCombine 二进制身份：可观察链与未证环节

## 结论

现有 EP8 的两个任务已经绑定到层、设备 task 和聚合算子 CSV，但仍不能确定实际 kernel 对象。通过只读检查 169 自有 stage 容器中的 CANN 9.2 和自定义算子安装包，找到了**可用于将来诊断的实际加载 API 候选**，并发现一种基于已安装头文件的关联方法；本次没有执行该方法、没有采集新设备工作，也没有改变运行中的目标进程。静态符号和静态指令调用不证明被测进程走了该分支，更不能绑定历史样本。

## 静态证据

1. 冻结源码 `sources/vllm-workspace/vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py:588–609` 在 `enable_fused_mc2 == 1` 且未选 mega_moe 的路径调用 `torch.ops._C_ascend.dispatch_ffn_combine`；本文件证明该源码分支的接口，不单凭其证明某一历史内核是某个安装对象。
2. 个人 stage 容器 `/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/config/ascend910_93/dispatch_ffn_combine.json` SHA-256 `992dae27e6cafd4bd835534d58f7ac73217c7b8b5c5299783f33c71fd8bca507` 的 `binList` 三条分别映射 `17df...`（BF16+NZ）、`de54...`（BF16+ND）、`fc6c...`（FP16+ND）JSON；各有不同 simplifiedKey。每项输入还声明 `format_match_mode: FormatAgnostic`，同一路径的三个 JSON 的 `tilingKey` 均为 `1000010`。因此 dtype/format/tilingKey 可提供候选线索，但不能替代运行时命中与实际加载证据；不能由 CSV 格式断定排除其他选项。`17df...` JSON 的 `binFileName` 和 `kernelName` 均是完整 `DispatchFFNCombine_17df546b46e1ea56007584fac7e6f429`，`magic` 为 `RT_DEV_BINARY_MAGIC_ELF`，自述对象 SHA `4cf1474cb1009b96fc743f5a396a9c07a6980146c41818b60cdbf0d3453a43a9`。这是安装元数据而非任务命中的记录。
3. 安装包 `.../op_api/lib/libcust_opapi.so` SHA-256 `5e7735b6b84626596a5eb7ac6e4b2a028fa227ee8f267f435ca6cd5f4928fbea` 的 `readelf -d` 列出 `libnnopbase.so` 为 NEEDED 依赖。`nm -D -C` 显示它导出 `aclnnDispatchFFNCombine`、`aclnnDispatchFFNCombineGetWorkspaceSize`、两个 `aclnnInnerDispatchFFNCombine...` 入口，并引用 `CreatAiCoreKernelLauncher(...)`。这些符号位于同一库，但不能据此确认动态执行时的精确调用顺序、具体库加载路径或功能分支。
4. `/usr/local/Ascend/cann-9.2.0/aarch64-linux/lib64/libnnopbase.so` SHA-256 `ab8145a5f6a7c3aa4f1e274263f206dcf4a8737e2266908112bea49d2b3a4534` 导出 `NnopbaseAclrtBinaryLoad`、`NnopbaseMC2KernelLaunch` 和 `NnopbaseExecutorKernelLaunch`，引用 `aclrtBinaryLoadFromData`、`aclrtBinaryLoadFromFile`、`aclrtBinaryGetFunction`、`aclrtBinaryGetFunctionByEntry`、`aclrtLaunchKernelWithHostArgs`、`aclrtStreamGetId`、`aclrtGetThreadLastTaskId` 及 `aclmdlRICaptureGetInfo`。`objdump -d --disassemble=NnopbaseAclrtBinaryLoad` 在 `0x28f4ec` 显示 `bl aclrtBinaryLoadFromData@plt`；`objdump -d --disassemble=NnopbaseMC2KernelLaunch` 在 `0x309b08` 显示一条分支的 `bl NnopbaseExecutorKernelLaunch@plt`，该函数中还有其他分支。以上证明该安装版本**存在**从 data 加载的候选路径及一条 MC2 下发分支；没有证明运行时是否命中、路径是否唯一路径，也没有证明 MC2 当时选择了该分支。
5. 同镜像 `/usr/local/Ascend/cann-9.2.0/aarch64-linux/include/acl/acl_rt.h` SHA-256 `62d127fbb899b1476e054b1815ca6e0c71a362868c655f0e3da4616fb7598fb7` 定义：`:4497` `aclrtBinaryLoadFromData(data,length,options,&binHandle)`；`:3527` 文件加载；`:3338` 按 kernelName 从 binHandle 获取 funcHandle；`:3549` 按 funcEntry 获取 funcHandle；`:5270` 以 funcHandle、stream 等 `aclrtLaunchKernelWithHostArgs`；`:2120` 当前线程 last task ID、`:2133` stream handle 到 stream ID；`:4783` `aclmdlRICaptureGetInfo` 获取流的捕获状态及 modelRI。`:4814` 的 `aclmdlRIDebugJsonPrint(modelRI,path,flags)` 只声明输出模型信息，未承诺能输出映射所需节点/函数身份。头文件没有保证 last task ID 与 profiler 物理 task ID 可直接等同，也未保证 replay 会再次触发加载或 host launch。

## 最小取证设计（未执行）

如后续有必要，在用户已授权范围内的独立诊断进程从启动期观察以上函数：记录 pid/线程、时间、返回码和对象生命周期；成功的 data 加载对调用时传入的精确 `data,length` 算摘要并映射到成功返回的 binHandle，文件加载记录路径与加载时文件内容摘要，再关联成功的 `GetFunction` 的完整 name 或 entry 与 funcHandle，随后关联成功 host launch 的 funcHandle、stream 及该调用所在 capture model 和已有提交层标记。需要时查询线程 last task ID、stream ID，但必须验证它们与 profiler model/physic stream/task 的对应关系，特别是 capture、异步任务、并发与 replay；如果不能验证，则仅报告 handle 和模型的已证明部分，模型/task 链为 UNKNOWN。

- data 内容有可能是包装或转换后的对象，摘要不匹配安装 `.o` 时不能直接断言对象不同；需验证真实数据结构及摘取方式，匹配也须核对装载前后内容。仅凭文件名、当前文件 SHA、地址、静态签名或共同 tilingKey 不足以证明历史绑定。
- 拦截候选仅是可能路径；预加载和符号替换能否覆盖以 `dlsym`、局部绑定、内部符号、其他 runtime 或图编排提交的路径尚未证明。验证覆盖要有加载/卸载与所有函数查询的完整生命周期，并防句柄重用、PID 重启、同名库和不同 rank 混淆；对 未观测调用不能当不存在。不要读取、打印或持久化张量、权重数据与用户输入；只计算二进制数据摘要及记录必要元信息。
- 在图 capture 时的 host launch 可能只发生一次，后续 replay 可能没有对应调用；需要运行实例与图节点的可靠对应，`aclmdlRIDebugJsonPrint` 输出是否包含节点身份、是否等价、是否有安全的现成采集方式都未验证。采集本身会改变 host 时间，不能当作未插桩基线性能。不要注入到已有共享服务，也不必为取证而新增跨 rank 同步。

### 可复现只读命令

以下在上述自有 stage 容器运行：

```sh
sha256sum /usr/local/Ascend/cann-9.2.0/aarch64-linux/lib64/libnnopbase.so /usr/local/Ascend/cann-9.2.0/aarch64-linux/include/acl/acl_rt.h
nm -D -C /usr/local/Ascend/cann-9.2.0/aarch64-linux/lib64/libnnopbase.so | grep -E 'NnopbaseAclrtBinaryLoad|NnopbaseMC2KernelLaunch|NnopbaseExecutorKernelLaunch|aclrtBinaryLoad|aclrtBinaryGetFunction|aclrtLaunchKernelWithHostArgs'
objdump -d --disassemble=NnopbaseAclrtBinaryLoad /usr/local/Ascend/cann-9.2.0/aarch64-linux/lib64/libnnopbase.so | grep -E 'NnopbaseAclrtBinaryLoad|aclrtBinaryLoadFromData'
objdump -d --disassemble=NnopbaseMC2KernelLaunch /usr/local/Ascend/cann-9.2.0/aarch64-linux/lib64/libnnopbase.so | grep -E 'NnopbaseMC2KernelLaunch|NnopbaseExecutorKernelLaunch'
```
