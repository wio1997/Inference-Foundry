# INT8 Engram Host UVA 与图重放组件验证

2026-11-17 UTC，在实际 Ascend910B1 上完成固定 SGLang A2 daily（CANN 9.1、torch_npu post6）和固定 vLLM-Ascend 0.30.0rc2（CANN 9.2、torch_npu post7）的各八卡组件验证。原镜像 ID、设备名、依赖版本、源码 SHA、逐 rank 数据及清理状态在 receipt 中保留。

| 对象 | 数值用例 | 图捕获 | 改变输入及主机权重后的重放 | 结果 |
|---|---:|---:|---:|---|
| SGLang G49 | 80 | 16 | 48 | 全部 BF16 逐位匹配 CPU float64 参考 |
| vLLM G50 | 80 | 16 | 48 | 全部 BF16 逐位匹配 CPU float64 参考 |

覆盖 INT8 代码、组32 E8M0 scale 的数值解释、NPU 索引、Host 注册/UVA 指针、多块寻址、chunk 边界、非连续 stride、头切片、填充头、零 token 和越界置零。对应 rank/图的三个重放输出 SHA 在两套镜像中相同，每次重放结果均变化。各八卡独立执行；这里没有 TP8/EP8 集体验证。

SGLang 复用固定 vLLM 镜像中未改写的 Triton kernel 与 Host buffer 源码声明，在原 CANN 9.1 环境编译。仅兼容实际头文件中两参数 aclrtMallocHost ABI，HostRegister 使用 flag=2；没有复制 CANN 9.2 二进制。vLLM 使用原生模块，HostRegister flag=268435458。两者均通过当前设备的数值与图重放用例，不声称两套 flags 可任意互换。

vLLM G49 首次探针在数值执行前发生 DeviceOperator 循环导入。G50 仅提前导入 vllm_ascend.ops，遵循框架初始化顺序；kernel 未改动。失败的八个 worker 日志和结果一并保存。容器最终 ExitCode=137 来自控制器停止承载 sleep 的容器；数值 worker 的退出码均为0，OOMKilled=false，清理后八卡没有进程。

范围：没有完整模型权重装载、Engram gate 的离线旋转验收、PD 状态传递或服务吞吐数据。本组件通过不能替代完整模型、目标/草稿图或性能验收。

- [SGLang G49 原始结构化证据](../evidence/components/G49_sglang/receipt.json)
- [vLLM G49 导入失败](../evidence/components/G49_vllm_import_failure/receipt.json)
- [vLLM G50 原始结构化证据](../evidence/components/G50_vllm/receipt.json)
- [原始与发布字节 SHA 对照](../evidence/components/provenance.json)
- [离线组件档案核验](../scripts/validate_components.py)
