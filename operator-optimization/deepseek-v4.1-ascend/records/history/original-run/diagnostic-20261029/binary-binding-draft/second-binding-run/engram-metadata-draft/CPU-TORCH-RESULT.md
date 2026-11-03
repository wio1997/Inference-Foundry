# 真实 CPU torch 切片兼容检查

本机 ZCode 使用 `deepseek-flash` 执行一次 `run_real_cpu_torch.py`，在 169 本人的 stage 容器里只使用 synthetic CPU tensor；脚本关闭 PyTorch backend autoload，明确检查 `torch_npu` 未导入。未启动 NPU、容器或推理服务，未接入生产进程。

CPU torch `2.10.0+cpu` 将 `[13,2,24]` 的两个 slot 切成 `[13,24]`：wrapper 分别记录 stride `[48,1]`、storage_offset `0/24`、non-contiguous、CPU 与 int32；默认禁用时没有 patch；启用后 fake 原方法的返回值与独立 CPU contiguous 结果一致。结果和 stderr 保存在 `real-cpu-stage.stdout` / `.stderr`；ZCode 会话与模型验收在 `real-cpu-zcode-output.jsonl` / `real-cpu-zcode-verified.json`，模型验收确认两次模型请求均为 `deepseek-flash`。

这验证普通 CPU torch.Tensor 在该 Python 环境上的入口属性读取和弱引用不会导致此特定测试失败，不能推广到 NPU storage、真实模型返回、性能、不同形状或编译/graph 模式。`storage_wrapper_instance` 只标记弱引用 Python wrapper 对象，不证明实际 storage 别名、NPU 内存生命周期或跨流读取完成；graph replay 是否执行 Python wrapper 未测试。代码本身仍是待主控复核草稿。
