# 固定模型和接入合同

权重来自 [Eco-Tech ModelScope 仓库](https://modelscope.cn/models/Eco-Tech/DeepSeek-V4.1-Flash-1017-w4a8)。主权重与配置文件固定在提交 `4fbbf8c78aeed000add9765003bd842099844006`；逐文件 revision、字节数和 SHA256 见清单。完整检查点约 528.8 GB，其中两个主 Engram 文件合计约 202.8 GB。下载使用独立有界进程、有限并发、Range 续传和逐文件 SHA256；当前下载仍未完成。

导出量化描述含 W4A8_DYNAMIC 专家、W8A8_DYNAMIC 层、FLOAT 层及两个 INT8_SYM Engram 表。Engram 权重为 INT8，块大小 32，scale 为 F8_E8M0；值投影存在离线旋转。不能把这些表当作 FP8 数值解释，也不能遗漏 export extra 表或全局旋转文件。

固定镜像的静态检查找到两个框架的模型与 DSPARK 路径，但尚未验证全模型加载和真实 NPU 分派。SGLang 的通用 Engram 默认创建 FP8 权重，CPU placement 路径包含 CUDA 专用调用；这说明需要检查 checkpoint 到参数、lookup 和旋转的完整调用路径，尚不能断言现成路径支持 INT8_SYM/NPU host offload。vLLM-Ascend 有专用 INT8/host UVA Engram 代码，也仍需以 910B 实测确认。

配置声明 DSPARK block size 5、三个 nextn 层。启动参数中“3 speculative tokens”与训练 block size 的关系须按各框架实现核对，不通过修改模型配置强行统一。保留 EP8、完整 D target/draft 图、异步重叠、前缀缓存、分块 prefill，以及用户要求的 BF16 KV；若混合 KV 实现存在约束，记录真实缺口并适配，不把降级设置当作目标完成。

验收顺序采用已有实现优先、必要薄适配：固定源代码和配置 → 完整权重映射/所有 rank → Engram 数值与状态 → PD 与 DSPARK/图 → 独立前缀全形状预热及编译检查 → 四档首轮冷热 → profile 与正式差异图。局部导入、源码存在、单算子或其他硬件结果均不代替全模型验收。
