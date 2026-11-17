# Engram 原生 API 与接入路径

昇腾 [CANN 9.1 host_register_v2 文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/API/runtimeapi/aclpythondevg_01_1162.html) 和 [host_get_device_pointer 文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/API/runtimeapi/aclpythondevg_01_1163.html) 列出 Atlas A2 训练/推理产品支持 Host 注册与设备映射指针。该文档支持优先验证原生 UVA 路线，并不证明当前容器、910B SoC、分片和 Triton 调用已通过。

固定 vLLM 镜像的 Engram 代码使用 aclrtMallocHost、aclrtHostRegisterV2、aclrtHostGetDevicePointer，并有 INT8/组32反量化与 F8_E8M0 scale 解码路径。下一步验证这些源码在两套固定 CANN/torch_npu 环境中的调用契约，优先复用现成实现，不搬运另一套 CANN 的二进制。

SGLang 的固定配置实现已将 DeepseekV41ForCausalLM 映射到 DeepseekV4ForCausalLM，因此仅看到模型类名不同不构成缺失。需要继续核对的具体缺口是 INT8_SYM Engram 参数存储/权重解释、Host 注册与 lookup、离线旋转、状态写回及图执行。

该阶段没有新增 DeepSeek 模型性能数据，不能升级为完整模型支持或优化收益。
