# 固定0.30镜像中的 PD / DSPARK prefix 契约

从保留的固定4d镜像停止容器读取真实源码，确认注册表把MultiConnector映射为AscendMultiConnector；原生实现具备配对target/draft prefix provider选择，故不必凭新版本报告重写该控制路径。

该SFA/DSPARK路径的P端明确要求CP乘积1、eager prefill、本地prefix关闭。外部配对prefix是另一路径：AscendStore需use_layerwise=true、GVA数据平面、producer角色，且MultiConnector只能选择一个配对provider。固定memcache backend声明GVA，draft页注册还要求MemCache atomic batch_write_finish。原生安全guard保留；用户参考的PCP4及non-layerwise Mooncake配置不直接等价于此路径。

AscendStore的layerwise配置还声明PIECEWISE图要求。P eager与D完整target/draft图须分角色核对实际解析结果，不能在D照搬use_layerwise后仍宣称FULL图，也不能为追求相同命令行而移除草稿KV正确性guard。

G365在CPU执行实际固定源码的9个控制分支，确认use_layerwise/data-plane/producer选择、原生P端CP/prefix/eager拒绝条件和图策略。key对象和aux层ID为接口夹具，不验证真实key hash、完整模型布局、SDK、store、PD、图或性能。下一步仍需实际DEVICE_RDMA读写/发布/关闭门槛、真实draft页注册、完整图和四档首次冷热测试。

- [固定源文件与逐项SHA](../evidence/connector_source_G359.json)
- [原生CPU控制分支结果和夹具范围](../evidence/connector_contract_G365.json)
- [SFA P端保护](../code/fixed_connectors_G359/vllm_ascend/distributed/kv_transfer/kv_p2p/sfa_pd_rd2h/dspark_context.py)
- [配对prefix provider选择](../code/fixed_connectors_G359/vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py)
- [配对draft页注册约束](../code/fixed_connectors_G359/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/dspark_prefix_cache.py)

离线复现控制分支：

```bash
python3 code/connector_contract_G365/verify_vllm_connector_contract_g365.py --source-root code/fixed_connectors_G359 --output connector-contract-local.json
```

此命令只执行CPU接口控制分支，没有NPU或模型负载。
