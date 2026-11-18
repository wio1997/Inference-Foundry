# 固定V2运行时与SDK证据范围

继续从保留的已停止任务容器读取固定0.30镜像源码，取回Ascend V2 worker/speculator及上游GPU speculator，逐文件SHA校验。全源取回共61个V2/旧spec目录文件及30个上游GPU spec文件；本分支只保存本轮选中的3个原始源文件及SHA指针。未因容器停止而重新启动，也未做包导入、服务启动或NPU操作。

Ascend V2初始化会配置DSpark常驻draft cache spec，保留persistent draft layer，避免将其别名到target的layerwise scratch。`_maybe_send_draft_kv` 在选定协议时将P侧辅助状态初始化为draft KV，再由D拉取已分配页面。该显式SFA协议只在connector选择含SfaRemoteD2HConnector时启用，MultiConnector递归识别；不能仅凭这条路径没有启用，就声称所有其他PD connector不支持DSpark。

SfaRemoteD2HConnector本身要求P不启用sparse offload、D启用sparse offload。用户PR使用generic MooncakeHybridConnector；其传输实现含MTP层处理。两者的内存层级、draft上下文与注册布局应分开核对，尚未凭源码快照决定公平对照的最终传输路径。实际draft上下文、常驻页面、图模式与接受率仍须运行验证，不移除原生安全guard。

G428另读取固定镜像memcache-hybrid包的24个Python/ELF文件，保留逐项SHA及方法名字节检查。`_pymmc` SHA与之前固定镜像SDK盘点一致，含batch_write_finish、batch_copy、batch_get_key_info等名字节。DSparkPrefixCache.register检查的是实际backend.store上的batch_write_finish方法；backend的旧SDK兼容包装不能代替这个要求。名字节存在只提供后续运行检查的依据，不证明Python类绑定、SDK初始化、DEVICE_RDMA/GVA读写、原子发布或真实命中通过。

- [V2选定源码与SHA](../evidence/fixed_v2_runtime_source_G430.json)
- [实际SDK文件与名字节结果](../evidence/fixed_SDK_symbols_G428.json)
- [Ascend V2 cache初始化与draft传输调用](../code/fixed_v2_runtime_G430/vllm_ascend/worker/v2/model_runner.py)
- [Ascend DSpark上下文初始化](../code/fixed_v2_runtime_G430/vllm_ascend/worker/v2/spec_decode/dspark/speculator.py)
- [上游GPU DSpark speculator](../code/fixed_v2_runtime_G430/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py)
- [原生配对prefix实际store要求](../code/fixed_connectors_G359/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/dspark_prefix_cache.py)
