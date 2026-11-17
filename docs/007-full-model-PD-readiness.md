# 完整权重与原生PD启动门槛

两个节点各95个固定文件、528805292902 bytes完成SHA校验。恢复程序复用原下载中已校验且inode/size/mtime保持不变的文件，只恢复缺失分片：P为43、D为14。原有失败记录保留。这里不声称对未改变的500GB重复计算了SHA。

G65按完整TP8/EP8、BF16 KV、DSPARK五个草稿预测（SG参数6含anchor）、context263296、C16与D侧目标/草稿full图提交P/D启动。运行时源码SHA、只读模型挂载、逐rank加载覆盖/PD注册日志及全部任务缓存目录均预先固定。两端在native feature validation阶段退出，未进入权重加载，因此没有模型服务或性能验收结论。

直接原因：固定daily源码的V4.1 DSpark PD gate只列出Mooncake/NIXL/MoRI，并要求static ragged verify、CP1/DCP1。原生AscendKVManager继承MooncakeKVManager，SDK有DEVICE_RDMA入口；这些是试接入依据，不能替代真实目标/草稿KV和Engram历史状态的跨机验收。

G68候选只扩展一个backend capability tuple，加入ascend；另加严格的NPU、原生继承、TP8/EP8/PP1/CP1/DCP1、static verify及DEVICE_RDMA条件，原始其它feature checks保留。完整模型加载、图、PD字节与功能仍未验证。代码与初次失败共同存档，不把候选当作已支持。

本任务AISBench输入为文本；候选使用language-model-only，后续vLLM臂须使用相同文本范围。固定daily源码明确不支持A2 vision与该DeepEP组合，本记录没有多模态质量结论。

固定vLLM镜像实测安装memcache-hybrid/memfabric-hybrid 1.3.0，已记录实际.so的SHA、协议入口字符串和env名称。其native V2 SFA PD要求成对保存目标/草稿prefix；AscendStore的GVA layerwise MemCache路径存在。服务、双向跨机字节一致、模型复用和无重算仍需实际验收，SDK存在不是这些门槛已通过。

- [P端G65退出状态](../evidence/full_model_G65/2203/state.json)
- [D端G65原始日志](../evidence/full_model_G65/2204/server.log)
- [P端95文件权重凭证](../evidence/full_model_G65/2203/verified_weight_snapshot.json)
- [D端95文件权重凭证](../evidence/full_model_G65/2204/verified_weight_snapshot.json)
- [G68局部接口接入候选](../code/full_model_candidates/sg_pd_g68/pd_gate_bridge.py)
- [固定镜像SDK二进制清单](../evidence/fixed_image_SDK_binary_inventory.json)

G68首次复验在新增检查中出现UnboundLocalError：cfg局部视图尚未赋值就被引用，未进入模型加载。G69将检查移到原生cfg=resolving_view(server_args)之后，并对原生函数该段的变量绑定做CPU回归检查；拓扑、静态verify、图和其它原始feature guard不变。此处保留包装缺陷，不能用它判定硬件或模型不支持。

- [G68实际失败观察](../evidence/full_model_G68/2204/failure_observation.json)
- [G69检查顺序修正](../code/full_model_candidates/sg_pd_g69/pd_gate_bridge.py)
