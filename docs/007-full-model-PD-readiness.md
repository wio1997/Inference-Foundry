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

G69修正配置绑定后通过原来的feature gate，随后scheduler的spawn子进程在导入启动入口时，再次执行创建server_help.txt，触发FileExistsError。两个完整P/D控制器均退出并核验全部8卡进程清空；尚未进入模型权重加载。此失败是启动包装缺陷，不是模型或硬件支持结论。

G70把入口副作用放入main函数，仅在__name__为__main__时执行。实际用Python spawn导入名__mp_main__在无candidate文件目录导入修正入口，验证无文件写入，main仍可调用。两端已提交同配置复验；这项CPU回归不代表完整模型、PD、图或性能通过。

- [G69 P端原始失败与spawn调用栈](../evidence/full_model_G69/2203/server.log)
- [G69 D端清理凭证](../evidence/full_model_G69/2204/state.json)
- [G70带主进程保护的启动入口](../code/full_model_candidates/sg_pd_g70/launch_pd.py)
- [G70 spawn导入回归](../evidence/full_model_G70/spawn_import_regression.json)

G70已进入八卡scheduler初始化，随后native ModelConfig给嵌入式草稿读取HF的ascend quant_method，报Unknown quantization method。该检查点的ModelSlim描述中，mtp.0/1/2共1226项均为FLOAT，和索引中的1226个mtp键集合一致；主模型专家仍为W4A8_DYNAMIC。草稿FLOAT是原检查点定义，不是把用户指定W4A8主模型降为非量化。

G72在父进程安装了只针对这个固定草稿的HF量化标签处理，但spawn worker在aux-hidden配置解析前尚未导入该包，仍报同一错误，没有嵌入式草稿凭证。两轮失败日志、源码和全部8卡清理凭证均保留。G75将适配显式放入worker的__mp_main__导入入口，在scheduler构造前安装，并逐PID记录。对隐式继承的target ModelSlim选择，草稿按检查点FLOAT恢复原生unquantized路径；实际分片头须全部为非打包浮点/整数表类型，否则失败，不修改检查点文件。

另外，固定ModelSlim用hc_head_前缀检测DSV4命名，而此描述缺少该前缀。原生模型需要model.layers.*.self_attn/mlp名称，检查点描述为layers.*.attn/ffn。用镜像原生remapper对46637项目标权重描述做静态映射，得到46637项且无格式冲突。G75仅添加原生名称alias、保留原项与格式，不改变W4A8/FLOAT数值语义；真实模型加载仍待验收。

- [G70 D端量化错误](../evidence/full_model_G70/2204/server.log)
- [G72 D端相同错误与worker缺少适配的记录](../evidence/full_model_G72/2204/server.log)
- [G75 worker适配安装入口](../code/full_model_candidates/sg_pd_g75/launch_pd.py)
- [G75草稿分片类型与量化范围检查](../code/full_model_candidates/sg_pd_g75/draft_quant_bridge.py)
- [G75原生量化名称alias](../code/full_model_candidates/sg_pd_g75/target_quant_alias_bridge.py)
- [原生名称静态契约结果](../evidence/target_quant_name_contract_G74.json)

G75在两端各8个worker中完成真实分片header核对：1226项草稿张量为1204项BF16与22项FP32，逻辑权重32428105992 bytes，三个MTP block全部覆盖；子进程安装入口确实执行。随后模型构造在MoE量化描述缺失处退出，全部8卡进程清理完成，尚未形成完整模型服务。

新取回的固定镜像get_quant_config显示：即使ModelConfig已选择modelslim，HF的ascend标签仍优先返回，构造器只收到quant_method、packed_modules_mapping、hf_config，没有真正的逐层描述。使用这段原生函数和记录型配置类在CPU执行，复现其在文件解析前返回的路径。该probe不执行模型或NPU。G78在同一固定模型范围内恢复已校验的ModelSlim描述文件，保留原生packed mapping，再做原生参数名alias；主模型W4A8/草稿BF16、完整拓扑、图和PD要求不变。真实完整模型仍待验收。

- [G75 D端模型构造失败](../evidence/full_model_G75/2204/server.log)
- [G75 P端草稿dtype凭证示例](../evidence/full_model_G75/2203/embedded_draft_quant.pid134.json)
- [原生loader HF优先路径](../code/fixed_native_loader/weight_utils.py)
- [CPU复现原函数的配置选择](../evidence/native_quant_precedence_probe_G78.json)
- [G78量化描述来源修正](../code/full_model_candidates/sg_pd_g78/target_quant_alias_bridge.py)

可在CPU执行 `python3 scripts/reproduce_native_quant_precedence.py` 重放原函数的字典元数据选择分支；[复现脚本](../scripts/reproduce_native_quant_precedence.py)使用原生metadata解析器、原生resolver和记录型配置类，不启动模型或访问NPU。
