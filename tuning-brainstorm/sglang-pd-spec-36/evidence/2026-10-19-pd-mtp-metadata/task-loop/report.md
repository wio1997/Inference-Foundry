# M031：PD＋MTP metadata 的实际 CPU 类与消费者

完整、未修改的安装版 `MetadataBuffers` 和 `build_eagle_disagg_draft_input` 已在 CPU 依赖环境中完成 **42 组参数用例、84 次消费者调用**。topk 有效前缀、hidden/DSA 值与 dtype、clone/stack 所有权检查均通过。这里的 wire 是 CPU tensor copy 夹具，接收字段恢复也是显式夹具；不是原生 PD 传输或模型推理。整体仍 **E0**，服务器 accepted_baseline=null。

## 固定实现和环境

安装源码 `ce0a8fea5f4245839d174b2b8c55692be1810a7f`，镜像 `sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263`。22 个输入源码文件在容器中与固定 Git 对象及本地快照逐字节校验；7 个实际导入模块另核对文件身份。没有 AST 抽取或替换这些类/函数。

consumer 的 `get_spec` 始终是实际 `runtime_context.get_spec`。配置来自库自带 `get_context().override_server_args(...)` 测试接口，接口创建 dummy 模型边界并投影真实配置 bag，每次恢复原状态。它不是完整服务启动或实际模型配置解析结果；states=3 是 multi-layer 参数夹具，不代表 GLM multi-layer 模型支持。

CPU venv 重用镜像包文件，排除可选 torch_npu 和启动 .pth，Torch2.10.0+cpu；所有输出 tensor 均在 CPU。任务容器限制1 CPU、4GiB、64pids，禁网、只读根、删除 capabilities，只读挂载四份校验文件，无权重/NPU设备。容器已按精确 ID 删除，现有服务 ID、启动时间、重启计数前后相同。

哈希固定的 config.json 给出 hidden_size=6144、index_share_for_mtp_iteration=True。实际配置 helper 计算 DSA metadata 宽度为 **2048**，两/三请求均覆盖。float32/bfloat16 是本实验声明的 hidden dtype 夹具，不是实际权重、激活或 KV dtype 结论。

## 用例和 oracle

| 部分 | 组数 | 内容 |
|---|---:|---|
| 参数矩阵 | 32 | batch2/3 × hidden16/6144 × fp32/bf16 × states1/3 × DSA维度0/8 |
| 实际配置宽度 | 2 | hidden6144、DSA2048、bf16、states1、batch2/3 |
| DSA缺省/全负 | 8 | batch2/3，各覆盖全缺失、混合缺失、全负、混合全负 |

每组都执行 overlap=false 和 true，后者仅使用 relay recorder 检查实际 `publish`→`stash` 调用顺序及 `RelayPayload` 对象字段，未验证 FutureMap 异步 fence。request 顺序与metadata slot顺序故意不同，slot为`[2,0,1]`的前N项。三请求用例还核对可选 checksum buffer。

源/目标各实例化真实类，按 `get_buf_infos` 的实际指针列表核对字段次序、完整字节和行字节数，再用 CPU `copy_` 模拟传输。接收夹具按真实14项tuple恢复请求字段，并沿用decode源码的“全负DSA行恢复None”规则；没有调用完整 `DecodeTransferQueue._commit_transfer_to_req`。完整 tensor 值、shape、dtype逐项与请求生成公式独立比较，结果中保留完整字节哈希。

## 所有权与有效长度

先向slot写16列旧topk，再写1或3列新前缀。旧尾部确实保留；真实 consumer 按当前 `num_states` 截取后stack，未把旧尾部带入有效输出。因此不能把“尾部没有清零”直接判为当前数据错误。

`get_buf`返回的各个tensor均有独立存储。拿到快照后清零全部目标wire buffer，所有快照保持不变；消费者stack完成后再改写请求中的topk/hidden/DSA，输出仍与oracle一致。与此同时，`bonus_tokens`按实现有意引用传入last-token tensor，overlap下`future_indices`引用batch请求索引；这些别名已单独验证，不能泛称所有字段都被clone。

DSA缺失或全负的接收行会使该批seed回退为None；有效全正行按原值进入消费者。这是当前的全批回退合同，未改变它来制造性能收益。当前实验不声称覆盖了consumer的每一条防御分支。

## Ascend适用性与条件观察

实际 remap selector 的首条件是 `is_cuda() or is_hip()`。本任务 Ascend-only 前提下不走 request-relative→local-slot 的 CUDA/HIP remap；本轮实际 CPU selector也返回False，没有改platform事实或强行执行NPU分支。

另记录三项边界观察：

- 复用slot后用hidden=None写入，会保留旧topk/hidden/DSA，同时写入新bootstrap_room。这是非正常生产者夹具；尚未证明真实EAGLE prefill会把这种请求交给消费者，不作为已发生PD故障，也未据此改代码。
- topk17超过固定16列容量，CPU `copy_`抛RuntimeError。
- hidden宽度15写入宽度16也抛RuntimeError。两种错误中bootstrap_room仍为旧值；此前字段可能已有部分写入，不能据此推断原生传输的原子性。之后有效写入可恢复。

这些观察保存在[完整结果](../cpu-result.json)，没有删掉异常或把条件反例记成原生NPU错误。

## 结论边界和下一步

L016只接受上述CPU类/消费者合同。CPU类使用uint64 bootstrap_room，NPU源码分支使用int64且分配于NPU；没有用CPU重定向冒充NPU D2D结果。logprob/sampling-mask分支、完整decode接收、真实FutureMap/fence、connector传输、native tree/verify、PD模型输出及性能均未验证。没有代码候选或生产变更，没有速度结论。

下一步追踪M025实际NPU DSA indexer：何时使用这份2048宽度seed、何时跳过或重算TopK、全批seed回退对短/长上下文分别意味着什么。先建立实际分支与必要工作证据，再决定优化；原生80个安全链式用例和任务自己的PD＋MTP基线仍等待明确资源分配。

## 复核

- [CPU结果](../cpu-result.json)、[容器身份与清理](../cpu-container.json)、[源码索引](../source-index.json)、[冻结用例](../specs/L016-cases-v1.json)。
- [探针](../scripts/probe_pd_metadata_cpu.py)、[冻结spec](../specs/L016-R003.json)、[ZCode精确session审计](../zcode-audit.json)、[产物索引](../artifact-index.json)。三个冻结命令各执行一次，实际元数据别名deepseek-flash，后端发布版本UNKNOWN；没有原始模型I/O或私有配置。
- 复现以此目录为scratch root，提供哈希匹配的四份模型/tokenizer文件和原SSH/Pod/镜像合同，再运行 `python3 scripts/execute_frozen.py specs/L016-R003.json`。探针只读config，其余三份文件仅校验；模型权重从未挂载。跨环境需新建位置合同与spec，不能编辑旧输入后沿用旧回执。
