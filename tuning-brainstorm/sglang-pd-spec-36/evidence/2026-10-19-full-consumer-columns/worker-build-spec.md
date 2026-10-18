# L005 限定脚本构建任务

这是主控明确授权的有限代码编写任务，属于 Task `sglang-pd-spec-36`、Loop `L005-full-consumer-cpu-profile`。本次只创建 `scripts/probe_full_consumer_cpu.py`，不运行该脚本、不改其他文件、不删除或覆盖任何 Run。允许读取下列指定输入与本任务 AGENTS、CONTRACT 和本 Loop goal/state。不要读取其他会话、其他 Task、`.private` 或认证材料。

输入：

- `scripts/probe_ascend_descriptor_cpu.py`
- `runs/L004-R001-descriptor-cpu/candidate-method.py`
- `runs/L004-R001-descriptor-cpu/result.json`
- `sources/installed/python/sglang/srt/disaggregation/ascend/conn.py`
- `sources/installed/python/sglang/srt/disaggregation/common/utils.py`
- `sources/installed/python/sglang/srt/disaggregation/mooncake/conn.py`

主控目标：补齐真实 Python consumer，尚不实现新优化。L004 原 sender 和已有 range sender 为两个变体；唯一变化是 fixture 的 `_transfer_data` 从 no-op 变为原样提取的 `MooncakeKVManager._transfer_data`。原生 engine 的 `batch_transfer_sync` 仍为受控 no-op，不读地址、不访问设备、不联网。

实现规格：

1. 只用标准库与 NumPy。ROOT 为脚本父目录的父目录。OUT 固定 `runs/L005-R001-full-consumer`，用 `mkdir(exist_ok=False)` 防覆盖。不得导入 SGLang、Torch、NPU 或执行旧脚本。
2. 从真实源码 AST 选择 `AscendKVManager.send_kvcache`、`group_concurrent_contiguous`、`MooncakeKVManager._transfer_data`，用 postponed annotations 隔离执行；从 L004 已冻结 `candidate-method.py` 选择 `range_metadata` 与 `candidate_sender`，不重新发明 candidate。不要触发旧脚本顶层代码。记录文件 SHA256、方法行号与 AST dump hash。
3. fixture 固定 PP=1、非 hybrid、custom_mem_pool=false，78层、地址与交替 item_len 沿用 L004。envelope 校验仍 no-op；明确 scope。用 `types.MethodType` 绑定真实 consumer。fake engine 需要在正确性阶段捕获三个 Python list 的有序内容、session id、调用次数与返回码；计时阶段 callback 不复制/遍历 list，只增加调用计数并返回0。
4. 沿用 L004 的151个确定性语义Case（包括随机seed、dtype、只读非连续视图、空侧、错长、HiSparse拒绝），可拷贝其独立逐页 oracle 与生成逻辑。检查两个 sender 传入 fake engine 的三个list与 oracle逐项一致；空描述符时真实consumer不调用engine；合法非空时调用一次。额外检查 engine 返回非零状态（例如7）透传且调用一次。不要把 envelope/NPU/完整取消说成已测。
5. 固定4个计时Case及每样本迭代数沿用 L004：短连续N8/R1/128次，连续N4096/R1/16次，连续N32768/R1/4次，碎片N4096/R4096/1次。A/B工作量一致。每个Case各3轮预热，5组A/A，然后9组交错AB/BA。先保存原始wall与thread CPU ns/调用，计时窗口一致；不要删异常值，不改变GC默认启用状态。每个样本的 `gc.get_stats()` collection计数差可在计时窗口外读取，作为附带信息。
6. 干净计时阶段不要启用 `gc.callbacks`，不要使用tracemalloc，也不要包装group/consumer计时，避免改变测量口径。计算wall与thread CPU各自的中位数/IQR/范围、配对几何平均A/B及4000次固定seed bootstrap 95%区间，保留所有原始样本和AB顺序。可给出描述性数字，不替主控接受优化。
7. 计时后另做分段诊断，不混入主样本：每个Case/变体各3次单调用。通过 env 中 group/range wrapper与真实consumer wrapper测 group、consumer、total 的wall与thread CPU；记录 sender residual=total-group-consumer。绑定consumer原方法不改其body。诊断阶段才启用一个 gc.callbacks 观察器，保存start/stop、generation、wall/thread时间和sample id。用try/finally移除仅本脚本注册的callback并恢复原env/binding。明确回调/包装有额外成本，这些数值是诊断。
8. fake engine 不做真实native类型转换或复制。真实 `_transfer_data` 的 zip/list必须实际执行。保存 native call count、descriptor count、payload bytes，保证每变体每Case相同。不要把consumer no-op时间当网络时间。
9. 保存 `result.json`、`receipt.json`，可保存提取方法快照；receipt至少有脚本、输入、结果 SHA256、exit_code、correctness、scope。失败应保留错误receipt并非零退出，禁止自行修复/重跑。成功stdout仅小JSON摘要，不打印所有样本。
10. result必须声明：本机CPU fixture，no-op engine，无NPU/模型/网络/PD基线；原生产源码未改；本任务整体serving证据仍E0。记录Python/NumPy/platform、GC阈值与启用状态。

返回只需：创建的文件、设计要点、未执行声明及主控需要检查的疑问。主控会独立审核、冻结哈希，然后另行授权执行。
