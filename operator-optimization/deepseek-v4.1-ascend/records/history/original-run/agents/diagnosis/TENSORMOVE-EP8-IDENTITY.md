# AIV 显示比例最低组：固定 TensorMove 的实际调用

选择已有严格可比排名里的 digest `1b826883b782ba77814d5bd40fcd3e46c759e380069db2e066ff0e9cb4bce89d`、层 0、EP8 的一条，回查已有 trace；只读、不启动 NPU 任务。该组的物理签名是 BF16 `[8184,5120]` 输入输出、ND 格式，8 EP 各 1 任务，均为 `AI_VECTOR_CORE`，显示的 `aiv_vec_ratio` 均为 0.0。组筛选、其他卡、周期条件与边界见 [RECAPTURE-ALL-KERNEL-RANKING.md](RECAPTURE-ALL-KERNEL-RANKING.md)。

## 绑定证据

EP8 的选中记录：flow `1793241328465293999`，主机起点 `1793241327169187.740 µs`、线程 `(737,737)`；设备终点 `1793241328465293.999 µs`、track `(755144,47)`。完整 trace SHA-256 `53ec20cf6d8b8f79b67a7f2f4ddaabf823b4c0305e88e2dbf9cf62d02e7c6f23` 经脚本逐字节复核。该 flow 有且只有一个 `s` 和一个 `f`；在设备终点、track 上唯一的 X 是 `aclnnInplaceCopy_TensorMoveAiCore_TensorMove`，task 43441、stream 47、core `AI_VECTOR_CORE`，trace 时长 37.0995 µs；此前 CSV 匹配为 37.100 µs、48 blocks、`aiv_total_cycles=2903326`、`aiv_vec_ratio=0.0`。两者为同一次采集的关联记录，并非独立实验。

相同主机线程中，flow 起点与唯一 `aclnnInplaceCopy` CPU 事件起点完全相同，该事件包含于 `aten::copy_`，后者记录的两个输入均为 BF16 `[8184,5120]`。两者调用栈的直接定位为 `attention/context_parallel/dsa_cp.py:1798 _forward_o_proj`，外层为 `dsa_v41_cp.py:224 _project_output` 与 `dsa_v41.py:716 forward`；它们同处 `vllm::dsa_v41_forward` 和相应 decoder layer CPU scope。**这是被跟踪到 attention 输出投影路径的单个 copy 任务，不需要根据算子名字猜测归属 Engram。** CPU 区间为主机包围范围，不是设备时长。

冻结副本 [dsa_cp.py](../../sources/vllm-workspace/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py) SHA-256 `7e8dd93eeda988866138342523ecfa7cc9c42ce67669d4b1507ee8c5f3452305` 第 1795–1799 行：先 `_apply_wo_b` 得到 `projected_output`，若 `output` 不为 None 执行 `output[...] = projected_output`。文件、行号、调用栈和 BF16 形状一致，支持该样本对应输出缓冲写入的解释；但并未在历史时刻绑定载入文件哈希与运行时张量指针。不能以此确认所有相同名字任务属于同一调用、断言实际缓存/存储别名，或宣称替换操作可保持语义。

每个逻辑 BF16 张量为 83,804,160 字节；这是形状字节数，不是 HBM 实际读写量，更不是可省去的流量。`aiv_vec_ratio=0.0` 是导出的向量指标显示值，不能证明设备闲置或任何数据搬运未发生；该字段同版本精确定义/舍入和不同硬件管线活动仍需证据，不能仅凭比例认定计算或内存瓶颈。

## 复算

脚本 [tensormove_ep8_trace.py](tensormove_ep8_trace.py) SHA-256 `669dae01136645e34d1778779980a8b8f7ea1110c45a3815b45c6c2bb51395fa` 固定文件与 hash，以流式方式遍历 786,948 个事件，输出 [tensormove-ep8-trace.json](tensormove-ep8-trace.json) SHA-256 `76791983ccb058e483cedd80d7be0057c91f4abe99d7d83a92940d5f3dcc5367`。输出含内部路径和时间，仅保存在任务目录，不应直接公开；发布时应只摘录脱敏结论。flow 起点与事件的精确时间一致及单个设备 X 已在本机复核。脚本输出的主机范围包含关系是时间与线程关联，不单凭它证明张量指针身份或全部因果。
