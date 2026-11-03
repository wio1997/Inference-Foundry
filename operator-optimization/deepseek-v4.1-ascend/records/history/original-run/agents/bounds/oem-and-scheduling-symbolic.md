# OEM 规格及固定服务的符号调度上界

## OEM 一手资料

已有 stage 查询显示容器可见的 `sys_vendor=iSoftStone`、`product_name=ChaoqiangA800T A3`。软通华方 [产品目录](https://tongfangpc.com/zh-cn/htmls/chanpin/index.html) 的检索索引有“软通华方超强A800T A3”，与 DMI 型号字符串相近；保存的 `isoftstone-catalog-search.txt` 是搜索索引快照，不是完整技术规格书。2026-10-28 的限定型号检索未找到准确对应 OEM 版本/BOM 的一手 INT8 算力、HBM 和互联技术参数。目录直接读取尝试返回超时；没有使用浏览器绕过。是否为裸金属、该 OEM 与华为 Atlas 800T 的具体配置关系均未核实。

因此 `P_INT8,max`、`B_HBM,max` 和指定互联 cut 的 `B_link,max` 均保留 UNKNOWN；不能因为型号相近而采用其他产品的数字。后续最短资料路径是该机的 OEM 型号版本、芯片 SKU 和 8 卡 BOM 对应的正式规格书，逐项确认数据类型、稠密/稀疏、频率、算子范围、单/双向及资源共享。

## 已识别的服务约束与缺口

冻结启动脚本 `evidence/frozen-serve.sh` 配置 `data-parallel-size=2`、`max-num-seqs=4`，启用 DSpark；工作负载每轮有效返回 `Q=8192` 个 token（8 个请求各 1024）。同一提交号的 [vLLM SchedulerConfig](https://raw.githubusercontent.com/vllm-project/vllm/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/config/scheduler.py) 将 `max_num_seqs` 定义为**每次迭代最多处理的序列数**，不能直接当作一个请求在服务器整个生命周期的并发槽位或时间约束。

同一提交号的 [rejection_sampler.py](https://raw.githubusercontent.com/vllm-project/vllm/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/v1/sample/rejection_sampler.py) 包含 `max_spec_len+1` 宽度的采样输出张量。这提供了某类采样器的候选边界线索；**尚未确认运行容器内的文件内容与公开提交一致、当前调用实际采用这一采样器、预填首 token 和所有输出均经相同路径**。因此本轮不声称实际每次迭代能输出多少 token。两份公开源码摘录和版本 SHA 来源见 `scheduler-fixed-commit-web.txt`、`sampler-fixed-commit-web.txt`；目标容器源码复核已交主控安排。

## 一个不要求事先知道 NPU 峰值的上界模板

先固定以下**需要独立验证**的服务执行域：

1. 有 `E` 个输出服务 engine，engine `d` 的每个提交事件最多处理 `S` 个有效序列，每条序列该事件最多提交 `g` 个有效 token；首 token、缓存/批处理输出与结束 token 都被正确计数，全部 `Q` token 无重复地归属到测量窗内的这些事件。
2. 同一 engine 内相邻输出提交事件的真实完成时刻至少相隔 `δ>0` 秒；这里 `δ` 是**对该执行域所有合法调度均成立的下界**，不是某次 trace 里的平均间隔、最小间隔或测得峰值倒数。
3. `E`、`S`、`g` 和 `δ` 覆盖服务中所有可能的输出路径。允许各 engine 异步运行、不同请求并行与通信计算重叠。

第 1 条给出每次事件最多 `Sg` 个 token。全体 engine 的事件数 `N_total` 因而满足：

`N_total ≥ ceil(Q/(Sg))`。

抽屉原理保证至少有一个 engine 的事件数 `N_max ≥ ceil(N_total/E) = ceil(Q/(E·S·g))`。对该 engine，将事件按真实完成时刻排序，第一个事件不需事先计一段间隔，但后续每对至少相隔 δ；所有这些事件位于测量窗口内，所以

`T ≥ max(0, ceil(Q/(E·S·g)) − 1) · δ`。

当右侧严格为正时，整轮聚合吞吐满足

`TPS = Q/T ≤ Q / ([ceil(Q/(E·S·g)) − 1]·δ)`。

这个推导没有把两个 engine 的时间相加，也不要求两个 engine 的事件同步。它从事件容量和同 engine 的必要间隔推导；若实现允许跨迭代并行、多个 sampler 提交融合或服务端直接复用结果，必须重新证明事件划分与 δ。

把已配置的 `Q=8192、E=2、S=4` **作为待确认的固定服务约束**代入，只得到 `T ≥ max(0, ceil(1024/g)−1)·δ`；`g` 与 `δ` 均未知，所以不产生数值 TPS。不要将更宽理想实现域硬性限制为这套当前调度；这个式子只能描述保留上述提交约束的执行域。

### 不同 engine 的非对称形式

若每个 engine 的严格最小间隔分别为 `δ_0、δ_1`，而有效事件数为整数 `n_0、n_1`，则

`T ≥ max((n_0−1)_+ δ_0, (n_1−1)_+ δ_1)`，

并且 `n_0+n_1 ≥ ceil(Q/(Sg))`。当分配未知时，可以对满足该不等式的非负整数分配取右侧最小值，得到不预设 DP 均衡的保守界；不能直接把两个 DP 的时间累加。该形式同样需要证明每个事件都落在同一端到端测量窗内，并正确处理跨时钟映射。

## 为什么依然没有数值

`g` 的最大值和事件路径尚未通过 baseline/stage 源码一致性、真实 sampler 调用、首 token 及最终输出路径的闭环验证。`δ` 需要证明由哪些不可绕过的语义依赖与物理操作构成，以及这些操作真正的最短延迟；纯粹的当前软件 API 调用顺序、代码中存在的同步或一次 trace 的事件间隔都不能成为跨所有合法方案的下界。没有这些证据就保留符号，不报告实际速度差距或优化收益。

原 `fixed_scheduler_bound.py`、`fixed-scheduler-bound.json` 包含一个附有未验证前提的算术示例，保留历史用途；在真实 sampler 和首 token 路径验证前，不把示例的每组 token 数提升为事实或用于本任务的正式计算。
