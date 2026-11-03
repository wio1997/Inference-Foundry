# 执行日志

- `2026-10-28T14:44:37Z` Loop 已冻结。下一步：检查已执行的只读 Run 并继续追查；此 Loop 根据本轮证据补登记，保留实际执行记录。

- `2026-10-28T14:44:37Z` 为用例 `frozen-169` 创建 Run `run-001`（review）。

- `2026-10-28T14:44:37Z` Run `run-001` 记录为 `pass`；正确性为 `not-applicable`。已核对 stage146：七个边界各有一个同连接 CANN 记录；flow关联有条件，同域调用间隔不证明根因。

- `2026-10-28T14:44:57Z` 为用例 `frozen-169` 创建 Run `run-002`（review）。

- `2026-10-28T14:45:05Z` Run `run-002` 记录为 `pass`；正确性为 `not-applicable`。stage148 只读查询完成；3个CANN同globalTid窗口已有API并集分别49.09、20.17、49.64us，不证明线程空闲或根因。

- `2026-10-28T14:48:38Z` 为用例 `frozen-169` 创建 Run `run-003`（review）。

- `2026-10-28T14:48:38Z` Run `run-003` 记录为 `pass`；正确性为 `not-applicable`。stage149查询完成：三个关联 PyTorch 与 CANN globalTid 不相等，跨表线程及时间域需继续核实。

- `2026-10-28T14:52:19Z` 为用例 `frozen-169` 创建 Run `run-004`（review）。

- `2026-10-28T14:52:28Z` Run `run-004` 记录为 `error`；正确性为 `not-applicable`。stage151 只读查询因 argparse 名称错误退出1，未产生数据；修复后另起stage，保留失败记录。

- `2026-10-28T14:53:37Z` 为用例 `frozen-169` 创建 Run `run-005`（review）。

- `2026-10-28T14:53:37Z` Run `run-005` 记录为 `pass`；正确性为 `not-applicable`。stage152只读查询3个固定锚点；各找到同连接集合的另一线程候选，队列语义与因果需独立审查。

- `2026-10-28T14:56:00Z` 为用例 `frozen-169` 创建 Run `run-006`（review）。

- `2026-10-28T14:56:00Z` Run `run-006` 记录为 `pass`；正确性为 `not-applicable`。读取一个指定的同镜像profiler源码，进一步审查队列时间来源；历史导出版本绑定尚待证实。

- `2026-10-28T15:01:07Z` 为用例 `frozen-169` 创建 Run `run-007`（review）。

- `2026-10-28T15:01:07Z` Run `run-007` 记录为 `pass`；正确性为 `not-applicable`。stage154找到OP_MARK时钟转换源码；固定三组同表enqueue结束到dequeue开始2.610、2.580、2.710us，历史版本/误差及其因果未闭合。

- `2026-10-28T15:05:28Z` 为用例 `frozen-169` 创建 Run `run-008`（review）。

- `2026-10-28T15:05:28Z` Run `run-008` 记录为 `pass`；正确性为 `not-applicable`。stage156查询6个PYTORCH同表邻域；producer 100%覆盖由长外层op造成，不证明线程活动或根因；worker有多组queue记录。

- `2026-10-28T15:09:01Z` 为用例 `frozen-169` 创建 Run `run-009`（review）。

- `2026-10-28T15:09:01Z` Run `run-009` 记录为 `pass`；正确性为 `not-applicable`。stage158三条长外层操作均无callchainId和connectionId，不能归属源码或解释主流缺口，按计划转向DP分组。

- `2026-10-28T15:09:01Z` 主控结论为 `INCONCLUSIVE`。已关联选定queue和主机调用，但固定外层行无调用栈，不能从捕获活动解释主流长缺口；转向可比较层分组。
