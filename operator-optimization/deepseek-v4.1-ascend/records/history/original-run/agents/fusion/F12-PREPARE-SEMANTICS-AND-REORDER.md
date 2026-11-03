# F12 索引整理的语义与重组条件（源码条件分析）

本记录是 `INDEXER-ATTENTION-EVENT-IDENTITY.md` 的增量，不重复把 32 对相邻设备任务当作数据依赖证明。它们有相同 stream 和兼容 shape；未证明实参地址、对象身份或同名任务对应下列冻结源码。`F12-LAYER24-RUN2-REVIEW.md`、`INDEXER-HELPER-REVIEW.md` 已记录相关融合及空输入分支，此处补充精确的变换边界。

## 被读取的范围

- `frozen-source/vllm_ascend/ops/triton/prepare_indexer_indices.py:15–56,59–149`：冻结 helper 内核与调用包装。
- `sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/indexer.py:181–205,208–231,254–297`：所保存源码的非 packed indexer 路径；上面两份源码是否等于 32 个事件当时执行的二进制尚未绑定。
- `evidence/model-config.json:33`：`max_position_embeddings=1048576`；该配置本身并未证明所有运行时 `positions` 实参的取值，约定用例为 32K 输入、1K 输出。

## 行为合同

令输入为行数 R、每行 K 的连续 INT32 `selected`（包装会对输入调用 `.contiguous()`）；一维 `positions` 为 INT32/INT64，长度 R，压缩比 c 为 1 或 2。每行 offset 为 `r*K+j`，其掩码仅允许 `r<R`、`j<K`。对有效行，内核计算 `visible=(positions[r]+1)//c`，取 `0 <= selected[r,j] < visible` 的条目，其余填 INT32_MAX。先将原整数按代码中的位映射作为 FP32 位模式排序，不是数值上的 int→float 转换，最后反映射并把 INT32_MAX 改为 -1。保留重复值；输入 K 非 2 次幂时的补齐列仅参与排序，最终不写入超出 K 的列。若给定 `indices_output`，其必须为 `[R,K]` 连续 INT32 并与 selected 同设备，且写到调用者提供的该地址；否则新分配。可选 `lengths_output` 为 `[R,1]` 连续 INT32。R=0 时提前返回。

存在长度输出且 `R * max(next_power_of_2(K), 32) <= 1024*512` 时，内核直接对原 valid 求和；否则运行后另以 `output>=0` 求和（没有长度输出则不做）。32 个选中窗口的所报告物理行数为 595、643 或 1023，K=512；**如果**这些窗口确实调用此源码，K 和 R 与该 shape 绑定，且给定长度输出，那么三种行数均落在融合写长度阈值内（1023×512=523776 < 524288）；单靠 profiler 输入形状无法证明长度实参或源码绑定。

## 两种重组的等价边界

1. **QLI 最终输出 + 本内核的过滤、按位置排序和共享输出发布融合**：必须保留原 QLI 的 score TopK 选择及 tie、候选 block 副输出，以及此后的可见性规则、逐行排序、重复项、-1、长度写入、地址及跨层消费者寿命。两端 native 接口是否支持指定输出地址/逐行完成通知未知，不能据此估算减少的时延。
2. **过滤移到 score TopK 之前**，一般不与“先选 K，再过滤并填 -1”交换：在一个假设的 K=1 例子中，若得分最高的索引不可见、次高项可见，后过滤返回 -1，预过滤返回次高项；这是数学反例，并非已观测的实际候选。可交换需要额外证据，例如证明 QLI 原本已严格排除所有当前行不可见项，或证明替换方法保留原选中项及无效槽。保存的 `indexer.py:262–270` 已向 native QLI 传入 `seqused_k`、`cmp_residual_k`、`mask_mode=3` 和 `cmp_ratio`，并不能仅凭 Python 参数确认算子内部掩码对当前每个 position 是否完全等同于 helper 谓词；应查该版本 native 实现/合同。
3. **把整理融入 attention 或按完整行/行块流水**：每行进入 attention 前应保持排序结果、重复项与无效槽一致；如果要求位级等价，还要保留 attention 的约简顺序及数值规则。若仍需后续层读取共享 indices，也必须按既有消费者与 graph 地址/可见性发布，且不得在未读完时覆盖。已有设备时间顺序不是数据地址、同步依赖或无内部跨行依赖的证明。

## 源码接口域的特殊值，不等于模型负载发现

令 INT64 position=2147483647、c=1、selected=INT32_MAX，则依源码 valid 判断为真（visible=2147483648），但所选值与 sentinel 相同，编码/排序/反编码后被写为 -1。直接融合长度路径按 valid 计为 1；fallback 对输出 `>=0` 计为 0。这是源码表达式对一个极端接口输入的条件推导：helper 本身未在包装中约束 positions 上限；未证明 Triton 实测、该值是调用方允许输入、或历史执行出现。若位置严格受冻结模型给出的 1,048,576 上限约束，或者是约定 32K 负载中的正常位置，则该反例不可达；不可据此认定当前模型存在 bug。它提示后续讨论“全部接口输入严格等价”与“约定模型负载等价”时必须分别说明位置域。

当前仍缺 QLI 原生行为与输出地址、每行实际 positions、生命周期和设备任务到源码/二进制的可信绑定；无需启动额外 NPU 负载即可继续查已有快照，不能从这些条件论证直接推出实际收益或根因。
