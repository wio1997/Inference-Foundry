# F26：末层残差复制、collapse 与 SP gather 的分块关系

## 既有审查与本次增量

`FINDINGS.md` 已登记 collapse→norm、MTP residual 双消费者及 SP/MTP 四分支；`CONFIGURATION-REVIEW.md` 已说明 DSpark+PP default 和 SP 只可作条件推导。本次只分析 **SP=true 且 MTP buffer 存在** 时 gather/collapse 的可交换条件、两个消费者的潜在并行以及变更通信域的后果，不将已登记的 collapse+norm 再作为新成果。

## 具体调用和不可删除的消费者

`model.py:1592–1607` 在 SP+MTP 条件下，依次对末层 `hidden_states` 和 `pre_mix` 分别 `sp_all_gather` 并按 `full_num_tokens` 截断，将完整 `[T,4,5120]` residual flatten 复制到 `_mtp_hidden_buffer[:T]`，然后用最后 decoder 的 `hc_collapse` 混合并做最后 RMSNorm；`get_mtp_target_hidden_states()` 将该 buffer 暴露到外部。collapse 按 `model.py:854–856` 用 FP32 `pre_mix * x.float()`，沿 4 路求和再转回 x 的 dtype。此 buffer 是完整 HC residual 的可见消费者，不等同于 collapse 后的 `[T,5120]` 值。

`evidence/frozen-serve.sh` 请求 DSpark K=5 和 TP8/DP2；`frozen-configuration-evidence.json` 指出在声明的默认 PP=1、条件满足且无覆盖时可推导 MTP buffer 存在，SP 属性也可在相应 backend/patch 条件下推导为 true。它们均非已读取的历史 resolved 对象，不能仅据此断言该请求实走这四分支之一。

## 新增：交换 gather 与 collapse 的条件、以及数据量

设本 rank 的局部残差 R 是 `[L,4,5120]`、局部 mix M 是 `[L,4]`，原路径是 `gather(R)`、`gather(M)` 后按每行 collapse，MTP 复制 full R。因为源码 collapse 的算式只跨同一 token 的 HC 轴、不跨 token，**若**两次 gather 对 token 的排序/裁剪完全一致，且原数值算法、dtype 和舍入对分片/完整 shape 一致，则 `collapse(gather(R),gather(M))` 与 `gather(collapse(R,M))` 在对应有效行上可相同。当前本地快照未提供 SP all-gather 实现的全部通信顺序/异步契约，也未证明多 shape 下 reduction lowering 完全相同，这不是无条件位相等证明。

可研究的替代路径是本地先 collapse、各 rank gather collapse 后 y，同时**另行保留** full residual 的 gather 与 MTP 复制；不能因为 collapse 提前就取消 MTP full residual。若用配置中的 BF16 residual/y 和 FP32 mix 计算每个有效行的原始逻辑数据体量：

- 当前需要 gather 的 mix 是 4×4 = **16 B/行**；替代的 collapsed y 是 5120×2 = **10,240 B/行**，比例 640:1。
- 两条路径还都需形成被 MTP 使用的 full residual，每行 4×5120×2 = **40,960 B/行**；若只替换 mix gather 为 y gather，逻辑行载荷由 40,976 B 增为 51,200 B，差额 10,224 B/行。

这是假定稠密且不压缩、每元素只按一次逻辑行载荷计算，并且 BF16/FP32 类型按配置/接口成立的**条件字节会计**；不是链路实际传输字节、必要通信下界或耗时。TP group、每 rank 所有远端行数、padding、collective 算法、复用和压缩会改变物理字节。此例说明“把小系数 gather 换成大 hidden gather”可能增量明显，不能以算式可交换就宣称变快。

另一条候选是保持原 gather 数据域，并在 full residual 与 mix 的对应块已可读后，**并行**执行 residual→MTP copy 与该块的 collapse；这两个源码消费者都读取 residual，并分别写 MTP buffer 和 collapsed output。成立条件是源/目标存储不意外 alias、缓冲在两消费者完成前不被覆盖、copy 正确保留全部 HC 值、collective 提供被安全消费的块与跨流事件；源码目前只显示两次完整 gather 后同程序顺序调用，不证明逐块 ready、并行实现或性能。MTP buffer 的外部读取还必须在正确 copy 完成后发生；模型返回 normalized hidden 的可见结果不能删。

collapse 与 norm 的后续融合仍要先完成各 token 4 路 FP32 混合并按原语义舍入回输入 dtype，norm 归约依赖该 token 完整 5120 特征、epsilon/gamma 和实际顺序；再接 LM head 要保留 logits processor 的切片/词表与其他返回消费者。并行的 copy 和 collapse 都可能争用访存与通信资源，理论独立不代表免费隐藏。

## 现有采样范围与恢复

已冻结的 layer39 窗口在末层 HcPost 结束时截止，因此其内部任务统计不自动覆盖后续 final norm / logits；`evidence/TRACE-SUMMARY.json` 中的全段 Mul、RmsNorm 和 allGather 可能来自其他层/请求，不能把次数或时长直接归属于 F26。只读 `query_f26_tail_window.py` / `run_f26_tail_window.py` 固定 EP8 layer39 run2 末标记后 500 µs、只列有限算子名并校验输入 SHA；**邻近时间不证明同一次 target tail 或 tensor 的关联**。结果见下节，缺少直接绑定时分支、事件和收益继续 UNKNOWN。无目标代码修改、无 NPU 操作。

## stage171：末层标记后有限范围的观察

stage171 已完成上面的只读查询；独立核验 `/Users/wio/work/173-sglang-matched-172-benchmark/operator-ceiling169-v2-171-f26-tail.exit-code` 为 0、`.verified.json` 的 `model_verified=true` 且模型为 `deepseek-flash`。结果 `f26-tail-window-evidence.json` SHA256 `2bb2dd756e0c201731b5caacd89a5b0cbb1c25f0ad47abf2222266bea9fe61f0`，脚本 SHA256 `c77f7b3a6bdecb1c20ed4c1897bf6ab004cc9a20db01bde16713799c8d2f08e5`，EP8 CSV SHA256 `488d3e3736677d7d041edbe8307d3c06b79553362e54aef07aba37fd74229cc7` 与清单一致。无目标代码修改、无 NPU 操作。

以下是 EP8 target layer39 run2 已归属 marker 结束之后、选定 500 µs 内的相对时间；只查有限名称：

| 观察任务 | stream/model | 起止 µs | 物理输入 → 输出 |
|---|---|---:|---|
| 两段 `hcom_allGather_` 记录 | 88/49 | 0.640–17.000；18.260–29.839 | shape N/A；各另有一条同区间 stream N/A 记录 |
| Cast | 89/49 | 39.039–47.119 | `[24,4,5120]` BF16 → FP32 |
| Mul | 89/49 | 47.139–62.679 | `[24,4,1]`、`[24,4,5120]` FP32 → `[24,4,5120]` FP32 |
| ReduceSum | 89/49 | 63.719–75.579 | `[24,4,5120]` FP32 → `[24,5120]` FP32 |
| Cast → RmsNorm | 89/49 | 76.379–93.478 | `[24,5120]` FP32 → BF16，再输入 BF16 `[24,5120]` norm |
| MatMulV2 | 47/4294967295 | 201.476–348.873 | `[24,5120];[16160,5120]` BF16 → `[24,16160]` BF16 |

前五类任务的形状、dtype 及局部顺序与 `gather→collapse→norm` 源码相容；这个窗口比全段汇总更有针对性，但**邻近 layer39 marker、model/stream、shape 不证明它们是同一次 target tail、MTP 分支、同一 tensor 或同一请求**。两段 allGather 各有一条同区间 N/A 汇总样式记录，不能把每对计为两个独立通信，也不能在没有事件映射时断言一条 gather 是 residual、另一条是 mix。后续 MatMul 的 model/stream 与前面的不同；16160 恰等于配置 vocab 129280/TP8，这只是形状相容，不能证明它为 LM head 或消费该 norm 结果。列表未显示 MTP copy 不能证明其不存在：查询限定名称，可能有其它类型/融合/时段；未观测具体值、storage、event 或输出消费者。当前只可将原候选约束与条件字节会计关联到相容的历史形状，**无法推断实际执行分支或性能收益**。

## target 最终 norm → LM head 的条件代数重组与偏置约束

此节只审 target 的最后归一化与词表投影，不重复 F30 对 DSpark Markov bias / 词表采样的分析。冻结 `model.py:1592–1608,1626–1636,1778–1783` 在 collapse（必要时 SP gather）后执行 `self.norm` 并返回 hidden，另由 `compute_logits` 把 hidden 和 `ParallelLMHead` 交给 `LogitsProcessor`。因此 norm 结果有模型 forward 的可见返回用途；即使某种投影重组无需该中间结果，也不能未经验证删除其生成、改变 aux hidden 返回或 `_mtp_hidden_buffer` 的时序/内容。源码没有证明这两个调用的具体运行实例、间接消费者或 logit 处理规则。

仅在**理想实数、RMSNorm 无加法 bias、无投影前中间舍入**的简化模型下，对每 token 令 `α=rsqrt(mean(x²)+ε)`，norm 权重为 `γ`，LM head 矩阵为 `W`，则 `W(α(x⊙γ)) = α((W diag(γ))x)`。这使另一种条件结构在数学上可提出：collapse 得到行 x 后，分支 A 求整行平方统计/α 并生成原 forward 要求的 normalized hidden；分支 B 用预处理权重或等价计算投影 x，二者在输出 logits 前会合。与现有 collapse→norm 本地融合不同，这是跨 **norm、权重投影和 logits 生成** 的重排；若实际接口仍要求完整 norm hidden 后才可调用 logits，这个方案需要新的数据/事件契约，并非现有可启用开关。

当前 Ascend `ops/layernorm.py:49–102` 还存在条件性 bias：quant description 含匹配 `norm.bias` 的键时构造 bias；若该特定对象实际 `bias_loaded`，无 residual 路径在 `npu_rms_norm` 后执行 `x.add_(self.bias)`。所以必须先确认最终 norm 的实例、权重及该布尔值。若规范化输出在理想实数中是 `α(x⊙γ)+b`，则 logits 包含 `α W(x⊙γ)+Wb`：不能把 `α` 移过或施加到 `Wb` 上；是否存在实际 bias 和其数值尚未绑定。当前 layernorm 快照 SHA-256 `d2fda42ad99935725c98b9505a8d2e94696d9cdb2906e5cb2f1e37bf21fb27bc`，与历史执行文件及 norm 实例未绑定。

即便忽略 bias，上述等式也不证明浮点结果等价：原路径的 collapse 已转回输入 dtype，norm 输出及可能的 bias 加法还会有实际 kernel 的 dtype/舍入；先把 γ 融入 W 或先算投影会改变乘加与归约顺序、权重量化和中间舍入。需要与实际运行的 norm/LM head 后端、归约及容许误差契约逐项比对；不能将实数恒等升级为逐位等价。最终共享标量 α 必须在完整 hidden 维的 RMS 归约完成后可用，且 token、padding、分片与 logits 对齐；即使数学路径有独立支路，也可能因 Cube/Vector、内存带宽与流事件争用抵消重叠。

仅仅因为**无 bias、有限正 α**不会改变理想实数下未经任何后处理 logits 的 argmax，也不能省掉 α 后声称推理等价：softmax 概率通常改变，温度、top-p、随机采样及 speculative rejection 等依赖分布；一旦后续存在非等比项，winner 也可能变化。例如投影 `q=[2,1]`、`α=1/2` 且随后加 `b=[0,3/4]` 时，正确 logits 为 `[1,1.25]`，省掉 α 得 `[2,1.75]`，两者赢家相反。这里只是代数反例，不证明本次 workload 中存在该 bias 或采样配置；tie、NaN、溢出及分布式词表映射也必须按真实契约保留。

条件性存储核算：已存目标配置 vocab=129280、hidden=5120；若假设词表恰按 TP8 均分，则每 rank 16160×5120 个权重。单独保存这份 `W diag(γ)` 的稠密 FP32 表为 **315.625 MiB/rank**，BF16 则为 **157.8125 MiB/rank**，不含 padding、元数据和复制；这是假设的逻辑张量大小，而非实际分片、额外显存或 HBM 流量。不同 dtype、量化、共享或就地替换会改变账本与精度。stage171 的 `[24,5120]` norm 和 `[24,5120]×[16160,5120]` 后续矩阵形状与此范围相容，但没有张量/调用身份；不由任务时长估算收益或声称有实际重叠。要进一步收紧需历史 norm bias/weight、归一化及 LM head 实例和图、token/词表消费者、精度要求及同步依赖；缺失则保持 UNKNOWN。本轮没有运行 NPU或修改目标实现。

### norm 中间 BF16 舍入的精确反例

除前节 logits 附加项反例外，即使**无 norm bias、无 logits bias、数学上保留 α**，把 α 移到投影之后也不普遍保持 BF16 数值。下面是可手算的受限模型：`x=[1,7]`、`γ=[1,1]`、`ε=0`，于是 `α=1/5`；某词项权重为 `[1,-1]`。假设 RMSNorm 输出先按 BF16 最近偶数舍入，再以精确点积计算、最终 logit 再按 BF16 最近偶数舍入，则

- 原顺序的 norm 输出分别为 `205/1024`、`179/128`，最终 logit 为 **`-153/128`**。
- 先在实数中算未归一化点积、再乘 α 的重排为 `-6/5`，最终 BF16 logit 为 **`-77/64`**。

两种结果不同；权重、输入均可表示为 BF16。该例是一种精确数值契约下的反例，不声称真实 kernel 累加完全精确、采用 ε=0 或出现这组输入；已存模型配置记载的 `rms_norm_eps=1e-20`，实际配置与历史实例仍需绑定。真实使用允许误差而非逐位相同时，需按指定误差契约评估，不能由此反例宣称所有重排不可行。

可用下列 CPU `Fraction` 计算复核，不涉及设备：

```python
from fractions import Fraction as F

def bf16(x):
    if not x:
        return x
    sign = -1 if x < 0 else 1
    a, e = abs(x), 0
    while a >= 2:
        a /= 2
        e += 1
    while a < 1:
        a *= 2
        e -= 1
    q = a * 128
    n = q.numerator // q.denominator
    rem = q - n
    if rem > F(1, 2) or (rem == F(1, 2) and n % 2):
        n += 1
    return sign * F(n, 128) * F(2) ** e

original = bf16(bf16(F(1, 5)) - bf16(F(7, 5)))
reordered = bf16(-F(6, 5))
assert (original, reordered) == (-F(153, 128), -F(77, 64))
```

stage171 的已存相容 `RmsNorm` task1998 报告两个输出，分别 BF16 `[24,5120]`、FLOAT `[24,1]`；不能据形状断定第二输出的数值是本例的 α，也不能确认此任务为最后 norm。`sampler-patch-snapshot/adapt_patch.py:829–884` 当前静态注册表把 `RMSNorm` 映射为 `AscendRMSNorm`、`ParallelLMHead` 映射为 Ascend 类；这只说明一个源码注册关系，不能证明历史实际注册、对象、底层图或精度。已保存局部源中未找到 `logits_processor.py` 或 `vocab_parallel_embedding.py`，故词表重映射、通信和最终 logits 的完整契约仍未知；不能猜测它们是单纯逐项 GEMM。没有新增 NPU 或目标实现更改。

## EP8 尾部五个设备任务的有限 API 关联审计（2026-10-30）

为收紧上述 F26 的来源，针对所选 EP8 layer39/run2 尾部窗口中同流的 Cast、Mul、ReduceSum、Cast、RmsNorm 五条设备记录，按保存的 model、stream、task、起止时间在同次 `ascend_pytorch_profiler_8.db` 中查询 `TASK`：五条各恰有一条匹配。它们都使用 connectionId 184330，但该 connection 在整张 TASK 表共出现 **19,971 次**；五个 globalTaskId 每个各有 **7 条** TASK。按这个 connection 的直接等值查询没有匹配 `CANN_API` 行，`CONNECTION_IDS` 中也没有对应记录，因此此查询未能通过该连接找到 PyTorch API、调用栈或张量身份。

这是确定范围内的负结果：现有五条任务身份、shape/dtype 与源码 collapse→norm 的关系相容，但不能用高度复用的 connection 或 globalTaskId 将这些任务绑定到某个 host 调用、独立 replay 或同一输入输出存储，亦不能把“未找到”推广到其他关联字段、其他数据文件或全部 API。任务依次位于同一 stream 也不单独证明读取同一中间张量。此前源码提出的 FP32 HC 路归约、输出回 BF16 再做 norm，以及 `_mtp_hidden_buffer` 的完整 residual 消费者，仍是按该源码路径执行时必须保留的条件语义；五条历史记录尚未证明命中该路径、逐块就绪或候选的性能收益。

复现：`python3 agents/fusion/run_f26_tail_api_binding.py`，只在 169 已有 stage 容器上读固定 EP8 DB 与既有聚合 JSON，不启动设备。数据库 SHA-256 `67d14f7f386658f8c8c5d3c9059dff9b0f1decb200eba621a3432fcf1e23de58`、输入聚合 SHA-256 `2bb2dd756e0c201731b5caacd89a5b0cbb1c25f0ad47abf2222266bea9fe61f0`、查询脚本 SHA-256 `4398a456ed5e621120d33e27c5f337087feb914cdfa4dfa3a83740995b0204a1`、运行脚本 SHA-256 `069e8dbb3e6a11b218f9be97d0e063810d1926047e43dc778e6e818aebd58e69`，结果 `f26-tail-api-binding.json` SHA-256 `ac9fe91b0401c0eb01cd1ccec52467fb7d5a274cd3416aa099224db502e4d8f6`。该结果含内部标识和绝对时间，仅用于任务内核对。后续应换其他已有证据源，不能重复同一连接等值查询来补足数据身份。
