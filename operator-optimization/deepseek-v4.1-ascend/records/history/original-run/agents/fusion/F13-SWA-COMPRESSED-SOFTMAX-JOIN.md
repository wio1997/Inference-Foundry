# F13：SWA 与压缩 KV 的注意力合流能否避开 QLI 等待

只读检查现有源码、公开昇腾原始接口资料和一份历史 trace；未修改目标、不运行 NPU。这里讨论一次 attention 内**窗口部分和压缩部分能否拆成独立计算后合流**，不是重新审 F12 的 QLI 输出排序或 F14 的 inverse RoPE。

## 已有复合算子与具体来源

`attention/dsa_v41.py:613–680,704–716` 的 A3 常规分支先通过 `_select_sparse_indices` 得到 compressed indices，再等待 attention metadata，调用一次已有复合 `npu_sparse_flash_mla`。它同时接收 Q、该层 SWA 的 `ori_kv`、源层 long-KV 的 `cmp_kv`、各自 block table、各自的有效长度/掩码/窗口、`cmp_residual_kv`、压缩索引、FP32 `sinks`，并设置 `return_softmax_lse=False`。调用只使用首输出，之后才原地 inverse RoPE。packed backend 走 `backend.qsmla`，需另行研究。

[昇腾公开 sparse_flash_mla 接口文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-transformer/sparse_flash_mla.md) 描述 SWA 单用原始 KV、CSA 联合原始窗口与按索引选择的压缩 KV，注意力使用共享 KV；当返回 LSE 开关为 false，第二输出是占位。**公开页面属于 CANN Community 文档版本，不能视为 169 镜像中 `torch.ops._C_ascend` 二进制合同**：例如该公开版本声明 `cmp_ratio` 支持 1、4、128，而冻结代码对压缩支路使用 1 或 2，运行时真实算子版本、行为、精度和支持范围须以所装实现确认。不能凭这份文档认定 ratio2 可用或不可用，亦不能仅因 trace 形状中出现 FLOAT 占位就认定有可合并 LSE。

`model.py:519–547,714–745` 和 config：source 层 2/8/14/20 拥有 long-KV；2→2..7、8→8..13、14→14..19、20→20..39 是可能消费这些 cache 的静态角色。层 24 是 index source、candidate-filter，使用层 20 的 long-KV / index cache；层 25–27 可以复用层 24 发布的 topk，后续 index source 再发布新选择。运行时空 cache、不同 request 可见长度、角色及对象身份均未由这些配置直接绑定。非 index source 层未必每层都运行 QLI，不能将每层都写成等待一次 QLI。

## 一个可证的实数分块关系

作为候选，**假设**目标 169 的具体 SMLA 在某一 query/head 上，确实把原窗口和所选压缩位置作为同一个 softmax 的参与项，两个部分的 score、mask 和共享 KV 的读取可各自按原语义完成：对分支 A/B 保存 `m=max(score)`、`z=Σexp(score-m)`、`n=Σexp(score-m)*value`，可按

`m=max(mA,mB)`；`z=zA*exp(mA-m)+zB*exp(mB-m)`；`n=nA*exp(mA-m)+nB*exp(mB-m)`；最终 `out=n/z`。

这是实数代数恒等，空分支需单独处理 `z=0` 等边界；对 NaN/Inf 和全部 mask 的情形不能直接套用 `-inf - -inf`。`sinks` 在真实算子里的具体贡献必须从对应运行版本契约确定，重组时原贡献应精确出现一次；不能对每个分支都各放一个 sink 后直接相加。任何重复位置、mask、索引排序与遍历顺序必须保留原算子语义，不能擅自按物理位置去重。`cmp_residual_kv` 和位置、block table 影响压缩部分可见域，不能用静态 shape 代替。

简单反例：忽略 sink，窗口部分一项 score=0、value=0，压缩部分一项 score=ln(3)、value=4。分别 softmax 得到 0 和 4，平均是 2；共同 softmax 是 `(1×0+3×4)/(1+3)=3`。所以把两路结果直接相加或平均是错误的，必须保留归一化权重或其他等价的充分状态。例子只说明一般 softmax，**不是本模型测量值**。

**宽实现域的反例与数值边界（2026-10-30）：**忽略 sink，假设两项均有效、score 为有限值且采用精确实数 softmax，两个 BF16 value 都是 2，则 `(exp(s₀)×2+exp(s₁)×2)/(exp(s₀)+exp(s₁))=2`。在该值恒等对整个允许输入域已经证明、且目标数值契约、cache 更新与其他副作用都允许时，可复用这个结果；不能由一般 attention 公式断定每次都须独立计算两路 score、分别产生两路状态或等待两个具体任务。反过来取 `s₀=s₁=0, v₀=2, v₁=4`，联合值为 3，而只使用第一项为 2，说明对一般输入不能未经证明直接丢弃第二分支。纯 CPU 复算 `python3 -c 'import math; print((math.exp(0)*2+math.exp(1)*2)/(math.exp(0)+math.exp(1)),(2+4)/2)'`，输出 `2.0 3.0`。

这个反例只在**明确的无 sink、非空、有限输入、精确实数或经证等价的舍入契约**中成立；真实 SMLA 的 sink 可能改变归一化分母或输出，mask、不同 cache、重复索引、NaN/Inf、有限精度 exp/累加、BF16 舍入、缓存写入和其他观察者也可能使简化不等价。尚未证明本次 169 的 value 恒等、实际使用无 sink、或设备计算逐位符合该模型，故不据此声称该服务能省某分支或获得提速。证明某个真实分支的信息不可省略，同样需要固定实际权重、可达状态、初态与可观察输出；shape 和现有调用顺序不够。

## 调度和接口边界

- 如果完整的 Q、窗口 KV/metadata、slot 写入和其 stream event 已就绪，可以探索先计算 SWA 分支，同时 QLI 独立选择压缩位置。对这个**显式分两路计算并按前述公式合流的候选**，使用压缩分支时须取得对应索引、long-KV 和 metadata，合流须获得有影响的两路充分状态及原 sink 效果；这些不构成所有允许缓存、预计算、重算或等价融合的实现都必须等待同样任务的证明。是否真能重叠还要验证 Q 的地址、metadata 并发访问、cache 别名、两个算子对 Cube/Vector/MTE 的争用和 graph buffer 生命周期。
- 当前 Python 调用一次完整的 SMLA，`return_softmax_lse=False`，**没有**向调用方发布上述每分支的 `(m,z,n)`，也没有看到该调用能中途接入另一路或部分输出。公开接口能在某些设置下返回整次 LSE，并不证明可对当前 packed/ratio/图模式单独运行两路、正确处理 sink 或返回未舍入 numerator；更不证明具备逐 tile event。要实施该候选可能需要新算子或内部接口契约；不能称之为现有开关。
- 实数等价不保证逐位等价：分块后 max/exp/累加顺序、累加精度、概率处理和输出 BF16 舍入都可能变化。要保留用户要求的数值语义，并验证空分支、全 mask、非有限值、duplicate/sentinel、source 层复用和新旧 cache 页版本。对于源层的 index key 和 long cache，必须满足 F19 中各自发布及 SMLA 读取前的 happens-before；topk 的 graph-stable buffer 必须在所有复用层读取期间不被覆盖。
- 源码路径将结果交给 F14 的 inverse RoPE 和输出投影；对于一般输入，不能仅凭局部 SWA 输出、单分支 BF16 结果或未经全局归一化的部分结果就认定得到完整 attention 输出。若另证某特定可达域或缓存状态下部分结果与完整可观察结果等价，则可在保持舍入、sink 和所有副作用的条件下简化。允许其他算子的条件性早启动，也不单独证明该行最终值已可发布。

## 既有单窗口的证据与限制

`F12-LAYER24-RUN2-REVIEW.md` 与 `f12-layer24-run2-evidence.json`：EP8、配置角色层24、run2 窗口中，stream89 的 QLI 记录 193.216–691.707 µs，随后 index helper 692.527–705.606 µs，SMLA 706.886–751.985 µs；SMLA 输入 shape 中同时有两个 BF16 paged KV 和 `[3,1,512]` INT32 索引，与源码所描述双分支调用相容。它不提供有效长度、物理内容、tensor identity、调用栈、独立 SWA 部分时长或关键路径；不能以 498.491 µs 的 QLI 时间或 45.099 µs 的 SMLA 时间计算可隐藏时间、收益或性能上界。针对非 source 层和其他请求更不能照搬。没有新增性能采集或无目的测试。

## 分支输出已舍入时，即使 LSE 精确也不能保证位级合流

F13 的实数分支合流还需区分**高精度充分状态**与**已经舍入的普通 attention 输出加 LSE**。以下是按实数权重和 BF16 round-to-nearest-even 构造的反例，三个 value 都可精确表示为 BF16：A 分支两项 value 分别为 `1`、`1+1/128`，未归一化权重分别为 `3`、`1`；B 分支一项 value 为 `1+1/128`，未归一化权重为 `4`。A、B 总权重各为 4，所以即使提供完全准确的 LSE，最终两分支比例仍是 1:1。

- 联合 softmax 的实数输出为 `1+5/1024`，最后一次 BF16 舍入为 `1+1/128`。
- A 分支实数输出为 `1+1/512`，先舍入为 BF16 得 `1`；B 的 BF16 输出为 `1+1/128`。按精确 LSE 权重合并后得到 `1+1/256`，正好为中点，按最近偶数舍入为 `1`。

因此仅用已经舍入为 BF16 的两路输出和即使精确的 LSE 进行合并，无法保证结果与联合 attention 后仅舍入一次逐位一致。实际要验证该方案须先确认历史分支输出/累加 dtype、内部状态、LSE 定义、sink 与具体误差契约；若应用允许数值误差而非逐位一致，也需以该契约判定，而不能据此说所有这种合流都不合法。这是数学反例，不证明上述权重或输入出现在该 workload，也不模拟目标设备实现或计算速度。该证据与现有 F13 属于同一候选，不增加可独立计收益的项目。

可复算：

```python
from fractions import Fraction as F

def bf16_near_one(v):  # 本例均在 [1, 2) 内
    q = (v - 1) * 128
    n = q.numerator // q.denominator
    f = q - n
    if f > F(1, 2) or (f == F(1, 2) and n % 2):
        n += 1
    return 1 + F(n, 128)

a = (F(3) + (1 + F(1, 128))) / 4
b = 1 + F(1, 128)
assert bf16_near_one((a + b) / 2) == F(129, 128)
assert bf16_near_one((bf16_near_one(a) + bf16_near_one(b)) / 2) == 1
```

## 压缩路径中，SMLA 元数据和 SWA 索引来自不同字段（2026-10-30）

冻结 `attention/dsa_v41.py` SHA-256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`。源码 `_get_layer_metadata:272–281` 根据不同 prefix 分别查找 `swa` 和 `long_kv`，将它们放入 `DeepseekV41LayerMetadata` 的 `swa` 和 `attention` 字段。标准非 packed `_forward_attention:635–656` 在 `compress_ratio in (1,2)` 时选择 `metadata.attention.smla_metadata` 作为 operator metadata，并按其对象 ID 调用 `wait_for_device_metadata(ATTENTION, ...)`；`:657–675` 的同一次调用则从 `metadata.swa` 提供 `ori_sparse_indices`、`ori_topk_length`、block table 和窗口参数，从 `metadata.attention` 提供压缩部分的 block table、长度及 residual。对于无压缩分支，选中的 operator metadata 是 `metadata.swa.smla_metadata`；packed `backend.qsmla` 走另一条路径，不能套用上述结论。

`_publish_task:879–900` 可以按条件把构造任务登记为 `DeviceMetadataTask`，其他情况立即执行构造闭包；`build_smla_metadata:1161–1208` 把其所属 builder 的 `ori_topk_length` 传给 metadata 算子。由此，探索提前拆分 SWA 或重排元数据构建时，需要分别追踪**所需 SWA 索引/长度的生产与就绪**以及所选 operator metadata 的生产与就绪，并核对 metadata 生成所使用的描述与最终调用参数满足目标算子的实际接口合同；不能仅凭两个字段同形、相近时间或存在一个按 operator metadata ID 的等待，就证明全部 SWA 数据已经就绪，或证明两个 builder 之间的张量是同一个。

这只是源码中字段来源和显式等待位置的限制：SWA 侧也可能通过同流顺序、之前的事件、其他等待或已有调度保证就绪。以上检查未穷尽 executor 的事件边，也未绑定历史实例、storage、ABI 和该等待的具体事件实现；**既不能据此认定缺失同步、数据竞争或线上缺陷，也不能推断潜在等待时间和提速**。现有 F13 仍是同一项条件性结构重组，不另计融合收益。

### 共享键的额外条件

在同一冻结源码 `dsa_v41.py:955–960,1119–1121,1161–1208` 的非 packed、启用 query metadata 构建路径中，SWA builder 的 `operator_ratio` 固定为 0，压缩 long-KV builder 为其压缩比 1 或 2；对应传给 `_publish_task` 的键分别为 `smla:c0` 与 `smla:c1` 或 `smla:c2`。因此，即使两次调用确实收到同一个 `batch_shared` 字典，按该键查找也不会因为已有 `smla:c0` 条目而直接命中 `smla:c1/c2` 条目。反过来，每次任务不一定都重新构造：字典中可能已存在各自的键，`_publish_task` 会直接返回对应对象。

该结论是键级源码推导，不证明两键的值一定是不同 tensor、不同 storage 或不同设备任务：其他调用可以预先填入映射，同一个 builder 的持久 buffer 也可能被复用；是否真的共享字典、各次构建顺序、event 和生命周期在历史样本中未绑定。不能将键不同当作缺同步或额外耗时的实测。对提出的跨模块重排，需按各自键和真实对象追踪缓存命中、发布及最终消费者，不能只按字段名或压缩比推断自动共享。

## 分块重叠的条件关键路径（2026-10-30）

为避免把当前完整 QLI 任务的结束时刻当成不可避免的全局屏障，定义一个**假定的分块模型**。时间起点为指定 query/head 的模型窗口起点；`a` 是窗口分支对最终输出必需的信息最早就绪时间的已证下界；每个压缩分块 `i` 的必要索引/内容最早就绪时间下界为 `r_i`，自该信息就绪后到该分块对最终输出的必要贡献就绪，另有已证的顺序时长下界 `b_i`。若所定义输入确实不可提前、该贡献不能被更早的等价预计算消除，且最终输出依赖窗口分支与全部这些贡献，则

`T >= max(a, max_i(r_i + b_i))`。

证明只使用逐分块的数据依赖：每个必需贡献无法早于 `r_i+b_i`，窗口侧无法早于 `a`，最终值必须等待它们，因此取最大值。这个式子没有假设不同分块串行，也没有将通信、计算或不同分支时长相加。最后的合流可能被融合到某个分块工作中，所以没有额外强行加一个合流时长。若对同一资源另能证明全部合法算法在该窗口内至少消费 `W` 单位工作，资源严格服务包络为 `β+P·T`，还可取 `T >= max(原式, (W-β)_+/P)`；`W` 不能直接用当前算法的总操作数代替，`β` 是窗口初始在途/突发信用，`P` 是对应资源的严格容量。两者适用对象和时钟窗口必须一致，资源共享时不能在没有证明时把多类工作分别按不同满速容量同时计算。

一个纯示意例：两块 `r=[2,5]`，各 `b=2`，窗口分支 `a=7`，单位任意；上式给出 7。若有独立资源，两个分块可分别在 [2,4]、[5,7] 完成，窗口分支在 7 完成，全部结果在 7 合流。把所有分块都等到 `r=5` 后才开始、再把每块 2 单位工作串行相加，会得到 9 或更大的时间，但这是附加调度限制，不能作为允许分块、重排和重叠的通用下界。该例没有使用任何 169 实测值。

重要限制：现有历史 QLI 记录的整任务时长并不是上述 `r_i` 的已证明下界；如果可提前计算候选 score、重算或用别的精确等价表示，索引最终发布后仍必做的 `b_i` 可能为零。窗口工作也可能与压缩工作融合或重用，不能未经证明把两者的独立工作量求和。最终输出对每个分块的真实依赖、逐块就绪能力、实际内核精度和容量/信用都未闭合。因此这里给出可审计的**条件依赖框架**，没有导出当前设备、整模型或 token 的数值性能上界，也不证明候选可实施或带来收益。
