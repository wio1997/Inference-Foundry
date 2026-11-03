# F08：KV norm → RoPE → SWA 持久写入

本次针对每层 attention 的 F08 组合做源码级补查。它是**条件候选**，不代表已完成实现、逐位等价或能提速。本地 `captured-layer-attribution/operator_metrics.csv` 的 40 个 target 层窗口均含 `RmsNorm`、`InplacePartialRotaryMul`、`ScatterNdUpdateSk` 和 `SparseFlashMla` 的同名记录；`f08-target-window-summary.json` 给出脚本、输入 SHA 和各层计数。但每层还包含多个同名调用，窗口/名称并不证明它们就是一条 KV 张量依赖或这些次数都对应独立请求。

## 生产与消费

- `model.py:417–425` 把 `wkv` 定义为 `ReplicatedLinear(hidden_size, head_dim)`，`kv_norm` 是按 head_dim 的 RMSNorm。当前 169 checkpoint 与参考量化描述已核对 SHA 一致，描述文件中 40/40 层 `.attn.wkv.weight` 标记 FLOAT；文件标签不是运行实例 dtype、算子内部计算精度或当年加载对象的直接观测。
- 常规 `_project_kv`（`attention/dsa_v41.py:326–330`）顺序是 wkv → kv_norm → view → 在 KV 上原地执行 partial rotary，返回 squeeze 后值；`_write_swa_cache:297–308` 把该值写入所属层的 SWA cache。启动镜像 `_build_info` 为 A3，正常加载 A3 profile 时为标准 non-packed；运行时对象未替换时走 `scatter_cache_sk`。packed 后端是另一个条件分支，不能套用相同缓存布局。
- 标准 `scatter_cache_sk:227–242` 以 `slot_mapping[:values.shape[0]]` 确定目标，先 `values.to(cache.dtype).contiguous()` 再调用 `npu_scatter_nd_update_sk`。源码注明 cache 可能是更大 layer-outermost 分配的 view，必须尊重实际 page stride；`[-1,-1]` 表示跳过。`.to()` 和 `.contiguous()` 可能直接返回原张量，也可能分配/复制，单凭源码不能断言某次 HBM 流量。
- `_forward_attention:613–683` 的标准路径将同一层 `swa_cache_layer.kv_cache[0]` 作为 `ori_kv` 交给 `npu_sparse_flash_mla`，并携带 block table、长度、稀疏索引、mask 与 metadata。这个复合 attention 算子是已存在的融合；不要将其内部 softmax/注意力分解成尚未实现的若干普通算子。SWA 写入还对后续调用和请求的对应缓存状态有持久意义；不能因为当步注意力的一个 tile 不用它就删除 cache 更新。
- CP 的 `_prepare_inputs_and_caches`（`attention/context_parallel/dsa_v41_cp.py:188–192`）允许 query 仅是 rank 本地切片、cache 更新来自更大的 global hidden；`_preprocess:350–355` 在 query 为空时仍调用 `_update_caches`。因此“没有本地 query”不等于“没有 cache 写”。是否在某个实际 profile 窗口出现此情况尚未知。

## 两条现有执行路径和事件

普通分支 `dsa_v41.py:368–373` 显示 `_project_kv` 后写 SWA。多流条件成立时，`440–495` 显式将 KV 与 Q 交错：辅助流等待 `part2_start` 后做 KV matmul 并记 `kv_matmul_done`；主流做 Q norm/quant 后记 `part3_start`，并在 Q_b matmul 前等待 KV matmul；辅助流等 `part3_start` 后做 KV norm、RoPE 和条件 SWA 写入；主流在继续之前 `wait_stream(aux_stream)`，若 `write_cache_on_main` 则 join 后由主流写 cache。具体 A3/non-packed 条件下通常对应 aux 流写缓存，但所选 runtime 分支与各 event 记录尚未逐次观测。

事件既可能确保数据就绪，也构成现有的资源调度次序；只凭“某两次计算数据上无依赖”不能直接删除。新的组合必须保证所有 attention 消费到的指定 cache rows 已就绪，并保留其他 cache 请求和元数据先后。在 per-row/tile 能确切建立完成关系时，可以研究让一部分规范化/旋转/写入与其他独立投影或通信重叠；这不证明当前算子接口提供逐 tile ready，也不保证 overlap 不争用 Cube、Vector 和 MTE。

## 可重组方式及必要约束

1. **norm + partial RoPE 两两组合**：每个 KV 行的整行 RMS 归约需包含该行全部 head_dim；接着保留输入到 norm、归约/epsilon、权重、norm 输出舍入和旋转算术。模型配置 `head_dim=512`、`qk_rope_head_dim=64`，partial 旋转区域为 `[448,512)`，其他 448 维也要原样产生。interleave 每对旋转分量和对应 cos/sin 必须一起可用；不完整 feature tile 不能当作完整 norm 或独立旋转 pair。
2. **norm + RoPE + scatter 三者组合**：当一整行依赖已经满足时，可研究直接将该行经 RoPE 和目标 cache dtype 转换后写到相应合法页/槽；对无效坐标保持跳过，对 page stride、写时序、重复目标 slot、异常/别名和其他可见消费者分别证明。它可能减少中间逻辑读写，但当前没有 tensor 身份与存储层计数，不能声称少读写多少 HBM 或节省一个 kernel 时间。
3. **投影分块接入**：可以从不同 token 行并行开始 Wkv，但某一行的 norm/rotation 需依赖该行对应完整归约信息；若 feature 分块，必须先组合该行所有归约分量，再对完整旋转 pair 和合法目标槽完成写入。融合不允许假定整机所有 token 行都独立：同一物理槽可能重复、页可能共享或回收，冲突处理与可见性须按真实 metadata 证明。

所有候选还需确认原始 operator 的数学精度、shape/量化绑定、缓冲别名、stream 事件和真实每次执行。以上不测卡、不修改推理代码，不以性能排序替代真实瓶颈证据。

## 第二次已绑定采样：同一层范围内的三任务候选

离线复核 `diagnostic-20261029/target-layer-all-kernels-16rank.jsonl`（SHA-256 `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`），只取原导出中已按同线程层范围→唯一 flow→设备记录→CSV 匹配的任务。`check_f08_actual_triples.py` 在每个范围中寻找同一 stream 下 task ID 连续、开始时间递增、输入输出 shape 和 dtype 相容的三个独立任务：BF16 `RmsNorm([T,512])` → `InplacePartialRotaryMul([T,1,1,512], [T,1,1,64], [T,1,1,64])` → BF16 `ScatterNdUpdateSk([P,128,512], [T,2], [T,512])`。采样 1,280/1,280 个范围各有一组：16 个 rank、40 层、四个摘要各覆盖 8 rank；T 为 4,760、8,180 或 5,144。这比旧的按名称同窗计数更具体，但重复 scope 或 graph 记录不应按独立请求计数。

例如 EP8 第 14 层，scope `737-20`，stream 35、task 13274/13275/13276；T=8,180，scatter cache 形状 `[17314,128,512]`。三个 task 的 flow ID 分别是 `1793241329134062817`、`1793241329134098777`、`1793241329134135656`；CSV 数据类型分别为 BF16 norm、BF16 旋转（FLOAT cos/sin）和 BF16 cache+INT32 索引+BF16 值。全部逐项结果与条件保存在 `f08-actual-triples.json`，脚本 SHA-256 `abe6e09c3d95f4fecdbde8c8dd33a239aa6ced1385d2e53e419f4fb5c51f64f9`。冻结 `dsa_v41.py` SHA-256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`：第 327–330 行 `_project_kv` 依次 norm、reshape、partial rotary，第 297–308 行和 227–242 行条件下写 cache。采集任务序列在形状、精度和顺序上与该 F08 分支相容。

这说明**所选记录中确有三个分开的设备任务**，并为 F08 候选提供了跨范围的实际任务证据；仍不能由形状、名字、同流、相邻 ID 确认三者读写同一张量或调用源码分支，不能证明未记录任务不存在、cache 地址或 event 依赖、实际硬件流量及可节省的时间。scatter 为持久状态更新，不能删除；候选只研究在满足全行 RMS 归约、旋转 pair、slot 地址/stride、cache 生命周期、数值和 stream 依赖前提下合并其写入。冻结源码、采样完整性和 profiler 扰动限制维持不变。未启动 NPU，也未修改目标实现。


## 已选三任务的 profile 时间分布

对上节 1,280 组，通过同 scope、EP、flow、stream、task 和精确开始时间，把三任务与固定 SHA 的原提取 JSONL 逐条关联；3,840/3,840 匹配。按导出第一维 T 分组，scatter 是三者中耗时严格最长的组数分别为 T=4760 的 279/320、T=5144 的 319/320、T=8180 的 609/640，共 **1207/1280**。以 T=8180 为例，norm、rotation、scatter 各自的任务时间中位数分别为 **31.540、34.969、258.655 μs**；逐组三任务时间总和中，scatter 所占比例的中位数是 **79.43%**。各任务中位数并不一定来自同一组；该比例是逐组算出再取中位数，**不是总时长或端到端占比**。

这些都是开启 profiler 的 trace task 显示区间；形状、同流和 task 连续并不证明真实 tensor 边或具体 cache 调用，T 不证明实际有效 token/工作量。比较只说明在所选三任务中 scatter 的采样时间常较长，不能据此说 HBM 是瓶颈、计算或缓存写可以消除、三者融合就可节省哪段时间。区间之间还存在其他或未记录工作，不能把 gap 当设备空闲。与整模型 TTFT/TPS 的关联和无 profiler 耗时均未知。

复算：`python3 agents/fusion/f08_profile_timing_summary.py`，只读已保存聚合，结果 `f08-profile-timing-summary.json`；输入 JSONL SHA `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`，triple SHA `e6c18bd23809d47652cc58faf8d5ac81f7879be0e483fc72d70264cd3a3940d5`。未运行 NPU。

## scatter 的状态变化下界：限定域及反例

对于上面 EP8 L14 的**形状相容样本**，`T=8180`、每行 512 个 BF16 值；输入 values 的逻辑容量为 `8180 × 512 × 2 = 8,376,320` 字节，即 `7.98828125 MiB`。可用 `python3 -c 'print(8180*512*2, 8180*512*2/1024**2)'` 复算。这个值是输入 tensor 的形状账，不是 cache 必须写入量，更不是 HBM 流量；采样未提供可绑定的 values/slot/目标存储身份和实际值。

只考虑下列**受限接口和执行域**：输出必须在窗口结束前以精确 BF16 字节形式出现在预先指定、逐字节可观察的 cache 物理目的地址；地址映射和冲突语义固定，窗口前后该地址集没有外部免费修改，且所有改变都由纳入资源账的写机制完成。令 `D` 为这些地址在窗口初态与目标终态之间不同的**字节位置数**，不计其他位置。每个不同位置的值都必须经某次覆盖该地址的写机制改变，因此至少 `D` 个不同目的字节位置受到写覆盖。这是集合包含关系，与融合或是否重算上游无关；不是说发生了 `D` 字节 HBM 传输，也不是说每个字节有单独一条指令。允许重新编码 cache、改变接口、替换消费者、延迟持久化或让观察者接受其他位置时，这个目的地址界不自动成立。

若进一步证明覆盖该目的地址资源的所有写机制在任何长度为 `t` 的窗口内，最多覆盖 `b + B·t` 个不同目的字节位置（`b` 是经证明的初始在途/突发信用，`B` 为硬容量上限，而非实测平均值），则 `t ≥ max(0,D-b)/B`。如果窗口初态已纳入全部在途更新且没有突发信用，可取 `b=0`；否则不能无依据令其为零。`B` 的资源范围、仲裁及单位必须与所计写机制一致，不能把某条链路规格直接当作所有写机制的覆盖速率。cache 可在片上保留脏数据、稍后写回，因此这一公式不自动给出窗口内 HBM 写入下界。

当规定仅目标合法槽所属的 512×2 字节行可改变、且不同物理槽的行不重叠时，定义去重后合法目标行数 `U`，则 `0 ≤ D ≤ 1024·U ≤ 1024·T`。`[-1,-1]` 跳过使 `U` 可以小于 `T`；重复槽也使 `U` 小于合法输入行数，重复目标的最终结果必须遵循真实接口的冲突语义，不能假定最后一行获胜。即使假设所有 `T` 行合法且目标互不相同，若 cache 初态已等于要求的终态，则 `D=0`：允许跳过相同写入的实现不受正数状态变化界约束。相反，`D` 大也不等于实际 HBM 写字节。当前没有绑定的有效槽、冲突、cache 初态和末态，也没有适用的硬 `B`、`b`，故本样本只能写出条件式，不能给出正数时间下界或宣称这一量必然随融合消失。该域也不能取代全推理框架、不同缓存格式或任意实现的上界。

## 已绑定形状不足以识别状态变化：固定有效槽与写入值的反例

对 EP8 L14 task 13276 的已保存结构元数据（cache `[17314,128,512]` BF16、slots `[8180,2]` INT32、values `[8180,512]` BF16），构造一个**假设接口状态**：令第 `i` 行目的槽为 `(i//128, i%128)`，`0 ≤ i < 8180`，共 8,180 个互不重复的合法坐标，仅使用 64 页；令 8,180×512 个更新值均为 BF16 零。再分别假设指定目的行初态全为 BF16 零，或全为有限 BF16 位模式 `0x0101`。在上节限定的物理目标行不重叠、逐字节可观察且无外部免费更新条件下，**槽、更新值、shape 和 dtype 完全相同**，前一种状态变化 `D=0`，后一种 `D=8,376,320` 字节。两种都是假设，不是历史运行中观察到的 slot 或 cache；即使额外确认了所有槽合法互异并知道实际 updates，仍需要对应目标初态与目标终态才能求 `D`。实测记录中两者都未提供，不能根据 shape 把 `D` 设为正值。

该反例只证明这些**结构字段**不足以识别指定地址状态变化；不声称两种构造能重现历史 trace 的时间、PMU 或完整运行上下文，也不证明历史实际状态可任意替换。现有 profile 时间可以描述被测实现和采样条件下的执行，缺少经验证的逆向模型时不能据此算出 `D`；改变接口、缓存编码或允许别的观察位置时此界也不自动适用。相同状态时现有内核仍可能执行写，因此 `D=0` 不意味着其耗时或实际 HBM 流量为零；同时，缺本机覆盖能力上界也不能将 `D` 换算为正数时间下界。复算 `python3 agents/fusion/f08_state_observability_counterexample.py`：输入聚合 SHA-256 `7b22c498beefb88aef225d654fdf1f136b898d29d5575e0382225af8289fba2e`，脚本 SHA-256 `a059785f107ed454976cddb2ee32723417b23521ee26841f0fc83f48751b0ee3`；仅 CPU 构造与检查，没有读取历史张量数值或执行 NPU。


## 理想算术下的重排反例：不能直接把 norm 权重移到 RoPE 后

冻结源码把带权 `kv_norm` 放在 partial RoPE 之前；在理想无 bias 的模型中，对一个受旋转的 pair，若把 norm 的同一 RMS 标量分母记为 `r`，权重记为 `D=diag(g0,g1)`，旋转记为 `R=[[c,-s],[s,c]]`，则原顺序的该 pair 为 `R D x / r`，将逐维权重移到 RoPE 后则为 `D R x / r`。即使在假设旋转是精确正交变换的实数理想模型里，二者差别由 `s(g0-g1)` 决定，一般不相等。特别地，令 `x=(1,0)`、`(g0,g1)=(1,2)`、`(c,s)=(3/5,4/5)`，分母以外的两种结果分别是 `(3/5,4/5)` 与 `(3/5,8/5)`。分母正数与该差异无关。

因此，融合内可以在保持原有语义的前提下重新安排计算，但不能仅凭 RoPE 保持向量范数就把带权缩放任意跨过旋转；例如旋转对的两个权重相等或该次 `s=0` 是这对分量在理想算术中可交换的充分条件。这个反例使用构造值，未核对实际模型权重、历史位置的 cos/sin，也不证明历史执行发生了错误或无法融合。冻结的 AscendRMSNorm 还存在可选 bias，这一简化模型不覆盖它；若实际 bias 生效，重排还必须保留 bias 的变换和舍入。即便理想代数可交换，真实 kernel 中 norm 的精度、舍入、epsilon、cos/sin 表示和原地写入仍需按要求验证，不能因此宣布 bitwise 等价。精确分数复算：`python3 agents/fusion/f08_scale_rope_counterexample.py`，脚本 SHA-256 `9ae082a776c17b516342e8a493058e91757ac75558ea2bb2ea7217f1deff25d3`。

## F08 的 norm→RoPE：跳过中间 BF16 舍入的精确反例

这里给出与上一节“gamma 不能任意跨旋转”**不同**的舍入反例：gamma 全为 1，不交换运算顺序，仅比较原中间 BF16 舍入是否保留。构造一个 512 维行 `x=[7,17]` 重复 256 次，假设 RMSNorm 的 `epsilon=0`，理想实数归约得到 RMS=13；一对分量的 norm 结果是 `(7/13,17/13)`。这对位于旋转区间，输入和 gamma 可用 BF16 精确表示。旋转参数取 FP32 可表示值 `c=5033165/8388608`、`s=13421773/16777216`（分别接近 0.6 和 0.8）。

在**限定的数学模型**中，norm 结果先以 BF16 round-to-nearest-even 舍入、旋转内部用精确实数算术、最终结果再舍入 BF16，则中间值为 `(69/128,167/128)`，旋转结果为 `(-23/32,155/128)`。若融合时直接将未舍入的 norm 结果喂给同一个旋转并只在最后舍入，结果为 `(-185/256,39/32)`：两维都不同。

因此，对于要求保留这个中间 BF16 数值边界的契约，融合不能仅因为不再物化中间 tensor 就顺带删除舍入；可以在一个 kernel 内显式产生等价舍入后的值，再完成旋转和写缓存。反例不排除误差容许下的其他实现，也不声称历史 kernel 使用 ε=0、具有完全精确归约或旋转中间计算、采用本组参数，或实际出现该差异。历史 trace 的 BF16 输出/输入签名也没有证明 norm 输出与 RoPE 输入是同一 tensor；对 EP8 L14 的 host/flow 关联和 storage 缺口见 `F08-EP8-TENSOR-PROVENANCE.md`。

CPU 精确有理数复算：`python3 agents/fusion/f08_norm_rope_rounding_witness.py`；脚本 SHA-256 `3bd4dddfe00b3d3fb2166a0635f33c0c94dca29a625a1d966206dd9d5d61cfae`。ZCode 只读检查了既有两份 F08 文档，确认其已提出舍入条件、没有该具体数值反例；数值和源码摘要由主控独立复核。未启动 NPU 或修改推理实现。
