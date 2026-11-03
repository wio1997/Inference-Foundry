# F17：双 FP32 投影与 C2 池化的可分块边界

## 既有覆盖与本次增量

之前 `COMPRESSOR-ENGRAM-REVIEW.md` 对 F17 仅记录同一 input cast 的两投影可考虑共享，以及投影与 indexer query quant 的已有重叠；F18 已深入 pool/norm 与 ring 的先读后写反例。本次聚焦 **Wkv/Wgate 每行每通道计算何时足以交给 pool，以及两条投影还能被 ring 消费**。不会把 F18 现有融合或 ring 工作重复算成 F17 的新收益。

## 源码分支和生产/消费

配置角色层 2、8、14 为 ratio=2 KV source，20 为 ratio=1。只有 ratio=2 的 `_write_compressed_source`（`dsa_v41.py:519–532`）先将全局 cache hidden 转 FP32，启动条件 indexer-query quant，再以**同一个** `hidden_states_fp32` 分别调用 `compressor.wkv` 与 `compressor.wgate`；两者都是 `nn.Linear(5120,512,bias=False,dtype=float32)`。它们产生每 token 512 个 FP32 `kv` 和 `score`，交给 `pool_projected`，后者再调用 `compressor_from_projected`。ratio=1 的 compressor 不存在这里的 gate/ring 分支，不能套用该图。`hidden_states.float()` 是否实际分配/搬运及两个投影实际读取层级均未证明。

`compressor_triton.py:85–195,225–314,650–719` 对一个完整 ratio2 group，按**每个通道各自的 score**跨组内两 token 做 max、exp/denom、再按该 score 加权对应 kv；跨段首组还要读之前 ring 中的 score/kv。同一 group 的每个通道独立得到 pooled，但下游 RMSNorm 在该 token 的完整 512 通道上归约。pool 后必须保留 token 对齐 BF16 输出和空/未完成 group 的初始零，再走原 norm。原 kernel 已组合 max、exp、sum、加权的 pool 算法；不能按独立待融合任务重复计数。

pool 返回与当前 `kv/score` 有另一个共同后继：`_cache_update_kernel` 把最多 `min(max_query_len,32)` 的尾部投影写入 FP32 ring，用于未来 chunk 的历史值。已有 `COMPRESSOR-RING-WITNESS.md` 给出某些槽的旧值先被 pool 读取、再被新值覆盖的反例；本次候选即使把 projected 结果直接供 pool，也必须产生这些需要持续的 **FP32 kv 与 score**，不能只保留 pooled BF16 或提前覆盖还要读的旧历史。下游 normalized latent 还有 index-key 和 long-KV 两分支，不能将其任一持久 cache 消掉。

## 依赖与分块机会

1. 对本段内单个 token、通道 j，`kv[t,j]` 与 `score[t,j]` 分别依赖 5120 个 input feature 和对应两个权重行；在对应点积的完整累加及原 FP32 数值形成之前，不能把局部 K 的结果当完整投影交 pool。两个独立投影可以对同一 input tile 共用一次取数或按 N 方向将两个权重拼成合并线性输出，但必须保留两套权重/布局、独立 FP32 精度/舍入、shape、状态消费者及最终顺序；同 dtype 的实数矩阵拼接不自动证明新 GEMM 与原实现逐位相同。
2. 单个 `(group,j块)` 的 pool 等待两位置相应 `kv/score` 的最终值或可用的 ring 历史；不需要其他**pool 输出通道**的 score/kv 来决定该块。可研究 group/channel 级投影→pool 对接，但当前 Python helper 要求整张量 contiguous 与同一 stream 的原 launch，设备 kernel 没有对外暴露逐块 ready；alias、分块 buffer 寿命、完整数据与输入范围必须先证明。某个完整 pooled channel 到后续 norm 的对接还需该 token 512 个 channel 的完整 RMS 统计；不能把 pool 的通道独立扩展成“norm 无跨通道依赖”。
3. 分离 current 与历史读取后，仅当已证明某 ring 槽所有旧值的消费者完成、新值对应 request/page/slot 无重复冲突，才能考虑该槽更新与其他独立 group 的 pool 同时进行；否则保留已有读后写顺序。F18 的已复算 32-slot 冲突反例适用于这里，不表示当下样本真实命中。
4. `_quantize_indexer_query` 在 `hidden_states.float()` 后按条件开始辅助流，源码注释明确希望与 Wkv matmul 重叠；正常数据边上该量化处理 prepared query，不使用上述 score。把两个投影合成一个、更改先后或扩展 fusion 时须重新满足 quantization 事件、权重和缓存写的 readiness，同时考虑 Cube/Vector/MTE 竞争；同一个时间窗口或数据独立不证明物理重叠有效。

## 防止错误重排的反例

不能普遍把 channel-wise pool 移到一个**所有通道共用同一权重**的 hidden-state 平均之前，再对平均结果只投影一次。两维的实数构造：输入 `x1=(1,0)`、`x2=(0,1)`，令 `Wkv=I`、`Wgate=I`，则 token1 得 `score=(1,0), kv=(1,0)`，token2 得 `score=(0,1), kv=(0,1)`。逐通道 softmax 后的结果是 `(p,p)`，其中 `p=e/(e+1)>1/2`；任何共用同一标量权重的输入平均 `a*x1+(1-a)*x2=(a,1-a)` 两坐标之和均为1，而 `(p,p)` 之和大于1，所以无法等价。这只排除这一种未经证明的重排，不排除精确保留通道依赖的重组，也不说明该具体输入出现在 169。

## 历史样本、边界与恢复

先前 stage096 在 EP8 已有 profiler 数据库找到 `_pool_kernel`、`_cache_update_kernel` 的多组历史设备记录，部分 shape 包含 `[T,512]` 和 `[17307,32,1024]` FP32 ring 逻辑形状；任务 ID 可在图重放时复用，同名/shape 不能证明所属层、`kv` 与 `score` tensor 身份或某次实际调用的关联。固定 EP8 层2 run2 的现有 `f11-layer2-run2-evidence.json` 记录 `[24,5120] BF16→FP32` Cast，随后存在 `[24,512]` 类任务；仅形状与顺序和该源码一致，不能仅此断言投影/池化实跑的数据链。stage164 已完成上述有界只读查询。独立核验 `/Users/wio/work/173-sglang-matched-172-benchmark/operator-ceiling169-v2-164-f17-layer2.exit-code` 为 0，`.verified.json` 的 `model_verified=true`、模型 `deepseek-flash`；结果 `f17-layer2-run2-evidence.json` SHA256 `9f2a659e1e016dfaadb866595cd9abc2a7b1b756369e5a3d3e7659ce90737542`，脚本 `query_f17_layer2_window.py` SHA256 `13e7aac08cf91fc4caefde01d5f026e120692530d5f167344a405223fdc8f834`，EP8 CSV SHA256 `488d3e3736677d7d041edbe8307d3c06b79553362e54aef07aba37fd74229cc7` 与清单一致。具体窗口（相对起点 µs）：

| 记录 | stream/task | 时间 µs | shape / dtype |
|---|---|---:|---|
| Cast | 89/177 | 183.356–191.356 | `[24,5120]` BF16 → FP32 |
| MatMulV3 | 89/179 | 191.416–217.075 | FP32 `[24,5120];[512,5120]` → `[24,512]` |
| MatMulV3 | 89/180 | 217.095–237.355 | 同 shape 和 dtype |
| `_quantize_indexer_query_kernel` | 87/36 | 191.936–199.436 | shape/dtype N/A，model ID 4294967295 |
| `_pool_kernel` | 89/182 | 240.695–245.415 | shape/dtype N/A，model ID 4294967295 |
| `_cache_update_kernel` | 89/183 | 245.435–250.815 | shape/dtype N/A，model ID 4294967295 |
| RmsNorm | 89/184 | 250.835–261.134 | `[24,512];[512]` BF16 → `[24,512]` BF16 |

两个 FP32 MatMulV3 与投影的 shape 和顺序相容；辅助流上的具名量化任务时间完整落在第一个 MatMulV3 的 25.659 µs 区间内，重叠时间 7.500 µs。pool、update 与 BF16 norm 的同流先后也与源码路径一致。这证明所列设备记录和相对时间，**不能证明**两个 matmul 是 Wkv/Wgate、aux 任务就是此 prepared query 的 quant、pool/update 与 norm 操作相同张量、不同 stream 的 event 对应或该 7.5 µs 隐藏了多少端到端时间；Triton 任务的 model ID 和 shape 无法辅助唯一绑定。只列指定类型，未记录的内容不能当不存在。

该分析只读冻结代码和历史资料，不跑 NPU、不改目标实现；候选正确性、资源争用与性能收益均待证。

## ratio-2 跨段首组：三遍逻辑读取可研究重组（2026-10-30）

本节是已融合 `_pool_kernel` **内部**的条件重组，非另一个 kernel 融合、也不与 F18 ring 重排重复计收益。冻结 `compressor_triton.py` SHA-256 `1c26f996fa91bcd3ae497e1ae33e88a2e0bf38c2e734f8af85baee06492a87ee`，`compressor_from_projected:675–704` 对该 token-aligned 路径固定 `RATIO=RATIO_PAD=CHUNK_ROWS=2`、`SINGLE_BLOCK=NO_PAD=True`，并把连续 FP32 `kv`、`score` `[T,512]`、FP32 ring `[blocks,32,1024]` 和 INT32 元数据交给 `_pool_kernel`。输出为 BF16 `[T,512]`，完成组写在其完成 token 对应行，后由独立 RMSNorm 消费；FP32 `kv` 和 `score` 随后还用于更新 ring（:706–719），不能因直接对接 pool 而删除持久状态。

`_pool_kernel:230–264` 在有效 group 的 `residual>0` 时，无论 `SINGLE_BLOCK` 都走 `_pooled_blocked`；对固定 ratio=2，仅跨段第一个完整组可能进入此分支。举例，`start_pos=3, used_len=1`，完整组是 token 2 和 3：前者的每通道 `score/kv` 来自该请求的 ring `2 % 32`，后者来自投影行 `seg_row_base`；只有 block ID 非零且在范围内等 guard 通过时才执行。其后当前源码 `:101–196` 分三遍求 max、exp 总和和加权值；每遍都以历史/新行各取一遍 score，后两遍各算一次对应 `exp(score-max)`，第三遍才读两行 kv。相对地，同版本的非残余 `SINGLE_BLOCK` 分支 `:265–292` 已按该组一次载入 score 并复用指数，不应宣称所有 pool 都在做三遍。

可研究对该特定残余组把两行 score 与 kv 按同一选择规则装入局部状态，一次求 `m=max(s0,s1)`、`e0=exp(s0-m)`、`e1=exp(s1-m)`，然后 `p_i=e_i/(e0+e1)`、`pooled=p0*v0+p1*v1`；保留原分量选择、顺序、FP32 精度和最后 BF16 输出。此举只复用既有指数及读取，不需要把该 pooling 误当作缺失的外部融合。数学上与原分支同式，但实际是否数值等价需核查编译器 lowering、exp/div、加法归约与舍入、NaN/Inf 等规则，不能只凭公式保证逐位相同。

仅以每个**有效完整组**、`HEAD_DIM=512`、FP32 计数，源码表达式中原三遍合计读取 `3×2×512` 个 score 元素；候选一次性保留两行则为 `2×512` 元素，差额 2048 FP32 元素（8192 字节）的**逻辑读取量**，同时 exp 的表达式元素数由 `2×2×512` 降至 `2×512`。这不是已测 HBM 事务或带宽：掩码、编译器重复消除、cache/UB、向量化和实际 lowering 都会改变物理访问和指令；延长局部状态寿命也可能增加资源占用。只有 residual 分支匹配的请求和组才有此候选，不能乘整个输入所有组，亦无法据此直接计算运行耗时或收益。既存 EP8 的 `_pool_kernel` 记录与源码路径形状相容，但未绑定 `start_pos`、分支、历史张量内容、实际编译结果或这个组，因此不能说历史样本实际执行了它。

不得只用新投影行代替历史：例如同一通道两行分数均为 0，历史值为 10、新值为 2，正确 pooled 为 6，忽略历史得 2。须保留 ring 的请求/page/slot 映射、旧值读完后才允许同槽更新、所有 channel 的 score 和 kv、未完成组输出初始化以及所有后续消费者；一个值/分数及时就绪不能推定完整组或跨流安全。无设备负载和目标推理代码修改。

## 两行池化改写：有限 FP32 输入下的舍入反例（CPU 模型）

前述 ratio-2 候选中，一个可能的重组是用 `p=sigmoid(s1-s0)`，将两行加权和写为 `v0+p*(v1-v0)`。在理想实数、有限输入下，这与逐通道的稳定 softmax 加权和等价；但并不能据此保证现有 FP32 路径、尤其最终 BF16 输出逐位相同。冻结 `compressor_triton.py:85–195,675–704` 先计算每通道 max、指数与分母，再分别乘值并归约；同文件源码 SHA-256 `1c26f996fa91bcd3ae497e1e33e88a2e0bf38c2e734f8af85baee06492a87ee`。前述 EP8 层 2 记录了两个 FP32 `[24,5120] × [512,5120] → [24,512]`、后续 `_pool_kernel` 与 BF16 `[24,512]` 归一化，名称和形状支持该源码结构的条件候选；调用张量身份、具体 pool 参数及 device lowering 尚未证明。

在 CPU NumPy 2.0.2 中，对单通道输入采用每步 FP32 舍入的算术模型：`s0=0.7128095030784607`、`s1=7.908058166503906`、`v0=-64.67178344726562`、`v1=59.7182502746582`。max 稳定的分别加权计算为 `59.625003814697266`；上述 sigmoid/中心差写法为 `59.625`。随后按 BF16 round-to-nearest-even 截断编码得到 `0x426f`（59.75）和 `0x426e`（59.5）。因此即使都使用有限 FP32 输入且实数公式等价，只要采用该 CPU 模型的运算和舍入方式，最终 BF16 也可以不同。候选要满足具体服务的数值验收条件，或保留与原路径相应的操作顺序和语义；仅凭数学等价不能承诺逐位一致。

复算 `python3 agents/fusion/f17_pool_sigmoid_fp32_witness.py`，脚本 SHA-256 `9146ce418a8a4bb9a970e1a773f2b89cbdbc09ca5905d6b34e437c277aa48df8`，使用固定 FP32 数值并检查 BF16 位码。**这是 CPU 浮点计算模型反例**：NumPy exp、乘加、归约及转换不保证与 910C/Triton/CANN 版本相同；未证明这些值能由真实权重产生，亦未声明硬件实际出现该差异、候选不能通过容差验收或可以获得性能收益。无新设备负载或目标实现改动。

## C2 元数据跨 builder 共享的条件反例

冻结 `dsa_v41.py` SHA-256 `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`，构造函数为每个 builder 分配 ring、完成 mask、位置及 cos/sin 缓冲（:791–807）；`_publish_task` 遇到已有 `c2:compressor` 键直接返回 buffer，不提交当前 closure（:879–900）；C2 调用却忽略返回值并发布当前 builder 的独立 ring、mask、位置和其 mask 的 group ID（:1375–1388），消费者在压缩前按 group ID 等待（:523–533）。

假设两个新 builder 各自持有独立初始缓冲，却被人为传入同一个 `common_v41_metadata` 字典，且第一 builder 已完成构建。对于一请求 `query_start=[0,1]`、`seq_len=6`、`positions=[5]`、block table 首项 7、不跳过更新，非 packed Python 分支的公式（:1325–1348）令第一 builder 的 ring 为 `[5,1,0,0,7]`、mask 为 true、source position 为 4；第二个命中键后自己的任务不执行，却仍发布自身初始 ring、false mask 和位置 0。故仅共享键或 mask 返回值不足以保证其余独立状态和对应等待事件同步；研究复用需证明整体状态、group、block table、位置、缓冲生命周期及 ready 事件相匹配。

此例的两个实例共字典、缓冲独立和执行分支是显式假设；尚未证明历史运行存在此组合，不能称为现有 bug 或确定收益。已保存 pool/update 同名记录也不能绑定这些 builder、输入或状态。

### 上述共享假设的调用链范围

冻结 runner `agents/bounds/metadata-path-snapshot/ascend_runner.py` SHA-256 `568c3049205b9caff5c656a7f5dd4c6e79fb358abe526f89fb9fb3f70a60a43c`，`:3822–3829,3875–3883` 每次 build 在各 framework KV group 循环内创建独立的 `common_v41_metadata`，但跨这些组共享另一个 `common_v41_batch_metadata`；C2 使用前者。因此，不能因为不同 KV group 共享 batch 字典就断言上例发生。

`:6022–6110` 在每个 KV group 内按 backend 全名、逐层 KV cache spec、`use_mla_rope` 分桶，每桶创建一个 metadata builder；桶内多层共享该 builder。若 state 层的三项均相同而只形成一个桶，就不会在这一遍中出现两个独立 state builder 共享该组字典的条件；若一个 KV group 有多个不同桶且各桶都是 compressor-state builder，则组内确会向各 builder 传同一字典，需进一步检查状态和事件。Legacy CP 的 `context_parallel/dsa_v41_cp.py:48–61,87–95,112–114` 中各外层 builder 各有 global builder，组内通过 `setdefault('cp_global',{})` 传递共享子字典，仍需上述多个桶才满足该条件。组形成与逐层 padded spec 见 `cache_config.py:147–166,181–205`，不能仅凭同属一个组推定 spec 一定相等。

保存资料未绑定历史进程实际 resolved cache spec、分桶和 builder 身份；不能断言条件实际可达或排除实际可达，也不能断言线上有 bug。该 runner 文件在当时 baseline/stage 快照 SHA 一致，但不是历史进程加载对象的证明。

冻结配置的 KV source 层为 2、8、14、20，其压缩率分别为 2、2、2、1；按 `model.py:714–727` 的角色和 `compressor.py:21–46` 的条件，前三层创建 C2 state。它们的初始 state 构造参数均为 block 32、1 个 KV head、head size 1024、FP32、head size v 0，backend 相同；但 `cache_config.py:159–205` 又按每个关联 tuple 的其他 cache spec 计算并设置 padded page size，再将 state 名称放入同一 group。因此初始参数相同不足以证明最终三个 spec 相等或必为一个桶。现有材料未捕获三个最终 spec 和运行时 builder 分组，这仍是条件可达性的关键缺口。

### 三个 C2 tuple 页大小的条件计算

进一步对照冻结 `cache_config.py` 和 stage 容器当前安装的 `vllm/v1/kv_cache_interface.py`（SHA-256 `1cf202f1a44d5bc5c3832b70b41507687ecceea3784e67a4f003bc0210d6eecb`）、Ascend `core/kv_cache_interface.py`（`b7356c5e499446a1bbf92be9490753336ed7f7d31e521735659491b613f370b3`）的页大小公式。**仅在**这些源码与实际对象相符、block size 128、head dim 512、非 packed、target SWA BF16、没有额外或覆写 spec、目标元数据按所示排序分组的条件下，前三个 ratio2 tuple 的页大小为：

| 逻辑页面 | 条件字节数 |
|---|---:|
| C2 state：32 行 × 1024 FP32 | 131072 |
| long KV：64 行 × 512 BF16 | 65536 |
| index：64 行 ×（128 INT8 + 1 FP16 scale） | 8320 |
| target SWA：128 行 × 512 BF16 | 131072 |

前三个 tuple 的 `long + index` 为 73856 B；对应 C2 state 和 target SWA alias 的未填充页各为 131072 B。非 packed 条件下 indexer 不创建 folded cache，且额外 draft SWA 至多按排序给每个 tuple 一个 alias；`get_layer_tuples` 对 long+index+folded 和所有 alias **取最大值而非求和**。令第 i 个前三 tuple 的 draft alias 未填充页大小为 `P_i`，无 alias 令其为 0，则条件页大小 `C_i=max(131072,P_i)`。例如所有 `P_i≤131072`，或三个 `P_i` 相等时，三个 `C_i` 都相等；若仅部分 tuple 有大于 131072 的 alias，则它们可能不同。条件下第 20 层为第四个 source tuple、ratio1，不自动改变前三 tuple 页大小。

在三个最终 state spec **所有字段**相等、同 backend、同 rope 分类，且 runner 未被覆写的附加条件下，runner 归为同一桶、一个 builder，前述多个 builder 共字典的反例不经该正常路径触发。当前 stage 文件和历史模型配置不证明实际运行时 cache spec、draft alias 数量与大小、硬件分支或 builder 身份；条件可能不成立，不能将这项算术称为实际内存占用、生产错误或全局正确性证明。未启动 NPU。


## 同 scope 的 FLOAT 与 BF16 同形状投影记录（条件对照）

从同一已存重型 profiler 提取中，选 16 rank、层 2/8/14 各两种 digest，共 **96 scope**；每 scope 按完整签名唯一选出 1 条 BF16 和 2 条 FLOAT 的 `MatMulV2`：均为 `[M,5120]×[512,5120]→[M,512]`、ND、AI_CORE、block 24，原名相同。两条 FLOAT 同 stream、相邻 task ID 且时间有序；BF16 记录位于另一 stream。下表将**每条 FLOAT 单任务**除以同 scope 的 BF16 单任务时长，再在各 M 内取统计；不是两个 FLOAT 耗时之和：

| 物理行数 M | scope 数 | FLOAT/BF16 单项时长比 min/median/max | BF16 / FLOAT 单项时长中位数 µs |
|---:|---:|---:|---:|
| 4760 | 24 | 1.710 / 1.975 / 2.116 | 154.267 / 306.627 |
| 5144 | 24 | 1.909 / 2.216 / 2.356 | 165.157 / 368.203 |
| 8180 | 48 | 2.067 / 2.238 / 2.394 | 267.957 / 600.688 |

按组分别计算的时长中位数不一定来自相同 scope，不能拿它们的比值替代逐条中位比。EP3/L2 的一个例子：BF16 为 259.080 µs，两条 FLOAT 为 611.600 / 591.440 µs。此组的两条 FLOAT 与上文 ratio2 层的 Wkv/Wgate FP32 形状及角色相容，但这 192 条尚未逐条绑定 host 栈、权重对象或 tensor；BF16 是另外一条流，工作内容、输入/权重、资源状态与实际 lowering 均不保证相同。故该观察支持优先核查 FP32 投影的实际路径与资源，**不证明 FLOAT dtype 单独造成 1.98–2.24 倍耗时**，也不支持为了提速直接改掉源码要求的 FP32 数值或宣称可省时间。选择仅覆盖导出且匹配的记录，不等于 trace 全覆盖。

只读复算 `python3 agents/fusion/analyze_f17_fp32_bf16_candidates.py`，脚本 SHA-256 `7abc93ff5b079dcd551229457228e5ed397b6d11b7b73163c2a2e41534b63e92`；输入 `diagnostic-20261029/target-layer-all-kernels-16rank.jsonl` SHA-256 `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`。该脚本校验每 scope 候选数量、形状、dtype、format、core、同流与任务顺序，仅比较同设备轨道内的导出区间，不比较跨设备时间戳。

## 两个投影拼成一次 GEMM 的输出布局反例

冻结源码 `compressor_from_projected` 在传入 pool 前检查 kv 和 scores 为 FP32、同形状且连续；不满足连续性直接抛出 `ValueError`。pool 读取按 `row*HEAD_DIM+channel` 寻址，后续 ring 更新也依赖既有布局。假设将两套 `[512,5120]` 权重拼成 1024 输出通道，一次标准 row-major GEMM 产生连续 `Y[T,1024]`，再用 `Y[:,:512]` 和 `Y[:,512:]` 作为两个 `[T,512]` 的视图：当 T>1 时，两视图的行 stride 是 **1024**，而紧密连续输出需要 512，故不能直接传给现有接口。T=1 的单例维度可能被 PyTorch 判为连续，空行情况需要单独处理。

这是正确性限制，不只是效率问题。例如 T=2、以元素为单位，kv 视图的 `[1,0]` 在拼接缓冲偏移 1024；若绕过检查，旧 kernel 按连续行规则读取偏移 512，得到的是第一行另一半的数据。score 视图的相应偏移应为 1536，旧规则得到 1024。纯 CPU 地址算术复算 `python3 agents/fusion/f17_contiguous_split_witness.py`，SHA-256 `7c1ca9f0c4b98d06f76ba37cea821fd835eed9cd13a965635d741a423d51f63d`；它不模拟 NPU。冻结 `compressor_triton.py:647–675` 和对应内核索引的 SHA-256 `1c26f996fa91bcd3ae497e1ae33e88a2e0bf38c2e734f8af85baee06492a87ee`，未证明历史二进制版本绑定。

可考虑候选 GEMM 直接发布两块连续结果、改 pool 和 ring update 接受显式 stride，或增加拆包；分别还需保持 FP32 数值、ring 先读后写、tensor 生命周期与同步。仅以 M=8180、两张 `[M,512]` FP32 结果为例，合计逻辑 33,505,280 B；若某方案确实完整读取一次并写入独立连续结果一次，则拆包逻辑读写为 67,010,560 B。**这些不是实测 HBM 流量或必经成本**，其他合法输出布局和融合可改变它；历史这两条投影与 pool 的 tensor 身份尚未绑定，不能估计可实现提速。

## 双投影与量化的选定导出区间

沿用上述 96 scope，每 scope 另选一条 `_quantize_indexer_query_kernel` 记录，均与两个 FLOAT 任务位于同一设备轨道进程的不同 track 和 stream。按导出时间计算区间交集，**87/96** 的量化记录完整落在第一条 FLOAT 的区间内；剩余 **9/96** 的量化记录在第一条 FLOAT 开始前结束。第二条 FLOAT 与该选定量化记录在 96/96 中无区间交集。两个 FLOAT 记录的同流间隔在 87 个 scope 中不超过 0.0405 µs，另外九个为约 0.889–2.370 ms；100 µs 只是事后用于区分本样本两簇的阈值。九例集中于 EP2/EP3 的特定摘要和层，不代表其他运行。

这将先前单个小行数样本的重叠观察扩展到大行数范围，但量化名称和 host scope 不能证明 tensor、事件对象或关键路径身份；不同 track 的时钟误差及未导出活动未闭合。没有交集不代表设备空闲，不证明大间隔由量化、调度或同步造成；时间重叠也不能直接算作被隐藏的端到端时间。可用上述脚本的 `--public-json publication-draft/projection-pairs.json` 生成去除 rank、层、标识符和绝对时间的配对数据，再由 `publication-draft/verify-projection-pairs.py` 独立复算时长比、87/96 和 9/96；这只验证匿名派生数据的算术，不能替代原始 trace 的捕获、身份与时钟审计。


## 按 tile 融合时的两类新值写入与输出初始化约束

冻结 `_cache_update_kernel:346–365` 对单个请求仅写最后 `min(used_len,32)` 个投影行，槽位为 token 位置模 32。条件构造：`start_pos=0, used_len=33`、有效非零 page，原路径写 token 1–32，因此槽 0 保存 token 32 的 kv/score。若改成每个 token 一完成投影便无序写 ring，token 0 和 32 都写槽 0；即使旧历史读取已经完成，如果 token 0 最后写，则槽 0 错误地保留旧 token 0。这是**新值之间的写写冲突**，与此前旧 ring 内容先读后写的反例不同。重组至少要保持只写尾部 32 行或证明同槽版本顺序；不同请求能否共享同一 `(page,slot)` 及其顺序也须另证。该构造不是历史冲突实测。

冻结 `compressor.py:25–33,55–64` 在条件分支复用 `_ring_pooled`；`compressor_from_projected:675–683` 每次先将本次输出范围置零，然后只对有效完整组的完成行写 pooled 值。对 `start_pos=0, used_len=1` 的未完成二元组，若复用的对应输出行上次是非零、候选融合省略初始化又没有给它写零，残留值就会进入后续 norm。可将清零与其他工作合并，但须覆盖所有没有完整组、无效页及相应输出行，同时确保清零与有效写入没有无序写写冲突；不能仅由第一、第二投影完成证明池化输出已正确。

本机复算 `python3 agents/fusion/f17_ring_write_zero_witness.py`，SHA-256 `aa1fb48ce051f090827d749f7849e764b9ee1ef28b30f219150b903ab4c7ad13`。该脚本只演示槽位与输出初始化的 CPU 算术，不模拟设备或证明历史输入；源码快照与 hash 见本文件前文。以上候选可以通过其他正确状态表示或调度满足，不是强制维持现有两个 kernel，也不是性能收益。


## 96 个 scope 中 FLOAT 与 BF16 目标的同轨邻近记录

仍限定上文 16 rank × 层 2/8/14 × 两摘要，共 96 个已选 scope，在每个 scope **已匹配的任务列表内**按设备 track 和时间寻找目标前后的最近记录，结果 96/96 均有下列序列：

- FLOAT 两投影所在 track：第一项之前最近的已选任务为 BF16 `[M,5120]`→FLOAT `[M,5120]` Cast；第二项之后最近三项为 BF16 `[M,512]` 的 ZerosLike、`_pool_kernel`、`_cache_update_kernel`。pool 的前两个输入均为 FLOAT `[M,512]`，ring 为 FLOAT `[B,32,1024]`，pool 输出 BF16 `[M,512]`；cache update 的前两个输入也为 FLOAT `[M,512]`。
- BF16 投影所在的另一 track：后三项均为 RmsNorm、InplacePartialRotaryMul、ScatterNdUpdateSk；RmsNorm 主输入 BF16 `[M,512]`、权重 `[512]`，scatter 更新 BF16 `[M,512]`。

这些签名与源码中压缩器两投影→pool/ring 和 attention KV 投影→norm/RoPE/cache 写的**两类路径分别相容**，增强了两类记录可能有不同角色的解释；它们不能作为两个任务处理相同数据的证据。所选 EP8/L14 的 BF16 后续 norm/RoPE/scatter 之前已有调用栈及 flow 证据，但 BF16 matmul 以及前后 tensor 身份未闭合；不能把这条进一步绑定推广到其余全部 scope。不同流、不同 dtype 和潜在不同模块使 1.98–2.24 倍比值更不宜单独归因于精度。

复算 `python3 agents/fusion/audit_f17_neighbor_roles.py`，脚本 SHA-256 `f84a581bedef2879f5ba966756c1ac6c78649e9d6c9a00b3741167dffa7d9067`，输入 hash 同上。脚本对目标的 shape/dtype、每组数量和这七种邻居类型断言，输出逐种 shape/dtype 与计数；没有邻居缺失只限该 96 scope 中已保存列表，**不等于完整设备流连续，也不证明张量数据边、计算因果或线上耗时**。
## 同形 FLOAT 双任务时长与选定量化记录的关系（2026-10-30）

复用此前 96 个 scope 中同轨同流、连续 task ID 的两条 FLOAT MatMulV2 记录，物理签名均为 `[M,5120] × [512,5120] → [M,512]`。逐对比较第一条与第二条时长，第一条更长的有 95/96 对。事后按另一流上的**选定**量化记录与第一条的导出区间关系分组：87 对量化完整落在第一条区间内，其中 86 对第一条更长；其余 9 对量化在第一条之前结束，9 对第一条仍更长。

因此在这 96 个已选 scope 中，所选量化记录与第一投影发生导出时间交叠，并非第一投影较长的必要条件。这只排除了一个狭义解释：未选量化、其他流、设备竞争、缓存、权重、有效行、运行状态及调度均可能不同，无法据此判断原因或真实设备并发。两条同形任务的输入/权重身份、工作量及二进制尚未绑定，采集干扰和导出完整性未知；9 对集中于同一 rank/摘要，分组事后选择，不代表独立实验或普遍结论。

按 M 和组内逐对时差取中位数：4760 行交叠组 18/18 更长、中位差 8.98975 µs，提前结束组 6/6、中位差 7.04975 µs；5144 行交叠组 24/24、中位差 10.33975 µs；8180 行交叠组 44/45、中位差 17.8 µs，提前结束组 3/3、中位差 11.62 µs。不得将这些小样本差值解释为独立量化造成的耗时。

原始已选 scope 来自 `diagnostic-20261029/target-layer-all-kernels-16rank.jsonl`（SHA-256 `0493115eb0a8c9191297f32ee4f845dc69d5d60ec09fce6deb96ec1faeb9a3c4`），提取程序为 `agents/fusion/analyze_f17_fp32_bf16_candidates.py`（SHA-256 `7abc93ff5b079dcd551229457228e5ed397b6d11b7b73163c2a2e41534b63e92`）。匿名配对和复算脚本见 `publication-draft/projection-pairs.json`、`publication-draft/verify-projection-pairs.py`；后者只验证匿名数据上的算术，不验证历史张量绑定。
