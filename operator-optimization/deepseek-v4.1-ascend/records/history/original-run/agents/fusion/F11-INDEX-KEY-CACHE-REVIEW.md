# F11：Indexer key 的量化与双缓存发布

## 与现有分析的关系及实际角色

F11 指 compressor latent 经 Indexer 的 `wk → k_norm → partial RoPE → npu_dynamic_quant → INT8 key/FP16 scale 两次 scatter`。它与 F19 有重叠但不等同：F19 已分析同一 compressor latent 分成 index-key 和 long-KV 两个分支，并指出先 indexer 再对 latent 原地旋转；本次专门分析 index-key 分支内部的归约、量化、两张 cache 及下游消费者，不把相同的 latent 物化/读取重复计算为两个收益。F09 是 query 量化，不能将其 FP16-roundtrip scale 和 rounding 契约直接套给 `npu_dynamic_quant` key。

冻结 `model.py:714–745` 与 `frozen-layer-coverage.json`：只有 KV source 且同为 index source 的目标层拥有 `wk/k_norm/k_cache` 并写 index key；冻结配置中是 **2、8、14、20**，前三者 ratio 2，20 为 ratio 1。24、28、32、36 是 index source，但 `owns_k=False`，它们的 QLI 读取第 20 层的 index key cache；非 index-source 层复用先前选好的 topk 索引，不一定调用 QLI 读取 key cache。2/8/14 各自 key cache 的目标 QLI 消费者分别是其本层；20 的 key cache 在配置角色下由 20、24、28、32、36 的 QLI 读取。这里是源码/配置静态消费者，不证明每次请求 cache 非空、历史值或实际 storage 身份；相应 long-KV/索引结果还有自己的消费者和未来 forward 的持久用途。

## 精确输入、输出及现有实现

1. `dsa_v41.py:509–545` 在 ratio 2 时，pool 返回 normalized latent，使用 metadata 中 C2 source cos/sin 和 index slot；ratio 1 分支取 compressor 输出与普通 source cos/sin。调用 `update_keys` 后才对原 latent `view(-1,1,512)` 原地 RoPE 并写 long-KV。`index_slots` 与 `long_slots` 分属不同 metadata 的切片，即使维度相同也不能假设相同物理槽。latent 同时被两分支消费，F11 融合不能提前原地改写 latent。
2. `indexer.py:104–124` 非 packed 分支且 `owns_k`、latent 非空时：BF16 `wk` 映射 512→128；BF16 RMSNorm，接着对 key 后 64 个维度原地 partial RoPE；`npu_dynamic_quant(..., dst_type=int8)` 返回量化 key 和 scale；前者散写 key cache，后者先 `unsqueeze(-1).to(float16)` 后散写 scale cache。RMSNorm 要求按原 row 的整个 128 维做归约；旋转需要尾部 64 维的对应 pair 和 cos/sin；量化范围、rounding、zero 和异常值严格按实际 `npu_dynamic_quant`，此处源码没有显示其内部语义，不可套用 query Triton 的量化规则。
3. `cache_config.py:92–116` 非 packed index spec 为 INT8 head size 128、每 index 行一个 FP16 scale、`storage_block_size = block_size // compress_ratio`。实际 cache shape、page stride、block table 与 dtype 由绑定实例和 metadata 决定；`scatter_cache_sk` 在 `dsa_v41.py:228–242` 提醒 cache plane 可是 layer-outermost 大分配的 view，应使用真实页步长，`[-1,-1]` 跳过。该函数包含 `.to(cache.dtype).contiguous()`，可能分配也可能为 identity，不能凭源码认定内存搬运次数。相同/重复 slot、别名覆盖、排序及 host/graph 的写完成语义未获运行证据。
4. `select_projected` 的 non-packed QLI 将 key、scale squeeze 后传给 native operator，保留了 Hybrid cache page stride；模型建图根据最近的 KV source 取相应 `k_cache`。二者在模型语义上是成对状态；若应用契约要求每个 slot 的 key 和 scale 来自同一输入更新，则融合发布必须满足该要求并保留正确地址与可见性，不可只写 INT8 而忽略 scale。当前未证明 slot 唯一、重复写时赢家一致，或历史实现必然满足该一致性；须先核实原有允许结果集合及用户精度语义，不能将期望当成已证契约。packed 分支另有 `write_index_cache`，候选源可额外折叠 cache；A3 正常加载 STANDARD/non-packed 的证据见 `INDEXER-RUNTIME-REVIEW.md`，历史实例和每次实际分支尚未绑定。

## 历史样本：EP8 第 2 层 run 2

在已 source-role 归属窗口中（`f07-q-kv-streams.json`，model 49，main stream 89），以下事件在同一 stream 先后出现；起止相对窗口起点，单位 µs：

| 事件 | task | 起止 µs | 记录的关键 shape / dtype |
|---|---:|---:|---|
| MatMulV2 | 185 | 261.174–267.854 | `[24,512]×[128,512]` BF16 → `[24,128]` BF16 |
| RmsNorm | 186 | 267.874–278.334 | `[24,128];[128]` BF16 → `[24,128]` BF16 |
| InplacePartialRotaryMul | 187 | 278.354–285.174 | `[24,1,1,128]` BF16，cos/sin 尾维 64 |
| DynamicQuant | 188 | 285.214–288.014 | `[24,128]` BF16 → `[24,128]` INT8 与 `[24]` FP32 |
| ScatterNdUpdateSk | 189 | 288.034–299.674 | `[17307,64,128]` INT8，slots `[24,2]`，updates `[24,128]` INT8 |
| ScatterNdUpdateSk | 191 | 301.094–312.113 | `[17307,64,1]` FP16，slots `[24,2]`，updates `[24,1]` FP16 |
| partial RoPE / scatter | 192 / 193 | 312.133–331.233 | 512 维 BF16，后续 BF16 long-KV 缓存形状 |

这些顺序、形状及两种 cache dtype 与源码 F11→F19 一致，足以确认相关类型的任务出现；**缺 tensor 句柄及调用栈，不能证明这是同一 latent、slot 或某次算子的实际数据链**。其中 24 是物理 token 行数而不是有效行数；source 在 CP 下使用 global cache token 域，query/indexer 可只有 3 行，不能用 3 行估算本次缓存更新。scatter 同名事件还有 SWA/long-KV，不能把层内所有 scatter 都归给 index key。无需将各任务时长相加视为端到端或可节省时间。

## 条件结构重组

- **norm + partial RoPE**：每行完整 128 维 norm 后做尾维 RoPE，可按 token 行独立供给下一步；保留 BF16 中间舍入与算子精度。需验证旋转 pair、输入 cos/sin、逐行跨流完成及与 long-KV 读取 latent 的先后。
- **RoPE → dynamic quant → key/scale 双写**：研究同一 kernel 在完成整行量化所需归约后，分别按真实 page stride 写 INT8 key 和 FP16 scale。候选必须确保和现有 quant + FP32→FP16 cast 一致，包括 scale 精度、取整饱和、NaN/全零处理，合法槽/无效槽、写覆盖次序、page stride、两个输出的 ready 条件。两个 scatter 的小任务是否已被底层消除、合并的 kernel 是否实际支持以及效益均未知。
- **拆分异步**：可以研究在某行 index key 已完整读取 latent 后，将该行 long-KV RoPE/写入与其他行的 key 处理流水，但必须保持读后写、每个目标 slot 与重复行/跨 chunk 的竞争关系、以及两种 cache 的真实依赖。不同流可能争用同一 Vector/MTE；现有整 kernel API 未证明逐行通知或 event 能力。不能以一整批 latent 已写为假设取消必要同步。

## 恢复与下一步

stage155 已由 ZCode `deepseek-flash` 对同一窗口完成只读查询；独立核验 `/Users/wio/work/173-sglang-matched-172-benchmark/operator-ceiling169-v2-155-f11-layer2.exit-code` 为 0、`.verified.json` 为 `model_verified=true`。查询 `query_f11_layer2_window.py` SHA256 `69becac3779f4f3bc6ba90cd3ea3c9b50b3e01a831f592a25781ee1ebc52b7ac`，结果 `f11-layer2-run2-evidence.json` SHA256 `cc26327c36e716502563b5f56fe9e63d65c7617448cb966819824a7c3e161b98`，输入 EP8 CSV 的 SHA256 `488d3e3736677d7d041edbe8307d3c06b79553362e54aef07aba37fd74229cc7` 与清单一致。未执行 NPU。

新记录补齐了两点：上述两次 scatter 之间同一 stream 的 task 190（299.694–301.074 µs）是 `[24,1]` FP32 → FP16 Cast，和 `scale.unsqueeze(-1).to(float16)` 相容；后续 task 198（335.853–426.291 µs）的 QLI 输入包含 `[17307,64,1,128]` INT8 与 `[17307,64,1]` FP16，和两张已记录 cache 的逻辑形状及类型相容。形状中额外的 1 维与 cache view 可能相容，但 **shape 无法证明相同 storage 或当前写入值被 QLI 实际读取**；本次没有记录 slots、scale 的数值或地址，没有逐调用栈、事件映射，model 49/stream 89 和近邻先后也不能弥补这些缺失。整条链是否可融合、相关 task 是否处于实际关键路径以及能节省多少仍 UNKNOWN。

后续若已有材料无法绑定 tensor/事件，就保留上述条件候选，不额外跑无针对性测试。当前已知的 key 缓存消费者、共享状态和各种空请求/捕获分支须保留，缺少 native 动态量化与 scatter 的逐位契约时必须标 UNKNOWN。

## 双 cache 更新中的重复 slot：等待与赢家一致是两个问题（2026-10-30）

EP8 第 2 层 run 2 的既有选样中，task 189 和 191 分别记录 INT8 key 与 FP16 scale 的 `ScatterNdUpdateSk`，之间 task 190 记录 FP32→FP16 cast；后续 QLI 的输入形状和 dtype 与两张 cache 相容。该样本只提供形状、名称和同流时间，**没有绑定 slot 数值、tensor storage、真实 producer/consumer 或历史 kernel 对象**。当前 stage 的 arch22 HP 入口源码注释称重复索引跨核写入非确定性；tiling key 30 另走 LargeIndex，不能把 HP 注释推广到全部实现，也未证明这两条历史 task 使用该分支。源码及哈希见 [F08 Scatter 语义审计](F08-SCATTER-INSTALLED-KERNEL-SEMANTICS.md)。

以抽象模型说明原 F11 双写合并候选的一条额外边界：假设 A、B 两输入行被写到同一个有效 slot，意图分别提供 `(key_A,scale_A)` 与 `(key_B,scale_B)`；两个 cache 的 scatter 若各自可独立选择 A 或 B 作为该 slot 的最终赢家，则可能有 `(key_A,scale_A)`、`(key_B,scale_B)`，也可能有 `(key_A,scale_B)` 或 `(key_B,scale_A)`。消费者等到**两次完整操作都已完成**可以避免按该事件契约看到未完成的写入，但不能仅凭“两次完成”推出赢家来自同一输入行。这个反例不证明历史 slot 重复、不同 kernel 的赢家在真实硬件上独立或混合结果实际发生；旧 HP 注释也不等于完整 API 的重复写结果集合。

研究把量化后的 key 与 scale 在同一内核分别散写时，要先明确验证或限定：slot 在一批及跨在途批次是否唯一；重复写的输入值是否逐元素相同；原接口对重复位置是否保证赢家或允许哪些结果；消费者什么时候可见；key 和 scale 的页 stride、位置、读写别名及完整 tensor 生命周期。**仅把两次 scatter 写进同一个 kernel，不自动使两块内存的写入对其他流原子可见，也不自动解决重复 slot 跨核竞争。**如果需求要求一致配对，可以研究在证明唯一 slot 或先形成合法共同赢家之后再发布；如果原接口容许非确定输出，而新实现缩窄结果集合或改写顺序是否被接受，需依据明确的用户/精度及运行语义契约，不能凭此假设自动等价。读者需要的就绪、必要的事件或其他可见性同步仍须单独证明。

重复 slot、scale 的动态值和具体二进制在现有采样均 UNKNOWN；没有新设备采集，因此这只细化 F11 已登记的重组条件，不是发现线上错误或收益。若未来在用户授权下取证，最小元数据包括已绑定一次 task/replay 的有效和重复 slot **脱敏计数**、两条写的后端/tiling 契约和可验证的发布依赖；光看相邻时间或同 shape 不能确定它们。

本机 ZCode 第一次把两次操作误当成各自写完整 pair，其回答不适用本反例；改为明确 key-only 与 scale-only 两缓冲后，`zcode/f11-separate-winner-recheck.zcode.json` 的四种组合与本机 `itertools.product` 枚举一致。这只复核抽象模型的组合，不验证设备可达性或原接口行为。

## key scale 跨 cache 与 QLI 的舍入边界

这是对上文 F11 key/scale cache 发布与 F12 QLI 消费之间的**数值约束**，不是额外的性能收益。冻结 `sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/indexer.py` SHA-256 `d5d6a438bbced4e63d729dac94c08869df30dab4a71ceacecdea211386b3db33`，非 packed `update_keys:119–124` 调用动态量化后将 scale 转为 FP16 写 cache；`select_projected:254–280` 将 cache scale 连同 INT8 key 和量化 query 传入 QLI。保存的 `agents/fusion/frozen-source/csrc/attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_service_vector_arch22.h` SHA-256 `fcddf4e682e5c742f1190dbe673da909bcdf745dbd6dc1cb66f700fa9b41b8a8` 的 `:827–845` 将所取 half key scale 转 float 并乘入对应位置的 score，随后进入候选降级和排序路径。已有 EP8 的 FP32→FP16 cast、两张 cache 和 QLI 参数 shape/dtype 与这条链相容，见上文；这些 task 没有绑定 storage、有效 slot、量化生成值、单条调用实例与 arch22 历史二进制。

因此，若研究将量化、cache 更新和后续 QLI 合并或用片上值转发，即使能够免除某个中间张量的物理发布，也不能无条件让 QLI 消费**未经过原 FP16 舍入的 FP32 scale**。可以在新链条中保留等效 FP16 转换再转回，并同时保留该 scale 与正确 INT8 key/slot 的匹配、对应页 stride、非 owned/重复槽语义、producer ready、原有持久 cache 的其他读者与生命周期；或者需证明更弱的数值验收合同。已有 scatter、量化域和 cross-layer 发布约束全部仍然适用。

以下是 **CPU NumPy / IEEE FP16 最近偶数舍入的接口级构造性反例**，未证明这些 scale 或 score 能由历史模型生成。两个候选的未缩放 score 均假设为 1，FP32 scale `[2^-27, 2^-26]` 及其反序经 FP16 舍入后均为 `[0,0]`，转回 FP32 并乘 score 仍为同一对零。如果跳过 FP16、直接使用 FP32 scale，两例的严格最大项分别是下标 1 和下标 0；任何仅依据相同原分数的固定 tie 决策，至少在其中一例与跳过舍入后的选择不同。这不需要假定 QLI 的实际 tie 偏向哪一项，也不证明实际选中结果改变。

```python
import numpy as np
s = np.array([2**-27, 2**-26], dtype=np.float32)
for v in (s, s[::-1]):
    rounded = v.astype(np.float16).astype(np.float32)
    print(rounded.tolist(), int(v.argmax()))
# [0.0, 0.0] 1
# [0.0, 0.0] 0
```

设备 FP16 转换、次正规数处理和实际量化 scale 下界均未以历史运行绑定；该例不能用于断言线上精度问题、实际 top-k 分歧或优化方案必然失败。如果输入域及误差验收能另外证明变化不影响结果，仍可研究更宽 scale 的替代方案。未操作 NPU或修改目标实现。

## QLI paged scale 读取：block table 与真实页 stride 是两项独立条件

这是上述 index-key/scale cache 到 F12 QLI 消费的布局约束，不是额外计数的融合收益。冻结 `sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/indexer.py:113–124,254–280`（SHA-256 `d5d6a438bbced4e63d729dac94c08869df30dab4a71ceacecdea211386b3db33`）在 non-packed 路径分别写 INT8 key 和 FP16 scale cache，之后将 `key_scale.squeeze(-1)` 传给 QLI，源码注释要求保留 Hybrid cache 的 page stride。保存的 arch22 `agents/fusion/frozen-source/csrc/attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_service_vector_arch22.h:157–201`（SHA-256 `fcddf4e682e5c742f1190dbe673da909bcdf745dbd6dc1cb66f700fa9b41b8a8`）的 PAGE_ATTENTION 分支按 batch 的 block table 获取每个物理 block ID，以非零 `keyDequantScaleStride0` 或为零时的 `kCacheBlockSize_` fallback 作为 axis0 跨页步长，逐 block 搬运 half scale；第一和最后 partial block 有各自偏移与长度处理。代码先将该 stride 提升到 int64，再与 block ID 相乘。这里是保存在本机的源码语义，并不证明实际调用使用该分支或历史二进制与其一致。

一个页映射的 CPU 整数反例：假设 FP16 scale 每个逻辑页 64 元素、真实 axis0 步长为 128 元素，某请求逻辑两个页的 block table 为 `[5,1]`，且 backing 数组在元素位置 `i` 保存 `i`。正确读取 128 个连续逻辑位置应为物理偏移 `640..703`，然后 `128..191`。如果只从第一页物理偏移连续读 128 个值，第二页起点误为 704；如果将 axis0 stride 按逻辑每页大小强行当作 64，第一页和第二页起点误为 320 和 64。正确 stride 也不能代替 block table：逻辑相邻页不保证物理相邻。可复算：

```python
B, stride, block_table = 64, 128, [5, 1]
backing = list(range(8 * stride))
correct = [backing[p * stride + j] for p in block_table for j in range(B)]
assert (correct[0], correct[63], correct[64], correct[-1]) == (640, 703, 128, 191)
assert backing[block_table[0] * stride + B] == 704
assert (block_table[0] * B, block_table[1] * B) == (320, 64)
```

步长单位这里是 FP16 **元素**，不是字节。作为另一个地址宽度条件例，block ID 17306、stride0=131072 元素时乘积为 2,268,332,032，大于 INT32_MAX；若错误截断并按 32 位补码有符号解释，则为 −2,026,635,264。这个例子不证明当前 kernel 存在溢出，保存源码恰好显式以 int64 算页乘积；也不证明该 stride、block ID 或足够大的有效分配在历史中存在。

先前 EP8/L2 的 QLI 参数包含 INT8 `[17307,64,1,128]` 与 FP16 `[17307,64,1]` 的 paged 形状，与此 source 角色相容；shape 无法证明实际物理 stride、块号、地址、共同版本、tensor 身份或历史执行分支。跨 producer cache 写、scale gather、score 乘法及候选输出的重组，必须保持真实 block table、各页 stride 和偏移、有效位置/尾块、key/scale 配对、cache 持久状态和可见性事件。若上游先根据真实页整理并发布满足 QLI 契约的布局，可以研究避免重复物化；这不证明现有接口支持该方案、不建立实际 HBM 事务或耗时。无 NPU 工作或目标实现变更。
