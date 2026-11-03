# Loop-011：C2 pool → norm 的状态消费者与 tile 依赖

本报告新增一个可手算的读写冲突反例和分块约束；它不证明反例在冻结采样中发生，也不证明融合可行或有收益。源码和输入 SHA 见 `compressor-engram-input-hashes.json`，本地检索仅覆盖当前已快照源码，隐式别名、外部插件、运行期 hook、编译器生成物和其他未快照调用仍未知。

## 数据、状态和零行

目标 ratio-2 分支 `dsa_v41.py:523-532` 生成 `kv` 和 `score` 两个 FP32 张量，调用 `compressor.pool_projected`；`compressor.py:55-67` 令 token 数为 `T=kv.shape[0]`，把预分配 `_ring_pooled[:T]`、FP32 state、5 行 INT32 metadata 传给 `compressor_from_projected`，然后对全部返回行做 `self.norm`。当前快照中 `_ring_pooled` 的显式引用是初始化和本处传入；没有据此宣称外部 hook/alias 不存在。

`compressor_triton.py:675-720` 先清零所有 T 行；当 tokens、batch 或 max_query_len 为 0 时直接返回零 buffer，调用者仍执行 `self.norm`。非空时 pool kernel 只把**完整 pair 的结束 token 行**写 BF16 pooled，其余行由初始化提供；cache-update kernel 再写环状态。norm 结果随后 `update_keys(latent, ...)`，然后才对 latent 原地 RoPE 并写 long KV（`dsa_v41.py:544-564`）。所以：

- 新方案要处理全部需要的行，而不仅是 pool 完成行。假设有限 gamma 和有效正 epsilon 时零行通常规范化为零，但必须确认真实 RMSNorm 细节和权重值；对 NaN/Inf 或特殊精度，直接跳过零行的等价性不可从普通情形推出。
- latent 的先 RoPE 前 index key 和后 RoPE 的 long KV 是不同语义版本，并有后续 layer 和请求的持久消费者；state ring 的更新也会影响未来 chunk，不能因为本次立即返回 latent 就省掉。
- 可能的 pool+norm 组合应保留 FP32 pool 到 BF16 的原有中间舍入，然后按原 RMSNorm 的实现精度和规约顺序生成输出。归一化输出地址与 ring pooled 的原始稳定地址是否能改变，取决于 graph、别名和消费者契约；不是源码级唯一性结论。

## 一个具体不可乱序的例子

设某个有效请求 `start_pos=1`、`used_len=32`，环长度 32。第一个 ratio-2 组包含历史 token 0 和新 chunk 第一个 token 1。`_pooled_blocked:101-128,169-193` 读取历史 token 0 的 ring slot `0 % 32 = 0`；该请求末尾新 token 32 由 `_cache_update_kernel:348-364` 写到 `32 % 32 = 0`。如果提前写 token 32，pool 读到的就是新值，改变结果。

例如对任一 channel，两个 score 均为 0，旧历史 kv=2、token 1 kv=4；正确的 softmax 均分 pooling 为 3。如果 token 32 提前把 slot 0 写成 kv=10 且 score 仍为 0，则错误 pooling 为 7。这只是允许的输入与地址构造，不声称真实请求的值或此冲突已观测；它说明不能把整个 ring 写与 pool 读视为独立、无条件提前覆盖。源码现有同流次序 `pool` → `cache_update` 是一个充分的保护方式。

理论上按不同 request/page/slot 分块并行，须先证明每个被覆盖的 ring slot 的所有历史读者已经完成、页别名和回收不存在冲突、同一页内 token 位置与有效掩码正确；另一组读写可重叠的可能性不等于所有 slot 都可重叠。同一个 batch 中多个请求如果指向同一物理 page，也须依据真实分配不变量判断，不能仅看 request index 不同就认定无冲突。

## 可选重组的估算口径

如果在同一 tile 保留 BF16 舍入与完整 RMSNorm 语义，且原先每行 `[512]` BF16 pooled 确实写入、又供 norm 从同一存储层读取，则中间边界的两次逻辑访问最多按 `2 × T × 512 × 2 = 2048T` 字节计。这里将 T 定义为本次 pool 的 token 对齐行数，是条件性会计上限，**不是已测 HBM 字节、必省流量、时间下界或速度提升**：这只计至多一次 pooled 写和一次 norm 读，不包括当前独立的 `out.zero_()` 写入；不能把这两次视为已经发生的 HBM 流量或断言它们全部可省。零行、缓存命中、片上转发、融合后新读写及分块竞争都会改变实际结果；如果实际 norm 或 pool 已被其他机制融合，该口径也不成立。

## 实际执行路径的证明方式

单看目标层的通用 `RmsNorm`、`MatMulV2` 或 `ZerosLike` 名称和同一 layer 时间窗不能确定 pool→norm：一层有其他同名操作。已有 16 rank CSV 是按操作名称与层窗口聚合，不提供 pooled storage identity，也无法区分 token-aligned 分支。可以优先在既有 rank EP8 的 SQLite 按 kernel 名、图 model/task/stream、调用元数据、输入 shape 做有界查询；若找到名字，只能说明相应记录存在，只有在同一次 graph replay、输入/输出 identity 或可靠 dependency 与源码调用位置都可绑定时才能声称真实执行该数据链。已有 profiled DB 若缺必要字段，应留 UNKNOWN，而不是补做无目的 NPU 测试。

待主控的 ZCode 只读查询 `run_compressor_trace_inventory.py` 完成后，将在本节补充具体验证与局限。未经核验不能从查不到名称推断没有执行，也不能把 replay 复用的 globalTaskId 视为唯一 invocation。

### 只读查询结果（stage 096）

ZCode `operator-ceiling169-v2-096-compressor-inventory` 退出码 0、verified `deepseek-flash`；`compressor-trace-inventory.json` 保存 DB SHA、查询 SHA、名称/shape 聚合及 join 审计。主控侧已重算查询脚本 SHA 一致。只针对 EP8 一份现有 profiler 数据库：

- 有 `_pool_kernel` 和 `_cache_update_kernel` 的 host 及设备记录；两类各有 3 个标记输入行数分别 12、6400、6668、宽度 512 的设备元数据分组，另有 input shape 为 N/A 的记录。存在 shape 对应 `[T,512]` 和 `[17307,32,1024]` state 的 pool 元数据，和 ratio-2 源码所预期接口一致。这表明采样中有相应名称的工作，但名称/shape 不能证明某个记录就是 F18 的某一次特定 source layer、与相应 norm 连接或具有某个 tensor 地址。
- 例如 `_pool_kernel` 在 `N/A` 组有 21 条 TASK 行但只有 3 个 distinct globalTaskId；整库 TASK 有 3964 个 globalTaskId 复用组。`COMPUTE_TASK_INFO` 的 globalTaskId 重复组为 0 只保证 join 元数据行不因该表重复而倍增，**不证明该 ID 是唯一运行次数、也不证明 metadata 的 shape 是每个 replay 的实时 shape**。因此均报告数据库行，不声称它们是独立请求或 kernel invocation 数。
- 相关 `RmsNorm` 名称和 `[T,512]` shape 的记录存在；同类名称来自多处，不能仅凭它们和 pool 在同一文件、相近时间或 shape 匹配，就断言同一 pooled tensor 的读写链。仍缺 tensor/storage identity 和可证执行依赖。
- 该 DB 亦有 `_engram_host_uva_gather_dequant_kernel` 与 `_engram_req_index_kernel` 名称记录，与 F24 的 NPU Triton 路径一致；这些记录不是 CPU 上直接查表的证据，也不能单凭名称断言具体 Engram 层、每个请求或事件等待开销。

这些是旧 trace 的只读取证，不影响其他任务；既有数据不足以给 F18 确切设备耗时、HBM 流量或融合收益。

## 条件反例：不能仅用两个输入 token 的 RMS 统计提前得到 pooled 行的 RMS（离线数学验证）

在已冻结源码的 ratio-2、token-aligned 路径中，pool 按通道组合两位置的 FP32 `kv` 和 `score`，将完成行先转 BF16（`compressor_triton.py:85–195,285–314,675–704`）；`compressor.py:18–22,55–64` 随后以 `config.rms_norm_eps` 对这一 BF16 完成行运行 `RMSNorm`。配置快照的 `head_dim=512`、`rms_norm_eps=1e-20`（`evidence/model-config.json`）。这里的反例针对「把 pool 与 norm 融合/分块时，仅保留各输入行独立平方和，提前求 pooled 行的 norm 统计」：即使两位置每个通道的 score 都为零，各通道权重皆为 1/2，输入各行平方和也不足以决定输出行的平方和。

取长度 512 的有限 FP32 向量，未写出的元素均为 0；两种输入均有各自行平方和 `(1,1)`：

- A：`a=(1,0,...)`, `b=(1,0,...)`，池化结果 `p=(1,0,...)`，`sum(p²)=1`；
- B：`a=(1,0,...)`, `b=(0,1,...)`，池化结果 `p=(0.5,0.5,...)`，`sum(p²)=0.5`。

输出值 0、0.5、1 能精确表示为 BF16；以实数 RMS 模型和配置 epsilon 求出的倒均方根因子分别约为 22.627416998 与 32，若令示例 gamma 恒为 1，则前两维分别为 `(22.627416998, 0)` 和 `(16,16)`。两输入的各自平方和相同，而 `a·b` 分别为 1 和 0。对于这个等权特例，`sum(((a+b)/2)²)=(sum(a²)+2 sum(a*b)+sum(b²))/4`，交叉信息必须以点积或其他等价信息保留；真实 score 是逐通道的，因此不能把本特例中的单一等权公式直接用作一般路径。下例纯 Python 标准库可复算，不模拟设备内核逐步舍入或 BF16 RMS 输出：

```bash
python3 - <<'PY'
import math
D, eps = 512, 1e-20
a = [1.0] + [0.0] * (D - 1)
for label, b in [('A', a.copy()), ('B', [0.0, 1.0] + [0.0] * (D - 2))]:
    p = [(x + y) / 2 for x, y in zip(a, b)]
    s = sum(x*x for x in p)
    inv = 1 / math.sqrt(s / D + eps)
    print(label, 'input_sumsq', sum(x*x for x in a), sum(x*x for x in b),
          'dot', sum(x*y for x, y in zip(a, b)), 'pool_sumsq', s,
          'rms_inv', inv, 'normalized_first2', [x*inv for x in p[:2]])
PY
```

**适用边界：**这是独立数学反例，输入向量、全零 score 和单位 gamma 未绑定到历史请求，也未证明能由实际 Wkv/Wgate 联合投影产生；不是历史结果错误、硬件 bug、已测吞吐或收益证明。它限制只用单行 RMS 标量代替 pool 后 RMS 的重排；并不要求先将完整 pooled 张量写入 HBM，也不禁止按通道完成 pool 后累积平方和、保留或重算所需片段，再按完整行统计输出。融合仍需遵守原 BF16 中间舍入、RMS 实际算子精度及原有 ring 与消费者约束。

冻结文件 SHA-256：`compressor_triton.py` `1c26f996fa91bcd3ae497e1ae33e88a2e0bf38c2e734f8af85baee06492a87ee`；`compressor.py` `31516e5ba234e34fa01e73e864fbc88bd6473d4a5274a1d43da8a79bab3cf494`；`evidence/model-config.json` `43d3b2bc55626893d59275b32e04068c0fc916f5c382eb91de73bdbade9c88f7`。以上文件均位于本次 run 根目录的 `sources/vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/compressor/`、`sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/` 和 `evidence/` 下。
