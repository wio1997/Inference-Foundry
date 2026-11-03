# Index prepare：按行切分跨长度分支的语义边界

本记录针对已有 F12 长度处理候选补充一个源码级反例；不代表 169 历史调用已命中该输入或已测到性能差异。已有长度使用者和 1023 行条件见 `../fusion/F12-CROSS-LAYER-INDEX-PUBLICATION-REVIEW.md`、`../fusion/F12-PREPARE-SEMANTICS-AND-REORDER.md`；本次未重新采样设备。

冻结 helper `../fusion/frozen-source/vllm_ascend/ops/triton/prepare_indexer_indices.py` 的 SHA-256 为 `c5e8e475c560fb5187d2f51e2b88384fdeebd69a5cee5bd931992a818f577197`。当 K=512 且传入非空 `lengths_output`，行数 1023、1024 分别是 523776、524288 元素，满足 `<= 1024*512`，先排序时对原 `valid` 直接求和；1025 行为 524800 元素，不满足条件，改对最终 `output>=0` 再求和。没有长度输出时没有这两个长度分支的对比。不能仅依据 `INDEXER-ATTENTION-EVENT-IDENTITY.md` 的 1023×512 形状证明历史这次调用传入了长度缓冲、实际 helper 二进制和 branch。

考虑每行 `selected=[INT32_MAX,-1,...,-1]`，K=512，`positions` 为 INT64、值为 INT32_MAX，`compress_ratio=1`。在源码数学语义和 INT64 加法不溢出的前提下，`visible=INT32_MAX+1`，第一项通过 `(selected>=0)&(selected<visible)`，融合计数是 1；但同一个 `INT32_MAX` 也是 `SENTINEL`，排序/反编码后被替换为 -1，因此 fallback 的 `output>=0` 计数为 0。所有其余位置是 -1，也变成 sentinel 后输出 -1。本例不涉及排序顺序歧义。这个极端数值符合 helper 显式的 dtype、形状和 ratio 断言，但尚未证明满足该模型完整调用契约、官方算子约束或实际运行时有此输入；它不是对某个真实设备 bug 的断言。

由此，对原 1025 行 fallback 调用按行拆分（如 1024+1）使各块都走融合长度分支，并非**对 helper 全部显式允许的输入**必然等价。一个足够排除上述反例的条件是每个有效 selected 严格小于 INT32_MAX，并且 padding 列也不能被 valid 计数；另需检查 position 范围、输入与输出别名、每行独立、存储偏移/contiguous 要求、图捕获固定地址与流/缓冲生命周期等。对 K=512，无 padding 列；而对非 2 次幂 K，kernel 的 `valid` 在加载被 mask 掉的 sentinel 后不再显式叠加列 mask，所以还须单独验证 padded 列不会被计入。上述条件是分析前提，不声称穷尽完整正确性条件。

下面是可复算的 CPU 整数语义模型，非 Triton 编译或 NPU 执行：

```python
K = 512
selected = [2**31 - 1] + [-1] * (K - 1)
position = 2**31 - 1  # 假设为 INT64
visible = (position + 1) // 1
valid = [0 <= x < visible for x in selected]
output = [x if v and x != 2**31 - 1 else -1 for x, v in zip(selected, valid)]
assert (sum(valid), sum(x >= 0 for x in output)) == (1, 0)
assert [(r, r * K <= 1024 * 512) for r in (1023, 1024, 1025)] == [
    (1023, True), (1024, True), (1025, False)
]
```

历史候选中非 packed 主 attention 调用没有显式传入 topk_lengths，packed 路径则会使用该长度；具体调用及共享状态的所有消费者、别名、图副作用仍需证明。对于已选 1023 行的条件路径，长度计算本来就与排序融合，不能假称切分消除一次 sum；对于更大行数的分支，性能收益及 fallback 是否真实出现也没有测量依据。PMU 含义和同次调用绑定的限制见 `INDEX-PREPARATION-PMU-REVIEW.md`，该反例不能解释高 scalar 字段或建立数值性能界。
