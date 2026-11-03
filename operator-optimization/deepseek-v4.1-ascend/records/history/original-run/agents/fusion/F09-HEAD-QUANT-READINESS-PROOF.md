# F09：Indexer query 按 head 分块的就绪边界

范围：对已有 F09（投影→部分 RoPE→query 动态量化）补充可证伪的数据依赖与分块粒度；不是新增一个可单独计算收益的融合项目。只读现存源码及摘要、在 CPU 上计算数学反例；没有操作 NPU、修改推理实现或声称历史调用已绑定。

## 源码中的实际条件路径

- 已保存的 `sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/indexer.py:167–179`，`project_query` 按 `(n_heads,width)` 展开投影，`apply_query_rope` 原地处理部分维度；非 packed 的 `quantize_query` 走 `quantize_indexer_query`，packed 走不同实现。本分析只适用于前者。
- `agents/fusion/frozen-source/vllm_ascend/ops/triton/quantize_indexer_query.py:23–37,40–69`（SHA-256 `6065b79f0f764e99bafae71347c8c8dfd0625ad52b0d695639c47e71338d0bba`）要求输入 `[tokens, heads, 128]`，先 contiguous，把 token×head 展成行，每行 128 个维度取 abs-max，除以 127，将 scale 转 FP16 再转回 FP32，最低取 `2^-24`；再用该 scale 做逐元素除法、nearbyint、截取 `[-127,127]`、写 INT8，并写对应 FP16 scale。这里讨论的是源码数学和数据依赖，不等于历史实际已走该分支；设备实现、浮点异常及 tensor identity 仍须绑定。

## 必要依赖与反例

每个 head 的最终 scale 取决于**该 head 全部 128 个经 RoPE 的值**。可以随上游 tile 到达增量计算 max，但仅看到前一段时一般不能确定全 head 的最终 scale，也就不能无条件把使用该 scale 的最终 INT8 子块交给消费者。

可复算反例：两个 128 维输入的前 64 项完全相同：第一项 1，其他为 0；后 64 项分别为全 0、以及末尾 4 其余为 0。按所存 helper 的 FP16 scale 顺序以及其最近偶数意图，在 CPU 上第一项分别量化为 127 和 32：

```python
import struct
f16 = lambda x: struct.unpack('e', struct.pack('e', x))[0]
for absmax in (1.0, 4.0):
    scale = max(f16(absmax / 127.0), 2.0**-24)
    print(scale, min(127, max(-127, round(1.0 / scale))))
# 0.00787353515625 127
# 0.031494140625 32
```

两个输入的前半完全一致而正确输出不同，因而任何只看前半、未获得后半相关信息且要一次发布最终值的方案，都不能对这两种输入同时正确。此反例中的数均可精确表示为 BF16；没有使用特别的 tie 或非有限值。它是数据依赖证明，不是设备执行实验，`div_rn` 的设备细节仍须按照安装版本契约核实。本证明不禁止缓存局部结果后重算、使用已知的额外上界提前确定结果、或采用经证明语义等价的其他算法；它否定的是无条件地以局部 max 替代完整 head max。

## 哪些边界可以并行、哪些尚未证明

- 该 helper 的 max 和 scale **按 head 而不是跨全部 token/head 归约**，因此在已经完成该 head 所需投影 K 归约及 RoPE 的前提下，可以研究独立准备该 head 的量化结果，而无需在数学上等待其他 head 的 max；但现有下游 QLI 是否提供/支持 head 级逐块提交，实际上游是否能暴露该就绪点，stream 事件、buffer 可见性和其代价均未知。不能由此声称现成接口可以流水执行或估计收益。
- RoPE 覆盖的通道及其配对数据、投影的完整 K 归约、原位写后读取与中间舍入都要先满足；完整 head 可在临时存储、寄存器/片上缓存或其他方式保存，源码依赖不强制必须通过 HBM 物化。若改变计算或规约顺序，还要按实际精度合同验证结果；遇零 head、NaN/Inf、截断和 FP16 scale 下溢等情形应保留原语义。
- 所存 helper 将输出行每 16 行分一个 block，并注释目的是每核写完整的 **32 字节 FP16 scale 组**，以避免部分输出 cache line 竞争。分块融合若让不同执行单元写同一 16-scale 组，需确认具体平台写入粒度、非重叠写和同步；该注释不是完整硬件内存模型，也不能直接宣称任意两个 scale 写必然数据竞争。尾块、buffer 对齐和图复用尤其需要核实。
- 条件算术：若一个目标调用实为 3 token、每 token 32 head 且宽 128，则为 96 个归约行、6 个 16 行 scale 组、理论输出存储容量 96×128 INT8 字节及 96×2 FP16 字节。旧 trace 的 `[3,4096]` 与该投影形状相容，**未证明**此调用进入该 helper、张量身份相同、实际搬运量或设备时间；容量不是实测流量，也不是吞吐上界。

下一步仅在有既存 trace/tensor 或配置证据时核实是否命中 non-packed helper、scale 写与消费者接口及对应生命周期，再判定这个更细粒度能否被实现。不能把它与 F05/F09 已有候选重复计收益，更不能以这条依赖或 96 行假设推断性能瓶颈。
