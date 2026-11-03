# Engram 非连续视图：一次真实负载中的逻辑复制账及其界限

## 结论

新诊断支持一个更具体的融合候选，但**不能据此得出模型 TPS 的数值上界，也没有证明物理 HBM 的最小流量或可实现收益**。这里把可核对的逻辑张量大小与不可证明的物理时间分开。

同一诊断测量窗口记录 8 个请求、262144 个输入 token、8192 个输出 token，前缀查询 262144、命中 235904（89.990234375%），8/8 输出与冻结结果匹配。observer 保存 1513 个唯一的有效 begin；其中 826 个具有唯一的、原方法已返回的匹配 end。1513 个输入均为 NPU int32 非连续视图，列数 24、stride `[48,1]`。包括 1462 个 `[24,24]`，其余 shape 的行数为 3208、3328、6416、6656、6672。详见 [聚合结果](../../diagnostic-20261029/binary-binding-draft/second-binding-run/engram-metadata-draft/minimal-recovery-draft/audit_live_metadata.json) 与 [独立审计](../fusion/ENGRAM-LOCAL-RECOMBINATION.md)。日志达到大小上限且丢记录，样本不能代表完整窗口。

## 条件下可复算的大小

假设 wrapper 调用的原方法与冻结的 [embedding.py](../../sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/engram/embedding.py) 对应、观察的参数在该方法的 `gathered.contiguous()` 处未被替换、记录与实际调用配对正确：每个 `[T,24]` int32 输入在原方法语义中都要求得到一个 `[T,24]` 连续张量，其逻辑大小为 `T×24×4` 字节。非连续性在开始调用处有直接观测。

- 826 个配对且原方法已返回的样本，因每个 `T≥24`，连续结果大小相加**至少** `826×24×24×4 = 1,903,104` 字节，约 1.815 MiB。
- 对全部 1513 个 begin，按 shape 分组计算的连续结果总逻辑大小是 **27,935,232** 字节，约 26.641 MiB；这仅是观察输入所对应的条件账单。由于部分 begin 无匹配 returned end，不能说所有复制已成功提交，更不能将其当作设备实际完成量或全窗口总量。脚本如下所示可由保存的聚合复算。

这些数字是**每次操作的逻辑张量大小之和**，不是彼此独立的物理内存地址：不同调用可以复用分配、共享缓存、使用图 replay 或经后端改写。源码中的调用也不证明每一次对应独立设备 kernel，或者数据确实通过 HBM 而非其他缓存。Python 方法返回不代表异步设备工作完成。observer 使用的 window 开闭是应用程序边界，不能自动与所有硬件工作完成窗口等同。

## 不可直接把账单当性能极限

一种宽松理想模型允许把 lookup 按 `[48,1]` 输入视图读取，在满足同一数值语义、读取时序和存储生命周期时，不必单独生成连续中间张量。该假设成立时，**中间张量的语义分配与复制可以被消除**，所以其逻辑字节量不能作为所有合法融合实现的必要工作、也不能用于收紧用户所要求的通用理论吞吐上界。

反过来，也不能宣称实际必然节约上述字节的 HBM 读和写、将该字节除以标称带宽作为收益，或将收益与其他融合候选直接相加。直接 stride 访问可能产生更大的事务放大、不同缓存命中、额外寻址或资源竞争；原始路径可能被编译器/运行时改写。完成等价实现还需证明 lookup 的真实后端、offset 与行 stride 寻址、异步读取完成与源缓冲复用的依赖、graph replay 和数值一致性，见独立审计。没有这些证据时，收益及变化后的必要访存均 UNKNOWN。

## 复算

只读聚合文件、仅做整数运算，不操作设备、不修改模型：

```sh
python3 - <<'PY'
import json
p='diagnostic-20261029/binary-binding-draft/second-binding-run/engram-metadata-draft/minimal-recovery-draft/audit_live_metadata.json'
a=json.load(open(p))
s=a['signatures']
assert a['unique_begin_valid_metadata']==1513 and a['unique_returned_pairs']==826
assert sum(x['count'] for x in s)==1513
assert all(x['dtype']=='torch.int32' and x['device']=='npu' and x['contiguous'] is False and x['stride']==[48,1] and x['shape'][1]==24 and x['shape'][0]>=24 for x in s)
print('all begins, conditional logical bytes:', sum(x['count']*x['shape'][0]*x['shape'][1]*4 for x in s))
print('returned pairs, minimum logical bytes:', a['unique_returned_pairs']*24*24*4)
PY
```

本记录的下界只适用于所述“已有原方法的已观测调用及其连续结果逻辑大小”条件，不是设备字节吞吐下界、输出延迟下界或待优化后实现的下界；理论上界数值仍需独立证明必要工作和对应硬资源容量。


## 配对样本按实有形状进一步核算（2026-10-29）

在相同配对、进程状态、run/window 和原源码绑定的条件下，逐条使用 826 个配对 begin 保存的 shape，而不是用每条至少 24 行估算：795 条为 24 行，9 条为 3208 行，9 条为 3328 行，3 条为 6416 行，4 条为 6656 行，6 条为 6672 行。每行 24 个 INT32 值，**所记录配对样本的连续结果逻辑大小之和是 15,725,568 字节（14.997 MiB）**。它是上述 1,903,104 字节保守下界的精化，不代表未记录的调用和完整窗口，也不是物理流量或设备执行完成量。输出及源码为 `agents/diagnosis/engram-paired-copy-account.json` 与 `agents/diagnosis/engram_paired_copy_account.py`；后者在 stage 容器中只读固定 SHA 的原始日志、状态和进度，以进程身份/调用 ID 配对并只输出计数，不操作 NPU。
