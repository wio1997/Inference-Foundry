# C2 ring 槽位依赖：可复算反例

从本任务根目录执行：

```sh
python3 agents/fusion/reproduce_compressor_ring_conflict.py
```

脚本只用 Python 标准库和精确有理数，并检查被冻结 `compressor_triton.py` 的 SHA。保存输出为 `compressor-ring-conflict-witness.json`；脚本对关键索引、同一地址和池化结果都有断言。它不运行模型、NPU、profiler 或测试容器，也不修改任何服务。

## 假设的输入

一条合法的示例请求：`start_pos=1`、`used_len=32`、输入段起点 `seg_row_base=0`，`max_query_len=32`、ratio=2、每页 ring 32 行、`HEAD_DIM=512`。指定有效物理 page 为 1，总 page 数设为 2，所以满足有效性检查 `0 < cache_row < NUM_CACHE_BLOCKS`。这些数只是构造源码路径的例子，**没有证据表明 169 实际采样出现了它们**。

## 对照冻结源码计算

所有行号均指 `sources/vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/compressor/compressor_triton.py`，SHA 为 `1c26f996fa91bcd3ae497e1ae33e88a2e0bf38c2e734f8af85baee06492a87ee`：

1. `647–704`：当前 `compressor_from_projected` 路径校验 FP32 kv、scores、ring 和 BF16 out，并以 `RATIO=2`、`CACHE_SIZE=32`、`TOKEN_ALIGNED=True` 启动 `_pool_kernel`。`groups=(32+1)//2=16`。
2. `230–249`：请求中完成组数为 `(1+32)//2 - 1//2 = 16`。对 `local_group=0`，`group_idx=1//2+0=0`、`residual=1−0×2=1`，进入 `_pooled_blocked` 的历史残余分支。
3. `110–128,158–195`：组内 token 位置为 `[0,1]`，相对 segment 的偏移是 `[-1,0]`。token 0 在 segment 之前，读取历史 ring 行 `0 % 32 = 0`；token 1 从当前 projection 第 0 行读取。`310–314` 把 pooled 写入 token 对齐输出第 `0+(0+1)×2−1−1=0` 行。
4. `681–683,706–718`：cache 更新的写槽位上限是 `min(32,32)=32`。`348–364` 对 `write_idx=31`，`tail_count=32`，`token_in_seg=32−32+31=31`、`token_pos=1+31=32`，写回 `32 % 32 = 0`。score 与 kv 都写入同一个物理 page 的对应分量。
5. 更具体地，对 channel 0、page 1，历史 kv 读取及新 kv 写回的 FP32 元素偏移都是 `(1×32+0)×2×512=32768`；历史 score 与新 score 的偏移都是 `32768+512=33280`。这是**完全相同地址**，不是仅相邻或指向同一 buffer。
6. `683` 的注释及 `684–719` 的同流 launch 顺序明确要求旧值先被 pool 读到，再执行 ring 更新。这个顺序是当前实现采用的保护；某种未来重组可以用更细粒度的先读后写关系替代，不必据此断言所有 pool 和所有写操作永远不可并行。

## 数值区分

将两个相关 score 全设为精确的 0，softmax 权重各为 1/2。举两个 channel：旧历史 kv 是 `[2,3]`、token 1 的当前 kv 是 `[4,3]`，token 32 当前 kv 是 `[10,3]`。按源码先读历史时得到 pooled `[3,3]`；如果 token 32 提前覆盖 slot 0，再读取则得到 `[7,3]`。这些整数在 BF16 中均精确表示，因此差异发生在 norm 之前，无须假设一套浮点近似或误差阈值。

其余 510 个 channel 可取 0。这里只证明一个满足所列条件的源码级输入会改变 pooled 值；**不主张它在当前基线真实发生，也不据此推导输出 token、性能收益或强制全局串行**。实际实现的 metadata 可达性、跨页别名和运行时调用仍需分别证明。
