# F12：位置 / 序列长度关系的源码级条件证明

## 范围与结论

只读已保存源码和状态，未运行 NPU。对下述**相同一个真实请求、相同一次 attention metadata 构造**，若走 GPU `positions` 普通赋值路径、`query_start_loc` 保留该真实请求边界，且其后无其他修改，则无论 C 是零、已有前缀还是前一 prefill chunk 的累计值，对行号 `i` 都有 `p=L−Q+i`。在同一公共 metadata 上，DSA-CP 的 causal 连续切片也保持关系。此是对指定源码路径的推导，不是实际逐行 runtime 验证；尤其不能仅凭 launch 参数证明每次调用都走此路径，也不证明 arch22 源码与已执行算子二进制绑定。

## 普通路径

记请求已计算 token 数为 `C`，本轮 scheduler 分给它的真实 token 数为 `Q`。`agents/bounds/metadata-path-snapshot/ascend_runner.py:2351–2363` 按 input batch 请求顺序从 scheduler 取各 Q；`:1324–1334,1435–1437` 调用继承的 `_get_cumsum_and_arange` 并将累计和填 `query_start_loc`。该函数 `agents/bounds/metadata-path-snapshot/vllm_runner.py:1749–1775` 将每请求内部编号生成为 `0,...,Q−1`。对没有进入 DCP async rebuild 特殊赋值的路径，`ascend_runner.py:1505–1534,1595–1606` 用同一个 device `num_computed_tokens` 张量和同一调度 Q 生成 `positions=C+i` 和 `seq_lens=C+Q`，所以 `p=L−Q+i`。异步投机拒绝可能先修正 device C；只要走上述普通赋值，两处仍使用相同的修正后 device C，此推导不依赖 CPU 乐观值等于 device C。不证明被修正的 C 与正确真实请求状态一定一致。

普通 prefill、分块 prefill、普通 decode 只改变 C/Q，而不会破坏此代数关系。前缀命中路径 `agents/bounds/metadata-path-snapshot/vllm_scheduler.py:918–989,1259–1265` 可以将本地和外部命中数之和作为计算起点；`:1012–1014` 恢复已有请求使用已有起点。命中统计并不提供本次逐行 C，也不证明 cache 内容正确。`vllm_runner.py:1489–1491` 对已有 batch 项更新 CPU 计数；新请求等其他状态流不由该单段证明。关系仅要求普通路径实际使用的是它所使用的同一个 C，并非断言所有命中/请求的上游映射正确。

`ascend_runner.py:1060–1105` 的所示 padding 操作追加虚拟边界而不是改写已有真实边界；只对 `0<=i<Q` 的真实行推导。额外 dummy 请求/图捕获、未检查的 ubatch 切片与 DCP metadata 改写、CP async rebuild 的 `positions_ready_on_device` / CPU rebuilt 特殊分支，均排除在证明外。`:3655–3681` 传递 device seq_lens、query start、positions 给 common metadata；`dsa_v41.py:902–921` 读取该 common 的 positions 与 seq_lens。`:690–710,567–590` 的 attention 源路径从 layer metadata 取 positions 交给 indexer。还需实际证明所比较的 SWA positions、indexer cache 的 query start / length 来自同一个相应 request 与 slice，和数据未被其他路径改写，才能对具体历史 V3 调用套用等式。

## Causal DSA-CP 切片

`sources/vllm-workspace/vllm-ascend/vllm_ascend/attention/context_parallel/dsa_cp.py:1136–1240` 将平坦 token 轴按 rank 的连续区间分片、与各请求 query 区间相交，形成局部 query 长度；`dsa_v41_cp.py:118–150` 截到真实 token 末尾，positions 取同一连续切片，并在 causal 情况下从 device 全局 L 减去该请求落在本 rank 之后的 query 后缀长度。

设本请求全局 query 行是 `0,...,Q−1`，本 rank 与其相交部分为 `[a,b)`，非空，且原有 `p_j=L−Q+j`。此时 `Q_local=b−a`；要删除的后缀为 `Q−b`；源码给出 `L_local=L−(Q−b)`；局部第 `i` 行原坐标 `j=a+i`。因此 `L_local−Q_local+i = L−(Q−b)−(b−a)+i = L−Q+a+i = p_j`。实际要求这里没有额外重排、相交段是有效请求行，且同组索引元数据来源一致。此结论不覆盖 `is_noncausal=true`，后者明确令 suffix=0；例如 C=100、Q=6、局部切片 [2,4)，位置首行为 102，但 L−Q_local=104。这个反例只针对非 causal 逻辑解释，并非声称实际模型负载走非 causal；上述 common metadata 创建时显式设 `causal=True`。

同目录 `f12_position_invariant_check.py` 使用独立 Python 整数参考，穷举 573,600 个有限构造配置、1,705,920 个真实局部行及一个非 causal 反例；仅核对推导和边界示例，不执行产品代码、不代替形式化证明或历史运行时采样。

## 来源和边界

- 两份 runner 与 scheduler 的 SHA-256 及 stage/baseline 匹配见 `agents/bounds/metadata-path-snapshot-status.json`。CP `dsa_cp.py` SHA-256 `7e8dd93eeda988866138342523ecfa7cc9c42ce67669d4b1507ee8c5f3452305`、`dsa_v41_cp.py` `039ec9d18e0a8a4aa39f9d5b4bea15204cea3f397a3fce97a5097b266bc9822e`、`dsa_v41.py` `3505769857f83fa7e67f652984f50bd0478158f1d24f185cab091203e70392fd`。本次通过只读 `docker exec ... sha256sum` 核对与现 stage 容器源码一致，并非历史导入对象或 device 二进制证明。
- 现有 frozen serve 文件记录 `enable_dsa_cp=true`、异步调度和 DSpark；不能从配置推断每个请求执行了哪个分支。本证据尚未找到历史具体 p/L/Q/i 值；未核验 DCP 重建与 ubatch 特殊分支。历史操作如果与本源码、假设和身份条件相符，则本证明可收紧原 `F12-QLI-MASK-ROW-BOUND.md` 的数据缺口；当前不能报告历史所有行已满足，也不能据此删除过滤、交换 TopK 或给出提速数值。
