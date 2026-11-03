# Run 001：末层残差、MTP 缓冲与输出头的多消费者边界

## 直接结论

已冻结的 G1 将 `target.final` 合为 `HC_COLLAPSE_NORM_LM_HEAD`，未在该节点内列出 `mtp_hidden_buffer` 这个条件消费者；G5 的 12 个模板也没有独立展开目标输出头。源码证明：当 `_mtp_hidden_buffer` 存在时，最后一层残差在计算最终 collapse 前还要复制到此缓冲，后续可通过 `get_mtp_target_hidden_states()` 暴露。因此，任何围绕最终 collapse 的重组都必须保留这一分支及其可见结果，不能把末层残差视为只供最终输出头消费。这是源码条件分支事实，**不证明该分支在冻结样本实际执行**。

依据：`sources/vllm-workspace/vllm-ascend/vllm_ascend/models/deepseek_v41/model.py`：

- 975–986：仅当 speculative config 存在且 `use_eagle()` 或 `uses_draft_model()`，并处于 PP last rank，才分配缓冲。基线存在 5-token speculative decode 并不单独证明这个函数返回值和实际 PP 状态。
- 1592–1607：若有缓冲，SP 模式先分别 all-gather `hidden_states` 与 `pre_mix`，然后将 `hidden_states.flatten(1)` 复制到缓冲；之后计算 collapse；没有缓冲的 SP 分支则在 collapse 后 all-gather；再归一化并返回。
- 854–856：collapse 是 `pre_mix.unsqueeze(-1) * x.float()`，沿 HC 轴求和，转换回原 dtype。改变转换、归约及舍入顺序可能改变数值。
- 1776–1783、1798–1801：`compute_logits` 消费返回的 hidden；MTP accessor 可返回复制缓冲。这里仅识别接口内已知消费者，不宣称 runner 所有间接调用与跨请求生命周期已全部获知。

## 待复核的结构机会（均非已实现或收益证明）

1. **F-Tail-01，collapse 内部的转换、乘法、HC 轴归约、原 dtype 转换**：可作为一个源级组合检查对象，关联原始 4 个操作。需要保持 FP32 中间值、归约次序和目标 dtype 舍入语义；数学等价和降低内存流量仍需证明。当前图未给底层实现；已有编译器或硬件融合情况 UNKNOWN。
2. **F-Tail-02，collapse 输出与最终 RMSNorm**：有直接生产消费数据边，可登记为有条件跨边界机会。collapse 原 dtype 转换是语义边界，不得默认省掉；`return_hidden` 是可见输出，不能未经证明只生成 logits 而删去它。SP 且无 MTP buffer 时中间有 all-gather；其通信与布局语义必须保留。
3. **F-Tail-03，源残差并行分支**：MTP buffer 开启时，残差还流向 flatten/copy，SP 时先 gather；与 collapse 共用源值是多消费者机会，必须保证 MTP buffer 的字节值、有效长度、生命周期，以及 collapse 和归一化输出。source 并不证明可以移除 buffer 写，也不证明 runner 的全部下游。若 MTP buffer 关闭，此分支不存在。

三项是带交叠的结构检查对象，不能累加所谓节省；任何 side effect、全体消费者、精度和同步未证时状态 **CONDITIONAL / UNKNOWN**。针对本案的实际 SP/MTP 分支、每次有效 shape、张量别名和 runner 间缓存生存期需要额外源码或 trace 支持，不能由图推断。

## 可复现扫描与覆盖边界

只读源输入、CPU 写分析目录：

```sh
cd /Users/wio/work/169-vllm-ascend-reproduce-172/analysis/operator-ceiling-v2-20261028/operator-ceiling-agent-kit/runs/169-vllm-v41
python3 agents/fusion/scan_final_boundary.py
```

脚本根据 source 记录四种 SP/MTP 布尔分支，各自枚举连通的 2、3、4 节点集合：关闭/关闭 11/11/12，关闭/开启 13/14/16，开启/关闭 12/12/13，开启/开启 15/16/19。数的是结构集合（包括外部边界和状态复制节点），**不是可融合项数，也不能相加成独立收益**。候选详细图及输入、脚本 SHA-256 在 `final-boundary-structural-scan.json`。条件采样、资源容量、运行时 graph lowering、完整所有消费者未覆盖；不构成全图穷尽。

本次只写本目录；没有访问其他主机、修改目标代码或启动 NPU。
