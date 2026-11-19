# 32K P 动态量化：CPU 投递所在 step 标记

## 可证的局部结果

从[同源候选链](050-p-dynamic-quant-queue-chain.md)继续，分别只读SG和vLLM的 P rank0设备CSV和原trace，按同一个capture、rank、进程及线程，对目标动态量化的完整CPU enqueue区间寻找完全包含它的 `ph=X, cat=cpu_op, name=step[...]` 标记。没有跨机时间相减；以Decimal处理源数值；按0、1、多候选保留，不按最近时间强配。原trace完整事件仅在授权主机内存中解析，聚合输出不含原始ID、绝对时间或请求内容。

在 SG 的单次诊断中，120/120个目标签名的CPU enqueue各有唯一同线程包含标记：**112个归入 EXTEND 标记，8个归入 DRAFT 标记**。采用[事后100µs事件分界](048-p-dynamic-quant-subgroups.md)：80个慢事件的CPU投递都在 EXTEND；40个快事件中32个在 EXTEND、8个在 DRAFT。不能把此结果等同设备事件在CPU区间内完成、等同模型层调用、或判定用户请求及端到端根因。

trace中共16个可解析 SG-style标记：8个EXTEND、8个DRAFT。逐标记聚合后，**8个 EXTEND 各自包含14个目标CPU enqueue候选，其中4快、10慢；8个DRAFT各包含1快、0慢**。这表明候选现象分布在该采集的所有8个已记录EXTEND标记内，而不是仅一个被截取窗口；8个step同属一次profile，并不是独立重复实验。

SG EXTEND 标记自述 `bs=1,toks=8192`，`c_sk` 记录为8192、16384、24576、32768等；它是标签中的CPU侧长度聚合字段，**不是逐请求真实物理KV占用测量**。原记录各段CPU span约1.36–1.61秒，其duration是CPU marker范围，可能包含等待与异步提交、并不等于P NPU独占执行或可直接相加的请求时间。不要用本窗口事件数乘时长推TTFT收益。

vLLM P rank0在这次trace中按相同 `step[...]` 命名规则未找到可匹配标记，399/399目标CPU enqueue无这种同线程包含候选；**仅说明SG式标记未在此筛选中观察到，不代表vLLM没有forward、没有其他命名标记，或399个设备事件没有运行。** 两框架并未配对同一逻辑step或请求。

## 证据及限制

- [stage分类](../evidence/p_step_051/stage_counts.json) SHA `ddeaf429af978d170ea4a7fb480b35e24c9d127dae04f2b12dabe60ed528baa9`；[逐标记匿名分布](../evidence/p_step_051/step_detail.json) SHA `2ce7a374cdc738efcdcf64bd2d2e2fe813fdcfd4c4da4d5fa74068d2a51156d0`。
- 设备CSV与trace的source SHA、候选链、队列分位和每标记计数与前次独立读取一致；SG为G240冻结源，vLLM为G148派生分析副本而非原始冻结CSV/trace。
- 两进程只读分析，P节点资源门检查后执行、无推理请求；I/O PSI不可用，不声明其影响为零。窗口标记由历史profiling产生，不能替代请求贯穿ID、模型层标识、图重放映射和端到端对照。
- `c_sk`、`toks`、`bs`来自标记文本；该聚合保留序号和数值，不发布forward ID/进程号/线程号/绝对时间。模型实现源码对step命名提供静态线索，但未与历史G240现场逐文件绑定，不能升级为运行期命中证据。

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_p_step_051.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_archive.py
```

[复核脚本](../scripts/verify_p_step_051.py)检查同源SHA、唯一/未知的分母、逐标记快慢计数和原结果一致性；不验证设备与请求的因果关联。
