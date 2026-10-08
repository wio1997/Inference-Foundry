# 图尺寸与 P prefill MC2：静态证据及实验边界

本节仅说明参数设计和已安装 Python 源码的静态行为，不给出本轮吞吐、延迟、SLA通过或性能提升结论。若运行有效性、成功率、缓存状态及实际配置核验通过，才可使用对应性能结果评估候选。

**执行修订**：以下保留实验前静态设计。P开关单独启用在profile阶段因400 MB通信缓冲区不足失败；随后修订为MC2＋HCCL_BUFFSIZE=4096 MB，预算12288保持。最终实测结果见[REPORT.md](REPORT.md)。

## 目标与独立步骤

目标为 native 稳态 input throughput ≥278,625.26 token/s、TTFT P50 <5s、TPOT P50 <18ms，并保留其余四项 SLA 门槛；full 结果应独立披露。Input throughput 若包含缓存命中 token，应与新计算 token 吞吐区分。测试计划为 C64/N384，服务输入80,012 token、输出600 token，名义共享前缀93%；需核对实际成功请求长度和缓存指标。

两个候选按先后顺序分别测试：

1. **D 图尺寸候选**：只将 D 的 capture sizes 指定为 [6,12,…,192] 共32档。P 保持原配置；D 保持 DP16/TP2/EP32、MTP5、maxseq32、budget4096、FULL_DECODE_ONLY 及既有通信开关。此候选本身不改变 D fused MC2。
2. **P prefill MC2 候选**：在图尺寸结果作出保留或恢复决定后，仅将 P 的 `enable_prefill_mc2` 请求设置为 true，P budget仍为12,288。P 保持 DP4/TP16/EP64、MTP1、eager及既有 FlashComm1/shared-expert-DP/DSA-CP设置；D 应冻结在明确记录的状态。若 D 图候选保留，P 变更应与该图候选运行比较；若恢复 D，则应与相应恢复状态比较。

这两步不得合并归因为某个单独开关的收益；若运行间还有其他有效参数或代码变化，应另行披露。

## D 图尺寸静态证据

已观察到启动的 pre-worker 默认尺寸为 [1,2,4] 加8的倍数直到192。已安装 V1 worker 会针对 MTP5 的 uniform query length6，对尺寸向上取6的倍数并去重。在该路径及图模式保持不变的条件下，默认最终尺寸推导为：

`[6,12,18,24,36,42,48,60,66,72,84,90,96,108,114,120,132,138,144,156,162,168,180,186,192]`

默认25档已可通过 padding 覆盖1–32个 uniform q6 decode 请求。Dense32档仅补足下列精确形状：

| decode请求数 | 实际q6 token | 默认图token | dense图token | 少padding token |
|---:|---:|---:|---:|---:|
|5|30|36|30|6|
|9|54|60|54|6|
|13|78|84|78|6|
|17|102|108|102|6|
|21|126|132|126|6|
|25|150|156|150|6|
|29|174|180|174|6|

其余 uniform 请求数没有这项 padding 差异，常见每rank2–4请求对应的12/18/24 token已在默认集合。若上述缺失形状在真实调度中频繁且 padding 成本明显，dense尺寸可能有价值；全局并发均值不能证明每rank形状频率。多7档也可能增加捕获时间、图内存或stream资源。Capture完成及服务就绪仍应独立验证。

q6可被TP2整除，静态尺寸调整未见新增对齐冲突。主模型图捕获循环使用 worker resolve 后的 descriptors，并把同一resolved尺寸集合交给 target/draft GraphParams。专用 draft proposer 的具体继承/内部行为若尚未完整取证，应保留该限制，不将 base class 片段视为所有draft分支的证明。

**q1问题不解决**：真实首decode若 query length1而不是uniform q6，在 FULL_DECODE_ONLY 的独立 uniform图路径中仍不具备该资格；dense q6列表不改变 eligibility。若某DP rank选择NONE，既有全DP模式同步行为也不会因增加图档位而改变。

## P 普通 prefill MC2 静态边界

`enable_prefill_mc2` 是 P 的普通 MC2候选，不能等同于 D 的 `enable_fused_mc2` 或宣称开启了 D fused算子。已安装 Python A3 selector中，普通 MC2未见EP≤32的限制；fused dispatch的EP限制属于另一个分支。Python无显式禁止不等于已验证编译算子的EP64支持、workspace边界或正确性。

如果当前设备/模型实际使用该A3 selector、EP已启用且P fused MC2保持off，则预算12,288/TP16会先产生768 token/rank，再由普通MC2的512 token/rank上限限制到8,192总token；超过所选容量的batch会选ALLTOALL。因此12,288是调度预算，不能写成MC2本身支持12,288 token。保留该预算可保留调度余量，无需单凭MC2容量上限将预算回退。若effective fused/设备/模型调用路径不同，须按实际分支重新核验。

FlashComm1与PCP8同时启用时，安装代码要求budget按TP×PCP=128对齐；12,288已可整除128。配置层未见该组合的显式冲突，但MC2 prepare/finalize与FlashComm1/DSA-CP实际张量布局、mask及量化模型callsite仍需正确性验证；配置通过本身不足以证明运行兼容或性能收益。若出现正确性/服务失败，应先保留证据并按明确候选状态处理，不能把失败数据计为有效性能结果。

## 历史对照与因果限制

D 图候选使用历史 C64/N384 A运行 `diag_c64_cb4bdad1715f` 作为参考。即便请求设置和源码身份可核对一致，两轮时间不同，且未在候选当时重复该对照；缓存/温度/调度形状/系统状态漂移仍未被排除。若差异很小，单轮历史对照不能证明统计显著或将变化完全归因于dense尺寸。P后续单轮筛选亦不证明最优配置。达到或未达到目标应以完整有效结果及六项门槛分别报告，静态证据不预判结论。

静态出处为已安装源码：`config/compilation.py`1466–1474及1503–1548；`v1/cudagraph_dispatcher.py`72–91、132–156、207–230、272–324；Ascend `worker/model_runner_v1.py`5169–5209；`ascend_forward_context.py`255–317及414–430；`ascend_config.py`109–152及177–185。此摘要仅引用模块与行号，不包含内部路径、主机地址、凭据或源码全文。
