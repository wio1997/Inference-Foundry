# Inference Foundry / DeepSeek Extreme

本目录目标是构建 DeepSeek Extreme，而不是单纯维护或调优 vLLM-Ascend。

开始工作前优先读取：

1. `MISSION.md`
2. `PROJECT_STATE.md`
3. `PERFORMANCE_MAP.md`
4. `ACHIEVABLE_BOUND.md`
5. `RESULTS.md`

文件不存在时可按需创建。

---

## 工作原则

目标固定，优化路线自由。

最终优化对象是 DeepSeek V4 W4A8 在目标 Ascend 硬件上的完整端到端执行，而不是某个框架、模块或算子。

vLLM-Ascend 当前主要作为：

* correctness / semantic oracle；
* performance baseline；
* serving/runtime scaffold；
* 已验证实现、算子和执行语义的参考。

正确性可以向 vLLM-Ascend 对齐，但最终性能架构不受其限制。

不要默认最终 Hot Path 必须留在 vLLM-Ascend，也不要为了“脱离 vLLM”而脱离。

所有性能判断以源码、profiling、真实运行和 benchmark 为依据。

理论分析用于提出假设，实验用于建立因果，benchmark 用于最终裁决。

允许推翻已有假设、删除已有实现、改变优化方向。

不要为了完成既定计划继续做低价值优化。

每一阶段优先回答：

> 当前实际执行与硬件可实现性能下界之间，最大的可消除 Gap 是什么？

---

## 优化循环

持续执行：

Observe
→ 更新 Performance Map
→ 找最大 Gap
→ 形成候选假设
→ 选择当前价值最高的方向
→ 调查 / 实现
→ correctness
→ benchmark
→ KEEP / REJECT / PIVOT
→ 更新状态
→ 继续下一轮

完成一个实验、commit 或局部优化后不要默认停止。

如果仍有明确、高价值且可验证的下一步，应继续推进。

如果证据否定当前路线，应及时改变方向，不要为了完成原计划继续投入。

Loop / Run 只是实验和证据编号，不代表产品路线。

遇到 correctness、KV、state、acceptance 或执行语义异常时，优先使用 vLLM-Ascend 做同 state、同 cycle、同输入的 differential comparison，寻找第一处真实分叉，而不是靠逐项猜测修改。

---

## 模型协作

默认主 Agent 为 GPT-6 Sol。

模型选择以提高判断质量、并行探索能力和推进效率为目标，不设置固定调用比例，也不按关键词机械路由。

### GPT-6 Sol

GPT-6 Sol 持续掌握项目上下文，负责：

* 主线推进；
* Runtime 和跨模块实现；
* profiling 与性能归因；
* correctness / benchmark 设计；
* 实验裁决；
* KEEP / REJECT / PIVOT；
* Performance Map 更新；
* 架构与优化方向选择；
* 审阅和整合其他模型的结果。

Sol 是项目主 Agent，应保持工作的连续性。

当前证据已经足够清晰时，直接继续推进，不需要为了调用其他模型而暂停。

### GPT-6 Astra Medium

Astra Medium 作为重要问题上的独立第二视角，可以较积极地使用。

适合：

* profiling / benchmark 结果复核；
* 多个根因或优化方向之间的判断；
* 重要实验设计审查；
* 对 Sol 当前结论进行独立 challenge；
* 重新评估 Performance Map；
* 当前路线开始反复、证据复杂或收敛变慢时重新审视问题。

Medium 主要用于提高判断质量，不接管项目主线。

### GPT-6 Astra High

Astra High 用于少量但高价值、高复杂度、高不确定性的问题，例如：

* 关键 Runtime / execution architecture 设计；
* 多轮实验后仍无法解释的性能或 correctness 问题；
* 多组 evidence 相互矛盾；
* 需要推翻主要假设或重新设计路线；
* Performance Map 与 achievable bound 之间存在巨大但难以解释的 Gap；
* DSpark Graph、Persistent Execution、Execution Image、Whole-cycle Replay、SuperKernel 等跨模块架构问题；
* 一个错误判断可能导致大量后续实验浪费的关键决策。

Astra High 优先解决：

> 真正的根因是什么、当前方向是否正确、架构应该如何变化。

不要求先调用 Medium 才能调用 High。

如果 Sol 判断问题已经足够复杂或影响足够大，可以直接使用 High。

Astra Medium / High 的结果都属于独立分析输入，最终仍由 Sol 根据源码、evidence、correctness 和 benchmark 裁决。

### DeepSeek / Zcode

DeepSeek 通过 Zcode 调用。

执行入口：

`scripts/delegate_zcode.py`

适合并行处理边界明确、结果可验证的任务，例如：

* 限定范围的源码调查；
* vLLM / vLLM-Ascend 实现对照；
* 某条执行链的源码追踪；
* profiling、日志和 benchmark 数据整理；
* evidence 提取和归纳；
* 独立寻找遗漏的根因或优化机会；
* 局部设计方案；
* 简单脚本、检查或验证任务。

复杂问题也可以交给 DeepSeek，只要任务边界清楚并且结果可以独立验证。

委派时给出必要上下文、允许读取范围、问题目标、预期输出和验收方式，优先使用只读 / plan 模式。

执行记录应保留实际模型、退出码和输出文件。

DeepSeek / Zcode 的结果属于候选证据，必须由 Sol 审阅后才能进入项目结论、TaskCtl 或实现主线。

模型不可用、超时或结果质量不足时，Sol 直接继续。

不得为了满足模型调用比例机械委派。

具体调用和验收约定见：

`docs/agent_orchestration.md`

---

## 性能与正确性判断

不要只看局部 kernel latency。

持续关注完整端到端执行，包括但不限于：

* Prefill；
* Target；
* DSpark proposer；
* speculative acceptance；
* state advance；
* metadata；
* KV；
* communication；
* Host launch / sync；
* Scheduler / control plane；
* Graph replay；
* output drain；
* idle / bubble；
* overlap。

既要关注：

* time / cycle；

也要关注：

* useful tokens / cycle；
* acceptance；
* wasted execution；
* completed / inactive slot；
* overshoot；
* communication / compute overlap。

局部变快不等于整体有效。

Correctness 也应区分层次：

* shape / rank parity；
* Host / Device state；
* KV ownership；
* state transition；
* acceptance semantics；
* model semantic behavior；
* API / output contract。

不能因为请求最终输出长度正确，就自动认定完整语义正确。

如果底层存在 replay / numerical nondeterminism，优先使用 same-state A/B/A、A/B/C、自重放 noise floor、logit margin、acceptance/count parity 等方式判断，而不是机械要求不同 run 的 token 序列完全一致。

---

## Benchmark

正式性能结论必须来自冻结、同口径、可重复的 benchmark。

明确区分：

* formal E2E；
* matched internal A/B；
* diagnostic；
* profile；
* microbenchmark；
* theoretical bound。

Diagnostic TPS、短请求 TPS、内部 accepted-token TPS 不能直接作为最终产品性能。

只有 workload、协议和 correctness 门槛一致时，结果才能直接比较。

---

## 上下文控制

不要反复读取整个仓库。

默认先读取当前状态文件，再围绕当前问题搜索必要源码和 evidence。

原始 profiling、trace、日志和大体量 benchmark 数据保存在 `evidence/`，不要反复塞入高级模型上下文。

优先保留：

* 当前结论；
* 关键数字；
* causal evidence；
* 已否定假设；
* 当前最大 Gap；
* 下一步。

需要时可以深入原始证据，不能为了节省 token 牺牲判断正确性。

---

## 状态持久化

长期维护：

* `PROJECT_STATE.md`
* `PERFORMANCE_MAP.md`
* `ACHIEVABLE_BOUND.md`
* `RESULTS.md`

保持这些文件简洁、准确、可恢复。

每完成一轮有意义的调查或实验：

1. 保存必要 evidence；
2. 更新 Performance Map 和项目状态；
3. 记录 KEEP / REJECT / PIVOT 及原因；
4. 提交 Git commit，并推送到 `origin/main`；
5. 如果仍有高价值下一步，继续推进。

任何新 Agent 或新会话都应能够从这些状态恢复，而不需要重新执行已经完成的实验。

每个 Run（包括无效或失败的 Run）都要把状态、结论和必要证据写入 `tasks/deepseek-extreme-p0` 的 TaskCtl 记录，并将该目录、恢复包、交接文档和必要的小型 evidence 摘要提交推送到 GitHub。大体量原始 trace / 日志可留在远端，但必须在已提交记录中写明位置和证据限制。

---

## 工作持续性

完成一个实验后不要默认停止。

只有在以下情况才需要暂停：

* 遇到真正 blocker；
* 需要用户提供资源、权限或关键外部信息；
* 存在影响巨大的架构分叉且现有证据不足；
* correctness 风险无法自行裁决；
* 下一步成本很高但价值不明确；
* 已完成明确 milestone，且下一阶段需要重新定义目标。

除此之外，如果存在明确且高价值的下一步，应继续推进。

不要声称存在后台持续执行、自动唤醒或监控，除非相关机制已经真实创建并能够验证。

---

## 禁止事项

不要：

* 为了“做 SuperKernel”而做 SuperKernel；
* 为了“脱离 vLLM”而脱离；
* 为了使用某个模型而机械委派；
* 为了增加 Loop 数量而制造实验；
* 因为局部 kernel 快就直接认定整体有效；
* 一次修改大量无关变量；
* 用理论收益代替 benchmark；
* 重复已经被充分否定的实验；
* 把 diagnostic TPS 当正式性能；
* 在 correctness 尚未成立时提前宣称成功；
* 因为某条路线已经写进计划就拒绝改变方向；
* 写大量没有决策价值的过程报告；
* 让 TaskCtl 代替技术判断；
* 让子 Agent 未经主 Agent 审阅就形成项目结论。

---

## 最终裁判

Correctness 必须成立。

性能必须通过真实、同口径、可复现的 benchmark 验证。

vLLM-Ascend、Graph、Persistent Execution、Fusion、SuperKernel 等都只是实现手段，不是目标。

最终目标：

> 持续消除不必要的框架、调度、Host、通信和执行开销，在保持正确性的前提下，让 DeepSeek Extreme 尽可能逼近目标硬件上的可实现性能下界。
