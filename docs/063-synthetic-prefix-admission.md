# SG 候选准入：前缀锁为何不一定释放预算

## 只是一项可复现的合成机制检验

针对G156本机候选源码，抽取并执行其页数、full预算、请求需求、独立池 `prealloc_fits`、FULL节点锁/解锁等真实Python方法，在CPU假对象上构造条件完全明确的三种情况；**未调用真实cache、NPU或推理服务，不证明G181现场运行了这份代码、观察到以下资源状态或根因是容量。**

统一假设：输入fill长度32768，缓存前缀16384、页128、每活跃请求reserve 512，共6条active；锁前free+evictable=28160，扣除active reserve后的full预算25088；可撤回token=16384。请求的候选full准入需求 `R=16896`。这些全是合成数字，不是正式运行快照；这里只测独立full池这一门，不含SWA、Mamba、metadata、请求槽、HiSparse、LoRA、设备分配或实际prefix match。

|此前已受保护的前缀槽|新锁定的可驱逐槽 K|锁前预算|锁后预算|需求 R|独立full门|解锁后预算|
|---:|---:|---:|---:|---:|---|---:|
|16384|0|25088|25088|16896|通过|25088|
|8192|8192|25088|16896|16896|通过（边界相等）|25088|
|0|16384|25088|8704|16896|不通过|25088|

同样的前缀长度与锁前free+evictable，因前缀中有多少槽**此前已受保护、多少刚由可驱逐变为受保护**，锁后可准入预算不同。不能据“有prefix命中”直接认定预算一定足够，也不能据这项模拟认定现场发生full门失败。

## 候选公式与适用条件

在候选的非HiSparse、相同full池、radix可驱逐项可计入及扣项非负的简化路径，`B = available + evictable − max(active_reserve, single_req_guard) − 其他扣项`。prefix match后先锁定，再重新计算预算；新锁K只表示本次从evictable转为protected的节点槽数，不必等于prefix长度。真实并发下其他状态可能同时变化，不能直接用本图中的 `B_after=B_before−K` 断言现场行为。

候选full需求是 `max(new_alloc + reserve, fill_len − prefix_len + min(max_new, CLIP) − retractable)`。页对齐时新分配页数为 `ceil(fill/page) − ceil(prefix/page)`；若fill、prefix均页对齐且其他条件不变，prefix较无命中所降低的需求为P，预算因新增锁而降低K，门槛余量的理想变化为 `P−K`。K=P、K=0、0<K<P对应完全抵消、理想全收益和部分收益；这三个条件不能从客户端C16或服务metrics推断。

一个合成helper反例：page128、fill256、prefix129时，候选 `get_num_new_pages` 给0槽，而 `ceil((256−129)/128)*128` 给128槽。此例只说明这两个数学表达式不同；候选radix路径可能先将prefix页对齐，未证明129是该分支真实可达输入。

## 复核与边界

[合成结果](../evidence/admission_063/synthetic.json) SHA `a8e40ee3e65a4cab660a65eeb712023e83a0702373cf59b993b4ae84a1a65080`；[验证器](../scripts/simulate_admission_063.py)以显式参数接收本机候选源码根路径，逐文件核SHA后提取方法AST并在假CPU环境中执行，普通和 `python -O` 均验证。该源码包不包含在此摘要中，缺源时只能核验结果SHA而不能复演AST；所有相对文件名与源码hash在结果内列明，无宿主路径或身份。

要确认现场瓶颈，仍需对应同一匿名attempt、锁前/锁后实际有效预算与需求、prefix新保护槽、其他资源门与head-of-line条件、阶段时钟和现场实际import；即使命中full门也还须验证其处于客户端关键路径。新观测、埋点或服务改变需要单独授权及资源隔离。
