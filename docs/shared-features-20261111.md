# 2026-11-12：同卡组限定里程碑已核验

当前权威状态见[开发状态](../STATUS.zh-CN.md)与[脱敏摘要](../validation/shared-features-20261111/summary.json)。原生82/参考81均在die[0,1,2,3,4,6,10,14]、TP8/DP1/PP1/EP8，同一镜像、权重、共享配置和AISBench负载下完成。原生C4/C8为197.95191/275.71708 token/s，参考为177.40808/257.49225；本批+11.58%/+7.08%，未达30%。

源提交6b00f78已通过80请求与8rank正常退出；活动缓存解析器640/640、37次prefix命中及真实logprob输出均有回执。两档每档32成功、1024入/256出，原始时间戳独立重算通过。完整文本56/64相同；不据此宣称通用质量或稳定性能通过。

两方保持5draft/eager draft、FULL target、DSA-CP与其依赖SP MoE、AIV、shared-expert overlap、static/super、async、prefix/align；HCCL400、task queue1、每rank4GiB、context2048、maxseq8、batchtokens1792。原生仍内部提供实际sampled-token logprobs，参考HTTP不请求此项；没有向参考添加额外工作来降低其速度。

旧g79测量完成但退出失败、旧卡组结果和此前不公平KDA对比都保留为历史，不能覆盖本节或与本次卡组混算。下一阶段不改变精度、模型层数或正式负载来凑收益。

---

## 以下为历史状态

# 最新参考与实验源码范围

最新状态以[当前开发状态](../STATUS.zh-CN.md)和[脱敏摘要](../validation/shared-features-20261111/summary.json)为准。独有≥30%未达成，不能把旧约1.7倍或计分完成但退出失败的候选写成验收通过。

当前新卡组为8die/TP8/EP8、物理卡0/1/2/3/5/7；活动die [0,1,2,3,4,6,10,14]。这不是8物理卡。vLLM与后续DeepInfer须使用同一映射；旧卡组[0,1,2,4,6,8,10,14]的成绩单列，不混算。

固定vLLM core0.30.0+empty、Ascend0.30.0rc2 g8fde6d4e3、aarch64/910C、CANN9.2、W8A8/BF16和既有镜像。共有配置保持5draft/eager draft、FULL target、DSA-CP及依赖的FlashComm/SP MoE、AIV、shared-expert overlap、static/super、async、prefix/align；HCCL400、task queue1、4GiB cache/rank、context2048、maxseq8、batch tokens1792。fused MC2与shared-expert overlap互斥，选择后者，不声称全局最优已穷尽。

新参考C4/C8为177.40808/257.49225 token/s，各32成功、0失败、1024入/256出。原始SQLite重算与官方统计一致；仅各一批。新原生测试待可用资源，不存在新卡组原生收益结论。

新增[原生logprob取行](../DeepInfer/deepinfer/model_executor/models/glm5_next/full_model/native_mtp_logprob.py)只跳过raw词表scatter重建，保留既有采样、拒绝采样、实际token概率和rank；raw模式以外回退。原生C++输出接口需要真实概率，因此没有填零或伪造输出。旧卡组一次真实NPU同输入检查通过，256-token计分完成，但重复drain使退出失败。源码修复与缓存解析修复的最终硬件验证未完成。

启动选项`--native-mtp-logprob-gather`默认关闭；本轮候选显式开启。DeepInfer仍拥有请求生命周期、C++ scheduler与Worker，SDK提供计算/采样/缓存组件；没有用SDK引擎代理原生请求。记录的局部或工程检查不能代表通用模型质量。

---

## 以下为历史记录，不能覆盖以上当前状态

# 当前共有配置前沿与源码候选

**独有≥30%尚未证实。** 此前约1.7倍受KDA捕获差异影响；后来的子集测试也未控制最终HCCL/task queue环境。历史数字保留，不能据此声明完整公平收益。

## 当前实测参考

固定vLLM core0.30.0+empty / Ascend0.30.0rc2 g8fde6d4e3，910C/aarch64、8die/TP8/EP8、W8A8/BF16、每rank缓存4GiB、context2048、maxseq8、批token预算1792。共同开启5draft/eager draft、DSA-CP、依赖的FlashComm/SP MoE、FULL target、static/super、async及prefix/align。HCCL_BUFFSIZE=400、TASK_QUEUE_ENABLE=1、AIV、V1 runner均记录为实际有效值。SDK将请求捕获尺寸[6,12,24,48]按TP对齐调整为[24,48]，双方须遵循同一解析规则。

fused MC2与shared-expert overlap在固定SDK中互斥，按模型指南选择后者；不声称两者同时打开或穷尽全局最优配置。

|并发|vLLM输出token/s|TTFT中位ms|TPOT中位ms|DeepInfer至少30%所需token/s|
|---|---:|---:|---:|---:|
|C4|171.7722|1463.71|12.59|223.3039|
|C8|251.3519|1165.38|12.67|326.7574|

每档32请求、1024入/256出，2次排除的预热；0失败。只有一批每档，没有足够噪声证据声明稳定优势。当前只有参考分数，不能与旧原生结果混算倍率。固定工程用例通过不代表通用模型质量通过。

## 修复与源码范围

DeepInfer仍拥有原生引擎、C++调度、Worker与请求生命周期，SDK提供模型计算、设备采样与物理缓存。新增代码位于[模型适配目录](../DeepInfer/deepinfer/model_executor/models/glm5_next/full_model/)。

- 单draft原生短请求曾通过；5draft在旧平台代码下因EP通信缓冲实际只有16MiB、小于算子要求30,867,456字节而在预热失败，无性能分数。
- [平台初始化](../DeepInfer/deepinfer/platform.py)已修复为保留显式HCCL与task queue值；[计算运行时](../DeepInfer/deepinfer/model_executor/models/glm5_next/full_model/sdk_runtime.py)在加载权重前拒绝环境漂移，记录实际共有能力及KDA投影契约。
- 共有5draft、EP、DSA-CP、static/super为显式选项；未伪造draft、权重或已加载集合。索引加载只选择原严格加载器需要的分片，属于共有启动优化。
- 参考首次加载与图捕获完成后，被旧门禁错误要求SP关闭而中止。DSA-CP实际依赖SP；修正门禁后保持全部推理参数与helper字节不变，完成本表测试。该编排失败保留，不冒称模型故障。
- 原生当前候选源码`3999fcbc283404faeb149c458ceea7f71bf09af5`尚在同配置控制测试。该提交的静态检查与元数据检查不冒充NPU功能或性能验收。

独有融合仅为隔离草稿，不在本提交中声称成功。开启共有SP后，旧GEMM/all-reduce草稿不适用，不能通过关闭参考SP配合它。

[脱敏实测摘要与证据哈希](../validation/shared-features-20261111/summary.json)。原始请求、数据库、日志和本机路径保留在受控任务材料中，不公开凭据、服务器地址、权重或私有路径。

---

## 历史记录与旧数据（保留原数字，不作当前目标验收）

# 原生共同优化接入与公平对照状态

此前约1.7倍来自DeepInfer开启KDA捕获而参考保留KDA eager边界，不能作为双方同优化的框架优势。旧数据保留，原验收结论不适用于新的公平对照目标。

## 架构及源码

DeepInfer继续拥有原生LLMEngine、C++ scheduler、Worker与请求回调。固定SDK仅负责模型计算、物理缓存与设备采样；没有创建vLLM引擎或请求调度器。

- `sdk_schedule.py`：请求hasher、真实前缀命中、延迟发布；现有原生分块检查点保存中间递推状态。
- `sdk_async_schedule.py`：按固定SDK的计数语义管理最多两个在途批次、输出占位和缓存释放。
- `sdk_async_dispatch.py`：下一步计算先提交，随后回收上一输出；不猜测设备上尚未回到主机的token。
- `kda_graph_capture.py`：使用SDK稳定索引缓冲区修复align模式捕获；参考与原生使用同一文件，不能算独有收益。

已通过探测的源码提交：`5ba306cfd29839bb6f80b996ecb51ab01b2f1e8a`。
相关源码位于[模型接入目录](../DeepInfer/deepinfer/model_executor/models/glm5_next/full_model/)。

## 已实测范围

相同8die/TP8/DP1/PP1/EPoff、W8A8/BF16。更强参考开启async、prefix/align、PIECEWISE/breakable与同一KDA捕获，每rank缓存4GiB：C4为123.78067421 token/s，C8为196.88095174 token/s。每档32请求、1024入/256出、2次排除的预热；64条完整输出与旧参考一致。原始SQLite计时独立重算通过。

原生集成探测完成6请求，前缀实际命中640 token；28次设备token反馈，23次观察到回收上一输出时下一步NPU仍未完成。43个提交批次全部回收，8rank exit0，固定输出通过。EOS感知请求当步回收；声明的greedy/ignoreEOS负载实际使用两步重叠。这是共同能力接入验证，不是性能收益验收。

复用同一镜像及权重，原生CLI新增`--prefix-caching --async-scheduling`，与已有`--graph-decode --breakable-graph --kda-decode-capture`一起显式开启。独立重建仍需合法权重、固定SDK环境和DeepInfer本地扩展，未声称干净环境复现已通过。

## 波动及当前边界

参考每档仅一批，不据此声明稳定框架收益；旧原生g47的预热、缓存容量和共同优化不同，不能与新参考直接计算公平倍率。新原生正式C4/C8仍在执行，当前没有独有≥30%收益。

FlashComm1/SP-MoE在固定SDK的TP8/DP1/EPoff下不能激活。FULL、MTP组合及其他8die拓扑仍待审查；现场权重含MTP层，固定Ascend源也包含Glm5NextMTP，不能以缺少实现排除它。正式通用质量、长上下文及独立重建仍未验收。

[脱敏结果与证据哈希](../validation/shared-features-20261111/summary.json)。历史报告保留原数字，但其约1.7倍不满足新目标。


## 原生正式结果与已修复失败

g58首次正式交错批次因暂停请求重新入批时缺少token历史，在SDK读取all_token_ids时触发KeyError，未计有效分数。提交52c38617aba8779f4e6e2d458fd9791b9d5c00fa只补齐这一调用契约，g59在相同负载与共同开关下完成正式测试。

|并发|DeepInfer token/s|vLLM token/s|本批倍率|
|---|---:|---:|---:|
|C4|104.55924195|123.78067421|0.844714|
|C8|203.70500240|196.88095174|1.034661|

64条完整输出与参考一致，原始SQLite计时独立重算通过；72个原生请求完成、8rank exit0且占用释放。原生实际观察到3556次先提交下一步计算再回收上一步输出，其中3542次下一NPU任务尚未完成；3644个提交批次全部回收。

该批C4落后约15.5%，C8领先约3.5%，不满足独有≥30%目标。参考C8有9个请求出现超过1秒的输出间隔，最大约4.36秒；原生该档没有超过1秒的间隔。长尾原因尚未确认，每档仅一批，不能声明稳定框架优势。

共同开关子集已对齐，但不是所有执行工作完全相同：原生桥目前还请求内部采样token的logprob（logprobs=0），HTTP参考未请求此项，其开销未测。FULL/MTP组合与其他8die拓扑仍开放，当前对照不是全部共有技术已穷尽的最终基线。下一项是当前执行路径的有界CPU/NPU采样，收益结论必须来自后续正式对照。
