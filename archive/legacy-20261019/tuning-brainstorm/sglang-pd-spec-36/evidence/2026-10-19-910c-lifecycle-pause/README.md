# 910C：自然结束、分块取消、同ID重试与暂停交接

按用户要求，完成当前R092后提交记录并暂停，等待新服务器。原GLM-5.3-W8A8C8、Ascend、PD、MTP EAGLE三步/topk1/draft4、TP16/EP16、BF16 KV与原角色graph配置保留；无生产源码补丁。整体仍E0，accepted_baseline=null，未建立稳定性能基线或优化A/B。

## R091：保留整体失败，分别记录自然EOS和上限结束

这是额外生命周期用例：原两条512-token输入，greedy、max_new_tokens=4096、ignore_eos=false，无人工停止词或logit bias。原512→128、32768→256基准未改变。冻结断言要求两条都自然生成EOS，此条件未满足，内层真实退出1，Run保留FAIL。

| 请求 | 实际输出 | 结束原因 | 可支持的结论 |
| --- | ---: | --- | --- |
| 0 | 4096 | length，无EOS | 正常达到输出上限；不能要求任意模型输出必在预算内自然结束 |
| 1 | 359 | stop，最后ID154827 | 第359个token为配置EOS，此后无多余输出 |

四个P/D响应均HTTP200，原prompt IDs逐个匹配，P首token等于D首token，MTP接受直方图与计数一致，最终两端ready与空闲flush均200。这是一个自然EOS分支及一个上限分支的实测，不能将原Run改写为PASS。原客户端scope字符串在失败时也会写入，因此scope不是结论；以实际响应、断言和失败回执为准。

## R092：真实分块取消、其余请求完成、同ID换房间重试

额外生命周期测试保留原三条32768输入、每条256输出。先单独提交victim的真实P/D请求；44日志确认首个4096-token分块、剩余28672 token后，向46客户端传递事件，再释放两条survivor的四个独立HTTP线程，并对两端发送唯一victim RID的取消。引擎使用RID前缀匹配，所以三个RID特意互不为前缀；没有abort_all或服务重启。

取消API两端均200，但通过判据还要求victim实际结束：P返回HTTP200、finish=abort、零输出token；D返回HTTP500，消息含同一RID、原bootstrap_room及明确的`Aborted by AbortReq.`。实际源码中接收器abort会标记KV传输失败，Decode将其转成500；这是冻结前允许的显式取消表示，不能作为普通生成成功HTTP200，也不是从任意500推断正确。

| 完成请求 | 输入/输出token | MTP verify | 正确/提出草稿 | 完整输出 |
| --- | --- | ---: | --- | --- |
| survivor-1 | 32768 / 256 | 91 | 164 / 273 | 与R090请求1哈希相同 |
| survivor-2 | 32768 / 256 | 100 | 156 / 300 | 与R090请求2哈希相同 |
| victim重试 | 32768 / 256 | 92 | 163 / 276 | 与R090请求0哈希相同 |

原victim与两个survivor全部响应结束后，复用victim的同一RID、换新bootstrap_room重试。取消、survivor与重试之间**没有主动flush**。三条完整结果逐个prompt ID及P/D边界匹配，全部768个输出token等于各自返回目标最大分数，接受直方图守恒，完整输出哈希与冻结R090参考相同。父进程wait确认客户端真实退出0；最后两端ready与空闲flush均200，原P067/D073精确CID仍运行。

测试通过的是一次事件驱动的分块取消、survivor隔离及同RID/新房间复用，不替代三条同时到达的性能用例。P前端的通用“waiting queue”取消文案无法标识精确内部队列；阶段依据是首个分块时仅victim已提交，随后P在完整prefill前以零输出终止。已有分块可能仍在执行，不能声称即时取消或从跨机器时钟相减推算延迟。

功能复用和最终空闲flush不等于物理KV逐字节/每页代际与引用数oracle，也不证明长期无泄漏、全部断连/重复取消场景或独立模型精度。累计目标分数一致性观察为2816个token（此前2048，本轮768），仍不是BF16独立金标准。所有时延只作生命周期诊断，不报告加速比。

## 续跑交接

本轮后不再派发新推理实验。记录原模型文件、源码、镜像与通信配置，以及未执行的graph/内存候选。新服务器必须重新盘点资源归属、校验模型/环境、建立新运行合同，再验证原长短PD＋MTP及生命周期；旧硬件结果不能直接成为新服务器性能基线。之后先冻结适合数值路径的质量判据与稳态A/A，再做单变量优化。36机制覆盖、硬件可达上界和端到端优化仍未完成。

- [迁移与暂停交接](notes/L039-server-migration-handoff.json)：包括最新服务处置状态；没有在公开文件保存凭据。
- [EOS失败的独立复核](notes/L039-eos-review.json)、[取消/复用独立复核](notes/L039-cancel-review.json)。
- [R091全部输出与失败视图](runs/L039-R091-pd-functional/public-view.json)、[R092完整输出、分数及退出视图](runs/L039-R092-pd-lifecycle/public-view.json)。
- [原始数据SHA及视图变换说明](public-view-sources.json)：回执SHA对应原始数据，公开视图另命名；未删失败、输出或计时样本。
- [40条冻结命令各执行一次的元数据审计](notes/L039-group-memory-zcode-audit.json)，不公开ZCode执行器完整模型I/O。
- [上个检查点：原三路32K两次完成及实际容量边界](../2026-10-19-910c-target-scores-long/README.md)。
