# checkpoint176：MTP 图的 DCP metadata 生命周期

目标 Decode FULL 已开启。原 GLM proposer 强制 MTP eager；当前唯一研究方向是在 H6/H5 上支持受限 greedy K1 MTP 单层 ACL graph。没有新增代码级性能 KEEP，H11 尚未进入研究栈；正式 Current、完整 API、80K/600/93% SLA 仍开放。

Run272 的实际 capture 成功，但正常请求 replay 合同未通过。亲查全部16×2 witness，唯一不同字段是 DCPContext.block_table 地址，两次 runtime 地址也不同。两条2334/8请求 exact，却均为 eager fallback，不能算 graph 数值正确性或收益。完整 natural23 EOS 请求未执行。首个失败、raw/spec保留，终态 H6/H5/mode0、all16 P/D healthy/idle；没有额外 recovery reload。归约见 Run272/correctness_reduced.json，native raw SHA b17e25b3…。

真实源码显示，FULL `_adjust_tensor` 的 F.pad 分配临时表，SFA builder 先保留其 local view，proposer 之后才 copy/bind persistent clone。dummy 第0步又使用 runner 原表。最小 v5 patch 在 dummy 第0步和 real metadata builder 之前更新并绑定已有持久表，保留逻辑请求数、列步长、全部 pointer guard、owned output 与 eager 回退；同 ptr/stride 时跳过 self-copy，其它模型/K/路径保持原行为。

最终 proposer SHA9bd8d2fc…、observer6a1355c8…、runner4c90054a…。实际容器 Torch CPU + SFA local-view AST 的五个独立 padded 输入、无 padding、alias、容量拒绝均通过；strict实际context/proxy/observer16rank/9fallback/4dummy也通过。第一次 CPU missing snapshot dependency raw保留，补齐源后通过，不计作 NPU Run。独立 final diff review 无具体源码阻塞。

唯一 Run273 已于09:32:39.542012Z 启动，controller2537213/start311116750、spec47819c7c…/52pins；normal四固定小PD correctness请求，至多一次owned失败恢复/warm，不profile、不扫描。P249保护，H6/H5固定。当前实际状态以 Run273 state/active_epoch/guards 为准；截至09:37:15Z 正在加载，还无新 device replay/complete/性能结论。Reset见 RESET_GLM_K1_STABLE_DCP.md。

matched性能脚本仍为条件草稿。审查发现并修复 terminal health失败后的旧POSITIVE误晋升风险；最终有效裁决在warm/witness/guards之后发布。严格 native cache collector检查有效delta，并匹配四个完整请求 P/D query/hit工作量。固定一个graph priming warm保证A1也取得本轮新transition；normal10请求、failure-inclusive11，至多一次恢复。实际workflow/recovery AST CPU反例确认 terminal warm/witness/guard/recovery/measurement失败不会晋升。必须等Run273真实correctness独立归约通过，才冻结新matched Run。

AISBench80K/600/93%与每轮salt、分端点cache delta/GitHub同步已完成；当前小诊断不是93%正式缓存合同。继续推进图 replay 与完整PD正确性，成功后做同resident A/B/A/B，保留已有有效H6/H5，不因正式SLA未完成清空研究栈。
