# Target metadata 生产者与 FULL 图交付

## 介绍

沿 V1 `_build_attention_metadata` 的原方法核对 target 侧 CP/global/local 数据、packet 生产者、空 query 和 FULL 图读者。冻结范围为 910C、CANN 9.2.0、TP8 DP2 EP16 PP1、DCP1、DSACP、K5 异步调度。

## 到了哪里

- **整轮组装。** 正常路径先在 runner 第 3534 行取修正后的 CPU 最大长度。Capture 在这一处使用配置上限，后面仍交付 CPU 镜像的 view。按组浅拷贝 common，保留每组 block/slot 和 padding 覆写。跨组共享本帧 batch 字典及其 `cp_global` 子字典，物理 slot 字典每组重建。Global owner 供 cache，local owner 供 Q。目标 gid12 的 CM、raw table/slot setter 与 unpadding 发生在其 builder 之前。GDN、DCP、encoder-only、MM prefix、offload 与 ubatch 的分支按原入口保留，冻结判断用 A3/DSACP 选中腿。
- **首个发布者。** 在 13 组、26 builder 的源码计划上执行 runner、CP/base 组装和 executor 的原 Python 控制。分组按完整 spec 与原层序派发，固定缓冲和 native 回调用 stand-in。六个任务的生产者如下。层号从零开始。

| 交付 | 首个生产者 | 事件阶段 |
| --- | --- | --- |
| C2 ring | gid1 global，layer 2 compressor state | COMPRESSOR |
| SMLA C2 | gid0 local，layer 2 long KV | ATTENTION |
| SMLA C1 | gid0 local，layer 20 long KV | ATTENTION |
| SMLA C0 | gid2 local，layer 0 SWA | ATTENTION |
| QLI C2 | gid0 local，layer 2 index K | INDEXER |
| QLI C1 | gid0 local，layer 20 index K | INDEXER |

- **空本地 query。** 三个 A3 SMLA closure 清各自的完整 1024 个 INT32 字，不调用 SMLA metadata op。两个 A3 QLI closure 仍调用原入口，global C2 保留其缓存状态任务。FULL 的同 descriptor 仍注册六个事件键，删掉已登记任务会被 `submit` 拒绝。两组同 rank 的合成 FULL 几何用于检查有无本地 query 的生产者切换；实际 dispatcher 接受哪些混合形状需按原捕获族核对。
- **FULL 读者。** 通用 runner selector 在当前 MLA、非 sparse 的参数下选择 `AscendMLABackend`；V4.1 各层的 cache 绑定选择 `AscendDSAV41CPImpl`。前者传给 ACL 包装器。原 `use_updatable_graph` 只对 `AscendAttentionBackend` 子类成立，MLA 走固定地址 replay，ENPU 只改变前置同步分支。Dense 控制会进入 context update，保留其 provider 合同。Breakable wrapper 回放有自己的同步及分段入口。逐层 V4.1 从当前层 SWA、source KV、index K、C2 state 的 prefix 取值，target 侧不使用 `mtp.*` 的三项。新读取的 MLA、updatable graph 和 breakable 源码身份均来自同一冻结容器。
- **目标与草稿分离。** Target 的 gid12 最晚构建，`smla:c0` 已由 gid2 发布，gid12 命中已有 packet。Actual DSpark 使用自身的 builder 与本轮新生成的 query/length，仍读原 gid12 物理 view。按冻结计划，仅在 runner 内成对略过 target 的 build/take 后，五对控制保留全部 51 个 target cache 条目、其符号 storage、6 个生产事件键和 13 次物理 setter。三条 `mtp.*` 的 target 阶段条目消失，真实草稿入口仍取得原 CM 与物理表。失败后只保留 take 会有旧任务风险。

## 结果

| 状态 | 范围 | 判断 |
| --- | --- | --- |
| 待确认 | 成对略过 target gid12 的 build/take | 保留 CM、物理 table/slot、组初始化和独立 draft build。配对控制压缩了合法性缺口，继续核实际 FULL 捕获指针及读写代次、异常和 provider 退役，之后做逐位验收。 |
| 待确认 | 整帧 QSL/长度共享及上一轮计数等待交错 | 正常 CPU 镜像直接读点已定位。跨组计算共享仍保留原物理首生产者。QSL 进入 deferred packet closure，改变共享存储要重新闭合末读和首覆写。 |
| 已停止 | 因空 query 删 FULL 的原事件键 | 原 executor 拒绝同 descriptor 的键集变化。SMLA 已走清零，QLI 仍需按现有入口，不能认领三个 SMLA 调用的省时。 |
| 已停止 | 跳过整个 gid12 物理组，或只跳 build、保留无 reset 的 take | 前者会破坏 DSpark 输入交付，后者可能提交旧任务。 |
| 已确认可实现 | 新替换原语 | 本轮无新增逐位准入。 |

沿用匹配配平 C8 32K/1K 的 16 卡 176 轮。gid12 整次 CP/global/local 的原 guarded prepare-Free 为 **15.223438 ms/卡**，对应 1.484% 诊断窗。假设能删且暴露 50%、每轮新增 0.1 ms，条件净 **6.511719 ms/卡**。按旧配平轻量参照的 50% 命中及 50% 转移场景，条件净 **56.574 ms/批**。这是原价格在明确 producer 条件下的继续使用。独立 native packet 节省、原 C8/六普通的请求净收益均无新实测，也不与共同 frame 或上一轮计数等待的估计相加。

转到 target 采样输出的 prompt logprobs 与请求行回收，核对继承方法、分块回传和普通负载的实际分支，再追尚未闭合的输出消费者。 每个替换原语逐位一致。本段为只读源码、旧价格和本机控制。

证据见 [五对组装与三项图回放控制](target-metadata-delivery-controls.json)、[图源身份](target-metadata-graph-identities.json)、[分组及价格](target-metadata-delivery-review.json) 和 [原 gid12 定价](target-draft-metadata-price.json)。
