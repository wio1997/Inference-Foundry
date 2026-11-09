# BatchFrame 联合生产协议（研究设计，待确认）

## 介绍

目标是同一次runner build中，给global和local两个域各建一次batch事实，再与每个资源的物理状态组装。保留原算子结果/顺序；先省重复host工作和提交，megakernel/mini graph只是之后可选的实现边界。没有已部署补丁或native准入。

## 到了哪里

实际runner每次创建新 `common_v41_batch_metadata`，global CP再嵌套 `cp_global`；这种generation作用域比跨迭代按shape缓存更强。当前 `batch` 已共享首个builder的coordinates，并非26份完全独立的工作。旧CPU-plan会保留首producer、使用私有CPU快照，25命中仍保留各builder自己的device qsl发布。CP length-elision已跳过剩余25份无活读length计算，不能再次认领。

| 字段 | 真正生产时点 | 可共享条件与仍需保留的读者 |
|---|---|---|
| target/draft count、mode、B/K/Q、role/epoch | target协调前；draft query数从实际类推导 | 显式区分两角色；8组DP CPU collective一致。dummy、K0、skip或错误时全体同序fallback。 |
| local CP区间、CPU qsl/length、maxima | 本次输入CPU事实和CP partition确定后 | 仅同代、同actual/causal/count域；快照/私有值不能被公开metadata的原地变化污染。global域单独保留。 |
| local device qsl、authoritative seq_lens | 原CPU H2D和GPU reject校正之后 | 不能从CPU乐观长度替代GPU值。共享输出需要逐位相同、稳定地址和全callback末读；first producer仍按原顺序计算。 |
| positions、RoPE/cos/sin视图、query views | 对应batch/源表/ratio就绪后 | 共享只读原视图；global KV与local Q切片不同。原cached RoPE缓冲仍有global读者，不得被local第二次gather覆写。 |
| block table、slot mapping、压缩plane、C2 ring | 每资源与请求代次确定后 | 始终group-local；相同shape/ratio不代表物理页相同。不能把首group的整份common或返回对象给其余group。 |
| native SMLA/QLI metadata与compressor task | 所需scalars/tensors就绪后 | 原batch/group共享已存在，不能假设26次metadata native op都独立；保留reset/take/publish与stage/id(buffer)键、原标量与原输出字节。 |
| CPU输入slot、GPU静态地址 | 最后实际H2D读/所有device reader后 | CPU host fence与device reuse fence分开；不得用提前reply、Python返回或enqueue代替末读完成。 |

每组仍生成原global_metadata、local metadata、cp_token_range和group-specific block/slot视图。可以预绑定配置常量、一次构建公共视图和maxima、合并兼容发布；必须逐项证明被删除的Python/native调用已无独立副作用。保留per-groupreturn_replace，直到证明观察者与alias语义完全相同。

## 结果

**待确认且值得做原型的边界**是整个同代描述符/组装/发布，不是25次小DMA。保守净模型保留两个parent builder和每组返回，见[净价格](metadata-host-net-price.json)。f=.5/h=.25ms时，需要约51.2%的建模组件真正可移除并暴露，才到旧C8整批1%排序参考；这是一个可以否证的开发门槛。

开始native原型前仍需核清：

- V1 `DeviceMetadataExecutor` 的COMPRESSOR→ATTENTION→INDEXER队列，各eager/FULL/dummy/empty读者是否都在下一代覆写前完成。当前V1只见release；不能借V2 TargetDeviceMetadata.finish证明全task join。
- 所有公开scalar/tensor/alias读者、捕获的closure与graph地址。尤其native metadata attrs含max_seq_len/actual/count/ratio，shape-only mini graph可能保留旧标量；完整更新协议另行判断，原样shape缓存已停止。
- 原语逐位比较需覆盖qsl、GPU lengths、slot/physical metadata、native SMLA/QLI payload与各阶段consumer输入；最终token或文本一致不能替代。旧88次文本对照不是此准入。
- 测量新增frame/key/guard/object/descriptor提交成本、实际减少的CPU工作及被GPU/peer等待隐藏的比例；不把9.29759ms enclosing host包络当全可删，不把Gloo等待全删。

以上缺口不阻断其它路径扫描；没有再次探测已解决的pinned自动快照，也没有重跑旧CPU-plan性能。
