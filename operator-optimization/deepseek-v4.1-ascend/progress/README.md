# 算子优化逐条进展

这是正在逐条讨论和复核的工作看板。历史 v180 分析归档保留；本看板不把候选登记、报告成稿或重复讲解当作实现完成。

当前记录：2026-11-04（Asia/Shanghai）。

- 42/42 个既有候选族已建立独立记录入口；这是记录覆盖，不是优化验证完成率。
- 本轮有新的逻辑复核记录：8/42。复核范围见各页；子方案澄清不关闭整个候选族。
- 其余 34/42 沿用历史条件分析，待按用户顺序逐条复核。
- 新实现：0/42；新实现正确性通过：0/42；有同条件性能结论：0/42；已完成独立复核：0/42。

## 从这里看

- 当前讨论：[F07 · Q_B收尾→部分RoPE（待验证）](F07.md)。
- 更新规则：[记录规范](PROCESS.md)；新记录使用[模板](TEMPLATE.md)。
- 所有进展：[变更记录](CHANGELOG.md)；结构化状态：[candidates.json](candidates.json)；追加事件：[events.jsonl](events.jsonl)。
- 已登记 F01 Task 的机器恢复入口：[Task README](tasks/f01-early-y/README.md)和[恢复包](tasks/f01-early-y/resume-pack.json)。转向下一候选不代表关闭 F01。

## 候选看板

|编号|对象|优化类型|本轮状态|下一步|
|---|---|---|---|---|
|[F01](F01.md)|注意力前 mHC 复合边界|已有融合边界的拆分/调度（本轮仅 EARLY-Y）|EARLY-Y 源码逻辑已澄清；可实现性与收益待验证|完成请求中的独立复核|
|[F02](F02.md)|embedding → 首层 mHC 系数投影；mHC pre → 输入归一化|相邻操作融合（本轮仅 Attention 侧 PRENORM）|PRENORM 源码边界已澄清；数值/实现/收益待验证|只补整行片上保留与数值归约的决策缺口|
|[F03](F03.md)|attention 或 FFN mHC post → 下一次 mHC pre，跨边界重组|生产者/消费者融合或交接重组（本轮仅Attention Post→FFN Pre）|层内Post→Pre交接已澄清；新实现/数值/收益待验证|界定一个可实现的Post→下一Y接口|
|[F04](F04.md)|FFN 归一化与类型转换|已有归一化/转换融合；可选下游接口重组|已有融合已确认；整体耗时已分批统计，纯转换与新收益未知|决定是否选择Router内转换子方案|
|[F05](F05.md)|Q norm → 动态量化 → Q 投影，多算子链|归一化＋动态量化融合（本轮QR-NORM-QUANT）|Q链与多消费者已澄清；融合数值/接口/收益待验证|选择保留原数值的NormQuant实现和接入接口|
|[F06](F06.md)|Q 与 KV 投影，共享输入或合并投影|共享输入的多输出投影重组|同行输入联合投影已解释；已知行域反例及已有复用/重叠已复核，native与收益待验证|若实施，先选真实同行输入的分支|
|[F07](F07.md)|Q 投影 → RoPE，生产者与消费者组合|投影收尾与 RoPE 融合|Q_B→部分RoPE源链、BF16边界与tail范围已复核；收尾接口、正确性和收益待证|先判断能否低成本接入Q_B收尾|
|[F08](F08.md)|KV 投影 → norm → RoPE → SWA 缓存写入|投影/归一化/旋转/缓存发布融合|沿用历史分析；待本轮逐条复核|明确整行归约与最终 cache 写入边界|
|[F09](F09.md)|Index query 投影 → RoPE → 量化|Index query 的 RoPE 与量化融合|沿用历史分析；待本轮逐条复核|复用 query 专有的 scale 与舍入合同|
|[F10](F10.md)|Index weight 投影 → 转换与缩放|投影收尾的转换/缩放融合|沿用历史分析；待本轮逐条复核|固定三次精度阶段和有效消费者|
|[F11](F11.md)|Index key norm → RoPE → 量化 → key/scale 缓存写入|Index key 处理与持久缓存发布融合|沿用历史分析；待本轮逐条复核|复用 key 与 scale 的联合发布合同|
|[F12](F12.md)|Indexer TopK → 索引修整 → 共享索引发布；候选块号排序跨层复用或位图 mask|索引发布融合、表示重组与条件复用|沿用历史分析；待本轮逐条复核|先选定 TopK 发布或 membership 表示子方案|
|[F13](F13.md)|稀疏注意力复合边界|已有稀疏 Attention 内部流水/拆分|沿用历史分析；待本轮逐条复核|选定内部队列或 SWA 提前子方案|
|[F14](F14.md)|Attention → 逆 RoPE → 布局恢复 → 输出投影|Attention 收尾/逆 RoPE/布局接合|沿用历史分析；待本轮逐条复核|保留真实跨卡 head 恢复依赖|
|[F15](F15.md)|Attention 输出填充/恢复 → 投影|填充/通信布局与投影流水重组|沿用历史分析；待本轮逐条复核|查清投影每块需要哪些参与者的数据|
|[F16](F16.md)|两级输出投影与 reduce-scatter 的分块重组|两级投影与 reduce-scatter 流水重组|沿用历史分析；待本轮逐条复核|复用两级投影和归约前舍入合同|
|[F17](F17.md)|Compressor 两路输入投影，共享转换或分块|Compressor 双投影共享加载/分块|沿用历史分析；待本轮逐条复核|固定 C1/C2 类型域和两套独立输出|
|[F18](F18.md)|Compressor 状态池化 → norm|状态池化/归一化与 ring 维护重组|沿用历史分析；待本轮逐条复核|明确旧 ring 读取与新值覆写顺序|
|[F19](F19.md)|Compressor latent 分叉 → index key 与压缩 KV|latent 双分支共享读取/交接|沿用历史分析；待本轮逐条复核|固定未旋转 latent 的读者与覆写边界|
|[F20](F20.md)|MoE router → TopK → dispatch；输入 ID 准备融合与跨层复用|已有TopK内部算法优化、周边重复消除与跨算子融合候选|TopK已有融合与周边重复清理已复核；加载路径、正确性、收益及其他变体待验证|若实施，先验证同INT64接口的私有清理消除|
|[F21](F21.md)|MoE 内 dispatch、两次分组矩阵计算、激活、combine 的分块流水|已有 MoE 融合内部流水|沿用历史分析；待本轮逐条复核|复用已有融合事实并选择一个竞争机制|
|[F22](F22.md)|共享 expert 的投影 → 激活量化 → 下投影|共享 expert 投影/激活量化/下投影衔接|沿用历史分析；待本轮逐条复核|固定 clamp 与整行量化统计边界|
|[F23](F23.md)|共享 expert 与 routed expert 汇合 → scaling/add，可延伸至 HC|expert 汇合与 HCPost 接合|沿用历史分析；待本轮逐条复核|复用两路舍入和真实 join 条件|
|[F24](F24.md)|Engram hash/lookup → ready → 行视图或分片；host-UVA 寻址|Engram 公共准备/查表/行组织重组|沿用历史分析；待本轮逐条复核|复用当前 backend 与全写合同|
|[F25](F25.md)|Engram 投影 → gate/残差|Engram 投影提前与 gate/残差接合|沿用历史分析；待本轮逐条复核|区分可提前的纯投影与依赖 hidden 的 gate|
|[F26](F26.md)|最终 HC collapse → norm 及 logits 边界|terminal collapse/norm/head 边界接合|沿用历史分析；待本轮逐条复核|从全部读者反推真正必需行|
|[F27](F27.md)|DSpark 辅助上下文投影 → norm，候选流水|DSpark 上下文多输入投影/通信/归一化重组|沿用历史分析；待本轮逐条复核|固定三个 checkpoint 和完整特征归约|
|[F28](F28.md)|DSpark embedding → HC 重复/初始化 → decoder|草稿 embedding/HC 展开/初始化重组|沿用历史分析；待本轮逐条复核|复用目标/草稿已有前置 shard 的区别|
|[F29](F29.md)|DSpark 上下文 KV 投影 → norm/RoPE → 组装与缓存写入；按请求复用 SWA 索引|三路 context KV 多输出投影与缓存发布|沿用历史分析；待本轮逐条复核|复用三套参数/cache 独立事实|
|[F30](F30.md)|DSpark collapse/norm → Markov bias、logits/置信度/验证|DSpark head/Markov/选择及回馈重组|沿用历史分析；待本轮逐条复核|固定 Markov 前一步 ID 与 raw confidence 输入|
|[F31](F31.md)|Attention/DSpark 相关 metadata 构造、共享与发布|metadata 多输出生产、共享与发布|沿用历史分析；待本轮逐条复核|按字段目的、query/cache 域及代际判重|
|[F32](F32.md)|SP 输入 gather → Q_a/KV 投影，研究前移投影后按需通信|投影前移与通信载荷重组|沿用历史分析；待本轮逐条复核|确认哪些原 hidden 读者使 gather 仍必要|
|[F33](F33.md)|C1 compressor 与 SWA KV 的同域投影共享、联合分块与前移通信|C1 与 SWA 同输入双投影共享/前移|沿用历史分析；待本轮逐条复核|固定 C1 域与两个独立 norm|
|[F34](F34.md)|C2 ring 状态的可观测性裁剪、稀疏维护与表示重组|C2 ring 可观察状态裁剪/表示重组|沿用历史分析；待本轮逐条复核|复用 rollback 反例并选定一个状态子方案|
|[F35](F35.md)|目标采样的按需raw logits、行选择/转换/argmax与拒绝输出重组|采样按需输出/选择转换融合与重组|沿用历史分析；待本轮逐条复核|复用 greedy 不等于 score 全死的结论|
|[H01](H01.md)|响应等待路径的轮询开销与重复等待工作|CPU 等待路径调度/记账重组|沿用历史分析；待本轮逐条复核|确认当前同请求的 on-CPU 与 producer-ready 边界|
|[H02](H02.md)|输入准备中剩余小拷贝、锁页和元数据工作的合并重组|输入准备的小拷贝/缓冲/metadata 重组|沿用历史分析；待本轮逐条复核|复用有序 gather 与跨 plane 别名合同|
|[H03](H03.md)|前缀块哈希的跨请求重复工作|跨请求精确前缀哈希结果复用|沿用历史分析；待本轮逐条复核|核对完整 hash key 并给当前 CPU 成本定价|
|[H04](H04.md)|前端长提示 tokenization 的精确复用、排队与批处理|tokenization 精确复用与排队重组|沿用历史分析；待本轮逐条复核|固定完整 tokenizer key 与当前排队成本|
|[H05](H05.md)|前端 DecodeStream 早段长调用与输出就绪流水|DecodeStream 准备/串行卸载/发布流水|沿用历史分析；待本轮逐条复核|区分 CPU 工作与 wall 等待，并保留逐 token 语义|
|[H06](H06.md)|固定 speculative 宽度下的 padding 域拆分与图 bucket 重组|padding 域与图 bucket 重组|沿用历史分析；待本轮逐条复核|复用24行约束和远端 receiver 工作事实|
|[H07](H07.md)|原始 token ID 图回放与图外 embedding 的单一生产路径|图内外 embedding 单一生产路径|沿用历史分析；待本轮逐条复核|绑定实际 raw-ID capture/replay 分支和有效读者|

## 读表时的边界

42 个是候选族，包含融合、拆分、复用、状态重组和调度。不是 42 个新增融合算子。相互重叠的方案不能重复计算收益。

源码逻辑成立、当前二进制实际执行、可实现性、数值正确性、性能收益分别记录。旧 166 时序、新 171 图回放、不同采集批次的工作量不能混成同一次实验。

本次只更新分析与计划，没有采集新 profile、修改服务器或启动独立子代理。原始 profile 不上传。后续执行仍遵循当时的任务范围。
