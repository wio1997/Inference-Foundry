# 逐条进展变更记录

由 events.jsonl 生成；新的纠正通过追加事件记录。Git 提交保存每次记录变更，事件不替代技术证据。

## 2026-11-03 · 20261103-F04-EXISTING-NORMCAST-REVIEW

涉及：[F04](F04.md)

类型：EXISTING_FUSION_AND_CONDITIONAL_INTERFACE_REVIEW；依据：核对归档DeviceOperator守卫/回退、BF16 NormCast kernel输出舍入及扩展，并复用v29/v63路由消费者证据。

归一化+转换已融合，首次融合机会排除；FP32来自舍入BF16。新改法仅保留ROUTER-WIDEN条件候选，不声称原API直接支持或已有收益。

## 2026-11-03 · 20261103-F03-USER-UNDERSTANDING

涉及：[F03](F03.md)

类型：USER_UNDERSTANDING_NOT_VALIDATION；依据：用户回复“ok 了解了下一个”。

记录用户了解F03解释并转F04；不记作新接口、数值正确性或性能通过。

## 2026-11-03 · 20261103-F03-POST-PRE-REVIEW

涉及：[F03](F03.md)

类型：SOURCE_CONSUMER_AND_INTERFACE_REVIEW；依据：核对固定模型同层Attention Post→FFNPre→后续FFNPost的消费者，HcPost底层舍入及v147源合同。

先研究Post结果直接交下一Y生成的一项读取；R仍须保留给下一系数生产和后续Post。不声称省掉R写入、跨过Engram/aux或预折叠等价；未实施、无新profile或性能对照。

## 2026-11-03 · 20261103-F02-USER-BOUNDARY-CONFIRMATION

涉及：[F02](F02.md)

类型：USER_UNDERSTANDING_CONFIRMATION_NOT_VALIDATION；依据：用户确认“把归一化放到hcpre里面”，随后说“这个可以记录一下，下一个”。

记录用户已确认的方案表述及转向下一条；保留原系数输出和Y精度边界。此事件不新增正确性、实现或性能验收。

## 2026-11-03 · 20261103-F02-PRENORM-REVIEW

涉及：[F02](F02.md)

类型：SOURCE_BOUNDARY_AND_ALGORITHM_REVIEW；依据：用户转向第二条后，核对固定模型的HcPre→input_layernorm关系、RMSNorm定义及F02-PRENORM历史数值合同；原源码SHA匹配归档。

本轮仅澄清Attention侧Y→RMSNorm融合；保留BF16中间舍入和完整行归约。条件式60KiB接口不是HBM或收益证据。其他F02子方案未关闭，F01仍待验证；没有新实现、NPU执行、独立代理或性能对照。

## 2026-11-03 · 20261103-F01-INDEPENDENT-REVIEW-REQUEST

涉及：[F01](F01.md)

类型：REQUEST_RECORDED_NOT_EXECUTED；依据：用户要求subagent使用GPT-6.1/xhigh；当前临时侧聊禁止创建或联系子代理。

独立审查登记为已请求但未启动，无模型产物；后续只有实际取得审查结果才可更新。不得将本次主执行者复核伪装成独立代理复核。

## 2026-11-03 · 20261103-F01-CORRECTIONS

涉及：[F01](F01.md)、[F02](F02.md)

类型：EXPLANATION_AND_CLASSIFICATION_CORRECTION；依据：用户追问融合定义与真实代码后，回到源依赖和profile能力边界。

纠正把EARLY-Y当新增融合、把Y描述为等全部系数完成才生成、把F02的Y+norm混入当前方案的讲解。两队列/部分硬件空闲不保证并发；旧166约1µs间隔不证明全局CPU从不瓶颈，35µs不是可省时间。F02只记录边界澄清，不记为本轮完成其分析。

## 2026-11-03 · 20261103-F01-DEPENDENCY-REVIEW

涉及：[F01](F01.md)

类型：SOURCE_REVIEW_AND_BOUNDED_PROFILE_REUSE；依据：本侧聊主执行者只读核对归档模型/A3源码、旧166数据库选定TASK及对应聚合、171图回放记录；可重现摘要见evidence/F01-review-20261103.json。

有外部pre_mix时Y不依赖当前新系数；当前源码先投影/同步，在第二阶段每块先写Y再处理post/comb。EARLY-Y是交付边界重组候选；未建立native改动可行性或收益。

## 2026-11-03 · 20261103-REGISTER-42

涉及：[F01](F01.md)、[F02](F02.md)、[F03](F03.md)、[F04](F04.md)、[F05](F05.md)、[F06](F06.md)、[F07](F07.md)、[F08](F08.md)、[F09](F09.md)、[F10](F10.md)、[F11](F11.md)、[F12](F12.md)、[F13](F13.md)、[F14](F14.md)、[F15](F15.md)、[F16](F16.md)、[F17](F17.md)、[F18](F18.md)、[F19](F19.md)、[F20](F20.md)、[F21](F21.md)、[F22](F22.md)、[F23](F23.md)、[F24](F24.md)、[F25](F25.md)、[F26](F26.md)、[F27](F27.md)、[F28](F28.md)、[F29](F29.md)、[F30](F30.md)、[F31](F31.md)、[F32](F32.md)、[F33](F33.md)、[F34](F34.md)、[F35](F35.md)、[H01](H01.md)、[H02](H02.md)、[H03](H03.md)、[H04](H04.md)、[H05](H05.md)、[H06](H06.md)、[H07](H07.md)

类型：TRACKING_CREATED_NOT_NEW_ANALYSIS；依据：用户要求每个候选的逻辑与后续工作可在GitHub观察和回溯；来源为既有CANDIDATES/v180记录。

为42个既有候选建立稳定页面和状态维度；41条沿用历史条件分析，不声称本轮重新验证；42页建档不记为42项优化完成。
