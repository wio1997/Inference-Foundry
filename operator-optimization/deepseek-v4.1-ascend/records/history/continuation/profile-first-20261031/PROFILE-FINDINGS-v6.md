# Profile 主导复核 v6：HC、Engram、模型尾部与辅助上下文

本轮继续采用“先充分查询 profile；仍有影响结论的问题，再查明确源码位置”。**整体目标未完成**：本轮证据细化了 F02/F03/F24/F25/F26/F27，首个 draft 初始化 F28 等范围仍待补查。候选条目复核数不代表模块闭合数或完成百分比。

## Profile 已经回答的问题

| 范围 | 直接观察与关联 | 对优化判断的影响 |
|---|---|---|
| 层内 HC 边界 | 1280 个保留窗口各有两个 HcPre、两个 HcPost；attention post→FFN pre 间隔中位 1.18 µs，最大 1.50312 ms。两处选定 EP3 调用分别绑定 model.py:902、905。 | 当前已经使用复合 HC 算子；关注内部输出分支及跨边界流水，不能重复登记为从零融合。间隔不是可省时间。 |
| 跨层 HC | 1248 对相邻层；64 个配置 Engram 边界与其他 1184 对分组。四个抽查的 Engram 边界各有 40 条计算记录；37–39 层前还观察到辅助 ReduceMean 和通信。 | 层内侧车没有列出这些层外任务，不表示中间没有计算。跨层融合必须覆盖真正中间图。 |
| 首层 target 初始化 | 选定 Tile39646 经 API/调用栈绑定 target repeat:1551；premix 初始化绑定1552–1553。 | 此次 target 的 repeat 已有物理执行证据；不能把它推广为 draft 首层的证据。 |
| Engram 普通 gate | 选定 40 条记录分别为 WKV1、参数转换/乘法3、普通旋转 gate34、模型复制2。WKV 与 gate 均有独立调用栈。 | 当前所选 gate 确实分解成多条物理任务，可明确登记归约、标量非线性、残差的组合候选。不能把整个边界时长当预计收益。 |
| 尾部 | 两个 EP3 窗口绑定完整 HC 复制、collapse、回 BF16、最终 RMSNorm、选行、LM head。归一化物理行数8184/4760，LM head 行数3/14。 | 新增按实际消费者行集合前移选行的条件变体；全量归一化并非仅凭 LM head 行数就可删。 |
| 辅助上下文 | 同一次选定尾窗中，三路 `[8180,5120]` concat→`[8180,15360]`，FC 本卡输出640列，TP8 AllGather，再对5120列 norm。 | 已绑定 DSpark 上下文投影，不再只按同名 MatMul 猜归属。本卡640列 norm 不等于完整5120列 norm。 |

四个明确 collective 连接进一步分别绑定辅助 hidden gather、MTP residual gather、MTP premix gather、DSpark context FC gather。它们属于单个历史 rank 的记录，不能替代完整跨 rank 的消息和张量所有权证明。

## Profile 无法独立回答后，只补查了这些合同

定点读取了 Engram common/layer 的已绑定表达式、model 的辅助消费者和尾部、runner 的选行与状态消费者、DSpark combine 函数和 MTP getter wrapper；复用了既存 HC 内核合同。未进行仓库全面审计。问题与允许位置保存在 [HC-ENGRAM-CODE-QUESTION-GATE.json](HC-ENGRAM-CODE-QUESTION-GATE.json)。

- Engram 用反旋转后的 FP32 hidden 计算范数和点积，最终残差却加原始 hidden；两者不能混用。key 和 hidden 的完整特征归约、BF16 WKV 输出边界、各步 FP32 舍入、mask 与特殊值语义均要保留。所选调用还实际执行了参数的两次 cast 和一次乘法；不可变且有版本管理时可研究复用。
- HC 外部 pre_mix 的混合输出可以与本次系数分支分别建模，但 post/comb/pre 仍有后续消费者；原生 kernel 完成事件不等于支持中途发布 tile。跨层残差可能被 Engram、辅助 hidden、下一次 HC pre 和稍后的 HC post 读取。
- runner 将完整 normalized hidden 保存进 ExecuteModelState；prompt logprobs 和部分非 aux 草稿分支还可读取。因此“只保留 logits 行”的无条件变体被排除；可研究 logits、prompt、draft 等所有有效消费者所需行的并集，并保留图接口、行序、重复索引和数值边界。
- 同一选定窗口既复制 raw MTP HC buffer，又用三路辅助 hidden 形成 DSpark context。已查的 aux 分支使用辅助上下文，不能仅因发生了复制就认定 raw 值一定被有效消费。若最终证明该分支中 raw getter/API 与其他读者均不可观察，可条件研究取消 raw MTP 生产并本地 collapse 后再 gather；本轮尚未证明 raw buffer 实际无读者。若所需功能合同包含 getter 每步返回完整 residual，则这项删除不成立。

## 尾部候选必须区分两个数据域

按 BF16、HC4、hidden5120 的稠密逻辑行计算，完整 HC 为40960B，FP32 mix 为16B，collapsed hidden 为10240B：

- raw HC 是有效必需消费者时，原 gather 行载荷40976B；保留 raw HC 而将 mix gather 换成 collapsed hidden，变为51200B，不能笼统认定减少通信。
- raw HC 被证明不可观察时，本地 collapse 后只 gather collapsed hidden 的条件模型为10240B/行；仍需证明 token 排列、裁剪、数值算法、事件和输出合同等价。

这只是不同前提下的逻辑载荷会计，**不是实测链路流量、通信下界或提速倍数**。它与按行选择、collapse→norm 融合相互重叠，不能累计计算收益。

## 证据与未完成范围

完整数据、源码片段指纹与条件记录见 [hc-engram-tail-reconciliation-v6.json](hc-engram-tail-reconciliation-v6.json)，摘要见 [hc-engram-tail-summary-v6.json](hc-engram-tail-summary-v6.json)，候选和模块台账分别见 [candidate-profile-ledger-v6.json](candidate-profile-ledger-v6.json)、[module-profile-ledger-v6.json](module-profile-ledger-v6.json)。

初次16窗查询的本地 ZCode 控制端超时124；独立状态与保存文件读取均退出0，确认远端结果已完成并可核验，未重跑16窗。随后只将两个尾窗扩到20ms，并查询四个指定 collective 连接。原 DB SHA 校验通过，大小/mtime 不变。没有新增设备采集、推理负载或修改目标实现。

本次细化轮次仍余 F21 的旧 profile 与166计数对账、F28 的 first-draft 初始化、F29 的草稿 logits/Markov 等细查；还需完成全模块、条件分支、资源边界和跨迭代覆盖核对。**whole_modules_closed=0，overall_complete=false。** 所有候选的实际可实现性、精度和端到端收益不能由本轮记录直接宣布完成。
