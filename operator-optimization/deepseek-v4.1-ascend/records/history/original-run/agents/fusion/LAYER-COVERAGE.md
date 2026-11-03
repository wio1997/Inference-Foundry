# Loop：冻结 40 层主要算子链的融合覆盖核对

## 怎么读

此表将已有 F01–F31 边界检查和冻结源码、模型配置对照到具体层。**已扫描**指已有结构记录，不表示已证明可融合；**已有融合**指源码已有相关复合调用，不表示内部无法再重组；**禁止的简化**仅针对写明的删除或重排，不能泛化成“两个算子永远不能融合”。全部待重组项的实现可用性、精度、完整消费者和速度都尚未证明。

| 主要链或边界 | 当前覆盖与状态 | 必须保留/未决 | 依据 |
|---|---|---|---|
| mHC pre → attention norm；attention post → FFN pre/norm；FFN post → 下一层 | F01–F04 已记录；F01、F04 源码已有复合调用，F02、F03 有条件候选。存在不同层间消费者，不能把 `RULED_OUT_IN_SCOPE` 解释为进一步融合在物理上不可能。 | mHC pre 的 post/comb/pre_mix 输出、残差及 BF16/FP32 语义；跨层 Engram 和 aux 分支。 | `model.py:844-920`，G1、`candidates/F03-G2.json` |
| Q/KV 输入量化、Wq_a → Q norm → quant → Wq_b → Q RoPE | F05–F07 已记录；Q norm 同时给主 Q 以及指定 index source 的 indexer，F06 在满足条件时已有输入量化复用。 | 8 个 index source 不能假设 qr 是单消费者；CP 下 Q 与 KV 行域未必一致；保留原量化 scale 与中间舍入和已有多流重叠。 | `dsa_v41.py:395-431,440-495`、`evidence/QR-consumers-and-fusion-review.md` |
| KV norm → RoPE → SWA cache；SWA / compressed KV → sparse attention → output projection | F08、F13–F16 已记录；F13 的 sparse flash 已是复合算子。 | cache 是持久和跨层状态，不能因为只融合当步 attention 就删除写入；投影中跨 rank 的 head restoration / reduction 条件、通信布局和数值都需要保留。 | `dsa_v41.py:620-715`、`context_parallel/dsa_v41_cp.py:189-224` |
| Index query / key / weight → topk / fixup → shared index buffer | F09–F12 已记录；key 更新在 KV source 分支，query/选择在 index source 分支。 | index source 的结果被后续层复用，稳定地址及空缓存 `-1` 初始化不能忽略；跨层的 shared buffer 更新/alias 与所有读者细节尚未穷尽。 | `dsa_v41.py:344-345,377-395,497-562,567-611` |
| Compressor 双输入投影与状态池化 → latent 分成 index key 和 compressed attention KV | F17–F19 已记录；2-ratio 的 KV source 有投影/池化分支，1-ratio 与其不同；池化已有复合调用。 | latent 的两条消费者和持久 cache 不可当作单消费者临时量抹掉；C2 ring state、历史页、物理槽位和运行时空行分支需保留。 | `dsa_v41.py:497-562`，`model.py:510-550` |
| FFN norm → FP32 router / BF16 expert → MC2 dispatch、expert、combine → HC post | F04、F20–F23 已记录；RMSNormCast、融合 MC2、部分 scaling/add 已存在实现或条件分支。 | router 与 expert 两种精度输入、expert 路由/通信及结果消费者均需保留；MC2 内部完整消费者、逐 tile 就绪和实现等价未知，不能按纯 GEMM 解释。 | `model.py:262-328,844-920`；`evidence/MC2-known-dependencies.md` |
| 层 1、14 的 Engram lookup → gate → residual | F24、F25 已记录；G5 展开数据及 event 边。 | 查表及状态、ready event、mask/rotation 和后续残差消费者均需保留；不能把主机卸载 lookup 一概归为 CPU 计算；图中的分支与 runtime alias 尚未完全绑定。 | `model.py:1555-1592`、`graphs/source-templates-G5.json` |
| 最终 HC collapse → norm / logits，与 MTP buffer | F26 已记录；新增 F-Tail-01–03 将 collapse 内部和条件多消费者拆开。 | DSpark 配置可推导 MTP 条件，PP/SP 与实际分支需进一步验证；应保留 buffer 和输出各自值、精度与生命周期。 | `agents/fusion/FINDINGS.md`、`agents/fusion/CONFIGURATION-REVIEW.md` |
| DSpark 草稿及 metadata | F27–F31 已记录，F30/F31 当前 UNKNOWN；G5 补充顺序、概率与事件。 | 禁止跳过逐步采样/状态依赖；cache 和 metadata alias、全部消费者仍有未覆盖范围。 | `evidence/engram-dspark-g5-review.json`、`candidates/F31-metadata-partial.json` |

## 配置所声明的跨层消费者

`model.py:510-550` 用最新 source 层构建角色图；`dsa_v41.py:567-575,616-624` 说明非 index-source 读取共享索引、long-context 路径访问 source KV。由当前 40 层配置推得，KV source 2、8、14、20 对应角色层分别为 **2–7、8–13、14–19、20–39**；索引 source 2、8、14、20、24、28、32、36 分别对应 **2–7、8–13、14–19、20–23、24–27、28–31、32–35、36–39**。这里是配置和源码角色，不等于每个请求都发生同样的非空读；层 0、1 的压缩比例是 0。第 20 层是候选块 source；24、28、32、36 这些后续 index source 按源码角色使用候选过滤，但是否每次进入特定实际 kernel 分支仍取决于输入和 backend。

这说明例如 **层 20 压缩 KV 的状态要满足最多 20 个配置角色层**，不是只有其本层 attention；第 20 层候选结果也不能只按一次 topk 的本地临时结果删除。数值 tensor alias、实际每次读写区间与所有中间实现内部消费者仍 UNKNOWN。

## 可复现、未覆盖部分

从任务根目录执行 `python3 agents/fusion/map_frozen_layer_coverage.py`：输出 `frozen-layer-coverage.json`，包含所有 40 层 role、每层适用的既有 F ID、逐条 F 登记、输入和脚本哈希。该脚本执行的是配置角色映射和现有登记索引，不运行模型，不检查算子数学等价，也不证明完整覆盖。`F01/F04/F13` 的原 `RULED_OUT_IN_SCOPE` 是已实现复合调用的记录，不是对所有重组规则的否定。

仍需：以真实运行配置确认 SP/PP/MTP 分支；对每条候选结合同一调用的 trace 与输入 shape 确认是否执行、是否已由其他算子融合；深入自定义算子实现、别名及所有边界消费者；对每种允许的变换证明精度和状态语义。各项未证之前只能写条件候选或 UNKNOWN，不做代码改动或实际加速判断。
