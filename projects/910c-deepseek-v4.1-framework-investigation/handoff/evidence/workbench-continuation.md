# 工作台续扫：计划、物理读序与整段停止条件

2026-11-09。基于 GitHub 研究分支 `c5db03424cbf4ad917588fba2ea29de40ad3f0e7` 的工作台、冻结环境、12 模块图及 S00–S11 覆盖接续。该分支比上次研究发布多一份独立 deepinfer 基线；本次沿工作台的冻结 vLLM-Ascend 路径，不将另一套实现的数据混入。

**范围**：171、910C、CANN 9.2.0、W8A8、TP8/DP2/EP16/PP1，C8 输入32768/输出1024及原六普通请求。只新增静态消费者核对和离线算术。SSH 仅核对主机身份及 `/data/tiankuan/wio` 下的目录，未查容器/API/当前二进制，未加载模型、编译、测量或变更部署。冻结身份沿用原件，不能据此宣称当前服务已核验。

## 从哪里继续

S00–S11 已有首轮排除、保留或具名条件。它们不是未经分析的入口；早期 `progress.json` 的“暂停/下一模块”不能覆盖后来的工作台。因此先选尚未闭合的**目标 DP 计划→草稿 DP 计划**交付，实际追到 DSpark query、dummy/skip 与 EP 调用，再继续核对 Engram 复用、SMLA 物理 reader、QLI 主计算和 router 入口。其余路径引用原已扫记录，不能写成此次重新扫描。

## C04：目标与草稿共用协调

- **介绍**　一次 CPU DP 协调共同发布目标图模式、每 DP 请求数、草稿 query 数和角色，省第二次协调的净关键耗时。实际链为 runner `_determine_batch_execution_and_padding`→`_sync_metadata_across_dp`，以及 proposer `_propose`→DSpark `set_inputs_first_pass`→第二次 sync→metadata/三层专家集合调用。
- **到了哪里**　**待确认**。固定 K5、padded eager 下，正常草稿 query 数为 `5*B`，接受数改变位置/有效长度，不改变该数；但只复用目标 padded 数的子方案 **C04a 已停止**。B1 的6行与 B4 的24行均可进入目标24行桶，草稿却分别为5/20行。新/mixed prefill 更不能从目标 token 总数求 B。`_skip_drafting` 还会发完整单请求 dummy query 并保留 EP 服务；dummy 本身也参加 DP 协调。
- **结果**　扩展共同 packet 至少含 epoch、每 DP 的 B/K/query width、live/dummy/skip 角色及全 DP 一致 fallback；目标图模式另行保留，不能沿用到 eager 草稿。可取消的只是第二次协调及其提交，peer 等待不自动消失。旧整批1%=185.908ms；合法命中 N 次须 `Σ净关键省>=185.908ms`，N=200/250 时每次≥0.930/0.744ms。尚缺该净额和角色覆盖，登记后前进；没有为此新跑服务。

## C06：Engram 完整内容复用

- **介绍**　两个实际层各缓存 hidden-independent 的完整 hash tuple→lookup→WKV/key统计，保留 history、mask、miss 与 bank 发布；层/表/WKV/rotation 版本均属于键。线性 WKV 的分头分解也会改变归约顺序、缓存容量与残余调用，不能免费套用完整结果命中模型。
- **到了哪里**　**待确认**。当前 BF16 WKV 输出还要与本层 hidden 做 FP32 rotation、norm 和 gate，缓存 lookup 或单 head 相同不能代替完整 KV 结果。命中时仍须保留本代 history 和输入有效性更新。
- **结果**　原同周期所需消除可缓存耗时71.1%–96.3%（中位75.4%），**不是 row-hit 阈值**。部分 miss 可能仍保留整次 dense WKV、权重搬运和集体调用；必须用 hit/miss 后真实残余调用及净发布成本定价。具体缺口为按层完整内容命中、每波 miss 数与残余 WKV 成本。单独 lookup/gate/局部搬运仍沿原整段低值停止，不重复开启小方案。

## C09/K02：selected KV 到 Cube 的真实交付顺序

- **介绍**　以分页 cache 直读替代 Vector `cmpKvGm→kvMergUb→kvMergeGm`，让 QK/PV 直接消费同一逻辑 plane；保留 softmax、first partial 与原精度。源码新核对范围在 `workbench-continuation-source-review.json` 的 arch22 Vector/Cube 摘录。
- **到了哪里**　**待确认**。原 reader 并非一律按 raw TopK 顺序：`GetKeyGmOffset` 用 block table 求物理偏移；`CopyInKv` 的正常双块搬运从较小物理偏移开始，负 stride/溢出/尾部则回退两次单块搬运；`ProcessVec0L` 遇首项负索引会 flush 后停止。Cube 的压缩 QK 与 PV 都读取这一已压实 plane。故仅按原 TopK 做新稀疏加载并不自动复现原 reduction 顺序。
- **结果**　CPU 分支见证 `[4096,1024]` 会按第二块、第一块交付；重叠偏移回退保持原顺序，尾部回退只交付有效块。这只是源码合同见证，未验证全核数值或速率。新 reader 须复现原物理 pair/compact 顺序、有效长度、V 子块分界与 buffer epoch，或另过数值验收。原 K256/等速17.78ms TTFT 情景仍待有效 K、完整 kernel 净差；不能从 capacity512、QLI 可见key数32769–32899或本 CPU 见证得出。direct 与共享 panel 仍共用一份预算，OA 净增量同窗另量。

## C11：压实评分留下多少替代成本

- **介绍**　逐 key 压实 QLI 评分，省候选外 MAC；保留原不同消费者的 Q/score、filler、tie、非有限值及次序恢复。2048-key 整段空块为0，128-key段可删 MAC仅0.0427%，同范围段跳过继续停止。
- **到了哪里**　**待确认**。以原四次QLI中位1.980801ms、理想评分MAC减少50.0899%和223个完全关键波作有利情景。令 c 为旧整段中可压实的评分时间比例，ρ 为压实后/原评分速率，h 为新增 gather/压实/恢复等净关键成本；`gain=f*c*T-(1-f)*c*T*(1/ρ-1)-h`。
- **结果**　旧整批1%需每四次净省0.833667ms。等速、h=0仍需 c≥84.024%；即极有利 c=1，h最多0.158514ms；h=0时ρ至少0.861817。两个原件 cohort 不是同一次stage分解，这些是联合假设下的去留门槛，非实测预计。缺评分主体 c、真实压实速率ρ及 h，保留后继续；原 member-only 服务组合仍按旧失败门槛停止。

## C12：结束同一预算内的边界融合

- **介绍**　把 norm、router GEMM、TopK、typed ID 包共同生产。FP32 router 输入与 BF16 expert 输入均有实际读者；动态384维 gate、选择 bias、sqrtsoftplus、6专家权重及原舍入要保留。当前无固定路由表，不能替换成只算6个dot。
- **到了哪里**　**待确认**仅限能改变 GEMM 主体/执行成本的较大方案。**C12a 已停止**：固定 GEMM、host 提交、段外就绪和专家服务，只消除 packet/gating/RMS 的全部边界。用112个同rank同周期逐窗并集复算，未用不同中位数或跨rank相加。
- **结果**　仅40层目标的非GEMM全免为同周期0.8144%–0.8963%，中位0.8570%；连3层草稿同类任务也全免为0.8764%–0.9637%，中位0.9214%。整组仍不过1%，其 cast/ID/TopK/logits/RMS 小变体一并停止。即给非GEMM全额免费，目标还需额外净删 router GEMM 的9.638%–18.819%（中位13.883%）；目标+草稿为3.156%–11.705%（中位7.129%）。新融合实际仍要执行TopK/norm，新成本只会收紧。以上是局部同周期条件筛查，未折成 TTFT 或新请求收益。

## 十二模块覆盖与剩余条件

“沿用”表示复用工作台 S 记录和既有 profile；“本次”只标新增核对处。待确认条件本身不是未扫执行入口，也不等于已经实现。

| 模块/数据交付 | 已扫路径与本次增量 | 结果与具体缺口 |
|---|---|---|
| M00 请求→输入/metadata | 沿用S00–03/S05；本次补目标/草稿DP计划 | C01/C02缺可删净主机提交；C03 B4已有实测，小批caller另判；C04a停止，扩展packet待确认。 |
| M01 history/hash→两层Engram | 沿用S04；本次明确完整结果与残余miss工作 | C06待完整键/权重代际及残余WKV成本；局部gate/交付停止。 |
| M02 残差→Attention HcPre/RMS | 沿用S06与K01 | C07缺可删Gram主体及原舍入验收；K01须同owner分kernel基线，保留Engram/aux/迟到系数。 |
| M03 Q/KV/cache/QLI→SMLA | 沿用S07各源/消费者；本次补物理packed读序和压实开销门槛 | C08缺换M数值/行序/窄AG净关键差；C09缺有效K/原序整核净差；C11缺c/ρ/h，已记条件。C2 ring完成/环绕/代次依原读者合同保留。 |
| M04 SMLA→RoPE/OA/OB/TP | 沿用S07/K02/F08 | C10只留新send producer、容量与同窗OA净增量；stock+pack停止；F08大prefill备选，短图保留回退。 |
| M05 残差→FFNPre/双精度norm | 沿用S06/S08；本次补完整非GEMM预算 | C07共享主体另判；C12a整组停止，FP32 router值不以BF16再widen替换。 |
| M06 路由→DFC/shared→归并 | 沿用S08含CombineV1/V2与远端专家；本次界定C12a | C05缺idle实际命中/全false净值；K03缺τ/ρ/h/层覆盖/关键迭代；草稿E8/K3不继承目标E24/K6定量。C12大方案缺GEMM净差。 |
| M07 FFNPost→下层/Engram/aux | 沿用S06/S08与K01 | 保留BF16 rounding、系数迟读及aux读者；单纯merge/Post原变体停止，不能复计残差预算。 |
| M08 末层→raw/collapse/norm/head | 沿用S09与原terminal逐行证据 | C05/K10缺紧凑行域设备数值、通信/原始raw bank合法退役及整段净差；不删其他DP真实expert服务。 |
| M09 采样/验证→seed/count | 沿用S09/S11，正常greedy与spec rejection | raw draft、first mismatch、bonus及CPU计数保留；局部head/Markov同预算低值停止；随机/概率分支按固定配置排除。 |
| M10 aux/context→三层五query→五Markov | 沿用S10；本次核5B与dummy/skip、per-group窗口位置 | C14/K05缺aux-owner额外权重/净发布成本；K04缺设备TP/epoch协议；context裁剪须每物理组的final writer和未来reader，不能只按接受前缀删。 |
| M11 输出/回收→下轮M00 | 沿用S11；本次闭合DP计划回退与epoch | C15缺普通请求ready与retired的实际净关键差；C02/K07共享输入预算扣重；K09跨轮常驻需新设备控制协议。 |

固定排除仍为 MRV2/PCP、A5 packed、MM请求内容、LoRA/EPLB、PP/池化/KV transfer、随机/entropy/synthetic验证、动态/置信度/reduced-vocab草稿等未命中配置。初始化/capture、全局空批、idle、new/resume/reorder、prefill/verify、空cache、skip-draft和stop/abort的去向沿S记录保留。不能把“未命中”写成该功能数学不可融合。

## 证据、复算与下一处

`workbench-continuation-source-review.json` 保存本次11份源码的hash、具名行段及摘录；不是全源码覆盖率或当前二进制认证。`workbench-continuation-inputs.json` 是原件hash和同窗小输入；`workbench-continuation-results.json` 可独立复算。命令：

```sh
python3 analyze_workbench_continuation.py --inputs workbench-continuation-inputs.json --output recomputed.json
```

本次实际收敛为两个停止子范围、三个更明确的去留门槛/消费者合同；没有新增设备性能值，不声称找到新根机制。12模块已有首轮处置保持；当前具名缺口可交给原型/补证，不再把它们当作未扫模块反复启动。

下一次若继续补证，先找**同一prefill调用的 SMLA selected索引/有效长度、page表和物理packed顺序**，连到完整QK/PV耗时，决定C09/K02是否过净值门槛；缺这些字段就保留条件，转C08同owner分kernel对照。K03沿已有τ/ρ/h缺口继续，无资料时才考虑最小阶段诊断。服务部署及设备实验继续留在各自独立任务，不在本次静态闭环内启动。
