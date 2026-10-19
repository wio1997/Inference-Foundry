# 当前交接 — 2026-10-17（Asia/Shanghai）

## 当前授权和目标

用户已恢复 **172.16.12.32 / 33** 的现场核查、准备、服务启动与测量，并进一步明确两台原有实验作业/预留全部可以停止。仅操作这两台；系统守护进程和访问入口保留。最新要求是 **全部使用 SDMA**，包含设备内、机内、跨机和最终 PD KV 数据路径。控制/初始化通信另记，不能用 TCP/RDMA 数据回退代替 SDMA。

最新用户要求是**持续推进原性能优化目标，官方部署仅作为已通过的前置步骤**。本轮官方基线所必需的Mooncake直传已获主控转达人类授权，覆盖此前「先不考虑Mooncake、优先MemCache」的阶段偏好；仍保持全SDMA，仅32/33，当前只操作自己实例/holder，其他作业不动。已实际启动原样MooncakeConnectorV2和原样官方代理，先交付可运行原生PD。MemCache准确组件为memcache_hybrid/memfabric_hybrid，后续官方MultiConnector池集成不由本轮自动扩张；不再以自写纯Store协议代替官方基线。

保持 GLM-5.3 W8A8C8 能力、约定精度和正确性；复用已有正确算子。以 vLLM-Ascend 为正确性/稳定性参考，研究状态、调度与执行机制，验证真正 P→KV 传输→D，排除 D 重做 prefill。参数扫描由其他工作负责。本线已完成8组负载的两轮官方图PD基线，MTP1有吞吐/TPOT实测收益且11项保护题无新增退化，但存在TTFT权衡和原生已有数值错误；业务请求分布、数值SLO和完整正确性验收标准待定，不能宣称完整精度或可持续容量验收。

分支 `codex/glm5-3-vllm-optimization`，本次接手 HEAD `f5571dc0e1c271ae533a347700f24b4a2c105762`，接手时工作区干净。继续 [draft PR #2](https://github.com/wio1997/Inference-Foundry/pull/2)，不自动合并。不改旧副本 `/Users/wio/work/Inference-Foundry` 或原 SGLang 研究分支。

## 协作分工（用户最新校正）

原主对话作为用户沟通、方向/预算与证据复核的主控；本执行对话继续32/33实现、实验、修复及证据交付。主控通过现有HANDOFF/wait_threads读取，无需跨聊天主动发送消息，不增任务/看板，不为普通工程动作增加逐步审批。本次分工校正不打断已加载的per-TP实验。诊断须结合匹配native边界与传输/布局/完成证据，逐token差异不单独证明传输错误；不改模型/量化来迎合结果。

## 当前执行与证据恢复


**18:03Z P11两轮及所有保护通过，当前正式同arrival/continuous suite74225**：P11实际graph capture/replay已确认、inference errors0。smoke60/60，C1/C4同24项PD/native各24/24，16reuse pairs+8个HTTP1↔4 max1/cancel transitions全部通过。正式两轮240/240，原算术11/12不变。对P8/D10 eager控制，TTFTgeo0.86373、TPOT0.98028、吞吐点估计1.08411，无>5%TTFT/TPOT单项退化；cold1024C1 formal稳定且未复现初smoke冷态偏高，不能用smoke代正式。见prefill-graph-v11-vs-eager-control-v10-v1.json。

**74225顺序已实际启动**：official-replay-v11-r1/r2（同A7两个源）→official-cont-prefill-graph-v11-r1/r2（180s/shape，同原bounds）→path-profile-v11（P11/D10 both，仅最后计时结束后2arming+stockproxy32，API stop）。此整链完成前不得切模型、追加profile、stage源或backup。源码新增--after-profile只是未来D12保留P11、其历史profile已停止时的显式gate，当前P11 timing不传此flag仍强拒绝profile文件；所有timing/helper AST hashes未变，parity回执已存。export手工revision guard已扩至12以支持P11/D12，非模型变化。

后续若保留P图并有余量，再D12 v3（当前仅staged）单独对照，P11必须保持。D12正式benchmark/continuous需--after-profile path-profile-v11，验证同P版本profile completed及所有roles stopped后才允许历史文件；新D12自身仍fresh。D10停后rank0 CPU export可趁D12加载空窗完成，完成后再正式计时。P11只profile一次，后续D12 profile仅D，避免P11多session造成rank0选择歧义。所有model仍19:21Z硬截止，holders19:31Z，不续租。



**17:43Z P11已实际launch，D10固定，当前suite28537等Ready**：D10两轮same-arrival均pass，r2结束后精确关闭P8（原CIDcc948f...），P11 CID **d82b591171b15560f4524fcf9a380537be82c10ee1a6a77df495d01f77d5cffc**，rootprefill-node32-v11，只改P target PIECEWISE/capture256,1024，无worker overlay，drafterMTP1仍eager。D10 CID6c89017... gate0原样保留；proxy/holders保留。P11计划在launch前刷新原expiryTTL，契约已frozen，未续租。

串行suite28537：prefill-graph-v11-smoke(3waves)→protected-c1-v11-r1→protected-c4-v11-r1→v11-reuse（新harness含4wavesHTTP1↔4切换）→v11正式r1/r2。任何失败即停止后续；当前不profile/backup。待正式结果判断，再排same-arrival2r/共同bounds持续2r及必要profile，不将requestedmode当实际图重放。

D10控制原件已Mac归档1124852B SHAa3e618aae7c9e2c3fadbf1fb20ad10a726dd74251ec3f3d808fcd4b25a9f6954。完整v2/on相对gate0/off same-arrival：256TTFTp50倍率0.97617，1024为1.00270；TPOTp50倍率1.02983/1.02091，925/925双方通过、各发送max≤0.527ms，结果见same-arrival-v2-v9-vs-gate0-v10-v1.json。只保留有限hot-C1 wave信号，不宣称v2普遍提速。

v3-singleton源已趁P11加载空窗stage33，SHA仍7eb57e...，**尚未launch/模型验证**。D12计划node33-decode-singleton-output-v12.json、离线契约singleton-output-loop-contract-v3.json已准备；referenceP必须等P图结果后填充冻结。guard handoff在已有latestfences被消费之后才观察latecopyevent.query，未实际完成不能关；status会记录[glm53/output-guard]日志。此源仅备选后续，P11图实验结果优先。



**17:37Z D10 wave/gold/reuse均通过，same-arrival r2运行（56732）**：两轮240/240，12题11/12；24 C4 PD/native各24/24，16reuse pairs通过。对同文件gate0D10，v2D9 hot C1 TTFT-8.20%/-7.53%，吞吐geo+1.98%、TPOTgeo+1.12%，有一个C4 TPOT+5.01%，仍不无条件采用。原先vsD6的13%退化有明显续写混合漂移：相同token序列TPOT只+1.46%/+1.83%，此离线分层是相关性、不替代原始无条件结果；见early-output-v2-token-strata-v1.json。D10 r1 same-arrival已过：v2比off 256 TTFT-3.93%、1024+0.83%，不泛化所有负载。r2完成前不得切P8/D10。

**下一现场动作仍P11图实验，D10固定**：全部replay完成且API无active后，精确停止P8 CIDcc948f194ff768021070ab1b0bac0cb3309606b60da6ca56bcea7a6b403cdd09（用r1/r2 replay gates），只刷新P11计划TTL到原19:21Z并launch。P11源图模块4SHA和实际image的vLLM捕获尺寸逻辑均已核验，后者SHA956b812e5a719bcbfa3a3958801361b38b9c928c96b8c073311bbb96376dfb7a。两capture256/1024会保留，实际runtime是否PIECEWISE仍需现场确认。P图后先smoke/C1+C4gold/reuse/两轮formal，对P8/D10正式参照；若有收益再同arrivals两轮与共同门槛连续两轮、最小P/D实际profile。若图不支持/资源失败就恢复原P，不做graph-size扫描。

**v3-singleton仅离线准备、未stage未launch**：fullsource /Users/wio/work/glm53-execution-evidence-backup/20261017-official-pd-optimization/early-output-v3-singleton/model_runner_v1.py，SHA **7eb57e7220c03d3c0b0780eed509fd28d98feafcefacb4a7df9292bfd895de32**，diff/manifest在runtime-patches/early-output-v3-singleton。只在实际singleton first-valid output提前copy，seen对所有有效req更新避免C4缩小误当首次。guard保留v2所有完成fence/ownership CPU wait，直到下一次原生late-copy event实际query完成；先消费最新pending fences再关闭额外guard，日志[glm53/output-guard]。这是一份未模型验证的有界原型，等P图固定后再决定D12对照，不能当收益。其patch已可从base568c字节精确重现并AST通过。reuse工具新加可选--transitions/--transition-waves8，检查1→4/4→1 max1/cancel后立即不同42回答；明确没有观测特定物理KV block回收，未执行新分支。status新加handoff日志字段。



**17:14Z D10开关关闭对照已launch，完整串行suite运行（session56732）**：D9两轮same-arrival 925/925功能通过，发送max≤0.527ms；256/1024 TTFT p50均值相对原无MTP为+23.25%/+13.01%，TPOT分别约0.277/0.282倍，明确首token代价。摘要same-arrival-v9-vs-baseline-v7-v1.json。D9在全回放结束及API空闲后精确停0；D10 CID **6c89017f3dfc68222100c15b48ef569260c057f5f076088afe1b1b64eb4d1efc** 已launch，P8保持。v2同sourceSHAa4a7...但GLM53_EARLY_SPEC_OUTPUT_COPY=0。TTL重新按原holder19:31Z减600s刷新，未续租。

D10 suite顺序：official-output-control-v10-r1/r2（先等Ready）→protected-c4-v10-r1→output-control-v10-reuse→official-replay-v10-r1/r2（同A7两个arrival源）。全部结束前不能切P8/D10或profile/backup。P11 graph计划仅离线准备；4个相关image模块的SHA核验已通过（node32-prefill-graph-image-source-audit-v1-bd93c7588d46），与匹配固定commit源码完全相同。待D10全suite完成再判定及切P11。

v9所有正式/保护/reuse/replay raw已Mac归档1059494B SHAdaf2f36128511c38da0818fc1168ef458297938b94ce858d56834cc2a699c17b。D6原始rank0 pre-export126files已在加载空窗完成Mac保全，66648476B原件压缩11768593B SHA9f722f7360d9dbff9e0634ba891d1874d365402fb1e4c56bf869e6b63676eb35，126项均在Mac逐文件核SHA；非新NPU profile。新的首output契约v2有继承source_manifest指向v1的文档旧字段，addendum明确实际v2 manifest/sourceSHA/launch证据，冻结原件未改，测量未受影响。



**17:06Z v9首轮同到达轨迹有效，第二轮运行中**：r1 256/1024共462/462请求通过，dispatch p95约0.042/0.041ms，max0.210/0.526ms，门槛通过。对原无MTP同arrival，TTFT p50为411.72→503.26ms、556.64→638.90ms；平均TPOT p50约41.03→11.33ms、43.00→12.08ms。收益有明确首token代价，不宣称同负载TTFT已改善。当前replay55201第二轮，不能切模型。

另已离线准备 **P11 PIECEWISE target graph** 单机制实验，计划node32-prefill-piecewise-v11.json、契约prefill-piecewise-loop-contract-v1.json，尚未部署。仅去P eager，capture总token256/1024两个必要覆盖shape，MTP1 drafter仍eager；现有kernel、精度、TP/SDMA/connector/proxy不变。匹配commit平台对PIECEWISE关闭Npugraph_ex/static/superkernel/use_inductor，保留现有FX fusion和ACLGraph；SFA attention仍eager boundary。先在D10加载期间跑CPU-only image module SHA audit，D10同P8两轮控制/保护完成后才能替换P。若capture/兼容/stream资源错误，保留失败即恢复原P，不展开graph-size参数网格。保持原holder expiry。



**16:59Z v9两轮完成，尚不采用，正在同到达轨迹回放**：240/240功能通过，原保护11/12不变，24 C4 PD/native及16 reuse pairs均通过。对早前stock D6，TTFT几何均值0.9691、吞吐0.9936、TPOT1.0202；1024 C1 cold TPOT+13.1%，故不能接受为单纯收益。D6该case每轮75 draft rounds/330接受token，v9 r1为83/313，变化值得用同worker gate0控制区分drift/时序/接受率，不能只猜同步成本。

**当前replay collector session55201**：official-replay-v9-r1(原A7r1 arrivals，前驱v9r2)→v9-r2(原A7r2 arrivals)，新绝对时钟driverSHA由collector冻结。全两轮结束前不切D9，不profile/backup。已准备但未launch的D10计划node33-decode-early-output-control-v10.json，v2完全相同RO worker SHA，唯一feature gate1→0，P8保持。需回放完成、无active/queued请求后精确停D9，再刷新原expiry TTL并launchD10，先两轮相同formal与保护，用同环境flag0对照区分v2负结果归因；不改核/精度/connector/proxy。



**16:55Z D9功能门槛已过，当前正式r1/r2计时中（collector33699）**：API Ready，16rank early-output activation，图replay正常，无inference errors。smoke60/60、原算术11/12（既有失败保留），24项C4 PD/native双方24/24，16组max1/cancel→立即不同答案replacement全部通过。正式r1/r2完成前不切模型、不profile/backup。绝对时钟CPU对照也已完成：12sec lookahead下原Python相对sleep迟发median12.039ms，新绝对wait median0.008675ms/max0.03654ms；node32-replay-clock-control-v1-25da53ff97cf。检查点b24d51be已push，PR2 body已更新，任务继续。



**16:44Z 已拒绝首token v1，D9第二版已launch，P8/proxy/holders保留**：v8两轮240/240功能通过，但相对stock MTP5吞吐几何均值仅1.0077、hot C1 TTFT无改善，1024 C1 cold TPOT+11.6%，不采用。24项C4 PD/native各24/24、16个max1/cancel→立即replacement pairs通过。D8精确CID停止，lifecycle回执保存；其所有已完成测试与失败回放原始请求已Mac归档913382B SHA373f0c0ec53ccc1124440630e485cfc89d69d7992c88e2f17bd10dce585915a0。

D9 CID **a860fbe99a982faa4137db64f0c0ecf1930f03f8193c5e6d7d53ab7852a74a83**，只读source `execution-patches/early-output-v2/model_runner_v1.py` SHA **a4a7f042642f4a4f4ce3d25897f8537377f9f22b9db637cef4eca135c1087e37**。请求ID seen集合代替可能回滚的optimistic长度；稳定decode用default/update stream等待旧完成事件，ownership变化CPU同步；每轮记录最新default/global fence，并保留旧event引用到后续output-ready。完整metadata仍返回前附上。D9为原型，未验收；P8 CIDcc948f...保持，expiry仍原6h holder减600s。

**当前串行suite collector session33699**：v9 smoke(3waves)等Ready → protected-c4-v9-r1 → v9 reuse16pairs → 正式r1/r2。任一失败停止后续。不要停止P8/D9或混入profile/backup。D8回放r1因driver普遍迟发约21ms失败，240/240模型功能过；r2因前驱失败自动退出，均无活跃进程。driver现改绝对CLOCK_MONOTONIC clock_nanosleep，先32-thread时序control，原p95≤10/max≤50ms门槛保留；未排新回放，待v9正式结果或stock恢复后排新run IDs。

共同门槛边界敏感性已用SHA核验Mac raw离线重算：原达标倍率2.8507，双方TTFT/TPOT门槛共同+1%为2.2577、+5%为2.2529，原始倍率2.2516不变。见continuous-common-bounds-sensitivity-v1.json；不要只报告2.85x。own analysis已更新MTP5/持续/反例/质量/SDMA覆盖边界，仍待v9与有效同到达负载结论。


**16:14Z持续对照完成，P8/D8首token调度原型已实际launch**：无MTP P7/D7持续r1/r2共925/925功能通过，扩展C4 native/PD双方24/24。对照MTP5持续2071/2071功能通过，四客户端输出率256为84.45→197.75tok/s、1024为77.61→168.03tok/s；等权倍率几何均值2.2516。预冻结共同门槛达标率baseline256/1024约76.7%/81.1%，MTP5约99.9%/99.8%，达标吞吐倍率3.0496/2.6647；baseline平均TPOT靠近门槛，应同时报告原始倍率和边界敏感性。实际TTFT及token gap p95在饱和C4下更大，不宣称全部延迟指标均改善。主摘要continuous-mtp5-v6-vs-baseline-v7-v1.json。

D6 profile rank0 CPU export v1因跨主机回执路径失败，v2改传经SHA核验的停止gate，return0。rank0摘要完成：452724事件、16200 SDMA任务、1个实际hccl_batchGet、无RDMA命名任务，矩阵LOCAL/SIO/HCCS；范围仅本次rank0窗口，旧全32rank地图复用。trace原始104903196B SHA508b8df36546012470597c7dedeee93ce43a64f30f05a40c74e83fbf9e7ce2eb，Mac gzip8654279B已核验，见node33-mtp5-rank0-trace-download-v1.json。

当前已精确关闭完成两轮及gold的P7/D7；P8 CIDcc948f194ff768021070ab1b0bac0cb3309606b60da6ca56bcea7a6b403cdd09，D8 CID3f51fdd82f7f9b54bc296739c3e8c901939c471b8fa1c4ed98a5586a5405d981，约16:08Z launch。P8仍MTP1无重叠，D8仍MTP5+共享重叠，**只新增一个RO worker patch及GLM53_EARLY_SPEC_OUTPUT_COPY=1**。原生connector/proxy/精度/算子不变。源diff/runtime-patches/early-output-v1；module SHA8b2cf6174a8d49a2c09ab145d194bbb43e0d8d6993e358921b9cc0aa225a0293。两份原始worker/async-output模块SHA在pinned image实际验证通过。lifetime在launch前刷新，仍原holder19:31Z减600s。原型尚未验收收益。

原型只对首个有效输出把原有D2H排在下轮draft之前，默认流等该微小读取结束防原地修改；最终ModelRunnerOutput在返回前附上完整metadata。下一次execute_model在任何input/KV/preemption复用前等待compute/global两流completion fence，保护跨请求生命周期，其他复杂特性走原分支。profile中8个PD verification轮的相关D2H间隔约13.5ms，copy本身约2.7us，仅作为实验预算。未新增kernel/copy primitive/传输协议。

**当前smoke已排队**official-early-output-v8-smoke（3waves，单独60请求，不作paired正式收益），随后official-protected-c4-v8-r1（24项×PD/native，同fixtureSHA）。collector sessions44961/47744。先确认实际first-token branch日志、gold与取消/max1立即复用，再排完整6wave r1/r2，对D6双轮reference；不因已写源码/PR结束。另需把A7固定到达轨迹在所选候选重放，比较同到达负载TTFT，避免仅凭更快闭环吞吐宣称更快首token。当前没有新的正式profile/CPU exporter运行。

**15:31Z持续MTP5两轮及最小真实D profile完成，无MTP对照P7/D7已实际launch**：MTP5持续r1/r2原件已Mac回收，四客户端每shape180s，全部请求功能通过；256达标输出约196.5/198.6tok/s，1024 r1约168.4tok/s，完整r2见收据。TTFT/平均TPOT使用预冻结共同门槛，尚未给与持续无MTP基线的相对容量结论。token成批返回的gap也保留，不将平均TPOT改善等同所有token间隔改善。

在两轮正式测量结束后，`path-profile-v6`仅启动D profile、两次native arming后经stock proxy生成32token，API stop成功；收据node32-opt-mtp5-path-profile-v6-b112627d0697。随后精确关闭P5/D6，关闭证据node32-cont-best-v5-lifecycle/node33-cont-best-v6-lifecycle。当前P7 CIDde3cee3098f375ad50a2b0b8c5254a77de859b7937392fca24813535042f13d4、D7 CID2341facb30bc379ca3a0eb06e5ba60fa1e00a4671faa5046435424cc40ebee5b，均原无MTP配置；lifetime在launch前重新按holder19:31Z减600s刷新。proxy/holders未动。

**当前D6 rank0 CPU export正在运行，尚未启动连续baseline计时。** 完成/停止CPU解析与备份后，再排队official-cont-baseline-v7-r1/r2（同180s、两shape、同bounds），r2后补同24项C4 native/PD保护，再恢复所选机制继续。旧D6的raw/profile在原目录，CPU导出只挂rank0及driver manager，不使用compute device，不重做全rank硬件地图。匹配公开固定SHA源码已核对：Ascend runner SHA568c3049与镜像一致；发现AsyncGPUModelRunnerOutput在draft后才wait_stream并D2H，可能影响TTFT，**仅源码假设，尚未实现worker patch，需当前profile确认及生命周期审查**。

**15:16Z MTP5两轮和C1/C4扩展保护完成，当前正在持续四客户端测量**：P5/D6各保持原实例。MTP5两轮240/240正式请求通过，原12题每轮仍11/12，无原保护题新增失败；同24项扩展任务在C1和C4下，PD/native D均24/24，fixture SHA完全相同。相对无MTP，波次输出吞吐几何均值2.3304、TPOT p50倍率0.2826；但TTFT倍率1.1408，C4部分case TTFT增加约19%–28%，不能无条件采用或当同延迟容量收益。见mtp5-v6-vs-*.json。

下一步已实际排队`official-cont-mtp5-v6-r1/r2`：P5/D6、256/1024冷前缀、64输出、4个连续闭环客户端，各180秒、两轮串行，取消wave barrier。共同实验门槛在测量前冻结为baseline两轮p95最大值：256 TTFT774.471ms/请求平均TPOT41.064ms；1024 TTFT1470.618ms/请求平均TPOT43.063ms。除数值/HTTP/长度通过外，只将达标请求计入timely_valid_output_tok_s，并保留token到达间隔/zero-gap比例和30秒吞吐窗口。不是业务SLO/全局容量声明。契约continuous-c4-loop-contract-v1.json。

**当前连续负载未结束，不能停止P5/D6或做profile/hash/backup。** 两轮完成后，计划必要的一次无MTP原配置重建P7/D7，运行相同两轮持续负载；再依据同门槛有效吞吐、TTFT、TPOT选择并恢复候选，完成最小真实请求profile。单rank CPU导出工具已准备，但尚未启动新profile；复用旧全32rank地图，不重做硬件微基准。proxy/6h holders始终保留，holder约19:31Z到期。

**14:57Z P5/D4角色隔离及扩展保护完成，下一候选P5/D6已实际launch**：两轮240/240正式请求通过，11项原保护题无新增失败。对MTP1的TPOTp50每case降低3.6%–4.8%，吞吐几何均值+4.12%（未超保守吞吐噪声阈值）；TTFT几何均值−0.77%，hot1024 C1两端重叠时的+23.5%退化消失，仅D重叠时对MTP1为+2.67%。冷256 C1仍有观察到+6.57%，而原对照该case两轮TTFT差16.92%，不把所有均值差当显著因果。对无MTP原基线，吞吐几何均值1.5694、TPOT倍率0.5533，仍有TTFT权衡。见decode-only-overlap-v5-vs-*.json。

新增24项固定数值/记录提取/2048和4096token检索，通过同一组token IDs分别走官方PD和native D，双方均24/24。原12题中的既有2+3=4错误仍单列保留，不能宣称完整模型精度。收据node32-official-protected-v5-r1-efce6a8b91ff；原始fixtures保留在远端tests/official-protected-v5-r1。

P-v5保持CID91dab72037316ac12f1a86918d9b664acc0f6951012fb306aa597197b3642d5c，D-v4在两轮及全部48个扩展请求完成、无active/waiting后按精确CID关闭。新D-v6 CID53102060219eeb0203b86efaae5e00340bdc2919d9f7bf3818edc73b58d5542d，仅MTP深度1→5，D共享重叠保留；P仍MTP1无重叠。固定官方参考P1/D5及MTP1约94%真实接受率支持这一次有边界的target摊销实验，不做参数网格。official-mtp5-v6-r1/r2已串行排队，official-protected-v6-r1在r2完成后执行；**不能在扩展检查运行时切换模型**。proxy/6h holders不动，profile尚未开始，正式计时后再做一次所选候选路径/执行验证。

**14:41Z第二候选完成，已实际切到P5/D4角色隔离对照**：v4两轮240/240请求通过，12题均11/12，无保护题新增失败。相对MTP1，8case吞吐几何均值仅+1.62%，未超原对照噪声门槛，不能记吞吐收益；TPOTp50每case下降3.5%–4.4%，但hot1024 C1 TTFT+23.5%。见overlap-incremental-comparison-v1.json。原生role histogram r1显示该case P prefill196.79→273.72ms（计算token同为256），D49.06→46.83ms；这支持把重叠仅保留D侧的可证伪测试，不是直接认定根因。

仅P-v4在两轮完成且running/waiting0后按精确CID关闭；P-v5无overlap已launch CID91dab72037316ac12f1a86918d9b664acc0f6951012fb306aa597197b3642d5c，D-v4保持CID8d175a3e6da5c01cd5b338a7f62f61c25b469b41ef25ba9d70ca30c39717cae6，proxy/holder均不动。MTP1/精度/传输/请求协议固定，只改变P flag。official-decode-overlap-v5-r1/r2已排队串行，benchmark明确revision4(D)/prefill-revision5，增加P launch存在性检查与等待状态展示修正；冻结计时函数未改。下一步完成比较并继续有效候选/保护验证，不因状态/PR结束。

**14:34Z现场恢复核验**：v4两端容器实际运行，P日志已确认`Shared expert split computation validation passed`，flag=True且未发现推理错误；正在完成初始化，正式计时尚未开始。r1状态仍显示waiting_predecessor是报告缺少转为waiting_ready造成的旧标签；前轮result已完成通过，进程实际在等双方API Ready，不是前轮缺失。只修下一版脚本状态展示，不改当前冻结实验/计时函数，不重复提交r1/r2。

**14:26Z首个机制对照已完成，继续第二个单机制候选**：MTP1两轮240/240正式请求通过，12题均11/12，全部11项baseline通过题无新增退化。相对无MTP基线，8case吞吐提升35.5%–65.0%，等权case倍率几何均值1.5073；TPOT降低40.8%–44.1%。但4case TTFTp50增5%以上（最多14.4%），所以结论为有延迟权衡的实测收益，非无条件采用/优化目标结束。见mtp1-comparison-v1.json。draft计数实际为正、接受率约94%，证明MTP实际执行。

热prefix每请求P/D少命中128token：256时128→0，1024时896→768。固定vLLM源码说明EAGLE/MTP drop最后匹配块以重建drafter隐藏态；本线不盲删安全guard。原生指标进一步显示hot P prefill真实计算128→256token，C1 hot P约193→229ms；D prefill_kv_computed_tokens仍0，不能把本地命中减少直接说成D全段重算。role-time-metrics-v1曾将token histogram误标ms，v2修正为mean_tokens，原件保留；模型/性能计时无影响。

第二候选v4仅在MTP1上开启原生multistream_overlap_shared_expert=true，P/D保持TP16无EP、MC2=0；源码TP路径确有stream switch，内置split/integrated一致性须通过。v4实际已launch：P d45d7ee1b05a77f1202ac505d34235b776d7a44f7c432d0b7884cb154f232faa，D8d175a3e6da5c01cd5b338a7f62f61c25b469b41ef25ba9d70ca30c39717cae6。原v3在两轮结束后按精确CID关闭，proxy-v2与6h holder不动。新两轮official-overlap-v4-r1/r2已排队，串行同8case/12任务；不重画SDMA图、不改精度/算子、不为备份/PR停目标。

**13:52Z基线两轮完成，首个优化候选已实际launch**：两轮共240/240正式64token请求的HTTP/长度/功能门槛通过，8case吞吐复测差异最大2.15%，额外算术均11/12。同一失败26token题D-native两次、官方PD一次都返回4；这是已有原生目标基线错误，不能单凭输出归因传输。正式基线摘要official-graph-baseline-summary-v1.json。

MTP1-v3已按唯一feature开关启动：P CID b1ee2e558be6c07527db10ada4e8bdbfdc48c5baaa60dfc1a28131df4e8dc548，D CID aaf99de0419cd3d5419125e07ed5acf04afe0d60382f8e608780e24224934694。v2模型仅在两轮计时及3个题目控制都完成、无active/waiting后按精确CID关闭；资源holder和官方proxy-v2未停，六小时归属不变。v3 P/D均native MTP1、eager draft、原target模式，其他能力/SDMA/用例相同；新engine ID防旧peer缓存混淆。lifetime按现有6h余量减600秒生成，目标约19:21Z。两个MTP benchmark已排队并严格串行，等待双方Ready，不能因当前阶段或PR完成停止。

**13:47Z首轮完整基线已回收**：r1=8case/120请求全部HTTP/64token/功能5通过；冷C1为23.0252/21.6302tok/s（256/1024），冷C4=80.1560/61.5100tok/s；热C1=23.5247/22.5788、热C4=79.3038/76.0428。首次连接/编译warmup TTFT15.105s单列，未计入正式；初始1024warmup TTFT476.95ms。完整本地收据node32-official-baseline-v2-r1-a323133ef485。第二轮串行运行中。

额外12题算术仅11/12：短26token few-shot提示的2+3实际返回4，属于数值错误，不是5/5.0格式差异，原件保留。不要通过放宽解析隐藏它，也不据此仅凭token认定传输根因。第二轮结束后只做这个失败题的少量D-native/同PD控制（正式计时外），明确原生目标/缓存边界问题，避免重启纯池或payload诊断。MTP对照保护全部baseline通过的题目，并报告完整数值错误，不宣称全模型精度通过。

**13:42Z首组基线已实测，仍持续优化**：official-baseline-v2-r1的256/64 C1冷prefix：6/6 HTTP/64token/数值5功能通过，23.0252tok/s，TTFTp50/p95=321.279/341.742ms，TPOT=39.0266/39.0721ms，P/D本地prefix hits均0。其余7个case继续，r2 collector2457按前轮result成功后串行启动，禁止与r1重叠；r1 collector14577。不将此与旧eager5.16诊断值算正式优化收益。

MTP候选已冻结mtp1-loop-contract-v1.json；两端同启1draft token、draft eager、target保持官方FULL_DECODE_ONLY，其余配置固定。权重索引准确文件名quant_model_weights.safetensors.index.json，含2345个layer78/MTP条目，不需要新权重；首次假定标准index文件名读取失败保留，改读取真实索引后通过。仅在r1/r2全套测完后按原生官方同MTP cache拓扑切换，两端一次必要重载；不因基线/备份/PR结束服务或目标。

**13:37Z已恢复实际运行队列**：新6h holders32 UID3b3ad135-298d-4a5d-96d1-b5070a34dc90、33 UID6049992b-4112-4bd2-9578-0a2bbde40314；客户端约19:31Z/北京时间10月18日03:31到期。P-v2 CID1188b5bb966e3180a682ce0222be0b538f8291844b36e692257bf74ebf638b0d、D-v2 CIDa0c8377377b387a4db19704d65b725dea74223c2f3e162eaf3c708edb743d3de已于13:31Z launch，模型21000s约19:21Z到期，官方proxy-v2已health Ready。baseline collector14577已排队，双方Ready自动开始：先两个连接/编译warmup单列，再256/64与1024/64、C1/C4、冷/热prefix，各6个封闭wave；C4是同步到达wave负载，不宣称无间隙稳态容量。正式时不做profile/hash/backup。最后12项算术任务检查与格式差异分层记录。

已吸收主控9562fe59离线profile审查：不把冷建连1.3s算带宽，不把41–44ms allreduce生命周期累计当关键路径；每周期157个TP allreduce、未重叠通信10–12ms/已profile周期是候选依据，不当无profile正式TPOT。源码核对shared-expert DP在无EP时被硬性关闭，排除该无效开关；multistream overlap另有TP实现但需实际分支证明，暂次于官方MTP1。正式权重config证实num_nextn_predict_layers=1；MTP先核P/D cache拓扑，倾向按官方两端同启一个draft token，target FULL_DECODE_ONLY、draft eager，单机制无新kernel。不得等待主控消息才继续。

**最新恢复，13:28Z（北京时间21:28）**：主控转达人类明确继续原优化目标；部署只是前置，不能在备份/PR/阶段总结后停止。旧P/D/proxy均按一小时timeout退出124，holder均DeadlineExceeded；两台各8卡空闲且无其他有效NPU claim，旧状态已记录resume-inventory-v1.json。正在只替换自己过期holder为21600s（6h），模型/官方proxy21000s（5h50m）留资源回收裕量；不再1h重载。基线v2仅改run/name/lifetime，原样V2/proxy/TP16/8K/W8A8C8/SDMA与既有权重stat校验复用，不重做全payload哈希、不重画SDMA地图。

下一段工作持续执行：先256/64、1024/64各C1/C4的官方图PD冷/热前缀可信基线，再利用现有真实trace选单机制候选并对照；首选考察官方MTP减少每输出token的target启动/TP通信成本。不得把5/5.0格式差异自动当整路线失败，功能/数值/任务精度分层记录；不得完全取消正确性。至少交付一次优化前后实测，有收益继续保护用例/迭代，无收益按证据换候选。只32/33、全SDMA、原生官方基础、能力/精度保持、无新kernel/参数扫描。若他人占用不抢占；维护自己的有限到期状态。历史“最终交付完成”段仅指此前部署阶段，不表示原优化目标已完成。


**最终交付：官方原生PD功能及32rank路径验证完成，服务保持运行。** P/D与原版官方proxy最后健康检查正常；两组真实请求各16rank完成远程KV READ；P/D共32份有效trace均有SDMA任务，D各rank有hccl_batchGet，32矩阵仅LOCAL/SIO/HCCS、无RDMA命名任务。结论限于本次实际profile窗口，不把计数当每byte归因/普遍证明。当前是两节点8K的官方直传PD基线，完整精度、长期稳定性、性能收益及MemCache池集成仍独立待验收。

P/D原始archive都已回传Mac并核SHA：P302,908,782B、`4fd60f8abf5faf882abb8c474009425f5d3f84cdd9b77282b7c92ac02584c715`；D416,348,365B、`bb0a10fb1ac4d9b01abb23436f60744f83932c834b179ed4501f259b3b1e36ef`。两次kubectl stdout的rc0截断坏文件和运输失败JSON保留，重传已有不可变archive成功，没有重做NPU实验。两端CPU离线export均return0，32份CANN输出完整；此前自动daemon解析失败已解决，非推理故障。无剩余设备实验或导出任务。

集群内入口 `http://172.16.12.32:29880`，served model `glm5`；P/D约12:37–12:38Z（北京时间20:37–20:38）有限到期，holder约12:52Z。没有关闭当前官方模型或proxy。最终收据为`evidence/20261017-official-pd/deployment-validation-v1.json`及`node*-official-pd-backup-v1.json`。120个工具AST、1906 JSON、313冻结脚本SHA及git diff --check已通过。原draft PR2已按实际官方部署重写标题和说明，不自动merge。

**12:06Z最终路径验收与收尾**：P/D32个rank全部有效CANN trace，SDMA任务P56,944/D437,740；D每rank都有hccl_batchGet，32份通信矩阵类型仅LOCAL/SIO/HCCS、无RDMA命名任务。真实请求窗口的有界路径证据通过，不把计数当每byte归因/普遍传输证明。两端CPU离线export均return0；自动daemon解析报错已按其提示离线处理，非推理故障。P侧“停止CPU导出”检查发现容器早已自动删除，未实际停止任何作业；D亦已自然完成。首次D摘要读临时SQLite文件遇导出删除竞态，失败保留，v2仅冻结完整有效CANN trace/matrix，避免派生临时文件竞态，无NPU重跑。

最终合成结果deployment-validation-v1.json；路径收据node32-official-prefill-path-check-v1-6c298f2f3f5b与node33-official-decode-path-check-v2-e8a08b218fae，D末次API仍Ready、完成READ48条（含profile新增16rank），仅automatic profiler parse日志错误、无推理错误。两份原始archive已生成：P302,908,782B SHA4fd60f8abf5faf882abb8c474009425f5d3f84cdd9b77282b7c92ac02584c715已Mac验证；D416,348,365B SHAbb0a10fb1ac4d9b01abb23436f60744f83932c834b179ed4501f259b3b1e36ef正在重新回传。kubectl大stdout首回传均rc0却截断，坏文件按SHA保留，重试仅已有不可变archive，writer flush+8s排空后P成功；没有重做设备实验。服务/holder保留当前有限窗口，未为备份停模型或代理。

**11:54Z当前已跑通官方原生PD功能**：两模型与官方proxy均Ready；256/16、1024/16通过官方proxy HTTP200并回答2+3。D日志对两条request_id各16rank记录成功batch_transfer_sync_read：256输入223,346,688逻辑接收字节、1024输入893,386,752字节（合计按16rank计，非去重线速）；D真实FULL graph capture15s/0.15GiB及replay日志均有，无native错误。已排除仅代理存活/仅D重算的假通过。256直接D与PD文本相同，1024存在5.0/5格式分歧，逐token和完整任务精度单列未验收，不再阻断官方部署。

原smoke客户端以Prometheus计数>0判失败的原件保留；匹配固定源码V2 pull_worker.py实际执行/记录成功READ但从不调用record_transfer或xfer_stats，因此该计数器为0不代表无传输。CPU-only validator验证镜像源SHA、两条原请求与32条完成日志、全部16rank/正字节/remote P32→D33及graph replay，已通过，未改官方源码或重跑请求。见node32-official-smoke-validation-v1-77036ee6dfde、node33-official-decode-transfer-proof-v1-97ff6b6f2a31。

真实服务路径profile已调用并正常stop两端，正式功能/计时不混入其开销。原始P约345MB、D约1.45GB；自动未导出trace，现仅CPU离线export（各16CPU/64GB、无compute devices，仅driver manager查询）进行中，先独立复制并SHA保全raw，collectors65200(D)/25376(P)，最长420s。没有新NPU微基准/模型重载或SDMA地图扩张。服务继续运行至各自有限3600s期限约12:37Z；下一步读导出trace核当前实际SDMA/HCCS、原件归档、PR更新。

**11:39Z官方原生PD已实际launch**：P32 CID `080a1e20b6403206286eaffd5eae4d784165197ba7cbb7f3eb1628b0665971ed`、D33 CID `dd2c0b5bf7c5fa21fd4a1cc5c2010927d00ff159adae0e1ec0e4538fae168e6a`均启动初始化，官方root `20261017-official-pd/prefill-node32-v1` / `decode-node33-v1`，API29481/82、served glm5。原样MooncakeConnectorV2（固定SHA5f828ad）+原样官方load_balance_proxy，proxy源码SHA85b89829已与镜像实文件核对。CPU-only官方proxy已healthcheck Ready，入口 `http://172.16.12.32:29880`，不代表P/D已Ready。P/D各TP16/PP1/DP1、8K/4seq/1024；P eager、D默认FULL_DECODE_ONLY，未沿用编译诊断纠正；全量化、AI_CPU/HCCS、INTRA_ROCE=0、LOCAL_COMM_RES1.2、只读/etc/hccn.conf保留，移除全部自写运行补丁、MemCache overlay及其config；无EP/MC2/DSACP/FlashComm/MTP。KV ports P36000..36016 /D36100..36116、engine_id各异。计划及匹配源码在official-pd-native-20261017，收据在evidence/20261017-official-pd。下一步双方Ready后通过官方proxy发实际请求，核remote-transfer与运行路径，不继续图逐token诊断。 官方proxy CID `4a8af0d5eea318ca9821b98eaa28b0a225fea3674a497f82e66ed0431849ce8c`。smoke collector10677等待双方Ready后自动发真实256/1024输入，先D-native再官方proxy，各16token；只对基础2+3功能烟测与真实D2D计数设门槛，原逐token差异独立记录，不作为路线阻断。Checkpoint f0a2f3d0已push PR2，主控analysis未暂存；模型仍加载，未做新微基准/诊断重载。

旧native均已关闭：32 eager Exit0；33 v2初始化期间主动TERM后timeout入口返回124、无action前错误、无OOM；其等待Ready的CPU客户端精确SIGTERM(-15)，未发模型请求。8卡空闲、无其他claim核对后只替换自己的旧holder；新4500s holder32 UID `34f87a93-f960-475b-8de1-a54f59eb9379`、33 UID `5e25d440-b7bf-4d4f-8ee8-a4da417c0e92`约12:52Z到期，P/D/Proxy各3600s约12:37–12:39Z有限窗口，不重复短超时加载。V2源码底层仍用同一Mooncake TE的ascend协议，旧析构133风险保留，不能称已绕过；先以实际官方服务结果判断。

**最新优先级，11:33Z主控转达用户「先按官方跑」**：立刻转向匹配v0.30.0rc1官方PD连接器+官方代理的实际部署。主控明确本轮官方基线必需的Mooncake直传覆盖先前「先不考虑Mooncake」阶段偏好；全SDMA、仅32/33、既有正式W8A8C8权重/固定镜像仍不变。不能假称官方四节点PP2/TP2×DP16在两台原样复现；采用两节点P TP16/D TP16、8K容量适配，IP/路径/端口/拓扑和已知C8/AI_CPU兼容项以外保持官方连接器/代理。停止自写纯池协议/同步补丁和逐token图诊断扩张。原60分钟图预算不再驱动新实验，启动结果或单一具体不支持条件优先交付。

当前动作：已开始正常关闭33 native-graph-v2（初始化中，取消等待Ready的collector43742，不再提交图请求），随后关闭32 native-eager；保留正式eager12/12、默认图真实replay和文本数值5与5.0的诊断差异，不将逐token差异当整条官方PD部署阻断。32/33有限holder仍为afd3a15d/0212957b，约12:24Z到期，未动其他资源。正直接HTTP取固定SHA官方GLM/通用PD/V2连接器/代理，现有Mooncake0.3.11.post1 READ双向payload已成功但析构133，默认不重跑旧探针；先核官方protocol/config/版本适配。未启动新的官方模型，下一步尽快生成并执行官方两节点启动计划。

**当前唯一优先交付：官方图执行原生C1对照（60分钟预算，11:02Z→12:02Z）**。主控最新转达用户明确要求：缓存集成诊断已投入过多，不再启动新PD诊断或分支，不等待长输入问题全部清零。已在执行的最小单B正常结束：同31key重新生成后，输出及P/D实际缓存、两侧私有池读回控制全部一致，仅保留为正面修复候选，未做完整回归/性能推广。所有旧PD模型P15/D14与Meta-v6均正常退出0；不再由历史Loop决定下一动作。

**11:31Z原生首轮实际结果/唯一纠正**：eager正式12/12通过，256/64=5.3976tok/s、TTFTp50=195.53ms、TPOTp50=184.17ms；1024/64=5.4743tok/s、306.00/180.50ms。33官方默认FULL_DECODE_ONLY捕获[1,2,4]且已出现真实ACL graph replay日志，但256/64首warmup第2token即newline198而eager为period13，64token文本5与5.0分歧，正式计时未接受。客户端首次父目录缺失在发请求前退出已修，原失败保留，与模型无关。按60分钟合同仅用一次具体错误支撑的纠正：固定源码确认enable_npugraph_ex默认true，下一实例仅设ascend_compilation_config.enable_npugraph_ex=false，保留FULL_DECODE_ONLY原生ACLGraph、TP16/SDMA/量化及其余项。该选择是可证伪编译路径假设，非已证明根因；再失败直接交付阻塞，不扩PD/参数网格。旧33图先保全正常关闭，新root native-graph-node33-v2，预算仍12:02Z。 新33 v2 CID `3bd1a84a9f4b160dfcd1d50494f4dc6d7b1d5326c81537a1bd52e39a2b730590`已于客户端11:29Z启动，lifetime1800s约11:59Z；旧33 v1关闭0。纠正collector43742等待双方Ready后自动同gate测试，max1600s但12:02Z硬工作期限优先。旧graph失败归档52517B SHA b39c943c626d1e6c78a50c5bf6deeecdd9690182717a598e742366c6ceee1a9e，eager正式原件87316B SHA15bd54738d05df1dce11f8a79172be15bbf112f5babd3f9cedddefbdf11666cc均已回传验证。graph v1关闭期间AsyncLLM EngineDeadError日志发生在主动TERM之后，关闭Exit0；不据此反称正式请求崩溃。官方PD文档主控新校正已吸收：当前纯Store路径是独立适配，不是Mooncake+Store官方组合的完整复现；不扩展PD实验。

**11:13Z原生实例/实测队列**：32 native-eager `51d352d33c01e751f33d3aa38a8814721f0fb8e4d3f06abc106b40927b8535c0`、33 native-graph `18a3ae9ffe500990f4e4dafb42575d00c6c245f06791cb1cf053a5c9bfb96d4d`均已启动初始化，root `native-eager-node32-v1` / `native-graph-node33-v1`、API29481/82，served name `glm53-w8a8c8-native`。完全移除KV connector、对齐/metadata/worker/backend/源fence等payload观察补丁及其env；同固定模型/seed/TP16/DP1/8K/4seq/1024批token、AI_CPU/HCCS-SDMA、必要C8关闭与原NZ/MLAPO/融合。仅graph移除enforce-eager并显式`FULL_DECODE_ONLY`，默认auto capture sizes，均加既有cudagraph-metrics。无EP/MC2/DSA-CP/FlashComm/MTP或新kernel。

同一Python调度器文件只有构造时一次info日志，SHA `f86d9f`，记录实际Scheduler类/async/effective graph mode/capture sizes/enforce_eager；没有任何step/forward日志或审计操作。两边profiler只配置、正式阶段不启动。补丁是读取有效配置的启动遥测，不改调度算法。正式客户端每模式在自身主机本地HTTP运行，先等双方Ready以免加载/初始化干扰，再逐模式执行256/64与1024/64，各1warmup+6measurement。保留旧native-prefix参考，另建立同冷缓存边界的新native-eager golden，graph严格比较该golden，不混入PD错误收益；两宿主放置差异必须报告。

新4500s holder32 UID `afd3a15d-6f48-4ec6-8227-61ca6ec8ac92`、33 UID `0212957b-1aad-4006-8f3e-32ca878bc1fa`均Running/16die，约12:24Z Mac到期；模型各3600s、约12:10/12:12Z。续期只在旧模型退出、两台各8卡空闲且无其他有效claim后替换自己精确UID，未停止其他作业。见`native-graph-reservations-v1.json`。60分钟工作预算优先于服务超时：只允许一次正式图配置起步，若失败至多一个明确错误支撑的修正；到预算须交付测量或具体阻塞，不自动回PD。`native-eager-c1-v1` collector45718已启动（等待双方Ready后执行，timeout1850），graph测量必须等eager完并取得其golden，不能两边并行正式计时。下一步只看native图初始化/effective/capture/replay/fallback和正确性。

**10:51Z用户最新优先级校正（主控转达）**：5.16 tok/s仅标为eager诊断基线，不能作为框架能力或独立优化收益基准。停止自动扩张PD诊断套件；已冻结但尚未启动的B/A/B缩为**一次最小B**，与已捕获的失败A同输入/salt/31key对照，保全后转官方常规优化，尤其图执行，不等待全部PD长输入清零。新脚本/contract已改为单B，原方案未在远端执行。PD最终目标保留；原生C1用于辨别模型执行开销，不混入不正确PD收益。

主控已核固定Ascend SHA `5f828ad638ee4f9b12db87f9d5481f29076d37f9` 的官方`docs/source/tutorials/models/GLM5.3.md`：Decode参考显式`FULL_DECODE_ONLY`，但其TP2×DP16/EP/AIV/Mooncake组合不可照搬。下一对照保持TP16、W8A8C8、seed、SDMA/HCCL AI_CPU与必要C8关闭项，只移除`--enforce-eager`并设置官方支持图模式；capture sizes≤4需按固定源码确认。必须核effective mode、capture/replay/fallback；不单凭argv报开启图。不要开启MTP、改精度、static/super-kernel、替换驱动或批量乱开所有融合。

只读当前argv确认P15/D14均`--enforce-eager`；MC2=0、DSA-CP=false、FlashComm1=false，是功能/生命周期/profile隔离配置，**尚无本线图启动失败证据支持长期强制eager**。不能因CLI没写而说async scheduling关闭，实际默认/effective待核。旧已核trace实际仍有MlaPrologV3、GroupedMatmulSwigluQuant、KvQuantSparseFlashAttentionVllm、QuantMatmul/IndexerQuant等融合算子；关闭上述三项不等于无融合。C8 reshape关闭是已有P首请求`StoreKvBlockMetadata→rtsRegisterCpuFunc`挂起及固定文档的必要兼容措施，保留标准C8/Indexer精度；不能随图一起盲目打开。

**10:28Z新判别实例已启动**：P15 `02a0afe89e095e166d3a4bf8bdee497f9e6a4aa52d8c71fdff4dbc6694bf163d`、D14 `962082beac1f632b1fce2de76e3babac52b4a4dbd61203c973071fc71f6f97fd`正在初始化，root `pd-node32-v15` / `pd-node33-v14`，API29481/82；新Meta-v6三端口已Ready。各3000s，新3600s holder32 UID `f45dad7a-6ba1-435a-97e5-1077e169f7df`、33 UID `aa49d45a-5fa3-4e84-9f56-f75a9d51f12f`已Running且16die。旧holder32正常删除时30秒等待超时、33尚未删除，已记录实际空闲状态并只按精确旧33 UID恢复，未碰其他claim；见recovery receipts。新资源约11:24Z Mac到期，模型约11:16Z到期，先查实际状态。

v2后置池观察同时挂载P0/D0，相同backend `a0e95b`，root `post-pool-readback-v2`；v1 D-only `01fe3f`只stage、未挂载，保留。P/D各用独立7MB NPU arena（int8 fill(-128)，0x80），在首次D128完成后的既有postkv-source/target请求读同31key，再算小整数控制；观察不改真实KV目的地。源码注册/注销模式与已测SDK一致，所有初始P1均无源屏障。额外SDK GET每次31/每端点单独记`pool_readback`，不能把model-loader计数0叫作没有诊断GET。哨兵可区分未写目的区与读到全零；控制矩仍非密码学全内容证明。

`post-pool-state-control-v1` collector27781已启动，timeout1600、client1500、Ready等待600，先4条A，不在Ready停止：三条历史salt+一条新key。保留原D golden、前D13 native和新D14 native，P16副本、D16实际缓存、P0/D0独立池读回共同判别。新启动只为加载这一必要只读模块，不做参数网格/性能套件。检查点`e81522e2`已push原PR2，包含此前强反例/原始归档和v1计划；本v2启动/代码在追加提交中。主控仍掌握方向预算，无跨聊天主动消息。

**10:21Z当前现场**：P14、D13、Meta-v5均已主动正常退出0，完整模型已无运行进程；本轮所有请求/控制/生命周期已SHA回传Mac，见`node*-source-ready-final-v1-backup.json`和`node*-source-ready-close-v1-backup.json`。当前剩余问题是已存对象/稳定GET路径，未宣称修复。下一Loop已冻结`post-pool-state-contract-v1.json`与P15/D14计划：仅4条初始无屏障A请求，D rank0在后置目标请求把31个池对象SDK读到独立6979584B int8 arena（0x80哨兵），115 entry四矩；注册/注销沿用已通过小测API，不读回真实KV bulk到CPU。D新backend `01fe3f`已stage并核SHA，其余模型/算子/精度/SDMA与worker/meta观察不变；源fence仍仅按已有request_id选择，初始4条均关闭。此启动模块变更是重载的必要原因。

原1h租约在两台8卡空闲且无其他有效NPU claim核对后，仅替换自己精确UID，为该判别Loop再取得一次3600s有限资源；不停止其他作业，不增优先级。新租约结果见`post-pool-reservations-v1.json`（若尚未生成则仍在取得中），Meta-v6/新模型尚未启动。下一次启动仍须核租约、空卡、固定镜像、权重stat和只读patch SHA；主控可继续调整预算/方向，不新增看板/任务或跨聊天主动发消息。

**10:15Z稳定外部重读反例**：历史salt重放3条仅2/3匹配；旧content坏请求再次失败，但坏块变为block22、entry104–114，821248B/每D rank全零，P16副本仍一致且与独立D-native相等。是同请求复现，不是同key/同entry固化。两机`source-ready-final-v1`不可变归档已回传核SHA；`known-failure-content-final-v1.json`保留176条entry/rank差异。ABBA6A/6B全控制匹配，不证明源fence有效。

当前OpenAPI无`/reset_prefix_cache`，未启用新API。用41条普通4096/1原生D请求产生1312完整块，淘汰旧本地缓存；实际D1288块，冻结脚本沿用旧参考1289，41对两者都足够，contract已更正。源P空闲、D不写池，淘汰阶段SDK GET与本地prefix hits均0。随后同bad salt D128新增GET496、本地hit0，却全部128token精确复现坏结果；再D-local8命中3968、GET0，top20及token logprobs逐值相等。见`node32-same-key-external-reread-v1-*`。已排除仅随原D本地驻留的问题；剩余是已存对象或稳定GET路径，输出不能独自证明池字节内容。

下一必要判别是在D128完成后的现有后置请求，用SDK经SDMA把同key读入独立7MB NPU arena，仅D rank0计算四矩，比较池读回、真实D缓存、P副本及D-native。复制前不观察内容，不增kernel/精度/端点/RPC/热更新平台。该观察必须启动只读模块更改，才需要下一次重载。先冻结CPU实现和SDK注册/注销API；P14/D13当前仍活，约10:26Z自身超时。当前循环原始证据先保全、正常关闭，再只在两台空闲且无其他有效claim时取得一次有限资源，不停止其他作业。主控仍可调整方向/预算。

**10:00Z ABBA结束且不采用候选**：12/12输出与后置控制均匹配，A6/B6均通过、无对照故障，不能证明源屏障有效。P16设备/选择/FIFO审计已通过，D逐请求/计数已冻结，仍待最终不可变内容归档复核。继续复用P14/D13，冻结`known-failure-key-replay-contract-v1.json`，仅3条A重放历史原salt/key：P13后置wave3全零块、P12最终FIFO的wave2/9输出失败，初始P本地命中须0；D仍真实3968，之后P/D-local控制，当前和原D参考分别保留。没有改模型/精度/算子/参数、无重载、无性能套件；P465/D496≤原512控制预算，复用原D-native控制，不新增参考。该Loop能区分确定性key触发与时序/启动历史，不能独自定责SDMA；原零内容反例继续有效。

**09:55Z中途判别**：8/12条已完成（A3+B5），所有输出/后置控制仍匹配；第一组A尚无反例。只读源日志已观察6条B共192次PUT前屏障，实际current_device均等于tp_rank，等待min/median/max约0.054/432.9655/1214.223ms，无原生错误；这包括未完成设备计算/后续chunk工作，绝不是SDMA/网络时延或全局关键路径，不能相加16rank。B不是空干预，但没有故障对照则无修复/收益结论。仍按冻结顺序收末组A，不改参数/精度、不加性能套件；最终audit与归档为准。

**09:49Z开始实测**：P14/D13已Ready；当前D-native128仍匹配原D golden。ABBA前两条无屏障A已完成，P/D本地命中均3968、额外D GET0、输出和后置31key×16rank控制均一致（CPU只读汇总`source-ready-content-progress-v1.json`，无新增NPU观察）。12条仍按冻结顺序运行，A尚无反例，不能因通过而宣称屏障有效。最终全量内容/设备与选择日志/外部命中/SDK计数、两机归档后决定；不新增性能套件。源码API已验证当前设备all-stream同步，发送线程set_device；逐rank实际设备仍要审计。此前原始内容归档回放`post-consume-replay-analysis-v1.json`精确复现block18、每rank66entry共4077824B全零，源与独立D-native全部一致。

**09:37Z本轮已启动**：Meta-v4已确认自然超时退出并保留生命周期，Meta-v5三端口已就绪、3600s。P14 `b98d4fc61254edbaaacd63023d7dbc714c9a6ba82525feea39dc096f30ea8a84` / D13 `4ee8fcb56c860f2a23246549ab1da2889c4adbc727be8d2022cde5ea5be49b83`已在新UID下启动、尚在初始化；启动重新核8卡空闲/权重stat/镜像/只读patch SHA。模型3000s，同后置observer和FIFOguard；source-ready-v2 patch363547。`source-ready-abba-v1`已启动（collector43743、timeout2500），有界等Ready后12条A3B6A3，不在Ready停止；没有pre PUT/GET内容读取。代码和此前否定/全零内容证据已提交push `da83f101`，不包含主控analysis修改。当前未宣称修复/性能收益。

**09:36Z源就绪ABBA冻结**：原过期资源仅在两台8卡空闲且无其他有效NPU请求后重新取得3600s有限预留，32 UID `8058e02a-abe3-4c77-93ca-539946844948`、33 UID `8e163546-e858-4c61-955a-cba01b6229c4`，均Running，见`source-ready-reservations-v1.json`；未停止其他作业。P13/D12已关闭0，下一部署P14/D13尚未启动。冻结`source-ready-abba-contract-v1.json`：A3 B3 B3 A3共12条，唯一B变量为P1已有request_id标签选择在SDK PUT前`torch.npu.synchronize()`；原event/FIFOguard、后置观察、模型/算子/精度/SDMA不变。kv_transfer镜像原c7af0a→363547；非标签对照也记录CPU日志，复制前不读KV。Meta-v4须确认已退出再启动新v5，3600s；模型3000s、测试2400s。若A不复现或B仍坏，不能宣称修复，不扩完整性能套件。新源码/计划正在stage，启动须再次核租约、卡空闲、镜像及patch SHA。第一次stage因旧审计副本多一尾部空行触发SHA断言，未写patch/启动模型；已比对镜像代码正文一致，v2固定精确镜像字节并保留失败。

**09:29Z明确内容反例与下一判别Loop**：后置4条输出全匹配，但wave3的logical block18（tokens2304–2431）在D16个rank均有连续entry49–114、模型32–77层及对应Indexer、4077824B被观察为全零；P16个副本均一致且非零，独立D-native在这66项与P完全相同。其他目标key/rank控制值匹配。本轮排除了这些样本中P副本不一致的解释，也说明输出相等不能替代KV内容保真；不能由后置数据独自归责PUT、GET或消费后的改写。证据`post-consume-content-analysis-v1.json`、`post-consume-entry-differences-v1.json`及两台`node*-post-consume-final-v1-backup.json`已SHA备份Mac。

P13/D12均已正常关闭0。原6h资源已DeadlineExceeded；09:26Z核对两台各8卡无进程、无其他有效NPU请求。准备仅在仍空闲的原两台重新取得有限1h预留，验证单变量“SDK PUT前源设备全部stream就绪”，不是再试PUT后的FIFO等待；不停止其他作业。既有后置观察与FIFO匹配边界guard保持，按现有request_id对源P1选择未加屏障/加`torch.npu.synchronize()`的ABBA，对照原D golden、P-native、实际3968命中与后置内容。若未捕获对照臂异常或屏障臂仍有零区域，则不宣称修复；暂不扩完整性能套件、无性能收益。先冻结/复核API及源码，再加载；主控继续掌握方向/预算，普通工程动作不加逐步审批。

**08:58Z当前Loop/现场**：最终FIFO候选12条收齐，仅9/12匹配P本地/原D golden，P本地12/12稳定；全部D确实加载3968、P每条16rank完成事件齐、SDK无错误/type2。三条失败2/3/9在D本地再次128token仍精确复现原坏输出，额外SDK GET=0、每条两次本地命中合计7936，故差异随已加载缓存保留；见`rejected-final-put-mechanism.json`和`node32-failed-cache-reuse-control-v1-*`。不推广、不跑该失败候选完整C1门槛，不把它当性能收益。

P12/D11已正常关闭0，失败请求/源码/审计已SHA回传Mac（`node*-final-put-rejected-v1-backup.json`）。现在P13 `928d7de3cd8b48c6c3982c8968c7fefaf1d0a550a4dbace5863de0b4d1113a70`、D12 `2027f4ae13f5ba0c01f68744e32c0dd711167f880c7438582bb55884acd1d47d`正在加载，root `pd-node32-v13` / `pd-node33-v12`、API29481/82、同world32 Meta-v4与6h UID。各1200s，约09:14/15Z Mac到期，原资源约09:18Z Mac到期；执行前重查。只为启动时挂载新的只读后置观察而重载，当前不是已验收服务。

新观察不在首次PUT/GET前后读KV：现有标准request_id已实测可用（response `cmpl-glm53-postkv-capability`）。仅在D128完成后的P-native128与D-local8请求用`glm53-postkv-`标签，读取已缓存前3968token的31key×115entry四矩；P可比较全部16副本、D比较持续保留的本地缓存。保留原D golden；每次须验证local hits3968且额外GET=0。三处冻结模块worker cf7c38、backend09d86b、metadata d71851，两台SHA一致；scheduler仍48387a，模型精度/算子/SDMA不变，无新API/RPC/文件触发或热更新机制。最终FIFO仅作为已知负结果下的匹配边界诊断guard，非采用修复。`post-consume-cache-control-v1` collector已启动，有界等Ready后跑初始4条，整体900s；不在Ready停止。一次共享SSH通道配额拒绝了stage32下载，未重跑stage，已串行恢复并核全部SHA。后续SSH尽量串行。

**08:35Z反例，候选不能推广**：即时交接重复至少8条时，P-native同盐3968前缀参考8/8仍等于原D golden，而D池消费有2条第3token分歧（含5.0重复格式）。P的prefix_cache_hits总计与原生复用次数×3968吻合，不能把P参考误叫冷重算。前三条16rank均记录最终PUT FIFO完成、源SDK无错误/type2，但完成fence尚未解决长输出一致性；不可宣称修复或性能收益，不对失败机制跑完整推广门槛。先收齐12条及每条真实命中/SDK计数、完成事件，保留反例。当前P12/D11为诊断候选，不是已验收配置。最小后续应区分源KV副本/写就绪、搬运与消费状态，不能用token差异独自归责SDMA；先读固定源码/已有证据，不猜测加同步或改精度。

**08:24Z当前执行**：P12/D11均Ready，readiness99162完成0；同UID双16die预留再次核查Running并写`final-put-control-lease-v1.json`。正在运行12条`final-put-immediate-control-v1`，collector84965、timeout1000s。先建立当前D-native参考，再每条P1→立即D128（证据写盘也放D之后）→P-native128，前两条local top20。无源key轮询/固定等待/观察钩子。接下来逐请求3968命中、SDK计数/错误、16rank source完成事件审计；必要C1边界/取消恢复回归后追加提交。尚未宣称候选通过，实例仍有限至约08:58Z Mac。

提交检查点`42b6116f`已push到原分支并更新draft PR2；包含本轮基线、真实KV/数值状态诊断与当前候选代码/计划，未把在跑候选写成已修复。后续结果追加提交，不改主控analysis目录。08:20Z P12/D11的16个worker均已启动，仍在初始化、未Ready，暂无原生错误；readiness99162继续有界等待。

**最新08:17Z现场**：P11/D10均正常关闭0；诊断请求、真实KV控制已冻结，原件在`node{32,33}-diagnostic-final-v1-backup.json`对应Mac归档并核SHA。P12 `95623b287047e8054863673673a2d13fe05608ae035d0d67ad91499a155c46ac`、D11 `579479baeaf6d4a3194ec7929990d2935799b1b8fbad0b80f2ccc31ad2dc95cf`已启动，正在初始化/加载；root `pd-node32-v12` / `pd-node33-v11`，API与world32 Meta-v4不变。每实例2700s，约08:58Z Mac到期；同6h UID，launch已重查Running/16die/设备空闲。没有payload观察或per-TP保存；P唯一候选为已有FIFO完成fence72ff56，D恢复原worker/backend+对齐scheduler。readiness collector99162有界运行中，Ready后执行`pd_final_put_suite.py`：12条即时P1→D128，再P-native128参考；原D golden/当前D-native单列，不静默换参考。前两条额外比较P/D池local top20。SDK错误、实际3968命中、16个source rank完成事件另审。当前不作机制收益或完整精度验收。主控独立analysis已提交1b78e0d9，本线可继续提交自己的文件。

**08:11Z诊断与下一Loop（优先于下面现场）**：v3无D checksum的4/4同完成边界128输出一致；numeric6/6的8token输出一致但top20最大差0.87248、chosen最大0.17888。追加复用最后同salt的最小对照：P-local、D首次pool、D-pool-local所采top20逐值完全相同；D-native与其重复逐值相同，但与P状态不同。见`numeric-cache-state-analysis-v1.json`，不能继续把原生P/D状态差当SDMA错误。当前正冻结最后source/D控制，之后只为移除启动观察与启用最小交接候选重载P12/D11，有限2700s、无audit/per-TP键。候选`wait_for_final_prompt_put=true`只在最终prompt块等待已有batch.done FIFO事件（worker原3d39b8→72ff56），不固定sleep、不改模型算子精度/SDK/SDMA；事件表示处理结束，不是耐久成功ACK，必须另审SDK错误/实际D命中。验证立即P1→D128时3968命中与原固定参考，同时在D之后用P本地128作为匹配状态参考，不能偷偷换golden或把不同参考合并成“全正确”。若不能保持预定命中/输出则拒绝候选；先完成真实测试，不停在Ready。

**现场08:04Z（以下历史段落不可作实时状态）**：P11 `3d154bff72037bf2e3d11c1ee6a0bb23fd74aa7cf4f92835dbc3ee53e787b6e7`、D10 `c497efe6c455941ff503705714c024914c188b1682385dfe89d7d1c6353883d4`已Ready，API29481/29482，root `pd-node32-v11` / `pd-node33-v10`。原world32 Meta-v4保留27400/01/02。模型有限3600s、约08:31Z Mac到期；08:03Z重查两台6h预留仍Running，UID32 `2cd9108a-eaf0-4164-856b-968bb65b79c6`、33 `ffd5ca30-273d-4646-aa79-7b35478b9416`，约09:10Z节点时钟到期，执行前重查。仅本线服务/资源，不停止其他作业。

恢复原共享MLA键+whole-block对齐，未启用已拒绝的per-TP保存；只加`GLM53_KV_CONTROL_AUDIT=1`观察。worker原3d39b8→b24869、backend原cf2d7a→e0a6f4，scheduler48387a，32/33 staging一致。P PUT前/后在NPU私有stream计算每entry四整数矩、D GET后计算，真实shape/stride/dtype/format记录；只回传控制数组，KV主路径仍device_sdma。控制矩非密码学payload SHA/字节相等证明，观察有同步与时序扰动，不作性能验收。ACL bulk方案仅staging未挂载，不用于真实KV读回。

**当前测试**：`payload-control-v3`四轮4096/128运行中，collector79446。与已完成v2使用相同的真实32key PUT完成guard及同轮bare native-prefix参考；D每进程128key预算已耗尽（文件计数核对），本轮无D checksum计算，P观察仍开。结果决定是否还需要D侧同步诊断。下一步`pd_numeric_boundary_control.py`已准备：等相同真实源key完成后比较6轮4096/8的top20概率，单独标记其不同输出长度/观察条件，尚未执行。不能在Ready/四轮通过处停止，不新增热更新平台，不为只读归因重载。

`payload-control-v2`四轮完成：128源key PUT前/后稳定，1984消费key×rank、每key115条entry的四整数矩全部一致，32rank布局一致，4/4 pool输出等于同轮native。见`real-kv-control-analysis-v2.json`及两台`node*-payload-control-v2-backup.json`，完整源/D控制与请求已SHA回传Mac。v1使用错误默认NONE_HASH种子，0/32key匹配且尚未D GET时失败；改为现场PYTHONHASHSEED=0后复用实例，失败保留。

`payload-observer-budget-control-v1`12轮ABBA完成：native12/12稳定，pool6/12一致。逐请求连接发现立即交接6轮均只加载3072token且输出不同；等待1秒6轮均加载3968且一致。此脚本没有保留v2源完成guard，源审计拖慢发布导致缓存边界混杂，**不能用失败归责D观察停止/传输**。`observer-budget-boundary-analysis-v1.json`保留全部12条；当前v3修正该混杂，不采用固定sleep作为修复。

SDK默认同步完成源码已冻结于`sdk-completion-source-receipt.json`：Python→native BatchGet调用及MemFabric无ASYNC默认等待分支。`KVCacheStoreBatch.done`有FIFO完成事件，wait_for_save目前只enqueue；源码另有释放请求保存fence，不能把异步enqueue直接叫内存安全问题。只有同完成边界的控制结果支持时，才考虑最终prompt保存完成交接候选；当前未改保存完成语义。

模型无关小测：7/31key、512buffer/key、每对象6979584B、异stride、双向one-batch/逐key全部SHA一致、padding未改、close0；只排除该向量小测故障，不证明真实框架stream顺序。1MiB合成ACL async trace明确PCIE_DMA_SQE，真实KV bulk读回已拒绝；NPU int8/BF16四矩小测匹配独立CPU整数参考、trace有AIVEC/MIX无CPU fallback名。上述6份原始小测归档均已SHA回传Mac；一次SSH multiplex会话上限失败的0B partial保留，串行重试通过。世界2小测Meta已正常关闭0，原world32不动。

供主控边界预研的自然产出：`real-kv-layout-capacity-v2.json`记录当前每rank54,528B/token、P1108/D1289个128token块、原例外/scales保留；不是普遍最小状态或服务最大长度。`profile-decode-structural-reference-v1.json`从旧512ctx profile离线提取rank0 B1 decode：78层MLA/attention、5 BF16+16量化Indexer、75组MoE、157all-reduce+1all-gather，少量真实shape/dtype和源SHA。旧profile有扰动，303ms窗口不能替代当前190ms TPOT；次数不是普遍必要资源，SIO/HCCS与SDMA计数有重叠，不相加。无新增NPU采样/重载；主控新analysis目录由主控提交，本线不纳入。

本线继续同PR2，原HEAD7d33725；本轮尚未commit/push。只使用现有HANDOFF与冻结evidence记录，不新增任务/看板。长前缀和C4尚未验收，没有机制性能收益，不改模型/量化/既有算子或SDMA要求。

### 已完成且可接受的基线

`evidence/20261017-pd-baseline/accepted-short-medium-baseline.json`已完成客户端、native命中、SDK总计数和资源覆盖审计。P9/D8完整模型TP16+TP16，固定8K context/4seq/1024批token、eager、`c8_enable_reshape_optim=false`、`mlapo_keep_prefill_weights=true`、loader-fix-v3、whole-block Python对齐补丁；1GiB/rank DRAM池、0HBM、无SSD、纯MemCache。没有启动profiler；正式计时期间没有权重传输、payload全哈希或证据归档。

临时固定C1负载，每组两轮、每轮6条测量，另各轮1条warmup。256输入/64输出：12/12一致，5.1607输出tok/s、TTFTp50/p95=404.34/415.70ms、TPOTp50/p95=190.91/192.79ms；1024/64：12/12一致，5.1606tok/s、TTFT=521.70/542.88ms、TPOT=188.38/190.57ms。所有28条（含warmup）D external_hit=need_load分别128/896，SDK expected=actual=1792get keys、448rank-get，平均rank-get约3.66ms（非全局关键路径）。D无原生错误，16个客户端bmDataOpType均2。5s资源样本各480条，P/D内存约118.3/120.8–120.9GiB，HBM每die约59.9–60.2GiB；资源窗口覆盖基线，但不覆盖较晚的已拒绝首token候选。没有生产SLO/容量或全局最优结论。

原始请求/响应、失败、resources和生命周期console已归档Mac并核SHA：`/Users/wio/work/glm53-execution-evidence-backup/20261017-pd-baseline/node{32,33}-completed-pd-tests-shared-replica-v1.tar.gz`；receipt在本evidence根目录。首次完整模型profile的32-rank SDMA/TP LOCAL/SIO/HCCS证据复用，不拿旧profile耗时当新shape成本。

### 严格正确性边界和实际否定结果

- `correctness-v4`的12项C1单次保护用例通过：127/128/129/255/256/257边界、1024/4096输入、repeat/local命中、stop、256-token部分共享前缀；同缓存边界的native local-prefix与池消费token IDs/text一致、所采logprob差0，冷P/D一致。冷full与缓存路径在部分格式决策不同，独立保留，不宣称完整精度验收。`correctness-c1-gate-v1`追加真实流取消：8token后断开，约0.253s观察running/waiting/delayed-release归零，恢复输出一致。
- C4原始严格输出3/4不一致；随后原生D local-prefix的3轮重复中2轮也有变化，不能归因MemCache，也不能验收C4有效吞吐。
- `baseline-c1-v1`长4096/128重复：6条测量中3条偏离固定native-prefix参考；全部7条（含warmup）external_hit=need_load=3968，无SDK错误。原生D fresh-prime/local复用4次一致（logprobs=5），之后不带logprobs的Pnative→Dpool3组也全一致。长前缀问题仍未闭环，不能推广其计时。
- `handoff-append-first-c1-v1`：P首token立即交客户端并追加到D输入，D生成63个剩余token。warmup即严格输出失败，已拒绝，不计收益。D确实加载1024token，朴素896token，SDK240keys精确吻合，说明机制实际发生而非未执行。详见`rejected-append-first-mechanism.json`。
- `long-handoff-abba-v1`：立即/等待1秒按ABBA各6条，立即6/6一致、等待4/6一致；等待不能修复。固定镜像wait_for_save只enqueue且_previous_save_batch无读者的源码事实不构成根因证明，不采用猜测性同步PUT补丁。config的Indexer topk=2048；4K重复上下文的稀疏选取/量化副本为可调查因素，尚未证明。

### 已拒绝的per-TP保存机制（历史，实例已关闭）

P10/D9只增加`preserve_mla_tp_kv=true`，每TP rank保存原C8 KV并按rank键消费，检验共享副本假设，最多16x逻辑PUT/池存储成本。12条4096/128匹配当前bare native参考的ABBA：native12/12等于原参考，pool8/12（立即4/6、等待1秒4/6）。因此拒绝该机制、不跑完整C1推广门槛、不记收益，见`rejected-per-tp-mechanism.json`。候选代码/计划/失败原件保留，P10/D9正常关闭0、资源采样停止，不是当前服务。

metadata-only探针64个key（4block×16rank）均存在，每对象6979584B，NPU initialized始终false、close0。该证据只证明独立rank键实际建立，不校验payload、不推定逻辑PUT字节均跨机。`conditional-work-accounting.json`与真实布局保存当前图字节计算，非普遍理论下界。之后的小SDK层batch和真实KV控制观察见当前段。

## 前一轮基础部署结果（历史）

纯MemCache P写池→D读池已在完整GLM5.3 W8A8C8/Indexer-int8、TP16+TP16、512 context下通过一个新480-token用例；16-token输出一致，D从池加载384token、本地命中0，外部get keys增加48。不是P→D直传，不是吞吐/精度验收。

上一轮已结束的功能服务：32 P `glm53-memcache-pd-32-fixed-v7-20261017`（29481）；33 D `glm53-memcache-pd-33-fixed-v6-20261017`（29482）；32 metadata `glm53-memcache-meta32-production-v3-20261017`（27400/27401/27402）。只读叠加loader-fix-v3隔离MemCache符号；P关闭官方已知冲突的C8 reshape可选优化。模型、量化、既有算子及SDMA约束保留。D API Ready不是单独的PD证明，上述外部命中与输出对照才是本次功能证据。

服务是本次有限期实验实例：P宿主启动00:57:43、timeout3600秒；D宿主启动00:30:47.691、timeout4500秒；仍受两台7200秒资源预留期限约束。不可把本文当作长期可用保证，继续执行前重查现场。模型profile已完成全32-rank核验；两端大文件备份均已回传Mac并验证归档SHA。最终结果摘要见 [deployment-validation.json](evidence/20261017-memcache-fix/deployment-validation.json)。

## 本线已实测

证据目录：[20261017-inventory](evidence/20261017-inventory/)。后续以每个 run 的 `context.json`、`result.json`、原始 stdout/stderr 和校验值核对；失败也保留。

- 32 宿主机 `B4-33U-Atlas900RCKA3-32`，33 为 `B5-5U-Atlas900RCKA3-33`；K8s 节点名仍为 `a3-5u-atlas900rcka3-{32,33}`。访问 Pod 的 hostPID/hostNetwork 和宿主挂载命名空间已核对。
- 两台均为 8 张物理卡 / 16 die，逻辑 ID=Phy-ID 0..15，每卡两个 die；npu-smi 每 die 65536 MiB。设备内 SIO、不同卡 HCCS_SW；两台全部32个die逐一查询均报告 superPodID=1，server index 分别31/32；各die SDID已留原始证据。同超节点不证明跨机性能。
- CPU：Kunpeng 920 7280Z，4 socket×80 core×2 thread=640 逻辑 CPU，8 NUMA，宿主约 2 TiB RAM。NPU PCI sysfs NUMA=-1，CPU 亲和仍未知。具体网卡/链路/限制见 [HARDWARE_MAP](HARDWARE_MAP.md)。
- 两台实际镜像均为 `quay.io/ascend/vllm-ascend:v0.30.0rc1-a3-arm64`，imageID `sha256:090eeee66e347509127601e565b971a0801701e32e4e865d89cc8dec72ca7d83`，arm64，RepoDigests 为空。
- 镜像：Python 3.12.13，CANN 9.1.0，torch 2.10.0+cpu + torch-npu 2.10.0.post4，vLLM 0.30.0+empty，vLLM-Ascend 0.30.0rc1；Mooncake 实际发行包为 `mooncake-transfer-engine-npu==0.3.11.post1`，并非未安装。vLLM 源码 SHA `ced6857afa0ea7b2e3f0846a62e1394e90f15607`，Ascend SHA `5f828ad638ee4f9b12db87f9d5481f29076d37f9`，镜像内两仓库工作区干净。宿主驱动 25.5.2，采样固件 7.8.0.7.220。
- 正式权重 `/data/wio/GLM53/GLM-5.3-w8a8c8/` 两台均已存在，incoming 已不存在；196 文件 / 764240832702 bytes（含 `optional/quarot.safetensors`）。当前逐路径 stat 和文件大小匹配，校验清单规范化 SHA `b6843ccf0ffee3cd22357f443dec1006107f958225d13871edc19bebfc88f3be` 与 `WEIGHTS_READY.json` 相符。复用了传输流水线的全量 payload 校验凭据；**本线未重复全量 payload 哈希**，不把 stat 说成独立 payload 校验。
- 量化描述含 `W8A8_DYNAMIC:176830`、`INT8_DYNAMIC:88`、`FLOAT:644`；config 为原 GLM `glm_moe_dsa`、78 层、256 专家/top8、hidden6144。目录名/静态元数据尚不证明运行时 W8A8C8、KV 存储或实际算子路径。
- 33 单 die0 无权重检查通过：容器只见 1 个设备，整数算子输出正确；torch 可见总内存 65787658240 bytes。仅环境功能检查，不是全卡、模型正确性或通信性能证据。该检查发生于旧预留释放前，采样无 NPU 计算进程；不能据此声称当时已取得独占。

## 资源状态及已执行变更

初始 K8s 证据中，32 有 testportal 工作负载请求 2 个 Ascend（逻辑 1、12），33 有 `default/sglang-pd4p2d-reserve-33-20261014` 请求全部 16 个。33 无预留的早期判断是漏读，已纠正；首次新预留被调度器拒绝后已删除，没有绕过进行通信测量。

用户随后明确两台全部可以 kill。已核对 32 的 Deployment `testportal-project/containerworkload-694f3t4n` 只有一个副本且仅位于 32、无对应 HPA，将其副本数 1→0；已删除 33 无控制器的旧预留 Pod。原 Pod 删除完成，系统及访问 Pod 未动。执行凭据：[workload-release.json](evidence/20261017-inventory/workload-release.json)。恢复原业务需另行决定，不能在实验期间自动恢复。

本任务 `default/glm53-execution-reservation-{32,33}-20261017` 首次预留均已 Running/Ready，每台请求16个Ascend、有限7200秒，约07:51 CST复核均为Failed/DeadlineExceeded。此次修复续接已删除本任务过期Pod并重新取得两台16-die预留，见下方“MemCache 崩溃修复与模型部署”的新UID和取证。**每次 NPU 执行前必须重新核对实际 Running/UID/节点/资源归属，不能只看本文。** 集群 admission 拒绝 Pod 直接指定 `preemptionPolicy=Never`；已记录失败，改用集群默认策略，节点 affinity 仅限指定节点，未创建集群级 PriorityClass。

## 前期 SDMA 与单机验证（历史）

- 已核对官方 A3 配置语义：`HCCL_INTER_HCCS_DISABLE=FALSE` 为超节点内 HCCS/SDMA；`HCCL_INTRA_PCIE_ENABLE=1` 配对 `HCCL_INTRA_ROCE_ENABLE=0` 禁用 RoCE；A3 `HCCL_OP_EXPANSION_MODE=AI_CPU` 避免选择 AIV 展开。配置与运行证据分开，具体支持及 communicator 级覆盖仍要核对。
- [hccl_probe.py](tools/hccl_probe.py) 已完成同卡 33[0,1]、异卡 33[0,2] 及跨机 32[2]↔33[2] 的 int8 双向 send/recv 与 bf16 all_reduce；五档字节、预热3+计时12次/项的 payload 校验通过。INFO 诊断日志与 ERROR 级计时分开保存；只有前者用于选路分析。初始缺少 PCIE 配对、CONNECT_TIMEOUT=30低于实际最小120、容器物理/逻辑可见号混用三次失败均保留，未把它们记作性能数据。初始化/控制 TCP 不作为数据通路；不能由 backend 名或环境变量宣称已证实 SDMA。
- **实际 PD connector 的 SDMA 选路和 C8 消费路径尚未运行验证**。HCCL 通过不能代替这项。不要把原来的 AIV/MC² 或 RDMA 配置无审查搬过来；全部 SDMA 约束下，逐个核实实际通信实现。暂停的 Mooncake 探针经过 READ 地址顺序修正后，两台双向64KiB/1MiB/16MiB数据校验完成，但两边析构均出现 `corrupted size vs. prev_size`、退出133；不能记为稳定通过。初次33设备初始化失败发生于模型仍占用die2时，空闲复测已通过初始化，未证明两容器共享设备可用。没有启动 Mooncake PD 服务。
- 已取得本任务资源，基础2-rank跨节点 SDMA 通信功能通过。实际 ACL 映射已核对；容器挂载物理0,2后应使用容器逻辑0,1，不能再用物理0,2筛运行时可见集。跨机测试无 HCCL_LOGIC_SUPERPOD_ID 或外部 rank-table 环境变量覆盖，使用 HCCL root-info/TCPStore 动态建域。继续按真实模型消息补相关组与争用。5 秒背景窗口无明显传输/磁盘读取，只是瞬时诊断；每次测量仍核实相关干扰。
- [TP16 单机容量计划](tools/capacity-tp16-node33.json) 已完成一次完整模型启动和短请求；容器 `62481295791bd1a9e086d5a833cac59a3832a654aa252d79fbbf44e5637bc012` 已主动停止、退出0，保留容器和日志，当前不提供服务。固定 TP16/DP1、无EP、MC2/DSA-CP/FlashComm关闭、int8 KV/indexer、V1、eager、512 context/1 seq。全部16个rank从 `model_runner_v1.py` 加载；每rank日志权重47.21GiB，warmup后torch allocated54.54GiB/reserved54.62GiB（软件口径，非物理HBM），KV预算约7.33–7.35GiB。宿主日志API Ready18:36:16、容器启动18:20:29，加载186shard约477秒；启动耗时仅同宿主时钟范围。短请求HTTP200，12输入/32输出token，以“5”开头回答2+3；单请求功能验证，不是精度或性能验收。原始console183178bytes已拉回Mac校验。实际量化method/缓存tensor布局与C8消费仍需运行对象或trace核验；CLI dtype不是证明。单机容量结果不等于1P1D验收。

## MemCache 初次核实（修复前历史）

- 镜像内 `memcache_hybrid==1.2.0`（VERSION git `4310d68fa8964a3032b71c09296b0ffa80330033`）、`memfabric_hybrid==1.2.0`（`999034e9320abf27bbde7be83a32d3bf4d0f7940`）。SDK API 已加载检查；未安装替换包或编译新kernel。MemFabric的导入provision分支只为950自动编译，本线A3不走该分支。
- 实际 `AscendStoreConnector` 注册的 `MemcacheBackend` 使用 `memcache_hybrid.DistributedObjectStore`，支持显式 `backend=memcache`；SDK配置支持 `device_sdma`。现有PD文档示例是MultiConnector，仍含Mooncake，所以不能用它证明纯MemCache PD。源码中单独Store消费者的 `consumer_is_to_load` **默认false**，纯池消费必须显式true；先按同TP/MLA/Indexer布局核对，不能直接照搬异TP稀疏布局。
- SDK1.2.0提供 `put_from`、`batch_put_from_layers`、`batch_get_into`、`batch_get_into_layers`、key-info/GVA、lease/写完成接口；不等于全部语义已经验证。`COPY_L2G=0`、`COPY_G2L=1`。已核对配置与API签名，后续以真实返回码、payload和运行选路判断。
- 32本任务元数据/配置服务 `glm53-memcache-meta32-20261017` 已主动停止，当前不监听；未占NPU或数据池。运行期间172.16.12.32:27400/27401/27402为metadata/config/metrics，不能叫TCP数据回退。初始12秒就绪观察过短记失败，容器未重启，随后三端口及SDK日志证实MetaService启动（约26秒）。SIGTERM后的服务清理日志已完成，但进程随后同样报heap错误、最终退出134，不能视为干净退出。配置/最终console在 `/data/wio/glm5-3-vllm-execution/20261017-memcache/meta32-v1`，不改全局配置；关闭后的完整归档已备份Mac并核验SHA。
- 空闲32/33物理die2已执行两rank最小合成检查：每rank临时1GiB DRAM池、0HBM池、显式device_sdma、SSD关闭。64KiB/1MiB双向共4对象的P写池/D读池均返回0，接收SHA全部匹配。实际HYBM初始化`bmDataOpType=2`，已与镜像内`HYBM_DOP_TYPE_SDMA=1U<<1`头文件核对；profiler两台各4个`SDMA_SQE`，每个put/get标注区间恰有1个SDMA任务。数据链为源HBM→源侧DRAM池→对端HBM，每对象两次SDK搬运；这不是P→D直传或模型KV消费的证明，也不是建议引入永久缓存层。Trace、日志原件及SHA已备份Mac，仓库留 `memcache-trace-analysis.json`。不拿profile耗时作性能基线。
- 生命周期未通过：仅隐式析构的case退出139；显式`close()`返回0后仍退出133。先释放tensor/清空torch缓存再close也复现。进一步32单客户端只init/close、不建tensor/不注册buffer/不传payload，torch allocated/reserved均0，仍退出133。具体根因未定，不能仅凭此归责到某个库或宣称SDK已稳定集成。第一次LocalConfig拒绝仅MetaConfig支持的`ock.mmc.log_output_target`，已删除该键后固定配置重测，未改协议或回退；所有失败保留。
- 指南要求A3 `device_sdma` 配套灵渠计算网络>=1.5。宿主HDK/CANN版本已取证，**现场灵渠网络版本仍未知**，不从同超节点标签或SDK包版本推定满足；本次正面证据只覆盖已测试的数据路径。纯Store的原GLM C8/Indexer注册、完成通知和D消费仍待模型验证。下一步先对最小init/close复现定位原生退出/动态库生命周期与版本配套，再用单独AscendStoreConnector、`backend=memcache`、`consumer_is_to_load=true`验证P写池→D读池。没有安装替换依赖，没有启动双完整模型，也不回退已暂停的Mooncake路线。

## 官方适配文档核对（最初查询）

用户要求先查 Ascend 官方适配文档。通过 Web Search 与公开 GitCode Git 获取了对应标签/文档，未使用浏览器，未编译、安装、替换依赖或启动服务/NPU作业。来源、固定提交、原文SHA与结论边界见 [official-docs.json](evidence/20261017-memcache-adaptation/official-docs.json)。

- 官方现行[完整底软配套](https://gitcode.com/Ascend/memcache/blob/c3ab6e59caba5f1df5e34222d05df5212da1a6cd/docs/zh/compatibility.md)对A3/HCCS device_sdma要求：灵衢L1>=1.5.0（优选1.5.1）、HDK>=25.5.1、固件>=7.8.0.6.201、OS kernel>=5.10.0；BIOS按内存容量另列要求。现场HDK25.5.2、CANN9.1.0及采样固件7.8.0.7.220达到已核字段的数值下限；L1版本仍未知，未由小测成功推定整套适配通过。v1.2.0 MemFabric README另明确A3 DRAM池化需配套1520L1升级。这里只记录要求，不操作交换机。
- 两个现场VERSION SHA均精确匹配官方v1.2.0标签。MemCache标签的MemFabric子模块固定为`9a9cfbc8a1e026e7c73a558ff27d75372cb2e803`，与独立包`999034e9320abf27bbde7be83a32d3bf4d0f7940`仅两文件的rank-leave/link-down日志及leaveRank_成员改动；ACC/ConfigStore及公开BM头未变化，不能把SHA不同直接判成配套故障。实际二进制构建ABI与加载路径还未核实。
- 官方[issue281](https://gitcode.com/Ascend/memcache/issues/281)/[MR536](https://gitcode.com/Ascend/memcache/merge_requests/536)记录编译期/运行期MemFabric C++ ABI或小版本差异引起C++符号抢占、MetaService coredump，改为纯C ABI与DEEPBIND隔离。源码核对：现场v1.2.0及本次release/1.2 tip1.2.1尚无此隔离，master1.5.0已有；不把新分支修复视为已适配本镜像。上游主要描述启动时string-copy崩溃，现场主要是close后堆错误；目前是相关线索，尚非根因。下一诊断优先实际SO/ELF/ABI与最小case原生栈。
- 已核对v1.2.0 API明确提供close()、device_sdma及1GiB池对齐，所读安装/API文档未找到能解释当前堆错误的特殊关闭顺序。vLLM-Ascend固定v0.30.0rc1指南中A3 fabric环境变量与现场一致；纯Store D消费仍需显式consumer_is_to_load=true。当前没有修复或纯MemCache PD验收结论。

## MemCache 崩溃修复与模型部署（2026-10-17 续接）

用户要求直接解决基础部署问题，已实施隔离修复并继续模型接入。证据：[20261017-memcache-fix](evidence/20261017-memcache-fix/)。

- 真实进程 maps 确认 wheel 原名库与 auditwheel 哈希名库重复加载；原生栈将 close 后 heap 错误定位到 CANN/ATB 全局析构期间的 free。基线 MetaService 可启动且 main 返回0，但进程退出133；仅合并重复库仍退出133。单纯 import 库本身退出0。
- 最终隔离副本 `loader-fix-v3` 保留原1.2.0二进制，将四个 libmf 原名路径链接至 wheel 内 canonical 哈希库，并把 MemCache/UBSIO 的 ctypes 预加载改为 `RTLD_NOW|RTLD_LOCAL|RTLD_DEEPBIND`。通过 PYTHONPATH、LD_LIBRARY_PATH 和只读挂载使用；未替换基础镜像、依赖版本或算子，没有绕过 close/析构。对照证明符号隔离能消除当前复现，尚未确定被抢占的单个 C++ 符号。
- MetaService 连续两次启动/停止/进程退出0。32 原先只 init/close 也失败的 NPU 用例现退出0。32/33 物理die2双向64KiB/1MiB共4对象：put/get返回码、SHA全部通过；两端 close 返回0且进程退出0；每端4个 SDMA_SQE，每个put/get区间恰1个。两个 `memcache-trace-analysis.json` 和归档SHA已保存。这是SDK数据路径及生命周期通过，尚非模型PD验收。
- 两台隔离副本均在 `/data/wio/glm5-3-vllm-execution/20261017-memcache/loader-fix-v3/site-packages`，库校验值一致。小测元数据服务 `glm53-memcache-meta32-fixed-v3-20261017` 已停止且退出0；其ConfigStore会保留首次连接的world_size=2，不能直接供world_size=32模型复用。已重建为 `glm53-memcache-meta32-production-v3-20261017`，仍为27400/27401/27402、有限7200秒，已接受32-rank并挂载P端16个池。首次重建前置bind检查受TIME_WAIT影响，已用SO_REUSEADDR修正；未夺用其他监听端口。预留已续建，32 UID=`a29da17a-3c31-40ca-9f4c-7f72d61c0c27`，33 UID=`7a546c86-8e53-425d-a30e-190454720288`，本次核对Running；仍须执行前重查，有限7200秒。
- 正在部署32 producer、33 consumer：完整GLM5.3 W8A8C8及Indexer-C8，同TP16/DP1，纯AscendStoreConnector/backend=memcache、D端consumer_is_to_load=true、device_sdma、1GiB/rank DRAM、无SSD。v3为补请求级profiler而停止；v4因PYTHONPATH覆盖CANN路径而导入acl失败，已改为保留原值并增加启动前模块定位检查。D的v5因上述world_size旧状态失败，现用 `tools/memcache-pd-node33-fixed-v6.json`，已API Ready且12输入/1输出token请求HTTP200（输出空格，非答案验收）。
- P的v5已API Ready，但首个14-token请求180秒超时。py-spy定位在attention mask构建；进一步GDB原生栈确认Torch队列实际阻塞在前序`aclnnStoreKvBlockMetadata`的`rtsRegisterCpuFunc`，并非已发生KV写池。对应源码`indexer.py:1152`受`c8_reshape_optim_enabled`控制，`ascend_config.py:933`仅对producer启用，公开配置键为`c8_enable_reshape_optim`。D端该分支有效值为false。P已正常停止、退出0，现用 `tools/memcache-pd-node32-fixed-v7.json` 显式关闭该可选reshape优化，继续标准C8缓存写算子；W8A8C8、Indexer-int8、SDMA和模型语义保持。没有编译/引入新kernel。新P已于宿主01:07:00 API Ready，首个真实请求HTTP200（12输入/8输出token，文本以“ 5”开头，19.40秒仅为功能记录）；证据 `node32-memcache-pd-standard-c8-smoke-5f271d0ca7e0/metrics.json`。D保持运行，API端口29481/29482。`memcache_pd_functional.py` 已通过完整模型P写池→D读池功能检查：新480-token提示词、16-token输出，P/D文本完全一致并以“ 5”开头；D外部get key计数0→48。D日志明确vllm_cached=0、kvpool_cached=384，说明加载3个完整128-token块并正常处理余下96token，没有整段重做prefill。两端profiler均正常停止。功能证据 `node32-memcache-pd-functional-v1-600eeb949b16/metrics.json`。模型SDK32个rank均bmDataOpType=2；全32个rank的Ascend trace均有效且含SDMA_SQE，32份节点内TP通信矩阵均只有LOCAL/SIO/HCCS。没有再发NPU请求。当前仅此合成用例功能通过，尚无完整精度/长上下文/并发稳定性或性能验收。
- 官方固定标签[additional_config文档](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0rc1/docs/source/user_guide/configuration/additional_config.md)明确建议池化/AICPU冲突时关闭`c8_enable_reshape_optim`；[issue15896](https://github.com/vllm-project/vllm-ascend/issues/15896)补充HDK26.0以前受影响。现场25.5.2和实际注册栈与此机制吻合，关闭后首请求通过。文档原文字节SHA与摘录见 `official-c8-aicpu-adaptation.json`，不升级驱动、不改变C8精度。

## 完整模型运行证据（本次最终核验）

[deployment-validation.json](evidence/20261017-memcache-fix/deployment-validation.json) 汇总有界用例及所有来源SHA；`node{32,33}-memcache-model-profile-v2-43b402dac851` 保留日志、trace SHA、矩阵摘要及派生计数。

- 32 P 的16个rank共480361个SDMA_SQE，33 D共409744个；每个rank均为正，RDMA命名任务计数均0。每端16份通信矩阵均仅LOCAL/SIO/HCCS；这些是模型TP通信与KV池搬运混合采样，不能把全部SDMA计数归给KV或拿profile耗时作收益。
- 实际量化调用：P的`aclnnQuantMatmulWeightNz` 77520次、`aclnnLightningIndexerQuant` 4864次；D分别65280/4096次。是运行路径证据，不等于完整精度验收，也不能把CPU API与设备kernel计数相加。
- 最初采集器假设已导出trace，实际只有raw profile，保留失败；CPU离线导出第一次因缺少驱动查询设备而失败，且会生成无效/不完整派生trace，未晋升为通过。原生`msprof`单rank对照证明只需`/dev/davinci_manager`查询接口。最终离线容器未挂载davinci0..15计算设备，使用固定镜像和CPU完成解析，全32个rank导出/JSON/SDMA检查通过；无需重跑模型请求。
- 最初原始采样32共3074文件、33共2722文件，已在`profile-raw-preserved`硬链接保全，逐文件与初始manifest的大小/SHA全部匹配，防止解析器data_simplification清理原始采样。原件及派生结果均已备份Mac：32归档1632329001bytes、SHA `ae8e0699a2112bcc1ed34d0b150c610568f7df1687a0bfb397be391ff34d9dc0`；33归档1366352967bytes、SHA `48c928e4053d095361e203efc91497a63285772eb20471906ce8478d8dc62cee`。各自`profile-backup.json`记录Mac路径和完成时间；旧P关闭console亦已单独压缩备份并匹配原始SHA。
- 最终资源快照 `pd-final-resource-check.json` 于Mac01:38:20Z复核两台原UID、16设备预留仍Running，metadata亦Running；`node{32,33}-memcache-pd-final-running-4e69af062ba9` 复核两个模型API与容器。交付前再次通过 `pd-delivery-api-check.json` 核对两个API和metadata仍可用。有限期限沿用上文，后续继续必须重查，未延长期限或创建监控任务。

## 前一轮收尾状态（历史）

Mac 2026-10-16T19:17Z重新核对两台本任务16-die预留仍Running，UID分别 `dfb6ef4d-746f-41ba-a163-20034e9b237f` / `d4d731d9-2d8b-43e1-b986-621b66fa7adf`，7200秒有限期限。仅sleep holder继续保留供续接，**不代表后续时刻仍有效或剩余时间足够**。证据 `final-task-resources.json`。`node{32,33}-final-device-idle-*` 均退出0：两台所有8卡均无NPU进程，本轮模型/Meta容器均已停止，系统和访问入口保留。初版收尾脚本把K8s sleep holder误归为运行实验容器，已按容器名前缀修正后复核，原始失败未删除。

## 操作入口与记录

已有 SSH 复用 socket `/Users/wio/work/sglang-glm53-pd/ssh-control` 已验证，可到用户指定入口，再经既有访问 Pod 进入 32/33 宿主。未找到用户所称 KeyPair-27.pem，当前无需为此停工；未关闭主机身份检查、未使用口令或保存私钥。连接记录不当作长期凭据保证，socket 失效再安全定位。

Mac 固定脚本→上传 `/data/wio/glm5-3-vllm-execution/20261017-inventory/<run>/`→运行/保存原始证据→拉回 Mac 校验。`collect_node.py` 检查 imageID 和本任务资源；结果记录命令、脚本 SHA、git SHA/差异摘要、时间、退出状态。大 trace 原件保留目标节点与 Mac，仓库留摘要和校验值。宿主时钟相对 Mac 有分钟偏差，不调系统时间；跨机不能直接相减 wall clock。

Zcode 入口已确认 0.16.9，常规采集/脚本实现已交其处理并经主控审阅修正；未读取/打印 provider 配置，实际 provider 型号未在本线独立核实。未创建自动化或新 goal。

资料归约已并入 HARDWARE_MAP：CloudMatrix384论文仅取§3.3/4.2.1/4.3.2/4.3.3与CANN9.1正文。论文AIV/独立RDMA平面不当作本线可用能力；全部SDMA约束下优先研究发起/等待、缓冲复用与KV对decode通信的竞争。没有照搬全超节点带宽或EP320结果。
