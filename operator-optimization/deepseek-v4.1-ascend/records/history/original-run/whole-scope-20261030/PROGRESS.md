# 整体优化机会分析进度

整体未完成，Goal active；26个暂定模块中0个通过全部覆盖门槛。文件、AST或报告规模不作为完成比例。

当前主线按用户最新指示：先充分分析既有profile、算子、调用栈和事件关联；profile能定案直接判定，不能定案才明确具体缺口并定点补查源码。停止把整库代码审阅作为前置任务。

冻结源码底账沿用573文件、47个选定局部合同、19条局部状态边；这些是局部进度，不是全部语义闭包。当前机器为资产10.55.0.166（实际105），CANN9.2.0；用户接受驱动/固件差异。

166新增实测：完整C8/32768输入/1024输出/235904÷262144前缀命中验证通过；单次参照511.546 token/s，不是稳定性或提速结论。single/TensorList图重放观察接口真实通过3/8/1及别名覆盖前快照检查。

有界MoE观察完成并恢复原服务健康：4736快照，296个完整16rank组逐专家路由守恒，覆盖40target+3draft。正式C8阶段能直接绑定87组、prefix seeds169组；40个captured replay组阶段仍未由Python-entry时间证明。全部值的接口合同在观察域内验收，未声称全迭代/分支/消费者覆盖；旧169时间不能与这些新计数拼成同调用数据。

首轮模型观察因脚本未证实的512行断言失败，原记录保留；修订版已成功取证与恢复。当前无在跑的设备采集。

当前Task活动Loop为loop-004：Profile主导的全模块融合候选证据分级。恢复入口为166协调目录RECOVERY.json；下一步复用profile-summary.json、完整模块/算子表和F01–F32证据，不重复整库索引或无限扩充观察器。


本轮Profile优先复核：旧16rank/481548行/87type、官方双窗980002行/87与83type分域汇总，48项计数检查通过。聚合与门槛见 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/README.md。未新增目标源码审阅或设备工作；候选/栈/flow复核尚未结束，Loop004继续，整体未完成。


Profile分级v2：32候选/26暂定模块均登记证据入口。F07/F08/F12/F20选定切片已复核；F05/F09仅共享输入量化变体已看汇总；F30/F31新增176个主图段间记录关联。官方32个sample按原始CSV SHA唯一绑定16rank×2window。21项仍只有family入口，整模块关闭为0。详见 /Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/PROFILE-FINDINGS-v2.md。下一步定向检查F06/F10/F16/F32现有投影/通信记录。


Profile v3：16rank全部1280 target scope的输出两投影对与shared expert三任务链已核。F16导出间隔20–41ns；F22 activation→down最长1.7652ms待查。按profile缺口仅补3冻结源码文件，target40层分36非KV-source/3C2-source/1C1-source；F32与F17/F19的消费者依赖已明确。F06/F10/F16/F22/F32纳入更细证据，本轮16项family入口仍待查，全部候选/模块闭包开放，0整模块关闭。/Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/PROFILE-FINDINGS-v3.md


Profile v4：96C2与32C1源scope确认，F17/F18/F19局部layout、ring WAR/WAW、BF16/RMS及双cache发布合同复用。原始EP3 DB已在166迁移归档定位并SHA084b...核对，F22长/短、F17长3个区间端点精确匹配：长间隔无对应WAIT类TASK，不解释为真实idle/可删时间；host API间隔分时钟计算。C2联合接口只给10240→5120 B/行条件账。本轮13项细粒度材料尚待纳入，全部候选/整模块闭包开放。/Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/PROFILE-FINDINGS-v4.md


Profile v5：128 key链候选、1280 merge/HC及postattention布局候选已查，7明确API/栈锚点定位调用位置。F23 runner Tensor返回合同与外层legacy tuple分开，保留scaling/reduction/rounding及context-aware replay语义。F15显式host431394→device102545→COMM247补出AlltoAll，侧车零计数为关联缺口。3冻结源码文件仅按profile问题定点审阅；本轮9项细粒度材料待对账，0整模块关闭。/Users/wio/work/166-vllm-analysis-continuation-20261030/profile-first-20261031/PROFILE-FINDINGS-v5.md
