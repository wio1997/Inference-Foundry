# 第24代：HCCL原微测出处与计量口径审计（阶段完成，待root验收）

本代闭合原HCCL独立微测的原输出、采集包络、部署绑定及历史端口失败，没有重跑性能矩阵、消融、模型、服务、NPU或SDK。仅使用任务private transport读取29份小文件并逐项SHA/bytes核对。新进程仅为已结束的CPU读取/分析，没有外部后台任务。启动argv安全字段核对为gpt-6.1-sol/high；实际服务端模型身份未暴露。

两套既有容器各四rank，7个BF16 AllReduce形状和2个FP32 AllGather形状，共72个rank/case。独立核算7200个eager事件、280个graph replay事件的形状/正确性/摘要/20调用摊销。所有72行原逐rank中位数与此前comparison精确相同；AllGather无graph。每case每rank的graph只有5个回放事件，各包含20个collectives，不是100个独立graph样本。没有把rank求和为墙钟、没有抹去等待或换成物理带宽。

原采集时钟：vLLM 2026-10-25T13:42:12.590872578Z至13:42:37.496756796Z；SG 13:42:37.714258735Z至13:43:17.829576305Z。这是controller保存的docker-exec启动到退出包络，非逐设备collective绝对时钟、非模型请求/EOF时间，不用mtime或事件耗时倒推。容器/image与此前部署inventory、baseline-handover一致；ready时钟早于采集；显式OMP=1、HCCL端口auto、world4、图20调用及原55s预算绑定。先前同容器package/git证据按原download manifest重核，micro各worker实际动态库/全环境身份未保存，不能从旧inventory穷尽推定。

首次SHA门禁揭示8份remote rank原件比以前local副本各多一个末尾LF；全文件其余字节与parsed JSON完全相等。原local SHA仍匹配旧comparison，remote原件另保存完整SHA；未改动任何原件。用原analyze_hccl.py（算法不变，仅AST重绑输入根）重算全部JSON，差别仅为这8个sources.sha256；按已证明字节桥归一化这8项后，完整JSON与既有comparison相等，任何数值未归一化。失败尝试和差异独立落盘，不能描述成原文件SHA全等或旧分析器raw bytes完全相等。

源码另揭示两个口径限制：controller的idle(k)前后调用写同一*-before.metrics，postcheck覆盖precheck；剩余gauges为0仅证明最后idle，不证明两份快照保留或连续idle。process_import_seconds_before_init的起点在imports之后，是相邻monotonic读数间隔，不能用于import成本。当前controller/standalone源码SHA无取数前freeze凭据，源码机制与原输出桥分别表达，不虚构历史执行文件SHA。

首轮独立目录有4份HCCL初始化resource unavailable错误，日志明确16666端口已占用，0份rank timing结果；后轮显式auto完成。它是通信端口资源故障，不是技术候选退化。两组CANN9.2/9.0、torch-npu、固定卡组等混杂保留；微测直接调用torch.distributed，不是两框架服务通信封装的替换实验，不推出模型收益、纯框架因果、输出/greedy等价或优化采用。

新UNALLOCATED archive-draft继承gen23，不覆盖历史草稿，新增3份小CSV，15份全<64KB、旧12份字节不变。20份草稿文件SHA、262表格行、37内部链接、16 RESULT字段及72行原数值检查通过；不是全repo CI或技术最终验收。未分配R、未commit/push/发布；项目规则与账号仍as-of-gen20，临分号/发布刷新。实际204合并需显式附加gen21/22/23/24 provenance，MAP_SUPPLEMENT.json列出本代入口，旧autostage/assembler目录未改。

现场只观察一次：108 verified_archived / submitted108 / inflight4 / pending92。四个原master PID/start/command与旧检查点一致，下游waiting；SDK原进程不存在、自有container停、slot resumed。原all8逻辑资源归属保留，没有抢锁/借slot/资源释放/终止作业。四CPU worker个体本代未另查，父记录照存；完整host/boot/PID/start/命令/日志/终止条件写external-jobs-receipt.json。

下一轮先观察原204一次。若完成则核对真实input/export/audit/archive SHA，gen19合并显式接入gen21–24及现有补充，再对实际最终地图做HTTP/browser QA。若仍在进行，可独立审计GDN micro/其它短诊断的原采集时钟和部署出处。Q06维持fallback，整体接受由root核验，overall_complete=false。
