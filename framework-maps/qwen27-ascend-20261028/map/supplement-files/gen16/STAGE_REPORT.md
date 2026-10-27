第16代完成真实 SG 稀疏 shadow 诊断、构造来源及逐进程线程退出原始回执闭合。阶段完成，待主会话验收；整体未完成。

现场为 EXPERIMENT_HOST/aarch64、Ascend910B1物理4–7、原SG/CANN9.0镜像/权重，TP4 bucket1、512/C1/out32。controller PID1564514/start_ticks515492355 非阻塞持原performance-workflow.lock后检查并启动；未改变调度、数学、状态写入，未跑正式矩阵/SDK/native采集。独立代次源/manifest/plan/上传SHA均核对。

真实请求返回512输入、32输出、length、[DONE]/EOF。四真实rank均由既有probe显式rank字段映射，各4个SHA-bound模块，合计16绑定；没有按PID顺序或时间猜rank。稀疏观察保留选定语义记录，健康空队列preplan与普通planner-return只保留过滤计数。四rank记录均从序号1连续至EOF，errors0/dropped0，构造→分配→提交pool/epoch来源审计通过。本代不是全方法调用无过滤trace，也未恢复gen15丢失前缀。

15个真实进程的EOF close、records SHA、PID/start_ticks/boot/namespace、collector TID全部闭合；每个容器内/proc probe原始stdout、解析值和container identity单独保留并独立审计，均为进程仍存在而collector线程消失。辅助进程也逐个验证，未仅凭controller控制流判断。20bootstrap/collector、11身份/rawprobe、7构造来源CPU用例通过；这些CPU用例不是NPU正确性或性能结论。

实测记录中168个pending对象/launch关联、160个完整process/RID/pool epoch/slot/generation匹配，8个末token预算模式。这些记录包含startup warmup及诊断两个显式engine RID，未按数量/时间选出所谓纯正式窗口；每个decision仍fallback。仅确认原路径的观测身份与构造来源；pending accepted-token下界、copy完成、GDN/cache内容/生命周期/复用安全、干预后的TP一致性与精确输出/greedy等价仍未知。未做Q06调度优化，也没有优化收益/采用结论。本代API request ID→engine RID未重新审计，不按唯一请求/计数/时间补连。

独立终态核对controller退出、原SG容器/image一致，恢复sleep且StartedAt与自身清理回执一致，performance锁当前无占用。不终止原pipeline/4CPUworker，不释放原8卡逻辑所有权。无新SDK借用。

原native单次快照：rank状态{'verified_archived': 92}；pipeline{'submitted': 92, 'inflight': 4, 'pending': 108}，后级仍等待204。此是进度快照，不是最终归档SHA或实际204 QA。本代证据须与gen8–15一起显式overlay/repackage/browserQA；私有main/index归档、R分配、整体/root最终验收仍未完成。

下一步优先检查native204是否ready；ready后做实际204输入/导出/审计/归档SHA闭合和gen8–16 overlay/repackage/浏览器QA。仍等待时，可基于本代完整语义记录与现有冻结源码，选择最低成本动作澄清pending accepted-token下界/必要fallback条件；不得仅凭末token计数模式直接启用Q06，不重跑本代诊断或性能矩阵。
