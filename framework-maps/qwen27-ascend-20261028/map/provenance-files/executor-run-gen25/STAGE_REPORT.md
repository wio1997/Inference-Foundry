# 第25代：GDN原微测出处及测量范围（阶段完成，待root验收）

本代复核两套已部署底座的原GDN独立API微测，没有重跑矩阵、消融、模型、服务、NPU或SDK。启动安全参数核对为gpt-6.1-sol/high；实际服务端模型身份未暴露。19份小原件逐项SHA/bytes核对，7个pt远端全文件流式SHA复读，3个T128输入/输出小文件另下载并核对，原始二进制未入archive-draft或Git。

10个case共50个事件批次、1000次计时API调用，另50次预热。5个共享输入SHA与fixture-manifest、两臂fixture_sha256均闭合；原results与原stdout/旧本地JSON相等，20项均值/中位数独立核算。原controller明确v采集2026-10-25T09:33:03.412794002Z–09:33:15.145800380Z，SG09:33:15.147008088Z–09:33:32.240994309Z，均为docker-exec启动至退出包络，不是逐kernel或请求时钟。

脚本每臂只使用logical device0，并非TP4；可见卡组为原0–3与4–7，micro worker没有独立保存物理映射。随机共享CPU operands，固定Nk4/Nv12/Dk128/Dv128、qkv/state BF16、g FP32，不是模型激活。每批20次eager API调用取事件区间/20，仅5批/case，不作100独立样本，不是captured graph；无法仅靠event隔离host-feed、等待或分配成本。显式OMP4、PYTHONPATH runtime及python3 -P按原command核对；SG服务独有compat环境不能自动套到另一个docker-exec。源码一次前置零队列检查未保存metrics，也无后置或连续idle证据。

两臂T128保存输出文件全字节相同，历史FP32参考报告与原stdout/旧本地报告相等。原evaluator在本机x86_64、torch2.14.1+cpu单线程复读同一历史保存张量，pairwise字段相等，但against_reference的12个描述性指标未与原aarch64报告完全相等；最大指标字段绝对差1.2900659274346638e-08。严格全JSON回归首次拒绝后，保留双方报告及差异，不归一化、不设临时容差、不当作技术候选失败。差异机制未实证归因，不用本机结论替代aarch64，参考指标本来无接受阈值。其余4case每臂仅有末次finite/shape，没有已保存成对/参考输出；[4096,4096]参数语义、initial_state不变性等未独立验收。模型输出/greedy等价仍unknown。

新UNALLOCATED archive-draft新增4份小CSV（采集/部署/逐case/配对摘要），旧15份字节不变，共19份均小于64KB。24文件SHA、290表格行、45内部链接、16 RESULT字段、10case原值与5项描述性ratio检查通过；不是全repoCI或性能验收。没有分配R、commit/push或发布。当前源码SHA不是取数前freeze，micro worker模块与CANN动态库全身份没有收据，CANN/依赖/固定卡组混杂保留。未推导纯框架因果、模型收益或优化采用。

只观察原204一次：parent ranks列表报告111 verified_archived，但同快照pipeline_counts为submitted110/inflight4/pending90。这是记录字段瞬时不一致，未把差值补齐、未当204完成、未将submitted当completed；该快照不能给出冻结终态。四原master PID/start/command匹配；下游仍waiting。旧SDK确切PID不存在、自有container停、slot resumed。all8原逻辑归属保留，无新外部后台作业、无锁抢占/借slot/资源释放/终止。完整外部身份见external-jobs-receipt.json，未观察的旧作业明确保留旧时间；4CPUworker为原parent管理，本代不另探个体。

另一次草稿验证因局部变量遮蔽构建receipt而KeyError，已修正后通过；这是本代验证脚本问题，不是原数据或技术候选失败。

下一轮先观察204一次；完成则核对真实input/export/audit/archive全SHA，再gen19合并显式加入gen21–25及current supplements，对实际204做HTTP/browser QA。如果仍pending，可把gen21–25溯源接入新的独立地图预览与合并runbook，做浏览器QA，保留预览/最终204区别。Q06维持fallback，质量与因果未知不放宽；整体接受仍由root核验，overall_complete=false。临分号/发布刷新项目规则、远端heads、编号占用及leafsys1权限。
