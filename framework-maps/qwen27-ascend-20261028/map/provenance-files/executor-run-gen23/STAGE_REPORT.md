# 第23代：cache历史部署与采集出处（阶段完成，待root验收）

72条原cache记录逐一核对完成清单、manifest/params SHA与332条逐index请求身份；原完成state与先前保存workflow-final-state字节内容一致。原record仅summary减少为旧分析器需要的字段，保留其余顶层信息、原完整文件SHA和原per-request audit。72波包括32 prime、16 qual、12正式off、12质量未闭合诊断on；没有重跑请求、原矩阵/消融、服务/NPU/SDK。

435份小证据下载后逐项SHA/bytes核对，含144份原before/after.metrics，它们还与record保存的全文件SHA/bytes逐一相同。最初缺metrics文件，分析器fallback保留HELP/TYPE注释行，与原文件读取分支输出不同；1752处差异全部位于native_cache_metrics，独立重现确认。补齐metrics后，未修改的原分析器完整JSON与既有cache results-final相等，仅归一化input_work_root。修正的是证据投影；没有修改数值、历史分析源码或技术候选。

cache-off继承核心消融最后g4-restored-baseline实例，没有虚构新的off部署。两份继承state及wrapper字节与第22代下载副本相同；核心restore→core completed→cache等待后off waves→显式on deploy和mode→最终off restore的AST锚点和历史时钟拒绝门禁保留。6份原部署state/launcher行及日志全SHA绑定，CPU拦截execvp重放6次argv/semantic JSON相等，未执行实际exec/model/device。当前源码hash观察不当取数前freeze，继承全环境未穷尽。

3组SG（继承off/on/最终off）共36份真实rank module/JIT收据、6份父进程收据及12份weight gates，PID→rank闭合，三覆盖模块与HV,V,K JIT源SHA和第21代冻结基线一致。weight gate不证明输出/greedy等价；原CANN9.2/9.0、aarch64/910B1、固定卡组等混杂保留，不能宣称纯框架因果。

四组C5/C10 cache严格门禁均rejected，同baseline/candidate重复token IDs不稳定；formal qualified on为0，四组N3仅诊断观察，质量等价/GDN状态恢复仍未知，eligible_for_optimization_adoption=false。没有把审计失败或模型容量问题当技术候选失败。原fixture manifest→client prompts_file哈希相同，但本代没有重读原prompt数组或重哈希SSE正文；完整record SHA是本次远程读取身份，旧分析JSON无逐record旧SHA，证据桥由完整分析输出一致、旧state相同及原manifest/metrics收据提供。

原采集包络：2026-10-25T13:01:59.035178+00:00至2026-10-25T13:27:44.000287368Z。起点为manifest创建（发请求前），终点为record finished_ns（summary及after-metrics后）；不是精确request-start/EOF。逐波72行另存，整数ns保留9位，不用mtime/耗时倒推。archive-draft继承gen22，新加cache时段/部署/N3三份CSV，共12份，各<64KB；旧9份保持。17份草稿SHA、242表格行、29链接、16 RESULT字段和24条原N3臂值检查通过；不是全repo CI、发布或最终204验收。

原生导出只观察一次：106个verified_archived，pipeline submitted106/inflight4/pending94；4个原master PID/start_ticks/command与前检查点一致，下游waiting。SDK旧进程不存在、自有container停、slot resumed。原all8逻辑归属保留，没有起停或杀任何外部作业，没有新借slot；4个CPU worker个体未另查。精确host/boot/PID/start/command/log/终止条件见external-jobs-receipt.json。该快照不能当204完成或补齐旧计数差异。

下一轮先观察204一次；若完成则核验真实input/export/audit/archive SHA，gen19合并显式附加gen21/22/23 provenance和最新supplement，再做实际最终HTTP/browser QA。若仍等待，可闭合HCCL微测/其它短诊断的原采集时钟和部署出处。未分配R/未发布；规则/refs/leafsys1权限临发布须刷新。Q06继续fallback，整体接受由root负责，overall_complete=false。
