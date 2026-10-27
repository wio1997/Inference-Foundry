# 第22代：核心消融部署与采集时钟（阶段完成，待root验收）

原78正式波、26个完整N3场景的record SHA与client manifest SHA逐项匹配既有results-final；690条逐index输入指纹（prompt/seed/body/max_tokens）一致。原139条完成记录均有SHA绑定投影和原manifest，其中61条guard/暖机/drain诊断维持排除。没有重跑原始请求、基线、消融、NPU或原SSE/token质量审计。

10个正式variant按控制器显式deploy/score绑定；SG async-on的owned_stall_restore与vLLM的deploy分别留源码锚点。原ready、manifest创建、record完成、下个variant启动四时刻严格有序；时间只作拒绝冲突的门禁，不拿近邻猜部署。10条原launcher日志JSON及原日志全文件SHA保留，CPU拦截execvp重放现存wrapper，argv/semantic JSON逐项相同。3个原控制器文件与本地task历史副本字节相同。现存wrapper/source hash的观察时间不冒充取数前freeze。

5组正式SG的60份真实rank module/JIT收据、额外10份parent收据与20份weight gate按PID→rank闭合，三覆盖模块与第21代冻结候选SHA相同，JIT source SHA及HV,V,K布局一致。原10组variant的container/image与baseline-handover相同。依赖、aarch64、910B1和分组沿已冻结部署及容器身份引用；没有新增全权重重哈希或完整继承环境快照。weight gate/装载收据不等于输出/greedy等价；没有凭当前源码补造历史加载时间。

正式采集包络为2026-10-25T09:45:43.893212+00:00至2026-10-25T12:53:05.423474153Z；逐phase/arm见ablation-collection-periods.csv。起点是manifest创建（发请求前），终点为record finished_ns（summary/after-metrics之后），不是精确request-start/EOF；ns整数转换保留精度，没有mtime或duration倒推。

两个download receipt共491项小文件逐项重哈希；其中原完整record没有传输，只传SHA绑定投影和summary协议有效性，formal78原record SHA仍与原final分析一致。首次下载因冗余summary超3MB小文件门禁而安全退出，改为projection；本地审计源码语法及anchor匹配问题已修正，AST明确了实际双分支。详见failed-attempts.json；这些不是技术候选或实验失败。

最新未编号私有archive-draft继承gen21，新增部署/时段2份CSV，9份总表均<64KB，旧7份字节不变。验证14草稿文件SHA、220表格行、22链接和16RESULT字段；不是完整repo CI、技术/性能接受或发布。未分配R、未commit/push；规则/refs/账号仍as-of-gen20，临发布须刷新。gen19最终地图合并需显式附加gen21/22provenance，本轮未重建/验收最终204地图。

一次原生导出观察：104个verified_archived条目，pipeline submitted104/inflight4/pending96。该快照计数相同不代表204完成，也不对旧快照差异作猜测解释。4个原master的PID/start_ticks/command与检查点一致，assembler/sharder/autostage仍在等待。SDK旧精确进程不存在、自有container已停、slot resumed。未借slot、未起停任何服务或设备作业；原all8逻辑所有权保留，CPU审计workers及主pipeline不动。worker个体/proc本轮未逐个核查；master精确身份、host/boot、命令、日志与终止条件见external-jobs-receipt.json。

下一轮先观察native204一次；完成才闭合实际input/export/audit/archive SHA并进行gen19合并、补充最新证据及实际204 HTTP/browser QA。若仍等待，可补cache/其它诊断的原部署/采集时钟，或推进有源码依据的候选门禁。Q06继续fallback，质量/greedy等价未知，未采用优化；root负责整体最终验收，overall_complete=false。
