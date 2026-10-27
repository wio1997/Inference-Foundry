# Q27 gen9 阶段结果：优化候选与源码范围审查

阶段完成，standalone executor 核验，待 root 验收；整体未完成，未发布、未采用优化。本轮没有模型、服务或 NPU 工作，没有借 SDK slot，没有终止外部进程或释放卡组。写入仅限本代目录及 executor-checkpoint.json，unattended/ 未改。

对已有16候选逐项明确目标、Work/Data Movement/Dependencies/Resources、观察、限制假设、干预位置、收益上限、边界与最低成本下一步；另将已存在证据中的确定长度尾回放 Q06、HCCL 等待/传输边界 Q07 明确列为候选，共18项。排序基于信息增益与成本，不是预测收益。所有 numeric_e2e_benefit_bound 为 null，优化采用为 false。

- 58个引用涉及35份冻结源码/配置文件，逐份重算 SHA，核对行界或 JSON 字段。这个计数只证明字节/结构，不表示所有引用语义或部署命中均被重新验证。
- 明确语义审查发现两处原引用行号错误：D03 的89–98行是3FS/EPLB相关内容，时间成本切块在105–119行；D04 的148–158行不能支撑近似SWA论据，定义在179–187行。文件SHA不变；原引用、原因、新范围和精确commit URL同时保留，不修改历史输入。
- 对4 bucket、4 budget、4 async-off/fresh async-on场景的既有N3配对百分比逐对重算，全部一致。只核对既有证据算术，不作新实验/新因果收益判定；质量等价仍false，固定phase顺序/无反序漂移控制等原限制保留。
- 4 cache cohort的strict ID gate均未过，baseline重复本身也不稳定。更新“本机未测”的旧展示，同时保留GDN/state正确性未知；不把失败gate当cache性能失败。
- 四rank step42零增量提交与原生回放按完整RID+step核对，读取并重算7份原始observer文件SHA，复用原root-native join receipt，不冒充重新导出/扫描全部native profile。调度器先run当前batch后process上一batch、finish cap及已finished请求丢弃路径具有源码契约；25.8–26.0ms只是原生端点跨度，无法推导可回收EOF时间，四rank不相加。
- 派生地图在机制页新增候选表，当前机会卡片/三条机制状态显式overlay，旧记录仍可展开；冻结baseline/mechanisms/design语义SHA保持不变。D03/D04源码链接用新范围。新 gen9-build-receipt governs当前产物，拷贝的旧receipt仅属历史。
- Chromium file:// 检查18行/18有效候选、2条校正链接、12场N3/4cache/4rank尾身份；回归124 vLLM回放、7936层区间、128 SG选项、15源码链接、8 FX sidecar、41 CPU shard及4请求身份节点。无JS/console错误或外部网络请求；桌面与移动截图已查看。仅适用于本代预览，非最终204 QA。

证据入口：candidate-audit.json、source-anchor-audit.json、CANDIDATE_REVIEW.md、map-preview/gen9-build-receipt.json、browser-QA.json、package-receipt.json。包采用固定allowlist并重新读回成员SHA，不含transport/key/原始profile。

现场只观察一次：原native204主进程4068537/start_ticks510787256及assembler7390、sharder7391、autostage18358均与原command/start/boot ID一致。rank记录74 verified_archived；pipeline_counts为submitted73/inflight3/pending127，两个进度字段存在差异，未当作完整档案复核。三个下游仍等待。gen82 controller/watchdog不存在、自有SDK container exited、slot phase resumed；未新借slot。所有8卡保留原逻辑占用。精确host/PID/start/command/log/终止条件在external-jobs-receipt.json，跨本轮结束保留外部作业。

本地driver的runner PID/start/argv完全匹配，Codex PID/start及直接父runner匹配，worker flock唯一；Codex当前argv SHA与启动记录不同的旧差异继续原样保留。服务端实际模型身份没有独立可见证据，不声称已核实。本轮没有新的失败尝试。

下一轮先单次核对native204精确现场与gate。若仍未齐，可对Q06作CPU源码不变量设计：明确已提交输出、inflight query、确定长度预算和req/KV/state所有权；覆盖EOS/grammar/取消/混批/spec边界，不能直接按输出计数删回放或全局关闭overlap。若204齐，核对完整input/export/audit/archive闭合和计数字段差异，将gen9 overlay/renderer及gen8请求身份应用到实际最终204，再重包和浏览器QA。

保留unknown：逐step→输出token因果、HCCL纯传输、端点exclusive/全stream关键路径、质量/greedy等价、历史gen64 rank0缺口与RECORD-stop警告。私有main/index R归档前刷新规则/heads/leafsys1权限；未分配R、未发布，root最终接受仍待办。
