# Reset161 — upload complete;first GLM-5.3 native PD functional reference

用户本轮明确确认166/167的`/data/tiankuan/wio/GLM-5.3-w8a8`均已传完。上传gate解除；不是成功加载或payload checksum证明。研究branch/head为`glm5-3-autonomous-20261001`/`df534e5d`，规则GLM-RESEARCH-RULES-v2；无Current/active PERF_KEEP。标准PD完整动态产品与既有TTFT/TPOT分位SLA沿MISSION；本次不放宽合同。

fresh `GLM53-PD-READY-20261006`真实Job确认两机全部182文件/177474tensor结构稳定，index/config身份与checkpoint160一致；32 NPU使用者严格匹配旧native workers，native running/waiting均0。原controller244 PID已不存在、owner记录failed；锁存在不等于持有，新controller须实际独占flock。旧public、observer及完整副本仅按fresh boot/start ticks/父子树退出，旧Run/STORE/配置/source保留。

产品最大缺口仍为5.3无真实PD加载/功能证据。唯一条件性能假说H3沿Reset160：全贡献者device-safe KV后是否因晚观测错过合法D机会；最大可信可消除Gap/time unknown，无性能patch。Run245是功能前置证明，不是H3计时、正式SLA或matched性能A/B；初始CPU observer不装入live进程。

旧Run101仅证明5.2部分PD与nativeAPI，旧helper→dispatch0.58/0.58/4.54ms不证明5.3收益。旧P/D startup/config只复用角色/connector/CANN环境知识；本次不同artifact必须实际加载。既有DSA-CP/MoE初始化失败不重放，`enable_dsa_cp=false`，不调用旧prepare里的计算路径补丁或全量clean。原controller/phase_runner/ACL寿命wrapper复用未改。

唯一Run245：P166与D167各DP1/TP16/PP1/EP16/DCP16，K1 MTP，eager，固定1GiB KV预算，原生scheduler/runner/connector，fail-policy显式fail。此为一个fit/功能参考，配置不计代码收益。首个工作包只发一个长于block的原生Chat P helper(内部1token)→KV metadata→D原body(32有效token)，P token不计用户输出；D不允许transfer失败时recompute fallback。

Hypothesis(functional)：该5.3 artifact在此布局可由现有正确算子实际加载、两角色ready并通过原生KV接收产生有效D输出。
Distinguishing evidence：实际两机root/rank/source/argv、加载与初始化raw、health/model身份；P metadata、D cached KV及token IDs/有效计数，真实exit/错误；没有fullAPI或性能结论。
Decision table：初始化失败→保留raw、停止本Run拥有的子树、读具体原因，不扫描配置；ready后PD失败→保留健康角色及raw只供定位，不提升Current；基本PD成功→随后补完整API/取消/错误/回收与H3的device/合法机会/commit/公开输出证明，不自动转长E2E。若H3首个合法batch已接纳则关闭；否则按Reset160证据门槛决定唯一代码候选。

Run目录内新helper只替代此次实验的危险全量清理/旧队列重放，不是新增产品Runtime层；REJECT时保留证据、停止本次身份进程即可退休。3项CPU guard测试覆盖PID/boot/tick复用拒绝、树作用域及非vLLM foreign NPU owner，0设备调用。Astra独立Review已在checkpoint160，结论仍适用。

本阶段没有新增代码级性能 KEEP；真实完整5.3 PD E2E Gain=unknown；Current=None。
