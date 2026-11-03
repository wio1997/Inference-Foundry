本轮以保存的64次decode图为起点：其中16次调度2请求/12tokens、48次4请求/24tokens，记录的目标图物理域仍为24行。没有把窗口频率当全部负载频率，也没有补配16次未知图阶段。

补核了37处原mode/DP分支及4处对齐helper分支，累计M22核心本地数据流记录110处；全部1101个原静态导航ID及16个语义范围保留。新增源码只涉及明确问题的方法/小helper，DP同步、dispatcher和旧配置片段直接复用。

捕获尺寸的时序得到澄清：Engine启动日志的[8,16,24]不等于最终worker registry键。worker先按各backend/cache spec的最低支持聚合；满足FULL-decode且CP/SP/pass-SP条件时，会在mode resolver前按LCM(6,8)=24预对齐，并把resolver用TP改为1。该分支下初始列表变为[24]；之后还会resolve mode、建立descriptor，registry取其实际descriptor尺寸。当前尚无最终registry快照，因此没有把条件推导当成全部运行时值。

Uniform decode还需要初始状态、逐请求最大宽度匹配和总tokens=宽度×请求数；DCP/hybrid/recompute、force参数、LoRA、encoder/cascade各有分支。dispatcher优先FULL key，其次放宽的PIECEWISE key，最后NONE fallback。保持FULLuniform6+TP8接口时16行无效；16行只满足TP约束，不能靠改capture-size设置替换24行。

DP非skip路径的CPU all_reduce汇集每rank写入的独立列，之后显式取最小graph mode和最大token数。任一rank为NONE就协调为NONE；图回放、draft或规定的MoE/finegrained通信条件仍可能要求统一最大carrier。force_eager只限制图分派，不取消DP协调。skip策略按模型角色、KV-consumer、hierarchy/MegaMoe、prefill/decode通信与recompute等条件每次重算，不能按配置对象id或MC2名称推断。81个DP例、512个skip逻辑赋值只验证源码条件域，不是实测耗时或实际rank配对。

H06补充一个条件变体：先确定backend最终mode，再仅对实际FULLuniform域施加joint alignment。它可用于评估真正支持非uniform/PIECEWISE的分支；现有源码TODO也指出预对齐早于mode降级的限制。保留8条约束，包括metadata/descriptor、TP、DP/EP carrier、实际row map和bank/native代际。没有宣称当前固定FULLuniform路径可降到16，或少1/3行就少1/3MoE工作/延迟。

另补齐历史源码证据：v16 immutable receipt中的utils.py SHA与当前冻结文件一致，给前两轮imported predicate提供历史166stage hash支持；实际backend、ENPU、registry和native epoch仍未快照。旧receipt的full-result有未返回的timestamp/envelope，不能用现代子集compact serializer重算其全文SHA；本轮保留该失败，改核原receipt的历史manifest SHA与源码SHA，未掩盖差异。

登记5项：3 pass、2 error。两次错误分别为把远端结果名误作本地文件、旧receipt格式不适用新hash校验；均在分析阶段保留并修正。无远端登录、模型/torch/NPU/服务/健康操作。

整体仍未完成：实际worker最终keys、DP skip及rankwise字段、graph/task/参数bank/事件/handle代际与全部读者、16次未知阶段、capture/warmup和worker邻接责任、资源上界及完整EngineCore/transport仍待证。35模型候选、7框架候选、0整模块关闭，26理论比较INCOMPLETE、3136资源行、849缺PMU、16phase-unbound保持。Goal active。
