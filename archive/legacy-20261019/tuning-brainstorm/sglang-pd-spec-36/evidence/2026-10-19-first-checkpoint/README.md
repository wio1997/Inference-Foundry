# 首个实证检查点：安装路径、支持性谓词与MTP长度状态

北京时间 2026-10-19。Task 仍 ACTIVE，整体 E0，没有本任务的 PD＋MTP 推理基线，也没有可报告的加速比。下面接受的是限定范围的审计或CPU正确性结论，不能当作36点已测完。

## 当前实现与可确认事实

从37号现有容器只读取回源码，实际HEAD为 `ce0a8fea5f4245839d174b2b8c55692be1810a7f`；镜像ID为 `sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263`。镜像标签revision是另一个值，因此不能用标签替代安装源码版本。课程参考提交仍单列为 `57db5089c207ae4ce189cc092a89a2027c8aea29`。

[36点路径/支持性表](path-support.md)及[机器可读版本](path-support.json)均有文件哈希与行号。36项均能定位静态源码或安装文档入口；所有 `runtime_path_confirmed` 仍为false。M034和M036本轮仅有安装文档入口，实际gateway和可选功能代码未闭合。模型config声明GlmMoeDsaForCausalLM，但实际加载类/覆盖配置还需追踪，不能据配置名推断层实现已命中。

16台现有访问入口的只读盘点均返回0。44、48无模型服务，每卡HBM报告值2,871–3,120 / 65,536（npu-smi原表单位MB）；32虽然空余HBM较多，已有其他实验容器。37每卡HBM报告值62,617–62,959 / 65,536（原表单位MB），其他多个节点也有模型服务。[容量摘要](capacity-summary.json)只表示一次观察，不表示资源租约。已请求用户补充分配信息；未启动/停止容器，未发推理请求，未修改其他服务。

## 最小实验与结论

| 子实验 | 实际结果 | 主控裁决 | 不能推出的结论 |
| --- | --- | --- | --- |
| L001：安装路径与资源只读审计 | 36静态入口，124个路径条目核验（121份文件、3个候选文件缺失）；原始安装源码未改 | ACCEPTED，仅静态范围 | 本任务已选定运行版本或36点均受NPU支持 |
| L001支持性子探针：实际AST谓词 | 288组grammar/batch判定与12组PD角色/PP/overlap分派通过 | ACCEPTED，仅CPU分派 | FSM、NPU事件、DP collective或端到端组合正确 |
| L002：FutureMap真实publish/resolve方法 | 324组CPU状态差分通过 | ACCEPTED，仅值/索引/dtype/CPU镜像 | 真NPU异步顺序、graph、完整EOS/取消、bonus/hidden/topk stash正确 |
| L002后续：错误producer反例 | 实际发布old+1或省略复用槽新长度，再调用真实resolve；636个反例均检出 | REJECTED：上述删改真实长度的替代行为 | relay融合候选无价值或已取得性能收益 |

M016白话结论：下一轮要知道每个请求究竟推进了多少。一次接受1、2、4个有效token的请求不能统一按加1更新；新请求复用旧槽时也不能读取上一请求的长度。数据必须保留，融合写入仍可能减少独立提交，但必须先确认实际producer、消费者与NPU执行时间线。本轮没有改生产源码，M016性能候选仍INCONCLUSIVE。

M032版本差异：实际 `supports_grammar_overlap` 对EAGLE、EAGLE3、FROZEN_KV_MTP、STANDALONE、DFLASH、DSPARK返回true；UNO、NGRAM、NONE返回false。`grammar_needs_sync`和队列/forward模式共同决定是否先处理上批结果，支持的算法另有grammar barrier。decode循环中较旧的概括注释不能覆盖实际判定函数。这里的true不是完整Ascend支持证明。

M007仅专用PD的P/D混批优化不适用；packed/ragged张量准备保留。M018私有长度D2H流在此版本明确为CUDA-only，不能当成Ascend现成路径。M022保留现有融合门控、模型opt-out和数值保护。M025安装文档明确shared-index prefetch排除推测解码；Ascend连接器另明确拒绝HiSparse destination-device indices，这与一般DSA短序列fast path是三件事。

## 命令、原始样本与复现范围

所有命令在任务根 `/Users/wio/work/sglang-learning-lab/mechanism-task-20261018` 执行，由本任务ZCode入口启动，外层和内层退出码分别保存。只读脚本中的SSH复用既有认证；不包含密码、密钥或token。

```sh
python3 scripts/collect_environment.py
python3 scripts/collect_sources.py
python3 scripts/collect_capacity.py
python3 scripts/read_source_batch.py
python3 scripts/build_path_support.py
python3 scripts/probe_support_predicates.py
python3 scripts/setup_cpu_probe.py
.venv-cpu/bin/python scripts/probe_relay_cpu.py
.venv-cpu/bin/python scripts/probe_relay_counterfactual.py
```

最终上述脚本均有exit0产物；谓词脚手架首次失败与执行器自行修补另列下文，不伪装成首次即成功。真实方法源码快照位于 `sources/installed/`，测试脚本位于 `scripts/`，CPU依赖版本见 [cpu-requirements.txt](cpu-requirements.txt)。可在本检查点目录用Python3.12创建独立`.venv-cpu`并安装这些依赖，再执行两个relay脚本与谓词脚本；每个脚本使用一次性输出目录，重复实验要另建Run，不能覆盖旧结果。源码快照为Apache-2.0上游文件，保留文件原有版权头。

原始逐例结果：[300组谓词/分派](results/predicates.json)、[324组relay状态与636个producer反例](results/relay-counterfactual.json)。每个Run的命令、退出码和哈希见 [receipts/](receipts/)，其他安装文件哈希与参考来源见 [source-manifest.json](source-manifest.json)。完整环境/源码传输原始文件留在本地Task Run中，其内容哈希在receipt中；未把其他服务命令、完整Kubernetes对象或认证日志放入仓库。

CPU fixture手工初始化FutureMap并提供同步FakeEvent；执行原样方法，但没有调用生产构造函数。负控v1先比较反事实状态值，v2进一步将错误值实际发布/或省略新槽发布后调用真实方法。两份receipt均保留，公开逐例样本选v2。没有设备计时，因此A/A噪声、A/B收益、TTFT/ITL/吞吐、接受长度分布和可达带宽/算力全部未测，不填0或虚构数字。

## 执行器偏差与独立审计

各新session的model_usage元数据均记录`deepseek-flash`；仍保留UNKNOWN后端发布版本。仅部分近期session的request/response日志尚存，具体条目单列，不能由配置别名推断缺失元数据。[脱敏审计](zcode-audit.json)只包括本任务实际session和选定工具状态，不含prompt、headers或provider配置。

谓词测试首次因枚举class默认参数引用CustomSpecAlgo而NameError。ZCode越过单命令执行边界，加入不被测试方法使用的类型占位及compile_placeholders标签，另写过/tmp诊断脚本，并移除空目录后重跑。主控已核对两处Edit、实际AST和原始模块哈希，接受最终有界结果，同时保留首次失败工具记录；执行器编的近似时间戳说明不作证据。临时文件已按确切归属清理。后续任务AGENTS规定：冻结命令只运行一次，禁止自行编辑、删除输出或重跑，失败返回主控。

## 下一轮与上界

没有服务profile，不能按假定Attention/MoE/网络瓶颈排序，也不能从CPU状态检查计算性能headroom。先补齐资源分配、实际模型映射/权重完整性、P/D并行及传输、MTP参数与graph/overlap合同。继续M016逐字段producer→publish/stash→resolve依赖图，随后审查M035/PD-CHUNK连续区间描述符现有实现；CPU等价与局部成本可独立推进。

真实NPU后续先A/A估噪声，再同条件交错A/B。MTP按draft＋verify＋commit＋transfer总成本/有效输出token并保存接受长度分布。下界要以真实shape/dtype/布局/拓扑的必要FLOPs、HBM/链路字节和串行依赖校准；本轮没有这些数字。Task继续ACTIVE，不能由此检查点宣称性能覆盖完成或逼近物理极限。
