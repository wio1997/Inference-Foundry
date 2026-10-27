# R0228 — Qwen27冻结部署性能与机制地图

实际最终204地图与完整分发已核验；随本记录归档；root最终接受待完成。KEEP仅保存研究证据。

| 字段 | 值 |
|---|---|
| R ID | R0228 |
| Model Context | `qwen3-5-27b-w8a8` |
| Base Stack | — |
| Base Commit | — |
| Depends On | — |
| Related OPTs | — |
| Question | 对同一实际checkpoint的两套冻结部署，从请求到EOF的性能差异及可证明的源码/原生profiling机制覆盖是什么？ |
| Hypothesis | 同负载下部署组合存在可重复的端点差异；源码与显式任务身份可验证局部机制。若身份/区间不唯一则保留unknown，不填补关键路径。 |
| Method | 探索性研究，归档说明为事后补记；7长度×C1/5/10/16×2框架×N3=168正式波，另56暖机，AB/BA配对；输出1024，主计分无profiler。消融78正式波/26 N3场景、cache 4 N3 cohort；独立HCCL原微测72 rank/case与采集/部署来源见EVIDENCE，非框架同路径；GDN单设备API原10case/50事件批次的采集/共享输入/描述性参考边界见EVIDENCE；诊断插桩单列。波内请求P50/均值分开，跨3波取中位、MAD与极差；无确认性验收阈值补造。 |
| Changes | 未采用新的推理优化；已有冻结SG正确性修复/FIA属于对比底座。研究新增观察器、原生身份审计与地图代码均为诊断，不能视为无开销候选臂。 |
| Result | aarch64；8×Ascend910B1分为两组TP4；输入512..256000、C1/5/10/16、输出1024、每臂N3。例512/C1：v/SG TPS=53.3567/49.4354，变化7.93217%；请求TTFT P50=0.274009/0.336138s，变化-18.4831%；端点TPOT=18.4941/19.9239ms，变化-7.17648%。全部28场景、逐臂min/max/MAD见README与assets；原始输出绝对位置/SHA/依赖/配置见EVIDENCE。冻结部署28场景中，vLLM TPS与端点TPOT均占优；SG按请求TTFT中位18/28、均值15/28占优。每臂每场景N3；质量/greedy等价与优化采用未确认。 204原始导出与归档完整性已闭合；不等同逐层归因、质量或整体接受。 相同payload的这两个插桩短诊断输出文本不一致；不证明根因或原性能矩阵质量。协议/长度有效不等于质量通过。整体质量/greedy未闭合，未采用优化。 |
| Decision | KEEP |
| Produces | 请求到EOF的冻结部署性能/机制/源码/原生profiling及18候选地图；实际204分片、完整补证合并、file/HTTP选择QA与完整分发包已核验。质量/greedy、外围完成语义与全stream关键路径保留unknown；未采用优化，待root最终接受。 |
| Final Commit | — |
| Code Snapshot | scripts/ 下七个诊断与地图交付组件；逐源 SHA 见 CODE_INDEX.json（sha256 65b6196204ba58e16fc673fa6e8739148b97028d075667e0acb2d0a755a08561）；最终地图、QA与完整分发收据见CURRENT_DELIVERY.md；未作为推理优化采用。 |
| Evidence | [EVIDENCE.md](EVIDENCE.md)；小数字表入库候选，原始trace/log/归档不入Git |

## 字段说明（第29代整理，不改变事实）

以下为历史草稿字段说明。本次本地归档已登记模型研究索引；最终提交前仍须刷新远端编号占用。

- Model Context原说明：`qwen3-5-27b-w8a8`（候选slug；main尚未登记，需与本轮归档一起核验登记）。
- Base Commit原说明：—（跨多个冻结栈的观测研究，无单一代码改动起点；原始源码/补丁与文件SHA见EVIDENCE）。
- Depends On原说明：—（不把历史同名Qwen研究自动视为本轮依赖）。
- Final Commit原说明：—（本轮未发布研究代码commit）。

## SG输出谱系补证（第36代）

[32个输出token谱系](SG_OUTPUT_LINEAGE.md)仅补充既有cleanSG短诊断：四rank及frontend ID序列相同，31正输出decode+1零输出tail/rank关联原生端点。没有新性能数字或跨框架根因结论；原Result字段不变。


## 采样源码契约补充（gen39，待验收）

[采样契约与边界](SAMPLING_CONTRACT.md)：同一2758字节短诊断请求：两栈在所审查的源码路径均选argmax；数字温度/top_k归一化不同不构成已证随机分歧。 历史sampler遥测、logits、vLLM采样ID与质量/greedy等价仍未知。仅增加保存输入分析，不改性能结果；204归档已闭合；最终地图与root接受仍待验收。


当前交付状态以 [CURRENT_DELIVERY.md](CURRENT_DELIVERY.md) 为准；各代补证说明保留历史时点，不代表最终地图已经通过QA。
