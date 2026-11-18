# C16已复现数值与code截断错误，启动target-only隔离诊断

本轮将[完整质量超时](039-D974-quality-timeouts-and-C16-diagnostic.md)推进为可观察的语义故障：[C16诊断999](../evidence/C16_faults_targetonly_G1006/G1005_parent_C16_diagnostic_assessment.json)在全部32个HTTP请求成功、SSE正常DONE的条件下，复现4个错误算术结果、3个synthetic code截断，以及14题GSM输出重复并达到256-token诊断上限。客户端逐SSE重建文本/usage/SHA均与记录一致。捕获器保存至data: [DONE]行后停止读取，未读取其后空行；原捕获字节没有补写或修改，全部content/usage事件均已重建核对。不能再只用grader格式或网络超时解释问题；服务端detokenizer、模型计算与DSPARK/verify仍未分离。

## 实际输入输出

| synthetic case | 预期 | 实际 |
| --- | --- | --- |
| math1：14×6+7 | 91 | 98 |
| math3：16×8+7 | 135 | 131 |
| math4：17×9+7 | 160 | 147 |
| math5：18×10+7 | 187 | 180 |
| bind0：复制合成code | KD32190DFD | KD32190DF |
| bind1：复制合成code | K0E411E800 | K0E411E |
| bind4：复制合成code | K1985DE174 | K1985DE |

这些code由测试脚本确定性生成，是测试输入，不是真实凭据。16个简单请求中8个严格匹配；其中算术3/8 exact、copy5/8 exact。math6输出解释式“19*11=209; 209+7=216.”，数值正确但违反只输出整数的要求，单独列为格式问题，不把它与四个错误数值混淆。copy错误保留各自输入的前缀，没有观察到完整复制另一请求code；这仍不证明完整PD缓存状态正确。

[原GSM前16题的流式诊断](../evidence/C16_faults_targetonly_G1006/raw/output/GSMprefixC16/results.jsonl)全部正常返回：index1/3在43/52 tokens停止，其余14个到256-token上限，原始内容可见重复括号、重复等式和反复复述题目。该组使用诊断预算256，不能报告原max2048/1319题准确率，也不能当作新的完整质量结果。

## 冻结与受控替换

诊断于20:48:56UTC完成，实际进程退出、CPU客户端只剩sleep；[107份原始文件及D974退休回执](../evidence/C16_faults_targetonly_G1006/G1003_diagnostic999_D974retire_2204_parsed.json)保留请求、SSE、实时partial、最终结果、源码、日志及SHA。再次核对两端running/queue为零、SDK无活动、原Profile冻结SHA未变和D974源码/容器/八卡owner后，才请求其原控制器停止。20:52:31UTC确认实际控制器退出、容器停止、八卡清空且owner释放；未删除旧容器、原始证据或回滚源码。P832、路由和guard980保留。

随后[实际启动D1004](../evidence/C16_faults_targetonly_G1006/G1004_TARGET_ONLY_D_ACTUAL_START_JSON_parsed.json)，移除六个原生DSPARK/draft CLI选项，请求target-only诊断。权重、固定daily A2镜像、TP8/EP8、上下文、P/D资源、分块、static memory fraction及target full graph配置沿用；[所有kernel、backend和频率helper字节相同](../evidence/C16_faults_targetonly_G1006/sg_D_targetonly_preparation_g1004_manifest.json)。仅复制本任务已退休D974的compiler cache，原目录保留，不将复制视作无编译干扰或性能证明。

[20:56:21UTC现场](../evidence/C16_faults_targetonly_G1006/G1005_D1004_actual_2204_parsed.json)确认新控制器及容器实际运行、正在加载权重，API尚未就绪。不能因候选CLI没有DSPARK参数就宣称实际speculative_algorithm=None；需继续核对native API、八rank严格target加载、无draft加载及完整图，再为新容器绑定guard，执行相同诊断输入。

target-only是隔离实验，未提升为最终对齐配置。缺少draft同时改变内存、缓存容量和query/verify图形状，因此其结果也不能单独证明DSPARK算法是唯一根因，更不能作端到端性能A/B。最终仍须修复并恢复要求的优化组合，通过相同1319题质量、SG四档AISBench及最终P8/D8匹配Profile和收益验收。

21:00:13UTC[新现场](../evidence/C16_faults_targetonly_G1006/G1007_D1004_effective_actual_2204_parsed.json)记录8个target各1384参数严格加载通过、missing为空，尚无draft账本；八rank开始target decode full graph、query宽1，尚无完成/API就绪证明。

## Target-only启动失败的实际终态与1010诊断实例

[1004实际失败及清理](../evidence/targetonly_layout_G1013/G1009_D1004_ACTUAL_FAILURE_JSON_parsed.json)发生在八rank target严格装载及full decode graph完成之后：现有lowratio_radix_compat.supported要求C2 request_scoped且ring_size=8，非SP原生ring为2，故decode-cache校验拒绝。没有API就绪或生成结果；实际控制器退出、容器停止、八卡owner释放及健康/空占用均已核对，原失败没有覆盖。

1010仅移除1004的原生disaggregation-decode-enable-radix-cache标志，未伪造unified/layout属性或放宽校验器，也未改kernel/backend/pool代码。[新实例](../evidence/targetonly_layout_G1013/G1012_D1010_actual_2204_parsed.json)于21:15:54UTC实际ready200，native API的speculative_algorithm=None、decode radix=False；八个target各1384参数严格通过、missing为空，无draft账本，八rank target decode full graph已完成。这仍只是target-only且decode-cache-off的诊断配置，没有提升为最终对齐方案。

### 生成前的PD状态契约

[固定源码与实际P配置核对](../evidence/targetonly_layout_G1013/G1011_targetonly_C2_PD_contract_preflight.json)发现另一个诊断前提：P832仍为DSPARK6，原生C2 ring推导为8；D1010的原生非SP ring为2。C2实际仍是request-scoped，先前1010候选说明中的page-scoped用语不准确，此处更正，保留已部署候选/hash原记录。get_request_state_buf_infos按ring大小声明C2 item bytes，所以跨SP的P/D不能在未核实实际wire契约时直接做对比。

这项差异只约束新的target-only诊断，不能反过来解释原P/D974双方均SP/ring8的质量错误。PP1等长指针列表提前返回，不能臆断C128 PP切片分支已触发；C2 transfer_indices的实际分支返回request slot，也不能套用generic pair-page公式推断索引错误。

1010尚未提交生成请求，guard980仍保留旧P832/D974身份。下一步需验证实际PD状态负载；可评估为target-only D真实分配并正确计价8-row C2 ring以保持wire几何，或用受控的双端target-only重建匹配2-row几何。两者都需要独立验证，不能伪造布局标志。最终仍要恢复所要求的DSPARK/前缀缓存配置并完成完整质量与性能目标。
