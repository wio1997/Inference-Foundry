# Eager 仍复现错误：真实 attention 与缓存历史核对

固定 SGLang daily A2 关闭 prefill/decode graph、保持观测关闭时，16条原始短题全部完成传输，严格匹配8/16，仍有算术错误及三条代码尾 token 遗漏。随后在两条请求上取得八个rank的240份真实 attention 快照；采样输出符合 CPU FP32 参考的预设容差，按原生索引顺序比较的192组缓存历史逐元素一致。这些结果缩小排查范围，未修复模型质量。

## 服务与变更范围

硬件、权重及依赖沿用[上一阶段](043-standalone-and-serial-semantic-isolation.md)：910B1/aarch64、W4A8，SGLang镜像 `sha256:7da963869ee54984dfeea2445befdf181710ef846448466db2361247759904ec`、CANN9.1、torch2.10/torch_npu2.10post6。vLLM固定0.32基线没有重跑，也没有把此处的诊断配置当作最终对齐配置。

D1060在所有诊断结束、原始文件冻结和两次空队列检查后，通过原控制器受控退休；八卡owner释放、原容器、缓存与日志保留。[退休记录](../evidence/eager_real_attention_G1113/G1097_D1060_actual_retirement_parsed.json)与[新D1098启动](../evidence/eager_real_attention_G1113/G1098_eager_D_actual_start_parsed.json)可核对实际身份。P1038和路由保留空闲。工作负载锁在新实例就绪后受控交接，未抢锁或重复起实例。

D1098仍为 standalone/null、DSPARK=None、TP8/EP8、prefix开启、overlap调度保留、max-running16、上下文263296、BF16 KV、FP4 index及原适配计算。仅关闭decode graph，并增加默认未开启的有界观测包装。包装先调用原attention，随后在明确开启时复制少量输入/输出；未修改喂给模型的张量。CPU导入检查只证明原函数身份、代码对象与签名保留，检查未初始化NPU。

[实际八卡门槛](../evidence/eager_real_attention_G1113/G1103_eager_D_actual_gate_parsed.json)确认各1384参数严格装载、missing=[]、无draft，API的prefill/decode backend均为disabled，没有target decode图捕获完成记录。C2 ring8、17请求槽、FP32 `[146,1024]`、三层1794048字节计价均与预期一致。启动器继承的scope字符串仍有旧full-graph措辞，实际argv/API和本门槛为权威；disabled配置中的默认max_bs不代表启用了图。

完整60份部署文件的[原始SHA与公开源码索引](../evidence/eager_real_attention_G1113/G1113_eager_deployment_source_index.json)复用未改文件，记录新增版本及脱敏副本SHA。计数和装载账本不等于逐expert物理字节或模型正确性证明。

## 观测关闭时的短题

[父级核查](../evidence/eager_real_attention_G1113/G1105_parent_unarmed_eager_semantics.json)重建16条SSE文本、usage、DONE、SHA，并确认请求JSON与上一C1诊断逐字节相同：T0、seed0、thinking=false、max128、零重试。此时ARM文件不存在，未产生张量快照。

- 16/16传输成功，严格匹配8/16：算术3/8、代码5/8。
- “13×5＋7”输出78，预期72；另有98/131/170等错误整数。
- `KD32190DFD`、`K0E411E800`、`K46C76D22E`仍分别缺少末尾D、800、E。
- 相比旧C1，部分输出变化；重载epoch、打包和缓存状态也有差异，不能把分数变化当作只有一个变量的质量收益。

故障在实际图关闭且观测未开启时仍存在，足以说明decode graph开启并非故障出现的必要条件；不排除其它图相关问题。完整1319题质量仍未验收。

## 两条请求的真实快照

开启观测后，仅执行原提示中的math0和bind1，使用已核对的原始prompt ID、原生非流式`/generate`、T0/sampling_seed0、max128和logprob返回。每rank每压缩比最多12次，只取首个对应层及每次张量的首行/末行。实际得到八rank、压缩比0/1/2各一个采样层，共240份快照、78170672字节。[冻结索引](../evidence/eager_real_attention_G1113/G1106_armed_attention_raw_2204_parsed.json)逐文件记录SHA。原始`.pt`继续留在受控服务器，没有公开或下载到分析客户端。

两条请求实际完成；math0输出72，bind1仍输出`K0E411E`并生成EOS=1，缺少800。观测同步、接口和logprob路径均有变化，不能把math0答对称为修复。客户端退出、P/D ready及空队列核对后，ARM文件改名保留，观测已关闭，再开始CPU分析。

实际C1张量仍有padding：prefill为32/40行，decode为8行，每行64heads×512维；它们不是32/40/8个真实请求。采样首行对应本次真实请求；张量末行属于padding。prefill的末个真实prompt行未被本次采集覆盖，不得把末行padding的检查外推给它。

## 数值参考、真实行与缓存历史

[CPU参考结果](../evidence/eager_real_attention_G1113/G1107_actual_attention_CPU_result_2204_parsed.json)在服务器无NPU客户端运行，torch2.12.0+cpu、4CPU限制。它将采样Q与实际选中BF16 KV转FP32，计算共享K/V attention，并把每head的sink作为值为0的额外softmax项。先用独立可手算的重复键、无效索引及不同sink用例验证参考程序；这不是NPU模型验收。

240份快照的480个采样行含真实行及padding行，均有限值。与FP32结果四舍五入到BF16后的最大绝对差为0.00390625；没有元素超出`max(2×BF16 ULP, 2e-5+2e-5×|FP32参考值|)`这一诊断容差。该容差是本次算子排查的判据，不是完整模型精度标准。

缓存核对保留了两次无效解释及更正：最初用张量末行做历史比较，因它是padding得到0组，不能记通过；改用真实首行后，若直接比较SWA数组前缀会出现192组差异。固定[原生元数据源码](../code/eager_diagnostics_G1113/native_attn_metadata_kernels.py)的`build_causal_swa_page_indices`按`pos-arange(window)`排列，即当前到历史；C1/C2压缩索引则按位置升序。未重排的差异不证明缓存损坏。

[按原生顺序的最终核查](../evidence/eager_real_attention_G1113/G1111_native_order_history_2204_parsed.json)先逐条验证真实首行的prompt ID、生成ID和位置，再将SWA有效行倒序为逻辑历史顺序：

| 检查范围 | 结果 |
| --- | --- |
| 八rank、三采样层的真实行输入/生成ID/位置 | 240条全部对齐，0错误 |
| 短因果前缀的SWA及压缩有效行计数 | 240条符合当前短上下文计数，0差异 |
| 同一请求相邻采样步骤的历史KV | 192组、4100096个元素，逐元素一致 |
| EOS作为下一输入的额外forward | 48份快照，来自八rank×三采样层×两请求 |

缓存一致性使用BF16元素的`torch.equal`，未声称原始bit模式相同。最后一项是本诊断下实际观测的overlap后续计算，不能据此推算最终DSPARK/PD配置的无效计算比例或性能收益。原始无效解释的CPU记录与源码保留，没有覆盖或伪装成通过。

[父级结论](../evidence/eager_real_attention_G1113/G1113_parent_real_attention_assessment.json)仅支持：在这两条短请求、三个采样层、八rank的范围内，采集到的attention计算、真实行对齐及已比较的KV历史未见异常。它不验证上游Q/K构造、其它层、实际权重数值、LM head/logits/采样、长上下文稀疏选择或完整质量。

## 后续边界

下一步保持原请求和实际padding信息，核对LM head、logits及采样实际使用的行，必要时对真实hidden state和选中权重做参考计算。修复语义后，仍需恢复最终对齐的DSPARK/PD和图配置，完成1319题质量、SGLang四档AISBench首轮冷/热、最终匹配P8/D8 profile、差异地图及端到端收益验证。[当前状态](../metadata/current_eager_D1098.json)持续将这些目标标为未完成。
