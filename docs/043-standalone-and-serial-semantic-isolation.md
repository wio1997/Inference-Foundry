# 单节点与串行诊断仍复现语义错误

固定 SGLang daily A2 在单节点 D、关闭 DSPARK、直接 HTTP 及单请求串行条件下仍出现算术错误和代码复制遗漏。32 个原始用例的实际提示 token 与固定 vLLM 0.32 编码器完全一致。当前证据缩小了故障范围，尚未定位具体算子，也未通过完整质量验收。

## 适用环境与实际服务

沿用原 910B1/aarch64、W4A8 权重，SGLang 镜像为 `sha256:7da963869ee54984dfeea2445befdf181710ef846448466db2361247759904ec`，CANN 9.1、torch 2.10、torch_npu 2.10 post6。实际参数、装载账本和图捕获记录见 [D1060 门槛](../evidence/standalone_semantics_G1094/G1068_parent_standalone_D_allrank_gate.json)及[运行快照](../evidence/standalone_semantics_G1094/G1068_standalone_D1060_actual_2204_parsed.json)。

旧 D1038 已[受控退休](../evidence/standalone_semantics_G1094/G1062_D1038_ACTUAL_RETIREMENT_JSON_parsed.json)，原容器、日志和缓存保留。新 D1060 的 `disaggregation_mode=null`、DSPARK=None、prefix 开启、TP8/EP8、最大运行请求16、上下文263296；prefill 图关闭，decode query1 full graph 的 bs=[8,16] 在八卡完成捕获。八份 target 账本各1384参数严格装载、missing=[]，没有 draft。

物理 C2 ring8 与原生共享计价继续保留；单节点原生请求槽为17，实际 FP32 pool `[146,1024]`，每层598016字节、三层1794048字节。这个门槛证明分配及计价，不证明完整计算正确。P1038 和原路由保留空闲，以下生成直接访问 D，未经过 P、路由或 PD 传输。[guard 记录](../evidence/standalone_semantics_G1094/G1067_standalone_D_guard_transfer_2203_parsed.json)绑定当前服务身份；后续实际生成前再次核对 canonical flock、服务和空队列。

## 原始请求和结果

全部客户端均为零重试、temperature=0、seed=0、thinking=false。短请求内容沿用 G999；各阶段原始请求、SSE、usage、DONE、源码和实际句柄均保存，SHA 与公开脱敏副本的对应关系在[归档索引](../evidence/standalone_semantics_G1094_provenance.json)。

| 诊断 | 请求与预算 | 传输结果 | 语义观察 |
| --- | --- | --- | --- |
| 单节点 C16 | 16短题/max128，加16条GSM前缀/max256 | 32/32完成 | 短题严格匹配5/16，算术1/8、代码4/8；GSM为3 stop、13 length |
| C16 首 token | 相同16短题，仅max_tokens改为1 | 16/16完成 | 全部completion_tokens=1、cached_tokens=0；部分算术首输出即错误 |
| 单节点 C1 串行 | 相同16短题/max128，仅客户端并发改为1 | 16/16完成 | 严格匹配7/16，算术3/8、代码4/8；全部cached_tokens=0 |

[单节点 C16 父级核查](../evidence/standalone_semantics_G1094/G1072_parent_single_NoSP_semantic_assessment.json)确认32份请求与此前 paired no-SP 诊断逐字节相同，重建32条 SSE 内容、usage、DONE 和 SHA。四条代码遗漏在两种部署中相同。GSM max256 仅用于观察生成前缀，不能作为1319题、max2048的完整质量分数。

[首 token 核查](../evidence/standalone_semantics_G1094/G1084_parent_first_token_assessment.json)确认只有预算128→1发生变化。例如“13×5＋7”首输出78，预期72；“14×6＋7”首输出92，预期91；另两题首输出131/170，预期135/160。其他题可能先输出计算中间值，因此不能把每个首 token 不匹配都评分为答案错误；max1也不用于代码完整性评分。该结果只说明部分错误输出无需后续生成步骤即可出现，尚不能排除共享计算、图初始化或输出路径问题。

[C1 父级核查](../evidence/standalone_semantics_G1094/G1087_parent_C1_semantic_assessment.json)重建16条响应并确认请求正文与 C16 完全相同。“13×5＋7”仍输出65。已核对的代码遗漏示例均缺少最后一个 tokenizer token，例如 `KD32190DFD` 输出 `KD32190DF`，`K0E411E800` 输出 `K0E411E`。全部八条代码输出与 C16 相同。算术结果随打包形态有变化，不能据此认定唯一的 overlap 或量化根因；但请求之间的并发不是现有故障出现的必要条件。

## 提示编码与采样字段核对

普通 HF tokenizer 没有 Jinja chat_template，不等于该服务没有模型编码器。固定 SGLang 源码优先按 model_type 选择 dsv41，通过 deepseek-recipe 0.2.0 编码。native vLLM 0.32 的固定镜像编码器来自 `tokenizers/deepseek_v41.py` 及 `deepseek_v41_encoding.py`，没有替用旧镜像同名文件。

[实际32条比对](../evidence/standalone_semantics_G1094/G1080_actual_prompt_token_comparison_2203_parsed.json)分别使用原生 vLLM 0.32 编码器、SG recipe 编码器和当前服务 `/tokenize`：三者的 token ID 序列全部完全相同。checkpoint tokenizer SHA 为 `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b`。结论仅覆盖这些单轮文本、thinking=false用例，不外推工具、多轮或视觉请求。

[固定协议源码](../code/actual_sampling_G1090/srt/entrypoints/openai/protocol.py)对显式 temperature=0 不做默认值替换，并将 Chat 的 seed 映射为 sampling_seed；[原生 SamplingParams](../code/actual_sampling_G1090/srt/sampling/sampling_params.py)将近零温度变为 top_k=1 的 greedy。当前源码证据不支持“temperature=0被当作默认采样”这一解释。

原生 `/generate` 的首次四用例探针误用了 sampling_params.seed，首个请求HTTP400后退出；失败的7份原始文件保留在[接口失败记录](../evidence/standalone_semantics_G1094/G1090_raw_generate_failure_2204_parsed.json)。首次错误响应正文未捕获，不能伪造其错误文字；固定源码确认该接口字段为 sampling_seed，后续独立探针修正客户端字段并增加错误正文保存，没有修改服务。

## 当前结论与剩余工作

已实测故障出现在没有 DSPARK、P、路由、PD 分离及请求间并发的配置中；额外的 DSPARK/PD 故障仍可能存在。提示编码差异在32条用例范围内已排除。[原生生成四用例核查](../evidence/standalone_semantics_G1094/G1093_parent_raw_token_assessment.json)已完成：bind0、bind1 返回的 output_ids 与 logprob token ID 一致，序列直接以 EOS=1 结束，没有生成所需的 D/800 尾 token，文本与这些 ID 一致。两例末步 EOS 分别领先正确尾 token 2.5/0.25 logit，不能解释为仅在流式输出或文本解码时丢字。该接口采用非流式和 logprob 返回，执行路径有变化；math0 此次输出正确72，不能宣称修复，也不能据此唯一归因采样或某个算子。下一步需沿共享 target forward、logits 和采样继续定位。

完整1319题质量、恢复最终对齐的 DSPARK/PD 配置、SGLang 四档 AISBench 首轮冷/热、最终匹配 P8/D8 profile 和差异地图、端到端优化收益仍未完成。当前诊断结果不能作为性能收益。见[当前状态](../metadata/current_standalone_D1060.json)。
