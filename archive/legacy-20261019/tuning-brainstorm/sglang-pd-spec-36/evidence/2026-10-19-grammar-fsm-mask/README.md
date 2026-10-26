# M032：真实 grammar 缓存、状态隔离、回滚与 MTP host mask

固定 SGLang `ce0a8fea5f4245839d174b2b8c55692be1810a7f` 的 27 个源码/依赖文件已核验。本轮使用真实 XGrammar 0.2.7 编译器、matcher 与 CPU mask 应用，执行原样完整 backend/cache/grammar-manager 方法及 MTP host DFS，**19 组、256 次断言通过**。已有的 288 组能力谓词只复核来源与哈希，未重复执行。

结果支持现有缓存复制、请求独立状态、EOS/rollback 和 host 树 mask 的限定正确性；没有新增优化候选、原生推理或性能结论。整体仍 **E0_SPEC_ONLY / 无接受基线**。目标模型 config 身份恢复失败、实际 NPU 库/参数、reasoner wrapper、分布式同步及设备生命周期仍有明确门槛。

## 当前做法与本轮边界

已有缓存存放可复制的 grammar。命中时 `copy()` 重建 matcher、共享编译 context；它返回**初始 FSM**，不是“复制当前生成进度”的检查点。实际 grammar manager 在编译 Future ready 后缓存一份 fresh copy，再交给请求；同 key 的请求应有独立 matcher 与可变统计列表。

MTP 的 `traverse_tree` 从已经提交的 root 开始，依父行 mask 判断草稿是否合法，接受有效节点、填当前节点的下一 token mask，再逐个 rollback。`build_grammar_vocab_mask` 先调用上一批已提交 token 的 barrier，随后解析 host tree、生成 mask，并清掉遗留的 extend mask。这些是现有实现，不能重新称为融合或缓存优化。

原型 CPU 夹具使用精确历史 GLM-5.3 tokenizer，`V=len(tokenizer)=154856`、`EOS=tokenizer.eos_token_id`、`require_reasoning=False`。**它们不是目标模型 padded vocab、完整 EOS 集合或实际 parser 配置的替代值**，也不是原生 512/32768 输入负载。使用这套夹具只验证 grammar 组件；完整 Ascend PD+MTP 合同保持未完成。

## 先复核旧证据，未重复计算覆盖

[旧证据复核](prior-evidence-review.json)检查了原 `L002-R001-predicates/result.json` 与脚本：288 组 grammar/batch 判定、12 组 PD 角色分派，相关源哈希仍一致。旧脚本使用 CustomSpecAlgo 编译占位，不导入完整服务，不运行真实 FSM、设备或 collective；这些边界保留。它证明 EAGLE 家族具有 grammar-overlap 能力谓词，不能替代本轮 matcher 状态或原生支持实测。

源码默认 grammar 字段先为 None，serving hook 解析为 xgrammar；实际运行配置 UNKNOWN。历史进程采集器只记录了选定 CLI flags，未采集 grammar/parser/sampling flags，因此“记录中没有这些键”不等于真实服务关闭了它们。

## 版本与身份：三个层次分别记录

| 证据 | 可确认的版本/身份 | 不能推出什么 |
| --- | --- | --- |
| 固定源码 pyproject | torch 2.13.0、transformers 5.12.1、xgrammar 0.2.7 声明 | 目标容器安装了这些版本 |
| 历史镜像 CPU 观测 | torch 2.10.0+cpu；目标 xgrammar 未采集 | 当前任务已拥有该运行环境 |
| 本机独立 `.venv-grammar` | Python 3.12.14、Torch 2.8.0、XGrammar 0.2.7、TVM FFI 0.1.10、HF 5.12.1 / tokenizers 0.22.2、jsonschema 4.26.0 | 与目标镜像或 NPU 等价 |

[环境结果](environment/result.json)与[依赖锁](environment/pip-freeze.txt)保存实际导入路径。新环境通过任务内 `.pth` 只读复用既有 CPU/HF 包；保护文件哈希检查未变化。XGrammar macOS arm64 wheel 与官方 PyPI SHA256 匹配，包内 39 个文件再次冻结哈希；二进制不分发。

同一个公开 GLM-5.3 revision 的 config 恢复结果为 **UNAVAILABLE / hash mismatch**：历史期望 `ca8f2f47…426ca`，下载值 `3ac72612…2e69`。没有用它覆盖或替代原模型配置；[身份记录](environment/model-config-identity.json)保留完整 SHA256。历史 tokenizer 两个 SHA256 仍精确匹配，但 tokenizer 身份不能证明 model EOS/padded vocab。没有下载权重。

## 执行清单

| Run | 内容 | 结果与范围 |
| --- | --- | --- |
| R001 | 11 个 grammar/sampler/能力/config 文件 | PASS，固定 blob |
| R002 | 6 个依赖声明、schema 校验、mask/CPU helper 文件 | PASS，固定 blob |
| R003 | 独立真实 grammar 环境、官方 wheel 与 config 恢复 | 环境 PASS；config 身份子门槛 UNAVAILABLE，不伪装通过 |
| R004 | 6 个 MTP DFS、默认解析、平台/调用相关文件 | PASS，固定 blob |
| R005 | 真实 FSM/编译缓存、原方法与独立 oracle | PASS，19 组、256 次断言；不是 256 个独立用例 |
| R006 | 4 个 EAGLE V2/common verify/scheduler/D2H facade 文件 | PASS，调用链静态范围 |

六次冻结命令各执行一次、内外层退出码均为 0、输入哈希保持不变。[ZCode 审计](zcode-audit.json)只公开 alias、工具与退出元数据，后端模型 release UNKNOWN。没有由源码采集 PASS 推导设备成功，也没有把 R003 的 config mismatch 隐去。

## 实测内容与独立对照

[逐组 CPU 结果](grammar-cpu-result.json)和[冻结审查](controller-review.json)给出方法 AST、完整配置与边界。

- **2/3 请求冷/热缓存**：真实 ThreadPoolExecutor Future 经原 manager ready 处理进入缓存；消耗一个请求不改变另一请求或 cache seed 的 mask。热命中共享 compiled context，matcher 与统计列表各自独立。清空 compiler/backend cache 后，已有请求仍能使用原 context 完成。没有测量 C++ 内部真实编译次数或冷/热性能。
- **真实嵌套 JSON**：3 个合法对象先由独立 jsonschema 校验，再对每个 token 检查 mask/accept，完成后允许 EOS；EOS rollback 和 1/3-token rollback 对照独立顺序 replay。4 个在 enum、数值范围、数组长度或额外属性上违规的对象被独立 validator 和真实 FSM 拒绝。失败 accept 不改变 matcher mask/accepted history；`current_token` 是源码诊断字段，未冒充 FSM checkpoint。
- **mask/bit 对照**：原 SamplingBatchInfo mask 更新覆盖 active、finished、terminated、None 混合行；跳过的行保留全允许值。真实 XGrammar CPU mask 应用与独立 packed-bit 算术对照相等；下批更新后无旧约束行残留。CPU token-filter 的重复 ID、31/32 位边界、allow/reset 组合保持其他行不变。
- **MTP host 树**：2/3 请求、4 行链按历史三步/topk1/draft4 的形状建立合成 tree，root 已提交。原 barrier 先推进到该前缀、再 resolve tree；每行 mask 对照独立顺序 replay，DFS 后 FSM 与已提交 history 恢复。另测合法/非法 EOS sibling 和 `V+4` 越界 ID；拒绝节点不继续访问，未访问行仍为全允许。真实 CPU logits 应用也已执行。
- **队列生命周期策略**：受控真实 Future 分别为 pending / running；原 abort 对 pending 可取消、running 不可取消，ready 处理不发布已中止请求的 cache。running 后续完成也不“幽灵回填”。这部分的慢编译 job 是明确夹具，不是假称真实 compiler 超时、真实 scheduler cancel 或 KV 回收。

JSON validator 是独立语义对照；状态 replay 使用相同真实 matcher 作对照，验证缓存/遍历的状态管理，不构成对 XGrammar 底层算法的形式证明。完整服务构造、grammar factory/Reasoner/parser、RNG/完整 Sampler 未运行。

## 保留的边界与反例

1. `SamplingBatchInfo.grammars=[None,None]` 直接调用更新会抛 StopIteration。它与 None / 空列表不同；完整 caller 在 filter/merge 后是否会生成此状态尚未核验。**这是前提见证，不声明当前服务 bug，也未做修补。**
2. 受控编译 timeout 会缓存 InvalidGrammarObject；后续同 key 请求继续 abort，直到 backend reset。running Future 晚完成不会自动替换该失败项。这是观察到的策略，真实阈值/业务触发/分布式一致性未实测。
3. 源 schema 校验拒绝 `uniqueItems` 等无法保持的语义；NUL（包括 JSON 内转义的 NUL）在进入 converter 前被拒绝。没有降低 schema 强度换取可执行或性能。
4. 未访问的草稿行保持全允许，安全性还依赖真实 verify 的父 mask/树接受约束；仅检查各行不能证明完整接受算法安全。

## Ascend / PD / MTP 支持的剩余门槛

[固定源调用图](source-index.json)确认 EAGLE 家族总选 V2（包括 non-overlap）。common verify 在 target forward 前构造 device GrammarTree，forward 后借 scheduler 提供的 pending-FSM barrier 生成 mask；scheduler 实际回调继续调用 result processor 的 FSM advance。本轮 barrier 是显式 host fixture，不能替代真实 committed-token 顺序。

NPU mask 分支导入 `sgl_kernel_npu` 并调用 `torch.ops.npu.apply_token_bitmask`；CPU object 分支调用 XGrammar CPU API。没有将后者作为前者的成功。GrammarTree 的 D2H、完成 event、pin/nonblocking、TP/DP/PP collective、图执行/生命周期、原生 sampler/RNG 与真实模型输出均未运行。

还须冻结真实模型 config、EOS/think_end 集合、实际 grammar/sampling/parser/strict-thinking 参数，以及已安装 XGrammar/NPU 算子注册与依赖。grammar 是请求条件，不应成为所有推测请求的统一门槛；也不能通过关闭推测/PD或放松 schema 来制造支持结论。

## 必要工作、成本与裁决

保持当前 schema 语义所需的状态转移、候选合法性与请求隔离不可删除；compile cache 只能复用不可变产物，状态相关 mask 不可按 schema 永久复用。当前 packed-mask 接口为每行 `4*ceil(V/32)` 字节，verify tree 为 `B*N` 行；这是**当前接口体积**，不是不可消除的硬件下界。实际目标 V、CPU 暴露时间、D2H/H2D 重叠与 NPU apply 成本 UNKNOWN，不能由组件断言估计 headroom。

Loop **INCONCLUSIVE**：限定 CPU 假设获得支持，原生 PD+MTP、Reasoner/完整 sampling 集成与物理生命周期未完成。无候选改动、生产应用或性能收益。缓存命中/复制/树回滚都是既有能力；先测稳定原生基线与 cold compile / steady mask 分项，再选择优化。

[原生后续合同](native-next-contract.json)保留原 GLM-5.3、Ascend、PD 和推测解码，以及 2×512→128 / 3×32768→256 用例。44/48 资源尚未分配。历史参考仍为 CLI NEXTN / steps3 / topk1 / draft4，当前规范化 runtime UNKNOWN；本轮没有替换模型配置或切成单步。
