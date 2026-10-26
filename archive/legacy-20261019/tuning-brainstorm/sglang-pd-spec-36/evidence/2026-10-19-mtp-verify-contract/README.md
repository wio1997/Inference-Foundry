# M031：Ascend EAGLE 接受合同与源码模型

本轮把安装版 EAGLE 调用追到 NPU 贪心验证文件，明确了观测 `steps=3 / topk=1 / draft_width=4` 的必要形状条件，并用固定源码完成123组链式CPU源码模型对照、72组局部门控夹具和32组贪心分支判定。**没有执行 Triton/CANN/NPU 算子或本任务 PD＋MTP 模型推理**，整体仍E0，服务器 accepted_baseline=null。

## 两个代码身份

- SGLang安装源码固定为 `ce0a8fea5f4245839d174b2b8c55692be1810a7f`。本轮收集的原文件均与该Git对象逐字节比对；[源索引](source-index.json)记录每个文件和采集Run。
- Python路径扫描找到 `sgl_kernel_npu 2026.9.0`，贪心文件SHA-256为 `defb03e6dd0a479e550a4ddfcd5f88f1ecd0bb37de9ac2171c01d210e03413c8`。这是安装包文件候选身份，缺少Git来源，也未内省本任务加载的原生算子。不能把主SGLang commit当成NPU包commit。

包初始化会加载torch_npu及`libsgl_kernel_npu.so`；本轮只读文件，没有导入该包。`eagle_utils.verify_tree_greedy_func`的NPU分支把`predicts/accept_index/accept_token_num/candidates/retrive_*/target_predict`传到此Python入口。拼写`retrive_*`在两端一致，不是本轮发现的接口错误。

## 配置门控与形状

固定pipeline中PD hook、NPU hook、speculative hook分别位于127、191、285行。NPU默认前缀强制attention/prefill/decode backend为`ascend`，未给page时设128。EAGLE尾部在topk=1时将草稿宽度归一化为`spec_steps+1`；topk>1且page>1时，只允许`flashinfer/fa3/triton`，所以在保持ascend时此尾部条件会拒绝。

72组夹具覆盖topk1/2/4、page未给/1/128、四种输入backend、宽度4/8：32组被所检查尾部门控拒绝；topk1全部归一化为宽度4。page=1的树形组合在这几段语句中通过，不证明完整ServerArgs、模型、attention或NPU算子支持它。测试执行的是实际默认前缀和EAGLE尾部语句，resolution对象由显式夹具提供；**不是完整启动检查**。

`EagleVerifyInput.max_tree_depth=spec_steps+1`，主调用按`[batch,max_tree_depth]`分配int32 accept_index。安装NPU内核按`req_idx*draft_width`定位该数组行，同时按连续候选顺序比较`candidate[i]`和`target[i-1]`，未读取child/sibling两张拓扑表。故链式且`draft_width==depth`是这段实现的关键前提。观测3/1/4满足`4==4`，这只是形状相容；原生tree builder是否产生对应有序链尚待实测。

## 最小源码模型：做了什么

[协议](specs/L015-source-model.json)冻结实际内核、wrapper、调用方、NPU默认和pipeline的哈希。探针从实际AST提取kernel和wrapper，kernel只移除decorator/annotation，body AST保持不变；用带边界检查的纯Python标量/向量/指针夹具执行。独立oracle沿child/sibling树寻找匹配路径，计算预测、接受索引和bonus。

| 检查 | 数量 | 结果及范围 |
|---|---:|---|
| batch2/3、width=depth=4的所有接受前缀向量 | 80 | 完整预测/索引/草稿计数一致 |
| 其他宽度和batch的代数、padding样本 | 43 | 一致；不代表这些参数可启动 |
| 手工树形、宽度不等的条件反例 | 2 | 与树oracle不一致；其中一例模型边界检查报告store accept_index[4] size4 |
| NPU默认前缀＋EAGLE尾部门控夹具 | 72 | 32拒绝；所有原始输入与决议保留 |
| 完整`_verify_uses_greedy`布尔判定 | 32 | 记录真实分支结果；NPU身份本身不强制greedy |

源码模型用-1哨兵检查未写槽位；真实caller的predict初始值是0。内核body没有改动，但该模型不覆盖Triton编译、CANN扩展、原生指针、整数dtype溢出、stream/并发或性能。分配、加载和store的Python夹具不是NPU运行结果。两个反例是条件构造，不能命名为已发生的NPU越界或当前3/1/4故障。

首个R004在门控夹具出现NameError：复制的源码使用`server_args`，生成helper误用了`args`参数。该Run标为INVALID，内层退出1、外层CLI退出0和[stderr](first-results/R004-stderr.log)原样保留。R005只修复helper参数和新输出目录，源码、case协议和内核body均未改变。[修复说明](specs/L015-R005-repair.json)和[成功结果](source-model-result.json)可复核。123组结论以R005完整结果为准。

## 接受长度的三个不同含义

内核输出的`num_correct_drafts`只计草稿，观测配置的范围0–3。`eagle_sample`返回时用不原地的`+1`加上尾部/bonus，因此`accept_lens`范围1–4。`run_eagle_verify`用它增加seq_lens，提取bonus时以`accept_index.shape[1]`作stride；`_draft_extend_for_decode`再以`accept_lens-1`恢复正确草稿数。本模型实际执行了源文件的`+1`返回表达式，核对有效前缀、bonus与逆变换；其他消费者目前仍是源码证据。

不能把`draft_width=4`写成每轮固定接受4，也不能把配置steps3当接受长度分布。后续成本应以总draft/verify/commit工作除以实际有效输出token，保留每轮0–3草稿的分布。

## PD交接已追到的边界

Prefill的EAGLE结果把topk概率/ID、hidden state及可选DSA topk保存到请求；hidden/DSA路径存在显式CPU复制。`disaggregation/utils.py::MetadataBuffers`分配topk容量16、参数化hidden宽度及可选int32 DSA行，`get_buf`逐项clone；NPU分支把这些metadata放在NPU，bootstrap_room改用int64。`set_buf`在hidden非空的分支写推测状态，并在DSA缺省时填-1。Decode从相同tuple取出字段并保存到请求，DSA全负行恢复成None。

这证明存在对应字段和消费者，**不证明PD传输完成或slot复用安全**。固定的16列容量、有效topk如何截取、缺省/旧slot清理、clone后的生命周期、NPU D2D可见性与bootstrap_room校验仍须通过真实消费者或最小差分确认。不能仅凭字段名称把PD＋MTP标为已支持。

## 下一道原生和服务门槛

[待执行的原生检查合同](native-next-contract.json)先限制为width/depth4、batch2/3、全部80种前缀组合，记录实际导入函数、包文件、Torch/Torch-NPU/Triton/CANN及设备身份。只在明确分配给本任务的资源上执行，先比较真实NPU kernel与独立oracle，再验证真实tree builder生成的retrieve/positions/mask。条件越界示例不会直接送入原生内核。

随后在任务自己的P/D角色上冻结相同模型/权重/量化/采样/输入ID、合法MTP参数和资源，验证首次输出、bonus、seq_lens、请求取消/EOS/重试及slot复用，最后建立TTFT/ITL、接受长度和有效token/s基线。当前已有单引擎观察不作PD基线，44/48历史空闲不作资源分配证据。本轮没有运行模型权重或修改服务，也没有更改MTP深度或接受阈值。

Loop总体INCONCLUSIVE：源码和数学边界已有证据，native、实际PD交接、非greedy采样及性能仍未证明。下一轮先把实际MetadataBuffers和prebuilt消费者的topk/clone/slot合同做最小CPU差分，再推进上述原生门槛。

## 复核

- [源索引](source-index.json)、sources/installed和sources/installed-kernel-npu；原生包版本与Git身份分别记录。
- [源码模型结果](source-model-result.json)、specs、scripts、[ZCode精确session审计](zcode-audit.json)、[产物索引](artifact-index.json)。每个冻结命令执行一次，deepseek-flash是会话元数据别名，后端发布版本UNKNOWN；不以CLI0替代内层成功。
- 源码模型可在scratch root保留本目录关系后执行`python3 scripts/execute_frozen.py specs/L015-R005.json`，无Torch、Triton或NPU依赖。只读采集脚本保留原SSH/Pod/容器合同；跨环境需要新spec，不修改旧输入后沿用旧证据。

无模型I/O、私有provider配置、凭据、权重或二进制库上传。
