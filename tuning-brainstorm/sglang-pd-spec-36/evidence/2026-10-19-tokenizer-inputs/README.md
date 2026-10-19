# M001：真实 tokenizer 加载、模板差分与输入夹具

固定安装源码 `ce0a8fea5f4245839d174b2b8c55692be1810a7f`，目标镜像为 `sha256:c5cf2deb674cd1c8aba10a70d8dcb3e413d3739f6a54808c568e11d711d2e263`。**本轮运行了完整 SGLang tokenizer 加载器、后处理修复及原样异步分词类，取得 CPU 正确性和输入身份结果；没有 HTTP、模型权重、NPU 或性能对照。整个 PD＋MTP 任务仍 E0。**

## 身份与执行范围

| 项目 | 实际记录 |
| --- | --- |
| Transformers / tokenizers / Jinja2 | 5.12.1 / 0.22.2 / 3.1.6 |
| PyTorch | 2.10.0+cpu，显式 CPU engine |
| 模型候选 config 哈希 | ca8f2f47b07919a514c0ca223dc2ea2bc7445afaa5ac76c013a3784e096426ca |
| tokenizer.json | 20,217,442 字节；19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d |
| chat_template.jinja | 10,734 字节；3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c |
| 实际 tokenizer 类 | transformers.tokenization_utils_tokenizers.TokenizersBackend，is_fast=True |
| vocab / EOS / PAD | 154856 / 154820 / 154820 |
| 底层组件 | pre_tokenizer=Sequence，decoder=ByteLevel |

tokenizer_config 的哈希、4 个模型文件哈希、13 个 canonical 源文件及 169 个实际导入 SGLang 文件哈希均已保存。额外 stop IDs 为空只表示没有附加这组 helper 的停止词；不能据此推断模型 EOS 集合或服务结束策略。

输入经过真实 `get_tokenizer → generic backend 重试/声明类加载 → _apply_post_load_fixes`。实际调用的 v5 组件恢复、BOS/EOS 兼容与特殊 token pattern 修复都有进入/返回记录，最终类仍是声明的 TokenizersBackend。没有换成裸 AutoTokenizer 或模拟 tokenizer 生成结果。

为运行 CPU 前处理，在同一镜像的独立 venv 中复用 490 个顶层包条目的原文件，排除 torch_npu 及两项 .pth 启动钩子；底层 TF/tokenizers/PyTorch 文件未编辑。CPU 环境与原 NPU 包可见性不同，结果不代表 NPU 服务进程的完整导入行为。

容器无网络/NPU 设备，根只读、cap-drop ALL、1 核/4 GiB；只挂载上述 config、tokenizer_config、tokenizer.json、chat_template 四个文件，均只读，没有挂载权重。缓存放在容器 /tmp。已删除任务容器；模型文件哈希与已有服务 ID/启动时间/重启计数前后相同。SGLang 导入曾尝试 npu-smi 信息查询，命令在此隔离环境不可用，没有可见设备或 NPU kernel。

## 实际正确性结果

见[成功结果的公开视图](cpu-result-public.json)和[容器回执](cpu-container-public.json)。完整 raw 结果保留本地，其哈希在 artifact-index 中；5 份批量输入移到独立文件，内容与 ID 哈希未变。

| 检查 | 实际结果 | 不能外推的部分 |
| --- | --- | --- |
| 8 种文本，含空串、中英文、组合 Unicode、空白、JSON、控制标记 | 普通编码与 RawTokenizer 一致；记录的自动 special 模式也一致 | 不证明全部语言/工具历史的穷尽等价 |
| 18 个 extra special tokens | 每个编码为一个相同 ID | 不代表完整 EOS/grammar 策略已验证 |
| 18 组模板参数 | tokenize=True 与先 render 再 encode 的 ID 流一致 | 只测直接模板调用；HTTP 层尚未执行 |
| 两个同时到达的 512-token 文本 | 合成一次 batch_size=2，逐请求输出与标量分词相同 | 没有测收益 |
| 三个同时到达的 32768-token 文本 | 合成一次 batch_size=3，逐请求输出相同 | 不证明 NPU KV 容量或端到端吞吐 |
| 两请求 kwargs 不同 | 回退两次标量调用，输出保持各自参数 | 不主张跨参数强行合批 |
| 0/10/20 ms 到达的短文本 | 记录为三次单请求，输出正确 | 到达时序是本 CPU 夹具 |
| max_batch_size=1 控制 | 两次标量调用，输出与 max=32 相同 | 参数速度比较仍待 A/A、A/B |
| 取消与错误恢复 | 取消一个 future 后另一个正常完成；真实 tokenizer 非法 padding 的两个 ValueError 均传回，下一请求恢复 | 不证明 HTTP/SSE、模型取消或页回收 |

异步生命周期共 7 组场景。取消用 CPU 线程门控控制时机，真正编码仍委派给同一个 tokenizer；错误使用真实非法 padding 参数触发，没有模拟 tokenizer 输出。

## 模板参数：相同长度仍可能不同输入

单用户探针三种 reasoning_effort 都产生 15 个 token，但 ID 哈希不同：

| 参数 | ID SHA256 前缀 |
| --- | --- |
| max / 未提供 / None / invalid | d32ff5d8f0 |
| low | e96906f179 |
| high | a7134cc492 |

所测直接模板调用中，enable_thinking 未提供/False/True 没有改变这些 ID，输出仍含 `<think>` 前缀。该结果不证明 HTTP API 会忽略这个参数；HTTP 层可能做额外处理，本轮未执行。

实际 ServingChat 的 roundtrip 探针原样提取运行，返回 False（该探针未发现损失），auto_adds_specials=False。这个单消息探针不能代替所有历史/工具消息验证。性能对照必须固定模板字节、参数、特殊 token 和实际 ID 流，只固定 token 数量不够。

## 冻结的 CPU 输入候选

[输入清单](input-fixtures/manifest.json)保存 5 份 synthetic 输入：2×512 和 3×32768，共 99,328 个 token。每份含 messages、rendered_text、input_ids、模板参数及 ID 哈希；分别用 c2-short-0/1 和 c3-long-0/1/2 标识。

这 5 份输入在模板直接编码、render 后普通编码及异步分词中保持相同 ID。固定 reasoning_effort=max、add_generation_prompt=True，已登记 tokenizer-cpu-c2 / tokenizer-cpu-c3 两个 CPU 用例。输出是 Python 整数列表，未声称此阶段已经产生特定设备 tensor dtype。

它们是**已固定的 CPU 输入候选**，尚未分配到本任务的 P/D 服务；权重摘要、服务资源、输出采样/EOS/长度策略和真正的 HTTP/PD token 流仍待冻结，不能算本任务服务基线。

## 首次失败与可用性记录

| Run | 实际结果 |
| --- | --- |
| R001 | 主控采集器准备阶段匹配到两个 print 标记而停止；外层 shell 错误继续，缺失 spec 的命令失败一次。原始缺失状态、命令错误与 setup audit 保留，没有读取远端模型 |
| R002 / R003 | 只读身份与依赖源码采集通过 |
| R004 | 完整导入因默认缓存目录只读而失败，未加载 tokenizer；容器已清理 |
| R005 | 缓存移到 /tmp 后，原镜像导入链探测可选 torch_npu，缺 libascend_hal.so，结果 UNAVAILABLE；不是分词通过 |
| R006 | 独立 CPU 依赖环境下完整 SGLang loader、修复及上述检查 PASS |

没有覆盖旧错误、挂驱动绕过隔离、修改生产源码或更换 tokenizer 文件。启动回执用时包含不同阶段，不能用作前后速度对照。

## 成本、上界与下一轮

前处理至少要读取输入字节、生成模型实际需要的 token ID，并处理模板和特殊标记。Rust 编码、Python 包装、排队与返回值分装的实际成本还需分别测；本轮没有可达 CPU 带宽、单位 token 时间或 NPU 校准值。

动态批量分词是已有能力。队列为空时源码立即处理；有积压时可能等待批窗口或最大批量，批量调用减少 Python 调用次数也可能增加首请求等待。下一轮固定相同输入、到达过程、2 ms 窗口、CPU 资源与版本，只比较 max_batch_size=1/32，先估计 A/A 波动，再做交织 A/B，并同时记录 wall/CPU 时间和限流状态。只报告局部 CPU 效果，另验证真实服务的配置开关和 TTFT。

M001 仍缺 HTTP 路由/模型前处理的整体验证、reasoning parser、更多历史/工具模板与正式性能数据；整个 36 点目标继续，Ascend、PD、MTP 约束保留。

## 可追溯入口

[source-index.json](source-index.json)、[zcode-audit.json](zcode-audit.json)、[artifact-index.json](artifact-index.json)记录固定来源、精确执行 session、命令、退出码及哈希。配置别名为 deepseek-flash，实际后台 release UNKNOWN；不发布认证设置或 raw model I/O。

脚本按本地 Task 布局运行；CPU 依赖视图、4 个只读挂载和缓存配置均有代码/回执，不应当作 NPU 部署模板。公开结果把大输入拆为独立 fixture，原始结果本地保留。模型 tokenizer.json 本身没有上传仓库。
