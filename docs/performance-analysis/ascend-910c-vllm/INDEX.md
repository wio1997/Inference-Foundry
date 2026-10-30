# 研究目录：DeepSeek V4.1 Flash W8A8 / Ascend 910C

本目录对应 2026-10-30 的研究交付：32K 输入、1K 输出、并发 8、约 90% 前缀缓存命中。研究与后续优化实现分开，当前没有候选的整模型提速结论。

## 推荐阅读顺序

| 内容 | 文件 |
|---|---|
| 完整推理过程与优化机会报告 | [README.md](README.md)；先读正文，历史折叠附录按需阅读。 |
| 一手资料、论文与适用差异 | [REFERENCES.md](REFERENCES.md) |
| 融合、拆分重组与已有机制 | [FUSION-INVENTORY.md](FUSION-INVENTORY.md) |
| 历史摘要与条件理论模型 | [evidence-summary.json](evidence-summary.json) |

## 最新数据及复算入口

| 数据范围 | 匿名数据 | 复算脚本 | 重要限制 |
|---|---|---|---|
| 同负载请求/服务计数/粗 NPU 快照 | [official-capture.json](official-capture.json) | [verify_official_capture.py](verify_official_capture.py) | 一轮无采集参考、一轮采集；NPU 实际约 2 s/帧，不是带宽或算子利用率。 |
| 官方双窗×16 rank 算子类型 | [official-operators.json](official-operators.json) | [verify_official_operators.py](verify_official_operators.py) | 各卡各窗所有 model/stream；不是完整请求关键路径。 |
| 与候选对应的 12 个完整形状组 | [official-operator-shapes.json](official-operator-shapes.json) | 同上 | 全 220 组和原 CSV 在私有任务目录；池化重复调用不是独立实验。 |
| 实际双 EngineCore CPU 栈/计数 | [framework-cpu.json](framework-cpu.json) | [verify_framework_cpu.py](verify_framework_cpu.py) | 20 s 栈采样含 idle 尾部，与设备 profile 不同次，栈比例不是可删时延。 |
| 历史主层/草稿及其他摘要 | [evidence-summary.json](evidence-summary.json) | [verify_summary.py](verify_summary.py) | 摘要算术复算不独立验证原始采样、函数归属或硬件上界。 |

在本目录运行上表四个 `verify_*.py` 只读现有 JSON，不需要 NPU。它们不重新发送私有负载，也不下载权重。

## 历史专项材料

| 主题 | 数据与对应脚本 |
|---|---|
| 两级输出投影的配对样本 | [projection-pairs.json](projection-pairs.json)、[verify-projection-pairs.py](verify-projection-pairs.py) |
| 旧基线的引擎直方图 | [baseline-engine-histograms.json](baseline-engine-histograms.json)、[verify-baseline-engines.py](verify-baseline-engines.py) |
| DSpark 提议/接受与前缀 | [engine-spec-decode.json](engine-spec-decode.json)、[spec-acceptance-prefix.json](spec-acceptance-prefix.json)、[verify-engine-spec-decode.py](verify-engine-spec-decode.py) |
| F16 条件理论示例 | [f16-conditional-example.json](f16-conditional-example.json)、[verify-f16-conditional-example.py](verify-f16-conditional-example.py) |

## 冻结环境与工作负载

| 项目 | 已记录值 |
|---|---|
| 模型/精度 | DeepSeek-V4.1-Flash-1017-w8a8；Ascend W8A8 动态量化，混合 BF16/FP32。 |
| 硬件/并行 | Ascend 910C/A3 项目标识；16 逻辑芯片，DP2×TP8，expert parallel。 |
| 负载 | 每轮 8 请求×32768 输入/1024 输出；命中 235904/262144，0 抢占。 |
| vLLM / Ascend | distribution `0.30.0+empty` / `0.30.0rc2`。 |
| torch / torch_npu | `2.10.0+cpu` / `2.10.0.post7`。 |
| CANN / Python | 9.2.0 / 3.12.13；aarch64。 |
| vLLM 源码 | `ced6857afa0ea7b2e3f0846a62e1394e90f15607`。 |
| vLLM-Ascend 源码 | `8fde6d4e335fe300e2b949aab0e31659cce0811d`。 |
| 镜像实际 ID | `sha256:ce42d65b2403992ad2e2b06c61f4f048babb57e455cfe307b018345e36c5e98b`。 |
| serve.sh SHA256 | `c297201185b5676254af4d7693279f89bfa60f9c76eb340058a78ecf730b7bba`。 |
| 已开启机制 | 异步调度/batch queue、fused MC2、DSA CP、FlashComm1、shared-expert DP、CPU binding、decode graph、DSpark 5、Engram CPU 卸载/DP 共享。详细参数见报告及私有环境快照。 |

原始 NPU 采集 run 为 `official-20261030T084726Z-cb252d1d`；CPU run 为 `framework-20261030T091949Z-3d9bcf87`。两者采集已完成，原常驻服务已恢复。新对话应核对实时状态，不使用旧 PID 或照搬历史实验入口重跑。

本目录只发布整理后的报告、匿名数据和复算脚本。完整原始 trace/CSV、私有冻结输入、响应、权重、容器挂载和 SSH 入口保存在本机与本人服务器任务目录。精确重跑需要这些原始条件；公开摘要不能替代它们。

研究交付 commit 为 `9e2870bb9982d83377b180f9523adadf8e5343bd`。本目录和环境交接是后续导航补充，不是新的性能实验。不要把仓库其他 W4A8/910B3/c12 项目的状态混入此研究。
