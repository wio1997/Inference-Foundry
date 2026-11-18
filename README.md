# vLLM · DeepSeek-V4.1-Flash W4A8 · 910B PD 对齐

本分支维护用户指定的vLLM-Ascend v0.32.0rc1适配、部署、AISBench和profile证据。当前[版本切换与原生源码核对](docs/008-required-vllm032-and-native-source-refresh.md)已归档；旧0.30源码、组件和G527配置仅保留历史身份，尚无新0.32完整模型或性能验收。SGLang维护于[独立分支](https://github.com/wio1997/Inference-Foundry/tree/sglang-deepseek-v41flash-w4a8-910b-pd)，历史混合资料固定于[原提交](https://github.com/wio1997/Inference-Foundry/tree/4573c8cadef92f8bd0622fd5543ffe27c011a13b)。此分支从同一历史来源提取vLLM原生源码、图契约、成功与失败组件；SG的完整装载或数值门槛不计为vLLM通过。

历史0.30原生Host UVA组件在8张910B1上通过80用例、16图、48次变化回放；该结果尚未在新0.32镜像重验，完整模型target/draft、跨机PD、完整图、质量及四档性能未验收。任务保持32K/64K/128K/256K、C16、输出1024，独立充分预热后首次cold32/hot32，记录TTFT、TPOT、输入/输出/总吞吐。

- [当前未验收范围](evidence/readiness.json)
- [固定镜像实际源码与SHA](evidence/source_provenance.json)
- [原生v2/DSPARK图源码契约](evidence/native_graph_contract.json)
- [G50实际Host UVA组件](evidence/components/G50_vllm/receipt.json)
- [原G49循环导入失败](evidence/components/G49_vllm_import_failure/receipt.json)
- [用户提供的已有适配记录适用性](docs/002-reference-compatibility.md)
- [分支范围与共享权重/数据权威指针](metadata/framework_scope.json)

公开产物脱敏，完整权重和大型trace不入Git。`python3 scripts/validate_vllm_archive.py`只做档案核验，不启动模型或NPU。

[固定镜像PD/DSPARK配对prefix源码契约](docs/003-fixed-image-PD-prefix-contract.md)

[PR #12 与固定镜像的逐项源码比较](docs/004-PR12-source-comparison.md)

[固定V2运行时和SDK符号检查范围](docs/005-fixed-v2-runtime-and-SDK-scope.md)

[固定原生hybrid PD双角色配置准备](docs/006-native-hybrid-PD-preparation.md)

[当前公共预算下的固定原生hybrid PD准备](docs/007-current-common-workspace-budget.md)

[原生0.32功能、计数与已知问题](docs/009-native032-functional-readiness-and-detokenizer-limit.md)；完整质量与性能验收仍待完成。

[AISBench启动目录失败与修复](docs/010-AISBench-readonly-cwd-failure-and-repair.md)。
