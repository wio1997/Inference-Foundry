# Ascend 950DT 部署与复现代码

对象：DeepSeek-V4.1-Flash，单机 8×Ascend 950DT。

| 入口 | 用途 |
| --- | --- |
| [DeepInfer 部署手册](../../docs/ascend950/README.md) | 下载、编译、兼容补丁、拓扑、启动与功能验收 |
| [DeepInfer 部署脚本](../../deploy/ascend950/) | 固定源码及构建、启动代码 |
| [vLLM Ascend 部署](../../deploy/ascend950/vllm-ascend/README.md) | 官方镜像、补充算子构建、DSA 补丁、服务验收 |
| [实验代码](performance-20261021/) | 服务适配器、AISBench 接入、全链路采集与分析工具 |

## 代码目录

- `adapter/`：DeepInfer DP8 独立请求服务、原生序列 fork 的有限前缀复用；诊断版本与普通测试版本分别保留。
- `scripts/remote/`：两框架的服务启动器；保留原运行目录约定，迁移时需核对挂载、网卡、Python 环境和模型目录。
- `tools/e26_ais_model.py`、`e27_ais_model.py`、`e29_ais_model.py`：AISBench 请求映射、DP 路由、缓存命名空间及长首包接收缓冲。
- `tools/ais_bench_prefix_cache/`：客户端依赖的接入插件源码。
- `tools/e27_prepare.py`、`e29_prepare80.py`：负载与配置生成器，需要调用方提供其输入数据。
- `tools/e27_run_window.py`、`e29_run_window.py`：单窗口执行及测前测后排空检查。
- `tools/e33_*`：完整 cohort 采集、CPU/内存/工作量探针、原生算子清单、图执行、通信与时钟分析。

公开代码中没有内部测试配置、冻结输入、实验数据或原始 profile。运行测试和分析时，需要提供相应配置及原始材料；这些脚本不能替代缺失输入，也不是直接运行整套历史实验的一键包。启动脚本中的目录和容器名是原代码的运行约定，执行前应按部署环境调整。

DeepInfer 的图模式、ZOS、MegaMoE、Engram，以及 vLLM 的静态 Kernel、Decode 图和异步调度来自各自框架原生实现。本项目保留服务适配、官方算子构建、兼容补丁和探针代码；没有开展 NVIDIA 优化向 950 的移植。最终性能归因与整理后新主机重部署尚未完成。
