# 按框架分支维护

用户确认继续在Inference-Foundry内按框架建立独立分支。SGLang新增适配、部署、数值/图门槛、AISBench与profile归本分支；vLLM新增内容归[独立vLLM分支](https://github.com/wio1997/Inference-Foundry/tree/vllm-deepseek-v41flash-w4a8-910b-pd)。原混合分支保留为历史，不继续承接新测量。既有证据不删除、不改写成另一框架的验收结果。

本分支从[混合历史固定提交](https://github.com/wio1997/Inference-Foundry/tree/4573c8cadef92f8bd0622fd5543ffe27c011a13b)连续继承，保留既有链接及SHA证明。旧vLLM源码/接口组件属于冻结比较背景，后续vLLM权威更新只在其分支进行；共享权重固定文件继续以本分支metadata为权威，另一分支通过固定commit链接引用。没有把两框架重复存放的历史快照当成两个独立实验。

用户提供的20261117其他910B部署记录已取得Git blob与repository commit。版本/硬件/负载/图执行差异、可复用机制和未验证项统一维护于[vLLM参考适用性分析](https://github.com/wio1997/Inference-Foundry/blob/vllm-deepseek-v41flash-w4a8-910b-pd/docs/002-reference-compatibility.md)，此处保留指针。继续沿用本任务固定SGLang镜像、完整DSPARK图和四档C16协议，不用参考环境的不同负载结果补齐当前缺失矩阵。
