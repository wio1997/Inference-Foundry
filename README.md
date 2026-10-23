# Inference-Foundry · Ascend 950DT

本分支保存 DeepSeek-V4.1-Flash 在单机 8×Ascend 950DT 上的部署与复现代码。

- [部署与复现代码入口](reproduction/ascend950/README.md)
- [DeepInfer 部署操作手册](docs/ascend950/README.md)
- [vLLM Ascend 部署脚本](deploy/ascend950/vllm-ascend/README.md)及[HTML 手册](docs/ascend950/vllm-ascend-deployment.html)
- [鲲鹏 950 硬件说明](docs/ascend950/KUNPENG950.md)

本次公开部署、适配、采集与分析代码，不包含内部测试配置、冻结输入、性能结果、原始 profile、模型或镜像。新主机重部署和最终性能归因尚未完成。

分支继承了其他模型和旧硬件的历史目录。可以在 950 分支删除无关文件并提交；main 不随之改变。将此分支合并回 main 时，需要审查这些删除。
