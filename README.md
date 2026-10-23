# Inference-Foundry · Ascend 950DT

本分支面向 DeepSeek-V4.1-Flash 在单机 8×Ascend 950DT 上的部署和性能复现。

- [950 部署与复现总入口](reproduction/ascend950/README.md)
- [DeepInfer 部署操作手册](docs/ascend950/README.md)
- [vLLM Ascend 部署脚本](deploy/ascend950/vllm-ascend/README.md)及[HTML 手册](docs/ascend950/vllm-ascend-deployment.html)
- [32K / 80K 性能测试复现](docs/ascend950/PERFORMANCE_REPRODUCTION.md)
- [32K C32 全链路分析状态与缺项](docs/ascend950/FULL_CHAIN_STATUS.md)
- [鲲鹏 950 硬件说明](docs/ascend950/KUNPENG950.md)

已保存部署、补丁、测试及采集分析代码。公开包不包含模型、镜像和完整原始 profile；最终因果归因尚未完成。新主机重部署未执行，不把代码整理当作新增测试结果。

分支由已有仓库派生，保留了其他模型和旧硬件的历史目录。当前 950 工作从上面的入口进入即可。可以在本分支删除无关文件并提交；这些删除只作用于本分支，合并到 main 时才可能带入 main。
