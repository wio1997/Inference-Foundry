# 原GLM53 2P2D接手提示

在wio1997/Inference-Foundry的codex/ascend-sglang-vllm-comparison分支，负责ascend-framework-comparison/tracks/glm53-2p2d。先读公共AGENTS、HANDOFF、PLAN、BENCHMARK_CONTRACT、RESOURCES和本轨道STATE；再核验实际进程/锁与设备，不能把归档当当前事实。

vLLM P30/31、D43/45；SGLang P34/35、D36/41。备用32/33/37；38和42排除，不再用42管理Pod。用户授权停掉的33个SGLang相关容器已经停用并保留，不自动恢复。新2P2D尚未部署；2P只有32die，重新核算TP/attentionDP/EP与KV/MemCache容量，备用机不暗中参与性能资源。

用户不要共享长前缀：C条独立74400-token前缀各自预热，连续复用四轮、新后缀，4C条完整输入各不相同且实际80000/600；C16/32/64对应64/128/256条。预热不计性能，正式保留自然缓存。旧RDMA共享前缀、旧4P2D与新2P2D不得直接算框架收益。

镜像权重先离线准备并核对路径/分片/身份，再真实已知答案、PD/SDMA、图/MTP、缓存正确性，最后新AISBench基线。实际构建/部署/模型/性能必须父审查SHA并经本地ZCode DeepSeek精确launcher一次；保留失败、全程/稳态与真实运行/排队。没有基线前不声称优化收益，不使用历史一次性launcher，不重启自动化，不干扰Flash轨道。


用户将本对话设为GPT6.1 Sol High主控。严格遵守../../EXECUTION_CONTRACT.md：所有实际资源/构建/部署/启动服务/模型请求/性能/profile操作均由本机ZCode DeepSeek执行，主控只读、方案、具体实现/命令审查和结果核验。ZCode入口/本机资产管线交接位置见公共文档，不读provider凭证。先读真实锁和receipt，不重复现有上传/导入，不自行开定时。与另一轨道独立worktree/任务目录/节点/Meta和端口，满载profile/性能错峰。


最终资源更新：本组权重与固定候选Docker镜像都已准备并独立验收，详见../../DOCKER_RESOURCE_ACCEPTANCE.md和最终JSON；不再等待或重跑B103上传导入。仍须核设备独占与本轮参数，再经本机ZCode DeepSeek启动真实模型验证。资源READY不等于Flash已适配或新2P2D性能通过。
