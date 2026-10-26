# 本次离线资源准备记录（禁止直接重放）

这些脚本是已经通过本机ZCode DeepSeek执行过的一次性资源管线源码，不是新对话的启动入口。V1遇到源旧vLLM tar blobSHA错误后停止；V2复用同一批15个node workers、Flash97SHA checkpoint，重新docker save固定image并压缩续传B103。当前正常后台工作不得重新执行V1/V2launcher，也不得改冻结文件/删锁。

目标只是统一/data/wio入口、局域网复制本地权重和Docker导入固定镜像，不启动推理容器、模型或性能请求。程序内部有限期后台等待不是Codex定时自动化；用户暂停的自动化保持暂停。程序完成后没有自动模型启动步骤。

本机当前任务目录与实际receipt见../../HANDOFF.md及../../DOCKER_ASSET_LAYOUT.md。需要修复时先读真实状态、退出/锁/partials，主控重新审查新的有界resumer和SHA，再经本机ZCode精确执行一次。不能放松归档层SHA或覆盖未知文件/标签。

仓库只保存代码/计划/校验索引，不保存二进制或provider配置。源1私有文件分发仅对本批15台LAN地址开放，含Range断点与逐文件SHA，无任意路径/目录浏览；没有改系统NFS/SDK/驱动/BIOS。源码快照是6efbc385，权重与镜像固定身份另见公共资源表。
