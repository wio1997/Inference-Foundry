# 本任务服务停止、NPU预留释放与暂停

用户明确要求完成当前R092、提交实验记录后，停止本任务的Prefill、Decode与共享存储，释放NPU预留并保留模型、容器及日志。实验已先提交为fb9f91a97582dc78f646abe56fbd0e3f1157fa07，随后才执行R093；没有再发送生成请求。

R093对三个完整CID、task/run标签及两个reservation Pod UID进行守卫。按Decode→Prefill→store顺序停止，停止前后完整自有日志写入各节点本任务R093目录并保存SHA；保留原容器及模型文件。核验原docker top进程的PID与启动tick，停止后均无同一进程仍活跃。容器退出码如实保留，停机中的非零引擎退出码不被改写为模型测试成功。

两个NPU预留Pod通过Kubernetes DeleteOptions UID precondition删除，随后确认旧UID消失；不会删除同名替代Pod。44与46各16逻辑NPU的本任务预留已释放。访问Pod、其他服务和模型磁盘未动；PriorityClass只保留无硬件占用的调度元数据。

本任务当前没有活跃模型服务、共享存储或实验客户端。按用户要求暂停，等待新服务器和明确恢复指令；目标仍未完成。原长短PD＋MTP功能、条件取消/重试和局部通信显存证据保留，稳定性能A/A、优化A/B、完整36机制原生覆盖与硬件前沿仍待继续。

- [独立停机复核与资源释放摘要](notes/L039-stop-review.json)。
- [完整停止响应、PID/start-tick与日志SHA](runs/L039-R093-shutdown/remote-result.json)。
- [最终迁移交接](notes/L039-server-migration-handoff.json)、[任务续跑入口](../../HANDOFF.md)。
- [先提交的EOS失败与取消/重试实验](../2026-10-19-910c-lifecycle-pause/README.md)。
- [冻结执行与退出回执](runs/L039-R093-frozen-controller/receipt.json)、[41条冻结命令元数据审计](notes/L039-group-memory-zcode-audit.json)。
