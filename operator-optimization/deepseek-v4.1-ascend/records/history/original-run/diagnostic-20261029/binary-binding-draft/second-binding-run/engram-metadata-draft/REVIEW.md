# Engram 入口元数据 wrapper：待审草稿

这不是已经启用的诊断，也没有修改模型、服务或容器。`engram_metadata.py` 导入不安装 wrapper；`install(cls)` 在 `WIO_ENGRAM_META_ENABLE` 不等于 `1` 时返回 False。目标只面向已核对的 `AscendParallelEngramEmbedding.embed_gathered(self, gathered, num_tokens)`，尚未接入现有启动脚本；以后由主控审核 ZCode 和恢复护栏后另行决定是否启动。

每次调用入口，只有 `WIO_DIAG_DIR/PROFILE_ACTIVE` 存在且是普通文件时才观察 `gathered`：shape、stride、storage_offset、dtype、device 类型和索引、contiguous、num_tokens 计数，以及随机进程代次、PID/TID、单调时间、调用序号。没有固定 shape 筛选，四次已观测的输入都可记录；未知维数、符号维、dtype 或过长字段拒绝并标记未知。`num_tokens` 是数量，不是 token 内容。仅普通 `torch.Tensor` 类型受支持，子类跳过，图编译或 replay 可能绕过 Python wrapper。

`tensor_instance` 和 `storage_wrapper_instance` 是只在当前进程代次内分配的整数，查找只用弱引用和 `is`，不调用 `id()`、`data_ptr()`、`_cdata`，不读取或输出真实地址和张量值。此 storage 标识只表示曾观察到同一个 Python storage wrapper 对象：包装对象可能被回收/重建，弱引用及 512 项表淘汰会产生新的匿名编号。它不是内存地址、物理 storage 的唯一 ID、生命周期证明、跨进程 ID，也不证明没有重用；`identity_scope=python_weak_object_only` 是强制限制。捕获内部 `contiguous()` 输出、底层 allocator 重用、NPU stream 完成、graph 节点/重放和 worker queue 等均不在本草稿的可证明范围。

两类普通记录包括 `embed_begin` 和 `embed_end`，结束只记 returned/raised，不记录异常文本。原方法的原始返回值和异常由 wrapper 透传；普通元数据/日志异常由观察器吞掉，不更改模型调用。异步 process-control 异常如 KeyboardInterrupt/SystemExit 保持正常进程控制，不伪装成成功。metadata 获取包括 `untyped_storage()`，可能本身分配、改变 Python wrapper 生命周期或在特定后端产生未知影响；此代码不主动调用设备同步/额外算子，但不能据此保证无同步、无图影响或低扰动。

文件固定在同目录 `engram-metadata.jsonl`，通过 `WIO_ENGRAM_META_PATH` 和 `WIO_DIAG_DIR` 指定，若不匹配或无绝对路径则不观察。多个配合写入进程以非阻塞 `flock` 同步文件大小，在新文件上实行合计 1 MiB 上限，预留 cap marker；每进程状态锁不阻塞，弱身份表最多 512 项。日志冲突、失败、未知 gate、截断、ID 淘汰和 cap 用计数/`state_incomplete` 表示；锁冲突、进程被强杀、磁盘短写或到达上限可能使最终 stats 和 marker 丢失，截断行可能损坏，缺少这些标记不证明完整。其他不遵守锁的写入者或已超限旧文件不在保证范围；必须使用新的专用文件。

## 测试与复核

`python3 test_engram_metadata.py` 是 fake Tensor/Storage 的 CPU 模拟，不载入 torch_npu、不启动 NPU：默认关闭、gate、原样返回和异常、元数据/日志异常、dtype/device index、无敏感值、匿名 ID、512 项淘汰、四个进程并发共享 16 KiB 测试 cap 通过。测试 cap 缩小以模拟同一上限机制；不能视作 1 MiB 的真实服务负载验证。Mac 和 169 stage Python 都通过；真实 torch Tensor 弱引用行为、所有调用路径、持久化、性能、模型输出、底层异步操作及原恢复流程都尚未验证。

源码 SHA-256 `05a2a59811bf049851ebad10e726542c9e4b4c6b940d996f4398b211947c5fd4`；测试 SHA-256 `8758d0624bd573f6b7c1a0bbecac2ccb70a7ab554da5bf7da793dd9f995772ae`。169 stage 的独立备份在 `.../second-binding-run/engram-metadata-draft-v2/`。如文件改动，应重新计算哈希并重新审查，不覆盖既有 capture。
