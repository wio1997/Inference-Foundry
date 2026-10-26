# 复现与范围

在新根/新环境/新Run核对spec哈希后运行，勿覆盖一次性目录/失败Run。所需CPU软件Python3.12.14、Torch2.8.0、numpy2.3.3、msgspec0.19.0；本轮不依赖Pillow/Transformers。分发64个固定公共源码闭包与2个合成M001输入，不分发软件/模型/tokenizer/权重或测试.bin文件。

File/LRU是完整原模块，实际文件操作仅在R/runs/本Run/owned-files；envflags由helper固定为任务参数，绝不使用默认/tmp/hicache或真实服务路径。原pagehash/nativeKV/后端constructor未运行，key是整个冻结输入ID的SHA256夹具，payload128int64仅用于字节身份。名称反例有共享目录/键/布局及不同语义模型的前提。

R003/R004共同295项比对和R005/R006共同38项比对分别保存，不把重复参考/候选累加。第二组隔离上层Scheduler bool丢失，已含controller-only补丁；不是从原生部署到最终服务的A/B。原msgspec结构与往返已执行，但HTTP/进程IPC未执行。清空失败为显式注入任务内os.remove异常。清空确认局部通过不修复8名称反例、旧rawtree或世代问题，也未添加idle/atomicbarrier。
