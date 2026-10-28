# 复现固定测试

1. 核对SOURCE-ARTIFACTS中的完整源码归档SHA和complete-source-SHA256；权重仍是指定ModelSlim W8A8，镜像ID见runtime。归档不含权重、镜像或编译缓存。
2. 原始正式控制器及基准驱动在raw/dspark-v15/scripts；两个AISBench配置在raw/dspark-v15/aisbench/measure1.py、measure2.py。配置引用服务器workload/frozen；409源码和冻结工作负载已逐文件SHA核对。
3. 当前服务已运行，不应再次启动。原始控制器自动跑基准且拒绝复用status路径；恢复控制器raw/current-controller.py使用新日志/control目录且禁用自动基准。再次部署必须选新版本目录、保留源码路径/manifest，先确认设备和8172空闲，不能触碰外部任务。
4. AISBench固定commit7bcb6440da11bef020943b2af9ed7c1ac8362306，num-warmups0、num-prompts8、max-num-workers1，batch_size8。驱动每轮先做规定prefix种子，再C8/32输出热身，然后跑一次正式8请求；不要将热身计入吞吐。
5. 正式指标以AISBench Duration和Total Generated Tokens计算，和服务器8192输出/262144输入/235904缓存计数交叉核对；完整1024token文本与raw/frozen-reference逐字节比较。16/16全等才可比较性能。第二轮使用独立phase输入与prefix种子。

源码依赖固定容器中CANN9.2/torch2.10/torch_npu2.10及custom OPP。CANN的PYTHONPATH不能被源码路径覆盖；实际启动命令见runtime与current-controller.py。CPU亲和性采用640CPU/16rank，每rank40CPU块内36主线程+1ACL+1release、2IRQ预留；只改变本项目线程。

这些步骤供复现，不是本次交付又执行的测试；现有正式两轮结果已通过独立审计。
