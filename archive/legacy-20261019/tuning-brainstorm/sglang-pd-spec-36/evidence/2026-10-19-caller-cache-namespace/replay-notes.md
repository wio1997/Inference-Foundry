# 复现边界

原命令引用本机解释器、只读CPU/frontend包根、独立Pillow overlay和一次性输出目录。复现时按receipt/package版本重建依赖，核对固定源码/脚本/两个合成输入哈希，在新根/新环境/新Run冻结；不要覆盖失败Run或改旧spec。manifest列出5个既有CPU输入候选，本轮仅使用和分发其中c2-short-0、c3-long-0两个合成文件，不能推断其他输入已在本轮执行。

输入闭包包括失败脚本与53个公共源码文件，不含软件/模型二进制。helper提取列明的完整原函数/方法体、完整Core/FULL等类；不是全服务模块导入。CPU外部夹具、类加载与真实调用范围见报告。候选仅替换完整Req构造函数中的namespace逻辑，继承其余原方法；没有生产应用。R003在执行前INVALID，没有ZCode/frozen收据。R006/R007同一覆盖不重复相加。

raw metadata indices是Torch CPU整数数组，未分配设备KV。旧缓存迁移反例假定旧entry保留；服务升级/清空/cachetiers未知。新缓存264PASS不能替代旧状态/streaming/native正确性。
