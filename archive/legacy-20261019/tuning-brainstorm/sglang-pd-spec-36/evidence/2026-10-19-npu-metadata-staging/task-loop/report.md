# L007 报告：Ascend metadata 与 staging

固定安装源码 ce0a8fea，整体服务 E0。R001/R002 完成源代码与观察配置采集，R001 schema假设导致 whitelist 为空已保留，R002 通过 flat resolved_dict 源码修正；观察 disaggregation_mode 为字符串 null，不是本任务 PD 资源。

R003 本地 Torch2.8 CPU 与 R004 目标 image 的 Torch2.10.0+cpu：768 gate、180 alias 用例及5 staging用例相同。共享int32反事实48组修改源，连续draft别名互相覆盖；borrowed view在注入caller改写后不保留原快照。实际NPU metadata只抽取int赋值和offset block，compact=None；没有执行kernel、graph或NPU copy。

Ascend继承needs_cpu_seq_lens=True，wrapper未声明也保守True；CPU镜像不能直接删。in-tree NPU pin能力声明False，显式pin=True子候选条件不适用；实际platform对象未观察。通用staging复用仍需NPU完成事件和所有权验证。

R004任务CPU容器runtime runc、无网络/device/bind、只读root、1核/2GiB，退出后按ID与labels删除。已有服务identity/start/restart count相同。远端与本地时钟不同，不跨时钟推算耗时。外层CLI与冻结controller/inner exit均0，所有冻结输入不变；配置model alias deepseek-flash，实际后台release未知。

没有性能数据、生产改动或PD基线。descriptor前轮N指page数，observed page128时32768 tokens约256页；N32768是函数压力用例，真实PD N/R分布待采。继续M021 graph边界及后续机制，不宣称整体完成。
