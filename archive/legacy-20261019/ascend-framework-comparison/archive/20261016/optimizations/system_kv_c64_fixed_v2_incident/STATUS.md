# 当前 C64 状态：未通过

64 条独立前缀预热已完成，649.233 秒，全部从正式吞吐排除，64×1743 个 L3 对象均存在。

正式 256 条中已保存 246 条成功响应，均实际 80000 输入、600 输出；243 条保留 74368 cached/storage tokens，3 条缓存明细缺失，保持缺失。另10条停滞，缺失ID记录在 INCIDENT_STATUS.json。观察到超过20分钟无新增成功记录。不得从246个幸存者计算整轮合格性能。

P DP0 等待10，全部P运行数0；D DP10/14分别等待4/6，全部D运行数0。经本地ZCode DeepSeek执行一次已审查的非阻塞线程快照：P TP0/15在批次MLP all_gather等待，TP1仍在HiCache ready-count all_reduce等待；5个D线程均在MLP all_gather。提示进程之间同步轮次/顺序不一致，尚需完整分组快照和可控复现确认最早分歧，不能把一张异步快照当作完整因果证明。

网关SMG1.13已实际部署，记录D健康超时；不能据此认定网关造成堵塞。8次快照不发模型请求、不暂停进程，不改服务。旧768/640容量、copy/pinned等历史问题不要重跑。

初版栈采集脚本因嵌套字符串空字节，在远程执行前的ast.parse失败，未采样；V2改用bytes([0])并独立校验嵌套源码AST，8个准确PID全部成功。原始失败记录保留。

当前任务没有重启、回滚或重复启动。C64原broker82415/session22843及锁保持，真实状态须下次复核；后续优先确认和修复同步活性，再进行合格基线，暂不叠加性能改动。所有服务/测试/采样动作继续经本地ZCode DeepSeek精确审查启动。

官方同类线索（不能直接套用不同cache类/硬件）：https://github.com/sgl-project/sglang/issues/30760 、https://github.com/sgl-project/sglang/issues/28429 、https://github.com/sgl-project/sglang/issues/42465 。
