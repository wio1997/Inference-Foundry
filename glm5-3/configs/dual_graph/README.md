# 双完整实例Graph部署候选

以下两个文件是Run7实际P与Run6实际D入口的逐字节副本；运行参数为Graph FULL_DECODE_ONLY、MTP3、batch16384，TP/EP/DCP16，保留PD角色、工具/reasoning parser及其他native flags。文件头的历史注释来自原入口，以实际argv/manifest为准。源文件未自动替换现场legacy scripts。

Run8核验两端真实进程/PID/boot/startticks/raw proc argv后接管等待与E2E，不重复加载权重。初始化/功能/性能结果以Run8 state与manifest为准，不由配置存在就宣称可用或提速。标准公共环境继续引用现场deploy/scripts/pd_common_env.sh，manifest含其SHA及冻结副本；资源由唯一controller管理，未配置自动恢复旧队列。
