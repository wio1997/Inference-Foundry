# 原生0.32：完整D0设备覆盖与独立CPU解析

P八rank的实际设备/CPU覆盖已验。D0现在由SDK Text和Db两条路径对同一完整原始窗口解析完成，[父级对照](../evidence/offline_profile_G856/parent_D0_full_format_parity.json)确认8081676条TASK、2358816条compute task、913个实际scheduler CPU注释和21450个scheduled query tokens。两种格式的device/model分组计数相同，原始逐文件SHA保持不变；这是完整窗口的聚合/分组覆盖一致，不宣称每个DB字段逐位相同。

[D0实际设备检查](../evidence/offline_profile_G856/D0_Text_device_review.json)另见1812次aclmdlRIExecuteAsync，等于906个非零scheduler步骤的两倍；存在26–30和37–41两个正ModelID家族。尚未逐家族关联源码与CPU/device correlation，不能直接给家族贴target/draft标签，也不能把图padding与真实query混为一谈。busy union包含WAIT，不能解释为计算利用率。

首个4CPU/600秒D0解析超时的失败继续保留在018；新的Text路径用16CPU完成，Db路径也完整完成。之后移到独立CPU容器处理剩余rank，固定同一0.32镜像、CPU槽24核/256GiB、无NPU设备。SDK可导入且NPU未初始化，但[首个rank1](../evidence/offline_profile_G856/CPU_rank1_frontend_guard_failure.json)失败于CANN能力检查，不能标解析成功。

现场发现同SHA的原生msprof前端在无设备环境里连--help也会超时；[无设备启动环境对照](../evidence/offline_profile_G856/frontend_no_device_timeout.json)和[仅管理设备对照](../evidence/offline_profile_G856/frontend_manager_only_timeout.json)均以20秒/124结束，停止继续增加设备。相同安装的[Python后台入口](../evidence/offline_profile_G856/same_image_CANN_sources_and_Python_help.json)则正常返回帮助。

独立CPU作业采用[限定CLI适配器](../code/offline_profile_G856/sitecustomize.py)：校验原生二进制与Python入口SHA，能力查询复用[同一镜像原生D0进程实际取得的帮助输出](../evidence/offline_profile_G856/actual_native_frontend_help_cache.json)；只把SDK已知DB导出/通信分析命令转给同一安装的Python后台，未知参数拒绝。SDK路径检查、数据处理和完整CPU/NPU任务导出仍走原SDK/CANN实现，未替换或缩减原始数据。独立PYTHONPATH仅作用该CPU作业及子进程，不改变推理服务。

[完整D0副本试验](../evidence/offline_profile_G856/CPU_full_D0_backend_pilot_actual_processing.json)已实际调用Python export db并产生host SQLite，设备节点为空，NPU未初始化；最终8081676 TASK/2358816 compute/913 CPU/21450 query/1812图调用及完整model分组的原生对照仍待验收。原失败rank1副本保留，不重发任何NPU采集。

当前[权威状态](../metadata/profile_epoch_G727.json)仍将D其余7rank、两框架逻辑/物理差异地图、SGLang完整质量与四档首轮矩阵、正确性加端到端优化收益列为未完成。完整Goal继续进行。
