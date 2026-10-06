> 2026-10-06：本目录可编辑启动模板的默认权重已切为`/data/tiankuan/wio/GLM-5.3-w8a8`，服务名`glm-53`。下方Run/验证结果属于历史模型，不证明5.3正确性或性能；本目录不是已裁决的当前PD入口，按[PLAN](../../PLAN.md)选择实际配置，上传完成前不执行。

# 双完整实例Graph部署候选

以下两个文件是Run7实际P与Run6实际D入口的逐字节副本；运行参数为Graph FULL_DECODE_ONLY、MTP3、batch16384，TP/EP/DCP16，保留PD角色、工具/reasoning parser及其他native flags。文件头的历史注释来自原入口，以实际argv/manifest为准。源文件未自动替换现场legacy scripts。

Run8核验两端真实进程/PID/boot/startticks/raw proc argv后接管等待与E2E，不重复加载权重。初始化/功能/性能结果以Run8 state与manifest为准，不由配置存在就宣称可用或提速。标准公共环境继续引用现场deploy/scripts/pd_common_env.sh，manifest含其SHA及冻结副本；资源由唯一controller管理，未配置自动恢复旧队列。

2026-10-01: These batch16384/gmu0.92 entries are historical candidates. Run8 D local-prefill and Run9 P PD-prefill MTP/DCP transient OOM: REJECT those legal load/config combinations. Resident finite-valid entries are ../dual_resident_safe. Larger KV reservation or profiling success does not prove runtime fit.
