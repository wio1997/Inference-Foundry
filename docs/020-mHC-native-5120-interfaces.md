# mHC 实际 H5120 接口门槛

固定 SGLang A2 镜像 `7da963869ee54984dfeea2445befdf181710ef846448466db2361247759904ec`、910B1、DeepSeek-V4.1 W4A8 配置的 hidden=5120 / HC=4。完整 D 图实际经过 RoPE 后触发 TileLang split/Sinkhorn 缺失，随后检查模型原生 mix_stats/combine/post 三个接口。

G357 证明当前 CANN 原生 `npu_hc_post` 拒绝 H5120，仅支持 4096/7168。候选对 NPU/HC4/H5120 精确分支复用同一固定镜像已有 Triton `mhc_post_split_h`，输入 contiguous，其他维度保持原路径；不更换镜像、驱动或 CANN。

G361 的首轮 post 比较失败来自夹具：combine 的变化回放修改了输入，post 的 CPU oracle 仍期待原始输入。该轮保留为无效夹具记录，不判定内核错误。G364 在每个接口前恢复全部输入，保持候选数学实现不变。

G364 实际 8 rank 通过 88 用例、88 图、264 次变化回放。combine/post 覆盖 rows=1/16/96/8192，实际模型 mix_stats 覆盖 1/16/96。BF16 combine/post 对 CPU FP64 后转 BF16，rtol=0.008、atol=0.003；FP32 mix_stats rtol=1e-4、atol=1e-5。最大绝对误差分别为 0.00390625、0.0078125、1.5497207641601562e-6；相对容差参与判据，不能仅凭最大绝对误差与 atol 比较。没有与不可用 CANN H5120 路径作位级等价声明。组件结束后控制器退出且 8 卡进程清空。

G373 将此前验证的 RoPE、split/Sinkhorn 与本次 post 分支集成到完整 P/D 候选。P/D 启动句柄已记录，但尚未完成完整模型质量、跨机传输、target/draft 图及四档 AISBench 验收。组件结果不能当作端到端性能收益。

- [组件汇总和精度范围](../evidence/mHC_interfaces_G371/acceptance.json)
- [原始硬件约束](../evidence/mHC_interfaces_G371/G357_hardware_failure/server.log)
- [无效夹具原始记录](../evidence/mHC_interfaces_G371/G361_invalid_fixture/server.log)
- [夹具修正说明](../evidence/mHC_interfaces_G371/fixture_correction.json)
- [实际 rank0 原始结果](../evidence/mHC_interfaces_G371/G364_accepted/evidence/rank0.json)
- [固定源 SHA 的 post 安装范围](../code/full_model_candidates/sg_pd_g373/post_install.py)
- [P 启动句柄](../evidence/full_model_G373/2203/start.json)
- [D 启动句柄](../evidence/full_model_G373/2204/start.json)

离线检查：`python3 scripts/validate_mhc_interfaces_G371.py`，只核对留存的结果和源码，不运行模型或 NPU。
