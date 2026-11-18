# NPU 复数频率索引的逐位兼容

G396完整D图越过此前MoE路由故障位置后，在压缩KV路径 `DeepseekV4AttnBackend._low_ratio_write_group` 的 `layer.freqs_cis[group_pos]` 报错：aclnnIndex不支持complex64。完整D日志、终态和原始归档SHA保留，16份target/draft worker权重ledger均passed且无missing。D端控制器退出并清空8卡；这轮没有完成PD或性能测量。

G415候选对complex64表先取FP32实数视图，完成相同索引后恢复复数视图；没有重新生成角度或频率，也没有改变RoPE和量化数学。对固定镜像三个源文件作SHA门槛，在5个原生函数中的6处频率索引作定点替换，保持函数对象及其已导入别名。安装仅在V4.1/NPU后端工厂内调用，其他设备的helper仍走原索引。

G423父级验收：8张910B1、224用例、192图、576次变化回放，含32个空行eager用例。263296×32 complex64频率表覆盖连续/非连续视图、1D/2D索引、0/1/16/96/8192行，频率结果逐位一致；原生RoPE/FP4链在128/512维上的CPU比较最大误差为0。链路参考为未改变的原生CPU函数；NPU压缩KV正E4M3 scale桥接已在此前组件单独检查，本轮不构成独立完整模型质量验证。实际执行源码与归档候选逐字节一致，结束后8卡进程清空。

G425将该已验收索引路径集成到同一权重、量化、EP、DSPARK、完整D图的全模型候选，保留之前的路由和NPU finalizer。实际完整前向、压缩状态跨机传输、奇数长度功能、质量与四档性能仍须验收。

- [G396原始错误](../evidence/full_model_G396/2204/server.log)
- [原始归档和16份权重ledger索引](../evidence/full_model_G396/2204/assessment.json)
- [组件验收与精度范围](../evidence/frequency_gather_G423/acceptance.json)
- [rank0完整原始结果](../evidence/frequency_gather_G423/evidence/rank0.json)
- [8卡释放凭证](../evidence/frequency_gather_G423/state.json)
- [逐位索引helper](../code/full_model_candidates/sg_pd_g425/freqs_compat.py)
- [六处定点安装及原生SHA](../code/full_model_candidates/sg_pd_g425/freqs_install.py)
