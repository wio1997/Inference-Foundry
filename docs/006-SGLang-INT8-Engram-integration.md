# SGLang INT8 Engram：原生接口接入与图执行验证

固定 SGLang A2 daily、CANN9.1、torch2.10/torch_npu post6、八张实际910B1上验证。G60/G61使用小型索引表fixture，隐藏维度5120、hc4、group32；这不是完整检查点、完整模型PD或性能验收。

适配只补齐原生EngramEmbedding的INT8/FP32 Host UVA存储、分片索引加载回调、owned-row lookup及旋转gate。SGLang原生Embedding.forward、_lookup和EngramHasher保持原方法/源码；fixture以真实HCCL八卡all-reduce接入原生TP汇总接口。生产RuntimeContext发布和完整量化模型仍待验收。

逐rank两层各4种token数（0、1、16、80），共64个lookup数值用例，均与独立CPU float64参考得到的BF16逐位匹配。覆盖多chunk、非本rank所有行、越界置零、INT8与E8M0权重索引、两个参数回调。每rank捕获两个TP lookup图和一个原生Engram forward图，共24个设备图捕获、72次改变输入后的重放。

原生Engram forward的gate在hidden5120下与独立完整旋转矩阵FP64参考对照，rtol=0.008/atol=0.008；保留image token行不变，并以错误的identity rotation作负对照，必须被拒绝。重放同时改变hidden与hash ids。此容差只用于该fixture，不能推出完整模型质量。

G60完成lookup/TP图，但完整forward图在torch.copysign处触发CANN9.1同步Host拷贝而失败。G61将该算子替换为FP32的int32符号位掩码和设备where，保留负零符号语义；没有关闭图执行。修正后八rank全部通过，控制器已退出，容器停止、OOMKilled=false、八卡进程清空，Host buffer及参数alias释放。

- [G60失败现场](../evidence/components/sg_engram_int8_g60/evidence/rank4.json)
- [G61逐rank成功记录](../evidence/components/sg_engram_int8_g61/evidence/rank0.json)
- [G61适配源码](../evidence/components/sg_engram_int8_g61/adapter.py)
- [独立参考与图测试](../evidence/components/sg_engram_int8_g61/probe.py)
- [G61控制器清理状态](../evidence/components/sg_engram_int8_g61/state.json)

下一步以完整95文件SHA校验通过的W4A8权重，完成两节点TP8/EP8 PD、严格目标/草稿加载、DSPARK5预测、目标与草稿图、前缀缓存和异步调度，再做四档完整预热与首轮冷热。组件通过不替代这些条件。
