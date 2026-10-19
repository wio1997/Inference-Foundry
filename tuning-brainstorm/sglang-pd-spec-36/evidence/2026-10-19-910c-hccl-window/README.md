# 910C原生诊断：HCCL通信窗口不足

所有NPU计算均在910C44执行，Mac仅控制和整理证据。完整原模型PD＋MTP启动仍未通过，整体E0，accepted_baseline=null。

之前把失败概括成tiling过于粗略。回查已保存的R046日志，完整Prefill的CamMoeDispatchNormal在实际maxBs=1时明确要求213MB，默认HCCL_BUFFSIZE只有200MB；早先审查漏看了这一条stdout。[原始产物定位、SHA与逐字错误](notes/L039-hccl-window-prior-evidence.json)。旧Run与旧结论记录保留，本检查点追加更正。

R048/R049确认实际CANN9.1.0 B243与DeepEP1.0.0+00db6cd6.cann.9.1.0.b243匹配，包内有ascend910_93实现。文件存在不能证明算子可执行，run/plog没有提供更具体错误。随后采用局部分发检查，不重复加载764GB模型。

| 局部支持性对照 | 唯一有效环境变化 | 原生结果 |
| --- | --- | --- |
| R050 / A | HCCL_BUFFSIZE未设置，实际默认200MB | 每个rank的16rank HCCL精确求和通过；全部16rank在MoeLowLatencyDispatchV2 tiling失败。算子明确打印此maxBs256形状需1649MB，实际200MB；没有执行到数据golden |
| R051 / B | HCCL_BUFFSIZE=2048MB | 16rank均返回并同步，已越过旧tiling；检查器3D形状假设失败，记invalid，数据正确性未判定 |
| R052 / corrected checker | 与B同一算子/输入/环境，修正检查器布局和清理 | 更正检查器后16rank、256个本地专家分组全部counts/完整BF16行golden通过，共16384行201326592字节；显式group销毁16rank完成，随后进程退出超时，整体仍FAIL |

两组其余参数及脚本相同：16rank，128token/rank，hidden6144，256专家，topk8，BF16，不量化，max_dispatch_tokens_per_rank256，eager、同步分发。形状中的hidden/expert/topk来自已封存原模型config SHA。每个rank先初始化NPU与HCCL，再导入DeepEP；这是有意构造的局部支持性诊断，不声称重现完整模型所有初始化顺序、量化、graph或调度行为。

预先冻结的数据检查逐专家枚举源rank和token，对每个有效完整6144维BF16行计算SHA，再以独立CPU路由得到的多重集合核对数量与内容。payload前两列保存源rank/token。分发失败时不宣称数据正确；没有执行combine、真实MoE FFN、KV页或PD/MTP服务，亦没有性能结论。R051越过tiling后，检查器错误假设了3D布局，触发形状断言而未完成数据检查，记为invalid，不能当算子数据错误；R052保留独立golden，增加实际形状记录并接受文档列出的flat2D或expert3D布局。R052全部rank还记录了process_group_destroyed=true，说明显式销毁已经完成；后续进程退出仍超时，具体原因未知。早先Run摘要误说显式group未销毁，已另附纠正记录，保留原始结果。A在记录所有错误后torchrun清理未退出，达到240秒限制后只停止该精确CID，最终137、OOMKilled=false；失败容器与日志保留。[主控逐层判定](notes/L039-registration-review.json)。

初始怀疑自定义OPP在导入后才激活；A已经执行到vendor tiling的窗口校验，因此本轮没有继续更改OPP/LD路径，改为证据支持的单变量HCCL窗口对照。不能把窗口检查通过等同整个算子、完整启动或服务通过。

读原始证据：[A的16rank结果/加载库/错误与日志SHA](runs/L039-R050-deepep-registration/remote-result.json)、[B对应结果](runs/L039-R051-deepep-registration/remote-result.json)、[版本与OPP清单](runs/L039-R048-cann-tiling/stdout.json)、[安装策略源码及完整代表plog](runs/L039-R049-tiling-details/stdout.json)、[仅执行会话元数据的审计](notes/L039-registration-zcode-audit.json)。五个冻结命令各一次；CLI退出码与内层结果分别保留。

R052只覆盖平衡路由、每专家64个有效token；未覆盖空专家、倾斜、尾部、量化或combine。

下一步：先定位所有rank完成数据golden及显式group销毁后仍不退出的生命周期问题：用最小HCCL-only对照及精确进程/退出日志区分基础运行时与DeepEP；原模型PD+MTP的2048MB窗口候选已准备但尚未登记/派发，保持原模型/TP16/EP16/MTP/graph/长度。完整PD分离、MTP、原模型、并行与冻结长度继续保留。前序[完整启动tiling反例](../2026-10-19-910c-deepep-tiling/README.md)与[SDMA局部传输/原模型封存](../2026-10-19-910c-native-transfer-bootstrap/README.md)不改写。本次只更新M028支持性认识，其他35项机制条目不变。
