# 原生0.32全rank覆盖与实际DSPARK调用链

epoch727使用用户指定v0.32.0rc1的arm64不可变镜像d5492ecf，vLLM源码5df63a7203cdb9328d824a27b682e77c7b9928be、vLLM-Ascend源码24487f4f48ee99cf45d86d1c504e645549b8c1ef，910B1/aarch64、CANN9.2.1、torch_npu2.10post7。原始profile cohort为16条32768输入/1024输出、C16，SHA见018/metadata。P/D原服务均受控退休，原始trace保留；本节没有重发模型请求。

## 全部D rank实际数据库复核

独立无NPU设备的同镜像CPU SDK路径已完成其余七rank。[父级只读DB复核](../evidence/native_allrank_G909/all8_D_parent_review.json)逐rank检查真实device编号、913条scheduler注释、21450 scheduled query、相同规范化CPU序列SHA、1812条aclmdlRIExecuteAsync、有效时戳、device/model TASK分组和compute表关联。原始raw逐文件SHA未变；每个worker结果的SHA也被固定。之前D0完整Text/Db/独立CPU三路计数对照继续保留，不将局部pilot当其余rank通过。

七个rank各有2358816条compute，rank4为2358812；其原始TASK也少7行（6881093，相对普通非零rank的6881100）。实际数据保留，不补行、不强求各rank相等；当前证据证明每个原始捕获的完整解析与实际设备/CPU覆盖，不能据此宣称所有rank执行算子完全相同或解释这几行差异。rank0额外WAIT等TASK较多，同样不能当更多计算量或利用率。

结合此前P八rank的实际219768条compute、198条CPU注释和524272 scheduled query，原生诊断的P8+D8原始设备与CPU覆盖复核完成。它不是两框架逻辑工作归一化地图完成，更不代表优化收益；SGLang仍须最终相同cohort的P/D采集与有效工作匹配。

## Prefill 计算与 HCCL 设备时间

复核同一轮 G804 的八卡只读数据库，每卡计算区间并集为 28.121–28.993 秒，HCCL 的正字节 `Memcpy` 和 `Reduce_Inline` 区间并集为 3.478–3.490 秒。八张卡上，这两类区间的交集均为 0 ns。`Notify_Wait` 单列，每卡区间并集为 2.400–3.245 秒。每卡全任务并集和任务数与封存的覆盖回执一致。这些读数使用每张卡内部的时间轴，关键路径及 P/D 跨节点 KV 传输需要独立时序。 [八卡证据](../evidence/profile_capture_G804/derived/P8_compute_HCCL_coverage_G1490.json) [复算工具](../code/profile_capture_G804/audit_P_device_overlap_g1490.py)

## 实际DSPARK源码链，区分target与draft

[Ascend V2工厂](../code/native_DSPARK_G909/vllm-ascend/vllm_ascend/worker/v2/spec_decode/__init__.py)在use_dspark分支构造AscendDSparkSpeculator。[实际Ascend类](../code/native_DSPARK_G909/vllm-ascend/vllm_ascend/worker/v2/spec_decode/dspark/speculator.py)继承[上游DSparkSpeculator](../code/native_DSPARK_G909/vllm/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py)，继承DFlash propose路径。这里没有使用target_dependent_ar的多步自回归实现来解释DSPARK。

实际DSpark _generate_draft先一次parallel backbone forward，再在其hidden states上顺序Markov sampling，非五次自回归backbone。sample_from_anchor缺省true，num_query_per_req等于num_speculative_steps=5；target另验证root+5，共6 query/request。源文件原始SHA与镜像内实际来源保留在[883](../evidence/native_allrank_G909/G883_native_DSPARK_source_actual_source_receipt.json)、[890](../evidence/native_allrank_G909/G890_native_DSPARK_base_source_actual_source_receipt.json)及provenance。

V2上游[model runner](../code/native_DSPARK_G909/vllm/vllm/v1/worker/gpu/model_runner.py)的execute_model运行target；sample_tokens随后调用speculator.propose，DFlash propose按query_cudagraph_manager选择一次FULL draft图。外部CPU注释仅包execute_model，不包后面的sample_tokens/propose。结合020的直接connection证据，范围内ModelID37–41支持target家族归因，范围外26–30符合该独立draft源码链；该 draft 路径没有独立的 scheduler 步标记。后续以原始 CANN 调用交替顺序和设备图容量核对了 906 组配对。 [逐次重放核对](../evidence/profile_capture_G804/derived/D0_ordered_target_draft_replays_G1461.json)

[图管理器](../code/native_DSPARK_G909/vllm/vllm/v1/worker/gpu/cudagraph_utils.py)按decode_query_len将capture token数round_up。020中逻辑30 query映射实际36首维与此机制一致；1920个多余shape lane、8.951%比例仍不是已执行FLOPs或墙钟收益。尚未改变bucket或部署任何该方向优化。
