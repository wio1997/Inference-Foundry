# GLM-RUN-0218 — INVALID

Run218试同V2/K2/Graph3n配置加ASCEND_LAUNCH_BLOCKING=1定位原异步越界；被原生VllmConfig guard拒绝ACLGraph与blocking不兼容，FAILED/INVALID，权重未加载、0新worker/0推理、D0NPU0/D1native16全部计数及public217保留。未绕guard。217CANNplog/实际SFADCP源码快照留server：blocktable展开3*1128=3384与范围吻合，req_indices/slotmapping paddedcount可能不一致为可证伪假说，尚非rootcause。

Audit /data/tiankuan/wio/Inference-Foundry/glm5-3/records/points/GLM-OPT-0002/jobs/AUDIT-RUN218-CONFIG-20261005T1050Z/reduction.json SHA256 7869697f2f430ef9223e0560b1732a905de8d6c53c7c07457eeb3689f0f9c785.
