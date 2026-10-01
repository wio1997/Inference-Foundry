# GLM-RUN-0024

INVALID native config startup before weights/EngineCore/API inference: inherited recompute_scheduler_enable=true requires PD kv_consumer and is illegal in local DP mode (kv_role=None). Both rank logs and process absence verified; controller signalled by exact boot/start/PID and terminal cancellation observed. Frozen candidate sources/spec retained, zero model requests/outputs. Exact old P21/D22 and taskPDproxy previously stopped and their logs preserved. This is not EP32 memory/communication rejection. New run will use lawful local DP scheduler configuration; no old queue.
