# 假设记录

## H001 - ACTIVE

At real all-active DP1xTP8 layer4, moving identical per-token FP32 replicated gate after existing hidden AllGather removes the logits AllGather without changing routing and may shorten complete all-rank MoE critical path despite 176.16MFLOP/rank extra gate work
