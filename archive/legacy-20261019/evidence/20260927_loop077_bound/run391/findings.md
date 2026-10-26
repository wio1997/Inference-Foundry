# Run391 Astra High read-only Target HCCL ABI audit

Frozen current Target trace Run378 has one embedding ReduceScatter, 43 identical six-operation layer blocks, and six tail AllGather tasks: 1 + 43*6 + 6 = 265. The count and dtype signatures match the installed CANN 9.1 HCCL ABI and vLLM-Ascend source callsites. This promotes **115,107,840 B input and 145,342,464 B output per rank-cycle** from Run378 conditional count interpretation to a source/ABI-consistent **current schedule tensor inventory**. Reported HCCL count×dtype sum25,651,200 B uses mixed count definitions and is not a communication total.

## ABI

Installed `/usr/local/Ascend/cann-9.1.0/aarch64-linux/include/hccl/hccl.h` lines55–67 defines RS `recvCount` as output elements; lines105–117 defines AG `sendCount` as input elements; lines207–223 defines A2A `sendCount` as elements **per peer**. For TP8 and `C=count*dtype_bytes`, RS input/output=`8C/C`, AG=`C/8C`, A2A=`8C/8C`. vLLM distributed base communicator lines199–221,232–260 matches AG and RS tensor dimensions. The installed tree lacks ProcessGroupHCCL.cpp, so native binary per-call attribution was not independently observed.

## Ordered source map

- ordinal0 embedding BF16 RS `[96,4096] -> [12,4096]`: `ops/vocab_parallel_embedding.py:229-248`, `ops/register_custom_ops.py:72-90`.
- layer `l=0..42`: `1+6l` DSA hidden BF16 AG `[12,4096]->[96,4096]`, actual `attention/context_parallel/dsa_cp.py:1435-1444`; `2+6l` token/head BF16 A2A local `[12,64,512]`, peer chunk `[12,8,512]`, `dsa_cp.py:1645-1675`; `3+6l` wo_b BF16 RS `[96,4096]->[12,4096]`, `dsa_cp.py:1343-1391` and `ops/linear_op.py:506-519`; `4+6l` MoE hidden BF16 AG, `5+6l` router FP32 AG `[12,256]->[96,256]`, `6+6l` MoE BF16 RS, `ops/fused_moe/prepare_finalize.py:355-387,503-524`.
- ordinal259 MTP residual BF16 AG `[12,16384]->[96,16384]`, `models/deepseek_v4.py:1142-1161`; 260-263 hidden and three DSpark aux BF16 AG `[12,4096]->[96,4096]`, `models/deepseek_v4.py:1136-1140`, `worker/model_runner_v1.py:4483-4520`; 264 logits BF16 AG `[96,16160]->[96,129280]`, `ops/vocab_parallel_embedding.py:331-340`.

Installed model config gives hidden4096, 43 layers, 256 experts, 64 heads and head_dim512. Actual DSA CP A2A is `dsa_cp.py:1667-1674`; `dsa_v1.py` static exchange is not the current path. Counts: RS87, AG135, A2A43. A read-only script checked all43 ordered blocks and tail sizes exactly.

## Bound decision

This is current-schedule API tensor volume, **not** mathematical compulsory bytes, physical link traffic or a latency floor. Router logits are derived from gathered hidden, embedding RS input has masked zeros, and DSA layout, weight split, tail hidden/logits consumer placement can change under a valid architecture. Dense ring `7C` per-rank traffic would be a conditional algorithm model only. Native collective algorithm, topology path, compression, protocol and mixed resource contention remain unmeasured. Current vLLM-Ascend HEAD `36589852a1eb8f5e842ad920f7c80ebdf1376ee9`; `worker/model_runner_v1.py` has local modifications, so do not assert exact source snapshot identity with old Run246. This audit is read-only, no service or NPU mutation.
