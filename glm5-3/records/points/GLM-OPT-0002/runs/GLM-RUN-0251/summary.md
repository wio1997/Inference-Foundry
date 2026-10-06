# Run251 — recovery, native full-byte correctness and matched comparison

H4 only; [Reset167](../../research/decode_path_20261006/RESET167.md). Actual Zcode Job GLM53-GATHER-RECOVER-20261006 handed off to the unique controller; controller completed recovercompare at2026-10-06T13:58:45Z, phase exit0. Initial Job running Result remains its original handoff evidence. `execution_summary.json` is a new Sol reduction of actual raw results, not a replacement for the Job or Run manifest.

Retired only owned failed Run250 D, preserving P Run249. Without loading weights, tested16 rank ×20 HCCL cases: rows1/3/16/17/32, BF16 width6144, contiguous/noncontiguous, ordinary and NaN/Inf/signed-zero data. All320 cases passed full padded byte equality, independent rank-coded expected output, input bytes, unpad and retained output lifetime. Stock strided inputs are normalized in its reference because exact ProcessGroupHCCL source rejects strided stock input. Candidate normalizes internally. Per-rank raw: `micro_checks/rankN.json`.

After micro pass, recovered D once and ran A1/B1/A2/B2 in the same16 resident worker processes. Mode changes require both P/D idle; no profile or config scan. Both B warm gates passed full-output/input bytes on all16 ranks, including actual padding NaNs (B1 rank7=22, B2 rank14/15=24 each). Warm correctness is excluded from timings. Candidate function AST equals executed shim function; original/candidate/shim SHA54e8ac…/26e52e…/f2efbe… verified by `reduce_compare.py`.

| Pair | Stock short TPOT median | Patched | Reduction | Complete PD wall |
|---|---:|---:|---:|---:|
|A1/B1|263.842719ms/token|246.899785ms/token|6.42%|1.693434→1.660282s,1.96%|
|A2/B2|260.413292ms/token|233.540234ms/token|10.32%|1.745766→1.566286s,10.28%|

Each short mode has two timed samples; identical2334 prompt,8 generated IDs and chunks1/2/2/2/1, all external KV hits. Complete dynamic arithmetic body is identical across modes (`thinking_token_budget=0`, max96); four natural EOS answers all`2`,38 prompt,3 generated IDs `[154842,17,154827]`, chunks1/1/1 and full external KV hits. No cap was reached. These are direct native P→KV→D helper/client E2E, not the unaccepted unified public API. Internal P helper token is not credited as D output.

All four stock short samples range255.281–265.772ms/token; patched233.258–250.625. Complete baseline phase drift is3.09%, patched5.66%; first complete pair's1.96% reduction is near noise. Therefore **positive scoped code signal, INCONCLUSIVE for formal repeatable complete Product Gain/PERF_KEEP**. Short8 output is not a complete service workload, formal SLA percentile or per-token ITL. Complete3-token arithmetic is narrow E2E correctness, not full workload/capacity acceptance. CurrentNone; stack unchanged. 本阶段没有新增代码级性能 KEEP。

Complete D-client wall1.306208→1.264291s (3.21%) and1.335283→1.186958s (11.11%). Pair2 P wall also decreases31.16ms despite unchanged P; total PD wall improvement is not all attributable to D code. Event-based TPOT/IDs/chunks were independently recomputed; full PD wall is the actual client timing. Independent final review accepted the limited conclusion and did not approve PERF_KEEP.

Final source restored to original54e8ac…, selector mode0, both health200/idle/all16 NPU owners checked independently at14:04Z. P root1916718 unchanged, D fresh root2532689; inspect `final_site.json` before future operations, never reuse PIDs blindly. Resident D keeps the safe selector in memory at stock mode. Controller completed and exited. No new device profile, parameter scan, kernel or large workload.

Detailed causal attribution, actual SFA/DCP and EP dependencies: [CRITICAL_PATH](../../research/decode_path_20261006/CRITICAL_PATH.md). Largest localized region is eager within-model host supply; exact maximum removable profOFF budget and separate Scheduler/Executor/ModelRunner gaps remain unknown. The6080 list-gather output copies are an identified redundant implementation path, not the whole residual latency.
