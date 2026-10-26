# R247 B indexer-optimized whole-model profile (source only)

Status: **source-only candidate delta, not dispatched, no model/NPU/HTTP/SSH
action taken by this unit's preparation.** The actual R246 PASS/idle8 signoff
is not present, so the launcher refuses.

## What is integrated

R247 reuses the already-run R238 B execution path and the original C5 image,
then adds exactly two performance points that have paired-profile evidence:

1. **R246 runtime row-stride score** — four declarations in
   `flash_kpool_score.py` change from `tl.constexpr` to runtime scalars:
   `TABLE_SR` and `table_stride_r` are added to `do_not_specialize`.
2. **R210 query-parallel indexer** — the generic `query_parallel_indexer`
   splits independent prefill query rows inside the current native attention
   TP group (`parallel.attn_tp_group`, `attn_tp_rank`, `attn_tp_size`), scores
   local rows, and performs one `int32` `all_gather_into_tensor` back in
   original row order. The R210 caller differs from the R238 caller only at
   the row-split call site (two hunks).

No R233, Graph, MTP, KDA, FP32-scale, topology or cache-grid change is
integrated. The original data sources are read-only; R238/R246/R210 are not
modified.

The R247 B overlay has **47 files**: 44 original R238 files unchanged, two
replaced, one added. It is **not byte-identical to the original 46-file
overlay**, and no document in this unit claims otherwise.

## Exact source delta

| File | Original | R247 |
| --- | --- | --- |
| `sglang/srt/hardware_backend/npu/ops/flash_kpool_score.py` | `44a6f449…` | `b1821eb6…` |
| `sglang/srt/hardware_backend/npu/ops/flash_kpool_indexer_npu.py` | `facd47f9…` | `f0c0aaaa…` |
| `sglang/srt/hardware_backend/npu/ops/flash_indexer_query_parallel.py` | (new) | `b10d0b38…` |

The full unified diff, including the R210 new module and the R246 import-alias
composition note, is in [`PERFORMANCE_DIFF.md`](PERFORMANCE_DIFF.md).

## Scope

- Nodes: only 46 (P) and 47 (D). Nodes 14/15 are never contacted.
- Two cold prefixes (`C32-prefix000`, `C32-prefix001`), 74,400 tokens each,
  one output token each.
- Four generation POSTs total (two prefixes × Decode then Prefill), two cold
  pairs, zero formal requests.
- No warm-up, no warm32, no dummy/smoke, no retry, no flush.
- Three capture windows (`early`, `mid`, `late`) × two real P forwards.
- Only TP0 is sampled (TP16 DP1). Leader-view only.
- One 1,200 s work budget for B. Cleanup keeps an independent safety bound.
- The original R201 configuration is unchanged: TP16/DP1/EP1, eager
  (`--disable-cuda-graph`, `--disable-overlap-schedule`), MTP off, BF16
  weights/KV, FP32 SSM, `modelslim`, chunked prefill 4096, mamba
  `extra_buffer`.
- PD control stays TCP; the data plane is typed SDMA
  (`ASCEND_MF_TRANSFER_PROTOCOL=sdma`, asserted in preflight and
  transfer-engine initialization). There is no RDMA fallback.

## Profiling fixes

- Native info files are matched as `profiler_info*.json`, including
  `profiler_info_0.json`, in the host index, download list and analyzer.
- `WAIT`, `NOTIFY` and `SQE` are classified as runtime control and excluded
  from `core_compute_union_us`, category unions and kernel busy unions; they
  are reported in a separate control bucket.
- CPU enqueue/instruction spans are never attributed to a device forward.
  `cpu_enqueue_is_device_forward` is false and per-forward device attribution
  remains unavailable without native flow correlation.
- Device time uses the native hardware CSV inside the outer
  `r234_capture_window_*` marker, which is started after a device drain and
  stopped only after `torch.npu.synchronize()`.
- A missing CANN export product, a truncated host index, a collection timeout
  or an analyzer that is not `COMPUTED` makes the run FAIL. There is no empty
  PASS path.
- The read-only parser is the main-control
  `checks/analyze-recovered-native-profile.py` adapted for R247 paths; its raw
  file handling is not rewritten.

## R246 actual PASS/idle8 gate

R247 must not start on a source-only R246 claim. Before staging, the launcher
and gateway perform all of the following:

1. Require `r246-gate.json` signed by main control
   (`kind=R246_ACTUAL_PASS_IDLE8_SIGNOFF`, `actual=true`, `passed=true`,
   `r246_unit=r246`, `idle8.node46=true`, `idle8.node47=true`).
2. Read the actual node-side receipt
   `/data/wio/inference-foundry/glm53-flash/runs/20261019-r246-dynamic-rowstride-tp16-score-component/host-receipt.json`
   on node 46 and require its SHA256 to equal
   `r246_host_receipt_sha256` in the signoff.
3. Require the receipt itself to show `unit=r246`, `passed=true`,
   `rank_passed=16`, `stage_a.passed=true`,
   `owned_container_verified=true`, `post_inspect="false 0"`,
   `post_npu` with 8× `No running processes found`, no
   `bounded_wait_timeout`, the exact R246 container and the original C5
   image.
4. Run `npu-smi info` live on nodes 46 and 47 and require 8×
   `No running processes found` on each.
5. Require `r247_review_sha256` in the signoff to match the frozen R247
   review, and the authorization to allow exactly `python3 dispatch.py` once.

The controller repeats the receipt and live-idle check before it creates its
own once-lock or touches a node. The signoff file is an external mutable
authorization, not part of the reviewed source closure.

## Single entrance

After main control freezes the review and writes the R246 gate:

```bash
cd runs/20261019-r247-b-indexer-optimized-profile
python3 dispatch.py
```

`dispatch.py` creates `dispatch.once.lock`, starts `launcher.py` detached once;
`launcher.py` creates `once.lock`, verifies the frozen review, then stages the
full closure to the gateway. The gateway re-checks the R246 receipt and live
idle8, creates the independent controller root
`/data/wio/inference-foundry/glm53-flash/controllers/20261019-r247-b-indexer-optimized-profile`,
and starts `controller/controller.py` once. The run root is
`/data/wio/inference-foundry/glm53-flash/runs/20261019-r247-b-indexer-optimized-profile`.

To inspect the frozen payload SHA without any remote action:

```bash
python3 launcher.py --manifest
```

The current freeze digest and all file SHAs are recorded in
`freeze-input.json` and `results/FROZEN_IDENTITY.json`.

## Cleanup order

The R238 writer-first cleanup module is reused unchanged:

1. Signal P (node 46) to stop.
2. Wait for its container to be stopped and 8 NPUs idle.
3. Publish the writer-stop permit.
4. Only then signal D (node 47) to stop.
5. If the writer stop/idle is unproven, retain the D destinations; never
   destroy them on an unproven writer.

Cleanup is not clamped by the 1,200 s work budget.

## Source layout

- `controller/` — B-only controller, host probe, launch command, export policy,
  freeze policy, R246 gate schema.
- `stage/B/` — 55-file stage: R201 argv/config, cleanup fence, preflight and
  the 47-file candidate overlay.
- `observers/` — R238 SGLang observer, unchanged.
- `client/` — B-only two-prefix client, exact four POSTs.
- `analysis/` — R247 path adaptation of the main-control recovered-profile
  analyzer, plus `profiler_info*.json` and control-task handling.
- `checks/` — CPU static, source-delta, rendered-launcher, R246 gate and
  profiling/export tests.
- `PERFORMANCE_DIFF.md` — exact three-file diff.
