# Fixed target execution contract v0

Scope: DeepSeek V4 Flash W4A8, 8×Ascend 910B3, DP1×TP8, c12, DSpark7,
temperature 0.

## Setup-time bindings

- Loaded target weights and TP/EP process groups.
- The fixed c12 Ascend forward context and per-layer attention metadata.
- Physical DSA compressor, indexer and SWA KV tensors.
- Model forward and LM-head callables.

These bindings may initially be borrowed from the oracle process, but they are
installed once. No scheduler, request, dynamic batch selection or metadata
builder call belongs inside `FixedTargetAdapter.execute`.

## Per-cycle inputs

- `target_input_ids`: fixed `[96]`, int32, stable address.
- `target_positions`: fixed `[96]`, int64, stable address.
- `target_logits_indices`: fixed `[96]`, int64.
- Attention state already present in fixed device buffers: c12 sequence lengths,
  query starts, block tables and slot mappings.

## Per-cycle outputs and mutations

- Target hidden states used by sampling and DSpark.
- Target logits for the 96 verification positions.
- Auxiliary target hidden states required by the proposer, if configured.
- In-place writes to every target DSA compressor/indexer/SWA cache touched by
  those 96 positions.

DeepSeek V4 Flash in this checkout is not on the generic hybrid Mamba/GDN state
postprocess path. The state parity gate therefore fingerprints the physical DSA
cache groups rather than inventing a separate recurrent-state object.

The current oracle stores these physical tensors in its `kv_caches` list and
binds the same tensors into each layer of `static_forward_context`. The
bootstrap migration point is therefore after cache allocation/binding, not
inside `execute_model`: transfer the model callable, TP/EP groups, bound layer
operators and cache tensor tree to Extreme Runtime, then discard the generic
runner control objects before entering the continuous loop.

## Current implementation boundary

`runtime/target_adapter.py` enforces fixed token counts and stable input buffer
addresses, invokes the bound target forward directly, and computes logits. The
next integration step binds the live Ascend context/KV tensors and replaces the
normal c12 target invocation, then checks hidden/logit/token and touched-cache
fingerprints before moving attention metadata ownership into `runtime/`.
