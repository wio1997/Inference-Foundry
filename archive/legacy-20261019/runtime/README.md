# DeepSeek Extreme fixed runtime

This directory is the product execution path for the frozen DeepSeek V4 Flash
W4A8 / 8x910B3 / DP1xTP8 / DSpark7 configuration.  vLLM and vLLM-Ascend may
bootstrap weights, caches, process groups, and existing operators during early
bring-up, but their scheduler, request objects, metadata trees, dispatch, and
per-step host bookkeeping are not part of the runtime ABI.

## Decode v0

The first ABI is fixed at c12 and greedy decoding:

1. Twelve stable request slots own block-table rows and persistent device state.
2. Each target verification cycle materializes a fixed `[12, 8]` block:
   one last accepted token plus seven DSpark draft tokens.
3. Target forward mutates target KV/recurrent state and produces target logits,
   final hidden states, and the auxiliary hidden states used by DSpark.
4. Verification writes fixed-width accepted tokens plus `num_sampled` and
   `num_rejected` device tensors.
5. State advance updates positions, computed-token counts, last-token state,
   output counters, and the next target slot mapping without request objects.
6. The three-layer DSpark operator writes the next `[12, 7]` draft block and
   its KV state.

`fixed_decode.py` owns the state and cycle ordering.  `oracle_shadow.py` is a
temporary bridge: it predicts the next target input from standalone state and
compares it with the oracle's next cycle.  It must not become a production
dependency.

`target_adapter.py` is the runtime-owned direct target entry. It binds loaded
weights, the fixed Ascend attention context, and physical KV tensors once, then
accepts only preallocated token/position buffers per cycle. The current smoke
uses a synthetic binding; the next integration run substitutes the live
DeepSeek binding and fingerprints the touched DSA cache groups.

`extreme_decode.py` is the product control-plane boundary. It owns the fixed
stage order and never imports vLLM. The bootstrap layer is allowed to transfer
loaded operator callables, communication groups and cache tensors once, but a
ModelRunner, SchedulerOutput or request object cannot enter `step()`.

For the frozen greedy workload, `greedy_accept.py` replaces the general
rejection-sampling stack with a fixed operation: TP-global target argmax,
leading draft-prefix acceptance, mismatch recovery or all-accepted bonus, and
accepted-count publication.  Its natural future fusion region includes state
advance and next-target-input materialization; the TP argmax collective is the
main boundary to evaluate for overlap or integration.

## Current promotion boundary

- The fixed ABI has executed eight causally connected cycles with deterministic
  operators.
- Real-weight next-cycle target input and greedy acceptance parity have passed
  on all eight ranks.
- The fixed target adapter ABI passes CPU and Ascend NPU structure tests; live
  target weights, attention context and physical cache tensors are the active
  binding gate.
- KV and recurrent-state fingerprints are required before claiming real-model
  multi-cycle correctness.

No performance claim is valid until the real model runs continuously through
this entry and matches oracle outputs/state.  Once correct, the runtime DAG—not
the original module boundaries—will define graph, replay, persistent, overlap,
fusion, and SuperKernel candidates.
