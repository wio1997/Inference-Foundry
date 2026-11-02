"""DeepSeek V4.1 model modules on the DeepInfer side.

Files fall into four roles. Keep the role in mind when changing one, and put a new file
in the role it belongs to rather than next to the file it resembles.

1. **Parameter containers** — `containers.py`, `linear.py`, `norm.py`, `rope.py`,
   `mla_attn.py`, `moe.py`, `hc_blocks.py`, `engram_parameters.py`, `heads.py`.
   Allocate parameters with the names, shapes and dtypes of their HF mapping targets.
   They declare no forward.
2. **Runtime adapters and forward** — `attn.py`, `compress.py`, `index_attn.py`,
   `mlp.py`, `hc.py`, `engram.py`, `utils.py`. Wrap a container, convert it to the
   runtime layout, and own "how the model computes". `utils.py` holds the shared
   container-to-runtime conversions. These files may allocate a *converted*
   runtime-layout parameter (`mlp.py` repacks expert weights, `attn.py` merges the
   WV/O projection), but never an HF-shaped one.
3. **Tensor operators** — `mhc_ops.py`, `mhc_modeling.py`, `yoco.py`, plus the shared
   `hc_lazy.py` data type. No parameters and no forward; they operate on tensors only.
4. **Config-derived topology, metadata and assembly** — `cross_layer.py`,
   `engram_metadata.py`, `builder.py`. Derive and validate structure from the config,
   then build the parameter tree.

Container to runtime pairing:

| Parameter container | Runtime implementation |
|---|---|
| `mla_attn.py`: MLA + compressed KV + sparse indexer | `attn.py` (`V4Attention` / `DFlashAttention`), `compress.py`, `index_attn.py` |
| `hc_blocks.py`: folded mHC block and head | `hc.py` (`HCLayerMixin` / `HCHead`), `mhc_ops.py`, `mhc_modeling.py` |
| `moe.py`: routed experts, shared expert, gate | `mlp.py` (`DeepseekMoE`) |
| `engram_parameters.py`: Engram tables and projections | `engram.py` (`Engram` / `EngramRuntime`), `engram_metadata.py` |
| `heads.py`, `norm.py`, `rope.py`, `linear.py`, `containers.py` | inlined in the forward implementations above |

Container conventions:

- allocate parameters only; the paired runtime implementation owns the forward;
- preserve the parameter names, shapes and dtypes of the HF mapping targets, along with
  the loading and runtime metadata: `.sf`, `n_experts`, `einsum_info`, `is_engram`,
  `row_start`.

`hc_lazy.py` stays dependency-free on purpose: operator and decoder code needs to type,
inspect and rebuild the cross-layer HC bundle without importing the HC forward path and
its fused kernel stack.
"""
