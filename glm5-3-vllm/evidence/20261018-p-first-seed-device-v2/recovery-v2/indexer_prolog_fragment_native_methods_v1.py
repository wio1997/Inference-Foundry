from __future__ import annotations
import torch,torch_npu
from vllm_ascend.ops.triton.rope import rope_forward_triton_siso
HAS_TRITON=True
class AscendW8A8DynamicLinearMethod:pass
class AscendW8A8MXFP8DynamicLinearMethod:pass
class Prolog:
    def _sfa_preprocess_prolog_v3(
        self,
        hidden_states: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        cos: torch.Tensor,
        sin: torch.Tensor,
        slot_mapping: torch.Tensor,
        *,
        attn_metadata: M | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
    ]:
        assert slot_mapping.numel() == hidden_states.shape[0], (
            "SFA Prolog V3 requires one cache index per input token, "
            f"got token_x={hidden_states.shape[0]} and cache_index={slot_mapping.numel()}."
        )
        assert self.q_a_layernorm is not None, "q_a_layernorm must be initialized for PROLOG_V3"
        assert self.kv_a_layernorm is not None, "kv_a_layernorm must be initialized for PROLOG_V3"

        qt = self._quant_type
        use_c8 = self.enable_sparse_sfa_c8

        common: dict[str, Any] = dict(
            weight_dq=self.weight_dq,
            weight_uq_qr=self.weight_uq_qr,
            weight_uk=self.W_UK_T,
            weight_dkv_kr=self.weight_dkv_kr,
            rmsnorm_gamma_cq=self.q_a_layernorm.weight.data,
            rmsnorm_gamma_ckv=self.kv_a_layernorm.weight.data,
            rmsnorm_epsilon_cq=self.q_a_layernorm.variance_epsilon,
            rmsnorm_epsilon_ckv=self.kv_a_layernorm.variance_epsilon,
            query_norm_flag=self.has_indexer,
            qc_qr_scale=1.0,
            kc_scale=1.0,
            cache_mode="PA_BSND",
            query_quant_mode=0,
        )
        kv_cache_nope = kv_cache[0]
        extra_kwargs: dict[str, Any] = {}
        if use_c8:
            extra_kwargs.update(
                ckvkr_repo_mode=1,
                quant_scale_repo_mode=1,
                tile_size=self.sfa_qsfa_tile_size,
                k_nope_clip_alpha=self.sfa_qsfa_k_nope_clip_alpha,
            )
            kr_cache = self.sfa_qsfa_kr_cache_dummy
        else:
            kr_cache = kv_cache[1]
        rope_cos_ = cos.view(cos.shape[0], cos.shape[-1])
        rope_sin_ = sin.view(sin.shape[0], sin.shape[-1])
        # The caller forwards the per-step cached int64 slots, so the .to()
        # below is a no-op in production; kept for direct-call safety.
        cache_index = slot_mapping.view(-1).to(torch.int64)

        if qt is not None:
            if qt is AscendW8A8MXFP8DynamicLinearMethod:
                token_x, ds = torch_npu.npu_dynamic_mx_quant(hidden_states, dst_type=torch.float8_e4m3fn)
                branch = dict(
                    dequant_scale_x=ds.reshape(token_x.shape[0], -1).view(torch.float8_e8m0fnu),
                    dequant_scale_w_dq=self.weight_dq_scale.view(torch.float8_e8m0fnu),
                    dequant_scale_w_uq_qr=self.weight_uq_qr_scale.view(torch.float8_e8m0fnu),
                    dequant_scale_w_dkv_kr=self.weight_dkv_kr_scale.view(torch.float8_e8m0fnu),
                    weight_quant_mode=3,
                )
            else:
                assert qt is AscendW8A8DynamicLinearMethod, (
                    f"PROLOG_V3 only supports W8A8Dynamic or W8A8MXFP8 quant, "
                    f"got {qt}. Did _resolve_preprocess_type allow a new quant type?"
                )
                token_x, dequant_x = torch_npu.npu_dynamic_quant(hidden_states.contiguous())
                branch = dict(
                    dequant_scale_x=dequant_x.view(-1, 1),
                    dequant_scale_w_dq=self.dequant_scale_w_dq,
                    dequant_scale_w_uq_qr=self.dequant_scale_w_uq_qr,
                    dequant_scale_w_dkv_kr=self.dequant_scale_w_dkv_kr,
                    weight_quant_mode=2,
                )
        else:
            token_x = hidden_states
            branch = dict(
                dequant_scale_x=None,
                dequant_scale_w_dq=None,
                dequant_scale_w_uq_qr=None,
                dequant_scale_w_dkv_kr=None,
                weight_quant_mode=0,
            )

        ql_nope, q_pe, _, q_c, q_c_scale = torch_npu.npu_mla_prolog_v3(
            token_x=token_x,
            rope_sin=rope_sin_,
            rope_cos=rope_cos_,
            kv_cache=kv_cache_nope,
            kr_cache=kr_cache,
            cache_index=cache_index,
            kv_cache_quant_mode=3 if use_c8 else 0,
            **common,
            **branch,
            **extra_kwargs,
        )
        num_h = self.local_num_heads
        ql_nope = ql_nope.view(-1, num_h, self.kv_lora_rank)
        q_pe = q_pe.view(-1, num_h, self.qk_rope_head_dim)

        if self.has_indexer:
            if q_c is None:
                raise RuntimeError("npu_mla_prolog_v3 did not return query_norm for SFA indexer.")
            q_c = q_c.view(-1, self.q_lora_rank)
            if q_c_scale is not None:
                if qt is AscendW8A8MXFP8DynamicLinearMethod:
                    q_c_scale = q_c_scale.view(-1, q_c_scale.shape[-1])
                    q_c = (q_c, q_c_scale)
                else:
                    q_c = (q_c, q_c_scale.view(-1))
        elif self._quant_type is None:
            q_c = None

        return hidden_states, ql_nope, q_pe, q_c

class Indexer:
    def _quantize_li_tensor(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply Hadamard transform and quantize for LI C8 or C4 path."""
        assert self.q_hadamard is not None
        x = x @ self.q_hadamard
        shape_ori = x.shape
        x = x.view(-1, self.head_dim)
        if self.enable_sparse_li_c4:
            x, scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=self.c4_k_op_dtype)
            scale = scale.view(*shape_ori[:-1], *scale.shape[-2:])
            return x, scale.view(self.c4_k_scale_cache_dtype)
        x, scale = torch_npu.npu_dynamic_quant(x, dst_type=self.c8_k_cache_dtype)
        return x, scale.to(self.c8_k_scale_cache_dtype)

    def forward_k(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """k path: compute ``k_li`` (and ``k_li_scale`` when LI C8 is
        enabled) from the hidden-states stage SFA hands in (raw states on
        fused preprocess paths, prepared states on native paths). SFA then
        persists the result through ``write_cache`` before the top-k stage
        runs, since the top-k kernel reads the freshly written cache.

        Also returns the non-K tail of the ``wk_weights_proj`` GEMM output
        (``indexer_weights``) so the top-k stage can reuse it instead of
        re-running the same GEMM on the same hidden states."""
        assert self.wk_weights_proj is not None
        assert self.k_norm is not None

        kw, _ = self.wk_weights_proj(hidden_states)
        k_li = kw[:, : self.head_dim]
        indexer_weights = kw[:, self.head_dim :]
        k_li = self.k_norm(k_li).unsqueeze(1)
        k_li = k_li.view(-1, 1, self.head_dim)

        if HAS_TRITON:
            cos = cos.view(-1, self.qk_rope_head_dim)
            sin = sin.view(-1, self.qk_rope_head_dim)
            k_li = rope_forward_triton_siso(
                k_li, cos, sin, rope_dim=self.qk_rope_head_dim, is_neox_style=self.is_rope_neox_style
            )
        else:
            k_li_pe, k_li_nope = torch.split(
                k_li, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1
            )

            cos = cos.view(-1, 1, 1, self.qk_rope_head_dim)
            sin = sin.view(-1, 1, 1, self.qk_rope_head_dim)

            k_li_pe = k_li_pe.unsqueeze(2)
            k_li_pe = torch_npu.npu_rotary_mul(k_li_pe, cos, sin)
            k_li_pe = k_li_pe.squeeze(2)

            k_li = torch.cat([k_li_pe, k_li_nope], dim=-1)  # [b*s,128]

        if self.enable_sparse_li_quant:
            k_li, k_li_scale = self._quantize_li_tensor(k_li)
            if self.enable_sparse_li_c8:
                k_li_scale = k_li_scale.unsqueeze(-1)
        else:
            k_li_scale = None

        return k_li, k_li_scale, indexer_weights
