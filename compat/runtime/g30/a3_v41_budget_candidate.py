"""Explicit target-only sizing candidate; no installation or global patching."""
from sglang.srt.model_executor.pool_configurator import DSV4PoolConfigurator


class A3V41PoolConfigurator(DSV4PoolConfigurator):
    """Price actual A3 BF16/INT8 pools before the byte-budget solver runs.

    draft_swa_layers is a required, verified layout fact, including zero for a
    non-speculative target. It must match actual BF16 draft pools sharing the
    target's paged SWA capacity. This candidate does not infer it from ratios.
    """

    _draft_kv_priced_exactly = True

    def __init__(self, kvc, *, draft_swa_layers):
        if not isinstance(draft_swa_layers, int) or draft_swa_layers < 0:
            raise ValueError("Supply the verified number of BF16 draft SWA layers")
        if kvc.kv_cache_dtype_str != "bfloat16" or getattr(kvc, "is_draft_worker", False):
            raise ValueError("This configurator prices the BF16 A3 target only")
        self.a3_draft_swa_layers = draft_swa_layers
        super().__init__(kvc)
        if set(self.stage_compress_ratios) - {0, 1, 2}:
            raise ValueError("A3 V4.1 budget supports only ratios {0,1,2}")
        if self._unified or self.encoder_replay:
            raise ValueError("This budget requires the paged A3 SWA layout")
        if self.is_speculative and not kvc.spec_algorithm.is_dspark():
            raise ValueError("Other draft layouts need their own exact byte contract")
        if not self.is_speculative and draft_swa_layers:
            raise ValueError("A non-speculative target cannot price draft SWA layers")
        if self.attn_head_dim != 512 or self.indexer_head_dim != 128:
            raise ValueError("Unexpected target head dimensions")
        self.kv_bytes = self.attn_head_dim * 2
        self.low_ratio_index_bytes = self.indexer_head_dim + 2
        self._spec_infl = 1.0
        self.bytes_per_swa_token = self.kv_bytes * (
            self.num_swa_layers + draft_swa_layers
        )
        self.bytes_per_full_token = self._get_bytes_per_full_token()

    def _get_low_ratio_kv_padding_bytes(self, full_tokens):
        # Each owner allocates one reserved compressed page of width P/ratio.
        compressed_reserved = sum(
            self.page_size // ratio * (self.kv_bytes + self.low_ratio_index_bytes)
            * len(layers)
            for ratio, layers in self.stage_owner_layers.items()
        )
        # Target/draft SWA also allocate a reserved physical page zero.
        swa_reserved = self.page_size * self.bytes_per_swa_token
        return compressed_reserved + swa_reserved

    def actual_payload_bytes(self, *, full_tokens, swa_tokens, max_running_requests):
        """Known pool tensors only; does not include allocator/workspace overhead."""
        compressed_coeff = sum(
            (self.kv_bytes + self.low_ratio_index_bytes) / ratio * len(layers)
            for ratio, layers in self.stage_owner_layers.items()
        )
        return int(
            full_tokens * compressed_coeff
            + swa_tokens * self.bytes_per_swa_token
            + self._get_low_ratio_kv_padding_bytes(full_tokens)
            + self._get_c2_state_fixed_bytes(max_running_requests)
        )
