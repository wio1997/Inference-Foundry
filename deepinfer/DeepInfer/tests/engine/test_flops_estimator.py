"""Compare the native estimator with per-layer, per-query FLOPs accounting.

Like main's test_v4_flops_limiter, use real model metadata without loading
weights. The reference enumerates visible keys rather than using prefix sums
or grouping layers by their FLOPs coefficients.
"""

import copy
import random
from collections.abc import Iterator

import numpy as np
import pytest
import torch  # noqa: F401  Load torch's shared libraries before deepinfer._C.

from deepinfer._C.engine import FlopsEstimator  # pyright: ignore[reportMissingModuleSource]
from deepinfer.config import ModelConfig
from deepinfer.model_executor.model_metadata_utils import ModelMetadata
from examples._config import DEFAULT_MODEL_NAME, resolve_model_path

pytestmark = [pytest.mark.ci, pytest.mark.ci_npu]

MODEL_NAMES = [DEFAULT_MODEL_NAME]
KEEP_TOKENS = 140


@pytest.fixture(scope="module", params=MODEL_NAMES)
def metadata(request: pytest.FixtureRequest) -> ModelMetadata:
    model = resolve_model_path(request.param)
    config = ModelConfig(model=model, tokenizer=model, trust_remote_code=True, load_format="auto", seed=0, max_num_seqs=512)
    return ModelMetadata(config.model_cfg.gpt)


def _reference_flops(
    m: ModelMetadata, q: int, k: int, *, mode: str, is_prompt: bool, full_len: int, swa_valid_start: int
) -> tuple[int, int]:
    """Count logical GEMMs and visible keys; coefficients include multiply-add."""
    full_len = k if full_len == -1 else full_len
    positions = np.arange(k - q + 1, k + 1, dtype=np.int64)  # One-based query positions.
    cutoff = full_len - KEEP_TOKENS
    h, c = m.n_embed, m.hc_mult
    proj_coef = 2 if m.fp8_attn else 4
    mlp_coef = 2 if m.fp8_mlp else 4
    shared = m.shared_layer_metadata
    attn = non_attn = 0
    final_rows = q

    def hc_pre(rows: int) -> int:
        # Projection [rows, C*H] @ [C*H, C*(C+2)] and pre mix.
        return 8 * rows * (c * h * c * (c + 2) + c * h)

    for layer in range(m.n_layer):
        consumer = is_prompt and mode != "off" and layer >= m.n_prefill_layer
        queries = positions
        if consumer and mode == "compact":
            queries = positions[positions > cutoff]
            if len(queries) == 0:
                queries = positions[-1:]  # Fully clipped chunks still execute one row.
        rows = final_rows = len(queries)
        head_dim = m.q_head_dim
        use_swa = m.use_swa_list[layer]
        packed = m.has_packed_kv_list[layer]
        ratio = m.kv_compress_ratio_list[layer]
        topk = m.topk_tokens[layer]
        local_index = topk > 0 and shared.is_index_source_list[layer]
        idx_h = m.index_attn_num_heads[layer] if local_index else 0
        idx_d = m.index_attn_head_dim[layer] if local_index else 0

        # Q A/B, SWA KV, Wv, output projection and indexer head weights.
        attn += proj_coef * rows * h * (m.q_lora_rank + (head_dim if use_swa else 0))
        attn += proj_coef * rows * m.q_lora_rank * (m.n_head * head_dim + idx_h * idx_d)
        attn += 2 * rows * m.n_head * head_dim * m.o_head_dim
        attn += proj_coef * rows * m.n_wv_group * m.o_head_dim * h
        attn += 4 * rows * h * idx_h
        if packed and shared.is_kv_source_list[layer]:
            attn += 4 * q * h * head_dim * (1 if ratio == 1 else 2)
        if local_index and shared.is_index_k_source_list[layer]:
            # Count completed compression blocks directly; placeholders also execute.
            projected_rows = max(1, int(np.count_nonzero(positions % ratio == 0))) if is_prompt else q
            attn += 4 * projected_rows * head_dim * idx_d

        if use_swa and m.window_size > 0:
            valid_start = max(swa_valid_start, cutoff) if consumer else swa_valid_start
            keys = np.clip(queries - valid_start, 0, m.window_size)
            attn += 4 * 2 * m.n_head * head_dim * int(keys.sum())
        if packed:
            keys = queries // ratio
            core_keys = np.minimum(keys, topk) if topk > 0 else keys
            attn += 4 * 2 * m.n_head * head_dim * int(core_keys.sum())
            if local_index:
                blocks = shared.index_candidate_blocks[layer]
                if blocks > 0:
                    block_size = shared.index_candidate_block_size[layer]
                    keys = np.minimum((keys + block_size - 1) // block_size, blocks) * block_size
                attn += (1 if m.indexer_use_fp4 else 2) * idx_h * idx_d * int(keys.sum())

        non_attn += mlp_coef * 3 * rows * h * m.moe_intermediate_size * (m.moe_top_k + m.n_shared_experts)
        non_attn += 4 * rows * h * m.n_routed_experts
        if m.hc_enabled:
            # Each layer has attention and MLP HC pre/post contractions.
            non_attn += 2 * (hc_pre(rows) + 8 * rows * c * (c + 1) * h)
            if consumer and mode == "compact" and layer == m.n_prefill_layer:
                non_attn += hc_pre(q - rows)  # Boundary pre runs before compaction.
        if layer in m.engram_cfg.get("engram_layer_ids", []):
            embed = sum(m.engram_cfg["n_embed_per_ngram"])
            non_attn += (2 if m.engram_fp8_proj else 4) * rows * embed * h * (c + 1)
            non_attn += 8 * rows * c * h

    if m.hc_enabled:
        non_attn += 8 * final_rows * c * h  # Final head mix.
        if m.hc_shifted:
            non_attn -= 8 * q * c * h  # The first shifted pre has no mix.
        else:
            non_attn += 8 * final_rows * c * h * c
    logits = 4 * (1 if is_prompt else q) * h * m.vocab_size
    return attn, attn + non_attn + logits


def _cases(m: ModelMetadata, is_prompt: bool) -> Iterator[tuple[int, int, int, int]]:
    # Both sides of SWA, compression, topk and partial candidate-block boundaries.
    boundaries = {1, KEEP_TOKENS, m.window_size, 1 << 20}
    for ratio, topk, blocks, size in zip(
        m.kv_compress_ratio_list,
        m.topk_tokens,
        m.shared_layer_metadata.index_candidate_blocks,
        m.shared_layer_metadata.index_candidate_block_size,
        strict=True,
    ):
        boundaries.update((ratio, ratio * topk, ratio * size, ratio * blocks * size))
    for boundary in sorted(boundaries):
        for k in (boundary - 1, boundary, boundary + 1):
            if not 1 <= k <= 1 << 20:
                continue
            q = min(k, 257 if is_prompt else 6)
            for full_len in (-1, k + KEEP_TOKENS, k + KEEP_TOKENS - q // 2):
                yield q, k, full_len, 0
                yield q, k, full_len, k - q
    if is_prompt:
        yield 32768, 917504, 917504, 0
        yield 1_000_000, 1 << 20, 1 << 20, 0
    rng = random.Random(0)
    for _ in range(128):
        q = rng.randint(1, 512 if is_prompt else 16)
        k = rng.randint(q, 1 << 20)
        yield q, k, k + rng.randint(0, 2 * KEEP_TOKENS), rng.randint(0, k - q)


@pytest.mark.parametrize("mode", ["off", "mask", "compact"])
@pytest.mark.parametrize("is_prompt", [True, False], ids=["prefill", "decode"])
def test_flops_estimator(metadata: ModelMetadata, mode: str, is_prompt: bool) -> None:
    estimator = FlopsEstimator(metadata, yoco_mode=mode, keep_tokens=KEEP_TOKENS)
    for q, k, full_len, valid_start in _cases(metadata, is_prompt):
        kwargs = dict(is_prompt=is_prompt, full_len=full_len, swa_valid_start=valid_start)
        expected = _reference_flops(metadata, q, k, mode=mode, **kwargs)
        actual = estimator.calc_flops(q, k, **kwargs)
        assert actual == expected, (mode, q, k, kwargs, actual, expected)


@pytest.mark.parametrize("shifted", [False, True])
def test_non_attention_variants(metadata: ModelMetadata, shifted: bool) -> None:
    # Exercise BF16 projections and both HC layouts even if checkpoints use FP8.
    m = copy.deepcopy(metadata)
    m.fp8_attn = m.fp8_mlp = m.engram_fp8_proj = False
    m.hc_enabled = True
    m.hc_shifted = shifted
    kwargs = dict(is_prompt=True, full_len=1024, swa_valid_start=768)
    actual = FlopsEstimator(m, yoco_mode="compact", keep_tokens=KEEP_TOKENS).calc_flops(256, 1024, **kwargs)
    assert actual == _reference_flops(m, 256, 1024, mode="compact", **kwargs)


def test_mask_consumer_engram(metadata: ModelMetadata) -> None:
    # Unlike compact, mask mode permits Engram in consumer layers.
    m = copy.deepcopy(metadata)
    assert m.n_prefill_layer < m.n_layer
    m.engram_cfg["engram_layer_ids"] = [m.n_prefill_layer]
    kwargs = dict(is_prompt=True, full_len=1024, swa_valid_start=0)
    actual = FlopsEstimator(m, yoco_mode="mask", keep_tokens=KEEP_TOKENS).calc_flops(256, 1024, **kwargs)
    assert actual == _reference_flops(m, 256, 1024, mode="mask", **kwargs)
