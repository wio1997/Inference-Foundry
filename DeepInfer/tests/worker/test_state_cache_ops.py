"""Test script for state cache operations in GPUCacheInterface.

This script tests the select_state_slice_to_store and load_state_from_store_to_prefill_cache
methods in the GPUCacheInterface class.

Note: Tests are split into two suites based on DEEPINFER_PREFILL_STORE_EXTRA_STATE value,
as the initialization logic differs between True and False.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Generator
from typing import Any

import pytest
import torch

from deepinfer import AttrDict, envs
from deepinfer.model_config_loader import load_model_config
from deepinfer.model_executor.layers.tilelang_ops.cache_common import V41_FORMAT
from deepinfer.model_executor.model_metadata_utils import ModelMetadata
from deepinfer.worker.cache_engine import (
    KVSTORE_GRAIN,
    GPUCacheInterface,
    KVCache,
    KVCacheManager,
    clear_compress_block_size_cache,
    get_num_state_slots,
)
from examples._config import DEFAULT_MODEL_NAME, resolve_model_path

pytestmark = [pytest.mark.ci, pytest.mark.ci_npu]

# Use the real C1/C2 configuration without APE/overlap; only the config is read, no weights are loaded.
BASE_MODEL_NAME = DEFAULT_MODEL_NAME


@pytest.fixture(scope="module")
def raw_base_config() -> Any:
    """Load matching HF config.json once; each test receives an isolated deepcopy."""
    return load_model_config(resolve_model_path(BASE_MODEL_NAME)).gpt


@pytest.fixture
def model_config(raw_base_config: Any) -> Any:
    """A per-test deep copy of the config so rewrites cannot pollute other tests."""
    return _deepcopy_unalias_layers(raw_base_config)


def _deepcopy_unalias_layers(cfg: Any) -> Any:
    """Deep-copy the config and break the inter-layer aliasing of attn_module_list.

    When the checkpoint's attn_module_list is built as ``[A, B] * N``, layers of the same kind share one AttrDict;
    a whole-object deepcopy keeps that alias through memo, so per-layer rewrites would overwrite each other — each position must be deep-copied separately.
    """
    cfg = copy.deepcopy(cfg)
    attn_module_list = cfg.get("attn_module_list") or []
    for pos, attn in enumerate(attn_module_list):
        if attn is not None:
            attn_module_list[pos] = copy.deepcopy(attn)
    return cfg


def _hit_position(last_process_position: int) -> int:
    """kvstore hit positions: the processed position is rounded down to KVSTORE_GRAIN."""
    return last_process_position // KVSTORE_GRAIN * KVSTORE_GRAIN


def _plus_one_wrapped(state: torch.Tensor) -> torch.Tensor:
    """Add 1 byte-wise to uint8 (wrapping) to build state that differs from the original everywhere."""
    return ((state.to(torch.int32) + 1) % 256).to(torch.uint8)


def _stored_swa(state_flat: torch.Tensor) -> torch.Tensor:
    """Read the effective bytes of the dumped SWA.

    The stored SWA segment is viewed per layer and each layer's trailing 32B alignment pad is removed (pad contents are undefined and not compared).
    """
    swa = state_flat[: GPUCacheInterface.swa_store_total_bytes_inference_pad].view(
        GPUCacheInterface.swa_store_num_layer, GPUCacheInterface.swa_bytes_per_layer_inference_pad
    )[:, : GPUCacheInterface.swa_bytes_per_layer_inference]
    return swa


class TestStateCacheOperationsBase:
    """Base test class for state cache operations."""

    @pytest.fixture
    def setup_envs_base(self, monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
        """Base setup for environment variables."""
        # envs is a descriptor class; setattr replaces the class attribute. Note that when monkeypatch.setattr saves the old value,
        # getattr(envs, name) triggers EnvVar.__get__, which replaces the descriptor with its resolved value, so undo restores
        # the resolved value rather than the descriptor; this relies on the autouse _clean_env teardown in tests/conftest.py
        # calling rearm_all() after monkeypatch's undo to re-attach the descriptors (the autouse fixture is set up first
        # and torn down last, which guarantees the order). Copying this pattern in a directory without that conftest requires rearming yourself.
        monkeypatch.setattr(envs, "DEEPINFER_MAX_BACKTRACKING_TOKENS", 10)
        # The real model has YOCO consumer SWA, so disk restore must enable truncation.
        monkeypatch.setattr(envs, "DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE", "mask")
        monkeypatch.setattr(envs, "DEEPINFER_YOCO_PREFILL_KEEP_TOKENS", 128)
        # the compress block size lru_cache reads env inside the function body; clear the cache after changing env, otherwise stale results are hit
        clear_compress_block_size_cache()

        yield

        clear_compress_block_size_cache()

    @pytest.fixture
    def state_cache(self, setup_envs: None, model_config: Any) -> torch.Tensor:
        """A state cache with random contents; shapes come from _kvcache_layout.

        NOTE This explicitly depends on setup_envs defined by the subclass so _kvcache_layout computes the class fields only after the test env is active,
             instead of relying on parameter order in the test signature.
        """
        # block_size = -1 means only the state shape is computed and no kv block is allocated; 10 state sequences
        _, _, _, state_shape, _ = GPUCacheInterface._kvcache_layout(model_config, -1, 10)
        return torch.randint(0, 256, state_shape, device="cuda", dtype=torch.uint8)


class TestStateCacheOperationsWithExtraState(TestStateCacheOperationsBase):
    """Test suite for DEEPINFER_PREFILL_STORE_EXTRA_STATE = True."""

    @pytest.fixture
    def setup_envs(self, setup_envs_base: None, monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
        """Setup environment variables for testing with extra state."""
        monkeypatch.setattr(envs, "DEEPINFER_PREFILL_STORE_EXTRA_STATE", True)
        clear_compress_block_size_cache()

        yield

        clear_compress_block_size_cache()

    def test_select_state_slice_to_store_basic(self, setup_envs: None, state_cache: torch.Tensor) -> None:
        """Test basic functionality of select_state_slice_to_store with extra state."""
        state_tensor = torch.empty(GPUCacheInterface.state_size_to_disk, dtype=torch.uint8, device="cuda")

        result = GPUCacheInterface.select_state_slice_to_store(
            state_id=0,
            last_hit_position=_hit_position(500),
            state_cache=state_cache,
            state_tensor=state_tensor,
        )

        assert result is not None
        assert result.shape[0] == GPUCacheInterface.state_size_to_disk
        assert result is state_tensor  # must return the preallocated tensor as-is
        assert result.dtype == torch.uint8
        assert result.is_cuda

    def test_load_state_from_store_to_prefill_cache_basic(self, setup_envs: None, state_cache: torch.Tensor) -> None:
        """Test basic functionality of load_state_from_store_to_prefill_cache with extra state."""
        original_state_cache = state_cache.clone()
        state_tensor = torch.randint(0, 256, (GPUCacheInterface.state_size_to_disk,), dtype=torch.uint8, device="cuda")

        GPUCacheInterface.load_state_from_store_to_prefill_cache(
            state_id=0,
            last_hit_position=_hit_position(500),
            state_tensor=state_tensor,
            state_cache=state_cache,
        )

        # with extra state, load goes through the SWA ring-buffer reshuffle; this only verifies the cache was actually written,
        # content-level validation is covered by test_round_trip_consistency
        assert not torch.equal(state_cache, original_state_cache)

    def test_swa_only_state_round_trip(self, setup_envs: None, model_config: Any) -> None:
        """Only the stored SWA region participates in the disk round trip."""

        _, state_size_to_disk, _, state_shape, _ = GPUCacheInterface._kvcache_layout(model_config, -1, 2)
        assert GPUCacheInterface.swa_num_layer > 0
        assert state_size_to_disk > 0

        state_cache = torch.randint(0, 256, state_shape, dtype=torch.uint8, device="cuda")
        selected = GPUCacheInterface.select_state_slice_to_store(
            state_id=0,
            last_hit_position=KVSTORE_GRAIN,
            state_cache=state_cache,
        )
        assert selected is not None
        GPUCacheInterface.load_state_from_store_to_prefill_cache(
            state_id=0,
            last_hit_position=KVSTORE_GRAIN,
            state_tensor=selected,
            state_cache=state_cache,
        )

    @pytest.mark.parametrize("last_position", [500])
    def test_round_trip_consistency(self, setup_envs: None, state_cache: torch.Tensor, last_position: int) -> None:
        """Test that select and load operations are consistent (round-trip) with extra state."""
        state_id = 0
        last_hit_position = _hit_position(last_position)
        test_state_cache = state_cache.clone()

        # select -> rewrite -> load back into cache -> select again; the rewritten contents must come back
        selected_state = GPUCacheInterface.select_state_slice_to_store(
            state_id=state_id,
            last_hit_position=last_hit_position,
            state_cache=test_state_cache,
            state_tensor=None,
        )
        modified_state = _plus_one_wrapped(selected_state)
        GPUCacheInterface.load_state_from_store_to_prefill_cache(
            state_id=state_id, last_hit_position=last_hit_position, state_tensor=modified_state, state_cache=test_state_cache
        )
        reselected_state = GPUCacheInterface.select_state_slice_to_store(
            state_id=state_id,
            last_hit_position=last_hit_position,
            state_cache=test_state_cache,
            state_tensor=None,
        )

        reselected_swa = _stored_swa(reselected_state)
        modified_swa = _stored_swa(modified_state)
        assert torch.equal(reselected_swa, modified_swa)

    @pytest.mark.parametrize("last_position", [0, 128, 256, 384, 512, 640, 768, 896])
    def test_select_state_slice_with_different_positions(self, setup_envs: None, state_cache: torch.Tensor, last_position: int) -> None:
        """Test select_state_slice_to_store with different last_process_position values with extra state."""
        result = GPUCacheInterface.select_state_slice_to_store(
            state_id=0,
            last_hit_position=_hit_position(last_position),
            state_cache=state_cache,
            state_tensor=None,
        )

        assert result is not None
        assert result.shape[0] == GPUCacheInterface.state_size_to_disk
        assert result.dtype == torch.uint8
        assert result.is_cuda

    @pytest.mark.parametrize("state_id", [0, 1, 2, 3])
    def test_select_state_slice_with_different_state_ids(self, setup_envs: None, state_cache: torch.Tensor, state_id: int) -> None:
        """Test select_state_slice_to_store with different state_id values with extra state."""
        result = GPUCacheInterface.select_state_slice_to_store(
            state_id=state_id,
            last_hit_position=_hit_position(500),
            state_cache=state_cache,
            state_tensor=None,
        )

        assert result is not None
        assert result.shape[0] == GPUCacheInterface.state_size_to_disk
        assert result.dtype == torch.uint8
        assert result.is_cuda


class TestStateCacheOperationsWithoutExtraState(TestStateCacheOperationsBase):
    """Test suite for DEEPINFER_PREFILL_STORE_EXTRA_STATE = False."""

    @pytest.fixture
    def setup_envs(self, setup_envs_base: None, monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
        """Setup environment variables for testing without extra state."""
        monkeypatch.setattr(envs, "DEEPINFER_PREFILL_STORE_EXTRA_STATE", False)
        clear_compress_block_size_cache()

        yield

        clear_compress_block_size_cache()

    def test_select_state_slice_to_store_basic(self, setup_envs: None, state_cache: torch.Tensor) -> None:
        """Test basic functionality of select_state_slice_to_store without extra state."""
        state_tensor = torch.empty(GPUCacheInterface.state_size_to_disk, dtype=torch.uint8, device="cuda")

        result = GPUCacheInterface.select_state_slice_to_store(
            state_id=0,
            last_hit_position=_hit_position(500),
            state_cache=state_cache,
            state_tensor=state_tensor,
        )

        assert result is not None
        assert result.shape[0] == GPUCacheInterface.state_size_to_disk
        assert result is state_tensor  # must return the preallocated tensor as-is
        assert result.dtype == torch.uint8
        assert result.is_cuda

    def test_load_state_from_store_to_prefill_cache_basic(self, setup_envs: None, state_cache: torch.Tensor) -> None:
        """Test basic functionality of load_state_from_store_to_prefill_cache without extra state."""
        state_id = 0
        original_state_cache = state_cache.clone()
        state_tensor = torch.randint(0, 256, (GPUCacheInterface.state_size_to_disk,), dtype=torch.uint8, device="cuda")

        GPUCacheInterface.load_state_from_store_to_prefill_cache(
            state_id=state_id, last_hit_position=_hit_position(500), state_tensor=state_tensor, state_cache=state_cache
        )

        assert not torch.equal(state_cache, original_state_cache)

        # without extra state, load is an identity copy and the first state_size_to_disk bytes should match the written tensor
        assert torch.equal(state_cache[state_id, : GPUCacheInterface.state_size_to_disk], state_tensor)

    @pytest.mark.parametrize("last_position", [500])
    def test_round_trip_consistency(self, setup_envs: None, state_cache: torch.Tensor, last_position: int) -> None:
        """Test that select and load operations are consistent (round-trip) without extra state."""
        state_id = 0
        last_hit_position = _hit_position(last_position)
        test_state_cache = state_cache.clone()

        # select -> rewrite -> load back into cache -> select again; the rewritten contents must come back
        selected_state = GPUCacheInterface.select_state_slice_to_store(
            state_id=state_id,
            last_hit_position=last_hit_position,
            state_cache=test_state_cache,
            state_tensor=None,
        )
        modified_state = _plus_one_wrapped(selected_state)
        GPUCacheInterface.load_state_from_store_to_prefill_cache(
            state_id=state_id, last_hit_position=last_hit_position, state_tensor=modified_state, state_cache=test_state_cache
        )
        reselected_state = GPUCacheInterface.select_state_slice_to_store(
            state_id=state_id,
            last_hit_position=last_hit_position,
            state_cache=test_state_cache,
            state_tensor=None,
        )

        # without extra state there is no reshuffle, so all bytes must match exactly
        assert torch.equal(reselected_state, modified_state)


class TestStateCacheOperationsWithExtraSizeZero(TestStateCacheOperationsBase):
    """Test suite for DEEPINFER_PREFILL_STORE_EXTRA_STATE = True with DEEPINFER_PREFILL_STORE_EXTRA_SIZE = 0."""

    @pytest.fixture
    def setup_envs(self, setup_envs_base: None, monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
        """Setup environment variables for testing with extra state enabled but extra size 0."""
        monkeypatch.setattr(envs, "DEEPINFER_PREFILL_STORE_EXTRA_STATE", True)
        monkeypatch.setattr(envs, "DEEPINFER_PREFILL_STORE_EXTRA_SIZE", 0)
        clear_compress_block_size_cache()

        yield

        clear_compress_block_size_cache()

    def test_select_state_slice_to_store_basic(self, setup_envs: None, state_cache: torch.Tensor) -> None:
        """Test basic functionality of select_state_slice_to_store with extra size 0."""
        state_tensor = torch.empty(GPUCacheInterface.state_size_to_disk, dtype=torch.uint8, device="cuda")

        result = GPUCacheInterface.select_state_slice_to_store(
            state_id=0,
            last_hit_position=_hit_position(500),
            state_cache=state_cache,
            state_tensor=state_tensor,
        )

        assert result is not None
        assert result.shape[0] == GPUCacheInterface.state_size_to_disk
        assert result is state_tensor  # must return the preallocated tensor as-is
        assert result.dtype == torch.uint8
        assert result.is_cuda

    def test_load_state_from_store_to_prefill_cache_basic(self, setup_envs: None, state_cache: torch.Tensor) -> None:
        """Test basic functionality of load_state_from_store_to_prefill_cache with extra size 0."""
        state_id = 0
        original_state_cache = state_cache.clone()
        state_tensor = torch.randint(0, 256, (GPUCacheInterface.state_size_to_disk,), dtype=torch.uint8, device="cuda")

        GPUCacheInterface.load_state_from_store_to_prefill_cache(
            state_id=state_id, last_hit_position=_hit_position(500), state_tensor=state_tensor, state_cache=state_cache
        )

        assert not torch.equal(state_cache, original_state_cache)

        # With EXTRA_SIZE=0 YOCO still keeps keep_tokens; verify by re-selecting on logical positions.
        restored = GPUCacheInterface.select_state_slice_to_store(state_id, _hit_position(500), state_cache)
        assert torch.equal(_stored_swa(restored), _stored_swa(state_tensor))


class TestKVCacheLayoutCompressRatios:
    """Layout-level tests for the refactored cache engine.

    These tests cover the currently retained C1/C2 compression ratios.
    Key checks:
      1. ``ModelMetadata`` accepts the C1/C2 ratios.
      2. ``KVCacheManager.finalize()``'s drift assertion holds for these combinations
         (StateBlock's precomputed byte count matches the per-layer accumulation in process).
      3. layout / state shapes are self-consistent (offsets monotonic and non-overlapping, total length aligned to align_to).

    NOTE Only layout computation is done (gpu_blocks/state_blocks=None); it is pure config math and needs no real kv cache buffer,
         so it does not allocate GPU memory, but constructing ModelMetadata still needs torch.
    """

    BLOCK_SIZE = 128
    NUM_SEQS = 8

    @pytest.fixture(autouse=True)
    def setup_envs(self, monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
        """Pin the env vars that affect the state token count and reset GPUCacheInterface's class fields so assertions reflect only the current run."""
        monkeypatch.setattr(envs, "DEEPINFER_MAX_BACKTRACKING_TOKENS", 10)
        monkeypatch.setattr(envs, "DEEPINFER_PREFILL_STORE_EXTRA_STATE", False)
        clear_compress_block_size_cache()

        # class fields leaking across tests let cache_info_ready / SWA assertions pass on stale values (false green), so reset everything to the declared defaults first
        for name, ann in GPUCacheInterface.__annotations__.items():
            monkeypatch.setattr(GPUCacheInterface, name, False if ann in (bool, "bool") else 0)
        # the layout logging in _kvcache_layout calls dist.get_rank(), which hangs in tests without an initialized PG;
        # setting this True short-circuits `not kvcache_layout_logged and dist.get_rank()==0` and skips the logging
        monkeypatch.setattr(GPUCacheInterface, "kvcache_layout_logged", True)

        yield

        clear_compress_block_size_cache()

    @staticmethod
    def _rewrite_compress_ratio(cfg: Any, new_ratio: int) -> Any:
        """Rewrite the supported compressed layers in the config to the given ratio.

        - Only layers that are already compressed (kv_compress_ratio > 0) are rewritten; the others (full / swa-only) stay unchanged.
        """
        cfg = _deepcopy_unalias_layers(cfg)
        attn_module_list = cfg.get("attn_module_list") or []
        rewrote = 0
        for attn in attn_module_list:
            if attn is None:
                continue
            if attn.cfg.get("kv_compress_ratio", 0) > 0:
                attn.cfg["kv_compress_ratio"] = new_ratio
                rewrote += 1
        assert rewrote > 0, "the base config has no rewritable compressed layer"
        return cfg

    def _assert_layout_self_consistent(
        self,
        layout: list[tuple[int, int, int]],
        kv_shape: tuple[int],
        state_shape: tuple[int, int],
        align_to: int,
        state_align_to: int,
    ) -> None:
        (total_kv,) = kv_shape
        num_slots, total_state = state_shape

        assert num_slots == get_num_state_slots(self.NUM_SEQS)
        assert total_kv > 0
        assert total_kv % align_to == 0
        # the base configuration always keeps SWA or non-c1 compressed state
        assert total_state > 0
        assert total_state % state_align_to == 0

        # the layout's (offset, length) pairs must increase monotonically in offset and be in-bounds and non-overlapping
        prev_end = 0
        for layer_id, offset, length in layout:
            assert length > 0
            assert offset >= prev_end, f"layout overlaps or is out of order: {layer_id=}, {offset=}, {prev_end=}"
            assert offset % 32 == 0, f"the attention kernel requires 32-aligned kv offsets: {offset=}"
            assert offset + length <= total_kv
            prev_end = offset + length
        assert total_kv - prev_end < align_to

    @pytest.mark.parametrize("ratio", [1, 2])
    def test_layout_finalize_drift_and_shape(self, model_config: Any, ratio: int) -> None:
        """For every compression-ratio combination, _kvcache_layout (which triggers finalize's drift assertion) must pass and the shapes must be self-consistent."""
        cfg = self._rewrite_compress_ratio(model_config, ratio)

        # ModelMetadata must accept c1/c2 (the new assertion); otherwise the compatibility change is missing
        model_metadata = ModelMetadata(cfg)
        bytes_per_token = 288 if model_metadata.use_fp4_sparse_kv else 528
        align_to = math.lcm(model_metadata.kvcache_align_to, bytes_per_token)

        # only layout / shapes are computed and no real buffers are allocated; this still runs the full KVCacheManager.process + finalize()
        layout, state_size_to_disk, kv_shape, state_shape, cache_list = GPUCacheInterface._kvcache_layout(
            cfg, self.BLOCK_SIZE, self.NUM_SEQS
        )

        # SWA stays FP8 (528 bytes/token); its state-slot stride also requires 32B alignment.
        state_align_to = math.lcm(32, 528)
        self._assert_layout_self_consistent(layout, kv_shape, state_shape, align_to, state_align_to)

        # Every non-lmhead layer should produce one KVCache.
        # _kvcache_layout's logic: walk attn_module_list, skip only idx == cfg.num_hidden_layers (the lmhead placeholder),
        # and every other module (including any nextn/MTP layers after n_layer) is processed into a KVCache.
        attn_module_list = cfg.get("attn_module_list") or []
        expected_num_cache = sum(1 for idx in range(len(attn_module_list)) if idx != cfg.num_hidden_layers)
        assert len(cache_list) == expected_num_cache

        # without overlap there is no dump reshuffle and only the swa part is dumped; state_size_to_disk must not exceed the total state
        assert 0 <= state_size_to_disk <= state_shape[1]

    @pytest.mark.parametrize("extra_state", [False, True])
    def test_to_ce_copies_complete_dropped_region(
        self,
        model_config: Any,
        monkeypatch: pytest.MonkeyPatch,
        extra_state: bool,
    ) -> None:
        """PE->CE must transfer all of C2's main/index state."""
        monkeypatch.setattr(envs, "DEEPINFER_PREFILL_STORE_EXTRA_STATE", extra_state)
        clear_compress_block_size_cache()
        cfg = self._rewrite_compress_ratio(model_config, 2)
        _, _, _, state_shape, _ = GPUCacheInterface._kvcache_layout(cfg, self.BLOCK_SIZE, self.NUM_SEQS)

        assert GPUCacheInterface.dropped_bytes_inference > 0
        cache_offset = GPUCacheInterface.swa_total_bytes_kvstore_pad
        tensor_offset = GPUCacheInterface.swa_total_bytes_inference_pad
        dropped_size = GPUCacheInterface.dropped_bytes_inference - GPUCacheInterface.swa_drop_total_bytes_inference_pad

        state_cache = torch.zeros(state_shape, dtype=torch.uint8, device="cuda")
        _, _, _, _, cache_list = GPUCacheInterface._kvcache_layout(
            cfg,
            self.BLOCK_SIZE,
            self.NUM_SEQS,
            state_blocks=state_cache,
        )
        expected_dropped_size = sum(
            state[0].numel() * state.element_size()
            for cache in cache_list
            for state in (cache.ica_state, cache.ca_state)
            if state is not None
        )
        assert dropped_size == expected_dropped_size
        state_cache[0, cache_offset : cache_offset + dropped_size] = 0x3C
        state_tensor = torch.full(
            (GPUCacheInterface.total_bytes_inference,),
            0xA5,
            dtype=torch.uint8,
            device="cuda",
        )
        result = GPUCacheInterface.select_state_slice_to_store(
            state_id=0,
            last_hit_position=1,
            state_cache=state_cache,
            state_tensor=state_tensor,
            TO_CE=True,
        )

        assert result is state_tensor
        assert torch.all(state_tensor[tensor_offset : tensor_offset + dropped_size] == 0x3C)

    def test_to_ce_copies_compress_state_after_swa(
        self,
        model_config: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The EXTRA path must copy compressed state from the tail of the kvstore/inference SWA respectively."""
        monkeypatch.setattr(envs, "DEEPINFER_PREFILL_STORE_EXTRA_STATE", True)
        clear_compress_block_size_cache()
        _, _, _, state_shape, _ = GPUCacheInterface._kvcache_layout(model_config, self.BLOCK_SIZE, self.NUM_SEQS)

        assert GPUCacheInterface.swa_total_bytes_kvstore_pad > GPUCacheInterface.swa_total_bytes_inference_pad > 0
        assert GPUCacheInterface.dropped_bytes_inference > 0
        cache_offset = GPUCacheInterface.swa_total_bytes_kvstore_pad
        tensor_offset = GPUCacheInterface.swa_total_bytes_inference_pad
        dropped_size = GPUCacheInterface.dropped_bytes_inference - GPUCacheInterface.swa_drop_total_bytes_inference_pad

        state_cache = torch.zeros(state_shape, dtype=torch.uint8, device="cuda")
        state_cache[0, cache_offset : cache_offset + dropped_size] = 0x3C
        state_tensor = torch.full(
            (GPUCacheInterface.total_bytes_inference,),
            0xA5,
            dtype=torch.uint8,
            device="cuda",
        )
        GPUCacheInterface.select_state_slice_to_store(
            state_id=0,
            last_hit_position=1,
            state_cache=state_cache,
            state_tensor=state_tensor,
            TO_CE=True,
        )

        assert torch.all(state_tensor[tensor_offset : tensor_offset + dropped_size] == 0x3C)

    def test_c1_allocates_kv_without_compress_state(self, model_config: Any) -> None:
        """c1 keeps main/index KV but allocates no compress state."""
        cfg = self._rewrite_compress_ratio(model_config, 1)
        _, _, kv_shape, state_shape, _ = GPUCacheInterface._kvcache_layout(cfg, self.BLOCK_SIZE, self.NUM_SEQS)
        gpu_blocks = torch.empty((1, *kv_shape), dtype=torch.uint8)
        state_blocks = torch.empty(state_shape, dtype=torch.uint8)
        _, _, _, _, cache_list = GPUCacheInterface._kvcache_layout(
            cfg,
            self.BLOCK_SIZE,
            self.NUM_SEQS,
            gpu_blocks=gpu_blocks,
            state_blocks=state_blocks,
        )

        attn_modules = [attn for idx, attn in enumerate(cfg.attn_module_list) if idx != cfg.num_hidden_layers]
        for cache, attn_module in zip(cache_list, attn_modules, strict=True):
            if attn_module.cfg.get("kv_compress_ratio", 0) != 1:
                continue
            assert cache.ca_cache is not None
            if GPUCacheInterface._is_sparse(attn_module):
                assert cache.ica_cache is not None
            assert cache.ca_state is None
            assert cache.ica_state is None

    def test_pure_c1_has_no_state_buffer(self, model_config: Any) -> None:
        cfg = self._rewrite_compress_ratio(model_config, 1)
        c1_layer = next(attn for attn in cfg.attn_module_list if attn is not None and attn.cfg.get("kv_compress_ratio", 0) == 1)
        c1_layer.cfg["sink_mode"] = ""
        c1_layer.cfg.pop("layer_info_shared_cfg", None)
        cfg.num_hidden_layers = cfg.n_prefill_layer = 1
        cfg.attn_module_list = [c1_layer, None]

        manager = KVCacheManager(cfg, kv_tensor=None, state_tensor=None)
        _, state_size_to_disk, _, state_shape, _ = GPUCacheInterface._kvcache_layout(cfg, self.BLOCK_SIZE, self.NUM_SEQS)

        assert manager.state_total_in_bytes == 0
        assert manager._state_block.total_bytes_inference == 0
        assert manager._state_block.all_compress_block_size == {}
        assert state_size_to_disk == 0
        assert state_shape == (get_num_state_slots(self.NUM_SEQS), 0)
        assert GPUCacheInterface.state_size_to_disk == 0
        assert GPUCacheInterface.total_bytes_inference == 0
        assert GPUCacheInterface.dropped_bytes_inference == 0

    def test_manager_finalize_asserts_directly(self, model_config: Any) -> None:
        """Construct KVCacheManager directly and verify finalize()'s SWA/compression offset assertions hold after processing all layers."""
        cfg = self._rewrite_compress_ratio(model_config, 2)

        manager = KVCacheManager(cfg, kv_tensor=None, state_tensor=None)
        attn_module_list = cfg.get("attn_module_list") or []
        cache_list: list = []
        layer_id = 0
        for idx, attn_module in enumerate(attn_module_list):
            if idx == cfg.num_hidden_layers:
                continue
            kvcache, _ = manager.process(layer_id, attn_module, self.BLOCK_SIZE, cache_list)
            cache_list.append(kvcache)
            layer_id += 1
        # if StateBlock's precomputation disagrees with the per-layer accumulation, finalize's internal assert raises
        manager.finalize()

        sb = manager._state_block
        # all C2 compressed state sits in the dropped region and is not dumped
        assert manager._dropped_state_offset == sb.swa_drop_total_bytes_kvstore_pad + sb.compress_total_bytes_kvstore


def _allocate_yoco_test_state(cfg: AttrDict) -> tuple[KVCacheManager, torch.Tensor, list[KVCache]]:
    from deepinfer.model_executor.layer_utils import iter_runtime_attn_modules

    sizing = KVCacheManager(cfg, None, None)
    state = torch.randint(0, 256, (2, sizing.state_total_in_bytes), dtype=torch.uint8, device="cuda")
    manager = KVCacheManager(cfg, None, state)
    caches = []
    for layer_id, _, module in iter_runtime_attn_modules(cfg.attn_module_list, cfg.num_hidden_layers):
        cache, _ = manager.process(layer_id, module, 128, caches)
        caches.append(cache)
    manager.finalize()
    manager.update_gpu_cache_interface_class_fields()
    return manager, state, caches


def _yoco_state_parts(cache: KVCache, state_id: int) -> list[torch.Tensor]:
    """Read effective data from each layer's public view, without relying on _carve_state_views' partition offsets."""
    assert cache.swa_cache is not None
    ring_size = cache.swa_cache.shape[1]
    swa = cache.swa_cache[state_id].view(ring_size, -1)
    main_size = V41_FORMAT.quantized_bytes
    parts = [swa[:, :main_size], swa[:, main_size:]]
    return parts + [state[state_id] for state in (cache.ca_state, cache.ica_state) if state is not None]


@pytest.mark.parametrize("extra_state", [False, True])
def test_yoco_swa_disk_and_ce(monkeypatch: pytest.MonkeyPatch, extra_state: bool) -> None:
    """PE compact and CE off have identical layouts; the dump keeps only the stored region while transfers keep all state."""
    monkeypatch.setattr(envs, "DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE", "compact")
    monkeypatch.setattr(envs, "DEEPINFER_YOCO_PREFILL_KEEP_TOKENS", 32)
    monkeypatch.setattr(envs, "DEEPINFER_PREFILL_STORE_EXTRA_STATE", extra_state)
    monkeypatch.setattr(envs, "DEEPINFER_PREFILL_STORE_EXTRA_SIZE", 128)
    monkeypatch.setattr(envs, "DEEPINFER_MAX_BACKTRACKING_TOKENS", 3)
    # Producer/consumer C2 and a DFlash SWA ring cover persisted SWA, dropped SWA and compression state.
    cfg = AttrDict(
        num_hidden_layers=2,
        n_prefill_layer=1,
        dense_ffn_dim=256,
        use_ape=False,
        use_fp4_sparse_kv=True,
        kv_lora_rank=448,
        qk_rope_head_dim=64,
        window_size=(17, 0),
        moe=AttrDict(moe_intermediate_size=256),
        attn_module_list=[
            AttrDict(
                cfg=AttrDict(kv_compress_ratio=2, sink_mode="swa", index_topk=8, attn_index=AttrDict(index_head_dim=128, index_n_heads=1))
            ),
            AttrDict(cfg=AttrDict(kv_compress_ratio=2, sink_mode="swa")),
            None,
            AttrDict(cfg=AttrDict(kv_compress_ratio=0, sink_mode="swa")),
        ],
    )
    clear_compress_block_size_cache()
    try:
        pe, state, caches = _allocate_yoco_test_state(cfg)
        sb = pe._state_block
        assert (sb.swa_store_num_layer, sb.swa_drop_num_layer) == (1, 2)
        assert sb.state_size_to_disk == sb.swa_bytes_per_layer_inference_pad
        with monkeypatch.context() as ce_env:
            ce_env.setattr(envs, "DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE", "off")
            ce_env.setattr(envs, "DEEPINFER_PREFILL_STORE_EXTRA_STATE", False)
            clear_compress_block_size_cache()
            ce, ce_state, ce_caches = _allocate_yoco_test_state(cfg)
        clear_compress_block_size_cache()
        pe.update_gpu_cache_interface_class_fields()
        assert ce.state_total_in_bytes == sb.total_bytes_inference
        assert ce._state_block.state_size_to_disk == sb.state_size_to_disk
        assert ce._state_block.swa_store_num_layer == sb.swa_store_num_layer

        state_id, hit = 1, 385  # a non-zero state row; the transfer position is not dump-granularity aligned and the ring wrapped.
        GPUCacheInterface.select_state_slice_to_store(state_id, hit, state, ce_state[state_id], TO_CE=True)
        for src_cache, dst_cache in zip(caches, ce_caches, strict=True):
            for src, dst in zip(
                _yoco_state_parts(src_cache, state_id),
                _yoco_state_parts(dst_cache, state_id),
                strict=True,
            ):
                positions = torch.arange(hit - dst.shape[0], hit, device="cuda")
                assert torch.equal(dst[positions % dst.shape[0]].view(torch.uint8), src[positions % src.shape[0]].view(torch.uint8))

        disk_hit = 384
        disk = GPUCacheInterface.select_state_slice_to_store(state_id, disk_hit, state)
        assert disk is not None and disk.numel() == sb.state_size_to_disk
        restored = torch.full_like(state, 0xA5)
        GPUCacheInterface.load_state_from_store_to_prefill_cache(state_id, disk_hit, disk, restored)
        drop_start = sb.swa_store_total_bytes_kvstore_pad
        assert torch.all(restored[state_id, drop_start:] == 0xA5)
        assert torch.all(restored[0] == 0xA5)
        restored_disk = GPUCacheInterface.select_state_slice_to_store(state_id, disk_hit, restored)
        assert torch.equal(_stored_swa(disk), _stored_swa(restored_disk))
    finally:
        clear_compress_block_size_cache()


@pytest.mark.parametrize(
    "removed_option,value,error",
    [("kv_compress_ratio", 4, "C4"), ("kv_compress_ratio", 128, "C128"), ("compress_overlap", True, "overlap"), ("use_ape", True, "APE")],
)
def test_removed_compression_features_rejected(model_config: Any, removed_option: str, value: Any, error: str) -> None:
    attn = next(attn for attn in model_config.attn_module_list if attn is not None and attn.cfg.get("kv_compress_ratio", 0) > 0)
    if removed_option == "use_ape":
        model_config.use_ape = value
    else:
        attn.cfg[removed_option] = value
    with pytest.raises(ValueError, match=error):
        ModelMetadata(model_config)
