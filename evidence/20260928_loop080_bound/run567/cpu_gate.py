"""CPU-only negative gate for scoped Run567 pause state capture."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import sys

import torch

sys.path.insert(0, "/data/wio/Inference_Foundry/diagnostics")
sys.path.insert(0, "/data/wio/Inference_Foundry")
from loop080_pause_state_probe_run567 import capture, compare  # noqa: E402
from runtime.fixed_decode import FixedDecodeState  # noqa: E402


def tensor():
    return torch.arange(4, dtype=torch.int64)


@dataclass
class Cache:
    name: str
    tensor: torch.Tensor


class TargetOwner:
    def __init__(self):
        self.assets = SimpleNamespace(caches=(Cache("kv", tensor()),),
                                      _mutable_tensors=(Cache("recurrent", tensor()),))

    def forward(self, *args):
        raise AssertionError("probe must not run Target")


class Common:
    def __init__(self):
        self.query_start_loc_cpu = tensor()
        self._seq_lens_cpu = tensor()
        self.seq_lens_cpu = tensor()
        self.query_start_loc = tensor()
        self.num_reqs = 12
        self.lazy_reads = 0

    @property
    def num_computed_tokens_cpu(self):
        self.lazy_reads += 1
        self._num_computed_tokens_cpu = tensor()
        return self._num_computed_tokens_cpu


def fixture():
    state = FixedDecodeState(**{
        name: (0 if name == "cycle_index" else tensor())
        for name in FixedDecodeState.__dataclass_fields__
    })
    common = Common()
    proposer_inner = SimpleNamespace(
        _dspark_draft_buffer=tensor(), _dspark_seed_buffer=tensor(),
        hidden_states=tensor(), _dflash_hidden_states=tensor(),
        positions=tensor(), _slot_mapping_buffer=tensor(),
        draft_attn_groups=[SimpleNamespace(kv_cache_group_id=2),
                           SimpleNamespace(kv_cache_group_id=3)],
        _layer_group_idx=[2, 3],
        _per_group_block_tables={gid: tensor() for gid in range(4)},
        _per_group_slot_mappings={gid: tensor() for gid in range(4)},
        _per_group_block_table_buffers={gid: tensor() for gid in (2, 3)},
        _per_group_query_slot_mapping_buffers={gid: tensor() for gid in (2, 3)},
        _per_group_context_slot_mapping_buffers={gid: tensor() for gid in (2, 3)},
        _context_slot_mapping_buffers=[tensor(), tensor()],
    )
    handoff = SimpleNamespace(
        common_attn_metadata=common, proposer=proposer_inner,
        _host_copy_pending=False, _host_count_copy=tensor(),
        _committed_emitted_count=tensor(),
        _spec_metadata=SimpleNamespace(cu_num_draft_tokens=tensor()),
    )
    owner = TargetOwner()
    runtime = SimpleNamespace(
        state=state, target=SimpleNamespace(binding=SimpleNamespace(forward=owner.forward)),
        proposer=handoff, _schedule_pending=False,
        schedule_lifetime=lambda: {"launches": 0, "commits": 0},
    )
    serving = SimpleNamespace(
        runtime=runtime, generation=5, request_ids=tuple(str(i) for i in range(12)),
        initial_output_counts=[0] * 12, remaining=[1024] * 12,
        _progress_baseline=[0] * 12, _initial_positions=[32768] * 12,
        _parked=[True] + [False] * 11,
        _last_decision_progress=[1024] + [0] * 11, _cycles=2,
        _token_history=torch.arange(32).reshape(4, 8),
        _count_history=torch.arange(16).reshape(4, 4),
    )
    return serving


def must_reject(before, serving, label):
    try:
        compare(before, capture(serving))
    except AssertionError as error:
        assert "pause changed enumerated state" in str(error), label
    else:
        raise AssertionError(f"{label} accepted")


s = fixture()
base = capture(s)
assert s.runtime.proposer.common_attn_metadata.lazy_reads == 0
assert compare(base, capture(s))["captured_fields_unchanged"]
assert compare(base, capture(s))["full_target_kv_content_certified"] is False
s.runtime.state.num_sampled[0] += 1
must_reject(base, s, "state tensor")
s = fixture(); base = capture(s)
s.runtime.proposer.common_attn_metadata._seq_lens_cpu[0] += 1
must_reject(base, s, "Host mirror")
s = fixture(); base = capture(s)
s._cycles += 1
must_reject(base, s, "serving scalar")
s = fixture(); base = capture(s)
s.runtime.proposer.common_attn_metadata._seq_lens_cpu = tensor().clone()
must_reject(base, s, "Host mirror address")
s = fixture(); s.runtime._schedule_pending = True
try:
    capture(s)
except RuntimeError as error:
    assert "scheduled metadata pending" in str(error)
else:
    raise AssertionError("pending scheduling accepted")
s = fixture(); s.runtime.proposer._host_copy_pending = True
try:
    capture(s)
except RuntimeError as error:
    assert "Host count copy pending" in str(error)
else:
    raise AssertionError("pending Host copy accepted")
s = fixture(); s.runtime.proposer.common_attn_metadata.query_start_loc_cpu = None
try:
    capture(s)
except RuntimeError as error:
    assert "required Host mirror absent" in str(error)
else:
    raise AssertionError("missing Host mirror accepted")
s = fixture(); s.runtime.state.extra = tensor()
try:
    capture(s)
except RuntimeError as error:
    assert "FixedDecodeState field drift" in str(error)
else:
    raise AssertionError("unknown state field accepted")
s = fixture()
try:
    capture(s, max_cloned_bytes=1)
except RuntimeError as error:
    assert "clone budget exceeded" in str(error)
else:
    raise AssertionError("clone budget exceeded without failure")
s = fixture(); base = capture(s)
s.runtime.proposer.common_attn_metadata._seq_lens_cpu = tensor().reshape(2, 2)
must_reject(base, s, "Host mirror shape")
s = fixture(); base = capture(s)
s.runtime.target.binding.forward.__self__.assets.caches = (
    Cache("kv", tensor()), Cache("kv2", tensor()))
must_reject(base, s, "Target cache registry")
s = fixture(); s.runtime.proposer.proposer.hidden_states = None
try:
    capture(s)
except RuntimeError as error:
    assert "required proposer buffer missing" in str(error)
else:
    raise AssertionError("missing proposer buffer accepted")
s = fixture()
s.runtime.proposer.proposer._per_group_block_tables = {0: tensor(), "0": tensor()}
try:
    capture(s)
except RuntimeError as error:
    assert "duplicate tensor path" in str(error)
else:
    raise AssertionError("duplicate tensor path accepted")
s = fixture(); s.runtime.proposer.proposer._per_group_block_tables = {2: None}
try:
    capture(s)
except RuntimeError as error:
    assert "non-tensor required leaf" in str(error)
else:
    raise AssertionError("missing proposer dict tensor accepted")
s = fixture(); s.runtime.proposer.proposer._context_slot_mapping_buffers = [None, tensor()]
try:
    capture(s)
except RuntimeError as error:
    assert "non-tensor required leaf" in str(error)
else:
    raise AssertionError("missing context slot tensor accepted")
s = fixture(); s.runtime.proposer.proposer._per_group_slot_mappings.pop(0)
try:
    capture(s)
except RuntimeError as error:
    assert "input group-key mismatch" in str(error)
else:
    raise AssertionError("input group mismatch accepted")
s = fixture(); s.runtime.proposer.proposer._per_group_block_table_buffers.pop(2)
try:
    capture(s)
except RuntimeError as error:
    assert "draft buffer group-key mismatch" in str(error)
else:
    raise AssertionError("draft group mismatch accepted")
s = fixture(); s.runtime.proposer.proposer._per_group_block_tables.pop(2)
s.runtime.proposer.proposer._per_group_slot_mappings.pop(2)
try:
    capture(s)
except RuntimeError as error:
    assert "draft group absent from input maps" in str(error)
else:
    raise AssertionError("draft gid missing from inputs accepted")
s = fixture(); s.runtime.proposer.proposer._context_slot_mapping_buffers.pop()
try:
    capture(s)
except RuntimeError as error:
    assert "context slot mapping layer count mismatch" in str(error)
else:
    raise AssertionError("context layer count mismatch accepted")
s = fixture(); base = capture(s)
s.runtime.state.num_sampled = torch.tensor([float("nan"), 1, 2, 3])
must_reject(base, s, "NaN/value change")
print("Run567 CPU scoped-state gate PASS")
