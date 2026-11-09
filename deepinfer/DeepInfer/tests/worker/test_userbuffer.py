from types import SimpleNamespace
from typing import Any

import pytest

import deepinfer.model_executor.parallel_utils.userbuffer as userbuffer_module
from deepinfer.model_executor.parallel_utils.userbuffer import _UserBufferManager


class _Group:
    """Stand-in for dist.ProcessGroup: identity-hashed, rank/size only feed logging."""

    @staticmethod
    def rank() -> int:
        return 0

    @staticmethod
    def size() -> int:
        return 1


class _FakeSymBuffer:
    destroyed: list["_FakeSymBuffer"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.base = kwargs.get("base")
        self.buffer = SimpleNamespace(nbytes=4096)
        self.tensor_alias = object()

    def destroy(self) -> None:
        self.destroyed.append(self)


class _FakeElasticBuffer:
    def __init__(self) -> None:
        self.destroyed = False

    def destroy(self) -> None:
        self.destroyed = True


@pytest.fixture
def manager(monkeypatch: pytest.MonkeyPatch) -> _UserBufferManager:
    monkeypatch.setattr(userbuffer_module.deep_gemm, "SymmBuffer", _FakeSymBuffer)
    locality = getattr(userbuffer_module.deep_gemm, "locality_domain", None)
    if hasattr(locality, "get_balanced_sm_locality_domains"):
        monkeypatch.setattr(locality, "get_balanced_sm_locality_domains", lambda: None)
    monkeypatch.setattr(_FakeSymBuffer, "destroyed", [])
    return _UserBufferManager()


def _allocate_sym_buffer(manager: _UserBufferManager, group: _Group, max_num_tokens: int) -> _FakeSymBuffer:
    return manager.allocate_sym_buffer(
        group,
        num_experts=8,
        max_num_tokens=max_num_tokens,
        num_topk=2,
        hidden_size=256,
        moe_intermediate_size=512,
        num_shared_experts=1,
        mma_type="bf16xbf16",
        activation="swiglu",
    )


def test_allocate_sym_buffer_views_share_first_base(manager: _UserBufferManager) -> None:
    group = _Group()
    base = _allocate_sym_buffer(manager, group, max_num_tokens=128)
    assert base.base is None
    assert _allocate_sym_buffer(manager, group, max_num_tokens=128) is base
    view = _allocate_sym_buffer(manager, group, max_num_tokens=64)
    assert view.base is base
    assert _allocate_sym_buffer(manager, group, max_num_tokens=32).base is base


def test_close_releases_all_userbuffer_resources(manager: _UserBufferManager) -> None:
    group = _Group()
    base = _allocate_sym_buffer(manager, group, max_num_tokens=128)
    view = _allocate_sym_buffer(manager, group, max_num_tokens=64)
    elastic_buffers = [_FakeElasticBuffer() for _ in range(3)]
    manager.ep, manager.cp, manager.engram = elastic_buffers

    manager.close()

    assert all(buffer.destroyed for buffer in elastic_buffers)
    assert _FakeSymBuffer.destroyed == [view, base]  # views destroyed before their base
    for buffer in (base, view):
        assert vars(buffer) == {}
    assert manager.ep is manager.cp is manager.engram is None
    assert not manager._sym_buffers

    manager.close()  # idempotent on the emptied manager
