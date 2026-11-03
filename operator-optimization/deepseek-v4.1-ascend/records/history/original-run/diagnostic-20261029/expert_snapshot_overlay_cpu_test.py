"""CPU-only failure-path tests; mock mode lifecycle, no PyTorch/NPU imports."""

import builtins
import sys
import types
import unittest
from unittest.mock import patch

import expert_snapshot_overlay as overlay


class OverlayFailures(unittest.TestCase):
    def test_missing_torch_leaves_class_unmodified(self):
        class Comm:
            def fused_experts(self, inp):
                return "ok"
        original = Comm.fused_experts
        real_import = builtins.__import__

        def no_dispatch(name, *args, **kwargs):
            if name == "torch.utils._python_dispatch":
                raise ImportError("missing torch")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=no_dispatch):
            installed = overlay.install_metadata_observer(
                Comm, enabled=True,
                main_bindings={14: overlay.LayerBinding(14, object(), "unused")},
                target_mc2_ranks=frozenset({0}), anonymous_key=b"test",
                execution_context=lambda: "capture", global_rank=lambda: 0,
                emit=lambda row: None,
            )
        self.assertFalse(installed)
        self.assertIs(Comm.fused_experts, original)

    def test_unreadable_schema_forwards_once(self):
        class BrokenSchema:
            calls = 0

            @property
            def _schema(self):
                raise RuntimeError("schema unavailable")

            def __call__(self, *args, **kwargs):
                self.calls += 1
                return "ok"

        func = BrokenSchema()
        observations = []
        self.assertEqual(overlay._forward_observed(func, (), {}, None, observations), "ok")
        self.assertEqual(func.calls, 1)
        self.assertEqual(observations, [None])

    def test_target_failure_does_not_retry(self):
        class BrokenOp:
            calls = 0
            _schema = types.SimpleNamespace(name="unrelated")

            def __call__(self, *args, **kwargs):
                self.calls += 1
                raise LookupError("target failed")

        func = BrokenOp()
        with self.assertRaisesRegex(LookupError, "target failed"):
            overlay._forward_observed(func, (), {}, None, [])
        self.assertEqual(func.calls, 1)

    def _make_wrapper(self, mode, context="eager_host_call", group="group"):
        torch = types.ModuleType("torch")
        utils = types.ModuleType("torch.utils")
        dispatch = types.ModuleType("torch.utils._python_dispatch")
        dispatch.TorchDispatchMode = mode
        expert = types.SimpleNamespace(layer_name="main")
        decoder = types.SimpleNamespace(
            layer_idx=14,
            mlp=types.SimpleNamespace(
                layer_idx=14, experts=types.SimpleNamespace(routed_experts=expert)
            ),
        )
        root = types.SimpleNamespace(model=types.SimpleNamespace(layers=[decoder]))
        bindings = overlay.bind_main_layers(root, frozenset({14}))

        class Comm:
            calls = 0
            token_dispatcher = types.SimpleNamespace(
                ep_rank_id=0, ep_world_size=8, moe_all_to_all_group_name=group
            )

            def fused_experts(self, inp):
                self.calls += 1
                if getattr(inp, "fail", False):
                    raise LookupError("target failed")
                return "ok"

        with patch.dict(sys.modules, {"torch": torch, "torch.utils": utils,
                                    "torch.utils._python_dispatch": dispatch}):
            overlay.install_metadata_observer(
                Comm, enabled=True, main_bindings=bindings,
                target_mc2_ranks=frozenset({0}), anonymous_key=b"test",
                execution_context=lambda: context, global_rank=lambda: 0,
                emit=lambda row: None,
            )
        return Comm(), types.SimpleNamespace(layer=expert)

    def test_invalid_anonymous_identity_passthrough_once(self):
        class Mode:
            pass
        comm, inp = self._make_wrapper(Mode, group=None)
        self.assertEqual(comm.fused_experts(inp), "ok")
        self.assertEqual(comm.calls, 1)

    def test_mode_entry_failure_does_not_execute_target(self):
        class Mode:
            def __enter__(self):
                raise RuntimeError("entry")
        comm, inp = self._make_wrapper(Mode)
        with self.assertRaisesRegex(RuntimeError, "mode entry failed"):
            comm.fused_experts(inp)
        self.assertEqual(comm.calls, 0)

    def test_mode_cleanup_failure_does_not_retry_target(self):
        class Mode:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                raise RuntimeError("exit")
        comm, inp = self._make_wrapper(Mode)
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            comm.fused_experts(inp)
        self.assertEqual(comm.calls, 1)

    def test_original_failure_preserved_when_cleanup_fails(self):
        class Mode:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                raise RuntimeError("exit")
        comm, inp = self._make_wrapper(Mode)
        inp.fail = True
        with self.assertRaisesRegex(LookupError, "target failed"):
            comm.fused_experts(inp)
        self.assertEqual(comm.calls, 1)


if __name__ == "__main__":
    unittest.main()
